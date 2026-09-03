"""Helpers to move data between Arrow columns and NumPy/PyTorch tensors.

Multimodal Universe catalogs store images and spectra as nested Arrow lists
(``list<list<list<float>>>`` for a ``(bands, height, width)`` cutout). Converting
those to dense arrays row by row is prohibitively slow, so the helpers here work
on whole columns with vectorised Arrow compute kernels and only fall back to
per-group processing when rows have different shapes.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc


def valid_mask(array: pa.Array) -> np.ndarray:
    """Boolean NumPy mask of the non-null entries of ``array``."""
    return np.asarray(pc.is_valid(array).to_numpy(zero_copy_only=False), dtype=bool)


def struct_field(array: pa.StructArray, *names: str) -> pa.Array | None:
    """Return the first child of a struct array whose name is in ``names`` (or ``None``)."""
    for name in names:
        index = array.type.get_field_index(name)
        if index >= 0:
            return array.field(index)
    return None


def nested_shapes(array: pa.Array, ndim: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-row shapes of an ``ndim``-deep nested list array.

    Returns ``(shapes, regular)`` where ``shapes`` is an ``(n_rows, ndim)`` int array
    and ``regular`` is a boolean mask that is ``False`` for rows that are null or ragged
    (sub-lists of unequal length inside the same row).
    """
    n_rows = len(array)
    shapes = np.zeros((n_rows, ndim), dtype=np.int64)
    regular = valid_mask(array)
    current = array
    row_of = np.arange(n_rows)
    for level in range(ndim):
        if not pa.types.is_list(current.type) and not pa.types.is_large_list(current.type):
            raise TypeError(f"Expected {ndim} nested list levels, found {level} in {array.type}")
        lengths = pc.fill_null(pc.list_value_length(current), 0)
        lengths = np.asarray(lengths.to_numpy(zero_copy_only=False), dtype=np.int64)
        if row_of.size:
            mins = np.full(n_rows, np.iinfo(np.int64).max, dtype=np.int64)
            maxs = np.full(n_rows, -1, dtype=np.int64)
            np.minimum.at(mins, row_of, lengths)
            np.maximum.at(maxs, row_of, lengths)
            touched = maxs >= 0
            regular &= ~touched | (mins == maxs)
            shapes[touched, level] = maxs[touched]
        current = pc.list_flatten(current)
        row_of = np.repeat(row_of, lengths)
    regular &= (shapes > 0).all(axis=1)
    return shapes, regular


def nested_to_numpy(array: pa.Array, shape: tuple[int, ...], dtype=np.float32) -> np.ndarray:
    """Convert a nested list array whose rows all have ``shape`` into a dense array.

    The result is a view on the Arrow buffer whenever the dtype already matches.
    """
    values = pc.list_flatten(array, recursive=True)
    flat = np.asarray(values.to_numpy(zero_copy_only=False), dtype=dtype)
    return flat.reshape((len(array), *shape))


def group_rows_by_shape(
    array: pa.Array, ndim: int
) -> tuple[dict[tuple[int, ...], np.ndarray], np.ndarray]:
    """Group row indices of a nested list column by the shape of each row.

    Returns ``(groups, invalid)``: ``groups`` maps a shape tuple to the array of row
    indices with that shape; ``invalid`` lists the rows that are null or ragged.
    """
    shapes, regular = nested_shapes(array, ndim)
    groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for row in np.flatnonzero(regular):
        groups[tuple(int(s) for s in shapes[row])].append(int(row))
    return {k: np.asarray(v, dtype=np.int64) for k, v in groups.items()}, np.flatnonzero(~regular)


def tokens_to_arrow(tokens: np.ndarray, valid: np.ndarray, arrow_type: pa.DataType) -> pa.Array:
    """Build the Arrow column for a block of ``(n_rows, n_tokens)`` tokens.

    ``arrow_type`` is either a plain integer type (``n_tokens`` must then be 1) or a
    struct with one list field, the nested layout LSDB recognises. Rows where ``valid``
    is ``False`` become nulls.
    """
    null = ~np.asarray(valid, dtype=bool)
    if pa.types.is_struct(arrow_type):
        field = arrow_type.field(0)
        values = pa.array(np.ascontiguousarray(tokens).ravel(), type=field.type.value_type)
        mask = pa.array(null)
        lists = pa.FixedSizeListArray.from_arrays(values, tokens.shape[1], mask=mask)
        return pa.StructArray.from_arrays([lists.cast(field.type)], [field.name], mask=mask)
    return pa.array(tokens.reshape(len(null)), type=arrow_type, mask=null)
