import numpy as np
import pyarrow as pa

from aion_hats.arrow_utils import (
    group_rows_by_shape,
    nested_shapes,
    nested_to_numpy,
    tokens_to_arrow,
)


def test_nested_shapes_flags_null_and_ragged_rows():
    arr = pa.array([[[1.0, 2.0], [3.0, 4.0]], None, [[1.0, 2.0], [3.0]], [[5.0, 6.0]]])
    shapes, regular = nested_shapes(arr, 2)
    assert shapes[0].tolist() == [2, 2]
    assert shapes[3].tolist() == [1, 2]
    assert regular.tolist() == [True, False, False, True]


def test_group_rows_by_shape_and_dense_conversion():
    arr = pa.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0]], [[7.0, 8.0], [9.0, 0.0]]])
    groups, invalid = group_rows_by_shape(arr, 2)
    assert invalid.size == 0
    assert groups[(2, 2)].tolist() == [0, 2]
    dense = nested_to_numpy(arr.take(pa.array(groups[(2, 2)])), (2, 2))
    assert dense.shape == (2, 2, 2) and dense.dtype == np.float32
    assert dense[1, 1, 0] == 9.0


def test_tokens_to_arrow_nested_and_scalar_with_nulls():
    tokens = np.arange(6).reshape(3, 2)
    nested = pa.struct([pa.field("token", pa.list_(pa.int64()))])
    col = tokens_to_arrow(tokens, np.array([True, False, True]), nested)
    assert col.type == nested
    assert col.to_pylist() == [{"token": [0, 1]}, None, {"token": [4, 5]}]
    col = tokens_to_arrow(np.arange(3), np.array([True, False, True]), pa.int32())
    assert col.to_pylist() == [0, None, 2] and col.type == pa.int32()
