"""Mapping between Multimodal Universe catalog columns and AION modalities.

A :class:`ModalitySpec` says which catalog column feeds which AION modality, and the
adapters below turn an Arrow column into batches of AION ``Modality`` objects ready
for the codecs. :func:`detect_modalities` proposes specs by inspecting a catalog
schema, using the column names that AION itself declares for each modality.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyarrow as pa
import torch
from aion.codecs.config import MODALITY_CODEC_MAPPING
from aion.codecs.preprocessing.band_to_index import BAND_TO_INDEX
from aion.modalities import (
    DESISpectrum,
    HSCImage,
    Image,
    LegacySurveyImage,
    Modality,
    Scalar,
    SDSSSpectrum,
    Spectrum,
)

from .arrow_utils import group_rows_by_shape, nested_to_numpy, struct_field, valid_mask

log = logging.getLogger(__name__)

#: Minimum image side (in pixels) the AION image codec can center-crop to.
IMAGE_MIN_SIZE = 96

#: Field name of the token list inside a multi-token output column. Such a column is a
#: ``struct<token: list<int>>``, which LSDB / nested-pandas load as a nested column
#: (``nested<token: [int64]>``), i.e. one sub-table of tokens per object.
TOKEN_FIELD = "token"

#: AION band label prefix of each image modality (``DES-G``, ``HSC-G``, ...).
SURVEY_PREFIX: dict[type[Image], str] = {LegacySurveyImage: "DES", HSCImage: "HSC"}
_IMAGE_BY_PREFIX = {prefix: modality for modality, prefix in SURVEY_PREFIX.items()}

#: Substrings of a catalog name that identify the survey of its image/spectrum columns.
IMAGE_NAME_HINTS: dict[str, type[Image]] = {
    "hsc": HSCImage,
    "legacysurvey": LegacySurveyImage,
    "legacy_survey": LegacySurveyImage,
    "decals": LegacySurveyImage,
}
SPECTRUM_NAME_HINTS: dict[str, type[Spectrum]] = {"desi": DESISpectrum, "sdss": SDSSSpectrum}

#: Struct fields of the MMU image and spectrum columns (name -> accepted aliases).
IMAGE_FIELDS = {"bands": ("band", "bands"), "flux": ("flux",)}
SPECTRUM_FIELDS = {
    "flux": ("flux",),
    "ivar": ("ivar",),
    "wavelength": ("wavelength", "lambda"),
    "mask": ("mask",),
}
SPECTRUM_REQUIRED = ("flux", "ivar", "wavelength")


def modality_kind(modality: type[Modality]) -> str:
    """``"image"``, ``"spectrum"`` or ``"scalar"``."""
    if issubclass(modality, Image):
        return "image"
    if issubclass(modality, Spectrum):
        return "spectrum"
    if issubclass(modality, Scalar):
        return "scalar"
    raise TypeError(f"{modality.__name__} is not an image, spectrum or scalar modality")


def _supported(modality: type[Modality]) -> bool:
    """Whether the adapters below can build this modality from a catalog column."""
    if not hasattr(modality, "num_tokens"):
        return False
    try:
        kind = modality_kind(modality)
    except TypeError:
        return False
    return kind != "scalar" or modality.num_tokens == 1  # vector scalars (Gaia XP) unsupported


#: Every modality this package can tokenize, keyed by class name.
MODALITY_REGISTRY: dict[str, type[Modality]] = {
    cls.__name__: cls for cls in MODALITY_CODEC_MAPPING if _supported(cls)
}
_REGISTRY_LOWER = {name.lower(): cls for name, cls in MODALITY_REGISTRY.items()}
_SCALAR_BY_NAME = {
    cls.name.lower(): cls
    for cls in MODALITY_REGISTRY.values()
    if issubclass(cls, Scalar) and isinstance(getattr(cls, "name", None), str)
}


def get_modality(name: str) -> type[Modality]:
    """Look up an AION modality class by (case-insensitive) name."""
    try:
        return _REGISTRY_LOWER[name.lower()]
    except KeyError:
        raise KeyError(
            f"Unknown AION modality {name!r}; known modalities: {sorted(MODALITY_REGISTRY)}"
        ) from None


@dataclass(frozen=True)
class ModalitySpec:
    """One column of the catalog to tokenize with one AION modality.

    Args:
        modality: the AION modality class, e.g. ``LegacySurveyImage``.
        column: the catalog column holding the raw data.
        token_column: name of the output column; defaults to the modality's AION token
            key (``tok_image``, ``tok_flux_g``, ...). Single-token modalities produce a
            plain integer column, the others a nested ``struct<token: list<int>>`` column.
        drop_source: whether to remove ``column`` from the output. Defaults to ``True``
            for images and spectra (the bulky columns) and ``False`` for scalars.
    """

    modality: type[Modality]
    column: str
    token_column: str | None = None
    drop_source: bool | None = None

    @property
    def kind(self) -> str:
        return modality_kind(self.modality)

    @property
    def name(self) -> str:
        return self.modality.__name__

    @property
    def output_column(self) -> str:
        return self.token_column or self.modality.token_key

    @property
    def drops_source(self) -> bool:
        return self.kind != "scalar" if self.drop_source is None else self.drop_source

    @property
    def num_tokens(self) -> int:
        return int(self.modality.num_tokens)

    def arrow_type(self, token_dtype: np.dtype) -> pa.DataType:
        scalar = pa.from_numpy_dtype(token_dtype)
        if self.num_tokens == 1:
            return scalar
        return pa.struct([pa.field(TOKEN_FIELD, pa.list_(scalar))])

    def to_dict(self) -> dict[str, Any]:
        return {
            "modality": self.name,
            "column": self.column,
            "token_column": self.output_column,
            "drop_source": self.drops_source,
        }

    @classmethod
    def parse(
        cls, text: str, schema: pa.Schema | None = None, catalog_name: str = ""
    ) -> ModalitySpec:
        """Parse a CLI-style spec: ``column``, ``ModalityName`` or ``column=ModalityName``."""
        if "=" in text:
            column, name = (s.strip() for s in text.split("=", 1))
            return cls(get_modality(name), column)
        if schema is None:
            raise ValueError(f"A schema is needed to resolve {text!r}")
        if text.lower() in _REGISTRY_LOWER:
            modality = get_modality(text)
            for spec in detect_modalities(schema, catalog_name=catalog_name):
                if spec.modality is modality:
                    return spec
            raise ValueError(f"No column of the catalog matches modality {modality.__name__}")
        if text not in schema.names:
            raise ValueError(f"Column {text!r} not found in catalog columns {schema.names}")
        candidates = detect_modalities(
            pa.schema([schema.field(text)]), catalog_name=catalog_name, strict=True
        )
        if not candidates:
            raise ValueError(
                f"Cannot infer an AION modality for column {text!r}; "
                f"use the explicit form {text}=<ModalityName>"
            )
        return candidates[0]

    def __str__(self) -> str:
        return f"{self.column} -> {self.name} ({self.output_column})"


# --------------------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------------------


def _has_fields(dtype: pa.DataType, fields: dict[str, tuple[str, ...]], required) -> bool:
    if not pa.types.is_struct(dtype):
        return False
    names = set(dtype.names)
    return all(any(alias in names for alias in fields[key]) for key in required)


def image_modality_for(
    catalog_name: str = "", bands: list[str] | None = None
) -> type[Image] | None:
    """Image modality from the catalog name, else from the survey prefix of the band labels."""
    name = catalog_name.lower()
    for hint, modality in IMAGE_NAME_HINTS.items():
        if hint in name:
            return modality
    if bands:
        prefix = bands[0].upper().split("-")[0]
        return _IMAGE_BY_PREFIX.get(prefix)
    return None


def spectrum_modality_for(catalog_name: str = "") -> type[Spectrum] | None:
    name = catalog_name.lower()
    return next((m for hint, m in SPECTRUM_NAME_HINTS.items() if hint in name), None)


def _sample_bands(sample, column: str) -> list[str] | None:
    """Band labels of the first row of ``column`` in ``sample`` (a table or a callable)."""
    table = sample() if callable(sample) else sample
    if table is None or not len(table):
        return None
    col = table.column(column)
    if isinstance(col, pa.ChunkedArray):
        col = col.combine_chunks()
    band_col = struct_field(col, *IMAGE_FIELDS["bands"])
    if band_col is None or not band_col[0].is_valid:
        return None
    return list(band_col[0].as_py())


def detect_modalities(
    schema: pa.Schema,
    *,
    catalog_name: str = "",
    sample: pa.Table | pa.RecordBatch | Callable[[], pa.Table] | None = None,
    strict: bool = False,
) -> list[ModalitySpec]:
    """Propose modality specs for the columns of a catalog.

    Image and spectrum struct columns are recognised from their fields; the survey is
    taken from ``catalog_name`` (e.g. ``mmu_hsc_pdr3_dud_22.5`` -> ``HSCImage``) or, for
    images, from the band labels of ``sample`` (a table, or a callable returning one so
    that no data is read when the name is enough). Scalar columns are matched against the
    column names AION declares for its scalar modalities (``FLUX_G``, ``EBV``, ``Z``,
    ``g_cmodel_mag``, ...), case-insensitively.

    Args:
        strict: raise instead of skipping a column whose survey cannot be determined.
    """
    specs: list[ModalitySpec] = []
    for field in schema:
        if _has_fields(field.type, IMAGE_FIELDS, IMAGE_FIELDS):
            modality = image_modality_for(catalog_name)
            if modality is None and sample is not None:
                modality = image_modality_for(bands=_sample_bands(sample, field.name))
            choices = "Legacy Survey or HSC", "LegacySurveyImage or HSCImage"
        elif _has_fields(field.type, SPECTRUM_FIELDS, SPECTRUM_REQUIRED):
            modality = spectrum_modality_for(catalog_name)
            choices = "DESI or SDSS", "DESISpectrum or SDSSSpectrum"
        elif pa.types.is_floating(field.type) or pa.types.is_integer(field.type):
            modality = _SCALAR_BY_NAME.get(field.name.lower())
            if modality is not None:
                specs.append(ModalitySpec(modality, field.name))
            continue
        else:
            continue
        if modality is None:
            msg = (
                f"Cannot tell whether column {field.name!r} is {choices[0]}; "
                f"pass {field.name}=<{choices[1]}>"
            )
            if strict:
                raise ValueError(msg)
            log.warning(msg)
            continue
        specs.append(ModalitySpec(modality, field.name))
    return specs


def resolve_modalities(
    schema: pa.Schema,
    modalities: str | list[str | ModalitySpec] | None,
    *,
    catalog_name: str = "",
    sample: pa.Table | pa.RecordBatch | Callable[[], pa.Table] | None = None,
) -> list[ModalitySpec]:
    """Turn the user's ``modalities`` argument into validated :class:`ModalitySpec` objects.

    ``None`` or ``"auto"`` detects everything that can be tokenized; otherwise each item
    is a :class:`ModalitySpec` or a string understood by :meth:`ModalitySpec.parse`.
    """
    if modalities is None or modalities == "auto":
        specs = detect_modalities(schema, catalog_name=catalog_name, sample=sample)
    else:
        if isinstance(modalities, (str, ModalitySpec)):
            modalities = [modalities]
        specs = [
            m if isinstance(m, ModalitySpec) else ModalitySpec.parse(m, schema, catalog_name)
            for m in modalities
        ]
    if not specs:
        raise ValueError(f"No column to tokenize was found in schema {schema.names}")
    kept = set(schema.names) - {s.column for s in specs if s.drops_source}
    seen: set[str] = set()
    for spec in specs:
        if spec.column not in schema.names:
            raise ValueError(f"Column {spec.column!r} not in catalog columns {schema.names}")
        if spec.output_column in seen or spec.output_column in kept:
            raise ValueError(
                f"Output column {spec.output_column!r} would clash with another column"
            )
        seen.add(spec.output_column)
    return specs


# --------------------------------------------------------------------------------------
# Adapters: Arrow column -> list of (row indices, Modality)
# --------------------------------------------------------------------------------------

ModalityBatch = tuple[np.ndarray, Modality]


def normalize_band(label: str, modality: type[Image]) -> str:
    """Map a catalog band label (``des-g``, ``g``) onto an AION label (``DES-G``)."""
    band = label.upper().replace("_", "-")
    if "-" not in band:
        band = f"{SURVEY_PREFIX[modality]}-{band}"
    if band not in BAND_TO_INDEX:
        raise ValueError(
            f"Band {label!r} is not supported by AION; known bands: {list(BAND_TO_INDEX)}"
        )
    return band


def _to_tensor(values: np.ndarray, device: torch.device | str, dtype=torch.float32) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(values)).to(device=device, dtype=dtype)


def _take(column: pa.Array, rows: np.ndarray) -> pa.Array:
    """``column.take(rows)``, without copying when ``rows`` is the whole column."""
    if rows.size == len(column) and rows[0] == 0 and rows[-1] == len(column) - 1:
        return column
    return column.take(pa.array(rows))


def image_batches(spec: ModalitySpec, column: pa.Array, device) -> list[ModalityBatch]:
    """Group the rows of an image struct column by band set and shape."""
    bands_col = struct_field(column, *IMAGE_FIELDS["bands"])
    flux_col = struct_field(column, *IMAGE_FIELDS["flux"])
    if bands_col is None or flux_col is None:
        raise ValueError(f"Column {spec.column!r} needs 'band' and 'flux' fields")
    struct_valid = valid_mask(column)
    groups, invalid = group_rows_by_shape(flux_col, 3)
    if invalid.size:
        log.debug("%s: %d rows with null or ragged images", spec.column, invalid.size)
    band_lists = bands_col.to_pylist()
    batches: list[ModalityBatch] = []
    for shape, rows in groups.items():
        rows = rows[struct_valid[rows]]
        if rows.size == 0:
            continue
        n_bands, height, width = shape
        if height < IMAGE_MIN_SIZE or width < IMAGE_MIN_SIZE:
            log.warning(
                "%s: skipping %d images smaller than %d px", spec.column, rows.size, IMAGE_MIN_SIZE
            )
            continue
        by_bands: dict[tuple[str, ...], list[int]] = {}
        for row in rows:
            labels = band_lists[row]
            if labels is not None and len(labels) == n_bands:
                by_bands.setdefault(tuple(labels), []).append(int(row))
        for labels, sub_rows in by_bands.items():
            idx = np.asarray(sub_rows, dtype=np.int64)
            flux = _to_tensor(nested_to_numpy(_take(flux_col, idx), shape), device)
            torch.nan_to_num_(flux, nan=0.0, posinf=0.0, neginf=0.0)
            modality = spec.modality(
                flux=flux, bands=[normalize_band(b, spec.modality) for b in labels]
            )
            batches.append((idx, modality))
    return batches


def spectrum_batches(spec: ModalitySpec, column: pa.Array, device) -> list[ModalityBatch]:
    """Group the rows of a spectrum struct column by length."""
    cols = {key: struct_field(column, *aliases) for key, aliases in SPECTRUM_FIELDS.items()}
    if any(cols[key] is None for key in SPECTRUM_REQUIRED):
        raise ValueError(f"Column {spec.column!r} needs 'flux', 'ivar' and 'lambda' fields")
    struct_valid = valid_mask(column)
    groups, _ = group_rows_by_shape(cols["flux"], 1)
    batches: list[ModalityBatch] = []
    for shape, rows in groups.items():
        rows = rows[struct_valid[rows]]
        if rows.size == 0:
            continue
        flux = nested_to_numpy(_take(cols["flux"], rows), shape)
        ivar = nested_to_numpy(_take(cols["ivar"], rows), shape)
        wavelength = nested_to_numpy(_take(cols["wavelength"], rows), shape)
        mask = ~np.isfinite(flux) | ~np.isfinite(ivar) | (ivar <= 0)
        if cols["mask"] is not None:
            mask |= nested_to_numpy(_take(cols["mask"], rows), shape, dtype=bool)
        modality = spec.modality(
            flux=_to_tensor(flux, device),
            ivar=_to_tensor(ivar, device),
            mask=_to_tensor(mask, device, dtype=torch.bool),
            wavelength=_to_tensor(wavelength, device),
        )
        batches.append((rows, modality))
    return batches


def scalar_batches(spec: ModalitySpec, column: pa.Array, device) -> list[ModalityBatch]:
    """Build one modality from the finite values of a numeric column."""
    values = np.asarray(column.to_numpy(zero_copy_only=False), dtype=np.float32)
    rows = np.flatnonzero(valid_mask(column) & np.isfinite(values))
    if rows.size == 0:
        return []
    return [(rows, spec.modality(value=_to_tensor(values[rows], device)))]


ADAPTERS: dict[str, Callable[[ModalitySpec, pa.Array, Any], list[ModalityBatch]]] = {
    "image": image_batches,
    "spectrum": spectrum_batches,
    "scalar": scalar_batches,
}


def build_modality_batches(spec: ModalitySpec, column: pa.Array, device) -> list[ModalityBatch]:
    """Dispatch to the adapter for the spec's modality kind."""
    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()
    return ADAPTERS[spec.kind](spec, column, device)
