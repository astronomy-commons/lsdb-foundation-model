"""Shared fixtures: a synthetic MMU-style HATS catalog and a fake codec manager."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from aion.modalities import Image, Scalar, Spectrum

from aion_hats.catalog import Partition, finalize_catalog, format_properties

IMAGE_SHAPE = (3, 100, 100)  # >= 96 px so the real codec could crop it
SPECTRUM_LENGTH = 40
BANDS = ["des-g", "des-r", "des-z"]


def make_rows(n: int, seed: int) -> pa.Table:
    """``n`` rows shaped like UniverseTBD/mmu_ssl_legacysurvey_north plus a spectrum."""
    rng = np.random.default_rng(seed)
    flux = rng.normal(size=(n, *IMAGE_SHAPE)).astype(np.float32)
    images = [
        {"band": BANDS, "flux": flux[i].tolist(), "psf_fwhm": [1.0, 1.1, 1.2], "scale": [0.262] * 3}
        for i in range(n)
    ]
    images[1] = None  # a missing image
    spec_flux = rng.normal(size=(n, SPECTRUM_LENGTH)).astype(np.float32)
    spectra = [
        {
            "flux": spec_flux[i].tolist(),
            "ivar": np.full(SPECTRUM_LENGTH, 2.0, np.float32).tolist(),
            "lambda": np.linspace(3600, 9800, SPECTRUM_LENGTH, dtype=np.float32).tolist(),
            "mask": [False] * SPECTRUM_LENGTH,
        }
        for i in range(n)
    ]
    flux_g = rng.uniform(0.1, 10, n).astype(np.float32)
    flux_g[0] = np.nan  # a missing scalar
    return pa.table(
        {
            "_healpix_29": pa.array(rng.integers(0, 2**60, n), pa.int64()),
            "image": pa.array(images),
            "spectrum": pa.array(spectra),
            "ebv": pa.array(rng.uniform(0, 0.1, n).astype(np.float32)),
            "flux_g": pa.array(flux_g),
            "z_spec": pa.array(rng.uniform(0, 1, n).astype(np.float32)),
            "ra": pa.array(rng.uniform(0, 360, n)),
            "dec": pa.array(rng.uniform(-90, 90, n)),
            "object_id": pa.array([f"obj{seed}_{i}" for i in range(n)]),
        }
    )


PARTITIONS = {Partition(4, 119): 10, Partition(5, 1239): 7, Partition(6, 5000): 5}


@pytest.fixture(scope="session")
def synthetic_catalog(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("catalogs") / "mmu_test_legacysurvey"
    dataset = root / "dataset"
    for seed, (partition, n_rows) in enumerate(PARTITIONS.items()):
        table = make_rows(n_rows, seed)
        path = dataset / partition.relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path, row_group_size=4)
    props = {
        "obs_collection": "mmu_test_legacysurvey",
        "dataproduct_type": "object",
        "hats_nrows": sum(PARTITIONS.values()),
        "hats_col_ra": "ra",
        "hats_col_dec": "dec",
        "hats_col_healpix": "_healpix_29",
        "hats_col_healpix_order": 29,
        "hats_max_rows": 8192,
        "hats_order": 6,
        "hats_skymap_order": 10,
        "hats_estsize": 1234,
    }
    (root / "hats.properties").write_text(format_properties(props))
    finalize_catalog(root, write_readme=False)  # partition_info.csv, _metadata, _common_metadata
    return root


class FakeCodecManager:
    """Deterministic stand-in for ``aion.codecs.CodecManager`` (no weights, no network)."""

    def __init__(self, device="cpu"):
        self.device = device
        self.calls: list[str] = []

    def encode(self, *modalities):
        out = {}
        for m in modalities:
            self.calls.append(type(m).__name__)
            n_tokens = type(m).num_tokens
            if isinstance(m, Image):
                key = m.flux.double().flatten(1).sum(1)
                assert m.flux.ndim == 4 and all("-" in b for b in m.bands)
            elif isinstance(m, Spectrum):
                key = torch.nan_to_num(m.flux).double().sum(1)
                assert m.mask.dtype == torch.bool and m.wavelength.shape == m.flux.shape
            elif isinstance(m, Scalar):
                key = m.value.double()
            else:
                raise TypeError(type(m))
            base = torch.round(key.abs() * 1000).long() % 4000
            tokens = (base[:, None] + torch.arange(n_tokens)) % 4096
            out[type(m).token_key] = tokens if n_tokens > 1 else tokens[:, 0]
        return out


@pytest.fixture
def fake_codecs() -> FakeCodecManager:
    return FakeCodecManager()
