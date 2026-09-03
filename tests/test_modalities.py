import pytest
import torch
from aion.modalities import (
    DESISpectrum,
    HSCImage,
    LegacySurveyEBV,
    LegacySurveyFluxG,
    LegacySurveyImage,
    SDSSSpectrum,
    Z,
)
from conftest import make_rows

from aion_hats import ModalitySpec, detect_modalities, resolve_modalities
from aion_hats.modalities import image_batches, normalize_band, scalar_batches, spectrum_batches


def test_detect_modalities_uses_catalog_name_and_aion_column_names():
    table = make_rows(3, 0)
    specs = detect_modalities(table.schema, catalog_name="mmu_desi_legacysurvey", sample=table)
    found = {s.column: s.modality for s in specs}
    assert found["image"] is LegacySurveyImage
    assert found["spectrum"] is DESISpectrum
    assert found["flux_g"] is LegacySurveyFluxG
    assert found["ebv"] is LegacySurveyEBV
    assert "z_spec" not in found  # AION's redshift column is called Z
    assert {"ra", "dec"} <= set(found)


def test_detect_modalities_reads_bands_from_sample_and_skips_ambiguous_spectra():
    table = make_rows(2, 0)
    specs = detect_modalities(table.schema, catalog_name="mystery", sample=table)
    found = {s.column: s.modality for s in specs}
    assert found["image"] is LegacySurveyImage  # from the des-* band labels
    assert "spectrum" not in found
    with pytest.raises(ValueError, match="DESI or SDSS"):
        detect_modalities(table.schema, catalog_name="mystery", sample=table, strict=True)
    specs = detect_modalities(table.schema, catalog_name="mmu_hsc_sdss")
    found = {s.column: s.modality for s in specs}
    assert found["image"] is HSCImage and found["spectrum"] is SDSSSpectrum


def test_resolve_modalities_string_forms():
    table = make_rows(2, 0)
    specs = resolve_modalities(
        table.schema,
        ["image", "LegacySurveyFluxG", "z_spec=Z", "spectrum=SDSSSpectrum"],
        catalog_name="mmu_ssl_legacysurvey_north",
        sample=table,
    )
    assert [(s.column, s.modality, s.output_column) for s in specs] == [
        ("image", LegacySurveyImage, "tok_image"),
        ("flux_g", LegacySurveyFluxG, "tok_flux_g"),
        ("z_spec", Z, "tok_z"),
        ("spectrum", SDSSSpectrum, "tok_spectrum_sdss"),
    ]
    assert specs[0].drops_source and not specs[1].drops_source
    with pytest.raises(ValueError, match="not found"):
        resolve_modalities(table.schema, ["nope"])
    with pytest.raises(KeyError, match="Unknown AION modality"):
        resolve_modalities(table.schema, ["flux_g=FluxQ"])
    with pytest.raises(ValueError, match="clash"):
        resolve_modalities(table.schema, [ModalitySpec(Z, "z_spec", token_column="ra")])


def test_normalize_band():
    assert normalize_band("des-g", LegacySurveyImage) == "DES-G"
    assert normalize_band("g", HSCImage) == "HSC-G"
    with pytest.raises(ValueError, match="not supported"):
        normalize_band("jwst-f200w", LegacySurveyImage)


def test_adapters_group_rows_and_skip_invalid_ones():
    table = make_rows(4, 1)
    spec = ModalitySpec(LegacySurveyImage, "image")
    batches = image_batches(spec, table.column("image").combine_chunks(), "cpu")
    assert len(batches) == 1
    rows, modality = batches[0]
    assert rows.tolist() == [0, 2, 3]  # row 1 has no image
    assert tuple(modality.flux.shape) == (3, 3, 100, 100) and modality.bands == [
        "DES-G",
        "DES-R",
        "DES-Z",
    ]

    ((rows, spectrum),) = spectrum_batches(
        ModalitySpec(DESISpectrum, "spectrum"), table.column("spectrum").combine_chunks(), "cpu"
    )
    assert rows.tolist() == [0, 1, 2, 3] and spectrum.wavelength.shape == (4, 40)

    ((rows, scalar),) = scalar_batches(
        ModalitySpec(LegacySurveyFluxG, "flux_g"), table.column("flux_g").combine_chunks(), "cpu"
    )
    assert rows.tolist() == [1, 2, 3]  # row 0 is NaN
    assert scalar.value.dtype == torch.float32
