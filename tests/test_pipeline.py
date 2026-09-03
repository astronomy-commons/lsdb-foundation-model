import json

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest
from aion.modalities import LegacySurveyImage
from conftest import PARTITIONS, make_rows

from aion_hats import (
    AionTokenizer,
    ModalitySpec,
    Partition,
    finalize_catalog,
    open_catalog,
    tokenize_catalog,
)
from aion_hats.catalog import read_properties

TOTAL_ROWS = sum(PARTITIONS.values())
NESTED_INT64 = pa.struct([pa.field("token", pa.list_(pa.int64()))])


def read_output(output):
    return ds.dataset(
        str(output / "dataset"), format="parquet", exclude_invalid_files=True
    ).to_table()


def test_open_catalog_local(synthetic_catalog):
    cat = open_catalog(synthetic_catalog)
    assert cat.is_hats and cat.is_local
    assert cat.name == "mmu_test_legacysurvey"
    assert cat.partitions == sorted(PARTITIONS)
    assert cat.schema.names[:3] == ["_healpix_29", "image", "spectrum"]
    assert cat.sample(3).num_rows == 3
    assert Partition.parse("Norder=4/Npix=119") == Partition(4, 119)
    assert Partition(5, 12345).relpath == "Norder=5/Dir=10000/Npix=12345.parquet"


def test_tokenizer_on_table(fake_codecs):
    table = make_rows(5, 3)
    tok = AionTokenizer(
        [ModalitySpec(LegacySurveyImage, "image"), ModalitySpec.parse("flux_g=LegacySurveyFluxG")],
        device="cpu",
        codec_manager=fake_codecs,
    )
    out = tok.tokenize_table(table, batch_size=2)
    assert "image" not in out.column_names and "flux_g" in out.column_names
    assert out.schema.field("tok_image").type == NESTED_INT64
    assert out.schema.field("tok_flux_g").type == pa.int64()
    tokens = out.column("tok_image").to_pylist()
    assert tokens[1] is None and len(tokens[0]["token"]) == LegacySurveyImage.num_tokens
    assert out.column("tok_flux_g").to_pylist()[0] is None  # NaN input
    # deterministic and independent of batching
    again = tok.tokenize_table(table, batch_size=5)
    for name in ("tok_image", "tok_flux_g"):
        assert again.column(name).to_pylist() == out.column(name).to_pylist()


def test_tokenize_catalog_end_to_end(synthetic_catalog, tmp_path, fake_codecs):
    output = tmp_path / "tokens"
    summary = tokenize_catalog(
        synthetic_catalog,
        output,
        ["image", "flux_g", "spectrum=DESISpectrum"],
        codec_manager=fake_codecs,
        device="cpu",
        batch_size=4,
        row_group_size=6,
        progress=False,
    )
    assert summary.ok and summary.rows == TOTAL_ROWS and summary.finalized
    assert sorted(summary.done) == sorted(p.label for p in PARTITIONS)
    for partition in PARTITIONS:
        assert (output / "dataset" / partition.relpath).exists()
    table = read_output(output)
    assert table.num_rows == TOTAL_ROWS
    assert set(table.column_names) == {
        "_healpix_29",
        "ebv",
        "flux_g",
        "z_spec",
        "ra",
        "dec",
        "object_id",
        "tok_image",
        "tok_flux_g",
        "tok_spectrum_desi",
    }
    assert table.column("tok_spectrum_desi").type == NESTED_INT64
    assert all(len(t["token"]) == 273 for t in table.column("tok_spectrum_desi").to_pylist())

    # finalize products
    assert (output / "partition_info.csv").read_text().splitlines()[0] == "Norder,Npix"
    props = read_properties((output / "hats.properties").read_text())
    assert props["hats_nrows"] == str(TOTAL_ROWS) and props["obs_collection"] == "tokens"
    assert "hats_skymap_order" not in props and props["aion_hats_modalities"].startswith("image=")
    assert pq.read_metadata(output / "dataset" / "_metadata").num_rows == TOTAL_ROWS
    prov = json.loads((output / "aion_hats.json").read_text())
    assert prov["modalities"][0] == {
        "modality": "LegacySurveyImage",
        "column": "image",
        "token_column": "tok_image",
        "drop_source": True,
    }
    readme = (output / "README.md").read_text()
    assert readme.startswith("---\n") and "data_files: dataset/**/*.parquet" in readme

    # the output is a HATS catalog our own reader (and lsdb, if present) can open
    out_cat = open_catalog(output)
    assert out_cat.partitions == sorted(PARTITIONS) and "tok_image" in out_cat.schema.names
    lsdb = pytest.importorskip("lsdb")
    cat = lsdb.open_catalog(output)
    df = cat.compute()
    assert len(df) == TOTAL_ROWS and str(df.dtypes["tok_image"]) == "nested<token: [int64]>"
    assert df["tok_image.token"].shape[0] == (TOTAL_ROWS - 3) * LegacySurveyImage.num_tokens

    # re-running skips everything; overwrite redoes it
    before = fake_codecs.calls[:]
    summary = tokenize_catalog(
        synthetic_catalog,
        output,
        ["image"],
        codec_manager=fake_codecs,
        device="cpu",
        progress=False,
    )
    assert (
        summary.done == []
        and len(summary.skipped) == len(PARTITIONS)
        and fake_codecs.calls == before
    )
    summary = tokenize_catalog(
        synthetic_catalog,
        output,
        ["image"],
        codec_manager=fake_codecs,
        device="cpu",
        progress=False,
        overwrite=True,
    )
    assert len(summary.done) == len(PARTITIONS)


def test_sharding_covers_every_partition_once(synthetic_catalog, tmp_path, fake_codecs):
    output = tmp_path / "sharded"
    summaries = [
        tokenize_catalog(
            synthetic_catalog,
            output,
            ["image"],
            codec_manager=fake_codecs,
            device="cpu",
            rank=r,
            world_size=2,
            progress=False,
        )
        for r in range(2)
    ]
    assert not any(s.finalized for s in summaries)
    done = [label for s in summaries for label in s.done]
    assert sorted(done) == sorted(p.label for p in PARTITIONS) and len(set(done)) == len(done)
    assert not (output / "partition_info.csv").exists()
    fin = finalize_catalog(output)
    assert fin.rows == TOTAL_ROWS and fin.partitions == len(PARTITIONS)


def test_rank_from_environment(synthetic_catalog, tmp_path, fake_codecs, monkeypatch):
    monkeypatch.setenv("SLURM_PROCID", "1")
    monkeypatch.setenv("SLURM_NTASKS", "3")
    summary = tokenize_catalog(
        synthetic_catalog,
        tmp_path / "env",
        ["image"],
        codec_manager=fake_codecs,
        device="cpu",
        progress=False,
    )
    assert (summary.rank, summary.world_size) == (1, 3)
    assert summary.done == [sorted(PARTITIONS)[1].label]


def test_max_rows_and_partition_selection(synthetic_catalog, tmp_path, fake_codecs):
    output = tmp_path / "demo"
    summary = tokenize_catalog(
        synthetic_catalog,
        output,
        "auto",
        codec_manager=fake_codecs,
        device="cpu",
        max_rows=12,
        batch_size=5,
        progress=False,
    )
    assert summary.rows == 12 and len(summary.done) == 2  # 10 rows + 2 of the next partition
    assert read_output(output).num_rows == 12
    assert "tok_ebv" in read_output(output).column_names  # auto picked scalars too

    summary = tokenize_catalog(
        synthetic_catalog,
        tmp_path / "one",
        ["image"],
        codec_manager=fake_codecs,
        device="cpu",
        partitions=["Norder=6/Npix=5000"],
        progress=False,
    )
    assert summary.done == ["Norder=6/Npix=5000"] and summary.rows == PARTITIONS[Partition(6, 5000)]
    with pytest.raises(ValueError, match="not in catalog"):
        tokenize_catalog(
            synthetic_catalog,
            tmp_path / "bad",
            ["image"],
            codec_manager=fake_codecs,
            partitions=["Norder=1/Npix=1"],
        )


def test_failed_partition_is_reported_not_fatal(
    synthetic_catalog, tmp_path, fake_codecs, monkeypatch
):
    calls = {"n": 0}
    real_encode = fake_codecs.encode

    def flaky(*mods):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return real_encode(*mods)

    monkeypatch.setattr(fake_codecs, "encode", flaky)
    output = tmp_path / "flaky"
    summary = tokenize_catalog(
        synthetic_catalog,
        output,
        ["image"],
        codec_manager=fake_codecs,
        device="cpu",
        progress=False,
    )
    assert not summary.ok and list(summary.failed) == [min(PARTITIONS).label]
    assert len(summary.done) == 2 and not summary.finalized
    assert not (output / "dataset" / min(PARTITIONS).relpath).exists()
    assert not list((output / "dataset").rglob("*.tmp"))
    with pytest.raises(RuntimeError, match="boom"):
        calls["n"] = 0
        tokenize_catalog(
            synthetic_catalog,
            tmp_path / "ff",
            ["image"],
            codec_manager=fake_codecs,
            device="cpu",
            progress=False,
            fail_fast=True,
        )


def test_cli(synthetic_catalog, tmp_path, fake_codecs, monkeypatch, capsys):
    from aion_hats import cli

    monkeypatch.setattr("aion.codecs.CodecManager", lambda device: fake_codecs)
    assert cli.main(["inspect", str(synthetic_catalog)]) == 0
    out = capsys.readouterr().out
    assert "image -> LegacySurveyImage (tok_image)" in out
    output = tmp_path / "cli"
    code = cli.main(
        [
            "tokenize",
            str(synthetic_catalog),
            str(output),
            "-m",
            "image",
            "-m",
            "flux_g",
            "--device",
            "cpu",
            "--no-progress",
            "--token-dtype",
            "int32",
        ]
    )
    assert code == 0 and (output / "partition_info.csv").exists()
    assert read_output(output).column("tok_image").type == pa.struct(
        [pa.field("token", pa.list_(pa.int32()))]
    )
    assert cli.main(["finalize", str(output)]) == 0
