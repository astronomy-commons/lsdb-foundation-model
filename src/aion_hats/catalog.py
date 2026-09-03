"""Reading HATS catalogs (local, Hugging Face, or any fsspec URL) and writing them back.

A HATS catalog is a directory with ``hats.properties``, ``partition_info.csv`` and a
``dataset/`` tree of parquet files named ``Norder=<k>/Dir=<d>/Npix=<p>.parquet``. Each of
those files is one HEALPix partition and is the unit of work for the tokenization
pipeline. The tokenized output keeps the same layout so it is again a HATS catalog
that ``lsdb`` can open and cross-match with the input.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import logging
import os
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq

log = logging.getLogger(__name__)

DATASET_DIR = "dataset"
PARTITION_GLOB = "Norder=*/Dir=*/Npix=*.parquet"
PARTITION_RE = re.compile(r"Norder=(\d+)/Dir=(\d+)/Npix=(\d+)\.parquet$")
HF_REPO_RE = re.compile(r"^[\w.-]+/[\w.-]+$")
DIR_SIZE = 10_000
DOWNLOAD_RETRIES = 3
STREAM_BLOCK_SIZE = 64 << 20  # bytes read per remote range request when streaming


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True, order=True)
class Partition:
    """One HEALPix partition of a HATS catalog."""

    order: int
    pixel: int

    @property
    def directory(self) -> int:
        return (self.pixel // DIR_SIZE) * DIR_SIZE

    @property
    def relpath(self) -> str:
        return f"Norder={self.order}/Dir={self.directory}/Npix={self.pixel}.parquet"

    @property
    def label(self) -> str:
        return f"Norder={self.order}/Npix={self.pixel}"

    @classmethod
    def parse(cls, text: str) -> Partition:
        """Parse ``Norder=4/Npix=1005``, ``4/1005`` or a full partition path."""
        match = PARTITION_RE.search(text)
        if match:
            return cls(int(match.group(1)), int(match.group(3)))
        numbers = re.findall(r"\d+", text)
        if len(numbers) == 2:
            return cls(int(numbers[0]), int(numbers[1]))
        raise ValueError(f"Cannot parse partition {text!r}; expected Norder=<k>/Npix=<p>")


def partition_file(root: Path, partition: Partition) -> Path:
    """Path of a partition inside a local catalog directory."""
    return root / DATASET_DIR / partition.relpath


def read_properties(text: str) -> dict[str, str]:
    """Parse a ``hats.properties`` file (Java properties syntax)."""
    props: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "!")) or "=" not in line:
            continue
        key, value = line.split("=", 1)
        props[key.strip()] = value.strip()
    return props


def format_properties(props: dict[str, Any], header: str = "HATS catalog") -> str:
    lines = [f"#{header}"] + [f"{k}={v}" for k, v in props.items() if v is not None]
    return "\n".join(lines) + "\n"


def _normalize_source(source: str | os.PathLike) -> str:
    source = os.fspath(source)
    if "://" not in source and not Path(source).exists() and HF_REPO_RE.match(source):
        log.info(
            "%s is not a local path; reading it as the Hugging Face dataset hf://datasets/%s",
            source,
            source,
        )
        return f"hf://datasets/{source}"
    return source


class HatsCatalog:
    """A HATS catalog opened for reading, wherever it lives.

    Use :func:`open_catalog` rather than the constructor.
    """

    def __init__(self, fs: fsspec.AbstractFileSystem, root: str, url: str):
        self.fs = fs
        self.root = root.rstrip("/")
        self.url = url
        self.name = PurePosixPath(self.root).name
        entries = {PurePosixPath(p).name for p in fs.ls(self.root, detail=False)}
        self.is_hats = "hats.properties" in entries
        self.properties: dict[str, str] = {}
        if self.is_hats:
            self.properties = read_properties(fs.read_text(f"{self.root}/hats.properties"))
            self.name = self.properties.get("obs_collection", self.name)
        self.dataset_dir = f"{self.root}/{DATASET_DIR}"
        self.partitions: list[Partition] = self._list_partitions("partition_info.csv" in entries)
        self._schema: pa.Schema | None = None

    # -- helpers ---------------------------------------------------------------------

    @property
    def protocol(self) -> str:
        proto = self.fs.protocol
        return proto[0] if isinstance(proto, (tuple, list)) else proto

    @property
    def is_local(self) -> bool:
        return self.protocol in ("file", "local")

    def _list_partitions(self, has_partition_info: bool) -> list[Partition]:
        if has_partition_info:
            text = self.fs.read_text(f"{self.root}/partition_info.csv")
            rows = csv.DictReader(text.splitlines())
            return sorted(Partition(int(r["Norder"]), int(r["Npix"])) for r in rows)
        found = self.fs.glob(f"{self.dataset_dir}/{PARTITION_GLOB}")
        parts = [Partition.parse(p) for p in found if PARTITION_RE.search(p)]
        if not parts:
            raise FileNotFoundError(
                f"{self.url} is not a HATS catalog: no partition_info.csv and no "
                f"{DATASET_DIR}/{PARTITION_GLOB} files"
            )
        return sorted(parts)

    def partition_path(self, partition: Partition) -> str:
        return f"{self.dataset_dir}/{partition.relpath}"

    # -- schema ----------------------------------------------------------------------

    @property
    def schema(self) -> pa.Schema:
        """Arrow schema, from ``_common_metadata`` if present, else the first partition."""
        if self._schema is None:
            names = {PurePosixPath(p).name for p in self.fs.ls(self.dataset_dir, detail=False)}
            for name in ("_common_metadata", "_metadata"):
                if name in names:
                    self._schema = pq.read_schema(f"{self.dataset_dir}/{name}", filesystem=self.fs)
                    break
            else:
                reader = self.open_partition(self.partitions[0])
                self._schema = reader.schema_arrow
                reader.close(force=True)
        return self._schema

    # -- reading ---------------------------------------------------------------------

    def open_partition(
        self, partition: Partition, local_path: str | os.PathLike | None = None
    ) -> pq.ParquetFile:
        """Open one partition for streaming reads (``ParquetFile.iter_batches``).

        Remote files are read through a large read-ahead buffer so that a row group
        costs one range request rather than one per page. Close with ``close(force=True)``.
        """
        if local_path is not None:
            return pq.ParquetFile(os.fspath(local_path))
        path = self.partition_path(partition)
        if self.is_local:
            return pq.ParquetFile(path)
        return pq.ParquetFile(
            self.fs.open(path, "rb", block_size=STREAM_BLOCK_SIZE), pre_buffer=True
        )

    def read_partition(
        self,
        partition: Partition,
        columns: list[str] | None = None,
        max_rows: int | None = None,
    ) -> pa.Table:
        """Read (the head of) one partition into memory."""
        reader = self.open_partition(partition)
        try:
            if max_rows is None:
                return reader.read(columns=columns)
            batch = next(reader.iter_batches(batch_size=max_rows, columns=columns), None)
            return pa.Table.from_batches(
                [batch] if batch is not None else [],
                schema=batch.schema if batch is not None else reader.schema_arrow,
            )
        finally:
            reader.close(force=True)

    def sample(self, n: int = 8) -> pa.Table:
        return self.read_partition(self.partitions[0], max_rows=n)

    def _fetch_file(self, src: str, target: Path, staging: Path) -> None:
        """Copy one remote file to ``target`` (atomically), by the best means for the protocol."""
        if self.protocol == "hf":
            from huggingface_hub import hf_hub_download

            resolved = self.fs.resolve_path(src)
            path = hf_hub_download(
                resolved.repo_id,
                resolved.path_in_repo,
                repo_type=resolved.repo_type,
                revision=resolved.revision,
                local_dir=staging,
            )
            os.replace(path, target)
        else:
            tmp = target.with_suffix(".part")
            self.fs.get_file(src, str(tmp))
            os.replace(tmp, target)

    def download_partition(self, partition: Partition, local_dir: str | os.PathLike) -> Path:
        """Copy one partition file to ``local_dir`` (no-op for local catalogs)."""
        src = self.partition_path(partition)
        if self.is_local:
            return Path(src)
        local_dir = Path(local_dir)
        target = local_dir / partition.relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        last_error: Exception | None = None
        for attempt in range(1, DOWNLOAD_RETRIES + 1):
            try:
                self._fetch_file(src, target, local_dir / "_staging")
                return target
            except Exception as err:  # noqa: BLE001 - retried, then re-raised
                last_error = err
                log.warning(
                    "Download of %s failed (attempt %d/%d): %s",
                    partition.label,
                    attempt,
                    DOWNLOAD_RETRIES,
                    err,
                )
                time.sleep(min(60, 5 * attempt))
        raise RuntimeError(f"Could not download {partition.label}") from last_error

    def __repr__(self) -> str:
        return f"HatsCatalog({self.url!r}, {len(self.partitions)} partitions)"


def open_catalog(source: str | os.PathLike, catalog: str | None = None) -> HatsCatalog:
    """Open a HATS catalog from a local path, an ``hf://`` URL, a Hugging Face dataset id
    (``UniverseTBD/mmu_ssl_legacysurvey_north``) or any fsspec URL.

    Hugging Face MMU repositories are HATS *collections*: ``collection.properties`` points
    at the main catalog directory, which is what gets opened unless ``catalog`` names
    another one (e.g. the margin cache).
    """
    url = _normalize_source(source)
    fs, root = fsspec.core.url_to_fs(url)
    root = root.rstrip("/")
    if catalog is not None:
        root = f"{root}/{catalog}"
    else:
        entries = {PurePosixPath(p).name for p in fs.ls(root, detail=False)}
        if "collection.properties" in entries and "hats.properties" not in entries:
            props = read_properties(fs.read_text(f"{root}/collection.properties"))
            root = f"{root}/{props['hats_primary_table_url']}"
    return HatsCatalog(fs, root, url)


# --------------------------------------------------------------------------------------
# Writing the output catalog
# --------------------------------------------------------------------------------------


def output_properties(
    catalog: HatsCatalog, output_name: str, extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Derive the ``hats.properties`` of the tokenized catalog from the input one."""
    from . import __version__

    props = dict(catalog.properties)
    for key in (
        "hats_estsize",
        "hats_nrows",
        "hats_skymap_order",
        "hats_skymap_alt_orders",
        "hats_primary_table_url",
    ):
        props.pop(key, None)
    props["obs_collection"] = output_name
    props["hats_builder"] = f"aion-hats v{__version__}"
    props["hats_creation_date"] = utc_now()
    props["aion_hats_source"] = catalog.url
    props.update(extra or {})
    return props


@contextmanager
def atomic_path(path: Path) -> Iterator[Path]:
    """Yield a temporary path that replaces ``path`` on success and is removed on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        yield tmp
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, path)


def write_atomic(path: Path, data: str | bytes) -> None:
    with atomic_path(path) as tmp:
        tmp.write_bytes(data) if isinstance(data, bytes) else tmp.write_text(data)


def hf_readme(name: str, source: str, specs: list[dict[str, Any]]) -> str:
    from huggingface_hub import DatasetCardData

    card = DatasetCardData(
        pretty_name=name,
        tags=["astronomy"],
        configs=[{"config_name": "default", "data_files": f"{DATASET_DIR}/**/*.parquet"}],
    )
    lines = "\n".join(
        f"- `{s['column']}` -> `{s['token_column']}` ({s['modality']})" for s in specs
    )
    return (
        f"---\n{card.to_yaml()}\n---\n\n# {name}\n\n"
        f"AION-1 tokens for [{source}]({source}), produced with "
        "[aion-hats](https://github.com/astronomy-commons/lsdb-foundation-model). "
        "The catalog keeps the HATS layout of its source, so it can be opened with "
        "`lsdb.open_catalog(...)` and joined back to the source on `_healpix_29`/`object_id`, "
        "or loaded with `datasets.load_dataset(...)`.\n\n"
        f"Tokenized columns:\n\n{lines}\n"
    )


@dataclass
class FinalizeSummary:
    output: Path
    partitions: int
    rows: int

    def __str__(self) -> str:
        return f"{self.output}: {self.partitions} partitions, {self.rows} rows"


def finalize_catalog(output: str | os.PathLike, write_readme: bool = True) -> FinalizeSummary:
    """Make ``output`` a complete HATS catalog from the partitions present on disk.

    Writes ``partition_info.csv``, ``dataset/_metadata`` and ``dataset/_common_metadata``
    (parquet footers of every partition, as ``lsdb`` expects), fills ``hats_nrows`` in
    ``hats.properties``, and adds a Hugging Face ``README.md`` so the folder can be
    uploaded as a dataset. Safe to re-run; run it once after all workers are done.
    """
    output = Path(output)
    dataset_dir = output / DATASET_DIR
    files = sorted(dataset_dir.glob(PARTITION_GLOB))
    if not files:
        raise FileNotFoundError(f"No partitions found under {dataset_dir}")
    partitions: list[Partition] = []
    metadata = []
    rows = 0
    for path in files:
        md = pq.read_metadata(path)
        md.set_file_path(str(path.relative_to(dataset_dir)))
        metadata.append(md)
        rows += md.num_rows
        partitions.append(Partition.parse(str(path)))
    partitions.sort()
    schema = pq.read_schema(files[0])

    with atomic_path(dataset_dir / "_common_metadata") as tmp:
        pq.write_metadata(schema, tmp)
    with atomic_path(dataset_dir / "_metadata") as tmp:
        pq.write_metadata(schema, tmp, metadata_collector=metadata)

    lines = ["Norder,Npix"] + [f"{p.order},{p.pixel}" for p in partitions]
    write_atomic(output / "partition_info.csv", "\n".join(lines) + "\n")

    props_path = output / "hats.properties"
    props: dict[str, Any] = read_properties(props_path.read_text()) if props_path.exists() else {}
    props.setdefault("obs_collection", output.name)
    props.setdefault("dataproduct_type", "object")
    props.setdefault("hats_col_healpix", "_healpix_29")
    props.setdefault("hats_col_healpix_order", "29")
    props.setdefault("hats_npix_suffix", ".parquet")
    props["hats_nrows"] = rows
    props["hats_order"] = max(p.order for p in partitions)
    props["hats_estsize"] = sum(p.stat().st_size for p in files) // 1024
    text = format_properties(props)
    write_atomic(props_path, text)
    write_atomic(output / "properties", text)

    provenance_path = output / "aion_hats.json"
    if write_readme and provenance_path.exists():
        prov = json.loads(provenance_path.read_text())
        readme = hf_readme(
            props["obs_collection"], prov.get("source", ""), prov.get("modalities", [])
        )
        write_atomic(output / "README.md", readme)
    summary = FinalizeSummary(output, len(partitions), rows)
    log.info("Finalized %s", summary)
    return summary
