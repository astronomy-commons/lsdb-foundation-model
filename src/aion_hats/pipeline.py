"""The partition pipeline: read a HATS catalog, tokenize, write a HATS catalog.

Every worker (rank) takes a round-robin share of the partitions, streams each one in
row batches through the :class:`AionTokenizer`, and writes the tokenized partition to
the mirrored path under ``output/dataset``. Outputs are written to a temporary file and
renamed at the end, so a partition either exists complete or not at all; existing
outputs are skipped, which makes re-running a job the resume mechanism. Remote
partitions are downloaded in a background thread while the previous one is being
tokenized, and within a partition the Arrow decoding of the next batch overlaps the
codec call on the current one.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import queue
import sys
import threading
import time
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm.auto import tqdm

from . import __version__
from .catalog import (
    HatsCatalog,
    Partition,
    atomic_path,
    finalize_catalog,
    format_properties,
    open_catalog,
    output_properties,
    partition_file,
    utc_now,
    write_atomic,
)
from .distributed import WorkerContext
from .modalities import ModalitySpec, resolve_modalities
from .tokenizer import AionTokenizer, CodecManagerLike

log = logging.getLogger(__name__)


@dataclass
class RunSummary:
    """What one worker did."""

    output: Path
    rank: int
    world_size: int
    done: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    rows: int = 0
    seconds: float = 0.0
    finalized: bool = False

    @property
    def rows_per_second(self) -> float:
        return self.rows / self.seconds if self.seconds else 0.0

    @property
    def ok(self) -> bool:
        return not self.failed

    def __str__(self) -> str:
        msg = (
            f"rank {self.rank}/{self.world_size}: {len(self.done)} partitions tokenized, "
            f"{len(self.skipped)} skipped, {len(self.failed)} failed, {self.rows} rows in "
            f"{self.seconds:.1f}s ({self.rows_per_second:.1f} rows/s) -> {self.output}"
        )
        if self.failed:
            msg += "\n  failed: " + ", ".join(f"{k} ({v})" for k, v in self.failed.items())
        return msg


# --------------------------------------------------------------------------------------
# Generic streaming helpers
# --------------------------------------------------------------------------------------


def take_rows(batches: Iterable[pa.RecordBatch], max_rows: int | None) -> Iterator[pa.RecordBatch]:
    """Yield batches until ``max_rows`` rows have been produced (the last one is sliced)."""
    total = 0
    for batch in batches:
        if max_rows is not None and total + batch.num_rows > max_rows:
            batch = batch.slice(0, max_rows - total)
        if batch.num_rows:
            yield batch
        total += batch.num_rows
        if max_rows is not None and total >= max_rows:
            return


def prefetch_iter[T](iterable: Iterable[T], depth: int = 2) -> Iterator[T]:
    """Iterate ``iterable`` in a background thread, keeping up to ``depth`` items ready.

    Exceptions raised by the producer are re-raised in the consumer; abandoning the
    consumer stops the producer.
    """
    items: queue.Queue = queue.Queue(maxsize=depth)
    stop = threading.Event()
    done = object()

    def produce():
        try:
            for item in iterable:
                while not stop.is_set():
                    try:
                        items.put(item, timeout=0.1)
                        break
                    except queue.Full:
                        continue
                if stop.is_set():
                    return
            items.put(done)
        except BaseException as err:  # noqa: BLE001 - forwarded to the consumer
            items.put(err)

    thread = threading.Thread(target=produce, name="prefetch", daemon=True)
    thread.start()
    try:
        while True:
            item = items.get()
            if item is done:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()


class PartitionFetcher:
    """Stages the partitions of a run so that tokenization never waits for the network.

    ``mode`` is ``"local"`` (read in place), ``"stream"`` (range requests straight from
    the source) or ``"download"`` (copy whole files to ``cache_dir``, ``num_prefetch``
    partitions ahead of the one being tokenized).
    """

    def __init__(
        self,
        catalog: HatsCatalog,
        mode: str,
        cache_dir: Path,
        partitions: list[Partition],
        num_prefetch: int = 1,
    ):
        self.catalog = catalog
        self.mode = mode
        self.cache_dir = cache_dir
        self.partitions = partitions
        self.num_prefetch = max(1, num_prefetch)
        self.pool = ThreadPoolExecutor(max_workers=self.num_prefetch, thread_name_prefix="fetch")
        self.futures: dict[int, Future] = {}

    def get(self, index: int) -> Path | None:
        """Local path of ``partitions[index]`` (``None`` means: stream it from the source)."""
        if self.mode == "local":
            return Path(self.catalog.partition_path(self.partitions[index]))
        if self.mode == "stream":
            return None
        for i in range(index, min(index + 1 + self.num_prefetch, len(self.partitions))):
            if i not in self.futures:
                self.futures[i] = self.pool.submit(
                    self.catalog.download_partition, self.partitions[i], self.cache_dir
                )
        return self.futures.pop(index).result()

    def release(self, local_path: Path | None) -> None:
        if self.mode == "download" and local_path is not None:
            local_path.unlink(missing_ok=True)

    def close(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)


def _choose_fetch_mode(catalog: HatsCatalog, fetch_mode: str, max_rows: int | None) -> str:
    if fetch_mode not in ("auto", "download", "stream"):
        raise ValueError(f"fetch_mode must be auto, download or stream, not {fetch_mode!r}")
    if catalog.is_local:
        return "local"
    if fetch_mode == "auto":
        return "stream" if max_rows is not None else "download"
    return fetch_mode


# --------------------------------------------------------------------------------------
# One partition
# --------------------------------------------------------------------------------------


def tokenize_partition(
    catalog: HatsCatalog,
    partition: Partition,
    tokenizer: AionTokenizer,
    out_path: Path,
    *,
    local_path: Path | None = None,
    batch_size: int = 64,
    row_group_size: int = 1024,
    max_rows: int | None = None,
    progress=None,
) -> int:
    """Tokenize one partition into ``out_path`` (written atomically). Returns the row count."""
    reader = catalog.open_partition(partition, local_path)
    out_schema = tokenizer.output_schema(reader.schema_arrow)
    batches = take_rows(reader.iter_batches(batch_size=batch_size), max_rows)
    prepared = prefetch_iter(map(tokenizer.prepare_batch, batches), depth=2)
    pending: list[pa.RecordBatch] = []
    pending_rows = 0
    total = 0
    try:
        with (
            atomic_path(out_path) as tmp_path,
            pq.ParquetWriter(tmp_path, out_schema, compression="zstd") as writer,
        ):
            for item in prepared:
                out = tokenizer.encode_prepared(item)
                pending.append(out)
                pending_rows += out.num_rows
                total += out.num_rows
                if progress is not None:
                    progress.update(out.num_rows)
                if pending_rows >= row_group_size:
                    writer.write_table(pa.Table.from_batches(pending, schema=out_schema))
                    pending, pending_rows = [], 0
            if pending:
                writer.write_table(pa.Table.from_batches(pending, schema=out_schema))
    finally:
        reader.close(force=True)
    return total


# --------------------------------------------------------------------------------------
# The whole catalog (this worker's share of it)
# --------------------------------------------------------------------------------------


def _dist_version(name: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _provenance(
    catalog: HatsCatalog, tokenizer: AionTokenizer, settings: dict[str, Any]
) -> dict[str, Any]:
    from aion.codecs.config import HF_REPO_ID

    return {
        "source": catalog.url,
        "source_catalog": catalog.name,
        "modalities": [s.to_dict() for s in tokenizer.specs],
        "token_dtype": str(tokenizer.token_dtype),
        "codec_repo": HF_REPO_ID,
        "aion_hats_version": __version__,
        "aion_version": _dist_version("polymathic-aion"),
        "torch_version": torch.__version__,
        "python": platform.python_version(),
        "command": " ".join(sys.argv),
        "created": utc_now(),
        "settings": settings,
    }


def _select_partitions(
    catalog: HatsCatalog, partitions: list[Partition | str] | None, max_partitions: int | None
) -> list[Partition]:
    selected = catalog.partitions
    if partitions is not None:
        wanted = {p if isinstance(p, Partition) else Partition.parse(p) for p in partitions}
        missing = wanted - set(selected)
        if missing:
            raise ValueError(f"Partitions not in catalog: {sorted(p.label for p in missing)}")
        selected = [p for p in selected if p in wanted]
    if max_partitions is not None:
        selected = selected[:max_partitions]
    return selected


def tokenize_catalog(
    source: str | os.PathLike | HatsCatalog,
    output: str | os.PathLike,
    modalities: str | list[str | ModalitySpec] | None = "auto",
    *,
    batch_size: int = 64,
    row_group_size: int = 1024,
    max_rows: int | None = None,
    max_partitions: int | None = None,
    partitions: list[Partition | str] | None = None,
    device: str | torch.device | None = None,
    rank: int | None = None,
    world_size: int | None = None,
    overwrite: bool = False,
    fetch_mode: str = "auto",
    cache_dir: str | os.PathLike | None = None,
    num_prefetch: int = 1,
    fail_fast: bool = False,
    progress: bool | None = None,
    codec_manager: CodecManagerLike | None = None,
    token_dtype: str | np.dtype = "int64",
    finalize: bool | None = None,
) -> RunSummary:
    """Tokenize a HATS catalog with the AION codecs and write a HATS catalog of tokens.

    Args:
        source: local path, ``hf://`` URL or Hugging Face dataset id of the input catalog,
            or an already opened :class:`HatsCatalog`.
        output: local directory for the tokenized catalog (created if needed).
        modalities: ``"auto"`` to tokenize every column AION has a codec for, or a list of
            :class:`ModalitySpec` / strings such as ``"image"``, ``"LegacySurveyImage"`` or
            ``"flux_g=LegacySurveyFluxG"``.
        batch_size: rows per codec call (and per read from the parquet file).
        row_group_size: rows per row group in the output parquet files.
        max_rows: stop this worker after that many rows (demo runs; the last partition
            is then partial).
        max_partitions: only consider the first N partitions of the catalog.
        partitions: explicit list of partitions (``"Norder=4/Npix=1005"``) to process.
        device: torch device; default is the GPU matching the local rank, else CPU.
        rank, world_size: worker identity; default from SLURM/torchrun/MPI environment
            variables, else a single worker. Partitions are assigned round-robin.
        overwrite: redo partitions whose output already exists (default: skip them).
        fetch_mode: ``"download"`` whole partition files in the background (default for
            remote catalogs), ``"stream"`` row groups straight from the remote file
            (default when ``max_rows`` is set), ``"auto"`` to choose.
        cache_dir: where downloaded partitions are staged (default: ``output/_cache``).
        num_prefetch: partitions downloaded ahead of the one being tokenized.
        fail_fast: raise on the first failing partition instead of continuing.
        progress: show a progress bar (default: only on rank 0).
        codec_manager: optional pre-built ``aion.codecs.CodecManager`` (or a test double).
        token_dtype: ``"int64"`` (default) or ``"int32"`` token columns.
        finalize: write ``partition_info.csv`` and parquet metadata at the end. Default:
            only when there is a single worker; with several, run
            :func:`finalize_catalog` once after all of them have finished.
    """
    started = time.time()
    ctx = WorkerContext.from_env(rank, world_size, device)
    catalog = source if isinstance(source, HatsCatalog) else open_catalog(source)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    specs = resolve_modalities(
        catalog.schema, modalities, catalog_name=catalog.name, sample=lambda: catalog.sample(2)
    )
    tokenizer = AionTokenizer(
        specs, device=ctx.device, codec_manager=codec_manager, token_dtype=token_dtype
    )
    log.info("%s: tokenizing %s with %s", ctx, catalog.url, "; ".join(map(str, specs)))
    tokenizer.load_codecs()

    selected = _select_partitions(catalog, partitions, max_partitions)
    assigned = ctx.shard(selected)
    summary = RunSummary(output, ctx.rank, ctx.world_size)

    if ctx.is_main or not (output / "aion_hats.json").exists():
        settings = {
            "batch_size": batch_size,
            "row_group_size": row_group_size,
            "max_rows": max_rows,
            "max_partitions": max_partitions,
            "partitions": [p.label for p in selected] if partitions else None,
        }
        write_atomic(
            output / "aion_hats.json",
            json.dumps(_provenance(catalog, tokenizer, settings), indent=2),
        )
        if catalog.is_hats:
            extra = {"aion_hats_modalities": ",".join(f"{s.column}={s.name}" for s in specs)}
            props = output_properties(catalog, output.name, extra)
            write_atomic(output / "hats.properties", format_properties(props))

    todo = [p for p in assigned if overwrite or not partition_file(output, p).exists()]
    todo_set = set(todo)
    summary.skipped = [p.label for p in assigned if p not in todo_set]
    if summary.skipped:
        log.info("%s: skipping %d partitions already tokenized", ctx, len(summary.skipped))

    mode = _choose_fetch_mode(catalog, fetch_mode, max_rows)
    cache = Path(cache_dir) if cache_dir is not None else output / "_cache"
    fetcher = PartitionFetcher(catalog, mode, cache, todo, num_prefetch)
    show_progress = ctx.is_main if progress is None else progress
    bar = tqdm(
        total=max_rows,
        unit="rows",
        disable=not show_progress,
        desc=f"rank {ctx.rank}",
        dynamic_ncols=True,
    )
    log.info(
        "%s: %d/%d partitions assigned, fetch mode %s", ctx, len(assigned), len(selected), mode
    )

    try:
        for index, partition in enumerate(todo):
            if max_rows is not None and summary.rows >= max_rows:
                break
            bar.set_postfix_str(partition.label)
            local_path = None
            t0 = time.time()
            try:
                local_path = fetcher.get(index)
                remaining = None if max_rows is None else max_rows - summary.rows
                n = tokenize_partition(
                    catalog,
                    partition,
                    tokenizer,
                    partition_file(output, partition),
                    local_path=local_path,
                    batch_size=batch_size,
                    row_group_size=row_group_size,
                    max_rows=remaining,
                    progress=bar,
                )
            except Exception as err:
                summary.failed[partition.label] = f"{type(err).__name__}: {err}"
                log.exception("%s: partition %s failed", ctx, partition.label)
                if fail_fast:
                    raise
                continue
            finally:
                fetcher.release(local_path)
            summary.rows += n
            summary.done.append(partition.label)
            log.info("%s: %s -> %d rows in %.1fs", ctx, partition.label, n, time.time() - t0)
    finally:
        bar.close()
        fetcher.close()

    summary.seconds = time.time() - started
    if finalize is None:
        finalize = ctx.world_size == 1
    if finalize and summary.ok and (summary.done or summary.skipped):
        finalize_catalog(output)
        summary.finalized = True
    log.info("%s", summary)
    return summary
