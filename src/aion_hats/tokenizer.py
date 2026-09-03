"""The :class:`AionTokenizer`: Arrow record batches in, Arrow record batches out."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pyarrow as pa
import torch
from aion.modalities import Modality

from .arrow_utils import tokens_to_arrow
from .modalities import ModalityBatch, ModalitySpec, build_modality_batches

log = logging.getLogger(__name__)


class CodecManagerLike(Protocol):
    """The part of ``aion.codecs.CodecManager`` the tokenizer relies on."""

    device: Any

    def encode(self, *modalities: Modality) -> dict[str, torch.Tensor]: ...


def default_device(local_rank: int = 0) -> torch.device:
    """``cuda:<local_rank>`` when CUDA is available, else CPU."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank % torch.cuda.device_count()}")
    return torch.device("cpu")


@dataclass
class PreparedBatch:
    """A record batch whose modality columns are already tensors on the codec device.

    Building it is CPU work (Arrow decoding, grouping, host-to-device copies) that the
    pipeline runs in a background thread while the codecs encode the previous batch.
    """

    batch: pa.RecordBatch
    groups: dict[str, list[ModalityBatch]]


class AionTokenizer:
    """Tokenize the columns described by ``specs`` with the AION codecs.

    The tokenizer is deliberately I/O agnostic: it maps a ``pyarrow.RecordBatch`` to
    another one with the raw modality columns replaced by token columns. This is what the
    partition pipeline calls, and it can equally be used inside an LSDB ``map_partitions``
    call or on an in-memory table.

    Args:
        specs: which columns to tokenize with which modality.
        device: torch device for the codecs (default: first GPU if available).
        codec_manager: an ``aion.codecs.CodecManager`` (created on demand). Anything
            with a compatible ``encode`` can be injected, e.g. a fake for tests.
        token_dtype: NumPy dtype of the token columns (``int64`` by default; ``int32``
            halves the size and is enough for every AION codebook).
    """

    def __init__(
        self,
        specs: list[ModalitySpec],
        device: torch.device | str | None = None,
        codec_manager: CodecManagerLike | None = None,
        token_dtype: np.dtype | str = np.int64,
    ):
        if not specs:
            raise ValueError("AionTokenizer needs at least one ModalitySpec")
        self.specs = list(specs)
        self.device = torch.device(device) if device is not None else default_device()
        self.token_dtype = np.dtype(token_dtype)
        if codec_manager is None:
            from aion.codecs import CodecManager

            codec_manager = CodecManager(device=self.device)
        self.codec_manager = codec_manager

    # -- schema ----------------------------------------------------------------------

    def output_schema(self, schema: pa.Schema) -> pa.Schema:
        """Schema of the tokenized batches for an input ``schema``."""
        dropped = {s.column for s in self.specs if s.drops_source}
        fields = [f for f in schema if f.name not in dropped]
        for spec in self.specs:
            fields.append(pa.field(spec.output_column, spec.arrow_type(self.token_dtype)))
        return pa.schema(fields, metadata=schema.metadata)

    # -- codecs ----------------------------------------------------------------------

    def load_codecs(self) -> None:
        """Download/load every codec now, so failures surface before any data is read.

        ``CodecManager`` only exposes lazy loading through ``_load_codec``; managers
        without it (test doubles) simply load on first use.
        """
        loader = getattr(self.codec_manager, "_load_codec", None)
        if loader is None:
            return
        for spec in self.specs:
            start = time.time()
            loader(spec.modality).to(self.device)
            log.info("Loaded %s codec on %s in %.1fs", spec.name, self.device, time.time() - start)

    # -- tokenization ----------------------------------------------------------------

    def prepare_batch(self, batch: pa.RecordBatch) -> PreparedBatch:
        """Decode the modality columns of ``batch`` into device tensors (no codec call)."""
        groups = {
            spec.output_column: build_modality_batches(spec, batch.column(spec.column), self.device)
            for spec in self.specs
        }
        return PreparedBatch(batch, groups)

    @torch.inference_mode()
    def encode_prepared(self, prepared: PreparedBatch) -> pa.RecordBatch:
        """Run the codecs on a prepared batch and assemble the output record batch."""
        batch = prepared.batch
        out_schema = self.output_schema(batch.schema)
        columns = {name: batch.column(name) for name in batch.schema.names}
        for spec in self.specs:
            columns[spec.output_column] = self._encode_column(
                spec, batch.num_rows, prepared.groups[spec.output_column]
            )
        return pa.RecordBatch.from_arrays([columns[f.name] for f in out_schema], schema=out_schema)

    def _encode_column(
        self, spec: ModalitySpec, n_rows: int, groups: list[ModalityBatch]
    ) -> pa.Array:
        """Tokenize one column; rows that could not be prepared become nulls."""
        tokens = np.zeros((n_rows, spec.num_tokens), dtype=self.token_dtype)
        valid = np.zeros(n_rows, dtype=bool)
        for rows, modality in groups:
            out = self.codec_manager.encode(modality)[spec.modality.token_key]
            out = out.reshape(len(rows), -1)
            if out.shape[1] != spec.num_tokens:
                raise RuntimeError(
                    f"{spec.name} produced {out.shape[1]} tokens per row, "
                    f"expected {spec.num_tokens}"
                )
            tokens[rows] = out.to("cpu").numpy()
            valid[rows] = True
        return tokens_to_arrow(tokens, valid, spec.arrow_type(self.token_dtype))

    def encode_batch(self, batch: pa.RecordBatch) -> pa.RecordBatch:
        """Tokenize one batch: source columns dropped/kept per spec, token columns appended."""
        return self.encode_prepared(self.prepare_batch(batch))

    def tokenize_table(self, table: pa.Table, batch_size: int = 64) -> pa.Table:
        """Tokenize an in-memory table (handy for LSDB ``map_partitions`` or notebooks)."""
        batches = [self.encode_batch(b) for b in table.to_batches(max_chunksize=batch_size)]
        return pa.Table.from_batches(batches, schema=self.output_schema(table.schema))
