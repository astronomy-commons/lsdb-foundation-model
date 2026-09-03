"""Tokenize Multimodal Universe HATS catalogs with the AION-1 codecs.

Typical use::

    from aion_hats import tokenize_catalog
    tokenize_catalog("UniverseTBD/mmu_ssl_legacysurvey_north", "ls_north_tokens", ["image"])

The output is again a HATS catalog (same partitions, images replaced by ``tok_image``)
that can be opened with ``lsdb`` or loaded with ``datasets``. Run one process per GPU
(``srun``, ``--num-procs``) to scale out; partitions are sharded by rank and completed
partitions are skipped on restart.
"""

__version__ = "0.1.0"

from .catalog import FinalizeSummary, HatsCatalog, Partition, finalize_catalog, open_catalog
from .distributed import WorkerContext
from .modalities import MODALITY_REGISTRY, ModalitySpec, detect_modalities, resolve_modalities
from .pipeline import RunSummary, tokenize_catalog, tokenize_partition
from .tokenizer import AionTokenizer, default_device

__all__ = [
    "MODALITY_REGISTRY",
    "AionTokenizer",
    "FinalizeSummary",
    "HatsCatalog",
    "ModalitySpec",
    "Partition",
    "RunSummary",
    "WorkerContext",
    "__version__",
    "default_device",
    "detect_modalities",
    "finalize_catalog",
    "open_catalog",
    "resolve_modalities",
    "tokenize_catalog",
    "tokenize_partition",
]
