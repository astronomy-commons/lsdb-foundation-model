"""Rank/device discovery and a tiny local multi-process launcher.

Tokenization is embarrassingly parallel over HATS partitions, so no collective
communication is needed: every worker just needs to know its rank, the world size and
which GPU to use. Those are read from the usual environment variables set by SLURM
(``srun``), ``torchrun`` or MPI launchers, or passed explicitly.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

log = logging.getLogger(__name__)

RANK_VARS = ("RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK", "PMI_RANK")
WORLD_VARS = ("WORLD_SIZE", "SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE", "PMI_SIZE")
LOCAL_RANK_VARS = ("LOCAL_RANK", "SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID")


def env_int(names: tuple[str, ...], default: int) -> int:
    """First of the environment variables ``names`` that is set, as an int."""
    for name in names:
        value = os.environ.get(name)
        if value not in (None, ""):
            return int(value)
    return default


@dataclass(frozen=True)
class WorkerContext:
    """Identity of this worker in a (possibly distributed) run."""

    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    device: torch.device = field(default_factory=lambda: _torch().device("cpu"))

    @classmethod
    def from_env(
        cls,
        rank: int | None = None,
        world_size: int | None = None,
        device: str | torch.device | None = None,
    ) -> WorkerContext:
        from .tokenizer import default_device

        rank = env_int(RANK_VARS, 0) if rank is None else rank
        world_size = env_int(WORLD_VARS, 1) if world_size is None else world_size
        local_rank = env_int(LOCAL_RANK_VARS, rank)
        if not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} is not in [0, {world_size})")
        dev = _torch().device(device) if device is not None else default_device(local_rank)
        return cls(rank, world_size, local_rank, dev)

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def shard(self, items: list) -> list:
        """The items this worker is responsible for (round-robin over ranks)."""
        return items[self.rank :: self.world_size]

    def __str__(self) -> str:
        return f"rank {self.rank}/{self.world_size} on {self.device}"


def _torch():
    import torch

    return torch


def spawn_local_workers(num_procs: int, argv: list[str]) -> int:
    """Run ``num_procs`` copies of the CLI on this node, one per GPU (or CPU worker).

    Composes with an outer launcher: with ``srun -n 4`` and ``num_procs=4`` there are
    16 workers in total. Returns the highest exit code of the children.
    """
    base = WorkerContext.from_env()
    n_gpus = _torch().cuda.device_count()
    if n_gpus and num_procs > n_gpus:
        log.warning("Spawning %d workers for %d GPUs; some will share a device", num_procs, n_gpus)
    cpu_threads = max(1, (os.cpu_count() or 1) // num_procs)
    procs = []
    for i in range(num_procs):
        env = dict(os.environ)
        env["RANK"] = str(base.rank * num_procs + i)
        env["WORLD_SIZE"] = str(base.world_size * num_procs)
        env["LOCAL_RANK"] = str(base.local_rank * num_procs + i if n_gpus == 0 else i)
        env.setdefault("OMP_NUM_THREADS", str(cpu_threads))
        cmd = [sys.executable, "-m", "aion_hats.cli", *argv]
        log.info("Spawning worker %s: RANK=%s WORLD_SIZE=%s", i, env["RANK"], env["WORLD_SIZE"])
        procs.append(subprocess.Popen(cmd, env=env))
    return max(p.wait() for p in procs)
