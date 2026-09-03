"""Command line interface: ``aion-hats tokenize | finalize | inspect``."""

from __future__ import annotations

import argparse
import logging
import sys

from . import __version__
from .distributed import RANK_VARS, env_int


def _add_tokenize_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "source", help="HATS catalog: local path, hf://datasets/<org>/<name> URL or <org>/<name>"
    )
    p.add_argument("output", help="Local directory for the tokenized HATS catalog")
    p.add_argument(
        "-m",
        "--modality",
        action="append",
        dest="modalities",
        metavar="SPEC",
        help="Column to tokenize: 'image', 'LegacySurveyImage' or 'flux_g=LegacySurveyFluxG'. "
        "Repeatable; default: everything AION has a codec for",
    )
    p.add_argument("--batch-size", type=int, default=64, help="Rows per codec call (default 64)")
    p.add_argument("--row-group-size", type=int, default=1024, help="Rows per output row group")
    p.add_argument("--max-rows", type=int, help="Stop this worker after N rows (demo runs)")
    p.add_argument("--max-partitions", type=int, help="Only consider the first N partitions")
    p.add_argument(
        "--partition",
        action="append",
        dest="partitions",
        metavar="Norder=K/Npix=P",
        help="Only these partitions (repeatable)",
    )
    p.add_argument(
        "--device",
        help="torch device, e.g. cuda:0 or cpu (default: GPU of the local rank, else CPU)",
    )
    p.add_argument("--rank", type=int, help="Worker rank (default: from SLURM/torchrun/MPI env)")
    p.add_argument("--world-size", type=int, help="Number of workers (default: from env, else 1)")
    p.add_argument(
        "--num-procs",
        type=int,
        default=1,
        help="Spawn N worker processes on this node (one per GPU)",
    )
    p.add_argument(
        "--overwrite", action="store_true", help="Redo partitions that already have an output"
    )
    p.add_argument(
        "--fetch-mode",
        choices=["auto", "download", "stream"],
        default="auto",
        help="How remote partitions are read (default: download, or stream with --max-rows)",
    )
    p.add_argument(
        "--cache-dir", help="Staging directory for downloaded partitions (default: OUTPUT/_cache)"
    )
    p.add_argument(
        "--num-prefetch", type=int, default=1, help="Partitions to download ahead of time"
    )
    p.add_argument("--fail-fast", action="store_true", help="Stop at the first failing partition")
    p.add_argument("--token-dtype", choices=["int64", "int32"], default="int64")
    finalize = p.add_mutually_exclusive_group()
    finalize.add_argument(
        "--finalize",
        dest="finalize",
        action="store_true",
        default=None,
        help="Write partition_info.csv and parquet metadata at the end",
    )
    finalize.add_argument(
        "--no-finalize",
        dest="finalize",
        action="store_false",
        help="Skip finalization (multi-worker runs; run 'aion-hats finalize' afterwards)",
    )
    p.add_argument("--no-progress", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aion-hats",
        description="Tokenize Multimodal Universe HATS catalogs with the AION-1 codecs",
    )
    parser.add_argument("--version", action="version", version=f"aion-hats {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    tok = sub.add_parser("tokenize", help="Tokenize a catalog (this worker's share of it)")
    _add_tokenize_args(tok)

    fin = sub.add_parser(
        "finalize", help="Write partition_info.csv and parquet metadata for a tokenized catalog"
    )
    fin.add_argument("output")

    ins = sub.add_parser(
        "inspect", help="Show properties, schema and detected modalities of a catalog"
    )
    ins.add_argument("source")
    return parser


def _configure_logging(verbose: bool) -> None:
    rank = env_int(RANK_VARS, 0)
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=f"%(asctime)s [rank {rank}] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    for noisy in ("httpx", "httpcore", "huggingface_hub", "fsspec", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def cmd_tokenize(args: argparse.Namespace, argv: list[str]) -> int:
    if args.num_procs > 1:
        from .distributed import spawn_local_workers

        # argparse keeps the last occurrence, so children see --num-procs 1 and do not re-spawn
        return spawn_local_workers(args.num_procs, [*argv, "--num-procs", "1"])

    from .pipeline import tokenize_catalog

    summary = tokenize_catalog(
        args.source,
        args.output,
        args.modalities or "auto",
        batch_size=args.batch_size,
        row_group_size=args.row_group_size,
        max_rows=args.max_rows,
        max_partitions=args.max_partitions,
        partitions=args.partitions,
        device=args.device,
        rank=args.rank,
        world_size=args.world_size,
        overwrite=args.overwrite,
        fetch_mode=args.fetch_mode,
        cache_dir=args.cache_dir,
        num_prefetch=args.num_prefetch,
        fail_fast=args.fail_fast,
        progress=False if args.no_progress else None,
        token_dtype=args.token_dtype,
        finalize=args.finalize,
    )
    print(summary)
    return 0 if summary.ok else 1


def cmd_finalize(args: argparse.Namespace) -> int:
    from .catalog import finalize_catalog

    print(finalize_catalog(args.output))
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    from .catalog import open_catalog
    from .modalities import detect_modalities

    catalog = open_catalog(args.source)
    print(f"{catalog.url} ({catalog.name})")
    for key, value in catalog.properties.items():
        print(f"  {key} = {value}")
    print(
        f"{len(catalog.partitions)} partitions, orders {sorted({p.order for p in catalog.partitions})}"
    )
    print("schema:")
    for field in catalog.schema:
        print(f"  {field.name}: {str(field.type)[:100]}")
    specs = detect_modalities(
        catalog.schema, catalog_name=catalog.name, sample=lambda: catalog.sample(2)
    )
    print("detected modalities:")
    for spec in specs:
        print(f"  {spec}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)
    if args.command == "tokenize":
        return cmd_tokenize(args, argv)
    if args.command == "finalize":
        return cmd_finalize(args)
    return cmd_inspect(args)


if __name__ == "__main__":
    sys.exit(main())
