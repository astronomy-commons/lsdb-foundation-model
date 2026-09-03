# lsdb-foundation-model

Tokenizing large astronomical catalogs with the [AION-1](https://github.com/PolymathicAI/AION)
codecs, starting from the [Multimodal Universe](https://github.com/MultimodalUniverse/MultimodalUniverse)
HATS catalogs hosted on Hugging Face (`UniverseTBD/mmu_*`).

This repository ships `aion-hats`, a small library and command line tool that turns any MMU
HATS catalog into a HATS catalog of AION tokens:

- **Input**: a HATS catalog on Hugging Face (`UniverseTBD/mmu_ssl_legacysurvey_north`), on disk,
  or behind any fsspec URL.
- **Output**: the same partitions (`Norder=k/Dir=d/Npix=p.parquet`), with the raw modality columns
  (images, spectra) replaced by token columns named after AION's token keys (`tok_image`,
  `tok_spectrum_desi`, `tok_flux_g`, ...). Multi-token columns are stored as
  `struct<token: list<int64>>`, which [`lsdb`](https://lsdb.io) loads as a nested column
  (`nested<token: [int64]>`, i.e. `df["tok_image.token"]`); single-token scalars are plain
  integers, and missing inputs become nulls. Every other column is kept, so the result opens with
  `lsdb` and joins back to its source on `_healpix_29` / `object_id`. It also
  loads with `datasets.load_dataset("parquet", data_files=".../dataset/**/*.parquet")` and can be uploaded to the
  Hub as is (a `README.md` with the dataset config is generated).
- **Scale-out**: partitions are the unit of work. Workers take a round-robin share, need no
  communication, write each partition atomically and skip partitions that are already done, so a
  job can be resumed by re-running it. One process per GPU (`srun`, `--num-procs`) is all it takes.

## Quick start

```bash
uv sync                      # environment (add --extra dev for tests, notebooks, lsdb)
uv run aion-hats inspect UniverseTBD/mmu_ssl_legacysurvey_north          # schema, detected modalities
uv run aion-hats tokenize UniverseTBD/mmu_ssl_legacysurvey_north data/tokenized_demo \
    --modality image --max-rows 100                                      # 100-object demo
```

or from Python:

```python
from aion_hats import tokenize_catalog

summary = tokenize_catalog(
    "UniverseTBD/mmu_ssl_legacysurvey_north", "data/tokenized_demo",
    modalities=["image"], max_rows=100,
)
```

`--modality` (or `modalities=`) takes a column name (`image`), an AION modality (`LegacySurveyImage`)
or an explicit pair (`z_spec=Z`); without it, every column AION has a codec for is tokenized
(`aion-hats inspect` lists them). Images and spectra are dropped from the output by default,
scalars are kept next to their token.

The same walkthrough as a Colab notebook:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/astronomy-commons/lsdb-foundation-model/blob/main/notebooks/tokenize_legacysurvey.ipynb)
(`notebooks/tokenize_legacysurvey.ipynb`, generated with `uv run python scripts/build_notebook.py`).

## Running the full catalog on several GPUs

The worker's rank and world size come from `SLURM_PROCID`/`SLURM_NTASKS`, `RANK`/`WORLD_SIZE`
(torchrun) or MPI variables, or from `--rank/--world-size`. Each worker uses the GPU matching its
local rank. On Perlmutter:

```bash
#!/bin/bash
#SBATCH -C gpu -N 4 --ntasks-per-node=4 --gpus-per-task=1 -c 32 -t 12:00:00
export HF_HOME=$SCRATCH/hf_cache                      # codec weights
srun aion-hats tokenize UniverseTBD/mmu_ssl_legacysurvey_north $SCRATCH/ls_north_tokens \
    --modality image --batch-size 256 --cache-dir $SCRATCH/ls_north_stage --num-prefetch 2
```

then, once every task has finished (re-submit the same job to resume after a time limit):

```bash
aion-hats finalize $SCRATCH/ls_north_tokens         # partition_info.csv, _metadata, hats_nrows
```

Notes:

- Remote partitions (about 2 GB each for Legacy Survey images) are downloaded in a background
  thread while the previous one is tokenized, and deleted afterwards; `--cache-dir` needs
  `(--num-prefetch + 1) x partition size` of free space per worker. With `--max-rows` (demos) row
  groups are streamed straight from the Hub instead. If the catalog is already on a parallel file
  system, pass its path and nothing is copied.
- On a single machine with several GPUs, `--num-procs 4` spawns one worker per GPU; it composes
  with `srun` (`srun -n 2 --ntasks-per-node=1 ... --num-procs 4` is 8 workers).
- A failing partition is logged and reported in the summary (non-zero exit code) without stopping
  the others; `--fail-fast` reverses that. Nothing is ever pushed to the Hub.
- `--token-dtype int32` halves the size of the token columns (every AION codebook fits).
- Set `HF_TOKEN` (a read token) for faster, rate-limit-free downloads from the Hub.

## Library layout

| Module | Role |
| --- | --- |
| `aion_hats.catalog` | open HATS catalogs (local, `hf://`, fsspec), list partitions, stream or download them, write and finalize the output catalog |
| `aion_hats.modalities` | `ModalitySpec`, detection of tokenizable columns, adapters from Arrow struct columns to AION `Image`/`Spectrum`/`Scalar` modalities |
| `aion_hats.tokenizer` | `AionTokenizer`: Arrow record batch in, record batch with token columns out (usable on its own, e.g. in `lsdb` `map_partitions`) |
| `aion_hats.pipeline` | `tokenize_catalog`: sharding, prefetching, atomic partition writes, resume, summary |
| `aion_hats.distributed` | rank/GPU discovery from the environment, local multi-process launcher |
| `aion_hats.cli` | `aion-hats tokenize / finalize / inspect` |

Tests (`uv run pytest`) run on a synthetic HATS catalog with a fake codec, so they need neither
network access nor GPU.
