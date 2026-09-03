"""Generate notebooks/tokenize_legacysurvey.ipynb, the Colab demo of the aion-hats library.

uv run python scripts/build_notebook.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat

REPO = "astronomy-commons/lsdb-foundation-model"
DATASET_ID = "UniverseTBD/mmu_ssl_legacysurvey_north"
NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "tokenize_legacysurvey.ipynb"

md = nbformat.v4.new_markdown_cell
code = nbformat.v4.new_code_cell

cells = [
    md(
        f'<a href="https://colab.research.google.com/github/{REPO}/blob/main/notebooks/'
        'tokenize_legacysurvey.ipynb" target="_parent"><img src="https://colab.research.google.com/'
        'assets/colab-badge.svg" alt="Open In Colab"/></a>'
    ),
    md(
        "# Tokenizing Legacy Survey images with AION-1\n\n"
        "This notebook tokenizes galaxy cutouts from the\n"
        f"[`{DATASET_ID}`](https://huggingface.co/datasets/{DATASET_ID}) Hugging Face dataset\n"
        "(the northern Legacy Surveys DR9 sample of the Multimodal Universe: 14M objects with\n"
        "g, r, z cutouts, stored as a HATS catalog) with the\n"
        "[AION-1](https://github.com/PolymathicAI/AION) image codec, using the `aion-hats` library\n"
        f"from [`{REPO}`](https://github.com/{REPO}).\n\n"
        "The output is again a HATS catalog with the **same columns as the original catalog, minus\n"
        "the images, plus a `tok_image` column** of 576 discrete tokens per object, stored as a\n"
        "nested column (`struct<token: list<int64>>`) that `lsdb` understands natively. The very same\n"
        "function scales to the whole catalog on a multi-GPU cluster (see the last section).\n\n"
        "For the demo we only process 100 objects and do not push anything to the Hub.\n\n"
        "### Enabling GPU access\n\n"
        "The codec runs on CPU too, but on Colab go to `Runtime > Change runtime type` and select a\n"
        "GPU to speed things up.\n\n"
        "### Installing dependencies"
    ),
    code(f"!pip install --quiet git+https://github.com/{REPO}.git lsdb matplotlib"),
    md(
        "## Step I: open the catalog\n\n"
        "`open_catalog` reads the HATS metadata (`hats.properties`, `partition_info.csv`, the\n"
        "parquet schema) without downloading any data. The catalog is made of ~11,000 HEALPix\n"
        "partitions of up to 8192 objects each; those partitions are the unit of work of the\n"
        "tokenizer. `detect_modalities` lists the columns AION has a codec for: the `image` struct\n"
        "(bands + flux) maps to `LegacySurveyImage`, and scalar columns are matched to AION's\n"
        "scalar modalities by name (`flux_g`, `ebv`, ...)."
    ),
    code(
        "from aion_hats import open_catalog, detect_modalities\n\n"
        f'SOURCE = "{DATASET_ID}"\n\n'
        "catalog = open_catalog(SOURCE)\n"
        "print(catalog)\n"
        "print(f\"{catalog.properties['hats_nrows']} rows, orders {sorted({p.order for p in catalog.partitions})}\")\n"
        "print(catalog.schema)\n\n"
        "for spec in detect_modalities(catalog.schema, catalog_name=catalog.name, sample=lambda: catalog.sample(2)):\n"
        "    print(spec)"
    ),
    md(
        "## Step II: tokenize\n\n"
        "`tokenize_catalog` streams the rows of each partition in batches, wraps the cutouts in\n"
        "AION's `LegacySurveyImage` modality (the codec knows the survey and the bands, and handles\n"
        "the missing `i` band), encodes them into 576 tokens (a 24x24 grid over the central 96x96\n"
        "pixels) and writes the tokenized partition next to the untouched columns.\n\n"
        'We only ask for the `image` column (`modalities="auto"` would also tokenize `flux_g`,\n'
        "`ebv`, ... into `tok_flux_g`, `tok_ebv`, ...), and stop after 100 objects, in which case\n"
        "only the needed row groups of the first partition are streamed from the Hub."
    ),
    code(
        "from aion_hats import tokenize_catalog\n\n"
        'OUTPUT = "tokenized_demo"\n\n'
        'summary = tokenize_catalog(SOURCE, OUTPUT, modalities=["image"], max_rows=100, batch_size=32)\n'
        "print(summary)"
    ),
    md(
        "## Step III: look at the result\n\n"
        "The output directory is a HATS catalog: `hats.properties`, `partition_info.csv` and a\n"
        "`dataset/` tree of parquet partitions. `aion_hats.json` records the provenance (source,\n"
        "modalities, codec and library versions)."
    ),
    code(
        "import pyarrow.dataset as ds\n\n"
        "!find $OUTPUT -type f | sort\n\n"
        'tokens = ds.dataset(f"{OUTPUT}/dataset", format="parquet", exclude_invalid_files=True).to_table()\n'
        "print(tokens.schema)\n"
        "row = tokens.slice(0, 1).to_pylist()[0]\n"
        'print({k: v for k, v in row.items() if k != "tok_image"})\n'
        'print("tokens:", len(row["tok_image"]["token"]), row["tok_image"]["token"][:16], "...")'
    ),
    md(
        "The parquet files also load as a regular Hugging Face dataset (the generated `README.md`\n"
        "carries the matching `data_files` config for when the folder is uploaded to the Hub):"
    ),
    code(
        "from datasets import load_dataset\n\n"
        'hf_dataset = load_dataset("parquet", data_files=f"{OUTPUT}/dataset/**/*.parquet", split="train")\n'
        "print(hf_dataset)"
    ),
    md(
        "Because the layout is preserved, `lsdb` opens the tokenized catalog like any other HATS\n"
        "catalog (and could cross-match or join it with the source on `_healpix_29`). The token\n"
        "column is recognised as a *nested* column: each object carries a small sub-table with a\n"
        "`token` column, and `tokenized[\"tok_image.token\"]` flattens it."
    ),
    code(
        "import lsdb\n\n"
        "tokenized = lsdb.open_catalog(OUTPUT)\n"
        "print(tokenized.dtypes)\n"
        "df = tokenized.compute()\n"
        'print(df["tok_image"].iloc[0].head())\n'
        "df.head(3)"
    ),
    md(
        "## Step IV: decode the tokens\n\n"
        "The same codec decodes tokens back into a 96x96 cutout, a useful sanity check of what\n"
        "information the tokenization retains. We read the original images of the first four\n"
        "objects directly from the source partition for comparison."
    ),
    code(
        "import numpy as np\n"
        "import pyarrow.compute as pc\n"
        "import torch\n"
        "import matplotlib.pyplot as plt\n"
        "from aion.codecs import CodecManager\n"
        "from aion.modalities import LegacySurveyImage\n"
        "from aion_hats import default_device\n\n"
        "device = default_device()\n"
        "codec_manager = CodecManager(device=device)\n\n"
        'originals = catalog.read_partition(catalog.partitions[0], columns=["object_id", "image"], max_rows=4)\n'
        'token_batch = torch.as_tensor(np.stack(pc.struct_field(tokens.slice(0, 4).column("tok_image"), "token").to_pylist()), device=device)\n'
        'decoded = codec_manager.decode({LegacySurveyImage.token_key: token_batch}, LegacySurveyImage, bands=["DES-G", "DES-R", "DES-Z"])\n\n'
        "fig, axes = plt.subplots(2, 4, figsize=(12, 6))\n"
        "for k in range(4):\n"
        '    flux = np.asarray(originals.column("image")[k]["flux"].as_py())\n'
        '    axes[0, k].imshow(flux[1, 28:-28, 28:-28], cmap="gray")\n'
        "    axes[0, k].set_title(f\"{originals.column('object_id')[k]} (r, input)\")\n"
        '    axes[1, k].imshow(decoded.flux[k, 1].cpu().numpy(), cmap="gray")\n'
        '    axes[1, k].set_title("decoded from tokens")\n'
        "for ax in axes.ravel():\n"
        '    ax.axis("off")'
    ),
    md(
        "## Scaling up\n\n"
        "The full catalog is 14M objects in 3.4 TiB of parquet. The same `tokenize_catalog` call\n"
        "(or the `aion-hats tokenize` command) does the whole job: without `max_rows`, each worker\n"
        "downloads its partitions in the background, tokenizes them, writes each one atomically and\n"
        "skips partitions that already exist, so a job resumes by re-running it. Workers take their\n"
        "rank and GPU from the environment (`SLURM_PROCID`, `RANK`, ...), so on a SLURM cluster such\n"
        "as Perlmutter one process per GPU is:\n\n"
        "```bash\n"
        "#SBATCH -C gpu -N 4 --ntasks-per-node=4 --gpus-per-task=1\n"
        f"srun aion-hats tokenize {DATASET_ID} $SCRATCH/ls_north_tokens --modality image \\\n"
        "    --batch-size 256 --cache-dir $SCRATCH/stage --num-prefetch 2\n"
        "aion-hats finalize $SCRATCH/ls_north_tokens   # once, after all workers are done\n"
        "```\n\n"
        "On a single multi-GPU machine, `--num-procs 4` spawns one worker per GPU. The finalized\n"
        "folder is ready for `huggingface_hub.HfApi().upload_folder(...)` (not done here)."
    ),
]

nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["colab"] = {"provenance": [], "gpuType": "T4"}
nb.metadata["accelerator"] = "GPU"
nbformat.write(nb, NOTEBOOK)
print(f"Wrote {NOTEBOOK}")
