# Hi-Scooby

Hi-Scooby predicts mouse chromatin contact maps from a wide single-cell RNA
count table and the mm10 reference sequence. Sequence embeddings are computed
on demand with the pinned AlphaGenome `all_folds` model; inference does not
require a precomputed embedding cache.

Two outputs are available:

- `smooth`: 5 kb, 1 Mb signed-log observed/expected maps for the 20 cell types
  learned by the phase-1 v2 smoothed head.
- `sparse`: pooled 10 kb expected contact counts, predictive intervals, and a
  deterministic NB2 simulated map from the raw-contact ranker.
- `both` (default): both outputs from one shared AlphaGenome pass.

The sparse v1 checkpoint is a diagnostic pooled model. It is RNA-independent
and must not be interpreted as a cell-type-specific predictor.

## Installation

Linux, Python 3.11, Conda, and a CUDA-capable JAX installation are required.
The live 1 Mb AlphaGenome forward was qualified on an NVIDIA H100.

```bash
conda env create -f environment.yml
conda env create -f environment-rna.yml
conda activate hi-scooby
python -m pip install -e .
```

The second environment preserves the historical `scvi-tools==1.1.3` RNA
encoder stack. If it was created at a path rather than as the named Conda
environment, point Hi-Scooby to its interpreter:

```bash
export HI_SCOOBY_RNA_PYTHON=/path/to/hi-scooby-rna/bin/python
```

## Resources

The downstream checkpoints and accepted RNA encoder live under `models/`.
Large reference and geometry files are not duplicated in the Python package.
Their paths and expected byte sizes are declared in
`configs/resources.yaml`.

For a source checkout, place the declared files at their repository-relative
paths. For an installed wheel or a shared data installation, set
`HI_SCOOBY_RESOURCE_ROOT` to a directory containing the same `data/...`
layout. The required inference resources are:

```text
data/
├── external/
│   ├── mm10_embeddings/window_manifest.mm10.parquet
│   └── mm10_genome/
│       ├── mm10.fa
│       └── mm10.fa.fai
└── processed/
    ├── multiome/tiles.parquet
    ├── raw_contact_ranker_10kb_inputs/tiles.10kb.parquet
    └── raw_contact_ranker_10kb_v1/
        ├── canonical_pairs.parquet
        └── distance_offset_curve.parquet
```

AlphaGenome weights are resolved at the pinned Hugging Face revision on first
use and may be supplied through the normal Hugging Face cache.

## Inference

The only required argument is a gzipped wide RNA TSV:

```bash
hi-scooby predict /path/to/rna_counts.tsv.gz
```

Select one output branch when needed:

```bash
hi-scooby predict /path/to/rna_counts.tsv.gz --mode smooth
hi-scooby predict /path/to/rna_counts.tsv.gz --mode sparse
```

The RNA table must begin with `barcode` and `cell_type`, followed by
unique gene columns containing nonnegative integer counts. Barcodes use the
form `rnaNN_<cell-barcode>` so the historical library batch is recoverable.
All 7,652 genes expected by the frozen RNA encoder must be present; additional
genes are allowed.

The output directory contains:

- `contact_maps.zarr`: chunked smooth maps and/or sparse visualization maps;
- `sparse_pairs.parquet`: canonical sparse predictions, expected counts,
  predictive intervals, and simulated counts;
- `tiles.parquet` and `cell_types.parquet`: coordinate and context tables;
- `run_manifest.json`: input and run metadata; and
- `README.txt`: an output-schema summary.

Sparse expected counts default to a filtered cis-pair depth of one million.
Use `--contact-depth` to change that scale and `--seed` to change the
simulated NB2 realization. Existing output directories are never overwritten.

## Training and validation entry points

The repository retains the preprocessing, training, and validation entry
points used for the released models. The RNA encoder rebuild and retained
loop validations were rerun for this inference-first release. Full phase-1
target regeneration and fresh end-to-end downstream training were not rerun.

Rebuild and validate the historical 14D RNA encoder:

```bash
conda run -n hi-scooby-rna python scripts/preprocess/train_rna_scvi.py
```

Generate the shared frozen AlphaGenome cache used for training:

```bash
python scripts/preprocess/cache_alphagenome_embeddings.py
```

Rebuild phase-1 targets and train the smoothed head:

```bash
python scripts/preprocess/build_phase1_targets.py
python scripts/model/train_phase1_model_v2.py \
  --config configs/phase1_v2.yaml \
  --output-dir results/phase1_model_v2
```

Train and evaluate the pooled sparse ranker:

```bash
python scripts/model/train_raw_contact_rate.py \
  --config configs/raw_contact_ranker_10kb.yaml \
  --feature-set alphagenome \
  --fold 0 \
  --output-dir results/raw_contact_ranker_10kb_v1/reproduction

python scripts/raw_contact_ranker/evaluate_shared_rate.py \
  --help
```

Peakachu and scDeepLUCIA validation entry points are retained in
`diagnostics/evaluate_b_only_peakachu_loops_10kb.py`,
`diagnostics/evaluate_family_peakachu_consensus_loops_10kb.py`, and
`diagnostics/evaluate_family_scdeeplucia_loops_10kb.py`. Use each script's
`--help` output to provide the external call sets and evaluation artifacts.

## Validation status

The accepted RNA encoder reproduced the retained cell-type centroids exactly.
A live one-tile RNA → AlphaGenome → both-head run published and reopened a
valid output. Against the historical float16 embedding control, smooth-map
RMSE was `0.001472`, sparse expected-count RMSE was `0.0000662`, coordinate
geometry and masks were identical, and all predictive intervals and simulated
counts were unchanged.
