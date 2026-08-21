---
name: flexynesis
description: Run flexynesis, a deep-learning suite for multi-omics data integration and clinical outcome prediction (drug response, cancer subtyping, survival analysis). Use whenever the user wants to train or run inference with flexynesis, mentions multi-omics data (gene expression, CNV, methylation, mutation, proteomics, etc.) alongside a clinical/outcome prediction task, asks about integrating omics layers with deep learning, or needs to format data / build a CLI command / choose a model class / interpret outputs for flexynesis. Applies even when the user doesn't say "flexynesis" by name but describes a matching task, e.g. "predict drug response from expression and CNV data" or "do survival analysis combining RNA-seq and mutations."
---

# flexynesis

Deep learning suite for multi-omics data integration and clinical outcome prediction
(drug response regression, cancer subtyping/classification, survival analysis).
Published in Nature Communications (DOI: 10.1038/s41467-025-63688-5). This skill is
for using flexynesis as a tool *from another project* — to run training/inference
on the user's own omics dataset, not for developing flexynesis itself.

Full docs: https://bimsbstatic.mdc-berlin.de/akalin/buyar/flexynesis/site/getting_started/

## Environment

Flexynesis lives in its own environment. Before running any flexynesis command:

```bash
mamba activate flexynesis
```

If the environment doesn't exist yet:

```bash
pip install flexynesis   # or: pip install -e . from a cloned flexynesis repo
```

Requires Python >= 3.11. `torch_geometric` is required only for `--model_class GNN`.

## Step 1: Get the data into the required layout

Flexynesis expects a directory with `train/` and `test/` subfolders. Both must
contain the same set of files:

```
data_path/
├── train/
│   ├── clin.csv        # rows = samples, columns = clinical/target variables
│   ├── gex.csv          # rows = features (genes etc.), columns = samples
│   └── cnv.csv          # same shape convention as above
└── test/
    ├── clin.csv
    ├── gex.csv
    └── cnv.csv
```

Rules that trip people up:
- Omics CSVs are **features as rows, samples as columns** — the transpose of the
  usual sample x feature convention. `clin.csv` is the opposite: samples as rows.
- Sample IDs must match between `clin.csv` and every omics file's column headers,
  and between train/test (they can differ in identity, but the column/row schema
  must align).
- Each modality file name (minus `.csv`) becomes the value passed to `--data_types`.
- Missing/low-variance features are handled internally (median imputation,
  variance filtering) — don't pre-clean unless you have a specific reason to.
- If the user's data isn't already in this shape (e.g. it's a single matrix, or
  samples are rows), write a small script to reshape/split it into this layout
  rather than trying to make flexynesis accept a different format.

## Step 2: Pick a model class

| `--model_class` | Use when | Notes |
|---|---|---|
| `DirectPred` | Default choice; regression/classification/survival on one or more targets | Fully connected encoders + MLP heads |
| `supervised_vae` | Same as above but also want a generative/reconstruction latent space | VAE with MMD loss; supports unsupervised mode if `--target_variables`/survival args are all omitted |
| `MultiTripletNetwork` | Want embeddings that cluster by class (e.g. subtype discovery, contrastive-style tasks) | First target variable **must be categorical** |
| `CrossModalPred` | Want to predict/reconstruct one omics layer from another (e.g. impute methylation from expression) | Set `--input_layers` and `--output_layers`; incompatible with `--fusion_type early` |
| `GNN` | Want to exploit gene-gene interaction structure (STRING PPI or a custom graph) | Requires `torch_geometric`; set `--gnn_conv_type {GC,GCN,SAGE}` |
| `DeepTSP` | Want a minimal, gene-set-structured pairwise gene panel (e.g. for a ddPCR-style assay) rather than a black-box predictor | Single omics modality only; requires `--gene_sets_path <standard-GMT-file>`; input must already be log2-scale (flexynesis's own normalization is skipped for this model); single target variable; see notes below |
| `RandomForest`, `SVM`, `XGBoost`, `RandomSurvivalForest` | Want a classical ML baseline instead of/alongside a DL model | Same CLI, no HPO search space needed; good sanity check via `--evaluate_baseline_performance` |

If unsure, default to `DirectPred` — it's the best-supported baseline and the
fastest way to confirm the data pipeline works end-to-end before trying anything
fancier.

### `DeepTSP` notes

`DeepTSP` scores within-gene-set candidate gene *pairs* (log2 fold change
features) rather than raw genes, with a hard global cap on the number of
surviving pairs — built for designing a minimal assay panel (e.g. ddPCR),
not general-purpose prediction. Specifics:
- `--gene_sets_path <file>` is required: a standard-GMT file (tab-separated:
  name, description, then one gene per remaining field). It is the caller's
  responsibility to supply a correctly formatted file.
- `--data_types` must resolve to exactly one modality.
- Input for that modality must already be library-size-normalized and
  log2-transformed — flexynesis's own log-transform/standard-scaling step is
  skipped entirely for this model class, so passing raw or z-scored data will
  silently corrupt the pairwise log2FC features.
- Supports exactly one target variable (or one survival event/time pair).
- `Trainer.min_epochs` is set automatically so `--early_stop_patience` can't
  fire before the pruning schedule (`target_k`/`prune_every` in the HPO
  config) has converged — expect training to always run at least that many
  epochs regardless of `--early_stop_patience`.
- Feature importance (`<prefix>.feature_importance.attentionweight.csv`) uses
  the model's own learned gate/attention weights, not Captum — the
  `--feature_importance_method` flag is ignored for this model class.

## Step 3: Build the command

Minimal required flags: `--data_path`, `--model_class`, `--data_types`, and
either `--target_variables` or the survival pair `--surv_event_var`/`--surv_time_var`.

```bash
flexynesis \
  --data_path /path/to/data_path \
  --model_class DirectPred \
  --data_types gex,cnv \
  --target_variables drug_response \
  --hpo_iter 50 \
  --outdir results/ \
  --prefix myjob
```

Flags worth knowing about, grouped by purpose:

**Task definition**
- `--target_variables col1,col2` — clin.csv columns to predict (comma-separated for multi-task)
- `--covariates col1,col2` — clin.csv columns to feed in as extra input features (e.g. age, sex, batch)
- `--surv_event_var` + `--surv_time_var` — required together for survival tasks; can be combined with `--target_variables` for a multi-task model
- `--join_key` — clin.csv column holding sample IDs, if not the index

**Fusion / architecture**
- `--fusion_type {intermediate,early}` — `intermediate` (default): one encoder per modality, concatenated then fused. `early`: concatenate raw modalities first, single encoder (internally keyed `"all"`)
- `--input_layers` / `--output_layers` — CrossModalPred only, which data types are encoded vs. reconstructed
- `--gnn_conv_type {GC,GCN,SAGE}` — GNN only
- `--string_organism` (default 9606=human), `--string_node_name {gene_name,gene_id}`, `--user_graph path.csv` (columns `GeneA,GeneB,Score`) — GNN graph source

**Feature selection (applies before training)**
- `--variance_threshold` (default 1), `--correlation_threshold` (default 0.8)
- `--features_min` (default 500), `--features_top_percentile` (default 20)
- `--restrict_to_features path.txt` — skip selection, use this exact feature list
- `--log_transform {True,False}`, `--subsample N` — downsample training set

**Training / HPO**
- `--hpo_iter` (default 100) — Bayesian HPO iterations; use a small number (1-10) for a quick smoke test
- `--use_cv` — 5-fold CV instead of a single 80/20 split (slower, more robust)
- `--config_path file.yaml` — custom HPO search space; see `examples/configs/hpo_configuration.yaml` in the flexynesis repo for the exact per-model-class YAML format (list of `{type: Integer|Real|Categorical, name, low/high or categories, [prior: log-uniform]}`). This is also the only way to raise `batch_size` past the default 32-128 range — useful for speeding up training on small (hundreds-of-samples) datasets. **Keep `batch_size` strictly below the post-`val_size`-split training set size**: the training `DataLoader` uses `drop_last=True`, so a `batch_size` at or above that size silently drops the only batch and trains on nothing (no error, no warning — looks like a normal run but the model never gets a gradient update).
- `--early_stop_patience` (default 10), `--hpo_patience` (default 10), `--val_size` (default 0.2)
- `--use_loss_weighting {True,False}` (default True) — learnable uncertainty-based multi-task loss balancing; leave on unless debugging a specific loss term
- `--finetuning_samples N` — fine-tune the trained model on N samples from the test set (transfer learning); not supported for `DeepTSP`

**Compute**
- `--device {auto,cuda,mps,cpu}` (default auto), `--threads`, `--num_workers`
- `--safetensors` — save weights as SafeTensors+JSON instead of `.pth`+`.joblib`

**Interpretability / comparison**
- `--disable_marker_finding` — skip Captum feature-importance step (saves time on quick runs)
- `--feature_importance_method {IntegratedGradients,GradientShap,Both}`
- `--evaluate_baseline_performance` — also fit RF/SVM/XGBoost/RSF for comparison

**Inference only (skip training)**
```bash
flexynesis \
  --pretrained_model job.final_model.pth \
  --artifacts job.artifacts.joblib \
  --data_path_test /path/to/new_test_data \
  --data_types gex,cnv \
  --outdir results/ --prefix myjob_inference
```
`data_path_test` needs the same modality files as training (feature harmonization
uses the stored artifacts), but does not need a `train/` subfolder.

## Step 4: Read the outputs

All outputs are written to `--outdir`, prefixed with `--prefix` (default `job`):

- `<prefix>.final_model.pth` / `.safetensors` — trained weights
- `<prefix>.final_model_config.json` — architecture/hyperparameters used
- `<prefix>.artifacts.joblib` / `.json` — fitted scalers, feature lists, label encoders (needed for inference on new data)
- `<prefix>.embeddings_train.csv`, `<prefix>.embeddings_test.csv` — latent sample embeddings
- `<prefix>.predicted_labels.csv` — predictions per sample per target
- `<prefix>.stats.csv` — performance metrics
- `<prefix>.feature_importance.<method>.csv`, `<prefix>.feature_logs.<modality>.csv` — per-feature importance (unless `--disable_marker_finding`)

When reporting results back to the user, read `stats.csv` for headline metrics and
`predicted_labels.csv` for per-sample detail rather than re-deriving them.

## Troubleshooting notes

- "Sample mismatch" or shape errors almost always trace back to Step 1 (row/column
  orientation, or clin.csv sample IDs not matching omics column headers).
- For a fast sanity check that the pipeline itself works, run with `--hpo_iter 1
  --features_top_percentile 5` before committing to a full HPO search.
- `MultiTripletNetwork` failing to train usually means the first target variable
  isn't categorical — check `clin.csv` dtypes.
