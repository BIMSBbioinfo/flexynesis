# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Flexynesis is a deep learning suite for **multi-omics data integration** and **clinical outcome prediction** (drug response, cancer subtyping, survival analysis). Published in Nature Communications (DOI: 10.1038/s41467-025-63688-5).

## Environment

Always activate the conda environment before running any Python or pip commands:

```bash
mamba activate flexynesis
pip install -e .   # install in editable mode after activating
```

## Commands

```bash
# Install (requires Python >= 3.11)
pip install -e .

# Lint (CI enforces these)
pip install isort flake8
isort --check-only --diff .
flake8 . --max-line-length=120

# Fix import ordering
isort .

# Run tests
pytest

# Run a single test
pytest tests/unit/test_smoke.py

# Run CLI
flexynesis --data_path ./data --model_class DirectPred --target_variables TARGET --data_types gex,cnv
```

## Architecture

### Core Design Pattern

Each omics modality is encoded separately by an `Encoder` into a latent vector. Latent vectors are concatenated and passed through a fusion Linear layer. Supervised prediction heads (`MLP`) sit on top for each target variable. All models are `pl.LightningModule` subclasses trained via PyTorch Lightning's `Trainer`.

**Fusion modes** (`--fusion_type`):
- `intermediate` (default): separate encoder per modality → concatenate → fusion layer → heads
- `early`: concatenate all modalities first → single encoder path (stored under key `"all"`)

**Multi-task loss balancing**: All models use learnable `log_vars` (`nn.ParameterDict`) — one parameter per loss term — for automatic uncertainty-based weighting. This avoids manual tuning when mixing regression, classification, reconstruction, and Cox PH survival losses.

### Model Classes (`flexynesis/models/`)

| Model | Key Feature |
|---|---|
| `DirectPred` | Fully connected encoders + MLP heads; the baseline DL model |
| `supervised_vae` | VAE with MMD loss; adds per-modality decoders; supports unsupervised mode (no targets) |
| `MultiTripletNetwork` | Triplet loss for discriminative embeddings; first target variable must be categorical |
| `CrossModalPred` | Cross-modality reconstruction; `--input_layers` encodes, `--output_layers` reconstructs; incompatible with `early` fusion |
| `GNN` | Graph convolution over STRING PPI network; `--gnn_conv_type {GC,GCN,SAGE,GAT}`; requires `torch_geometric` |
| `DeepTSP` | Gene-set-structured pairwise gene scoring with a hard global cap on surviving pairs (minimal ddPCR-style panel design); self-contained in `flexynesis/models/deep_tsp.py`, doesn't reuse `Encoder`/`MLP`; single modality, single target, requires `--gene_sets_path` |

### Key Modules (`flexynesis/modules.py`)

- `Encoder`: Linear → LeakyReLU → BatchNorm blocks → FC_mean + FC_var (VAE reparameterization)
- `Decoder`: Mirror of Encoder → sigmoid output
- `MLP`: single hidden layer (Linear → BatchNorm → ReLU → Dropout → Linear); used as prediction heads
- `flexGCN`: stacked graph conv + BatchNorm + Dropout + FC for graph-level output
- `cox_ph_loss`: custom Cox proportional hazards loss for survival tasks

### Data Pipeline (`flexynesis/data.py`)

**Input format**: directory with `train/` and `test/` subdirectories, each containing:
- `clin.csv`: rows=samples, columns=clinical/target variables
- `<modality>.csv`: rows=features (gene names etc.), columns=samples (e.g. `gex.csv`, `cnv.csv`)

**`DataImporter.import_data()` pipeline**:
1. Read CSVs, optional downsampling
2. `cleanup_data()`: drop low-variance features (percentile threshold), drop high-NA features, median impute remaining NAs
3. `select_features()`: Laplacian score ranking (graph-based, unsupervised local structure) + optional correlation-based redundancy removal; keep `--features_top_percentile`% (min `--features_min`)
4. Harmonize train/test to feature intersection
5. Optional log-transform
6. StandardScaler normalization (fit on train, transform test) — skipped entirely when `skip_normalization=True` (set automatically for `DeepTSP`, which requires pre-scaled log2 input; `DataImporterInference` falls back to the stored feature list instead of a fitted scaler in this case)
7. Wrap as `MultiOmicDataset` (PyTorch Dataset)

**Inference artifacts** (`flexynesis/inference.py`): After training, `<prefix>.artifacts.joblib` (or `.json` for safetensors mode) stores feature lists, fitted scalers, and label encoders for consistent preprocessing of new test data.

### Training Loop (`flexynesis/main.py`)

`HyperparameterTuning` runs Bayesian optimization (scikit-optimize `gp_hedge`) for `--hpo_iter` iterations. Each iteration uses either 5-fold CV (`--use_cv`) or a single 80/20 split. After HPO, retrains on the full dataset with the best config.

The training `DataLoader` uses `drop_last=True` — a `batch_size` at or above the (post-split) training set size silently trains on zero batches, no error raised. `get_batch_space()` caps the default HPO-sampled range at 128 for this reason; raising it (e.g. via `--config_path` for small cohorts) needs a value safely below the split size, not equal to or above it.

For `DeepTSP`, `HyperparameterTuning.min_epochs_for()` sets `Trainer(min_epochs=...)` from the sampled `prune_every`/`epochs`/`pruning_budget_frac`, so `EarlyStopping` can't fire before `GlobalPairPruner`'s hard-pruning schedule converges to `target_k` (deliberately `min_epochs`, not an inflated `patience` — the latter counts consecutive non-improving epochs from whenever the last improvement was, not from epoch 0, so padding it by the pruning schedule's length can silently consume most of the remaining epoch budget instead).

### HPO Search Spaces (`flexynesis/config.py`)

Most models share: `latent_dim` (16–128), `hidden_dim_factor` (0.2–0.5), `supervisor_hidden_dim` (8–32), `lr` (log-uniform 1e-4–1e-2), `batch_size` (powers of 2, 32–128, auto-capped to dataset size). `DeepTSP` has its own space instead: `hidden_dim`, `lr`, `epochs`, `target_k` (final surviving-pair count), `prune_every`, `max_genes_per_set`. Custom spaces can be provided via `--config_path <file.yaml>` (see `examples/tutorials/conf.yaml` or `examples/configs/hpo_configuration.yaml` for format).

### Model Interface

All DL models implement:
- `transform(dataset)` → sample latent embeddings
- `predict(dataset)` → predictions per target variable
- `compute_feature_importance(dataset, var, method)` → Captum IntegratedGradients or GradientShap scores (`DeepTSP` ignores `method` and reports its own gate/attention weights instead, labeled `attentionweight`)
- `decode(dataset)` → reconstructed omics (CrossModalPred and supervised_vae only)

### CLI (`flexynesis/__main__.py`)

Key flags not obvious from `--help`:
- `--surv_event_var` + `--surv_time_var`: required for survival tasks (target variable holds the group label)
- `--covariates col1,col2`: use clin.csv columns as additional input features
- `--pretrained_model` + `--artifacts` + `--data_path_test`: inference-only mode (skip training)
- `--safetensors`: save weights as SafeTensors + JSON instead of `.pth` + `.joblib`
- `--finetuning_samples N`: fine-tune on N test samples before evaluation (transfer learning); not supported for `DeepTSP` (its `encoders` is an `nn.ModuleDict`, not the `ModuleList`/`MLPs` shape `FineTuner.apply_freeze_config` assumes)
- `--gene_sets_path <standard-GMT-file>`: required for `DeepTSP` (single-modality only)
- `--disable_marker_finding`: skip Captum feature importance (saves time)
- `--evaluate_baseline_performance`: also run RF/SVM/XGBoost/RSF baselines for comparison
- `--string_organism` (default 9606=human), `--user_graph` CSV (GeneA, GeneB, Score columns) for GNN

### Output Files

All prefixed by `--prefix` (default `job`):
- `<prefix>.final_model.pth` / `.safetensors` — model weights
- `<prefix>.final_model_config.json` — architecture config
- `<prefix>.artifacts.joblib` / `.json` — scalers, feature lists, encoders for inference
- `<prefix>.embeddings_{train,test}.csv`, `predicted_labels.csv`, `stats.csv`
- `<prefix>.feature_importance.<method>.csv`, `feature_logs.<modality>.csv`

## Lazy Imports

`flexynesis/__init__.py` uses `LazyModule` proxies to defer heavy PyTorch/Lightning imports until actually needed. This is intentional to keep CLI startup fast — don't break it by eagerly importing at module level.
