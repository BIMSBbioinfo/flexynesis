"""DeepTSP: gene-set-structured, minimality-constrained pairwise gene scoring.

Ties within-gene-set candidate gene pairs through a sparsemax-gated, per-set
pairwise scorer and a per-class attention fusion across sets, with a hard
global cap on the number of surviving pairs enforced between training epochs
(see `GlobalPairPruner`).

v1 scope: single omics modality, exactly one target variable (or one survival
event/time pair folded into one target), no batch variables, no covariate
augmentation, no graph-gated pairs.

`--finetuning_samples` (flexynesis.main.FineTuner) is not supported: its
apply_freeze_config() hard-codes DirectPred-style attribute shapes (`encoders`
as a ModuleList, plus an `MLPs` dict) that DeepTSP doesn't have (`encoders` is
an `nn.ModuleDict` keyed by gene set, and there's no separate `MLPs` head
dict), so it raises an AttributeError if used with this model class.

Trainer `min_epochs` is set automatically for DeepTSP (see
`HyperparameterTuning.min_epochs_for` in flexynesis/main.py) so EarlyStopping
can't fire before GlobalPairPruner's hard-pruning schedule has converged to
target_k -- each pruning step changes model capacity discontinuously and can
cause a transient validation-loss dip that isn't a real plateau. Deliberately
a `min_epochs` floor rather than an inflated `--early_stop_patience`: patience
counts consecutive non-improving epochs from whenever the last improvement
happened, not from epoch 0, so padding it by the pruning schedule's length can
consume most of the remaining epoch budget once pruning actually converges
and disable early stopping in practice. `--early_stop_patience` itself is
left exactly as given; it only starts being evaluated once min_epochs is
reached.
"""

from itertools import combinations

import lightning as pl
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ..feature_selection import laplacian_score
from ..modules import cox_ph_loss

__all__ = ["DeepTSP"]


# ---------------------------------------------------------------------------
# Ported building blocks
# ---------------------------------------------------------------------------


def sparsemax(z: torch.Tensor) -> torch.Tensor:
    """Project a 1-D score vector onto the probability simplex (Martins &
    Astudillo, 2016). Sample-independent gates only need the 1-D form."""
    z_sorted, _ = torch.sort(z, descending=True)
    z_cumsum = torch.cumsum(z_sorted, dim=0)
    k = torch.arange(1, z.numel() + 1, device=z.device, dtype=z.dtype)
    support = 1 + k * z_sorted > z_cumsum
    k_star = support.nonzero(as_tuple=True)[0].max() + 1
    tau = (z_cumsum[k_star - 1] - 1) / k_star.to(z.dtype)
    return torch.clamp(z - tau, min=0)


class PairGate(nn.Module):
    """One competing, sample-independent sparsemax weight per candidate pair
    in a gene set. `pruned_mask` supports hard global pruning (see
    `GlobalPairPruner`): masked-out pairs are forced to -inf before sparsemax.
    """

    def __init__(self, n_pairs: int):
        super().__init__()
        self.n_pairs = n_pairs
        self.raw_logits = nn.Parameter(torch.zeros(n_pairs))
        self.register_buffer("pruned_mask", torch.ones(n_pairs, dtype=torch.bool))

    def gate_weights(self) -> torch.Tensor:
        if not torch.any(self.pruned_mask):
            return torch.zeros_like(self.raw_logits)
        logits = self.raw_logits.masked_fill(~self.pruned_mask, float("-inf"))
        return sparsemax(logits)

    def forward(self, pair_features: torch.Tensor) -> torch.Tensor:
        """pair_features: [batch, n_pairs] -> gated [batch, n_pairs]."""
        return pair_features * self.gate_weights()


class SetEncoder(nn.Module):
    """Small MLP turning gated pair features into per-class logits."""

    def __init__(self, n_pairs: int, n_classes: int, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_pairs, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, gated_features):
        return self.net(gated_features)


class ClassAttentionFusion(nn.Module):
    """Per-class attention across gene-set branches. `attention_logits` is
    [n_sets, n_classes], softmax-normalized down each class column."""

    def __init__(self, n_sets: int, n_classes: int):
        super().__init__()
        self.attention_logits = nn.Parameter(torch.zeros(n_sets, n_classes))

    def forward(self, per_set_logits: torch.Tensor):
        """per_set_logits: [n_sets, batch, n_classes] ->
        (final_logits [batch, n_classes], attention [n_sets, n_classes])."""
        attention = torch.softmax(self.attention_logits, dim=0)
        final_logits = torch.einsum("sbc,sc->bc", per_set_logits, attention)
        return final_logits, attention


class _PairIndex(nn.Module):
    """Holds one gene set's precomputed within-set pair gene-index buffers, so
    `DeepTSP.forward` can compute log2FC pair features batch-wise via tensor
    indexing + subtraction rather than a precomputed static matrix (flexynesis
    batches arrive one DataLoader minibatch at a time)."""

    def __init__(self, gene_a_idx: list, gene_b_idx: list):
        super().__init__()
        self.register_buffer("gene_a_idx", torch.tensor(gene_a_idx, dtype=torch.long))
        self.register_buffer("gene_b_idx", torch.tensor(gene_b_idx, dtype=torch.long))


# ---------------------------------------------------------------------------
# Ported global pruning
# ---------------------------------------------------------------------------


class GlobalPairPruner:
    """Periodically hard-masks the globally lowest-weighted pairs across all
    of a DeepTSP model's per-gene-set PairGate branches, converging to an
    exact total pair count (`target_k`). Runs between optimizer steps (see
    `DeepTSP.on_train_epoch_end`), not inside forward -- it's a discrete op on
    the persistent `pruned_mask` buffer, not meant to be differentiable.

    Ranking is each pair's within-set sparsemax weight scaled by its gene
    set's fusion attention (mean across classes), so pairs from sets the
    model relies on for the final prediction are favored over pairs that
    merely won local within-set competition in a set the fusion layer
    barely attends to.
    """

    def __init__(self, model, target_k: int, prune_every: int = 5, prune_fraction: float = 0.2):
        self.model = model
        self.target_k = target_k
        self.prune_every = prune_every
        self.prune_fraction = prune_fraction

    @classmethod
    def adaptive(
        cls,
        model,
        target_k: int,
        n_epochs: int,
        prune_every: int = 5,
        pruning_budget_frac: float = 0.5,
    ) -> "GlobalPairPruner":
        """Derive `prune_fraction` so pruning reaches `target_k` within
        `pruning_budget_frac` of `n_epochs`, regardless of starting pool size."""
        n_start = sum(int(gate.pruned_mask.sum()) for gate in model.gates.values())
        n_rounds = max(1, int((n_epochs * pruning_budget_frac) // prune_every))
        prune_fraction = 0.0 if n_start <= target_k else 1 - (target_k / n_start) ** (1 / n_rounds)
        return cls(model, target_k=target_k, prune_every=prune_every, prune_fraction=prune_fraction)

    def n_active_pairs(self) -> int:
        return sum(int(gate.pruned_mask.sum()) for gate in self.model.gates.values())

    def step(self, epoch: int) -> int:
        """Call once per epoch (1-indexed). Returns pairs remaining."""
        n_active = self.n_active_pairs()
        if epoch % self.prune_every != 0 or n_active <= self.target_k:
            return n_active
        n_to_prune = min(int(n_active * self.prune_fraction), n_active - self.target_k)
        if n_to_prune <= 0:
            return n_active

        # Cross-set comparison needs a common scale: a pair's own sparsemax weight only
        # reflects local competition within its set, not whether that set matters to the
        # final prediction, so scale by the set's fusion attention (mean across classes)
        # before pooling and sorting globally.
        set_attention = torch.softmax(self.model.fusion.attention_logits.detach(), dim=0).mean(dim=1)
        set_weight = dict(zip(self.model.set_names, set_attention.tolist()))

        scored = []
        for set_name, gate in self.model.gates.items():
            weights = gate.gate_weights().detach()
            for pair_idx in torch.nonzero(gate.pruned_mask, as_tuple=True)[0].tolist():
                scored.append((weights[pair_idx].item() * set_weight[set_name], set_name, pair_idx))
        scored.sort(key=lambda x: x[0])  # lowest attention-weighted score first
        for _, set_name, pair_idx in scored[:n_to_prune]:
            self.model.gates[set_name].pruned_mask[pair_idx] = False
        return self.n_active_pairs()


# ---------------------------------------------------------------------------
# Gene-set / pair I/O. Standard-GMT only -- no format auto-detection; the
# caller is responsible for supplying a correctly formatted GMT file.
# ---------------------------------------------------------------------------


def load_gene_sets(path) -> dict:
    """Parse a standard GMT file: name, description, then one gene per
    remaining tab-separated field (variable number of genes per line)."""
    gene_sets = {}
    with open(path) as f:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            name, genes = fields[0], fields[2:]
            gene_sets[name] = [g for g in genes if g]
    return gene_sets


def load_and_intersect_genesets(gmt: dict, available_genes, min_genes: int = 5) -> dict:
    """Restrict each gene set to genes present in the data; drop sets left
    with fewer than `min_genes` genes."""
    available = set(available_genes)
    out = {}
    for name, genes in gmt.items():
        kept = list(dict.fromkeys(g for g in genes if g in available))
        if len(kept) >= min_genes:
            out[name] = kept
    return out


def _rank_genes_by_laplacian(expr: pd.DataFrame, k: int = 5) -> pd.Series:
    """Rank genes within one gene set by unsupervised Laplacian score (lower =
    more informative). expr: genes x samples. Ascending Series, best first."""
    X = expr.T.to_numpy()  # laplacian_score expects samples x features
    scores = laplacian_score(X, k=k)
    return pd.Series(scores, index=expr.index).sort_values()


def cap_genesets(genesets: dict, expr: pd.DataFrame, max_genes: int = 50, k: int = 5) -> dict:
    """Cap each gene set to its top `max_genes` genes by a per-set Laplacian
    rank (only trims sets that exceed the cap; ranks each set against its own
    submatrix, not a global ranking, so generic globally-informative genes
    can't leak into unrelated sets)."""
    capped = {}
    for name, genes in genesets.items():
        if len(genes) <= max_genes:
            capped[name] = genes
            continue
        scores = _rank_genes_by_laplacian(expr.loc[genes], k=k)
        capped[name] = scores.index[:max_genes].tolist()
    return capped


def enumerate_pairs(genes: list) -> list:
    """All unordered gene pairs within an already-capped gene set."""
    return list(combinations(genes, 2))


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


class DeepTSP(pl.LightningModule):
    def __init__(
        self,
        config,
        dataset,
        target_variables,
        batch_variables=None,
        surv_event_var=None,
        surv_time_var=None,
        use_loss_weighting=True,
        device_type=None,
    ):
        super().__init__()

        if batch_variables is not None:
            raise ValueError(
                "DeepTSP does not support batch_variables (v1 is single-modality, "
                "single-target only)."
            )
        if len(dataset.dat) != 1:
            raise ValueError(f"DeepTSP only supports single-modality datasets, got {len(dataset.dat)}.")

        target_variables = list(target_variables) if target_variables else []
        if surv_event_var is not None and surv_time_var is not None:
            target_variables = target_variables + [surv_event_var]
        if len(target_variables) != 1:
            raise ValueError(
                "DeepTSP v1 supports exactly one target variable in total: either one "
                "entry in `target_variables`, or a survival event/time pair (which "
                f"folds into one target). Got {len(target_variables)}: {target_variables}."
            )

        self.config = config
        self.target_variables = target_variables
        self.batch_variables = batch_variables
        self.surv_event_var = surv_event_var
        self.surv_time_var = surv_time_var
        self.variables = self.target_variables
        self.feature_importances = {}
        # kept for constructor-signature parity with other model classes; a
        # single-target model has nothing to uncertainty-weight against
        self.use_loss_weighting = use_loss_weighting
        self.device_type = device_type

        self.variable_types = dataset.variable_types
        self.ann = dataset.ann
        self.layer = list(dataset.dat.keys())[0]
        self.layers = [self.layer]

        if "gene_sets_path" not in config or not config["gene_sets_path"]:
            raise ValueError(
                "config['gene_sets_path'] is required for DeepTSP: path to a "
                "standard-GMT gene-set file."
            )

        gene_names = list(dataset.features[self.layer])
        gmt = load_gene_sets(config["gene_sets_path"])
        genesets = load_and_intersect_genesets(gmt, gene_names, min_genes=int(config.get("min_genes_per_set", 5)))
        if not genesets:
            raise ValueError(
                "No gene sets from --gene_sets_path had enough overlapping genes with "
                "the dataset (see `min_genes_per_set`)."
            )

        # genes x samples matrix for per-set Laplacian capping; dataset.dat[layer]
        # is assumed already log2-scale by the time it reaches this model (see
        # DataImporter's skip_normalization).
        expr = pd.DataFrame(
            dataset.dat[self.layer].numpy().T, index=gene_names, columns=list(dataset.samples)
        )
        capped = cap_genesets(
            genesets,
            expr,
            max_genes=int(config.get("max_genes_per_set", 50)),
            k=int(config.get("laplacian_k", 5)),
        )
        pairs_per_set = {name: enumerate_pairs(genes) for name, genes in capped.items()}
        pairs_per_set = {name: p for name, p in pairs_per_set.items() if len(p) > 0}
        if not pairs_per_set:
            raise ValueError("No gene set had >= 2 genes after capping -- no pairs to form.")
        self.pairs_per_set = pairs_per_set
        self.set_names = list(pairs_per_set.keys())

        gene_to_idx = {g: i for i, g in enumerate(gene_names)}
        self.pair_indices = nn.ModuleDict(
            {
                name: _PairIndex(
                    [gene_to_idx[a] for a, b in pairs],
                    [gene_to_idx[b] for a, b in pairs],
                )
                for name, pairs in pairs_per_set.items()
            }
        )

        var = self.target_variables[0]
        if self.variable_types[var] == "numerical":
            self.n_classes = 1
        else:
            self.n_classes = len(np.unique(self.ann[var]))

        hidden_dim = int(config.get("hidden_dim", 16))
        n_pairs_per_set = {name: len(p) for name, p in pairs_per_set.items()}
        self.gates = nn.ModuleDict({name: PairGate(n) for name, n in n_pairs_per_set.items()})
        self.encoders = nn.ModuleDict(
            {name: SetEncoder(n, self.n_classes, hidden_dim) for name, n in n_pairs_per_set.items()}
        )
        self.fusion = ClassAttentionFusion(len(self.set_names), self.n_classes)

        target_k = int(config.get("target_k", 20))
        prune_every = int(config.get("prune_every", 5))
        if "epochs" in config:
            self.pruner = GlobalPairPruner.adaptive(
                self,
                target_k=target_k,
                n_epochs=int(config["epochs"]),
                prune_every=prune_every,
                pruning_budget_frac=float(config.get("pruning_budget_frac", 0.5)),
            )
        else:
            self.pruner = GlobalPairPruner(
                self, target_k=target_k, prune_every=prune_every, prune_fraction=float(config.get("prune_fraction", 0.2))
            )

    # -- forward / core compute ------------------------------------------------

    def _per_set_logits(self, x: torch.Tensor):
        """x: [batch, n_genes]. Returns (stacked [n_sets,batch,n_classes], gate_weights dict)."""
        per_set_logits = []
        gate_weights = {}
        for name in self.set_names:
            idx = self.pair_indices[name]
            pair_features = x[:, idx.gene_a_idx] - x[:, idx.gene_b_idx]
            gated = self.gates[name](pair_features)
            per_set_logits.append(self.encoders[name](gated))
            gate_weights[name] = self.gates[name].gate_weights().detach()
        return torch.stack(per_set_logits, dim=0), gate_weights

    def forward(self, dat):
        """dat: {layer: [batch, n_genes] tensor}.
        Returns (final_logits [batch, n_classes], {set: gate_weights}, attention)."""
        x = dat[self.layer]
        stacked, gate_weights = self._per_set_logits(x)
        final_logits, attention = self.fusion(stacked)
        return final_logits, gate_weights, attention

    # -- loss --------------------------------------------------------------------

    def compute_loss(self, var, y, y_hat):
        """Ported from DirectPred.compute_loss, specialized to one variable."""
        if self.variable_types[var] == "numerical":
            valid = ~torch.isnan(y)
            if valid.sum() > 0:
                loss = F.mse_loss(torch.flatten(y_hat[valid]), y[valid].float())
            else:
                loss = torch.tensor(0.0, device=y_hat.device, requires_grad=True)
        else:
            valid = (y != -1) & (~torch.isnan(y))
            if valid.sum() > 0:
                loss = F.cross_entropy(y_hat[valid], y[valid].long())
            else:
                loss = torch.tensor(0.0, device=y_hat.device, requires_grad=True)
        return loss

    def _target_loss(self, logits, y_dict):
        var = self.target_variables[0]
        if var == self.surv_event_var:
            return cox_ph_loss(logits, y_dict[self.surv_time_var], y_dict[self.surv_event_var])
        return self.compute_loss(var, y_dict[var], logits)

    def training_step(self, train_batch, batch_idx, log=True):
        dat, y_dict, samples = train_batch
        logits, _, _ = self.forward(dat)
        loss = self._target_loss(logits, y_dict)
        if log:
            self.log_dict(
                {self.target_variables[0]: loss, "train_loss": loss}, on_step=False, on_epoch=True, prog_bar=True
            )
        return loss

    def validation_step(self, val_batch, batch_idx, log=True):
        dat, y_dict, samples = val_batch
        logits, _, _ = self.forward(dat)
        loss = self._target_loss(logits, y_dict)
        if log:
            self.log_dict(
                {self.target_variables[0]: loss, "val_loss": loss}, on_step=False, on_epoch=True, prog_bar=True
            )
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config["lr"])

    def on_train_epoch_end(self):
        if self.pruner is None:
            return
        n_active = self.pruner.step(self.current_epoch + 1)  # pruner is 1-indexed
        self.log("active_pairs", float(n_active), on_step=False, on_epoch=True, prog_bar=True)

    # -- inference ----------------------------------------------------------------

    def predict(self, dataset):
        self.eval()
        from ..utils import create_device_from_string, to_device_safe

        device = create_device_from_string(self.device_type or "auto")
        self.to(device)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
        var = self.target_variables[0]
        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                dat, y_dict, samples = batch
                dat = {k: to_device_safe(v, device) for k, v in dat.items()}
                logits, _, _ = self.forward(dat)
                logits = logits.detach().cpu()
                if dataset.variable_types[var] == "categorical":
                    predictions.extend(torch.softmax(logits, dim=1).numpy())
                else:
                    predictions.extend(logits.numpy())
        return {var: np.array(predictions)}

    def transform(self, dataset):
        """Embedding = concatenated pre-fusion per-set logits (stacked branch
        outputs before ClassAttentionFusion) -- the DeepTSP analogue of
        DirectPred's pre-supervisor-head fused embedding."""
        self.eval()
        from ..utils import create_device_from_string, to_device_safe

        device = create_device_from_string(self.device_type or "auto")
        self.to(device)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
        embeddings_list, sample_names = [], []
        with torch.no_grad():
            for batch in dataloader:
                dat, _, samples = batch
                x = to_device_safe(dat[self.layer], device)
                stacked, _ = self._per_set_logits(x)  # [n_sets, batch, n_classes]
                embeddings_batch = stacked.permute(1, 0, 2).reshape(stacked.shape[1], -1)
                embeddings_list.append(embeddings_batch.detach().cpu())
                sample_names.extend(samples)
        embeddings_concat = torch.cat(embeddings_list, dim=0)
        columns = [f"{name}_C{c}" for name in self.set_names for c in range(self.n_classes)]
        return pd.DataFrame(embeddings_concat.numpy(), index=sample_names, columns=columns)

    def compute_feature_importance(self, dataset, target_var, method=None, **kwargs):
        """Ignores `method`/Captum entirely. For every pair with nonzero
        PairGate weight, emits one row per class of `target_var`, with
        importance = gate_weight * that gene set's fusion attention weight
        for that class.
        """
        if target_var != self.target_variables[0]:
            raise ValueError(
                f"DeepTSP only supports its single configured target variable "
                f"'{self.target_variables[0]}', got '{target_var}'."
            )
        attention = torch.softmax(self.fusion.attention_logits, dim=0).detach().cpu()
        label_mapping = dataset.label_mappings.get(target_var, {})
        rows = []
        for set_idx, name in enumerate(self.set_names):
            weights = self.gates[name].gate_weights().detach().cpu()
            for pair_idx, (gene_a, gene_b) in enumerate(self.pairs_per_set[name]):
                w = weights[pair_idx].item()
                if w <= 0:
                    continue
                for class_idx in range(self.n_classes):
                    rows.append(
                        {
                            "target_variable": target_var,
                            "target_class": class_idx,
                            "target_class_label": label_mapping.get(class_idx, ""),
                            "layer": self.layer,
                            "name": f"{gene_a}_{gene_b}",
                            "importance": w * attention[set_idx, class_idx].item(),
                        }
                    )
        df = pd.DataFrame(
            rows, columns=["target_variable", "target_class", "target_class_label", "layer", "name", "importance"]
        )
        self.feature_importances[target_var] = df
        return df
