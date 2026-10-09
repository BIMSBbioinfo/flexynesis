import numpy as np
import pytest
import torch

from flexynesis.data import MultiOmicDataset
from flexynesis.models.crossmodal_pred import CrossModalPred
from flexynesis.models.supervised_vae import supervised_vae
from flexynesis.utils import decode_dataset, evaluate_reconstruction

CONFIG = {"latent_dim": 4, "hidden_dim_factor": 0.5, "supervisor_hidden_dim": 8}


def make_dataset(n=20):
    rng = np.random.default_rng(0)
    layers = {"gex": 12, "cnv": 7}
    dat = {k: torch.tensor(rng.normal(size=(n, d)), dtype=torch.float32)
           for k, d in layers.items()}
    features = {k: [f"{k}_{i}" for i in range(d)] for k, d in layers.items()}
    samples = [f"s{i}" for i in range(n)]
    return MultiOmicDataset(dat, {}, {}, features, samples, {})


@pytest.mark.parametrize("cls", [supervised_vae, CrossModalPred])
def test_decode_and_stats(cls):
    ds = make_dataset()
    model = cls(CONFIG, ds, target_variables=[])
    decoded = decode_dataset(model, ds, batch_size=7)
    assert set(decoded) == {"gex", "cnv"}
    assert decoded["gex"].shape == (12, 20)
    assert list(decoded["gex"].columns) == ds.samples
    assert list(decoded["cnv"].index) == ds.features["cnv"]
    # decoding is deterministic and independent of batch size
    again = decode_dataset(model, ds, batch_size=20)
    np.testing.assert_allclose(decoded["gex"].values, again["gex"].values, atol=1e-5)
    stats = evaluate_reconstruction(model, ds, cls.__name__)
    assert set(stats["var"]) == {"reconstruction_gex", "reconstruction_cnv"}
    assert set(stats["metric"]) == {
        "mse", "sample_corr_mean", "sample_corr_median",
        "feature_corr_mean", "feature_corr_median",
    }


def test_perfect_reconstruction_gives_unit_correlation():
    ds = make_dataset()
    model = supervised_vae(CONFIG, ds, target_variables=[])
    model.decode_batch = lambda dat: {k: v.clone() for k, v in dat.items()}
    stats = evaluate_reconstruction(model, ds, "supervised_vae")
    vals = stats.set_index(["var", "metric"])["value"]
    for layer in ("gex", "cnv"):
        assert vals[(f"reconstruction_{layer}", "mse")] == pytest.approx(0, abs=1e-10)
        assert vals[(f"reconstruction_{layer}", "sample_corr_mean")] == pytest.approx(1)
        assert vals[(f"reconstruction_{layer}", "feature_corr_mean")] == pytest.approx(1)
