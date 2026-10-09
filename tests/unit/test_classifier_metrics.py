import numpy as np
import pytest
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from flexynesis.utils import evaluate_classifier, evaluate_wrapper


def probs(rng, n, k):
    p = rng.random((n, k))
    return p / p.sum(1, keepdims=True)


def test_matches_sklearn_when_all_classes_present():
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 4, 80), probs(rng, 80, 4)
    m = evaluate_classifier(y, p)
    assert m["average_auroc"] == pytest.approx(
        roc_auc_score(y, p, multi_class="ovr", average="weighted")
    )
    onehot = np.eye(4)[y]
    assert m["average_aupr"] == pytest.approx(
        average_precision_score(onehot, p, average="weighted")
    )


def test_absent_class_still_gives_scores():
    rng = np.random.default_rng(1)
    y, p = rng.integers(0, 3, 60), probs(rng, 60, 4)  # class 3 never in y
    m = evaluate_classifier(y, p)
    assert m["average_auroc"] is not None and m["average_aupr"] is not None


def test_single_class_is_empty_not_error():
    rng = np.random.default_rng(2)
    m = evaluate_classifier(np.ones(20, dtype=int), probs(rng, 20, 2))
    assert m["average_auroc"] is None


class DS:
    variable_types = {"y": "categorical"}

    def __init__(self, ann):
        self.ann = {"y": ann}


def test_negative_and_nan_labels_are_excluded():
    rng = np.random.default_rng(3)
    p = probs(rng, 40, 4)
    y = torch.tensor(rng.integers(0, 4, 40))
    y_missing = y.clone()
    y_missing[:5] = -1  # missing/unseen labels encoded as -1 (int)
    ref = evaluate_wrapper("m", {"y": p[5:]}, DS(y[5:]))
    got = evaluate_wrapper("m", {"y": p}, DS(y_missing))
    np.testing.assert_allclose(ref["value"].astype(float), got["value"].astype(float))
    # float labels with NaN
    y_nan = y.float()
    y_nan[:5] = float("nan")
    got = evaluate_wrapper("m", {"y": p}, DS(y_nan))
    np.testing.assert_allclose(ref["value"].astype(float), got["value"].astype(float))


def test_predicted_labels_handle_missing_labels():
    from flexynesis.utils import get_predicted_labels

    class D:
        variable_types = {"y": "categorical"}
        label_mappings = {"y": {0: "a", 1: "b", -1: "Unknown"}}
        samples = ["s0", "s1", "s2"]
        ann = {"y": torch.tensor([0.0, float("nan"), -1.0])}

    p = np.array([[0.9, 0.1], [0.2, 0.8], [0.5, 0.5]])
    df = get_predicted_labels({"y": p}, D(), "test", "m")
    known = df.drop_duplicates("sample_id").set_index("sample_id")["known_label"]
    assert known["s0"] == "a"
    assert known["s1"] != known["s1"]  # NaN
    assert known["s2"] == "Label_not_in_train"
