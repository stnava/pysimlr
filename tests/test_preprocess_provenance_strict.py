"""Replaying a preprocessing pipeline must use the training statistics.

Every per-method branch in `preprocess_data` was guarded by
`if provenance and KEY in provenance`. A missing key -- or an empty dict, since
`if {}` is False -- fell through to the `else` branch and recomputed the
statistic from whatever matrix was passed in, i.e. silently refit the transform
on held-out data with no error and no warning. Separately, a method appearing
twice in `scale_list` wrote both steps' statistics to one key.
"""
import pytest
import torch

from pysimlr.utils import preprocess_data


@pytest.mark.parametrize("scale_list", [
    ["centerAndScale"],
    ["center"],
    ["norm"],
    ["np"],
    ["sqrtnp"],
    ["eigenvalue"],
    ["centerAndScale", "np"],
    ["center", "norm", "sqrtnp"],
])
def test_replay_uses_training_statistics(scale_list):
    torch.manual_seed(0)
    train = torch.randn(100, 4) * 10.0 + 5.0
    test = torch.randn(80, 4) * 1.0 + 50.0        # deliberately different

    fitted, prov = preprocess_data(train, scale_list)
    replayed_train = preprocess_data(train, scale_list, provenance=prov)
    assert torch.allclose(replayed_train, fitted, atol=1e-5), (
        "replaying on the training data must reproduce the fit exactly"
    )

    replayed_test = preprocess_data(test, scale_list, provenance=prov)
    refit_test, _ = preprocess_data(test, scale_list)
    if "none" not in scale_list:
        assert not torch.allclose(replayed_test, refit_test, atol=1e-3), (
            "replaying the training transform on shifted test data produced the "
            "same result as refitting on it -- the training statistics were ignored"
        )


@pytest.mark.parametrize("bad", [
    {},
    {"nan_fill": 0.0},
    {"cas_mean": torch.zeros(4)},          # missing cas_std
    {"center_mean": torch.zeros(4)},       # wrong method's statistics
])
def test_incomplete_provenance_is_refused(bad):
    torch.manual_seed(1)
    test = torch.randn(50, 4) + 100.0
    with pytest.raises(KeyError, match="provenance"):
        preprocess_data(test, ["centerAndScale"], provenance=bad)


def test_repeated_method_gets_its_own_statistics():
    torch.manual_seed(2)
    x = torch.randn(60, 3) * 4.0 + 2.0
    fitted, prov = preprocess_data(x, ["center", "center"])
    assert "center_mean" in prov and "center_mean@1" in prov, (
        f"both occurrences must be recorded separately, got {sorted(prov)}"
    )
    replayed = preprocess_data(x, ["center", "center"], provenance=prov)
    assert torch.allclose(replayed, fitted, atol=1e-6)


def test_legacy_flat_provenance_still_replays():
    """Provenance written before positional namespacing used bare keys."""
    torch.manual_seed(3)
    x = torch.randn(40, 3)
    fitted, prov = preprocess_data(x, ["centerAndScale"])
    assert set(prov) == {"nan_fill", "cas_mean", "cas_std"}
    assert torch.allclose(
        preprocess_data(x, ["centerAndScale"], provenance=prov), fitted, atol=1e-6
    )


def test_none_scale_list_is_a_passthrough():
    x = torch.randn(10, 3)
    out, prov = preprocess_data(x, ["none"])
    assert torch.allclose(out, x)
    assert torch.allclose(preprocess_data(x, ["none"], provenance=prov), x)


def test_nan_fill_comes_from_training_not_test():
    train = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    test = torch.tensor([[float("nan"), 900.0], [900.0, 900.0]])
    _, prov = preprocess_data(train, ["none"])
    out = preprocess_data(test, ["none"], provenance=prov)
    assert out[0, 0] == pytest.approx(prov["nan_fill"]), (
        "NaNs in held-out data must be filled with the training fill value"
    )
