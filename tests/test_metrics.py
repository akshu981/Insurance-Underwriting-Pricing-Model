import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from metrics import gini, psi_binary, psi_continuous  # noqa: E402


def test_gini_high_when_claims_sit_on_top_ranked_policies():
    n = 1000
    predicted = np.arange(n)
    actual = (np.arange(n) >= 900).astype(float)
    assert gini(actual, predicted, np.ones(n)) > 0.85


def test_gini_negative_when_ranking_is_reversed():
    n = 1000
    predicted = -np.arange(n)
    actual = (np.arange(n) >= 900).astype(float)
    assert gini(actual, predicted, np.ones(n)) < -0.85


def test_gini_near_zero_when_claims_are_uniform():
    n = 1000
    assert abs(gini(np.ones(n), np.arange(n), np.ones(n))) < 0.01


def test_gini_uses_exposure_weights():
    n = 500
    rng = np.random.default_rng(0)
    predicted = rng.random(n)
    actual = rng.poisson(0.1, n).astype(float)
    exposure = rng.uniform(0.1, 1.0, n)
    assert gini(actual, predicted, exposure) != gini(actual, predicted, np.ones(n))


def test_psi_continuous_zero_for_identical_samples():
    x = np.random.default_rng(1).normal(size=5000)
    assert psi_continuous(x, x) < 1e-6


def test_psi_continuous_flags_a_shifted_sample():
    rng = np.random.default_rng(2)
    assert psi_continuous(rng.normal(0, 1, 10000), rng.normal(1, 1, 10000)) > 0.2


def test_psi_binary_zero_for_same_rate_and_large_for_big_shift():
    a = np.array([0] * 90 + [1] * 10)
    b = np.array([0] * 50 + [1] * 50)
    assert psi_binary(a, a) < 1e-9
    assert psi_binary(a, b) > 0.2