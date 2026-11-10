import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from model_gate import evaluate  # noqa: E402

METRICS = json.loads((ROOT / "docs" / "model_metrics.json").read_text())


def passed(metrics):
    return all(ok for _, ok, _ in evaluate(metrics))


def test_recorded_metrics_pass_the_gate():
    assert passed(METRICS)


def test_blocks_a_small_gini_improvement():
    m = copy.deepcopy(METRICS)
    m["challenger"]["test_gini"] = m["champion"]["test_gini"] + 0.005
    assert not passed(m)


def test_blocks_a_fairness_regression():
    m = copy.deepcopy(METRICS)
    m["challenger"]["di_region"] = 0.60
    assert not passed(m)


def test_blocks_bad_pure_premium_calibration():
    m = copy.deepcopy(METRICS)
    m["challenger"]["pp_calibration_adj"] = 1.25
    assert not passed(m)