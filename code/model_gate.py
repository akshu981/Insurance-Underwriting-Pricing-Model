"""Promotion gate: decides whether the challenger may replace the champion.

Usage: python code/model_gate.py docs/model_metrics.json
Exits with status 1 if any rule fails, so CI blocks the change.
"""
import json
import sys

RULES = {
    "min_gini_improvement": 0.01,   # challenger test Gini must beat the champion by at least this
    "max_di_drop": 0.03,            # challenger DI ratio may fall at most this far below the champion's
    "di_floor": 0.70,               # absolute minimum disparate-impact ratio
    "calibration_band": 0.10,       # adjusted pure-premium calibration must be within +/-10% of 1.0
}


def evaluate(metrics, rules=RULES):
    champ, chall = metrics["champion"], metrics["challenger"]
    checks = []

    gain = chall["test_gini"] - champ["test_gini"]
    checks.append((f"Gini improvement >= {rules['min_gini_improvement']}",
                   gain >= rules["min_gini_improvement"], f"{gain:+.4f}"))

    pp_gain = chall["pure_premium_gini"] - champ["pure_premium_gini"]
    checks.append(("Pure premium Gini not worse", pp_gain >= 0, f"{pp_gain:+.4f}"))

    for key, label in [("di_density", "density"), ("di_region", "region")]:
        drop = champ[key] - chall[key]
        checks.append((f"DI ({label}) within {rules['max_di_drop']} of champion",
                       drop <= rules["max_di_drop"], f"{chall[key]:.3f} vs {champ[key]:.3f}"))
        checks.append((f"DI ({label}) >= {rules['di_floor']}",
                       chall[key] >= rules["di_floor"], f"{chall[key]:.3f}"))

    cal = chall["pp_calibration_adj"]
    checks.append((f"Pure premium calibration within +/-{rules['calibration_band']:.0%}",
                   abs(cal - 1) <= rules["calibration_band"], f"{cal:.3f}"))
    return checks


def main(path):
    with open(path) as f:
        metrics = json.load(f)
    checks = evaluate(metrics)
    width = max(len(name) for name, _, _ in checks)
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    if all(ok for _, ok, _ in checks):
        print("\nGATE: PASSED - challenger may be promoted")
        return 0
    print("\nGATE: FAILED - challenger is not promoted")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "docs/model_metrics.json"))