"""Validation and monitoring metrics shared by the notebooks and the CI tests."""
import numpy as np
import pandas as pd

_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def gini(actual, predicted, exposure):
    """Exposure-weighted Gini: rank by predicted rate, cumulate claims vs exposure."""
    df = pd.DataFrame({"actual": actual, "predicted": predicted, "exposure": exposure})
    df = df.sort_values("predicted").reset_index(drop=True)
    cum_a = df["actual"].cumsum() / df["actual"].sum()
    cum_e = df["exposure"].cumsum() / df["exposure"].sum()
    return 1 - 2 * _trapezoid(cum_a, cum_e)


def psi_continuous(base, live, bins=10):
    """Population Stability Index for a continuous feature, using baseline quantile bins."""
    base, live = np.asarray(base, dtype=float), np.asarray(live, dtype=float)
    edges = np.unique(np.quantile(base, np.linspace(0, 1, bins + 1)))
    interior = edges[1:-1]
    n = len(interior) + 1
    b = np.bincount(np.digitize(base, interior), minlength=n) / len(base)
    l = np.bincount(np.digitize(live, interior), minlength=n) / len(live)
    b, l = np.clip(b, 1e-4, None), np.clip(l, 1e-4, None)
    return float(np.sum((l - b) * np.log(l / b)))


def psi_binary(base, live):
    """Population Stability Index for a 0/1 feature."""
    pb = np.clip(np.mean(base), 1e-4, 1 - 1e-4)
    pl = np.clip(np.mean(live), 1e-4, 1 - 1e-4)
    return float((pl - pb) * np.log(pl / pb) + ((1 - pl) - (1 - pb)) * np.log((1 - pl) / (1 - pb)))