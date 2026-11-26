import numpy as np


def regime_metrics(A, pred):
    """RMSE / MAE / R2 / bias overall, for A <= 15 and for A > 15 (µSv/h)."""
    out = {}
    for name, m in (("all", np.ones_like(A, bool)), ("low", A <= 15), ("high", A > 15)):
        e = pred[m] - A[m]
        out[f"rmse_{name}"] = float(np.sqrt(np.mean(e ** 2)))
        out[f"mae_{name}"] = float(np.mean(np.abs(e)))
        out[f"bias_{name}"] = float(np.mean(e))
        ss = np.sum((A[m] - A[m].mean()) ** 2)
        out[f"r2_{name}"] = float(1 - np.sum(e ** 2) / ss)
    return out
