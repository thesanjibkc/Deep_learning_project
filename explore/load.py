"""Load the ARMAS V3 time-series partitions for exploration.

Each JSON file is newline-delimited: one flight observation per line.
Scalar fields describe the observation time; each of the 35 environmental
variables is a list whose index 0 is the CURRENT value and later indices go
back in time (5M: 12 steps x 5 min, 1H: 24 steps x 1 h).
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGS = ROOT / "figures" / "explore"

SCALARS = ["Datetime", "ARMAS", "NAIRASV3", "NAIRASV2", "Latitude", "Longitude",
           "Altitude(Bar)", "Altitude(GPS)", "Geomagnetic_latitude",
           "Geomagnetic_longitude", "Geomagnetic_Rc", "Geomagnetic_Lshell",
           "Vehicle_ID"]


def load(cadence="5M", partition="P1"):
    """Return (df, X): df has scalars + current env values; X is
    [n, steps, 35] in CHRONOLOGICAL order (oldest first, current last)."""
    path = DATA / cadence / f"ARMAS_TIMESERIES_{cadence}_20132023_{partition}.json"
    rows = [json.loads(line) for line in path.open() if line.strip()]
    env = [k for k in rows[0] if isinstance(rows[0][k], list)]
    df = pd.DataFrame({k: [r[k] for r in rows] for k in SCALARS})
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    X = np.array([[r[k] for k in env] for r in rows], dtype=np.float32)  # n,35,steps
    X = X[:, :, ::-1].transpose(0, 2, 1).copy()                          # n,steps,35
    for j, k in enumerate(env):
        df[k] = X[:, -1, j]                                              # current value
    df["residual"] = df["ARMAS"] - df["NAIRASV3"]
    df["partition"] = partition
    return df, X, env
