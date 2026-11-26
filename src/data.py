"""Load the ARMAS data once, save a fast cache, and build model inputs.

Features
  static   (n, 8)      position + geomagnetic shielding at the observation time
  temporal (n, T, 35)  space-weather history, oldest -> current (T=12 for 5M, 24 for 1H)
  current  (n, 43)     static + the current (last) step of every temporal feature
Targets
  A = ARMAS (measured), N = NAIRAS V3 (physics), r = A - N (correction)
"""
from pathlib import Path
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STATIC = ["Latitude", "Longitude", "Altitude(Bar)", "Altitude(GPS)",
          "Geomagnetic_latitude", "Geomagnetic_longitude", "Geomagnetic_Rc",
          "Geomagnetic_Lshell"]
PARTS = ("P1", "P2", "P3")


def _build(cadence):
    static, temporal, A, N, part, times = [], [], [], [], [], []
    env = None
    for p in PARTS:
        path = DATA / cadence / f"ARMAS_TIMESERIES_{cadence}_20132023_{p}.json"
        for line in path.open():
            if not line.strip():
                continue
            r = json.loads(line)
            if env is None:
                env = [k for k in r if isinstance(r[k], list)]
            static.append([r[k] for k in STATIC])
            temporal.append([r[k][::-1] for k in env])   # reverse: oldest -> current
            A.append(r["ARMAS"]); N.append(r["NAIRASV3"])
            part.append(p); times.append(r["Datetime"])
    return dict(static=np.array(static, np.float32),
                temporal=np.array(temporal, np.float32).transpose(0, 2, 1),
                A=np.array(A, np.float32), N=np.array(N, np.float32),
                part=np.array(part), time=np.array(times), env=np.array(env))


def load(cadence="5M"):
    cache = DATA / f"cache_{cadence}.npz"
    if not cache.exists():
        np.savez(cache, **_build(cadence))
    d = dict(np.load(cache, allow_pickle=False))
    d["current"] = np.concatenate([d["static"], d["temporal"][:, -1, :]], axis=1)
    d["feature_names"] = STATIC + list(d["env"])
    return d


def split(d, train="P1", val="P2", test="P3"):
    """Boolean masks for one train/validation/test assignment."""
    return {k: d["part"] == p for k, p in (("train", train), ("val", val), ("test", test))}


class Scaler:
    """Standardize with TRAINING statistics only (no leakage)."""
    def fit(self, x):
        axes = tuple(range(x.ndim - 1))
        self.mu = x.mean(axes, keepdims=True)
        self.sd = x.std(axes, keepdims=True) + 1e-6
        return self

    def __call__(self, x):
        return (x - self.mu) / self.sd
