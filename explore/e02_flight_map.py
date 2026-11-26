"""Exploration 2: where were the flights, and where is the dose high?

Three world maps of all 92,415 observations (P1+P2+P3):
  (a) measured ARMAS dose rate
  (b) geomagnetic cutoff rigidity Rc (the strength of Earth's magnetic shield)
  (c) NAIRAS error r = A - N
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.insert(0, str(Path(__file__).parent))
from load import load, FIGS

df = pd.concat([load("5M", p)[0] for p in ("P1", "P2", "P3")], ignore_index=True)

panels = [
    ("ARMAS", "(a) Measured dose rate ARMAS (µSv/h)", "inferno",
     mcolors.Normalize(0, 25), "ascending"),
    ("Geomagnetic_Rc", "(b) Cutoff rigidity Rc (GV): low = weak magnetic shield", "viridis_r",
     mcolors.Normalize(0, 15), "descending"),
    ("residual", "(c) NAIRAS error r = A − N (µSv/h): red = NAIRAS too low", "RdBu_r",
     mcolors.TwoSlopeNorm(0, -10, 10), "abs"),
]

fig = plt.figure(figsize=(12, 15))
for i, (col, title, cmap, norm, order) in enumerate(panels):
    ax = fig.add_subplot(3, 1, i + 1, projection=ccrs.PlateCarree())
    ax.set_extent([-180, 180, -10, 90], ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="#eeeeee")
    ax.add_feature(cfeature.COASTLINE, lw=.4, edgecolor="#777777")
    ax.gridlines(draw_labels=True, lw=.3, color="gray", alpha=.5,
                 xlocs=range(-180, 181, 60), ylocs=range(0, 91, 30))
    v = df[col].values
    key = {"ascending": v, "descending": -v, "abs": np.abs(v)}[order]
    s = np.argsort(key)  # draw the most important points last (on top)
    sc = ax.scatter(df["Longitude"].values[s], df["Latitude"].values[s], c=v[s],
                    cmap=cmap, norm=norm, s=1.2, transform=ccrs.PlateCarree())
    ax.set_title(title, loc="left", fontsize=11)
    fig.colorbar(sc, ax=ax, shrink=.75, pad=.02)

fig.suptitle(f"Exploration 2 — Where are the flights and the high doses?  (n = {len(df):,})",
             fontweight="bold", fontsize=13)
fig.tight_layout()
out = FIGS / "e02_flight_map.png"
fig.savefig(out, dpi=140)
print("saved", out)

# numbers for discussion: dose and NAIRAS error by Rc band
df["Rc_band"] = pd.cut(df["Geomagnetic_Rc"], [-0.01, 1, 3, 6, 10, 20],
                       labels=["0-1", "1-3", "3-6", "6-10", ">10"])
t = df.groupby("Rc_band", observed=True).agg(
    n=("ARMAS", "size"), ARMAS_median=("ARMAS", "median"),
    share_A_gt_15=("ARMAS", lambda x: (x > 15).mean()),
    mean_error=("residual", "mean"))
print(t.round(2))
