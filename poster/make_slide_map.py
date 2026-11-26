"""Wide two-panel world map for the slides: measured dose, and NAIRAS error."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.colors as mcolors
import cartopy.crs as ccrs, cartopy.feature as cfeature
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import load
d = load("5M"); lat, lon = d["static"][:, 0], d["static"][:, 1]; A, N = d["A"], d["N"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13})
fig = plt.figure(figsize=(16, 4.6))
for k, (v, cmap, norm, title, key) in enumerate([
        (A, "inferno", mcolors.Normalize(0, 25), "Measured dose (µSv/h)", A),
        (A - N, "RdBu_r", mcolors.TwoSlopeNorm(0, -10, 10), "NAIRAS error A − N (red = too low)", np.abs(A - N))]):
    ax = fig.add_subplot(1, 2, k + 1, projection=ccrs.PlateCarree())
    ax.set_extent([-170, 40, -5, 85], ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="#ECEEF2"); ax.add_feature(cfeature.COASTLINE, lw=.4, edgecolor="#8A8F99")
    s = np.argsort(key)
    sc = ax.scatter(lon[s], lat[s], c=v[s], cmap=cmap, norm=norm, s=1.5, transform=ccrs.PlateCarree())
    ax.set_title(title, loc="left", fontsize=15, color="#14213D", fontweight="bold")
    fig.colorbar(sc, ax=ax, shrink=.8, pad=.02)
fig.tight_layout()
out = Path(__file__).parent / "figures" / "slide_map.png"
fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight"); print("saved", out)
