"""Exploration 3: does the Sun's condition change the dose?

One point per flight-day (all partitions). To remove the location effect we
compare like with like: only weak-shield observations (Rc < 1 GV) at cruise
altitude (10.5-12 km) in panel (c).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from load import load, FIGS

df = pd.concat([load("5M", p)[0] for p in ("P1", "P2", "P3")], ignore_index=True)
df["day"] = df["Datetime"].dt.floor("D")
alt = df["Altitude(GPS)"] / 1000
ctrl = df[(df["Geomagnetic_Rc"] < 1) & alt.between(10.5, 12)]

daily = df.groupby("day").agg(sunspots=("Solar_sunspots", "mean"), f107=("Solar_f107", "mean"),
                              oulu=("NM_OULU", "mean"), err=("residual", "mean"),
                              kp=("Index_Kp", "max"))
cd = ctrl.groupby("day").agg(A=("ARMAS", "median"), N=("NAIRASV3", "median"),
                             oulu=("NM_OULU", "mean"), n=("ARMAS", "size"))
cd = cd[cd["n"] >= 10]

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(4, 1, figsize=(11, 12), sharex=True)

a = ax[0]
a.scatter(daily.index, daily["sunspots"], s=8, color="orange")
a.set(ylabel="sunspot number", title="(a) Solar activity: the 11-year cycle (max 2014 → min 2019 → rising)")

a = ax[1]
a.scatter(daily.index, daily["oulu"], s=8, color="tab:green")
a.set(ylabel="Oulu neutron monitor\n(counts/s)",
      title="(b) Cosmic rays reaching Earth (neutron monitor) — moves OPPOSITE to the Sun")

a = ax[2]
a.scatter(cd.index, cd["A"], s=14, color="tab:red", label="ARMAS (measured)")
a.scatter(cd.index, cd["N"], s=14, color="tab:blue", marker="x", label="NAIRAS V3 (model)")
a.set(ylabel="dose rate (µSv/h)",
      title="(c) Same place, same altitude (Rc < 1 GV, 10.5–12 km): daily median dose")
a.legend(loc="upper left")

a = ax[3]
c = np.where(daily["err"] > 0, "tab:red", "tab:blue")
a.bar(daily.index, daily["err"], width=6, color=c)
a.axhline(0, color="k", lw=.8)
a.set(ylabel="mean error A − N\n(µSv/h)",
      title="(d) NAIRAS error per flight-day: red = too low, blue = too high")

fig.suptitle("Exploration 3 — The Sun's condition over 2013–2023 (one point per flight-day)",
             fontweight="bold")
fig.tight_layout()
out = FIGS / "e03_solar_cycle.png"
fig.savefig(out, dpi=140)
print("saved", out)

print("flight-days:", len(daily), " controlled days:", len(cd))
print(f"corr(sunspots, oulu) daily = {daily['sunspots'].corr(daily['oulu']):.2f}")
print(f"corr(oulu, ARMAS) controlled = {cd['oulu'].corr(cd['A']):.2f}")
print(f"corr(oulu, NAIRAS) controlled = {cd['oulu'].corr(cd['N']):.2f}")
yr = ctrl.groupby(ctrl["Datetime"].dt.year).agg(A=("ARMAS", "median"), N=("NAIRASV3", "median"),
                                                 oulu=("NM_OULU", "mean"), n=("ARMAS", "size"))
print(yr.round(2))
