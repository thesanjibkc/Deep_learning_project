"""All figures of the CVPR-format report, drawn in paper style (see src/paperstyle.py).

Usage: python3 src/paper_figures.py [name ...]      (no name = all)
names: nairas best map solar temporal training residual
Writes figures/paper/<name>.pdf. Poster/slide versions stay in figures/explore and figures/results.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paperstyle import (COL, FULL, save, BLACK, ORANGE, SKY, GREEN, BLUE, VERMILION, PURPLE,
                        GRAY, LIGHT)
from data import load, split, DATA, PARTS

ROTS = ["123", "132", "213", "231", "312", "321"]
SEEDS = [42, 1, 2]
_cache = {}


def data(cad="1H"):
    if cad not in _cache:
        _cache[cad] = load(cad)
    return _cache[cad]


def vehicles():
    if "veh" not in _cache:
        v = []
        for p in PARTS:
            for line in (DATA / "1H" / f"ARMAS_TIMESERIES_1H_20132023_{p}.json").open():
                if line.strip():
                    v.append(json.loads(line)["Vehicle_ID"])
        _cache["veh"] = np.array(v)
    return _cache["veh"]


def binned(ax, x, y, edges, color, label, min_n=30, band=True):
    mid, q = [], []
    for lo, up in zip(edges[:-1], edges[1:]):
        k = (x >= lo) & (x < up)
        if k.sum() >= min_n:
            mid.append((lo + up) / 2); q.append(np.percentile(y[k], [25, 50, 75]))
    q = np.array(q)
    if band:
        ax.fill_between(mid, q[:, 0], q[:, 2], color=color, alpha=0.2, lw=0)
    ax.plot(mid, q[:, 1], color=color, label=label)


# ---------------------------------------------------------------- Fig. 1: where NAIRAS fails
def fig_nairas():
    d = data("1H"); A, N = d["A"], d["N"]; r = A - N
    fig, ax = plt.subplots(1, 2, figsize=(COL, 1.75), gridspec_kw={"wspace": 0.8})
    a = ax[0]
    hb = a.hexbin(A, N, gridsize=40, extent=(0, 45, 0, 45), norm=LogNorm(), cmap="viridis",
                  mincnt=1, linewidths=0, rasterized=True)
    a.plot([0, 45], [0, 45], "--", color=BLACK, lw=0.7)
    a.axvline(15, color=GRAY, lw=0.6, ls=":")
    a.set(xlim=(0, 45), ylim=(0, 45), aspect="equal", xlabel="ARMAS $A$ (µSv/h)",
          ylabel="NAIRAS $N$ (µSv/h)")
    a.set_title("(a)")
    cb = fig.colorbar(hb, ax=a, pad=0.04, fraction=0.05, aspect=15)
    cb.ax.tick_params(labelsize=6)
    a = ax[1]
    edges = np.arange(0, 42, 2)
    binned(a, A, r, edges, VERMILION, "by measured $A$")
    binned(a, N, r, edges, BLUE, "by NAIRAS $N$")
    a.axhline(0, color=BLACK, lw=0.5); a.axvline(15, color=GRAY, lw=0.6, ls=":")
    a.set(xlabel="dose bin (µSv/h)", ylabel="$A-N$ (µSv/h)", ylim=(-7, 30)); a.set_title("(b)")
    a.set_box_aspect(1)
    a.legend(loc="upper left", handlelength=1.2, fontsize=6.5)
    save(fig, "nairas")


# ---------------------------------------------------------------- Fig. 2: NAIRAS vs best models
def best_predictions():
    d = data("1H"); A = d["A"]
    models = {"CNN–MLP": ("results/temporal_final/1H", "CNN-MLP_residual"),
              "Weighted Transformer": ("results/temporal_final_weighted/5M", "Transformer_residual")}
    pred = {}
    for lab, (root, stem) in models.items():
        p = np.full_like(A, np.nan)
        for r in ("123", "231", "312"):            # each partition is the test set exactly once
            te = split(d, *(f"P{c}" for c in r))["test"]
            p[te] = np.mean([np.load(ROOT / root / f"rot{r}_seed{s}" / f"pred_{stem}.npy")[te]
                             for s in SEEDS], 0)
        assert not np.isnan(p).any()
        pred[lab] = p
    return pred


def fig_best():
    d = data("1H"); A, N = d["A"], d["N"]; hi = A > 15
    pred = best_predictions()
    series = [("NAIRAS V3", N, BLACK)] + [(k, v, c) for (k, v), c in zip(pred.items(), (BLUE, VERMILION))]
    rm = lambda p, k=slice(None): float(np.sqrt(np.mean((p[k] - A[k]) ** 2)))
    fig, axs = plt.subplots(1, 6, figsize=(FULL, 1.5),
                            gridspec_kw={"width_ratios": [1, 1, 1, 0.05, 0.45, 1.25], "wspace": 0.3})
    axs[4].axis("off")
    for i, (lab, p, _) in enumerate(series):
        a = axs[i]
        hb = a.hexbin(A, p, gridsize=45, extent=(0, 45, 0, 45), norm=LogNorm(vmin=1, vmax=2000),
                      cmap="viridis", mincnt=1, linewidths=0, rasterized=True)
        a.plot([0, 45], [0, 45], "--", color=BLACK, lw=0.7)
        a.axvline(15, color=GRAY, lw=0.6, ls=":")
        a.set(xlim=(0, 45), ylim=(0, 45), aspect="equal", xlabel="ARMAS $A$ (µSv/h)",
              ylabel="predicted dose (µSv/h)" if i == 0 else "")
        a.set_title(f"({'abc'[i]})")
        a.text(0.04, 0.97, f"{lab}\nRMSE {rm(p):.2f}\n$A>15$: {rm(p, hi):.2f}", transform=a.transAxes,
               va="top", fontsize=6.5)
    cb = fig.colorbar(hb, cax=axs[3]); cb.set_label("observations per cell", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    a = axs[5]
    for lab, p, c in series:
        binned(a, A, A - p, np.arange(0, 42, 2), c, lab)
    a.axhline(0, color=BLACK, lw=0.5); a.axvline(15, color=GRAY, lw=0.6, ls=":")
    a.set(xlabel="ARMAS $A$ (µSv/h)", ylabel=r"$A-\hat{A}$ (µSv/h)"); a.set_title("(d)")
    a.set_box_aspect(0.85)
    a.set_ylim(-7, 38); a.legend(loc="upper left", handlelength=1.2, fontsize=6.5)
    save(fig, "best")


# ---------------------------------------------------------------- appendix: flight map
def fig_map():
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    d = data("1H"); A, N = d["A"], d["N"]; st = d["static"]
    lat, lon, rc = st[:, 0], st[:, 1], st[:, 6]
    panels = [(A, "(a) ARMAS dose rate (µSv/h)", "inferno", Normalize(0, 25), A),
              (rc, "(b) Cutoff rigidity $R_c$ (GV)", "viridis_r", Normalize(0, 15), -rc),
              (A - N, "(c) NAIRAS error $A-N$ (µSv/h)", "RdBu_r", TwoSlopeNorm(0, -10, 10), np.abs(A - N))]
    fig = plt.figure(figsize=(FULL, 6.3))
    for i, (v, title, cmap, norm, key) in enumerate(panels):
        ax = fig.add_subplot(3, 1, i + 1, projection=ccrs.PlateCarree())
        ax.set_extent([-180, 180, -10, 85], ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="#EEEEEE")
        ax.add_feature(cfeature.COASTLINE, lw=0.3, edgecolor=GRAY)
        g = ax.gridlines(draw_labels=True, lw=0.25, color=GRAY, alpha=0.5,
                         xlocs=range(-180, 181, 60), ylocs=range(0, 91, 30))
        g.top_labels = g.right_labels = False
        g.xlabel_style = g.ylabel_style = {"size": 6}
        s = np.argsort(key)
        sc = ax.scatter(lon[s], lat[s], c=v[s], cmap=cmap, norm=norm, s=0.25, lw=0,
                        transform=ccrs.PlateCarree(), rasterized=True)
        ax.set_title(title, loc="left")
        cb = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.015); cb.ax.tick_params(labelsize=6)
    save(fig, "map")


# ---------------------------------------------------------------- appendix: solar cycle
def fig_solar():
    d = data("1H"); A, N = d["A"], d["N"]; env = list(d["env"]); st = d["static"]
    cur = d["temporal"][:, -1, :]
    df = pd.DataFrame({"day": pd.to_datetime(pd.Series(d["time"])).dt.floor("D"), "A": A, "N": N,
                       "rc": st[:, 6], "alt": st[:, 3] / 1000,
                       "ss": cur[:, env.index("Solar_sunspots")], "oulu": cur[:, env.index("NM_OULU")]})
    df["err"] = df["A"] - df["N"]
    daily = df.groupby("day").agg(ss=("ss", "mean"), oulu=("oulu", "mean"), err=("err", "mean"))
    c = df[(df["rc"] < 1) & df["alt"].between(10.5, 12)]
    cd = c.groupby("day").agg(A=("A", "median"), N=("N", "median"), n=("A", "size"))
    cd = cd[cd["n"] >= 10]
    fig, ax = plt.subplots(4, 1, figsize=(FULL, 4.6), sharex=True, gridspec_kw={"hspace": 0.35})
    ax[0].scatter(daily.index, daily["ss"], s=3, color=ORANGE, lw=0)
    ax[0].set(ylabel="sunspot\nnumber"); ax[0].set_title("(a)")
    ax[1].scatter(daily.index, daily["oulu"], s=3, color=GREEN, lw=0)
    ax[1].set(ylabel="Oulu NM\n(counts/s)"); ax[1].set_title("(b)")
    ax[2].scatter(cd.index, cd["A"], s=5, color=VERMILION, lw=0, label="ARMAS")
    ax[2].scatter(cd.index, cd["N"], s=6, color=BLUE, marker="x", lw=0.6, label="NAIRAS V3")
    ax[2].set(ylabel="median dose\n(µSv/h)"); ax[2].set_title("(c)")
    ax[2].legend(loc="upper left", ncol=2, markerscale=1.5)
    ax[3].bar(daily.index, daily["err"], width=6, lw=0,
              color=np.where(daily["err"] > 0, VERMILION, BLUE))
    ax[3].axhline(0, color=BLACK, lw=0.5)
    ax[3].set(ylabel="mean $A-N$\n(µSv/h)", xlabel="date"); ax[3].set_title("(d)")
    for a in ax:
        a.yaxis.set_label_coords(-0.06, 0.5)
    save(fig, "solar")


# ---------------------------------------------------------------- appendix: history information
def fig_temporal():
    d5, d1 = data("5M"), data("1H"); env = list(d1["env"])
    X5, X1 = d5["temporal"], d1["temporal"]
    r = d1["A"] - d1["N"]
    sd = X1.reshape(-1, 35).std(0) + 1e-12
    move5 = np.median((X5.max(1) - X5.min(1)) / sd, 0)
    move1 = np.median((X1.max(1) - X1.min(1)) / sd, 0)
    rank = lambda a: pd.DataFrame(a).rank().values
    rr = rank(r[:, None])[:, 0]; rr = (rr - rr.mean()) / rr.std()

    def lagcorr(X):
        out = np.zeros((35, X.shape[1]))
        for t in range(X.shape[1]):
            z = rank(X[:, t, :]); z = (z - z.mean(0)) / (z.std(0) + 1e-12)
            out[:, t] = (z * rr[:, None]).mean(0)
        return out[:, ::-1]                       # column 0 = observation time

    C5, C1 = lagcorr(X5), lagcorr(X1)
    order = np.argsort(-move1); y = np.arange(35); names = [env[i] for i in order]
    fig = plt.figure(figsize=(FULL, 4.4))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 0.75, 1.3, 0.04], wspace=0.12)
    a = fig.add_subplot(gs[0])
    a.barh(y - 0.2, move5[order], 0.4, color=ORANGE, label="1-hour window")
    a.barh(y + 0.2, move1[order], 0.4, color=BLUE, label="24-hour window")
    a.set_yticks(y); a.set_yticklabels(names, fontsize=5.5); a.invert_yaxis()
    a.set_xlabel("median change in window\n(range / std)"); a.set_title("(a)")
    a.legend(loc="lower right")
    for k, (C, T, step, xl) in enumerate([(C5, 12, 5, "minutes before"), (C1, 24, 1, "hours before")]):
        a = fig.add_subplot(gs[k + 1])
        im = a.imshow(C[order], aspect="auto", cmap="RdBu_r", vmin=-0.3, vmax=0.3)
        ticks = list(range(0, T, 3 if T == 12 else 6))
        a.set_xticks(ticks); a.set_xticklabels([t * step for t in ticks])
        a.set_yticks(y); a.set_yticklabels([])
        a.set_xlabel(xl); a.set_title(f"({'bc'[k]})")
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[3])); cb.set_label("Spearman correlation with $A-N$")
    cb.ax.tick_params(labelsize=6)
    save(fig, "temporal")


# ---------------------------------------------------------------- appendix: training dynamics
def fig_training():
    z = np.load(ROOT / "results/tracking/tracking.npz")
    tc, fc = z["trf_curve"], z["flat_curve"]; best = int(tc[:, 1].argmin())
    hb = np.arange(23, -1, -1)
    d = data("1H"); t = pd.to_datetime(pd.Series(d["time"]))
    i = int(np.where(t.dt.strftime("%Y-%m-%d %H:%M") == "2017-09-09 06:24")[0][0])
    kp = d["temporal"][i, :, list(d["env"]).index("Index_Kp")] / 10
    att = z["trf_att_storm"]
    wm = z["flat_wmap"].sum(2); wm = wm / wm.sum(1, keepdims=True)
    fig, ax = plt.subplots(2, 2, figsize=(FULL, 3.7), gridspec_kw={"hspace": 0.55, "wspace": 0.3})
    a = ax[0, 0]; ep = np.arange(len(tc))
    a.plot(ep, fc[:, 0], "--", color=VERMILION, label="Flat MLP, train")
    a.plot(ep, fc[:, 1], color=VERMILION, label="Flat MLP, validation")
    a.plot(ep, tc[:, 0], "--", color=BLUE, label="Transformer, train")
    a.plot(ep, tc[:, 1], color=BLUE, label="Transformer, validation")
    a.axvline(best, color=GRAY, ls=":", lw=0.7)
    a.set(xlabel="epoch", ylabel="RMSE (µSv/h)", ylim=(1.6, 4.05)); a.set_title("(a)")
    a.legend(loc="lower center", ncol=2)
    for a, M, t_ in ((ax[0, 1], att, "(b)"), (ax[1, 1], wm, "(d)")):
        im = a.imshow(M, aspect="auto", cmap="cividis", origin="lower", vmin=att.min(), vmax=att.max(),
                      extent=(-0.5, 23.5, -0.5, len(M) - 0.5))
        a.set_xticks(range(0, 24, 3)); a.set_xticklabels(hb[::3])
        a.set(xlabel="hours before the flight", ylabel="epoch"); a.set_title(t_)
        cb = fig.colorbar(im, ax=a, pad=0.02); cb.ax.tick_params(labelsize=6)
        cb.set_label("attention" if t_ == "(b)" else "weight share", fontsize=7)
    ax[0, 1].axhline(best, color="white", lw=0.6, ls=":")
    a = ax[1, 0]
    a.bar(range(24), att[best], color=BLUE, label=f"attention, epoch {best}")
    a.axhline(1 / 24, color=GRAY, ls="--", lw=0.6)
    a2 = a.twinx(); a2.step(range(24), kp, where="mid", color=VERMILION, lw=1.0, label="Kp")
    a2.set_ylabel("Kp"); a2.spines["top"].set_visible(False)
    a.set_xticks(range(0, 24, 3)); a.set_xticklabels(hb[::3])
    a.set(xlabel="hours before the flight", ylabel="attention weight"); a.set_title("(c)")
    h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, loc="upper right")
    save(fig, "training")


# ---------------------------------------------------------------- appendix: what drives the residual
def eta2(groups, e):
    s = pd.Series(e).groupby(groups)
    between = (s.size() * (s.mean() - e.mean()) ** 2).sum()
    return float(between / ((e - e.mean()) ** 2).sum())


def residual_shares(cad="1H"):
    """Per assignment (residual Transformer, seed 42; cad 1H = 24-hour, 5M = 1-hour history): share
    of error variance explained by each grouping, for the NAIRAS error and the error after the model."""
    d = data(cad); A, N = d["A"], d["N"]; st = d["static"]
    veh = vehicles(); t = pd.to_datetime(pd.Series(d["time"]))
    day = t.dt.strftime("%Y-%m-%d").values
    flight = pd.Series([f"{v}|{x}" for v, x in zip(veh, day)]).values
    kp24 = data("1H")["temporal"][:, :, list(d["env"]).index("Index_Kp")].max(1) / 10   # max Kp over 24 h
    G = {"Aircraft": veh, "Year": t.dt.year.values.astype(str),
         "Altitude band": pd.cut(st[:, 3] / 1000, [0, 10, 11, 12, 20]).astype(str),
         "Shielding band ($R_c$)": pd.cut(st[:, 6], [-1, 1, 3, 6, 10, 20]).astype(str),
         "Storm (max Kp)": pd.cut(kp24, [-1, 3, 5, 9.5]).astype(str)}
    rng = np.random.default_rng(0); rows = []
    for r in ROTS:
        te = split(d, *(f"P{c}" for c in r))["test"]
        pred = np.load(ROOT / f"results/temporal_final/{cad}/rot{r}_seed42/pred_Transformer_residual.npy")
        for kind, e in (("NAIRAS", (A - N)[te]), ("model", (A - pred)[te])):
            row = {"rot": r, "kind": kind}
            for g, lab in G.items():
                row[g] = eta2(np.asarray(lab)[te], e)
            fl = flight[te]; row["Flight"] = eta2(fl, e)
            # chance level for aircraft: reassign aircraft labels across flights at random
            f_codes, f_idx = np.unique(fl, return_inverse=True)
            f_air = np.array([x.split("|")[0] for x in f_codes])
            row["Aircraft chance"] = np.mean([eta2(rng.permutation(f_air)[f_idx], e) for _ in range(20)])
            rows.append(row)
    return pd.DataFrame(rows)


def fig_residual(cad="1H"):
    imp = pd.concat([pd.read_csv(f) for f in sorted((ROOT / "results/explain").glob(f"importance_groups_{cad}_rot*_seed42.csv"))])
    imp = imp.groupby(["group", "kind"])["d_all"].agg(["mean", "min", "max"]).reset_index().sort_values("mean")
    sh = residual_shares(cad)
    sh.to_csv(ROOT / f"results/explain/residual_shares_{cad}_seed42.csv", index=False)
    m = sh.groupby("kind").mean(numeric_only=True); lo = sh.groupby("kind").min(numeric_only=True)
    hi_ = sh.groupby("kind").max(numeric_only=True)
    print(m.round(3).T.to_string())

    fig, ax = plt.subplots(1, 3, figsize=(FULL, 2.15),
                           gridspec_kw={"width_ratios": [1.15, 1.0, 0.5], "wspace": 1.0})
    a = ax[0]
    short = {"Geomagnetic shielding (Rc, L, geomag lat/lon)": "Geomagnetic shielding",
             "Solar activity (sunspots, F10.7, polar field)": "Solar activity",
             "Geomagnetic indices (Kp, ap, Dst)": "Kp, ap, Dst", "NAIRAS estimate (as input)": "NAIRAS estimate",
             "Position (lat, lon)": "Latitude, longitude"}
    y = np.arange(len(imp))
    col = [BLUE if k == "flight (current)" else ORANGE for k in imp["kind"]]
    a.barh(y, imp["mean"], color=col, xerr=[imp["mean"] - imp["min"], imp["max"] - imp["mean"]],
           error_kw={"lw": 0.5, "capsize": 1})
    a.set_yticks(y); a.set_yticklabels([short.get(g, g) for g in imp["group"]])
    a.set_xlabel("RMSE increase when shuffled (µSv/h)"); a.set_title("(a)")
    a.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)],
             labels=["flight (current)", "space weather (history)"], loc="upper center",
             bbox_to_anchor=(0.35, -0.24), ncol=2)

    a = ax[1]
    groups = ["Aircraft", "Year", "Altitude band", "Shielding band ($R_c$)", "Storm (max Kp)"][::-1]
    y = np.arange(len(groups))
    for off, kind, c, lab in ((0.2, "NAIRAS", GRAY, "NAIRAS error $A-N$"), (-0.2, "model", VERMILION, r"error after the model $A-\hat{A}$")):
        v = m.loc[kind, groups] * 100
        a.barh(y + off, v, 0.38, color=c, label=lab,
               xerr=[v - lo.loc[kind, groups] * 100, hi_.loc[kind, groups] * 100 - v], error_kw={"lw": 0.5, "capsize": 1})
    a.axvline(m.loc["model", "Aircraft chance"] * 100, color=BLACK, ls=":", lw=0.7)
    a.set_yticks(y); a.set_yticklabels(groups)
    a.set_xlabel("error variance explained (%)"); a.set_title("(b)")
    a.legend(loc="upper center", bbox_to_anchor=(0.4, -0.24), ncol=1)

    a = ax[2]
    for x, kind in enumerate(("NAIRAS", "model")):
        fl, air = m.loc[kind, "Flight"], m.loc[kind, "Aircraft"]
        parts = [(1 - fl, LIGHT, "within a flight"), (fl - air, GRAY, "between flights"),
                 (air, VERMILION, "aircraft")]
        bot = 0
        for v, c, lab in parts:
            a.bar(x, v * 100, bottom=bot * 100, color=c, width=0.6, label=lab if x == 0 else None)
            a.text(x, (bot + v / 2) * 100, f"{v * 100:.0f}", ha="center", va="center", fontsize=6.5,
                   color="white" if c == VERMILION else BLACK)
            bot += v
    a.set_xticks([0, 1]); a.set_xticklabels(["$A-N$", r"$A-\hat{A}$"])
    a.set(ylabel="error variance (%)", ylim=(0, 100)); a.set_title("(c)")
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1)
    save(fig, "residual" if cad == "1H" else "residual_1h")


# ---------------------------------------------------------------- appendix: which feature, which hour, whose history
NICE = {"Altitude(GPS)": "GPS altitude", "Altitude(Bar)": "Barometric altitude", "Geomagnetic_Rc": "Cutoff rigidity $R_c$",
        "Geomagnetic_latitude": "Geomagnetic latitude", "Geomagnetic_longitude": "Geomagnetic longitude",
        "Geomagnetic_Lshell": "L-shell", "NAIRAS": "NAIRAS estimate", "Latitude": "Latitude", "Longitude": "Longitude",
        "NM_NEWK": "NM Newark", "NM_SOPO": "NM South Pole", "NM_OULU": "NM Oulu", "NM_THUL": "NM Thule",
        "Solar_SPF20": "South polar field (filt.)", "Particles_P100": "Protons $>$100 MeV", "Particles_P1": "Protons $>$1 MeV",
        "SW_V": "Solar-wind speed", "Solar_sunspots": "Sunspot number", "SW_density": "Solar-wind density",
        "SW_B": "IMF $|B|$", "SXR_short": "X-ray (short)", "SXR_long": "X-ray (long)", "Index_Kp": "Kp", "Index_Dst": "Dst",
        "Index_Ap": "ap", "Solar_SPF": "South polar field", "Solar_NPF": "North polar field", "Solar_APF": "Mean polar field",
        "Solar_NPF20": "North polar field (filt.)", "Solar_APF20": "Mean polar field (filt.)", "Solar_f107": "F10.7",
        "Particles_P5": "Protons $>$5 MeV", "Particles_P10": "Protons $>$10 MeV", "Particles_P30": "Protons $>$30 MeV",
        "Particles_P50": "Protons $>$50 MeV", "Particles_E20": "Electrons", "SW_Bx": "IMF $B_x$", "SW_By": "IMF $B_y$",
        "SW_Bz": "IMF $B_z$", "SW_Vx": "Solar-wind $V_x$", "SW_Vy": "Solar-wind $V_y$", "SW_Vz": "Solar-wind $V_z$",
        "SW_temperature": "Solar-wind temperature", "SW_pressure": "Solar-wind pressure"}


def fig_features(cad="1H"):
    ex = ROOT / "results/explain"
    v = pd.concat([pd.read_csv(f) for f in sorted(ex.glob(f"importance_vars_{cad}_rot*_seed42.csv"))])
    v = v.groupby(["variable", "kind"])["d_all"].agg(["mean", "min", "max"]).reset_index()
    v = v.sort_values("mean", ascending=False).head(15).iloc[::-1]
    h = pd.concat([pd.read_csv(f) for f in sorted(ex.glob(f"importance_hours_{cad}_rot*_seed42.csv"))])
    h = h.groupby("steps_before").agg(["mean", "min", "max"])
    hz = pd.concat([pd.read_csv(f) for f in sorted(ex.glob(f"history_importance_{cad}_rot*_seed42.csv"))])
    hz = hz.groupby("group", sort=False)[["freeze_all", "shuffle_all"]].agg(["mean", "min", "max"])
    fig, ax = plt.subplots(1, 3, figsize=(FULL, 2.4), gridspec_kw={"width_ratios": [1.1, 0.85, 1.05], "wspace": 0.75})
    a = ax[0]; y = np.arange(len(v))
    a.barh(y, v["mean"], color=[BLUE if k == "flight (current)" else ORANGE for k in v["kind"]],
           xerr=[v["mean"] - v["min"], v["max"] - v["mean"]], error_kw={"lw": 0.5, "capsize": 1})
    a.set_yticks(y); a.set_yticklabels([NICE.get(s, s) for s in v["variable"]], fontsize=6.5)
    a.set_xlabel("RMSE increase when shuffled (µSv/h)"); a.set_title("(a)")
    a.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)],
             labels=["flight (current)", "space weather (" + ("24 h" if cad == "1H" else "1 h") + ")"], loc="upper center",
             bbox_to_anchor=(0.3, -0.25), ncol=1)
    a = ax[1]; x = h.index.values * (1 if cad == "1H" else 5)
    for col, c, lab in (("d_all", BLUE, "all observations"), ("d_high", VERMILION, "$A>15$")):
        a.fill_between(x, h[(col, "min")], h[(col, "max")], color=c, alpha=0.2, lw=0)
        a.plot(x, h[(col, "mean")], color=c, label=lab)
    a.axhline(0, color=BLACK, lw=0.5); a.invert_xaxis()
    a.set(xlabel=("hours" if cad == "1H" else "minutes") + " before the observation", ylabel="RMSE increase (µSv/h)"); a.set_title("(b)")
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=1)
    a = ax[2]
    short = {"Geomagnetic indices (Kp, ap, Dst)": "Kp, ap, Dst", "All 35 variables": "All 35 variables"}
    g = [s for s in hz.index]; y = np.arange(len(g))[::-1]
    for off, col, c, lab in ((0.2, "freeze_all", BLUE, "flatten past to current value"), (-0.2, "shuffle_all", ORANGE, "swap past with another reading")):
        m_ = hz[(col, "mean")].values
        a.barh(y + off, m_, 0.38, color=c, label=lab,
               xerr=[m_ - hz[(col, "min")].values, hz[(col, "max")].values - m_], error_kw={"lw": 0.5, "capsize": 1})
    a.axvline(0, color=BLACK, lw=0.5)
    a.set_yticks(y); a.set_yticklabels([short.get(s, s) for s in g], fontsize=6.5)
    a.set_xlabel("RMSE increase (µSv/h)"); a.set_title("(c)")
    a.legend(loc="upper center", bbox_to_anchor=(0.3, -0.25), ncol=1)
    save(fig, "features" if cad == "1H" else "features_1h")


# ---------------------------------------------------------------- appendix: occlusion (as in the preliminary report)
def fig_occlusion():
    """Replace an input by its training mean (0 in standardized units) and measure the change in test RMSE
    of the residual Transformer; mean over the six assignments (seed 42). Positive = worse without it."""
    ex = ROOT / "results/explain"
    rd = lambda pat: pd.concat([pd.read_csv(f) for f in sorted(ex.glob(pat))])
    v24 = rd("occlusion_vars_1H_rot*_seed42.csv").groupby(["variable", "kind"])["d_all"].mean()
    v1 = rd("occlusion_vars_5M_rot*_seed42.csv").groupby(["variable", "kind"])["d_all"].mean()
    top = (v24 + v1).sort_values(ascending=False).head(12).index[::-1]
    l24 = rd("occlusion_lags_1H_rot*_seed42.csv").groupby("steps_before")[["d_all", "d_high"]].agg(["mean", "min", "max"])
    l1 = rd("occlusion_lags_5M_rot*_seed42.csv").groupby("steps_before")[["d_all", "d_high"]].agg(["mean", "min", "max"])
    c24 = rd("occlusion_cells_1H_rot*_seed42.csv").groupby(["variable", "steps_before"])["d_all"].mean().unstack()
    rows = c24.abs().sum(1).sort_values(ascending=False).head(12).index
    c24 = c24.loc[rows, sorted(c24.columns, reverse=True)]                 # columns: 23 h before ... 0

    fig = plt.figure(figsize=(FULL, 4.3))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.95], width_ratios=[1.25, 1, 1], hspace=0.75, wspace=0.55)
    a = fig.add_subplot(gs[0, 0]); y = np.arange(len(top))
    a.barh(y + 0.2, [v24[i] for i in top], 0.38, color=BLUE, label="24-hour model")
    a.barh(y - 0.2, [v1.get(i, np.nan) for i in top], 0.38, color=ORANGE, label="1-hour model")
    a.axvline(0, color=BLACK, lw=0.5)
    a.set_yticks(y); a.set_yticklabels([NICE.get(i[0], i[0]) for i in top], fontsize=6.5)
    a.set_xlabel("RMSE change when occluded (µSv/h)"); a.set_title("(a) variables")
    a.legend(loc="lower right")
    for k, (L, unit, step, title) in enumerate(((l24, "hours", 1, "(b) time steps, 24-hour model"),
                                                (l1, "minutes", 5, "(c) time steps, 1-hour model"))):
        a = fig.add_subplot(gs[0, k + 1]); x = L.index.values * step
        for col, c, lab in (("d_all", BLUE, "all"), ("d_high", VERMILION, "$A>15$")):
            a.fill_between(x, L[(col, "min")], L[(col, "max")], color=c, alpha=0.18, lw=0)
            a.plot(x, L[(col, "mean")], color=c, label=lab)
        a.axhline(0, color=BLACK, lw=0.5); a.invert_xaxis()
        a.set(xlabel=f"{unit} before the observation", ylabel="RMSE change (µSv/h)"); a.set_title(title)
        a.legend(loc="upper left", fontsize=6.5)
    a = fig.add_subplot(gs[1, :])
    vm = np.nanmax(np.abs(c24.values))
    im = a.imshow(c24.values, aspect="auto", cmap="RdBu_r", vmin=-vm, vmax=vm)
    a.set_yticks(range(len(rows))); a.set_yticklabels([NICE.get(r, r) for r in rows], fontsize=6.5)
    a.set_xticks(range(0, 24, 2)); a.set_xticklabels([int(c) for c in c24.columns[::2]])
    a.set_xlabel("hours before the observation"); a.set_title("(d) one variable at one hour, 24-hour model")
    cb = fig.colorbar(im, ax=a, pad=0.01, fraction=0.03); cb.set_label("RMSE change (µSv/h)", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    save(fig, "occlusion")


# ---------------------------------------------------------------- appendix: learning curves of all models
MCOL = {"MLP, direct": (GRAY, "--"), "MLP": (BLACK, "-"), "Flat MLP": (VERMILION, "-"), "CNN–MLP": (GREEN, "-"),
        "TCN": (SKY, "-"), "Transformer, direct": (BLUE, "--"), "Transformer": (BLUE, "-"), "Transformer, $w=4$": (PURPLE, "-")}


def histories(rot="123", seed=42):
    """Per-epoch train/validation RMSE of every model for one assignment, by history length."""
    rd = lambda p: json.load(open(ROOT / p))
    basic = rd(f"results/basic/rot{rot}_seed{seed}/histories.json")
    out = {}
    for cad in ("5M", "1H"):
        tmp = rd(f"results/temporal_final/{cad}/rot{rot}_seed{seed}/histories.json")
        w = rd(f"results/temporal_final_weighted/{cad}/rot{rot}_seed{seed}/histories.json")
        out[cad] = {"MLP, direct": basic["MLP (direct)"], "MLP": basic["MLP (residual)"],
                    "Flat MLP": tmp["FlatMLP (residual)"], "CNN–MLP": tmp["CNN-MLP (residual)"], "TCN": tmp["TCN (residual)"],
                    "Transformer, direct": tmp["Transformer (direct)"], "Transformer": tmp["Transformer (residual)"],
                    "Transformer, $w=4$": w["Transformer (residual)"]}
    return out


def best_epochs():
    rows = []
    for cad, hist in (("5M", "1 h"), ("1H", "24 h")):
        for r in ROTS:
            for s in SEEDS:
                for k, v in histories(r, s)[cad].items():
                    rows.append({"model": k, "history": hist, "best": int(np.argmin([e["val_rmse"] for e in v]) + 1)})
    return pd.DataFrame(rows)


def fig_curves():
    H = histories()
    fig, ax = plt.subplots(1, 3, figsize=(FULL, 2.5), gridspec_kw={"width_ratios": [1, 1, 0.95], "wspace": 0.38})
    for a, cad, title in ((ax[0], "5M", "(a) validation RMSE, 1-hour history"), (ax[1], "1H", "(b) validation RMSE, 24-hour history")):
        for k, v in H[cad].items():
            c, ls = MCOL[k]; ep = [e["epoch"] for e in v]; vr = [e["val_rmse"] for e in v]; b = int(np.argmin(vr))
            a.plot(ep, vr, ls, color=c, lw=0.9, label=k)
            a.plot(ep[b], vr[b], "o", color=c, ms=3)
        a.set(xlabel="epoch", ylabel="RMSE (µSv/h)", ylim=(3.55, 4.25)); a.set_title(title)
    ax[0].legend(loc="upper right", fontsize=6, ncol=1, handlelength=1.6)
    be = best_epochs(); order = list(MCOL)
    a = ax[2]
    for i, k in enumerate(order):
        for off, hist, mk in ((-0.15, "1 h", "o"), (0.15, "24 h", "s")):
            v = be[(be.model == k) & (be.history == hist)]["best"].values
            a.scatter(np.full(len(v), i + off) + np.random.default_rng(i).uniform(-0.05, 0.05, len(v)), v,
                      s=5, marker=mk, color=MCOL[k][0], alpha=0.7, lw=0)
            a.plot([i + off - 0.12, i + off + 0.12], [np.median(v)] * 2, color=BLACK, lw=0.8)
    a.set_xticks(range(len(order))); a.set_xticklabels([k.replace("Transformer", "Trf.").replace(", direct", " dir.") for k in order],
                                                       rotation=60, ha="right", fontsize=6)
    a.set_ylabel("best epoch"); a.set_title("(c) best epoch, 18 runs (o 1 h, □ 24 h)")
    save(fig, "curves")


# ---------------------------------------------------------------- appendix: occlusion for every model
def fig_occlusion_models():
    ex = ROOT / "results/explain"
    rd = lambda pat: pd.concat([pd.read_csv(f) for f in sorted(ex.glob(pat))])
    v24 = rd("occl_all_vars_1H_rot*_seed42.csv"); v1 = rd("occl_all_vars_5M_rot*_seed42.csv")
    l24 = rd("occl_all_lags_1H_rot*_seed42.csv"); l1 = rd("occl_all_lags_5M_rot*_seed42.csv")
    models = ["MLP", "Flat MLP", "CNN–MLP", "TCN", "Transformer"]
    M24 = v24.groupby(["variable", "model"])["d_all"].mean().unstack()[models]
    top = M24.max(1).sort_values(ascending=False).head(14).index
    M24 = M24.loc[top]
    fig = plt.figure(figsize=(FULL, 3.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1, 1], wspace=0.75)
    a = fig.add_subplot(gs[0])
    vm = np.nanmax(np.abs(M24.values))
    im = a.imshow(M24.values, aspect="auto", cmap="RdBu_r", vmin=-vm, vmax=vm)
    for (i, j), v in np.ndenumerate(M24.values):
        if not np.isnan(v):
            a.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5, color="white" if abs(v) > 0.5 * vm else BLACK)
    a.set_xticks(range(len(models))); a.set_xticklabels(models, rotation=35, ha="right", fontsize=6.5)
    a.set_yticks(range(len(top))); a.set_yticklabels([NICE.get(t_, t_) for t_ in top], fontsize=6.5)
    a.set_title("(a) variable occlusion, 24 h (MLP: no history)")
    cb = fig.colorbar(im, ax=a, pad=0.02, fraction=0.05); cb.ax.tick_params(labelsize=6)
    cb.ax.set_title("$\\Delta$RMSE", fontsize=6.5, pad=3)   # title, not side label: a side label collided with panel (b)
    for k, (L, unit, step, title) in enumerate(((l24, "hours", 1, "(b) time-step occlusion, 24 h"),
                                                (l1, "minutes", 5, "(c) time-step occlusion, 1 h"))):
        a = fig.add_subplot(gs[k + 1])
        g = L.groupby(["model", "steps_before"])["d_all"].mean()
        for mdl in models[1:]:
            s = g.loc[mdl]; c, ls = MCOL[mdl]
            a.plot(s.index.values * step, s.values, ls, color=c, label=mdl)
        a.axhline(0, color=BLACK, lw=0.5); a.invert_xaxis()
        a.set(xlabel=f"{unit} before the observation", ylabel="RMSE change (µSv/h)"); a.set_title(title)
        if k == 0: a.legend(loc="upper left", fontsize=6)
    save(fig, "occlusion_models")


FIGS = {"nairas": fig_nairas, "best": fig_best, "map": fig_map, "solar": fig_solar,
        "temporal": fig_temporal, "training": fig_training, "residual": fig_residual, "features": fig_features,
        "residual_1h": lambda: fig_residual("5M"), "features_1h": lambda: fig_features("5M"),
        "occlusion": fig_occlusion,
        "curves": fig_curves, "occlusion_models": fig_occlusion_models}

if __name__ == "__main__":
    for n in (sys.argv[1:] or FIGS):
        FIGS[n]()
