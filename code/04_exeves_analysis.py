"""
ExEvEs + peak-precipitation analysis.

Two scenarios:
  GLEAM + MSWEP, 2001–2020
  FLUXCOM + MSWEP, 2001–2020

Four output figures (one per plot type, all scenarios overlaid/side-by-side):
  exeves_q95_density.png   – density of % q95 days during ExEvEs per cell
  exeves_q95_map.png       – same as spatial map
  exeves_pct_density.png   – density of % ExEvE days out of all days per cell
  exeves_lag.png           – standardised timing of q95 relative to ExEvEs
"""
from _paths import input_path, output_path
from _alignment import validate_alignment
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import warnings; warnings.filterwarnings('ignore')

GLEAM_FILE   = "gleam_e_mm_med_198001_202412_025_daily.nc"
FLUXCOM_FILE = "ET_2001_2020_025_daily_med.nc"
MSWEP_FILE   = "mswep-v2-8_tp_mm_med_197901_202012_025_daily.nc"

LAGS       = [-3, -2, -1, 0, 1, 2, 3]
LAG_LABELS = ["-3", "-2", "-1", "ExEvEs", "+1", "+2", "+3"]
SC_LABELS  = ["GLEAM\n2001–2020", "FLUXCOM\n2001–2020"]
SC_COLORS  = ["seagreen", "darkorange"]


# ── Algorithm functions ───────────────────────────────────────────────────────

def pentad_zscore(et, times_pd):
    """Pentad-normalised z-score. et: (T, nlat, nlon) float32."""
    T, nlat, nlon = et.shape
    pentad = np.minimum((times_pd.dayofyear.values - 1) // 5, 72)
    z = np.full_like(et, np.nan, dtype=np.float32)
    for p in range(73):
        idx = pentad == p
        if not idx.any():
            continue
        v  = et[idx]
        mu = np.nanmean(v, axis=0)
        sg = np.nanstd(v, axis=0)
        z[idx] = (v - mu) / np.where(sg > 0, sg, np.nan)
    return z


def detect_exeves(et, times_pd):
    """
    Runs ≥ 2 consecutive days with z ≥ z80 containing ≥ 1 day with z ≥ z95.
    Returns (T, nlat, nlon) bool.
    """
    T, nlat, nlon = et.shape
    N = nlat * nlon
    z    = pentad_zscore(et, times_pd).reshape(T, N)
    z80  = np.nanpercentile(z, 80, axis=0)
    z95  = np.nanpercentile(z, 95, axis=0)
    above   = z >= z80
    extreme = z >= z95
    out = np.zeros((T, N), dtype=bool)
    for j in range(N):
        if not above[:, j].any():
            continue
        d  = np.diff(above[:, j].astype(np.int8), prepend=0, append=0)
        ss = np.where(d ==  1)[0]
        es = np.where(d == -1)[0]
        for s, e in zip(ss, es):
            if (e - s) >= 2 and extreme[s:e, j].any():
                out[s:e, j] = True
    return out.reshape(T, nlat, nlon)


def make_q95_threshold(precip, min_q95=0.1):
    """Per-cell q95 from the supplied precipitation period and valid mask."""
    q95v  = np.nanpercentile(precip, 95, axis=0)
    return q95v, q95v > min_q95


def apply_q95(precip, q95v):
    """(T, nlat, nlon) bool: days exceeding per-cell q95 threshold."""
    return precip >= q95v[np.newaxis]


def expand_window(exeve, radius=3):
    """Dilate ExEvE boolean array by ±radius days."""
    T = exeve.shape[0]
    flat = exeve.reshape(T, -1).copy()
    win  = flat.copy()
    for k in range(1, radius + 1):
        fwd = np.roll(flat,  k, axis=0); fwd[:k]  = False
        bwd = np.roll(flat, -k, axis=0); bwd[-k:] = False
        win |= fwd | bwd
    return win.reshape(exeve.shape)


def per_cell_stats(exeve, q95m, valid, T):
    """
    Returns:
      pct_q95 (nlat, nlon): % of q95 days falling during ExEvEs (exact overlap only)
      pct_exeve (nlat, nlon): % of all days that are ExEvE days
    """
    n_q  = q95m.sum(axis=0).astype(float)
    n_qw = (exeve & q95m).sum(axis=0).astype(float)
    pct_q95 = np.where((n_q > 0) & valid, n_qw / n_q * 100, np.nan)
    pct_ev  = np.where(valid, exeve.sum(axis=0) / T * 100, np.nan)
    return pct_q95, pct_ev


def lag_profile(exeve, q95m, valid):
    """
    Mean across valid cells of: per-cell % of total q95 days at each lag position.
    lag=0 is divided by average event length to be comparable to single-day lags.
    Also returns the expected baseline if ExEvEs and q95 were independent.
    """
    T, nlat, nlon = exeve.shape
    N = nlat * nlon
    ev_f = exeve.reshape(T, N)
    q_f  = q95m.reshape(T, N)
    vf   = valid.reshape(N)
    rows, bls = [], []
    for j in range(N):
        if not vf[j]:
            continue
        ev, q = ev_f[:, j], q_f[:, j]
        nq = int(q.sum())
        if nq == 0:
            continue
        d  = np.diff(ev.astype(np.int8), prepend=0, append=0)
        ss = np.where(d ==  1)[0]
        es = np.where(d == -1)[0]
        if len(ss) == 0:
            continue
        avg_len = float(np.mean(es - ss))
        bls.append(len(ss) / T * 100)   # expected % per event-day if independent
        pcts = []
        for lag in LAGS:
            if lag == 0:
                pcts.append(float((ev & q).sum()) / avg_len / nq * 100)
            else:
                lag_ts = set()
                for s, e in zip(ss, es):
                    t = s + lag if lag < 0 else e + lag - 1
                    if 0 <= t < T and not ev[t]:
                        lag_ts.add(t)
                pcts.append(sum(int(q[t]) for t in lag_ts) / nq * 100)
        rows.append(pcts)
    profile  = np.nanmean(rows, axis=0) if rows else np.zeros(7)
    baseline = float(np.mean(bls))       if bls   else 0.0
    return profile, baseline


# ── Load data ─────────────────────────────────────────────────────────────────

print("Loading FLUXCOM (land mask + ET) ...")
fc_ds    = xr.open_dataset(input_path(FLUXCOM_FILE)).sel(time=slice("2001-01-01", "2020-12-31"))
land_2d  = fc_ds["land_fraction"].values >= 0.5
ref_lat  = fc_ds.lat.values
ref_lon  = fc_ds.lon.values
fc_et_np = (fc_ds["ET"].values * 24).astype(np.float32)  # mm/hr → mm/day
fc_times = pd.DatetimeIndex(fc_ds.time.values)
print(f"  FLUXCOM period: {fc_times[0].date()} → {fc_times[-1].date()}")

print("Loading MSWEP ...")
ms_ds    = xr.open_dataset(input_path(MSWEP_FILE)).sel(time=slice("2001-01-01", "2020-12-31"))
ms_np    = ms_ds["mask"].values.astype(np.float32)
ms_times = pd.DatetimeIndex(ms_ds.time.values)
print(f"  MSWEP period: {ms_times[0].date()} → {ms_times[-1].date()}")

# q95 threshold computed on the analysis period (fixed across both scenarios)
q95_vals, q95_valid = make_q95_threshold(ms_np)
valid_base = q95_valid & land_2d

print("Loading GLEAM ...")
gl_ds    = xr.open_dataset(input_path(GLEAM_FILE)).sel(time=slice("2001-01-01", "2020-12-31"))
gl_np    = gl_ds["E"].values.astype(np.float32)
gl_times = pd.DatetimeIndex(gl_ds.time.values)
print(f"  GLEAM period: {gl_times[0].date()} → {gl_times[-1].date()}")


# ── Analysis periods ──────────────────────────────────────────────────────────

def tidx(times, t0, t1):
    return np.where((times >= pd.Timestamp(t0)) & (times <= pd.Timestamp(t1)))[0]

gl_s2 = tidx(gl_times, "2001-01-01", "2020-12-31")
fc_s3 = tidx(fc_times, "2001-01-01", "2020-12-31")
ms_s2 = tidx(ms_times, "2001-01-01", "2020-12-31")
validate_alignment(gl_ds, fc_ds, ms_ds)


# ── Run all scenarios ─────────────────────────────────────────────────────────

scenarios = [
    ("S2: GLEAM 2001–2020",   gl_np,    gl_times, gl_s2, ms_s2),
    ("S3: FLUXCOM 2001–2020", fc_et_np, fc_times, fc_s3, ms_s2),
]

results = []
for label, et_np, et_times, et_idx, ms_idx in scenarios:
    print(f"\n{'─'*50}\n{label}")
    et_sub = et_np[et_idx]
    ms_sub = ms_np[ms_idx]
    t_pd   = pd.DatetimeIndex(et_times[et_idx])
    T      = len(et_idx)

    print(f"  Detecting ExEvEs ({T} days × {et_sub.shape[1]}×{et_sub.shape[2]} cells) ...")
    exeve = detect_exeves(et_sub, t_pd)

    q95m  = apply_q95(ms_sub, q95_vals)
    et_has_data = ~np.all(np.isnan(et_sub), axis=0)
    valid = valid_base & et_has_data

    n_exeve_cells = (exeve.sum(axis=0)[valid] > 0).sum()
    print(f"  {n_exeve_cells} / {valid.sum()} valid cells have ≥1 ExEvE day")

    print(f"  Computing per-cell stats ...")
    pct_q95, pct_ev = per_cell_stats(exeve, q95m, valid, T)

    print(f"  Computing lag profile ...")
    profile, baseline = lag_profile(exeve, q95m, valid)

    print(f"  ExEvE days: {np.nanmean(pct_ev):.1f}%  |  q95-in-window: {np.nanmean(pct_q95):.1f}%")
    print(f"  Lag profile: {np.round(profile, 2)}  baseline={baseline:.2f}%")

    results.append(dict(
        label=label, pct_q95=pct_q95, pct_ev=pct_ev,
        profile=profile, baseline=baseline,
    ))


# ── Plotting helpers ──────────────────────────────────────────────────────────

def density_panel(ax, vals_list, labels, colors, title, xlabel, xlim=(0, 100)):
    for vals, label, color in zip(vals_list, labels, colors):
        v = vals[np.isfinite(vals)].ravel()
        v = v[(v >= xlim[0]) & (v <= xlim[1])]
        if len(v) < 20:
            continue
        mu = np.nanmean(v)
        ax.hist(v, bins=60, range=xlim, density=True, alpha=0.3, color=color)
        kde = gaussian_kde(v, bw_method=0.15)
        x = np.linspace(*xlim, 500)
        ax.plot(x, kde(x), color=color, lw=2,
                label=f"{label.replace(chr(10), ' ')}  mean={mu:.1f}%")
        ax.axvline(mu, color=color, lw=1.2, ls='--')
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.set_xlim(*xlim)
    ax.grid(axis='y', alpha=0.3)


def map_panel(ax, vals, lat, lon, title, vmin, vmax, cmap="YlOrRd"):
    lon2d, lat2d = np.meshgrid(lon, lat)
    im = ax.pcolormesh(lon2d, lat2d, vals, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label="%", shrink=0.8)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_xlim(lon.min(), lon.max())
    ax.set_ylim(lat.min(), lat.max())


def lag_panel(ax, profile, baseline, color, title):
    x = np.arange(7)
    bar_colors = ["firebrick" if l == 0 else color for l in LAGS]
    ax.bar(x, profile, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax.axhline(baseline, color="gray", lw=1.3, ls='--',
               label=f"expected if random: {baseline:.1f}%")
    ax.set_xticks(x)
    ax.set_xticklabels(LAG_LABELS)
    ax.set_xlabel("Days relative to ExEvE")
    ax.set_ylabel("% of total q95 days\n(per event-day)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)


# ── Figure 1: density of % q95 in ±3-day window ──────────────────────────────
fig1, ax1 = plt.subplots(figsize=(9, 5))
density_panel(ax1,
              [r["pct_q95"] for r in results],
              SC_LABELS, SC_COLORS,
              "% of q95 days falling during ExEvEs (per land cell)",
              "Percentage (%)")
fig1.tight_layout()
fig1.savefig(output_path("exeves_q95_density.png"), dpi=150, bbox_inches="tight")
print("\nSaved exeves_q95_density.png")

# ── Figure 2: spatial maps ────────────────────────────────────────────────────
all_q95 = np.concatenate([r["pct_q95"][np.isfinite(r["pct_q95"])] for r in results])
vmax2   = np.nanpercentile(all_q95, 97)

fig2, axes2 = plt.subplots(2, 1, figsize=(10, 8))
for ax, r, slabel in zip(axes2, results, SC_LABELS):
    map_panel(ax, r["pct_q95"], ref_lat, ref_lon,
              f"% q95 days during ExEvEs — {slabel.replace(chr(10), ' ')}",
              vmin=0, vmax=vmax2)
fig2.suptitle("Spatial distribution of q95 precipitation during ExEvEs",
              fontsize=12, fontweight="bold")
fig2.tight_layout()
fig2.savefig(output_path("exeves_q95_map.png"), dpi=150, bbox_inches="tight")
print("Saved exeves_q95_map.png")

# ── Figure 3: density of ExEvE % of all days ─────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(9, 5))
density_panel(ax3,
              [r["pct_ev"] for r in results],
              SC_LABELS, SC_COLORS,
              "ExEvEs as % of all days (per land cell)",
              "Percentage (%)", xlim=(0, 30))
fig3.tight_layout()
fig3.savefig(output_path("exeves_pct_density.png"), dpi=150, bbox_inches="tight")
print("Saved exeves_pct_density.png")

# ── Figure 4: lag bar charts ──────────────────────────────────────────────────
fig4, axes4 = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
for ax, r, slabel, color in zip(axes4, results, SC_LABELS, SC_COLORS):
    lag_panel(ax, r["profile"], r["baseline"], color,
              f"Timing of q95 relative to ExEvEs\n{slabel.replace(chr(10), ' ')}")
fig4.suptitle("Standardised timing of peak precipitation relative to ExEvEs",
              fontsize=12, fontweight="bold")
fig4.tight_layout()
fig4.savefig(output_path("exeves_lag.png"), dpi=150, bbox_inches="tight")
print("Saved exeves_lag.png")

print("\nAll done.")
