"""
Compare GLEAM vs FLUXCOM ExEvEs for the common 2001-2020 period.

Outputs:
  compare_exeves_seasonal.png  – monthly ExEvE frequency bar chart
  compare_exeves_jaccard.png   – per-cell Jaccard overlap map
  compare_exeves_freq.png      – side-by-side ExEvE frequency spatial maps
"""
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')

GLEAM_FILE   = "gleam_e_mm_med_198001_202412_025_daily.nc"
FLUXCOM_FILE = "ET_2001_2021_025_daily_med.nc"


# ── Reuse detection functions ────────────────────────────────────────────────

def pentad_zscore(et, times_pd):
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


# ── Load datasets ────────────────────────────────────────────────────────────

print("Loading FLUXCOM ...")
fc_ds    = xr.open_dataset(FLUXCOM_FILE)
land_2d  = fc_ds["land_fraction"].values >= 0.5
ref_lat  = fc_ds.lat.values
ref_lon  = fc_ds.lon.values
fc_et    = (fc_ds["ET"].values * 24).astype(np.float32)
fc_times = pd.DatetimeIndex(fc_ds.time.values)

print("Loading GLEAM ...")
gl_ds    = xr.open_dataset(GLEAM_FILE)
gl_et    = gl_ds["E"].values.astype(np.float32)
gl_times = pd.DatetimeIndex(gl_ds.time.values)

# Common period: 2001-01-01 to 2020-12-31
t0, t1 = "2001-01-01", "2020-12-31"
gl_idx = np.where((gl_times >= pd.Timestamp(t0)) & (gl_times <= pd.Timestamp(t1)))[0]
fc_idx = np.where((fc_times >= pd.Timestamp(t0)) & (fc_times <= pd.Timestamp(t1)))[0]

gl_sub    = gl_et[gl_idx]
fc_sub    = fc_et[fc_idx]
common_pd = pd.DatetimeIndex(gl_times[gl_idx])
T         = len(gl_idx)
print(f"Common period: {common_pd[0].date()} → {common_pd[-1].date()}  ({T} days)")

# Valid land cells where both products have data
gl_valid = ~np.all(np.isnan(gl_sub), axis=0)
fc_valid = ~np.all(np.isnan(fc_sub), axis=0)
valid    = land_2d & gl_valid & fc_valid
print(f"Valid cells: {valid.sum()}")


# ── Detect ExEvEs ────────────────────────────────────────────────────────────

print("Detecting GLEAM ExEvEs ...")
gl_ev = detect_exeves(gl_sub, common_pd)

print("Detecting FLUXCOM ExEvEs ...")
fc_ev = detect_exeves(fc_sub, common_pd)

print(f"GLEAM  ExEvE days: {gl_ev[:, valid].mean()*100:.1f}%")
print(f"FLUXCOM ExEvE days: {fc_ev[:, valid].mean()*100:.1f}%")


# ── Figure 1: Monthly ExEvE frequency ───────────────────────────────────────

months = np.arange(1, 13)
month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]

# mean fraction of valid cells that are in ExEvE per month
gl_monthly  = np.zeros(12)
fc_monthly  = np.zeros(12)
for m in months:
    mask = common_pd.month == m
    gl_monthly[m-1]  = gl_ev[mask][:, valid].mean() * 100
    fc_monthly[m-1]  = fc_ev[mask][:, valid].mean() * 100

x = np.arange(12)
width = 0.35

fig1, ax1 = plt.subplots(figsize=(11, 5))
ax1.bar(x - width/2, gl_monthly,  width, label="GLEAM",   color="steelblue",  alpha=0.8)
ax1.bar(x + width/2, fc_monthly, width, label="FLUXCOM", color="darkorange", alpha=0.8)
ax1.set_xticks(x)
ax1.set_xticklabels(month_labels)
ax1.set_ylabel("% of land cells in ExEvE")
ax1.set_xlabel("Month")
ax1.set_title("Monthly ExEvE frequency — GLEAM vs FLUXCOM (2001–2020)", fontsize=12)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
fig1.tight_layout()
fig1.savefig("compare_exeves_seasonal.png", dpi=150, bbox_inches="tight")
print("Saved compare_exeves_seasonal.png")


# ── Figure 2: Jaccard overlap map ───────────────────────────────────────────
# Jaccard = |A ∩ B| / |A ∪ B|  per cell

gl_f = gl_ev.reshape(T, -1)
fc_f = fc_ev.reshape(T, -1)
vf   = valid.reshape(-1)
nlat, nlon = valid.shape

inter = (gl_f & fc_f).sum(axis=0).astype(float)
union = (gl_f | fc_f).sum(axis=0).astype(float)
jaccard = np.where(vf & (union > 0), inter / union, np.nan).reshape(nlat, nlon)

lon2d, lat2d = np.meshgrid(ref_lon, ref_lat)

fig2, ax2 = plt.subplots(figsize=(10, 4))
im = ax2.pcolormesh(lon2d, lat2d, jaccard, cmap="viridis", vmin=0, vmax=0.3)
plt.colorbar(im, ax=ax2, label="Jaccard index")
ax2.set_title("Temporal co-occurrence of ExEvEs: GLEAM ∩ FLUXCOM / GLEAM ∪ FLUXCOM (2001–2020)",
              fontsize=10)
ax2.set_xlabel("Longitude"); ax2.set_ylabel("Latitude")
med_j = np.nanmedian(jaccard[valid])
ax2.text(0.02, 0.05, f"Median Jaccard = {med_j:.3f}",
         transform=ax2.transAxes, fontsize=10,
         bbox=dict(boxstyle="round", fc="white", alpha=0.8))
fig2.tight_layout()
fig2.savefig("compare_exeves_jaccard.png", dpi=150, bbox_inches="tight")
print("Saved compare_exeves_jaccard.png")


# ── Figure 3: Side-by-side ExEvE frequency maps ──────────────────────────────

gl_freq = np.where(valid, gl_ev.sum(axis=0) / T * 100, np.nan)
fc_freq = np.where(valid, fc_ev.sum(axis=0) / T * 100, np.nan)
vmax = max(np.nanpercentile(gl_freq[valid], 97), np.nanpercentile(fc_freq[valid], 97))

fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5))

im = ax3a.pcolormesh(lon2d, lat2d, gl_freq, cmap="YlOrRd", vmin=0, vmax=vmax)
plt.colorbar(im, ax=ax3a, label="%")
ax3a.set_title(f"GLEAM ExEvE days (mean={np.nanmean(gl_freq[valid]):.1f}%)", fontsize=10)
ax3a.set_xlabel("Longitude"); ax3a.set_ylabel("Latitude")

im = ax3b.pcolormesh(lon2d, lat2d, fc_freq, cmap="YlOrRd", vmin=0, vmax=vmax)
plt.colorbar(im, ax=ax3b, label="%")
ax3b.set_title(f"FLUXCOM ExEvE days (mean={np.nanmean(fc_freq[valid]):.1f}%)", fontsize=10)
ax3b.set_xlabel("Longitude"); ax3b.set_ylabel("Latitude")

fig3.suptitle("ExEvE frequency as % of all days (2001–2020)", fontsize=12, fontweight="bold")
fig3.tight_layout()
fig3.savefig("compare_exeves_freq.png", dpi=150, bbox_inches="tight")
print("Saved compare_exeves_freq.png")

print("\nAll done.")
