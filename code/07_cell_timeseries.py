from _paths import input_path, output_path
from _alignment import validate_alignment
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings; warnings.filterwarnings('ignore')

GLEAM_FILE   = "gleam_e_mm_med_198001_202412_025_daily.nc"
FLUXCOM_FILE = "ET_2001_2020_025_daily_med.nc"
MSWEP_FILE   = "mswep-v2-8_tp_mm_med_197901_202012_025_daily.nc"


def pentad_zscore(et, times_pd):
    T, nlat, nlon = et.shape
    pentad = np.minimum((times_pd.dayofyear.values - 1) // 5, 72)
    z = np.full_like(et, np.nan, dtype=np.float32)
    for p in range(73):
        idx = pentad == p
        if not idx.any(): continue
        v = et[idx]
        mu = np.nanmean(v, axis=0); sg = np.nanstd(v, axis=0)
        z[idx] = (v - mu) / np.where(sg > 0, sg, np.nan)
    return z


def detect_exeves(et, times_pd):
    T, nlat, nlon = et.shape; N = nlat * nlon
    z   = pentad_zscore(et, times_pd).reshape(T, N)
    z80 = np.nanpercentile(z, 80, axis=0)
    z95 = np.nanpercentile(z, 95, axis=0)
    above = z >= z80; extreme = z >= z95
    out = np.zeros((T, N), dtype=bool)
    for j in range(N):
        if not above[:, j].any(): continue
        d  = np.diff(above[:, j].astype(np.int8), prepend=0, append=0)
        ss = np.where(d ==  1)[0]; es = np.where(d == -1)[0]
        for s, e in zip(ss, es):
            if (e - s) >= 2 and extreme[s:e, j].any():
                out[s:e, j] = True
    return out.reshape(T, nlat, nlon)


# ── Load ─────────────────────────────────────────────────────────────────────

print("Loading ...")
fc_ds    = xr.open_dataset(input_path(FLUXCOM_FILE))
land_2d  = fc_ds["land_fraction"].values >= 0.5
ref_lat  = fc_ds.lat.values
ref_lon  = fc_ds.lon.values
fc_et    = (fc_ds["ET"].values * 24).astype(np.float32)
fc_times = pd.DatetimeIndex(fc_ds.time.values)

gl_ds    = xr.open_dataset(input_path(GLEAM_FILE))
gl_et    = gl_ds["E"].values.astype(np.float32)
gl_times = pd.DatetimeIndex(gl_ds.time.values)

ms_ds    = xr.open_dataset(input_path(MSWEP_FILE))
ms_np    = ms_ds["mask"].values.astype(np.float32)
ms_times = pd.DatetimeIndex(ms_ds.time.values)

t0, t1 = "2001-01-01", "2020-12-31"
fc_idx = np.where((fc_times >= pd.Timestamp(t0)) & (fc_times <= pd.Timestamp(t1)))[0]
gl_idx = np.where((gl_times >= pd.Timestamp(t0)) & (gl_times <= pd.Timestamp(t1)))[0]
ms_idx = np.where((ms_times >= pd.Timestamp(t0)) & (ms_times <= pd.Timestamp(t1)))[0]
validate_alignment(fc_ds.isel(time=fc_idx), gl_ds.isel(time=gl_idx), ms_ds.isel(time=ms_idx))

fc_sub = fc_et[fc_idx]
gl_sub = gl_et[gl_idx]
ms_sub = ms_np[ms_idx]
dates  = pd.DatetimeIndex(fc_times[fc_idx])

# q95 threshold from the analysis period
q95_vals = np.nanpercentile(ms_sub, 95, axis=0)
q95m     = ms_sub >= q95_vals[np.newaxis]

gl_valid = ~np.all(np.isnan(gl_sub), axis=0)
fc_valid = ~np.all(np.isnan(fc_sub), axis=0)
valid = land_2d & gl_valid & fc_valid

# ── Pick a random valid cell ──────────────────────────────────────────────────

rng = np.random.default_rng(42)   # fixed seed for reproducibility
valid_idx = np.argwhere(valid)               # shape (N_valid, 2)
ci, cj = valid_idx[rng.integers(len(valid_idx))]
lat_c, lon_c = ref_lat[ci], ref_lon[cj]
print(f"Selected cell: lat={lat_c:.3f}, lon={lon_c:.3f}  (row={ci}, col={cj})")

# ── Detect ExEvEs for this cell only ─────────────────────────────────────────

print("Detecting ExEvEs ...")
gl_ev = detect_exeves(gl_sub, dates)
fc_ev = detect_exeves(fc_sub, dates)

gl_ts = gl_sub[:, ci, cj]
fc_ts = fc_sub[:, ci, cj]
gl_ex = gl_ev[:, ci, cj]
fc_ex = fc_ev[:, ci, cj]
q95_ts = q95m[:, ci, cj]

print(f"GLEAM  ExEvE days: {gl_ex.mean()*100:.1f}%")
print(f"FLUXCOM ExEvE days: {fc_ex.mean()*100:.1f}%")

# ── Plot — show one year at a time to keep it readable ───────────────────────

years = sorted(dates.year.unique())
n_years = len(years)
fig, axes = plt.subplots(n_years, 1, figsize=(18, n_years * 1.6), sharex=False)
fig.suptitle(f"ET time series + ExEvEs — lat={lat_c:.3f}°, lon={lon_c:.3f}°  (2001–2020)",
             fontsize=12, fontweight="bold")

gl_patch  = mpatches.Patch(color="steelblue",  alpha=0.4, label="GLEAM ExEvE")
fc_patch  = mpatches.Patch(color="darkorange", alpha=0.4, label="FLUXCOM ExEvE")
both_patch = mpatches.Patch(color="purple",    alpha=0.5, label="Both ExEvE")

for ax, yr in zip(axes, years):
    mask = dates.year == yr
    d    = dates[mask]
    gl_y = gl_ts[mask]
    fc_y = fc_ts[mask]
    gl_e = gl_ex[mask]
    fc_e = fc_ex[mask]
    q95_y = q95_ts[mask]

    ax.plot(d, gl_y, color="steelblue",  lw=0.8, label="GLEAM")
    ax.plot(d, fc_y, color="darkorange", lw=0.8, label="FLUXCOM", alpha=0.85)

    # shade ExEvE periods
    def shade(ev, color):
        in_ev = False
        for i, flag in enumerate(ev):
            if flag and not in_ev:
                start = d[i]; in_ev = True
            elif not flag and in_ev:
                ax.axvspan(start, d[i], alpha=0.3, color=color, lw=0)
                in_ev = False
        if in_ev:
            ax.axvspan(start, d[-1], alpha=0.3, color=color, lw=0)

    shade(gl_e & ~fc_e, "steelblue")
    shade(fc_e & ~gl_e, "darkorange")
    shade(gl_e & fc_e,  "purple")

    for dt in d[q95_y]:
        ax.axvline(dt, color="black", lw=0.6, ls="--", alpha=0.5)

    ax.set_ylabel(str(yr), fontsize=8, rotation=0, labelpad=28, va="center")
    ax.tick_params(axis='x', labelsize=7)
    ax.tick_params(axis='y', labelsize=7)
    ax.grid(axis='y', alpha=0.2)

axes[0].legend(handles=[
    plt.Line2D([0],[0], color="steelblue",  lw=1.2, label="GLEAM ET"),
    plt.Line2D([0],[0], color="darkorange", lw=1.2, label="FLUXCOM ET"),
    gl_patch, fc_patch, both_patch,
    plt.Line2D([0],[0], color="black", lw=0.8, ls="--", label="q95 precip"),
], fontsize=8, loc="upper right", ncol=6)

axes[-1].set_xlabel("Date")
fig.text(0.005, 0.5, "ET (mm day⁻¹)", va="center", rotation="vertical", fontsize=9)
fig.tight_layout(rect=[0.02, 0, 1, 0.97])
fig.savefig(output_path("cell_timeseries.png"), dpi=150, bbox_inches="tight")
print("Saved cell_timeseries.png")
