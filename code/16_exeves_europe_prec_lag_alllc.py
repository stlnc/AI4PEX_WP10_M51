"""
Lagged relationship: % of heavy-rainfall days (q50–q95) with an ExEvE day at
offset lag ∈ {-5…-1, +1…+5} days.

  lag < 0 : ExEvE precedes rain  (ExEvE → rain direction)
  lag > 0 : rain precedes ExEvE  (rain → ExEvE direction)

Plot (6 panels, one per LC type + All Europe):
  x-axis : lag in days, natural gap at 0
  y-axis : % of heavy-rainfall days with an ExEvE day at that lag
  ribbons : q50–q95 spread per scenario (fill); mean across quantiles (bold line)
Output : exeves_europe_prec_lag_alllc.png
"""
from _analysis import pentad_zscore, detect_exeves
from _alignment import validate_alignment
from _figure_style import EUROPE_LAND_COVER_COLORS
from _paths import input_path, output_path
import os
import multiprocessing
import warnings
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pyreadr
from concurrent.futures import ProcessPoolExecutor

multiprocessing.set_start_method('fork', force=True)
warnings.filterwarnings('ignore')

GLEAM_FILE   = "gleam_e_mm_eur_2001_2020_025_daily.nc"
FLUXCOM_FILE = "ET_2001_2020_025_daily_eur.nc"
MSWEP_FILE   = "mswep-v2-8_tp_mm_eur_2001_2020_025_daily.nc"
MASKS_FILE   = "masks.rda"

LC_TYPES  = ["Forests", "Croplands", "Savannas", "Grasslands", "Shrublands"]
LC_COLORS = list(EUROPE_LAND_COVER_COLORS[:5])

SC_KEYS   = ['S2', 'S3']
SC_LABELS = ["GLEAM 2001–2020", "FLUXCOM 2001–2020"]
SC_COLORS = ['#ff7f0e', '#2ca02c']

SEASON_MONTHS = [4, 5, 6, 7, 8, 9, 10]
QUANTILES     = np.arange(50, 96, 5)            # 50, 55, …, 95
LAGS          = list(range(-5, 0)) + list(range(1, 6))   # skip 0

N_WORKERS = min(8, os.cpu_count() or 4)
print(f"Using {N_WORKERS} parallel workers")


# ── Detection ─────────────────────────────────────────────────────────────────


def _detect_chunk(args):
    et_chunk, times_arr = args
    return detect_exeves(et_chunk, pd.DatetimeIndex(times_arr))


def detect_exeves_parallel(et, times_pd):
    T, N = et.shape
    if N == 0:
        return np.zeros((T, 0), dtype=bool)
    chunk_size = max(1, (N + N_WORKERS - 1) // N_WORKERS)
    chunks = [
        (et[:, i : i + chunk_size], times_pd.values)
        for i in range(0, N, chunk_size)
    ]
    with ProcessPoolExecutor(max_workers=N_WORKERS) as exe:
        parts = list(exe.map(_detect_chunk, chunks))
    return np.concatenate(parts, axis=1)


def lag_overlap_pct(heavy, exeve, lag):
    """
    Per-cell % of heavy-rainfall days that have an ExEvE day at exactly `lag`
    days offset.  heavy and exeve are (T, N) bool arrays.
      lag > 0  →  exeve[t + lag]  (ExEvE after rain)
      lag < 0  →  exeve[t + lag]  (ExEvE before rain, same formula)
    """
    T = heavy.shape[0]
    k = abs(lag)
    if lag > 0:
        h, e = heavy[: T - k], exeve[k:]
    else:
        h, e = heavy[k:], exeve[: T - k]
    n_h  = h.sum(axis=0).astype(np.float32)
    n_hw = (h & e).sum(axis=0).astype(np.float32)
    return np.where(n_h > 0, n_hw / n_h * 100.0, np.nan)


# ── Load masks ────────────────────────────────────────────────────────────────

print("Loading masks ...")
masks = pyreadr.read_r(str(input_path(MASKS_FILE)))['masks']
eur = masks[
    masks['ipcc_continent'].isin(['EUROPE', 'EUROPE-AFRICA']) &
    (masks['land_mask'] == 'land') &
    masks['land_cover_short_class'].isin(LC_TYPES)
].copy().reset_index(drop=True)
del masks
print(f"  Target cells: {len(eur)}")

lat_min = eur.lat.min() - 0.5;  lat_max = eur.lat.max() + 0.5
lon_min = eur.lon.min() - 0.5;  lon_max = eur.lon.max() + 0.5


def select_period_apr_oct(da, t0, t1):
    sub = da.sel(time=slice(t0, t1))
    return sub.where(sub.time.dt.month.isin(SEASON_MONTHS), drop=True)


# ── Load GLEAM ────────────────────────────────────────────────────────────────

print("Loading GLEAM ...")
gl_ds    = xr.open_dataset(input_path(GLEAM_FILE))
gl_ref   = gl_ds['E'].sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
eur_lats = gl_ref.lat.values
eur_lons = gl_ref.lon.values
gl_s1_da = select_period_apr_oct(gl_ref, '2001-01-01', '2020-12-31')
gl_s1    = gl_s1_da.values.astype(np.float32)
t_s1     = pd.DatetimeIndex(gl_s1_da.time.values)
gl_ds.close()


# ── Load MSWEP — wet-day quantile thresholds ──────────────────────────────────

print("Loading MSWEP ...")
ms_ds   = xr.open_dataset(input_path(MSWEP_FILE))
ms_eur  = ms_ds['mask'].sel(time=slice('2001-01-01', '2020-12-31'), lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
ms_full = ms_eur.where(ms_eur.time.dt.month.isin(SEASON_MONTHS),
                       drop=True).values.astype(np.float32)
ms_full[ms_full <= 0] = np.nan   # wet-day-only quantile basis

print(f"Computing {len(QUANTILES)} quantile thresholds ...")
q_grids = {int(q): np.nanpercentile(ms_full, q, axis=0) for q in QUANTILES}
q_valid = np.isfinite(q_grids[95])
del ms_full

ms_s1_da = select_period_apr_oct(ms_eur, '2001-01-01', '2020-12-31')
ms_s1    = ms_s1_da.values.astype(np.float32)
ms_ds.close()

t_s2_start = t_s1.searchsorted(pd.Timestamp('2001-01-01'))
gl_s2 = gl_s1[t_s2_start:]
ms_s2 = ms_s1[t_s2_start:]
t_s2  = t_s1[t_s2_start:]


# ── Load FLUXCOM — dask-chunked parallel I/O ──────────────────────────────────

print("Loading FLUXCOM (dask-chunked) ...")
fc_ds    = xr.open_dataset(input_path(FLUXCOM_FILE), chunks={})
fc_land  = (fc_ds["land_fraction"].values >= 0.5)
fc_s3_da = select_period_apr_oct(fc_ds['ET'], '2001-01-01', '2020-12-31')
validate_alignment(gl_s1_da, ms_s1_da, fc_s3_da)
t_s3     = pd.DatetimeIndex(fc_s3_da.time.values)
fc_s3    = (fc_s3_da.compute().values * 24).astype(np.float32)
fc_ds.close()

print(f"S2: {gl_s2.shape}  S3: {fc_s3.shape} (Apr–Oct only)")


# ── Grid index mapping ────────────────────────────────────────────────────────

dlat = eur_lats[1] - eur_lats[0]
dlon = eur_lons[1] - eur_lons[0]
nlat, nlon = len(eur_lats), len(eur_lons)

i_lat = np.round((eur.lat.values - eur_lats[0]) / dlat).astype(int)
i_lon = np.round((eur.lon.values - eur_lons[0]) / dlon).astype(int)

ok    = (i_lat >= 0) & (i_lat < nlat) & (i_lon >= 0) & (i_lon < nlon)
eur   = eur[ok].copy().reset_index(drop=True)
i_lat = i_lat[ok]
i_lon = i_lon[ok]
q_valid_c  = q_valid[i_lat, i_lon]
q_thresh_c = {q: q_grids[q][i_lat, i_lon] for q in QUANTILES}
print(f"Cells in bounds: {len(eur)}")


# ── Compute lagged overlap per (scenario × LC × quantile × lag) ───────────────
#
# records entries: {scenario, lc, quantile, lag, mean, p25, p75, n}
# lc='All' aggregates all non-Barren cells.

records = []

for sc_label, sc_key, et_grid, times_pd, ms_grid in [
    (SC_LABELS[0], 'S2', gl_s2, t_s2, ms_s2),
    (SC_LABELS[1], 'S3', fc_s3, t_s3, ms_s2),
]:
    print(f"\n{sc_key}: {et_grid.shape[0]} days — parallel ExEvE detection ...")
    et_cells = et_grid[:, i_lat, i_lon]
    ms_cells = ms_grid[:, i_lat, i_lon]

    exeve = detect_exeves_parallel(et_cells, times_pd)   # (T, N) bool

    has_data = ~np.all(np.isnan(et_cells), axis=0)
    valid    = has_data & q_valid_c
    if sc_key == 'S3':
        valid &= fc_land[i_lat, i_lon]

    # Build per-LC validity masks + one mask for all cells combined
    lc_masks = {lc: (eur['land_cover_short_class'] == lc).values & valid
                for lc in LC_TYPES}
    lc_masks['All'] = valid

    for q in QUANTILES:
        thresh = q_thresh_c[q][np.newaxis, :]
        heavy  = ms_cells >= thresh          # (T, N) bool

        for lag in LAGS:
            pct_cells = lag_overlap_pct(heavy, exeve, lag)   # (N,)

            for lc_key, lc_mask in lc_masks.items():
                pct = pct_cells[lc_mask]
                pct = pct[np.isfinite(pct)]
                if pct.size == 0:
                    continue
                records.append({
                    'scenario': sc_label,
                    'lc':       lc_key,
                    'quantile': int(q),
                    'lag':      lag,
                    'mean':     float(np.nanmean(pct)),
                    'p25':      float(np.nanpercentile(pct, 25)),
                    'p75':      float(np.nanpercentile(pct, 75)),
                    'n':        pct.size,
                })

        print(f"  {sc_key} q{q}  done")

    del et_cells, ms_cells, exeve, heavy

del gl_s1, ms_s1, fc_s3

df = pd.DataFrame([{k: v for k, v in r.items()} for r in records])


# ── Plot ──────────────────────────────────────────────────────────────────────
#
# Layout : 2 × 3 grid; first 5 panels = LC types, 6th = All Europe
# x-axis : lag days (-5…-1 | gap | +1…+5); natural 2-unit gap at 0
# y-axis : % of heavy-rainfall days with ExEvE at that lag
# Per scenario : ribbon (q50–q95 range) + bold mean-across-quantiles line

PANEL_ORDER  = LC_TYPES + ['All']
PANEL_TITLES = LC_TYPES + ['All Europe (excl. Barren)']
PANEL_COLORS = LC_COLORS + ['#333333']

# x positions: use actual lag values; gap between -1 and +1 is 2 units
X = np.array(LAGS, dtype=float)
X_NEG = X[X < 0]
X_POS = X[X > 0]

fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=False)
axes_flat  = axes.flatten()
global_ymax = df["mean"].max()

for ax, lc_key, title, title_col in zip(
        axes_flat, PANEL_ORDER, PANEL_TITLES, PANEL_COLORS):

    sub = df[df['lc'] == lc_key]

    for sc_label, sc_col in zip(SC_LABELS, SC_COLORS):
        sc = sub[sub['scenario'] == sc_label]
        if sc.empty:
            continue

        # For each lag: q50 mean, q95 mean, and grand mean across all quantiles
        q50_means, q95_means, grand_means = [], [], []
        for lag in LAGS:
            lag_df = sc[sc['lag'] == lag]
            if lag_df.empty:
                q50_means.append(np.nan); q95_means.append(np.nan)
                grand_means.append(np.nan)
                continue
            q50_means.append(lag_df[lag_df['quantile'] == 50]['mean'].values[0]
                             if not lag_df[lag_df['quantile'] == 50].empty else np.nan)
            q95_means.append(lag_df[lag_df['quantile'] == 95]['mean'].values[0]
                             if not lag_df[lag_df['quantile'] == 95].empty else np.nan)
            grand_means.append(lag_df['mean'].mean())

        q50_means  = np.array(q50_means)
        q95_means  = np.array(q95_means)
        grand_means = np.array(grand_means)

        # Ribbon: q50 (lower bound) to q95 (upper bound) of mean pct
        ax.fill_between(X, q50_means, q95_means,
                        color=sc_col, alpha=0.18, linewidth=0)
        # q50 and q95 boundary lines (thin)
        ax.plot(X, q50_means, color=sc_col, lw=0.8, linestyle='--', alpha=0.6)
        ax.plot(X, q95_means, color=sc_col, lw=0.8, linestyle='--', alpha=0.6)
        # Grand mean (bold)
        ax.plot(X, grand_means, color=sc_col, lw=2.2, marker='o',
                markersize=5, zorder=5, label=sc_label)

    # Visual guide at lag = 0
    ax.axvline(0, color='#888888', lw=1.2, linestyle=':', zorder=3)

    # Directional background shading
    ax.axvspan(X_NEG.min() - 0.5, -0.05, color='#d0e4f7', alpha=0.25, zorder=0)
    ax.axvspan(0.05, X_POS.max() + 0.5, color='#fde8cc', alpha=0.25, zorder=0)

    ax.set_title(title, fontsize=11, fontweight='bold', color=title_col)
    ax.set_xticks(LAGS)
    ax.set_xticklabels(
        [str(l) for l in LAGS], fontsize=8, rotation=45
    )
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0, global_ymax)

# Direction labels on top row only
for ax in axes[0]:
    ylim = ax.get_ylim()
    ypos = ylim[1] * 0.93
    ax.text(-3.0, ypos, "← ExEvE before rain", fontsize=7,
            ha='center', color='#3a6ea5', style='italic')
    ax.text(3.0, ypos, "rain before ExEvE →", fontsize=7,
            ha='center', color='#b55a00', style='italic')

# Shared axis labels
for ax in axes[1, :]:
    ax.set_xlabel("Lag (days)", fontsize=10)
for ax in axes[:, 0]:
    ax.set_ylabel("% of heavy-rainfall days with ExEvE at lag", fontsize=9)

# Legend + q50/q95 ribbon note
axes_flat[0].legend(fontsize=8, framealpha=0.85, loc='upper left')
ribbon_note = mpatches.Patch(facecolor='grey', alpha=0.3,
                              label='ribbon = q50–q95 range\nbold line = mean across quantiles')
axes_flat[0].legend(handles=axes_flat[0].get_legend_handles_labels()[0] + [ribbon_note],
                    labels=axes_flat[0].get_legend_handles_labels()[1] + [''],
                    fontsize=7, framealpha=0.85, loc='upper left')

fig.suptitle(
    "Lagged co-occurrence of heavy rainfall and ExEvEs — Europe (Apr–Oct)",
    fontsize=13, fontweight='bold', y=1.01,
)
fig.tight_layout()
fig.savefig(output_path("exeves_europe_prec_lag_alllc.png"), dpi=150, bbox_inches='tight')
print("\nSaved exeves_europe_prec_lag_alllc.png")
