"""
Scatterplot: heavy-rainfall threshold (q50–q95) vs % of those days inside ExEvEs.
All non-Barren land covers in Europe, 2 ET scenarios.
Output: exeves_europe_prec_quantile_scatter_alllc.png

Performance notes
─────────────────
• FLUXCOM is loaded with dask chunks so .compute() reads in parallel threads.
• detect_exeves (inner Python pixel loop) is parallelised via ProcessPoolExecutor,
  splitting the cell axis into N_WORKERS chunks.
• Rainfall quantiles are computed from WET DAYS ONLY (days with precip > 0 mm),
  consistent with standard heavy-rainfall definitions. Dry days are set to NaN
  before np.nanpercentile so they are excluded from the distribution.
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
import pyreadr
from concurrent.futures import ProcessPoolExecutor

# Force fork on Linux so workers don't re-import __main__ (avoids forkserver
# bootstrap loop on Python 3.14+ where forkserver became the default).
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
QUANTILES     = np.arange(50, 96, 5)   # 50, 55, …, 95

N_WORKERS = min(8, os.cpu_count() or 4)
print(f"Using {N_WORKERS} parallel workers")


# ── Detection helpers ─────────────────────────────────────────────────────────


# Top-level (picklable) wrapper used by ProcessPoolExecutor
def _detect_chunk(args):
    et_chunk, times_arr = args
    return detect_exeves(et_chunk, pd.DatetimeIndex(times_arr))


def detect_exeves_parallel(et, times_pd):
    """Split cell axis into N_WORKERS chunks and detect ExEvEs in parallel."""
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


# ── Load masks — all LC types except Barren ───────────────────────────────────

print("Loading masks ...")
masks = pyreadr.read_r(str(input_path(MASKS_FILE)))['masks']
eur = masks[
    masks['ipcc_continent'].isin(['EUROPE', 'EUROPE-AFRICA']) &
    (masks['land_mask'] == 'land') &
    masks['land_cover_short_class'].isin(LC_TYPES)
].copy().reset_index(drop=True)
del masks
print(f"  Target cells: {len(eur)}")


# ── Spatial bounding box ──────────────────────────────────────────────────────

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


# ── Load MSWEP — quantile thresholds (wet days only) ─────────────────────────

print("Loading MSWEP ...")
ms_ds   = xr.open_dataset(input_path(MSWEP_FILE))
ms_eur  = ms_ds['mask'].sel(time=slice('2001-01-01', '2020-12-31'), lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
ms_full = ms_eur.where(ms_eur.time.dt.month.isin(SEASON_MONTHS),
                       drop=True).values.astype(np.float32)
# ── WET-DAY ONLY: exclude dry days from the quantile distribution ─────────────
ms_full[ms_full <= 0] = np.nan

print(f"Computing {len(QUANTILES)} quantile thresholds from wet days ...")
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


# ── Load FLUXCOM — dask-chunked for parallel I/O ──────────────────────────────
# chunks={'time': 100} lets dask read the NetCDF in parallel slabs so that
# .compute() below uses the threadpool scheduler instead of a single serial read.

print("Loading FLUXCOM (dask-chunked parallel I/O) ...")
fc_ds    = xr.open_dataset(input_path(FLUXCOM_FILE), chunks={})
fc_land  = (fc_ds["land_fraction"].values >= 0.5)
fc_s3_da = select_period_apr_oct(fc_ds['ET'], '2001-01-01', '2020-12-31')
validate_alignment(gl_s1_da, ms_s1_da, fc_s3_da)
t_s3     = pd.DatetimeIndex(fc_s3_da.time.values)
# .compute() triggers dask's parallel chunk reads, then we convert immediately
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


# ── Compute % heavy-rainfall days in ExEvEs per cell × quantile ───────────────

records = []   # list of dicts: scenario, lc, quantile, mean, p25, p75, pct_all

for sc_label, sc_key, et_grid, times_pd, ms_grid in [
    (SC_LABELS[0], 'S2', gl_s2, t_s2, ms_s2),
    (SC_LABELS[1], 'S3', fc_s3, t_s3, ms_s2),
]:
    print(f"\n{sc_key}: {et_grid.shape[0]} days, {len(eur)} cells — parallel ExEvE detection ...")
    et_cells = et_grid[:, i_lat, i_lon]    # (T, N)
    ms_cells = ms_grid[:, i_lat, i_lon]    # (T, N)

    # Parallel ExEvE detection split across cell chunks
    exeve = detect_exeves_parallel(et_cells, times_pd)  # (T, N) bool

    has_data = ~np.all(np.isnan(et_cells), axis=0)
    if sc_key == 'S3':
        valid = has_data & q_valid_c & fc_land[i_lat, i_lon]
    else:
        valid = has_data & q_valid_c

    for lc in LC_TYPES:
        lc_mask = (eur['land_cover_short_class'] == lc).values & valid

        for q in QUANTILES:
            thresh = q_thresh_c[q][np.newaxis, :]
            heavy  = ms_cells >= thresh                          # (T, N)
            n_h    = heavy[:, lc_mask].sum(axis=0).astype(float)
            n_hw   = (exeve[:, lc_mask] & heavy[:, lc_mask]).sum(axis=0).astype(float)
            ok_q   = n_h > 0
            pct    = np.where(ok_q, n_hw / n_h * 100, np.nan)
            pct    = pct[np.isfinite(pct)]

            if pct.size == 0:
                continue
            records.append({
                'scenario': sc_label,
                'lc':       lc,
                'quantile': int(q),
                'mean':     float(np.nanmean(pct)),
                'p25':      float(np.nanpercentile(pct, 25)),
                'p75':      float(np.nanpercentile(pct, 75)),
                'pct_all':  pct,
            })
        print(f"  {sc_key} {lc:<12}  q95 mean={records[-1]['mean']:.1f}%  n={pct.size}")

    del et_cells, ms_cells, exeve

del gl_s1, ms_s1, fc_s3


# ── Aggregate records across all LC types (for the 6th panel) ────────────────

all_lc_records = []
for sc_label in SC_LABELS:
    for q in QUANTILES:
        recs_q = [r for r in records if r['scenario'] == sc_label and r['quantile'] == int(q)]
        if not recs_q:
            continue
        pct_combined = np.concatenate([r['pct_all'] for r in recs_q])
        pct_combined = pct_combined[np.isfinite(pct_combined)]
        if pct_combined.size == 0:
            continue
        all_lc_records.append({
            'scenario': sc_label,
            'quantile': int(q),
            'mean':     float(np.nanmean(pct_combined)),
            'p25':      float(np.nanpercentile(pct_combined, 25)),
            'p75':      float(np.nanpercentile(pct_combined, 75)),
            'pct_all':  pct_combined,
        })


# ── Plot — 5 LC subplots + 1 all-Europe panel ─────────────────────────────────

def _plot_panel(ax, recs_for_panel, sc_labels, sc_colors, rng, title, title_color='black'):
    for sc_label, sc_col in zip(sc_labels, sc_colors):
        recs = sorted(
            [r for r in recs_for_panel if r['scenario'] == sc_label],
            key=lambda r: r['quantile'],
        )
        if not recs:
            continue

        for r in recs:
            n      = r['pct_all'].size
            jitter = rng.uniform(-1.5, 1.5, n)
            ax.scatter(r['pct_all'], np.full(n, r['quantile']) + jitter,
                       color=sc_col, alpha=0.07, s=3, linewidths=0, rasterized=True)

        qs    = [r['quantile'] for r in recs]
        means = [r['mean']     for r in recs]
        p25s  = [r['p25']      for r in recs]
        p75s  = [r['p75']      for r in recs]
        ax.plot(means, qs, color=sc_col, lw=2, marker='o', markersize=5,
                zorder=5, label=sc_label)
        ax.fill_betweenx(qs, p25s, p75s, color=sc_col, alpha=0.12)

    ax.set_yticks(QUANTILES)
    ax.set_yticklabels([f"q{q}" for q in QUANTILES], fontsize=8)
    ax.set_title(title, fontsize=11, fontweight='bold', color=title_color)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=9)


rng = np.random.default_rng(42)
fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=True)
axes_flat = axes.flatten()

for ax, lc, lc_col in zip(axes_flat[:5], LC_TYPES, LC_COLORS):
    _plot_panel(ax, [r for r in records if r['lc'] == lc],
                SC_LABELS, SC_COLORS, rng, lc, title_color=lc_col)

# 6th panel: all Europe (excluding Barren)
_plot_panel(axes_flat[5], all_lc_records, SC_LABELS, SC_COLORS, rng,
            "All Europe (excl. Barren)", title_color='black')

# Shared axis labels
for ax in axes[1, :]:
    ax.set_xlabel("% of heavy-rainfall days within ExEvEs", fontsize=10)
for ax in axes[:, 0]:
    ax.set_ylabel("Precipitation quantile threshold", fontsize=10)

axes_flat[0].legend(fontsize=8, framealpha=0.8)
fig.suptitle("Heavy-rainfall days during ExEvEs — Europe (Apr–Oct), by land cover",
             fontsize=13, fontweight='bold', y=1.01)
fig.tight_layout()
fig.savefig(output_path("exeves_europe_prec_quantile_scatter_alllc.png"),
            dpi=150, bbox_inches='tight')
print("\nSaved exeves_europe_prec_quantile_scatter_alllc.png")
