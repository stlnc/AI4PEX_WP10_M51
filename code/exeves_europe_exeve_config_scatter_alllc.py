"""
Sensitivity of ExEvE–heavy-rainfall co-occurrence to the ExEvE detection
thresholds.  Precipitation is fixed at q95 (wet days).  The ExEvE definition
is varied across 10 logical combinations of:
  p_above   – minimum z-score percentile for a day to count as "elevated ET"
  p_extreme – minimum z-score percentile for at least one day in the event

Current definition: p_above=80, p_extreme=95  (marked with ★ on y-axis)

Plot (6 panels): same layout as exeves_europe_prec_quantile_scatter_alllc.py
  y-axis : ExEvE config label  (p_above | p_extreme)
  x-axis : % of q95 precipitation days falling within ExEvEs
Output  : exeves_europe_exeve_config_scatter_alllc.png

Performance
───────────
• z-scores are computed ONCE per scenario (most expensive step), then reused
  across all 10 configs — avoids 10× redundant pentad_zscore calls.
• Run-detection (inner pixel loop) is parallelised via ProcessPoolExecutor
  with fork: workers inherit the z-score array via CoW so only tiny
  (start, end, p_above, p_extreme) tuples are serialised.
"""
import os
import multiprocessing
import warnings
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import pyreadr
from concurrent.futures import ProcessPoolExecutor

multiprocessing.set_start_method('fork', force=True)
warnings.filterwarnings('ignore')

GLEAM_FILE   = "gleam_e_mm_land_198001_202412_025_daily.nc"
FLUXCOM_FILE = "ET_2001_2020_025_daily_eur.nc"
MSWEP_FILE   = "mswep-v2-8_tp_mm_land_197901_202012_025_daily.nc"
MASKS_FILE   = "masks.rda"

LC_TYPES  = ["Forests", "Croplands", "Savannas", "Grasslands", "Shrublands"]
LC_COLORS = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#D55E00"]

SC_KEYS   = ['S2', 'S3']
SC_LABELS = ["GLEAM 2001–2020", "FLUXCOM 2001–2020"]
SC_COLORS = ['#ff7f0e', '#2ca02c']

SEASON_MONTHS = [4, 5, 6, 7, 8, 9, 10]
PREC_Q        = 95   # fixed precipitation quantile

# 10 ExEvE threshold combinations (p_above, p_extreme)
# sorted by p_extreme (primary), then p_above (secondary)
EXEVE_CONFIGS = [
    (75, 90),
    (80, 90),
    (85, 90),
    (75, 95),
    (80, 95),   # ← current definition
    (85, 95),
    (90, 95),
    (75, 99),
    (80, 99),
    (85, 99),
]
CONFIG_CURRENT = (80, 95)
CONFIG_LABELS  = [f"{a} | {x}" for a, x in EXEVE_CONFIGS]

N_WORKERS = min(8, os.cpu_count() or 4)
print(f"Using {N_WORKERS} parallel workers")


# ── Detection ─────────────────────────────────────────────────────────────────

def pentad_zscore(et, times_pd):
    T, N   = et.shape
    pentad = np.minimum((times_pd.dayofyear.values - 1) // 5, 72)
    z      = np.full_like(et, np.nan, dtype=np.float32)
    for p in range(73):
        idx = pentad == p
        if not idx.any():
            continue
        v      = et[idx]
        mu     = np.nanmean(v, axis=0)
        sg     = np.nanstd(v, axis=0)
        z[idx] = (v - mu) / np.where(sg > 0, sg, np.nan)
    return z


# Module-level global so forked workers inherit it via CoW — no serialisation.
_z_shared = None


def _detect_z_chunk(args):
    """Worker: detects ExEvEs in a cell-slice using the inherited z-score array."""
    start, end, p_above, p_extreme = args
    z_chunk = _z_shared[:, start:end]
    T, M    = z_chunk.shape

    z_above_th   = np.nanpercentile(z_chunk, p_above,   axis=0)
    z_extreme_th = np.nanpercentile(z_chunk, p_extreme, axis=0)
    above   = z_chunk >= z_above_th
    extreme = z_chunk >= z_extreme_th

    out = np.zeros((T, M), dtype=bool)
    for j in range(M):
        if not above[:, j].any():
            continue
        d  = np.diff(above[:, j].astype(np.int8), prepend=0, append=0)
        ss = np.where(d ==  1)[0]
        es = np.where(d == -1)[0]
        for s, e in zip(ss, es):
            if (e - s) >= 2 and extreme[s:e, j].any():
                out[s:e, j] = True
    return out


def detect_exeves_config_parallel(z, p_above, p_extreme):
    """Detect ExEvEs for one threshold config, splitting cells across workers."""
    global _z_shared
    _z_shared  = z
    T, N       = z.shape
    chunk_size = max(1, (N + N_WORKERS - 1) // N_WORKERS)
    args = [
        (i, min(i + chunk_size, N), p_above, p_extreme)
        for i in range(0, N, chunk_size)
    ]
    with ProcessPoolExecutor(max_workers=N_WORKERS) as exe:
        parts = list(exe.map(_detect_z_chunk, args))
    return np.concatenate(parts, axis=1)


# ── Load masks ────────────────────────────────────────────────────────────────

print("Loading masks ...")
masks = pyreadr.read_r(MASKS_FILE)['masks']
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
gl_ds    = xr.open_dataset(GLEAM_FILE)
gl_ref   = gl_ds['E'].sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
eur_lats = gl_ref.lat.values
eur_lons = gl_ref.lon.values
gl_s1_da = select_period_apr_oct(gl_ref, '1980-01-01', '2020-12-31')
gl_s1    = gl_s1_da.values.astype(np.float32)
t_s1     = pd.DatetimeIndex(gl_s1_da.time.values)
gl_ds.close()


# ── Load MSWEP — q95 wet-day threshold ───────────────────────────────────────

print("Loading MSWEP ...")
ms_ds   = xr.open_dataset(MSWEP_FILE)
ms_eur  = ms_ds['mask'].sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
ms_full = ms_eur.where(ms_eur.time.dt.month.isin(SEASON_MONTHS),
                       drop=True).values.astype(np.float32)
ms_full[ms_full <= 0] = np.nan   # wet-day-only quantile basis

print(f"Computing q{PREC_Q} precipitation threshold (wet days only) ...")
q95_grid  = np.nanpercentile(ms_full, PREC_Q, axis=0)
q95_valid = np.isfinite(q95_grid)
del ms_full

ms_s1_da = select_period_apr_oct(ms_eur, '1980-01-01', '2020-12-31')
ms_s1    = ms_s1_da.values.astype(np.float32)
ms_ds.close()

t_s2_start = t_s1.searchsorted(pd.Timestamp('2001-01-01'))
gl_s2 = gl_s1[t_s2_start:]
ms_s2 = ms_s1[t_s2_start:]
t_s2  = t_s1[t_s2_start:]


# ── Load FLUXCOM — dask-chunked parallel I/O ──────────────────────────────────

print("Loading FLUXCOM (dask-chunked) ...")
fc_ds    = xr.open_dataset(FLUXCOM_FILE, chunks={'time': 100})
fc_land  = (fc_ds["land_fraction"].values >= 0.5)
fc_s3_da = select_period_apr_oct(fc_ds['ET'], '2001-01-01', '2020-12-31')
t_s3     = pd.DatetimeIndex(fc_s3_da.time.values)
fc_s3    = (fc_s3_da.compute().values * 24).astype(np.float32)
fc_ds.close()

print(f"S1: {gl_s1.shape}  S2: {gl_s2.shape}  S3: {fc_s3.shape} (Apr–Oct only)")


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
q95_valid_c = q95_valid[i_lat, i_lon]
q95_thresh_c = q95_grid[i_lat, i_lon]          # (N,) per-cell q95 threshold
print(f"Cells in bounds: {len(eur)}")


# ── Main computation ──────────────────────────────────────────────────────────
#
# For each scenario:
#   1. Compute z-scores ONCE  (expensive — reused across all 10 configs)
#   2. For each ExEvE config: detect ExEvEs in parallel, then tally q95-rain overlap

records = []   # {scenario, lc, config, config_label, mean, p25, p75, pct_all}

for sc_label, sc_key, et_grid, times_pd, ms_grid in [
    (SC_LABELS[0], 'S2', gl_s2, t_s2, ms_s2),
    (SC_LABELS[1], 'S3', fc_s3, t_s3, ms_s2),
]:
    print(f"\n── {sc_key} ({et_grid.shape[0]} days) ──")
    et_cells = et_grid[:, i_lat, i_lon]    # (T, N)
    ms_cells = ms_grid[:, i_lat, i_lon]    # (T, N)

    # z-scores computed once and shared with all config workers via fork CoW
    print("  Computing z-scores ...")
    z = pentad_zscore(et_cells, times_pd)  # (T, N) float32

    has_data = ~np.all(np.isnan(et_cells), axis=0)
    valid    = has_data & q95_valid_c
    if sc_key == 'S3':
        valid &= fc_land[i_lat, i_lon]

    # q95 precipitation mask — fixed for all ExEvE configs
    heavy_q95 = ms_cells >= q95_thresh_c[np.newaxis, :]   # (T, N) bool
    n_h_all   = heavy_q95.sum(axis=0).astype(np.float32)   # (N,)

    lc_masks         = {lc: (eur['land_cover_short_class'] == lc).values & valid
                        for lc in LC_TYPES}
    lc_masks['All']  = valid

    for cfg_idx, (p_above, p_extreme) in enumerate(EXEVE_CONFIGS):
        lbl = CONFIG_LABELS[cfg_idx]
        print(f"  Config ({p_above}, {p_extreme}) — parallel detection ...")

        exeve = detect_exeves_config_parallel(z, p_above, p_extreme)  # (T, N)

        n_hw_all = (heavy_q95 & exeve).sum(axis=0).astype(np.float32)  # (N,)
        pct_all  = np.where(valid & (n_h_all > 0),
                            n_hw_all / n_h_all * 100, np.nan)

        for lc_key, lc_mask in lc_masks.items():
            pct = pct_all[lc_mask]
            pct = pct[np.isfinite(pct)]
            if pct.size == 0:
                continue
            records.append({
                'scenario':     sc_label,
                'lc':           lc_key,
                'config':       (p_above, p_extreme),
                'config_label': lbl,
                'mean':         float(np.nanmean(pct)),
                'p25':          float(np.nanpercentile(pct, 25)),
                'p75':          float(np.nanpercentile(pct, 75)),
                'pct_all':      pct,
                'n':            pct.size,
            })
            if lc_key != 'All':
                print(f"    {lc_key:<12} mean={records[-1]['mean']:.1f}%  n={pct.size}")

        del exeve

    del et_cells, ms_cells, z, heavy_q95, n_h_all

del gl_s1, ms_s1, fc_s3


# ── Plot ──────────────────────────────────────────────────────────────────────

PANEL_ORDER  = LC_TYPES + ['All']
PANEL_TITLES = LC_TYPES + ['All Europe (excl. Barren)']
PANEL_COLORS = LC_COLORS + ['#333333']

rng = np.random.default_rng(42)
fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=True)
axes_flat  = axes.flatten()

# y positions for the 10 configs (0 … 9, bottom-to-top)
Y_POS = np.arange(len(EXEVE_CONFIGS))

for ax, lc_key, title, title_col in zip(
        axes_flat, PANEL_ORDER, PANEL_TITLES, PANEL_COLORS):

    recs = [r for r in records if r['lc'] == lc_key]

    for sc_label, sc_col in zip(SC_LABELS, SC_COLORS):
        sc_recs = [r for r in recs if r['scenario'] == sc_label]
        if not sc_recs:
            continue

        # Build ordered arrays aligned to EXEVE_CONFIGS
        cfg_map = {r['config']: r for r in sc_recs}
        means, p25s, p75s, pct_lists = [], [], [], []
        for cfg in EXEVE_CONFIGS:
            r = cfg_map.get(cfg)
            means.append(r['mean'] if r else np.nan)
            p25s.append(r['p25']   if r else np.nan)
            p75s.append(r['p75']   if r else np.nan)
            pct_lists.append(r['pct_all'] if r else np.array([]))

        # Raw cell scatter (jitter x for visibility)
        for y_pos, pct in zip(Y_POS, pct_lists):
            if pct.size == 0:
                continue
            jitter = rng.uniform(-0.3, 0.3, pct.size)
            ax.scatter(pct, np.full(pct.size, y_pos) + jitter,
                       color=sc_col, alpha=0.07, s=3, linewidths=0,
                       rasterized=True)

        # IQR band + mean line (horizontal — config on y-axis)
        ax.fill_betweenx(Y_POS, p25s, p75s, color=sc_col, alpha=0.15)
        ax.plot(means, Y_POS, color=sc_col, lw=2, marker='o',
                markersize=5, zorder=5, label=sc_label)

    # Mark the current (reference) definition with a dashed line
    ref_y = EXEVE_CONFIGS.index(CONFIG_CURRENT)
    ax.axhline(ref_y, color='#666666', lw=0.8, linestyle=':', zorder=2)

    ax.set_yticks(Y_POS)
    ax.set_yticklabels(CONFIG_LABELS, fontsize=8)
    ax.set_title(title, fontsize=11, fontweight='bold', color=title_col)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=9)

# Shared axis labels
for ax in axes[1, :]:
    ax.set_xlabel(f"% of q{PREC_Q} precipitation days within ExEvEs", fontsize=10)
axes[0, 0].set_ylabel("ExEvE config  (p_above | p_extreme)", fontsize=9)
axes[1, 0].set_ylabel("ExEvE config  (p_above | p_extreme)", fontsize=9)

axes_flat[0].legend(fontsize=8, framealpha=0.85)
fig.suptitle(
    f"Sensitivity of ExEvE–heavy-rainfall co-occurrence to detection thresholds\n"
    f"Europe (Apr–Oct), precipitation fixed at q{PREC_Q} (wet days)",
    fontsize=12, fontweight='bold', y=1.02,
)
fig.tight_layout()
fig.savefig("exeves_europe_exeve_config_scatter_alllc.png",
            dpi=150, bbox_inches='tight')
print("\nSaved exeves_europe_exeve_config_scatter_alllc.png")
