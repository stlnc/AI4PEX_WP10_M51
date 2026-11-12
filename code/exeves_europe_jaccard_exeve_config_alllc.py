"""
GLEAM–FLUXCOM ExEvE agreement across ExEvE detection threshold configurations.
Precipitation is fixed at q95 (wet days).  The Jaccard index is computed per
cell between GLEAM and FLUXCOM ExEvE masks restricted to q95 heavy-rain days,
for each of 10 (p_above, p_extreme) combinations.

  Jaccard = |ExEvE_GLEAM ∩ ExEvE_FLUXCOM ∩ heavy_q95|
          / |(ExEvE_GLEAM ∪ ExEvE_FLUXCOM) ∩ heavy_q95|

Plot (6 panels): y = ExEvE config  |  x = Jaccard index
Output: exeves_europe_jaccard_exeve_config_alllc.png

Performance
───────────
• z-scores computed ONCE per product (GLEAM, FLUXCOM), then _z_shared is
  swapped between them so each of the 10 configs reuses the precomputed arrays.
• Run-detection parallelised via ProcessPoolExecutor with fork (CoW).
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

SEASON_MONTHS = [4, 5, 6, 7, 8, 9, 10]
PREC_Q        = 95

# 10 configs sorted by p_extreme (primary), p_above (secondary)
EXEVE_CONFIGS = [
    (75, 90), (80, 90), (85, 90),
    (75, 95), (80, 95), (85, 95), (90, 95),
    (75, 99), (80, 99), (85, 99),
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


_z_shared = None   # inherited by forked workers via CoW


def _detect_z_chunk(args):
    start, end, p_above, p_extreme = args
    z_chunk = _z_shared[:, start:end]
    T, M    = z_chunk.shape
    z_ab    = np.nanpercentile(z_chunk, p_above,   axis=0)
    z_ex    = np.nanpercentile(z_chunk, p_extreme, axis=0)
    above   = z_chunk >= z_ab
    extreme = z_chunk >= z_ex
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


def detect_from_z_parallel(z, p_above, p_extreme):
    global _z_shared
    _z_shared  = z
    T, N       = z.shape
    chunk_size = max(1, (N + N_WORKERS - 1) // N_WORKERS)
    args = [(i, min(i + chunk_size, N), p_above, p_extreme)
            for i in range(0, N, chunk_size)]
    with ProcessPoolExecutor(max_workers=N_WORKERS) as exe:
        parts = list(exe.map(_detect_z_chunk, args))
    return np.concatenate(parts, axis=1)


def jaccard_conditional(exeve_a, exeve_b, heavy):
    """Per-cell Jaccard restricted to heavy-rainfall days. Returns (N,) float."""
    inter = (exeve_a & exeve_b & heavy).sum(axis=0).astype(np.float32)
    union = ((exeve_a | exeve_b) & heavy).sum(axis=0).astype(np.float32)
    return np.where(union > 0, inter / union, np.nan)


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


# ── Load GLEAM (S2: 2001–2020) ────────────────────────────────────────────────

print("Loading GLEAM ...")
gl_ds    = xr.open_dataset(GLEAM_FILE)
gl_ref   = gl_ds['E'].sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
eur_lats = gl_ref.lat.values
eur_lons = gl_ref.lon.values
gl_s1_da = select_period_apr_oct(gl_ref, '1980-01-01', '2020-12-31')
gl_s1    = gl_s1_da.values.astype(np.float32)
t_s1     = pd.DatetimeIndex(gl_s1_da.time.values)
gl_ds.close()

t_s2_start = t_s1.searchsorted(pd.Timestamp('2001-01-01'))
gl_s2 = gl_s1[t_s2_start:]
t_s2  = t_s1[t_s2_start:]
del gl_s1


# ── Load MSWEP — q95 wet-day threshold ───────────────────────────────────────

print("Loading MSWEP ...")
ms_ds   = xr.open_dataset(MSWEP_FILE)
ms_eur  = ms_ds['mask'].sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
ms_full = ms_eur.where(ms_eur.time.dt.month.isin(SEASON_MONTHS),
                       drop=True).values.astype(np.float32)
ms_full[ms_full <= 0] = np.nan

print(f"Computing q{PREC_Q} threshold (wet days only) ...")
q95_grid  = np.nanpercentile(ms_full, PREC_Q, axis=0)
q95_valid = np.isfinite(q95_grid)
del ms_full

ms_s1_da  = select_period_apr_oct(ms_eur, '1980-01-01', '2020-12-31')
ms_s1     = ms_s1_da.values.astype(np.float32)
ms_s2     = ms_s1[t_s2_start:]
ms_ds.close()
del ms_s1


# ── Load FLUXCOM (S3: 2001–2020) ─────────────────────────────────────────────

print("Loading FLUXCOM (dask-chunked) ...")
fc_ds    = xr.open_dataset(FLUXCOM_FILE, chunks={'time': 100})
fc_land  = (fc_ds["land_fraction"].values >= 0.5)
fc_s3_da = select_period_apr_oct(fc_ds['ET'], '2001-01-01', '2020-12-31')
t_s3     = pd.DatetimeIndex(fc_s3_da.time.values)
fc_s3    = (fc_s3_da.compute().values * 24).astype(np.float32)
fc_ds.close()

print(f"S2: {gl_s2.shape}  S3: {fc_s3.shape}  MS: {ms_s2.shape} (Apr–Oct only)")


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
q95_valid_c  = q95_valid[i_lat, i_lon]
q95_thresh_c = q95_grid[i_lat, i_lon]
print(f"Cells in bounds: {len(eur)}")

gl_cells = gl_s2[:, i_lat, i_lon]
fc_cells = fc_s3[:, i_lat, i_lon]
ms_cells = ms_s2[:, i_lat, i_lon]

valid_gl = ~np.all(np.isnan(gl_cells), axis=0)
valid_fc = ~np.all(np.isnan(fc_cells), axis=0) & fc_land[i_lat, i_lon]
valid    = valid_gl & valid_fc & q95_valid_c

lc_masks        = {lc: (eur['land_cover_short_class'] == lc).values & valid
                   for lc in LC_TYPES}
lc_masks['All'] = valid

# q95 heavy-rain mask — fixed for all configs
heavy_q95 = ms_cells >= q95_thresh_c[np.newaxis, :]   # (T, N) bool
del ms_cells


# ── Compute z-scores ONCE per product ────────────────────────────────────────

print("\nComputing z-scores — GLEAM 2001–2020 ...")
z_gl = pentad_zscore(gl_cells, t_s2)   # (T, N) float32

print("Computing z-scores — FLUXCOM 2001–2020 ...")
z_fc = pentad_zscore(fc_cells, t_s3)   # (T, N) float32

del gl_cells, fc_cells


# ── Compute Jaccard for each ExEvE config ─────────────────────────────────────

records = []   # {config, config_label, lc, mean, p25, p75, jac_all, n}

for cfg_idx, (p_above, p_extreme) in enumerate(EXEVE_CONFIGS):
    lbl = CONFIG_LABELS[cfg_idx]
    print(f"\nConfig ({p_above}, {p_extreme}) ...")

    _z_shared = z_gl
    exeve_gl  = detect_from_z_parallel(z_gl, p_above, p_extreme)

    _z_shared = z_fc
    exeve_fc  = detect_from_z_parallel(z_fc, p_above, p_extreme)

    jac = jaccard_conditional(exeve_gl, exeve_fc, heavy_q95)   # (N,)
    del exeve_gl, exeve_fc

    for lc_key, lc_mask in lc_masks.items():
        vals = jac[lc_mask]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        records.append({
            'config':       (p_above, p_extreme),
            'config_label': lbl,
            'lc':           lc_key,
            'mean':         float(np.nanmean(vals)),
            'p25':          float(np.nanpercentile(vals, 25)),
            'p75':          float(np.nanpercentile(vals, 75)),
            'jac_all':      vals,
            'n':            vals.size,
        })

    all_r = next((r for r in records
                  if r['lc'] == 'All' and r['config'] == (p_above, p_extreme)), None)
    if all_r:
        print(f"  All-Europe  mean Jaccard = {all_r['mean']:.3f}  n={all_r['n']}")

del z_gl, z_fc, heavy_q95


# ── Plot ──────────────────────────────────────────────────────────────────────

PANEL_ORDER  = LC_TYPES + ['All']
PANEL_TITLES = LC_TYPES + ['All Europe (excl. Barren)']
PANEL_COLORS = LC_COLORS + ['#333333']
SINGLE_COLOR = '#5b5ea6'

Y_POS = np.arange(len(EXEVE_CONFIGS))
rng   = np.random.default_rng(42)

fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=True, sharex=True)
axes_flat  = axes.flatten()

for ax, lc_key, title, title_col in zip(
        axes_flat, PANEL_ORDER, PANEL_TITLES, PANEL_COLORS):

    cfg_map = {r['config']: r
               for r in records if r['lc'] == lc_key}

    means, p25s, p75s, jac_lists = [], [], [], []
    for cfg in EXEVE_CONFIGS:
        r = cfg_map.get(cfg)
        means.append(r['mean']    if r else np.nan)
        p25s.append(r['p25']     if r else np.nan)
        p75s.append(r['p75']     if r else np.nan)
        jac_lists.append(r['jac_all'] if r else np.array([]))

    # Raw per-cell scatter with y-jitter
    for y_pos, vals in zip(Y_POS, jac_lists):
        if vals.size == 0:
            continue
        jitter = rng.uniform(-0.3, 0.3, vals.size)
        ax.scatter(vals, np.full(vals.size, y_pos) + jitter,
                   color=SINGLE_COLOR, alpha=0.06, s=3, linewidths=0,
                   rasterized=True)

    # IQR band + mean line
    ax.fill_betweenx(Y_POS, p25s, p75s, color=SINGLE_COLOR, alpha=0.2)
    ax.plot(means, Y_POS, color=SINGLE_COLOR, lw=2.5, marker='o',
            markersize=6, zorder=5)

    # Mark current definition
    ref_y = EXEVE_CONFIGS.index(CONFIG_CURRENT)
    ax.axhline(ref_y, color='#666666', lw=0.8, linestyle=':', zorder=2)

    ax.set_yticks(Y_POS)
    ax.set_yticklabels(CONFIG_LABELS, fontsize=8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_title(title, fontsize=11, fontweight='bold', color=title_col)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=9)

for ax in axes[1, :]:
    ax.set_xlabel(
        f"Jaccard index  (GLEAM vs FLUXCOM ExEvEs on q{PREC_Q} heavy-rain days)",
        fontsize=9)
axes[0, 0].set_ylabel("ExEvE config  (p_above | p_extreme)", fontsize=9)
axes[1, 0].set_ylabel("ExEvE config  (p_above | p_extreme)", fontsize=9)

fig.suptitle(
    "GLEAM–FLUXCOM ExEvE agreement across detection threshold configurations\n"
    f"Europe (Apr–Oct), precipitation fixed at q{PREC_Q} (wet days)",
    fontsize=12, fontweight='bold', y=1.02,
)
fig.tight_layout()
fig.savefig("exeves_europe_jaccard_exeve_config_alllc.png",
            dpi=150, bbox_inches='tight')
print("\nSaved exeves_europe_jaccard_exeve_config_alllc.png")
