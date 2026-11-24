"""
ExEvE detection threshold sensitivity: GLEAM–FLUXCOM agreement (Jaccard) vs
ExEvE frequency (% of Apr–Oct days classified as ExEvE).

For each of 10 (p_above, p_extreme) configurations and each cell:
  x = unconditional Jaccard(ExEvE_GLEAM, ExEvE_FLUXCOM)  over all Apr–Oct days
  y = ExEvE frequency (% of days) — shown separately for GLEAM and FLUXCOM

Stricter configs → fewer ExEvE days (lower y).
The plot reveals whether stricter definitions improve cross-product agreement.

Plot (6 panels): per-cell scatter coloured by config + bold mean marker per config
  circles = GLEAM 2001–2020   triangles = FLUXCOM 2001–2020
Output: exeves_europe_jaccard_vs_freq_alllc.png
"""
from _analysis import pentad_zscore
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
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
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

SEASON_MONTHS = [4, 5, 6, 7, 8, 9, 10]

# 10 configs sorted by p_extreme (primary), p_above (secondary)
EXEVE_CONFIGS = [
    (75, 90), (80, 90), (85, 90),
    (75, 95), (80, 95), (85, 95), (90, 95),
    (75, 99), (80, 99), (85, 99),
]
CONFIG_CURRENT = (80, 95)
CONFIG_LABELS  = [f"{a}|{x}" for a, x in EXEVE_CONFIGS]
N_CONFIGS      = len(EXEVE_CONFIGS)

# Sequential colormap across configs (permissive → strict)
CMAP        = cm.plasma
CFG_COLORS  = [CMAP(i / (N_CONFIGS - 1)) for i in range(N_CONFIGS)]

N_WORKERS = min(8, os.cpu_count() or 4)
print(f"Using {N_WORKERS} parallel workers")


# ── Detection ─────────────────────────────────────────────────────────────────


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

t_s2_start = t_s1.searchsorted(pd.Timestamp('2001-01-01'))
gl_s2 = gl_s1[t_s2_start:]
t_s2  = t_s1[t_s2_start:]
del gl_s1


# ── Load MSWEP — validity mask only (no precipitation threshold needed) ───────

print("Loading MSWEP (validity mask) ...")
ms_ds   = xr.open_dataset(input_path(MSWEP_FILE))
ms_eur  = ms_ds['mask'].sel(time=slice('2001-01-01', '2020-12-31'), lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
ms_s1_da = select_period_apr_oct(ms_eur, '2001-01-01', '2020-12-31')
ms_full = ms_s1_da.values.astype(np.float32)
ms_full[ms_full <= 0] = np.nan
q95_valid = np.isfinite(np.nanpercentile(ms_full, 95, axis=0))
del ms_full
ms_ds.close()


# ── Load FLUXCOM ─────────────────────────────────────────────────────────────

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
print(f"Cells in bounds: {len(eur)}")

gl_cells = gl_s2[:, i_lat, i_lon]   # (T, N)
fc_cells = fc_s3[:, i_lat, i_lon]   # (T, N)
T        = gl_cells.shape[0]

valid_gl = ~np.all(np.isnan(gl_cells), axis=0)
valid_fc = ~np.all(np.isnan(fc_cells), axis=0) & fc_land[i_lat, i_lon]
valid    = valid_gl & valid_fc & q95_valid[i_lat, i_lon]

lc_masks        = {lc: (eur['land_cover_short_class'] == lc).values & valid
                   for lc in LC_TYPES}
lc_masks['All'] = valid


# ── Compute z-scores once per product ────────────────────────────────────────

print("\nComputing z-scores — GLEAM ...")
z_gl = pentad_zscore(gl_cells, t_s2)

print("Computing z-scores — FLUXCOM ...")
z_fc = pentad_zscore(fc_cells, t_s3)

del gl_cells, fc_cells


# ── Compute Jaccard + ExEvE frequency for each config ────────────────────────

records = []

for cfg_idx, (p_above, p_extreme) in enumerate(EXEVE_CONFIGS):
    lbl = CONFIG_LABELS[cfg_idx]
    print(f"\nConfig ({p_above}, {p_extreme}) ...")

    exeve_gl = detect_from_z_parallel(z_gl, p_above, p_extreme)   # (T, N)
    exeve_fc = detect_from_z_parallel(z_fc, p_above, p_extreme)   # (T, N)

    # Unconditional Jaccard over all Apr–Oct days
    inter = (exeve_gl & exeve_fc).sum(axis=0).astype(np.float32)
    union = (exeve_gl | exeve_fc).sum(axis=0).astype(np.float32)
    jac   = np.where(valid & (union > 0), inter / union, np.nan)

    # ExEvE frequency per product (% of T days)
    freq_gl = np.where(valid, exeve_gl.sum(axis=0).astype(np.float32) / T * 100, np.nan)
    freq_fc = np.where(valid, exeve_fc.sum(axis=0).astype(np.float32) / T * 100, np.nan)

    del exeve_gl, exeve_fc

    for lc_key, lc_mask in lc_masks.items():
        both_ok   = lc_mask & np.isfinite(jac) & np.isfinite(freq_gl) & np.isfinite(freq_fc)
        jac_vals  = jac[both_ok]
        fgl_vals  = freq_gl[both_ok]
        ffc_vals  = freq_fc[both_ok]
        if jac_vals.size == 0:
            continue
        records.append({
            'config':        (p_above, p_extreme),
            'config_label':  lbl,
            'lc':            lc_key,
            'jac':           jac_vals,
            'freq_gl':       fgl_vals,
            'freq_fc':       ffc_vals,
            'mean_jac':      float(np.nanmean(jac_vals)),
            'mean_freq_gl':  float(np.nanmean(fgl_vals)),
            'mean_freq_fc':  float(np.nanmean(ffc_vals)),
            'p25_jac':       float(np.nanpercentile(jac_vals, 25)),
            'p75_jac':       float(np.nanpercentile(jac_vals, 75)),
            'p25_freq_gl':   float(np.nanpercentile(fgl_vals, 25)),
            'p75_freq_gl':   float(np.nanpercentile(fgl_vals, 75)),
            'p25_freq_fc':   float(np.nanpercentile(ffc_vals, 25)),
            'p75_freq_fc':   float(np.nanpercentile(ffc_vals, 75)),
            'n':             jac_vals.size,
        })

    all_r = next((r for r in records
                  if r['lc'] == 'All' and r['config'] == (p_above, p_extreme)), None)
    if all_r:
        print(f"  All-Europe  Jaccard={all_r['mean_jac']:.3f}  "
              f"freq_gl={all_r['mean_freq_gl']:.1f}%  "
              f"freq_fc={all_r['mean_freq_fc']:.1f}%  n={all_r['n']}")

del z_gl, z_fc


# ── Plot ──────────────────────────────────────────────────────────────────────

PANEL_ORDER  = LC_TYPES + ['All']
PANEL_TITLES = LC_TYPES + ['All Europe (excl. Barren)']
PANEL_COLORS = LC_COLORS + ['#333333']

rng = np.random.default_rng(42)
fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharex=True)
axes_flat  = axes.flatten()

for ax, lc_key, title, title_col in zip(
        axes_flat, PANEL_ORDER, PANEL_TITLES, PANEL_COLORS):

    recs = [r for r in records if r['lc'] == lc_key]

    for cfg_idx, (p_above, p_extreme) in enumerate(EXEVE_CONFIGS):
        r = next((r for r in recs if r['config'] == (p_above, p_extreme)), None)
        if r is None:
            continue
        col        = CFG_COLORS[cfg_idx]
        is_current = (p_above, p_extreme) == CONFIG_CURRENT

        # Per-cell scatter: GLEAM circles, FLUXCOM triangles
        ax.scatter(r['jac'], r['freq_gl'],
                   color=col, alpha=0.06, s=3, linewidths=0,
                   marker='o', rasterized=True)
        ax.scatter(r['jac'], r['freq_fc'],
                   color=col, alpha=0.06, s=3, linewidths=0,
                   marker='^', rasterized=True)

        # IQR cross-hairs — GLEAM (solid), FLUXCOM (dashed)
        ax.plot([r['p25_jac'], r['p75_jac']],
                [r['mean_freq_gl'], r['mean_freq_gl']],
                color=col, lw=1.0, alpha=0.7)
        ax.plot([r['mean_jac'], r['mean_jac']],
                [r['p25_freq_gl'], r['p75_freq_gl']],
                color=col, lw=1.0, alpha=0.7)
        ax.plot([r['p25_jac'], r['p75_jac']],
                [r['mean_freq_fc'], r['mean_freq_fc']],
                color=col, lw=1.0, alpha=0.7, linestyle='--')
        ax.plot([r['mean_jac'], r['mean_jac']],
                [r['p25_freq_fc'], r['p75_freq_fc']],
                color=col, lw=1.0, alpha=0.7, linestyle='--')

        # Mean markers — GLEAM: circle/star; FLUXCOM: triangle/filled-plus
        gl_marker = '*' if is_current else 'o'
        fc_marker = 'P' if is_current else '^'
        msize_gl  = 10  if is_current else 7
        msize_fc  = 10  if is_current else 7

        ax.scatter(r['mean_jac'], r['mean_freq_gl'],
                   color=col, s=msize_gl**2, marker=gl_marker,
                   edgecolors='black', linewidths=0.5, zorder=6)
        ax.scatter(r['mean_jac'], r['mean_freq_fc'],
                   color=col, s=msize_fc**2, marker=fc_marker,
                   edgecolors='black', linewidths=0.5, zorder=6)

        # Label at GLEAM mean
        ax.annotate(r['config_label'],
                    xy=(r['mean_jac'], r['mean_freq_gl']),
                    xytext=(3, 3), textcoords='offset points',
                    fontsize=5.5, color=col,
                    bbox=dict(boxstyle='round,pad=0.15', fc='white',
                              ec='none', alpha=0.6))

    ax.set_title(title, fontsize=11, fontweight='bold', color=title_col)
    ax.set_xlim(-0.02, 1.02)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=9)

# Legend — product markers
legend_handles = [
    Line2D([0], [0], marker='o', color='#555555', linestyle='None',
           markersize=6, label='GLEAM 2001–2020'),
    Line2D([0], [0], marker='^', color='#555555', linestyle='None',
           markersize=6, label='FLUXCOM 2001–2020'),
]
axes_flat[0].legend(handles=legend_handles, fontsize=7,
                    loc='upper left', framealpha=0.85)

for ax in axes[1, :]:
    ax.set_xlabel("Jaccard index — GLEAM vs FLUXCOM, unconditional", fontsize=10)
for ax in axes[:, 0]:
    ax.set_ylabel("ExEvE frequency — % of Apr–Oct days", fontsize=10)

# Colorbar for config progression
sm  = plt.cm.ScalarMappable(cmap=CMAP,
                              norm=mcolors.Normalize(vmin=0, vmax=N_CONFIGS - 1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, orientation='vertical',
                    fraction=0.012, pad=0.02, aspect=40)
cbar.set_ticks(range(N_CONFIGS))
cbar.set_ticklabels(CONFIG_LABELS, fontsize=7)
cbar.set_label("ExEvE config — p_above | p_extreme", fontsize=8)

fig.suptitle(
    "GLEAM–FLUXCOM ExEvE agreement vs ExEvE frequency\n"
    "across detection threshold configurations — Europe (Apr–Oct)",
    fontsize=12, fontweight='bold', y=1.02,
)
fig.tight_layout()
fig.savefig(output_path("exeves_europe_jaccard_vs_freq_alllc.png"),
            dpi=150, bbox_inches='tight')
print("\nSaved exeves_europe_jaccard_vs_freq_alllc.png")
