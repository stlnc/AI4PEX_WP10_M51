"""
ExEvE frequency vs precipitation co-occurrence across ExEvE configs and quantiles.

For each of 10 (p_above, p_extreme) ExEvE configurations and each precipitation
quantile (q50–q95, wet days only):

  y = ExEvE frequency  — % of Apr–Oct days classified as ExEvE (per config)
  x = % of heavy-rain days (≥ q_n threshold) that fall within ExEvEs

Both GLEAM 2001–2020 and FLUXCOM 2001–2020 are shown.

Each ExEvE config produces a horizontal band on y (fixed frequency), with 10 dots
varying in x as the precipitation quantile threshold changes (q50 → q95).
Stricter ExEvE configs → lower y (fewer ExEvE days).

Plot (6 panels): config mean lines coloured by config (plasma)
  circles / solid lines   = GLEAM 2001–2020
  triangles / dashed lines = FLUXCOM 2001–2020
  x-position encodes precipitation quantile (q50 … q95, labelled on current config)
Output: exeves_europe_freq_vs_prec_config_alllc.png
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
QUANTILES     = np.arange(50, 96, 5)   # 50, 55, …, 95

# 10 ExEvE configs sorted by p_extreme (primary), p_above (secondary)
EXEVE_CONFIGS = [
    (75, 90), (80, 90), (85, 90),
    (75, 95), (80, 95), (85, 95), (90, 95),
    (75, 99), (80, 99), (85, 99),
]
CONFIG_CURRENT = (80, 95)
CONFIG_LABELS  = [f"{a}|{x}" for a, x in EXEVE_CONFIGS]
N_CONFIGS      = len(EXEVE_CONFIGS)

CMAP       = cm.plasma
CFG_COLORS = [CMAP(i / (N_CONFIGS - 1)) for i in range(N_CONFIGS)]

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


# ── Load MSWEP — wet-day quantile thresholds ──────────────────────────────────

print("Loading MSWEP ...")
ms_ds   = xr.open_dataset(input_path(MSWEP_FILE))
ms_eur  = ms_ds['mask'].sel(time=slice('2001-01-01', '2020-12-31'), lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
ms_full = ms_eur.where(ms_eur.time.dt.month.isin(SEASON_MONTHS),
                       drop=True).values.astype(np.float32)
ms_full[ms_full <= 0] = np.nan   # wet-day-only quantile basis

print(f"Computing {len(QUANTILES)} quantile thresholds (wet days only) ...")
q_grids = {int(q): np.nanpercentile(ms_full, q, axis=0) for q in QUANTILES}
q_valid = np.isfinite(q_grids[95])
del ms_full

ms_s1_da = select_period_apr_oct(ms_eur, '2001-01-01', '2020-12-31')
ms_s1    = ms_s1_da.values.astype(np.float32)
ms_s2    = ms_s1[t_s2_start:]
ms_ds.close()
del ms_s1


# ── Load FLUXCOM ─────────────────────────────────────────────────────────────

print("Loading FLUXCOM (dask-chunked) ...")
fc_ds    = xr.open_dataset(input_path(FLUXCOM_FILE), chunks={})
fc_land  = (fc_ds["land_fraction"].values >= 0.5)
fc_s3_da = select_period_apr_oct(fc_ds['ET'], '2001-01-01', '2020-12-31')
validate_alignment(gl_s1_da, ms_s1_da, fc_s3_da)
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
q_valid_c  = q_valid[i_lat, i_lon]
q_thresh_c = {q: q_grids[q][i_lat, i_lon] for q in QUANTILES}
print(f"Cells in bounds: {len(eur)}")

ms_cells = ms_s2[:, i_lat, i_lon]   # (T, N) — shared across both products


# ── Main computation ──────────────────────────────────────────────────────────
#
# records: one entry per (product, config, quantile, lc)
#   mean_freq — y-axis: mean ExEvE frequency (% of T days), same for all quantiles in a config
#   mean_pct  — x-axis: mean % of heavy-rain days within ExEvEs

records = []

for prod_label, et_grid, times_pd, is_fc in [
    ('GLEAM',   gl_s2, t_s2, False),
    ('FLUXCOM', fc_s3, t_s3, True),
]:
    print(f"\n── {prod_label} ──")
    et_cells = et_grid[:, i_lat, i_lon]   # (T, N)
    T        = et_cells.shape[0]

    has_data = ~np.all(np.isnan(et_cells), axis=0)
    valid    = has_data & q_valid_c
    if is_fc:
        valid &= fc_land[i_lat, i_lon]

    lc_masks        = {lc: (eur['land_cover_short_class'] == lc).values & valid
                       for lc in LC_TYPES}
    lc_masks['All'] = valid

    # Precompute heavy-rain masks for all quantiles (fixed across ExEvE configs)
    heavy = {q: ms_cells >= q_thresh_c[q][np.newaxis, :]   # (T, N) bool
             for q in QUANTILES}
    n_h   = {q: heavy[q].sum(axis=0).astype(np.float32) for q in QUANTILES}

    print("  Computing z-scores ...")
    z = pentad_zscore(et_cells, times_pd)   # (T, N) float32

    for cfg_idx, (p_above, p_extreme) in enumerate(EXEVE_CONFIGS):
        lbl = CONFIG_LABELS[cfg_idx]
        print(f"  Config ({p_above}, {p_extreme}) ...")

        exeve = detect_from_z_parallel(z, p_above, p_extreme)   # (T, N) bool

        # y: ExEvE frequency (% of T days) — same for all quantiles
        freq_cells = np.where(valid, exeve.sum(axis=0).astype(np.float32) / T * 100, np.nan)

        for q in QUANTILES:
            n_hw     = (heavy[q] & exeve).sum(axis=0).astype(np.float32)
            pct_cells = np.where(valid & (n_h[q] > 0), n_hw / n_h[q] * 100, np.nan)

            for lc_key, lc_mask in lc_masks.items():
                ok_cells = lc_mask & np.isfinite(freq_cells) & np.isfinite(pct_cells)
                freq_v   = freq_cells[ok_cells]
                pct_v    = pct_cells[ok_cells]
                if freq_v.size == 0:
                    continue
                records.append({
                    'product':      prod_label,
                    'config':       (p_above, p_extreme),
                    'config_label': lbl,
                    'config_idx':   cfg_idx,
                    'quantile':     int(q),
                    'lc':           lc_key,
                    'mean_freq':    float(np.nanmean(freq_v)),
                    'mean_pct':     float(np.nanmean(pct_v)),
                    'p25_pct':      float(np.nanpercentile(pct_v, 25)),
                    'p75_pct':      float(np.nanpercentile(pct_v, 75)),
                    'p25_freq':     float(np.nanpercentile(freq_v, 25)),
                    'p75_freq':     float(np.nanpercentile(freq_v, 75)),
                    'n':            freq_v.size,
                })

        del exeve

    del et_cells, z

del ms_cells, heavy, n_h


# ── Plot ──────────────────────────────────────────────────────────────────────

PANEL_ORDER  = LC_TYPES + ['All']
PANEL_TITLES = LC_TYPES + ['All Europe (excl. Barren)']
PANEL_COLORS = LC_COLORS + ['#333333']

# Quantile label positions on x-axis (annotated on current config only)
Q_LABEL_QUANTILES = [50, 75, 95]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes_flat  = axes.flatten()

for ax, lc_key, title, title_col in zip(
        axes_flat, PANEL_ORDER, PANEL_TITLES, PANEL_COLORS):

    recs = [r for r in records if r['lc'] == lc_key]

    for cfg_idx, (p_above, p_extreme) in enumerate(EXEVE_CONFIGS):
        col        = CFG_COLORS[cfg_idx]
        lbl        = CONFIG_LABELS[cfg_idx]
        is_current = (p_above, p_extreme) == CONFIG_CURRENT
        lw         = 2.2 if is_current else 1.2

        for prod_label, marker, linestyle in [
            ('GLEAM',   'o', '-'),
            ('FLUXCOM', '^', '--'),
        ]:
            cfg_recs = sorted(
                [r for r in recs
                 if r['config'] == (p_above, p_extreme) and r['product'] == prod_label],
                key=lambda r: r['quantile'],
            )
            if not cfg_recs:
                continue

            xs = [r['mean_pct']  for r in cfg_recs]
            ys = [r['mean_freq'] for r in cfg_recs]
            qs = [r['quantile']  for r in cfg_recs]

            ax.plot(xs, ys, color=col, lw=lw, linestyle=linestyle,
                    alpha=0.85, zorder=4)
            ax.scatter(xs, ys, color=col, s=5**2, marker=marker,
                       edgecolors='black', linewidths=0.3, zorder=5)

            # Annotate quantile labels on the current config only, GLEAM only
            if is_current and prod_label == 'GLEAM':
                for x, y, q in zip(xs, ys, qs):
                    if q in Q_LABEL_QUANTILES:
                        ax.annotate(f"q{q}", xy=(x, y),
                                    xytext=(0, 5), textcoords='offset points',
                                    fontsize=6, color=col, ha='center',
                                    bbox=dict(boxstyle='round,pad=0.1',
                                              fc='white', ec='none', alpha=0.7))

        # Config label at the GLEAM q95 mean point
        ref = next((r for r in recs
                    if r['config'] == (p_above, p_extreme)
                    and r['product'] == 'GLEAM'
                    and r['quantile'] == 95), None)
        if ref:
            ax.annotate(lbl,
                        xy=(ref['mean_pct'], ref['mean_freq']),
                        xytext=(4, 0), textcoords='offset points',
                        fontsize=5.5, color=col, va='center',
                        bbox=dict(boxstyle='round,pad=0.1',
                                  fc='white', ec='none', alpha=0.6))

    ax.set_title(title, fontsize=11, fontweight='bold', color=title_col)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=9)

for ax in axes[1, :]:
    ax.set_xlabel("% of heavy-rain days falling within ExEvEs", fontsize=10)
for ax in axes[:, 0]:
    ax.set_ylabel("ExEvE frequency — % of Apr–Oct days", fontsize=10)

# Legend
legend_handles = [
    Line2D([0], [0], color='#555555', lw=1.5, linestyle='-',  marker='o',
           markersize=5, label='GLEAM 2001–2020'),
    Line2D([0], [0], color='#555555', lw=1.5, linestyle='--', marker='^',
           markersize=5, label='FLUXCOM 2001–2020'),
]
axes_flat[0].legend(handles=legend_handles, fontsize=8,
                    loc='upper right', framealpha=0.85)

# Colorbar for ExEvE config progression
sm  = plt.cm.ScalarMappable(cmap=CMAP,
                              norm=mcolors.Normalize(vmin=0, vmax=N_CONFIGS - 1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, orientation='vertical',
                    fraction=0.012, pad=0.02, aspect=40)
cbar.set_ticks(range(N_CONFIGS))
cbar.set_ticklabels(CONFIG_LABELS, fontsize=7)
cbar.set_label("ExEvE config — p_above | p_extreme", fontsize=8)

fig.suptitle(
    "ExEvE frequency vs heavy-rainfall co-occurrence\n"
    "across ExEvE configs and precipitation quantiles — Europe (Apr–Oct)",
    fontsize=12, fontweight='bold', y=1.02,
)
fig.tight_layout()
fig.savefig(output_path("exeves_europe_freq_vs_prec_config_alllc.png"),
            dpi=150, bbox_inches='tight')
print("\nSaved exeves_europe_freq_vs_prec_config_alllc.png")
