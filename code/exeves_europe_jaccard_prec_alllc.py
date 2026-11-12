"""
GLEAM–FLUXCOM ExEvE agreement as a function of precipitation intensity.

For each cell and precipitation quantile threshold q (q50–q95, wet days),
the Jaccard index measures agreement between GLEAM and FLUXCOM ExEvE masks
restricted to heavy-rainfall days (days where precip ≥ threshold_q):

  Jaccard(q) = |ExEvE_GLEAM ∩ ExEvE_FLUXCOM on heavy days|
             / |ExEvE_GLEAM ∪ ExEvE_FLUXCOM on heavy days|

Cells where the union is 0 (neither product flags any heavy-rain day as ExEvE)
are excluded from the distribution at that quantile level.

Plot (6 panels): y = precipitation quantile  |  x = Jaccard index
Output: exeves_europe_jaccard_prec_alllc.png
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
QUANTILES     = np.arange(50, 96, 5)   # 50, 55, …, 95

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


def detect_exeves(et, times_pd):
    T, N    = et.shape
    z       = pentad_zscore(et, times_pd)
    z80     = np.nanpercentile(z, 80, axis=0)
    z95     = np.nanpercentile(z, 95, axis=0)
    above   = z >= z80
    extreme = z >= z95
    del z
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
    return out


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


def jaccard_conditional(exeve_a, exeve_b, heavy):
    """
    Per-cell Jaccard of two ExEvE masks restricted to heavy-rainfall days.
    All inputs (T, N) bool.  Returns (N,) float; NaN where union == 0.
    """
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


# ── Load MSWEP — wet-day quantile thresholds ──────────────────────────────────

print("Loading MSWEP ...")
ms_ds   = xr.open_dataset(MSWEP_FILE)
ms_eur  = ms_ds['mask'].sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
ms_full = ms_eur.where(ms_eur.time.dt.month.isin(SEASON_MONTHS),
                       drop=True).values.astype(np.float32)
ms_full[ms_full <= 0] = np.nan   # wet-day-only quantile basis

print(f"Computing {len(QUANTILES)} quantile thresholds (wet days only) ...")
q_grids = {int(q): np.nanpercentile(ms_full, q, axis=0) for q in QUANTILES}
q_valid = np.isfinite(q_grids[95])
del ms_full

ms_s1_da = select_period_apr_oct(ms_eur, '1980-01-01', '2020-12-31')
ms_s1    = ms_s1_da.values.astype(np.float32)
ms_s2    = ms_s1[t_s2_start:]
ms_ds.close()
del ms_s1


# ── Load FLUXCOM (S3: 2001–2020) — dask-chunked ───────────────────────────────

print("Loading FLUXCOM (dask-chunked) ...")
fc_ds    = xr.open_dataset(FLUXCOM_FILE, chunks={'time': 100})
fc_land  = (fc_ds["land_fraction"].values >= 0.5)
fc_s3_da = select_period_apr_oct(fc_ds['ET'], '2001-01-01', '2020-12-31')
t_s3     = pd.DatetimeIndex(fc_s3_da.time.values)
fc_s3    = (fc_s3_da.compute().values * 24).astype(np.float32)
fc_ds.close()

print(f"S2: {gl_s2.shape}  S3: {fc_s3.shape}  MS: {ms_s2.shape} (Apr–Oct only)")
assert gl_s2.shape[0] == fc_s3.shape[0], \
    f"Time mismatch: GLEAM {gl_s2.shape[0]} vs FLUXCOM {fc_s3.shape[0]}"


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


# ── Detect ExEvEs for both products (2001–2020 only) ─────────────────────────

gl_cells = gl_s2[:, i_lat, i_lon]
fc_cells = fc_s3[:, i_lat, i_lon]
ms_cells = ms_s2[:, i_lat, i_lon]

print("\nDetecting ExEvEs — GLEAM 2001–2020 ...")
exeve_gl = detect_exeves_parallel(gl_cells, t_s2)   # (T, N) bool

print("Detecting ExEvEs — FLUXCOM 2001–2020 ...")
exeve_fc = detect_exeves_parallel(fc_cells, t_s3)   # (T, N) bool

# Valid cells: must have data in both products and valid MSWEP q95
valid_gl = ~np.all(np.isnan(gl_cells), axis=0)
valid_fc = ~np.all(np.isnan(fc_cells), axis=0) & fc_land[i_lat, i_lon]
valid    = valid_gl & valid_fc & q_valid_c

del gl_cells, fc_cells


# ── Compute conditional Jaccard per (LC × quantile) ──────────────────────────

lc_masks        = {lc: (eur['land_cover_short_class'] == lc).values & valid
                   for lc in LC_TYPES}
lc_masks['All'] = valid

records = []   # {lc, quantile, mean, p25, p75, jac_all}

for q in QUANTILES:
    thresh = q_thresh_c[q][np.newaxis, :]        # (1, N)
    heavy  = ms_cells >= thresh                   # (T, N) bool
    jac    = jaccard_conditional(exeve_gl, exeve_fc, heavy)  # (N,)

    for lc_key, lc_mask in lc_masks.items():
        vals = jac[lc_mask]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        records.append({
            'lc':       lc_key,
            'quantile': int(q),
            'mean':     float(np.nanmean(vals)),
            'p25':      float(np.nanpercentile(vals, 25)),
            'p75':      float(np.nanpercentile(vals, 75)),
            'jac_all':  vals,
            'n':        vals.size,
        })
    all_rec = next((r for r in records if r['lc'] == 'All' and r['quantile'] == int(q)), None)
    if all_rec:
        print(f"  q{q}  All-Europe mean Jaccard = {all_rec['mean']:.3f}  n={all_rec['n']}")

del ms_cells, heavy


# ── Plot ──────────────────────────────────────────────────────────────────────

PANEL_ORDER  = LC_TYPES + ['All']
PANEL_TITLES = LC_TYPES + ['All Europe (excl. Barren)']
PANEL_COLORS = LC_COLORS + ['#333333']
SINGLE_COLOR = '#5b5ea6'   # one comparison only: GLEAM vs FLUXCOM

rng = np.random.default_rng(42)
fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=True, sharex=True)
axes_flat  = axes.flatten()

for ax, lc_key, title, title_col in zip(
        axes_flat, PANEL_ORDER, PANEL_TITLES, PANEL_COLORS):

    recs = sorted([r for r in records if r['lc'] == lc_key],
                  key=lambda r: r['quantile'])
    if not recs:
        continue

    qs    = [r['quantile'] for r in recs]
    means = [r['mean']     for r in recs]
    p25s  = [r['p25']      for r in recs]
    p75s  = [r['p75']      for r in recs]

    # Raw per-cell Jaccard scatter (jitter y)
    for r in recs:
        n      = r['jac_all'].size
        jitter = rng.uniform(-1.5, 1.5, n)
        ax.scatter(r['jac_all'], np.full(n, r['quantile']) + jitter,
                   color=SINGLE_COLOR, alpha=0.06, s=3, linewidths=0,
                   rasterized=True)

    # IQR band + mean line
    ax.fill_betweenx(qs, p25s, p75s, color=SINGLE_COLOR, alpha=0.2)
    ax.plot(means, qs, color=SINGLE_COLOR, lw=2.5, marker='o',
            markersize=6, zorder=5)

    ax.set_yticks(QUANTILES)
    ax.set_yticklabels([f"q{q}" for q in QUANTILES], fontsize=8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_title(title, fontsize=11, fontweight='bold', color=title_col)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=9)

for ax in axes[1, :]:
    ax.set_xlabel("Jaccard index  (GLEAM vs FLUXCOM ExEvEs on heavy-rain days)",
                  fontsize=9)
for ax in axes[:, 0]:
    ax.set_ylabel("Precipitation quantile threshold", fontsize=10)

fig.suptitle(
    "GLEAM–FLUXCOM ExEvE agreement on heavy-rainfall days — Europe (Apr–Oct)\n"
    "Jaccard restricted to days ≥ precipitation quantile threshold (wet days)",
    fontsize=12, fontweight='bold', y=1.02,
)
fig.tight_layout()
fig.savefig("exeves_europe_jaccard_prec_alllc.png", dpi=150, bbox_inches='tight')
print("\nSaved exeves_europe_jaccard_prec_alllc.png")
