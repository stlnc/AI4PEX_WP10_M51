"""
Global ExEvEs analysis for a user-selected land cover type.
Optimised for machines with large RAM: sequential year-by-year I/O,
numpy fancy indexing in RAM, parallel ExEvE detection.

Outputs:
  exeves_<lc>_pct_q95_density.png
  exeves_<lc>_pct_q95_map.png

Set LC_CHOICE below:
  1 = Croplands  2 = Forests  3 = Savannas
  4 = Grasslands  5 = Shrublands  6 = Barren
"""

# ── User config ───────────────────────────────────────────────────────────────
LC_CHOICE  = 4
N_JOBS     = -1    # parallel workers for ExEvE detection (-1 = all cores)
CHUNK_SIZE = 4000  # cells per parallel worker
YEARS_PER_LOAD = 5 # years to load at once during I/O (tune for your RAM/storage)
IO_WORKERS = 4     # threads for parallel year-batch I/O (tune for your storage)
# ─────────────────────────────────────────────────────────────────────────────

import os, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import gaussian_kde
from joblib import Parallel, delayed
import pyreadr
import warnings; warnings.filterwarnings('ignore')

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

GLEAM_FILE     = "gleam_e_mm_land_198001_202412_025_daily.nc"
MSWEP_FILE     = "mswep-v2-8_tp_mm_land_197901_202012_025_daily.nc"
MASKS_FILE     = "masks.rda"
FLUXCOM_GLOBAL = "ET_2001_2020_025_daily.nc"

LC_MAP = {1: "Croplands", 2: "Forests", 3: "Savannas",
          4: "Grasslands", 5: "Shrublands", 6: "Barren"}
LC_TYPE = LC_MAP[LC_CHOICE]
LC_SLUG = LC_TYPE.lower()

SC_LABELS = ["GLEAM 1980–2020", "GLEAM 2001–2020", "FLUXCOM 2001–2020"]
SC_COLORS = ["steelblue", "seagreen", "darkorange"]


# ── Resource monitor ──────────────────────────────────────────────────────────

def log_resources(tag=""):
    if not HAS_PSUTIL:
        return
    proc = psutil.Process(os.getpid())
    vm   = psutil.virtual_memory()
    print(f"  [MEM {tag}]  process={proc.memory_info().rss/1e9:.1f} GB  |  "
          f"system={vm.used/1e9:.1f}/{vm.total/1e9:.1f} GB ({vm.percent:.0f}%)  |  "
          f"CPU={psutil.cpu_percent(interval=0.1):.0f}%")


# ── Fast cell loader (sequential year batches → numpy fancy index in RAM) ─────

def load_cells_fast(da, t0, t1, i_lat, i_lon, label=""):
    """
    Load the full spatial array one year-batch at a time (sequential I/O),
    then numpy fancy-index in RAM — avoids dask scattered reads.
    """
    all_times = pd.DatetimeIndex(da.sel(time=slice(t0, t1)).time.values)
    years = sorted(set(all_times.year))
    year_batches = [years[i:i+YEARS_PER_LOAD] for i in range(0, len(years), YEARS_PER_LOAD)]

    parts = []
    it = (tqdm(year_batches, desc=f"    {label}", unit="batch", ncols=72)
          if HAS_TQDM else year_batches)

    for batch in it:
        ts = f"{batch[0]}-01-01"
        te = f"{batch[-1]}-12-31"
        # Load full (T_batch, nlat, nlon) — one contiguous sequential read
        arr = da.sel(time=slice(ts, te)).values.astype(np.float32)
        parts.append(arr[:, i_lat, i_lon])   # numpy fancy index in RAM
        del arr                               # free spatial grid immediately
    return np.concatenate(parts, axis=0)     # (T, N)


# ── Detection (fully vectorised — no per-cell Python loop) ────────────────────

def pentad_zscore(et, times_pd):
    T, N = et.shape
    pentad = np.minimum((times_pd.dayofyear.values - 1) // 5, 72)
    z = np.full_like(et, np.nan, dtype=np.float32)
    for p in range(73):
        idx = pentad == p
        if not idx.any(): continue
        v  = et[idx]
        mu = np.nanmean(v, axis=0); sg = np.nanstd(v, axis=0)
        z[idx] = (v - mu) / np.where(sg > 0, sg, np.nan)
    return z


def detect_exeves(et, times_pd):
    """Vectorised via cumsum run-labeling + bincount — no per-cell loop."""
    T, N = et.shape
    z   = pentad_zscore(et, times_pd)
    z80 = np.nanpercentile(z, 80, axis=0)
    z95 = np.nanpercentile(z, 95, axis=0)
    above   = z >= z80[np.newaxis, :]
    extreme = z >= z95[np.newaxis, :]

    enter        = np.diff(above.view(np.int8), axis=0, prepend=0).clip(min=0)
    local_run_id = (np.cumsum(enter, axis=0) * above).astype(np.int32)
    max_per_cell = local_run_id.max(axis=0)
    cell_offset  = np.zeros(N, dtype=np.int32)
    np.cumsum(max_per_cell[:-1], out=cell_offset[1:])
    global_run_id = np.where(above,
                             local_run_id + cell_offset[np.newaxis, :],
                             0).astype(np.int32)

    n_runs      = int(global_run_id.max()) + 1
    flat_id     = global_run_id.ravel()
    flat_ext    = (above & extreme).ravel().view(np.uint8).astype(np.float32)
    run_len     = np.bincount(flat_id, minlength=n_runs)
    run_has_ext = np.bincount(flat_id, weights=flat_ext, minlength=n_runs) > 0
    valid_run   = (run_len >= 2) & run_has_ext
    valid_run[0] = False
    return valid_run[global_run_id]


def detect_parallel(et, times_pd):
    """Dispatch cell chunks to parallel workers with a tqdm progress bar."""
    N      = et.shape[1]
    slices = [(i, min(i + CHUNK_SIZE, N)) for i in range(0, N, CHUNK_SIZE)]
    jobs   = (delayed(detect_exeves)(et[:, s:e], times_pd) for s, e in slices)

    if HAS_TQDM:
        gen   = Parallel(n_jobs=N_JOBS, return_as="generator")(jobs)
        parts = list(tqdm(gen, total=len(slices),
                          desc="    detection chunks", unit="chunk", ncols=72))
    else:
        parts = Parallel(n_jobs=N_JOBS)(jobs)
    return np.concatenate(parts, axis=1)


# ── Load masks ────────────────────────────────────────────────────────────────

t_total = time.time()
print(f"Analysis: {LC_TYPE} — Global")
print("─" * 55)

print("Loading masks ...")
masks = pyreadr.read_r(MASKS_FILE)['masks']
cells = masks[
    (masks['land_mask'] == 'land') &
    (masks['land_cover_short_class'] == LC_TYPE)
].copy().reset_index(drop=True)
print(f"  {LC_TYPE} cells globally: {len(cells)}")


# ── Open datasets (no dask — we do manual time batching) ─────────────────────

print("\nOpening datasets ...")
gl_ds = xr.open_dataset(GLEAM_FILE)
ms_ds = xr.open_dataset(MSWEP_FILE)

try:
    fc_ds = xr.open_dataset(FLUXCOM_GLOBAL)
    has_fluxcom = True
    print("  FLUXCOM found")
except FileNotFoundError:
    has_fluxcom = False
    print("  FLUXCOM not found — S3 skipped")

glat = gl_ds['lat'].values;  glon = gl_ds['lon'].values
nlat, nlon = len(glat), len(glon)
dlat = glat[1] - glat[0];   dlon = glon[1] - glon[0]

i_lat = np.round((cells.lat.values - glat[0]) / dlat).astype(int)
i_lon = np.round((cells.lon.values - glon[0]) / dlon).astype(int)
ok    = (i_lat >= 0) & (i_lat < nlat) & (i_lon >= 0) & (i_lon < nlon)
cells  = cells[ok].copy().reset_index(drop=True)
i_lat  = i_lat[ok];  i_lon = i_lon[ok]
print(f"  Cells in global grid: {len(cells)}")
log_resources("datasets open")


# ── Load MSWEP once for the full record (q95 + both scenarios) ───────────────

print("\nLoading MSWEP (full record, year batches) ...")
t1 = time.time()
ms_cells_all = load_cells_fast(ms_ds['mask'], '1979-01-01', '2020-12-31',
                               i_lat, i_lon, label="MSWEP")
ms_times_all = pd.DatetimeIndex(ms_ds.time.values)
print(f"  Shape: {ms_cells_all.shape}  ({time.time()-t1:.0f}s)")
log_resources("MSWEP loaded")

print("  Computing q95 per cell ...")
q95_c  = np.nanpercentile(ms_cells_all, 95, axis=0)
q95v_c = q95_c > 0.1
print(f"  Valid cells: {np.sum(q95v_c)}")

# Pre-slice MSWEP for each scenario time window (no re-read)
def ms_slice(t0, t1):
    mask = (ms_times_all >= t0) & (ms_times_all <= t1)
    return ms_cells_all[mask]

log_resources("after q95")


# ── Run scenarios ─────────────────────────────────────────────────────────────

def run_scenario(sc_key, et_cells, ms_cells, times_pd):
    T, N = et_cells.shape
    n_chunks = (N + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"  Detecting ExEvEs  ({T}d × {N} cells, {n_chunks} chunks × "
          f"{'all' if N_JOBS == -1 else N_JOBS} workers) ...")
    t1 = time.time()
    exeve = detect_parallel(et_cells, times_pd)
    print(f"  Done in {time.time()-t1:.0f}s")
    log_resources(f"{sc_key} detection done")

    q95m     = ms_cells >= q95_c[np.newaxis, :]
    has_data = ~np.all(np.isnan(et_cells), axis=0)
    valid    = has_data & q95v_c
    n_q      = q95m.sum(axis=0).astype(float)
    n_qw     = (exeve & q95m).sum(axis=0).astype(float)
    pct      = np.where(valid & (n_q > 0), n_qw / n_q * 100, np.nan)
    print(f"  mean q95 = {np.nanmean(pct):.1f}%   valid = {np.sum(np.isfinite(pct))}")
    return pct


results = {}

for sc_key, t0, t1 in [("S1", '1980-01-01', '2020-12-31'),
                        ("S2", '2001-01-01', '2020-12-31')]:
    print(f"\n{'─'*55}\n{sc_key}  GLEAM {t0[:4]}–{t1[:4]}\n{'─'*55}")
    print("  Loading GLEAM cells ...")
    t1_ = time.time()
    et_cells = load_cells_fast(gl_ds['E'], t0, t1, i_lat, i_lon, label="GLEAM")
    print(f"  Shape: {et_cells.shape}  ({time.time()-t1_:.0f}s)")
    log_resources(f"{sc_key} GLEAM loaded")

    ms_cells = ms_slice(t0, t1)
    times_pd = pd.DatetimeIndex(gl_ds.time.sel(time=slice(t0, t1)).values)
    results[sc_key] = run_scenario(sc_key, et_cells, ms_cells, times_pd)
    del et_cells


if has_fluxcom:
    print(f"\n{'─'*55}\nS3  FLUXCOM 2001–2020\n{'─'*55}")
    fc_lat = fc_ds['lat'].values;  fc_lon = fc_ds['lon'].values
    fc_dlat = fc_lat[1] - fc_lat[0];  fc_dlon = fc_lon[1] - fc_lon[0]
    fc_ilat = np.round((cells.lat.values - fc_lat[0]) / fc_dlat).astype(int)
    fc_ilon = np.round((cells.lon.values - fc_lon[0]) / fc_dlon).astype(int)
    ok_fc   = ((fc_ilat >= 0) & (fc_ilat < len(fc_lat)) &
               (fc_ilon >= 0) & (fc_ilon < len(fc_lon)))

    print("  Loading FLUXCOM cells ...")
    t1_ = time.time()
    et_cells = (load_cells_fast(fc_ds['ET'], '2001-01-01', '2020-12-31',
                                fc_ilat[ok_fc], fc_ilon[ok_fc], label="FLUXCOM") * 24)
    print(f"  Shape: {et_cells.shape}  ({time.time()-t1_:.0f}s)")
    log_resources("FLUXCOM loaded")

    ms_cells = ms_slice('2001-01-01', '2020-12-31')[:, ok_fc]
    times_pd = pd.DatetimeIndex(fc_ds.time.sel(time=slice('2001-01-01', '2020-12-31')).values)
    pct_sub  = run_scenario("S3", et_cells, ms_cells, times_pd)
    pct      = np.full(len(cells), np.nan)
    pct[ok_fc] = pct_sub
    results['S3'] = pct
    del et_cells


# ── Density plot ──────────────────────────────────────────────────────────────

print(f"\n{'─'*55}\nPlotting ...\n{'─'*55}")
sc_plot = [(k, l, c) for k, l, c in zip(['S1','S2','S3'], SC_LABELS, SC_COLORS)
           if k in results]

fig, ax = plt.subplots(figsize=(10, 5))
for sc_key, sc_label, sc_color in sc_plot:
    v = results[sc_key][np.isfinite(results[sc_key])]
    if len(v) < 20: continue
    mu = np.nanmean(v)
    ax.hist(v, bins=50, range=(0, 100), density=True, alpha=0.15, color=sc_color)
    kde = gaussian_kde(v, bw_method=0.2)
    x = np.linspace(0, 100, 500)
    ax.plot(x, kde(x), color=sc_color, lw=2.5, label=f"{sc_label}  mean={mu:.1f}%")
    ax.axvline(mu, color=sc_color, lw=1.2, ls='--', alpha=0.7)
ax.set_xlabel("% of q95 days falling during ExEvEs", fontsize=10)
ax.set_ylabel("Density", fontsize=10)
ax.set_title(f"{LC_TYPE}: % q95 days during ExEvEs — Global", fontsize=12, fontweight='bold')
ax.legend(fontsize=9);  ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fname = f"exeves_{LC_SLUG}_pct_q95_density.png"
fig.savefig(fname, dpi=150, bbox_inches='tight')
print(f"  Saved {fname}")


# ── Spatial map ───────────────────────────────────────────────────────────────

dh        = abs(dlat) / 2
lat_edges = np.concatenate([glat - dh, [glat[-1] + dh]])
lon_edges = np.concatenate([glon - dh, [glon[-1] + dh]])
LON, LAT  = np.meshgrid(lon_edges, lat_edges)
cmap      = plt.cm.YlOrRd
norm      = mcolors.Normalize(vmin=0, vmax=100)
n_maps    = len(sc_plot)

fig, axes = plt.subplots(1, n_maps, figsize=(8 * n_maps, 4.5))
if n_maps == 1: axes = [axes]
for ax, (sc_key, sc_label, _) in zip(axes, sc_plot):
    grid = np.full((nlat, nlon), np.nan)
    grid[i_lat, i_lon] = results[sc_key]
    ax.pcolormesh(LON, LAT, np.ma.masked_invalid(grid), cmap=cmap, norm=norm, rasterized=True)
    ax.set_title(sc_label, fontsize=11, fontweight='bold')
    ax.set_xlabel("Longitude", fontsize=9);  ax.set_ylabel("Latitude", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.text(0.02, 0.02, f"mean={np.nanmean(results[sc_key]):.1f}%",
            transform=ax.transAxes, fontsize=9, va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                    ax=axes, fraction=0.02, pad=0.02)
cbar.set_label("% q95 days during ExEvEs", fontsize=9)
fig.suptitle(f"{LC_TYPE}: % of q95 days during ExEvEs — Global",
             fontsize=12, fontweight='bold', y=1.01)
fig.tight_layout()
fname = f"exeves_{LC_SLUG}_pct_q95_map.png"
fig.savefig(fname, dpi=150, bbox_inches='tight')
print(f"  Saved {fname}")

print(f"\nTotal runtime: {(time.time()-t_total)/60:.1f} min")
log_resources("done")
