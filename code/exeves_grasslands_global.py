"""
Global ExEvEs analysis for Grasslands land cover type.
% of q95 precipitation days falling within ExEvEs.
Output: exeves_grasslands_pct_q95_density.png, exeves_grasslands_pct_q95_map.png
"""
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import gaussian_kde
import pyreadr
import warnings; warnings.filterwarnings('ignore')

GLEAM_FILE   = "gleam_e_mm_land_198001_202412_025_daily.nc"
FLUXCOM_FILE = "ET_2001_2020_025_daily_eur.nc"  # Will need global version
MSWEP_FILE   = "mswep-v2-8_tp_mm_land_197901_202012_025_daily.nc"
MASKS_FILE   = "masks.rda"

LC_TYPE = "Grasslands"
SC_LABELS = ["GLEAM 1980–2020", "GLEAM 2001–2020", "FLUXCOM 2001–2020"]
SC_COLORS = ["steelblue", "seagreen", "darkorange"]


# ── Detection ─────────────────────────────────────────────────────────────────

def pentad_zscore(et, times_pd):
    T, N = et.shape
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
    T, N = et.shape
    z    = pentad_zscore(et, times_pd)
    z80  = np.nanpercentile(z, 80, axis=0)
    z95  = np.nanpercentile(z, 95, axis=0)
    above = z >= z80; extreme = z >= z95
    out = np.zeros((T, N), dtype=bool)
    for j in range(N):
        if not above[:, j].any(): continue
        d  = np.diff(above[:, j].astype(np.int8), prepend=0, append=0)
        ss = np.where(d ==  1)[0]; es = np.where(d == -1)[0]
        for s, e in zip(ss, es):
            if (e - s) >= 2 and extreme[s:e, j].any():
                out[s:e, j] = True
    return out


# ── Load masks ────────────────────────────────────────────────────────────────

print("Loading masks ...")
masks = pyreadr.read_r(MASKS_FILE)['masks']
grass = masks[
    (masks['land_mask'] == 'land') &
    (masks['land_cover_short_class'] == LC_TYPE)
].copy().reset_index(drop=True)
print(f"  Global {LC_TYPE} cells: {len(grass)}")


# ── Load datasets ─────────────────────────────────────────────────────────────

print("Loading GLEAM ...")
gl_ds    = xr.open_dataset(GLEAM_FILE, chunks={'time': 365, 'lat': 180, 'lon': 180})
gl_times = pd.DatetimeIndex(gl_ds.time.values)
eur_lats = gl_ds['lat'].values  # Global GLEAM grid
eur_lons = gl_ds['lon'].values

print("Loading MSWEP ...")
ms_ds = xr.open_dataset(MSWEP_FILE, chunks={'time': 365, 'lat': 180, 'lon': 180})

# For FLUXCOM: check if global file exists, else warn
try:
    fc_ds = xr.open_dataset("ET_2001_2020_025_daily.nc")  # Global FLUXCOM
    fc_times = pd.DatetimeIndex(fc_ds.time.values)
    has_fluxcom = True
except FileNotFoundError:
    print("  WARNING: Global FLUXCOM not found; FLUXCOM results will be skipped")
    has_fluxcom = False

# Grid mapping: map cell (lat, lon) to nearest grid index
nlat, nlon = len(eur_lats), len(eur_lons)
dlat = eur_lats[1] - eur_lats[0]
dlon = eur_lons[1] - eur_lons[0]

i_lat = np.round((grass.lat.values - eur_lats[0]) / dlat).astype(int)
i_lon = np.round((grass.lon.values - eur_lons[0]) / dlon).astype(int)

ok = (i_lat >= 0) & (i_lat < nlat) & (i_lon >= 0) & (i_lon < nlon)
grass = grass[ok].copy().reset_index(drop=True)
i_lat = i_lat[ok]
i_lon = i_lon[ok]
print(f"  Cells in global grid: {len(grass)}")

# MSWEP for grassland cells only — vectorized cell extraction via isel + dask
print("Loading MSWEP for grassland cells ...")
_cell = dict(lat=xr.DataArray(i_lat, dims='cell'), lon=xr.DataArray(i_lon, dims='cell'))
ms_grass = ms_ds['mask'].isel(**_cell).compute().values.astype(np.float32)
print(f"  MSWEP shape for grasslands: {ms_grass.shape}")
print("Computing q95 threshold per grassland cell ...")
q95_c = np.nanpercentile(ms_grass, 95, axis=0)
q95v_c = q95_c > 0.1
print(f"  Valid cells (q95 > 0.1): {np.sum(q95v_c)}")


# ── Run scenarios ─────────────────────────────────────────────────────────────

def load_cells(da, t0, t1):
    """Extract only grassland cells for a time period — avoids loading the full global grid."""
    return da.sel(time=slice(t0, t1)).isel(**_cell).compute().values.astype(np.float32)

scenarios = [
    ("S1", gl_ds['E'], ms_ds['mask'], '1980-01-01', '2020-12-31'),
    ("S2", gl_ds['E'], ms_ds['mask'], '2001-01-01', '2020-12-31'),
]

results = {}  # Store all results for plotting

for sc_label, et_da, ms_da, t0, t1 in scenarios:
    print(f"\n{sc_label}: Loading {t0} to {t1} ...")
    et_cells = load_cells(et_da, t0, t1)
    ms_cells = load_cells(ms_da, t0, t1)
    times_pd = pd.DatetimeIndex(gl_ds.time.sel(time=slice(t0, t1)).values)
    T = et_cells.shape[0]

    print(f"  Detecting ExEvEs ({T} days, {len(grass)} cells) ...")
    exeve = detect_exeves(et_cells, times_pd)
    q95m = ms_cells >= q95_c[np.newaxis, :]

    has_data = ~np.all(np.isnan(et_cells), axis=0)
    valid = has_data & q95v_c

    n_q = q95m.sum(axis=0).astype(float)
    n_qw = (exeve & q95m).sum(axis=0).astype(float)
    pct_q95 = np.where(valid & (n_q > 0), n_qw / n_q * 100, np.nan)

    results[sc_label] = pct_q95
    mean_val = np.nanmean(pct_q95[np.isfinite(pct_q95)])
    print(f"  {LC_TYPE}: mean q95={mean_val:.1f}%  valid_cells={np.sum(np.isfinite(pct_q95))}")

# Optional FLUXCOM
if has_fluxcom:
    print("\nS3: Loading FLUXCOM 2001–2020 ...")
    et_cells = load_cells(fc_ds['ET'], '2001-01-01', '2020-12-31') * 24  # mm/hr → mm/day
    ms_cells = load_cells(ms_ds['mask'], '2001-01-01', '2020-12-31')
    t_s3 = pd.DatetimeIndex(fc_ds.time.sel(time=slice('2001-01-01', '2020-12-31')).values)
    T = et_cells.shape[0]

    print(f"  Detecting ExEvEs ({T} days, {len(grass)} cells) ...")
    exeve = detect_exeves(et_cells, t_s3)
    q95m = ms_cells >= q95_c[np.newaxis, :]

    has_data = ~np.all(np.isnan(et_cells), axis=0)
    valid = has_data & q95v_c
    n_q = q95m.sum(axis=0).astype(float)
    n_qw = (exeve & q95m).sum(axis=0).astype(float)
    pct_q95 = np.where(valid & (n_q > 0), n_qw / n_q * 100, np.nan)

    results['S3'] = pct_q95
    mean_val = np.nanmean(pct_q95[np.isfinite(pct_q95)])
    print(f"  {LC_TYPE}: mean q95={mean_val:.1f}%  valid_cells={np.sum(np.isfinite(pct_q95))}")


# ── Plotting: Density ─────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5))

for sc_key, sc_label, sc_color in zip(['S1', 'S2', 'S3'], SC_LABELS, SC_COLORS):
    if sc_key not in results:
        continue
    v = results[sc_key]
    v_clean = v[np.isfinite(v)]
    if len(v_clean) < 20:
        continue

    mu = np.nanmean(v_clean)
    ax.hist(v_clean, bins=50, range=(0, 100), density=True, alpha=0.15, color=sc_color)
    kde = gaussian_kde(v_clean, bw_method=0.2)
    x = np.linspace(0, 100, 500)
    ax.plot(x, kde(x), color=sc_color, lw=2.5, label=f"{sc_label}  mean={mu:.1f}%")
    ax.axvline(mu, color=sc_color, lw=1.2, ls='--', alpha=0.7)

ax.set_xlabel("% of q95 days falling during ExEvEs", fontsize=10)
ax.set_ylabel("Density", fontsize=10)
ax.set_title(f"{LC_TYPE}: % q95 days during ExEvEs — Global", fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='upper right')
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig("exeves_grasslands_pct_q95_density.png", dpi=150, bbox_inches='tight')
print("\nSaved exeves_grasslands_pct_q95_density.png")


# ── Plotting: Global maps ─────────────────────────────────────────────────────

# Build 2D grids for pcolormesh
dh = abs(dlat) / 2
lat_edges = np.concatenate([eur_lats - dh, [eur_lats[-1] + dh]])
lon_edges = np.concatenate([eur_lons - dh, [eur_lons[-1] + dh]])
LON, LAT = np.meshgrid(lon_edges, lat_edges)

def build_grid(col_key):
    grid = np.full((nlat, nlon), np.nan)
    grid[i_lat, i_lon] = results[col_key]
    return grid

cmap = plt.cm.YlOrRd
n_maps = len(results)
figsize = (6 * n_maps, 5.5)
fig, axes = plt.subplots(1, n_maps, figsize=figsize)
if n_maps == 1:
    axes = [axes]

vmin, vmax = 0, 100
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

for ax, (sc_key, sc_label) in zip(axes, zip(['S1', 'S2', 'S3'], SC_LABELS)):
    if sc_key not in results:
        continue
    grid = build_grid(sc_key)
    masked = np.ma.masked_invalid(grid)
    ax.pcolormesh(LON, LAT, masked, cmap=cmap, norm=norm, rasterized=True)
    ax.set_title(sc_label, fontsize=11, fontweight='bold')
    ax.set_xlabel("Longitude", fontsize=9)
    ax.set_ylabel("Latitude", fontsize=9)
    ax.tick_params(labelsize=8)
    mean_val = np.nanmean(results[sc_key][np.isfinite(results[sc_key])])
    ax.text(0.02, 0.02, f"mean = {mean_val:.1f}%", transform=ax.transAxes,
            fontsize=9, va='bottom', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                    ax=axes, orientation='vertical', fraction=0.02, pad=0.02,
                    label="% q95 days during ExEvEs")
cbar.ax.tick_params(labelsize=8)

fig.suptitle(f"{LC_TYPE}: % of q95 precipitation days falling during ExEvEs — Global",
             fontsize=12, fontweight='bold', y=1.01)
fig.tight_layout()
fig.savefig("exeves_grasslands_pct_q95_map.png", dpi=150, bbox_inches='tight')
print("Saved exeves_grasslands_pct_q95_map.png")

print("\nAll done.")
