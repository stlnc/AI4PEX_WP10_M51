"""
Spatial maps of % q95 days during ExEvEs — 3 panels (scenarios), 6 LC types
each in a distinct colormap with its own colorbar.
Output: exeves_europe_q95_map_lc.png
"""
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pyreadr
import warnings; warnings.filterwarnings('ignore')

GLEAM_FILE   = "gleam_e_mm_land_198001_202412_025_daily.nc"
FLUXCOM_FILE = "ET_2001_2020_025_daily_eur.nc"
MSWEP_FILE   = "mswep-v2-8_tp_mm_land_197901_202012_025_daily.nc"
MASKS_FILE   = "masks.rda"

LC_TYPES   = ["Forests", "Croplands", "Savannas", "Grasslands", "Shrublands", "Barren"]
LC_COLORS  = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#D55E00", "#CC79A7"]

SC_KEYS   = ['S1', 'S2', 'S3']
SC_LABELS = ["GLEAM 1980–2020", "GLEAM 2001–2020", "FLUXCOM 2001–2020"]
SEASON_MONTHS = [4, 5, 6, 7, 8, 9, 10]  # April-October


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
    del z  # large float32 array no longer needed
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
eur = masks[
    masks['ipcc_continent'].isin(['EUROPE', 'EUROPE-AFRICA']) &
    (masks['land_mask'] == 'land') &
    masks['land_cover_short_class'].isin(LC_TYPES)
].copy().reset_index(drop=True)
del masks  # large table no longer needed
print(f"  Target cells: {len(eur)}")


# ── Load datasets ─────────────────────────────────────────────────────────────

def select_period_apr_oct(da, t0, t1):
    sub = da.sel(time=slice(t0, t1))
    return sub.where(sub.time.dt.month.isin(SEASON_MONTHS), drop=True)

lat_min = eur.lat.min() - 0.5;  lat_max = eur.lat.max() + 0.5
lon_min = eur.lon.min() - 0.5;  lon_max = eur.lon.max() + 0.5

print("Loading GLEAM ...")
gl_ds    = xr.open_dataset(GLEAM_FILE)
gl_ref   = gl_ds['E'].sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
eur_lats = gl_ref.lat.values
eur_lons = gl_ref.lon.values
gl_s1_da = select_period_apr_oct(gl_ref, '1980-01-01', '2020-12-31')
gl_s1    = gl_s1_da.values.astype(np.float32)
t_s1     = pd.DatetimeIndex(gl_s1_da.time.values)
gl_ds.close()

print("Loading MSWEP ...")
ms_ds    = xr.open_dataset(MSWEP_FILE)
ms_eur   = ms_ds['mask'].sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
ms_full  = ms_eur.where(ms_eur.time.dt.month.isin(SEASON_MONTHS), drop=True).values.astype(np.float32)
ms_full[ms_full <= 0] = np.nan  # wet-day-only threshold basis (>0 mm/day)
q95_grid  = np.nanpercentile(ms_full, 95, axis=0)
q95_valid = np.isfinite(q95_grid)
del ms_full  # full time-series only needed for percentile
ms_s1_da = select_period_apr_oct(ms_eur, '1980-01-01', '2020-12-31')
ms_s1    = ms_s1_da.values.astype(np.float32)
ms_ds.close()

# Derive S2 arrays as views of S1 (no copy)
t_s2_start = t_s1.searchsorted(pd.Timestamp('2001-01-01'))
gl_s2 = gl_s1[t_s2_start:]   # numpy view
ms_s2 = ms_s1[t_s2_start:]   # numpy view
t_s2  = t_s1[t_s2_start:]

print("Loading FLUXCOM ...")
fc_ds   = xr.open_dataset(FLUXCOM_FILE)
fc_land = fc_ds["land_fraction"].values >= 0.5
fc_s3_da = select_period_apr_oct(fc_ds['ET'], '2001-01-01', '2020-12-31')
t_s3     = pd.DatetimeIndex(fc_s3_da.time.values)
fc_s3    = (fc_s3_da.values * 24).astype(np.float32)
fc_ds.close()

print(f"S1: {gl_s1.shape}  S2: {gl_s2.shape}  S3: {fc_s3.shape} (Apr-Oct only)")


# ── Grid index mapping ────────────────────────────────────────────────────────

dlat = eur_lats[1] - eur_lats[0]
dlon = eur_lons[1] - eur_lons[0]
nlat, nlon = len(eur_lats), len(eur_lons)

i_lat = np.round((eur.lat.values - eur_lats[0]) / dlat).astype(int)
i_lon = np.round((eur.lon.values - eur_lons[0]) / dlon).astype(int)

ok    = (i_lat >= 0) & (i_lat < nlat) & (i_lon >= 0) & (i_lon < nlon)
eur   = eur[ok].copy().reset_index(drop=True)
i_lat = i_lat[ok]; i_lon = i_lon[ok]
q95_c  = q95_grid[i_lat, i_lon]
q95v_c = q95_valid[i_lat, i_lon]
print(f"Cells in bounds: {len(eur)}")


# ── Run scenarios ─────────────────────────────────────────────────────────────

for sc_label, et_grid, times_pd, ms_grid in [
    ("S1", gl_s1, t_s1, ms_s1),
    ("S2", gl_s2, t_s2, ms_s2),
    ("S3", fc_s3, t_s3, ms_s2),
]:
    print(f"\n{sc_label}: {et_grid.shape[0]} days ...")
    et_cells = et_grid[:, i_lat, i_lon]
    ms_cells = ms_grid[:, i_lat, i_lon]
    T = et_grid.shape[0]

    exeve = detect_exeves(et_cells, times_pd)
    q95m  = ms_cells >= q95_c[np.newaxis, :]

    has_data = ~np.all(np.isnan(et_cells), axis=0)
    if sc_label == "S3":
        valid = has_data & q95v_c & fc_land[i_lat, i_lon]
    else:
        valid = has_data & q95v_c

    n_q  = q95m.sum(axis=0).astype(float)
    n_qw = (exeve & q95m).sum(axis=0).astype(float)
    eur[f'pct_q95_{sc_label}'] = np.where(valid & (n_q > 0), n_qw / n_q * 100, np.nan)
    del et_cells, ms_cells, exeve, q95m  # free large per-scenario arrays

    for lc in LC_TYPES:
        sub = eur[eur['land_cover_short_class'] == lc]
        print(f"  {lc:<12}  q95={sub[f'pct_q95_{sc_label}'].mean():.1f}%")

del gl_s1, ms_s1, fc_s3  # all scenario data processed


# ── Spatial maps ──────────────────────────────────────────────────────────────

dh = abs(dlat) / 2
lat_edges = np.concatenate([eur_lats - dh, [eur_lats[-1] + dh]])
lon_edges = np.concatenate([eur_lons - dh, [eur_lons[-1] + dh]])
LON, LAT  = np.meshgrid(lon_edges, lat_edges)

geo_aspect = 1.0 / np.cos(np.radians(50))

# Per-LC colorscale range (99th pct across all scenarios)
lc_norms = {}
for lc in LC_TYPES:
    mask = eur['land_cover_short_class'] == lc
    all_vals = np.concatenate([
        eur.loc[mask, f'pct_q95_{k}'].dropna().values for k in SC_KEYS
    ])
    vmax = np.percentile(all_vals, 99) if len(all_vals) else 60.0
    lc_norms[lc] = mcolors.Normalize(vmin=0, vmax=vmax)

def build_grid_lc(col, lc):
    grid = np.full((nlat, nlon), np.nan)
    mask = eur['land_cover_short_class'] == lc
    grid[i_lat[mask], i_lon[mask]] = eur.loc[mask, col].values
    return grid

# Layout: 3 map columns + narrow colorbar column
fig = plt.figure(figsize=(17, 5.5))
gs  = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.12], wspace=0.06)
map_axes = [fig.add_subplot(gs[0, c]) for c in range(3)]

for ax, sc_key, sc_label in zip(map_axes, SC_KEYS, SC_LABELS):
    for lc, lc_color in zip(LC_TYPES, LC_COLORS):
        grid   = build_grid_lc(f'pct_q95_{sc_key}', lc)
        masked = np.ma.masked_invalid(grid)
        cmap_lc = mcolors.LinearSegmentedColormap.from_list(lc, ['#f5f5f5', lc_color])
        ax.pcolormesh(LON, LAT, masked, cmap=cmap_lc,
                      norm=lc_norms[lc], rasterized=True)

    ax.set_title(sc_label, fontsize=10, fontweight='bold', pad=4)
    ax.set_aspect(geo_aspect)
    ax.tick_params(labelsize=7)
    ax.set_xlabel("Longitude", fontsize=8)
    if ax is map_axes[0]:
        ax.set_ylabel("Latitude", fontsize=8)
    else:
        ax.set_yticklabels([])

# 6 stacked colorbars in the last column
cbar_gs = gs[0, 3].subgridspec(len(LC_TYPES), 1, hspace=1.2)
for i, (lc, lc_color, norm) in enumerate(zip(LC_TYPES, LC_COLORS, lc_norms.values())):
    cax = fig.add_subplot(cbar_gs[i])
    cmap_lc = mcolors.LinearSegmentedColormap.from_list(lc, ['#f5f5f5', lc_color])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap_lc),
                      cax=cax, orientation='vertical')
    cb.ax.tick_params(labelsize=6)
    cb.set_label(lc, fontsize=7, labelpad=3)

fig.suptitle("% of q95 precipitation days falling during ExEvEs — Europe by land cover",
             fontsize=11, fontweight='bold', y=1.01)
fig.savefig("exeves_europe_q95_map_lc.png", dpi=150, bbox_inches='tight')
print("\nSaved exeves_europe_q95_map_lc.png")
