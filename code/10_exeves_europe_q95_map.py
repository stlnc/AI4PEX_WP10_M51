"""
Spatial maps of % q95 days falling during ExEvEs — Europe land cover cells.
Two panels: GLEAM 2001–2020 and FLUXCOM 2001–2020.
Output: exeves_europe_q95_map.png
"""
from _analysis import pentad_zscore
from _alignment import validate_alignment
from _paths import input_path, output_path
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pyreadr
import warnings; warnings.filterwarnings('ignore')

GLEAM_FILE   = "gleam_e_mm_eur_2001_2020_025_daily.nc"
FLUXCOM_FILE = "ET_2001_2020_025_daily_eur.nc"
MSWEP_FILE   = "mswep-v2-8_tp_mm_eur_2001_2020_025_daily.nc"
MASKS_FILE   = "masks.rda"

LC_TYPES = ["Forests", "Croplands", "Savannas", "Grasslands", "Shrublands", "Barren"]

SC_LABELS = ["GLEAM 2001–2020", "FLUXCOM 2001–2020"]
SC_COLORS = ["seagreen", "darkorange"]


# ── Detection ─────────────────────────────────────────────────────────────────


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
masks = pyreadr.read_r(str(input_path(MASKS_FILE)))['masks']
eur = masks[
    masks['ipcc_continent'].isin(['EUROPE', 'EUROPE-AFRICA']) &
    (masks['land_mask'] == 'land') &
    masks['land_cover_short_class'].isin(LC_TYPES)
].copy().reset_index(drop=True)
print(f"  Target cells: {len(eur)}")


# ── Load datasets ─────────────────────────────────────────────────────────────

print("Loading GLEAM ...")
gl_ds    = xr.open_dataset(input_path(GLEAM_FILE))
gl_times = pd.DatetimeIndex(gl_ds.time.values)

print("Loading FLUXCOM ...")
fc_ds   = xr.open_dataset(input_path(FLUXCOM_FILE))
fc_land = fc_ds["land_fraction"].values >= 0.5

print("Loading MSWEP ...")
ms_ds = xr.open_dataset(input_path(MSWEP_FILE))

lat_min = eur.lat.min() - 0.5;  lat_max = eur.lat.max() + 0.5
lon_min = eur.lon.min() - 0.5;  lon_max = eur.lon.max() + 0.5

gl_ref   = gl_ds['E'].sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
eur_lats = gl_ref.lat.values
eur_lons = gl_ref.lon.values

ms_full  = ms_ds['mask'].sel(
    time=slice('2001-01-01', '2020-12-31'),
    lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max)).values.astype(np.float32)
q95_grid  = np.nanpercentile(ms_full, 95, axis=0)
q95_valid = q95_grid > 0.1

def load_period(da, t0, t1):
    return da.sel(time=slice(t0, t1)).values.astype(np.float32)

gl_s2 = load_period(gl_ref, '2001-01-01', '2020-12-31')
ms_s2 = load_period(ms_ds['mask'].sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max)),
                    '2001-01-01', '2020-12-31')
fc_s3 = (fc_ds['ET'].sel(time=slice('2001-01-01', '2020-12-31')).values * 24).astype(np.float32)

t_s2 = pd.DatetimeIndex(gl_ds.time.sel(time=slice('2001-01-01', '2020-12-31')).values)
t_s3 = pd.DatetimeIndex(fc_ds.time.sel(time=slice('2001-01-01', '2020-12-31')).values)
validate_alignment(
    gl_ref.sel(time=slice('2001-01-01', '2020-12-31')),
    ms_ds['mask'].sel(time=slice('2001-01-01', '2020-12-31'), lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max)),
    fc_ds['ET'].sel(time=slice('2001-01-01', '2020-12-31')),
)

print(f"S2: {gl_s2.shape}  S3: {fc_s3.shape}")


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

scenarios = [
    ("S2", gl_s2, t_s2, ms_s2),
    ("S3", fc_s3, t_s3, ms_s2),
]

for sc_label, et_grid, times_pd, ms_grid in scenarios:
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

    for lc in LC_TYPES:
        sub = eur[eur['land_cover_short_class'] == lc]
        print(f"  {lc:<12}  q95={sub[f'pct_q95_{sc_label}'].mean():.1f}%")


# ── Spatial maps ──────────────────────────────────────────────────────────────

# Build 2D grid for a given LC mask — only cells of that LC type are non-NaN
def build_grid_lc(col, lc):
    grid = np.full((nlat, nlon), np.nan)
    mask = eur['land_cover_short_class'] == lc
    grid[i_lat[mask], i_lon[mask]] = eur.loc[mask, col].values
    return grid

# Cell-edge meshgrid for pcolormesh
dh = abs(dlat) / 2
lat_edges = np.concatenate([eur_lats - dh, [eur_lats[-1] + dh]])
lon_edges = np.concatenate([eur_lons - dh, [eur_lons[-1] + dh]])
LON, LAT  = np.meshgrid(lon_edges, lat_edges)

# Geographic aspect correction at mean latitude ~50°N
geo_aspect = 1.0 / np.cos(np.radians(50))

cmap = plt.cm.YlOrRd
SC_KEYS = ['S2', 'S3']

# Per-LC vmax: 99th percentile of values across all scenarios
lc_vmax = {}
for lc in LC_TYPES:
    mask = eur['land_cover_short_class'] == lc
    all_vals = np.concatenate([
        eur.loc[mask, f'pct_q95_{k}'].dropna().values for k in SC_KEYS
    ])
    lc_vmax[lc] = np.percentile(all_vals, 99) if len(all_vals) else 60.0

fig, axes = plt.subplots(
    len(LC_TYPES), len(SC_KEYS),
    figsize=(14, 22),
    gridspec_kw={'wspace': 0.05, 'hspace': 0.15}
)

for r, lc in enumerate(LC_TYPES):
    norm = mcolors.Normalize(vmin=0, vmax=lc_vmax[lc])
    row_axes = axes[r]

    for c, (sc_key, sc_label) in enumerate(zip(SC_KEYS, SC_LABELS)):
        ax = row_axes[c]
        grid   = build_grid_lc(f'pct_q95_{sc_key}', lc)
        masked = np.ma.masked_invalid(grid)
        ax.pcolormesh(LON, LAT, masked, cmap=cmap, norm=norm, rasterized=True)
        ax.set_aspect(geo_aspect)
        ax.tick_params(labelsize=7)

        # Mean annotation
        mean_val = np.nanmean(eur.loc[eur['land_cover_short_class'] == lc, f'pct_q95_{sc_key}'].values)
        ax.text(0.03, 0.04, f"{mean_val:.1f}%", transform=ax.transAxes,
                fontsize=8, va='bottom', ha='left',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.75))

        # Column headers (top row only)
        if r == 0:
            ax.set_title(sc_label, fontsize=10, fontweight='bold', pad=4)
        # x-label (bottom row only)
        if r == len(LC_TYPES) - 1:
            ax.set_xlabel("Longitude", fontsize=8)
        else:
            ax.set_xticklabels([])
        # y-label (left column only)
        if c == 0:
            ax.set_ylabel(lc, fontsize=9, fontweight='bold', labelpad=4)
        else:
            ax.set_yticklabels([])

    # Per-row colorbar to the right of the three maps
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=row_axes, orientation='vertical',
                        fraction=0.018, pad=0.01, aspect=18)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("% q95 during ExEvEs", fontsize=7)

fig.suptitle("% of q95 precipitation days falling during ExEvEs — Europe by land cover",
             fontsize=12, fontweight='bold', y=1.002)
fig.savefig(output_path("exeves_europe_q95_map.png"), dpi=150, bbox_inches='tight')
print("\nSaved exeves_europe_q95_map.png")
