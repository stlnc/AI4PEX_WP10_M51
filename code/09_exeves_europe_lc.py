"""
ExEvEs analysis by land cover type for Europe (EUROPE + EUROPE-AFRICA).
Two scenarios per land cover subplot:
  S2: GLEAM 2001–2020
  S3: FLUXCOM 2001–2020

Outputs:
  exeves_europe_pct.png  – % ExEvE days of total days
  exeves_europe_q95.png  – % q95 days falling during ExEvEs
"""
from _analysis import pentad_zscore
from _alignment import validate_alignment
from _paths import input_path, output_path
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pyreadr
import warnings; warnings.filterwarnings('ignore')

GLEAM_FILE   = "gleam_e_mm_eur_2001_2020_025_daily.nc"
FLUXCOM_FILE = "ET_2001_2020_025_daily_eur.nc"
MSWEP_FILE   = "mswep-v2-8_tp_mm_eur_2001_2020_025_daily.nc"
MASKS_FILE   = "masks.rda"

LC_TYPES  = ["Forests", "Croplands", "Savannas", "Grasslands", "Shrublands", "Barren"]
LC_COLORS = ["forestgreen", "goldenrod", "peru", "yellowgreen", "sienna", "lightgray"]

SC_LABELS = ["GLEAM 2001–2020", "FLUXCOM 2001–2020"]
SC_COLORS = ["seagreen", "darkorange"]


# ── Detection functions (T, N) ────────────────────────────────────────────────


def detect_exeves(et, times_pd):
    T, N = et.shape
    z   = pentad_zscore(et, times_pd)
    z80 = np.nanpercentile(z, 80, axis=0)
    z95 = np.nanpercentile(z, 95, axis=0)
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
fc_ds    = xr.open_dataset(input_path(FLUXCOM_FILE))
fc_times = pd.DatetimeIndex(fc_ds.time.values)
# land_fraction from FLUXCOM (already on European grid)
fc_land  = fc_ds["land_fraction"].values >= 0.5

print("Loading MSWEP ...")
ms_ds    = xr.open_dataset(input_path(MSWEP_FILE))

# European bounding box (from masks extent)
lat_min = eur.lat.min() - 0.5;  lat_max = eur.lat.max() + 0.5
lon_min = eur.lon.min() - 0.5;  lon_max = eur.lon.max() + 0.5

# Reference grid from GLEAM European slice (exact coords)
gl_ref = gl_ds['E'].sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
eur_lats = gl_ref.lat.values   # descending
eur_lons = gl_ref.lon.values   # ascending

# MSWEP analysis period for q95 threshold
ms_full = ms_ds['mask'].sel(
    time=slice('2001-01-01', '2020-12-31'),
    lat=slice(lat_max, lat_min),
    lon=slice(lon_min, lon_max)).values.astype(np.float32)
q95_grid  = np.nanpercentile(ms_full, 95, axis=0)
q95_valid = q95_grid > 0.1

def load_period(da, t0, t1):
    return da.sel(time=slice(t0, t1)).values.astype(np.float32)

# Load time slices
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


# ── Map mask cells to grid indices ────────────────────────────────────────────

dlat = eur_lats[1] - eur_lats[0]   # negative
dlon = eur_lons[1] - eur_lons[0]   # positive

i_lat = np.round((eur.lat.values - eur_lats[0]) / dlat).astype(int)
i_lon = np.round((eur.lon.values - eur_lons[0]) / dlon).astype(int)

nlat, nlon = len(eur_lats), len(eur_lons)
ok = (i_lat >= 0) & (i_lat < nlat) & (i_lon >= 0) & (i_lon < nlon)
eur   = eur[ok].copy().reset_index(drop=True)
i_lat = i_lat[ok]; i_lon = i_lon[ok]
q95_c = q95_grid[i_lat, i_lon]
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
    # For FLUXCOM also mask non-land cells
    if sc_label == "S3":
        fc_land_c = fc_land[i_lat, i_lon]
        valid = has_data & q95v_c & fc_land_c
    else:
        valid = has_data & q95v_c

    n_q  = q95m.sum(axis=0).astype(float)
    n_qw = (exeve & q95m).sum(axis=0).astype(float)

    eur[f'pct_ev_{sc_label}']  = np.where(valid,             exeve.sum(axis=0) / T * 100, np.nan)
    eur[f'pct_q95_{sc_label}'] = np.where(valid & (n_q > 0), n_qw / n_q * 100,            np.nan)

    for lc in LC_TYPES:
        sub = eur[eur['land_cover_short_class'] == lc]
        print(f"  {lc:<12}  pct_ev={sub[f'pct_ev_{sc_label}'].mean():.1f}%  "
              f"q95={sub[f'pct_q95_{sc_label}'].mean():.1f}%")


# ── Plotting ──────────────────────────────────────────────────────────────────

def make_figure(col_prefix, suptitle, xlabel, xlim, fname):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=False)
    axes = axes.flatten()

    for ax, lc, lc_color in zip(axes, LC_TYPES, LC_COLORS):
        sub = eur[eur['land_cover_short_class'] == lc]
        for sc_key, sc_label, sc_color in zip(['S2','S3'], SC_LABELS, SC_COLORS):
            col = f'{col_prefix}_{sc_key}'
            v = sub[col].values
            v = v[np.isfinite(v) & (v >= xlim[0]) & (v <= xlim[1])]
            if len(v) < 20:
                continue
            mu = np.nanmean(v)
            ax.hist(v, bins=50, range=xlim, density=True, alpha=0.15, color=sc_color)
            kde = gaussian_kde(v, bw_method=0.2)
            x = np.linspace(*xlim, 500)
            ax.plot(x, kde(x), color=sc_color, lw=2,
                    label=f"{sc_label}  mean={mu:.1f}%")
            ax.axvline(mu, color=sc_color, lw=1.2, ls='--')

        n_cells = len(sub)
        ax.set_title(f"{lc}  (n={n_cells})", fontsize=10)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.set_xlim(*xlim)
        ax.legend(fontsize=7)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle(suptitle, fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_path(fname), dpi=150, bbox_inches='tight')
    print(f"Saved {fname}")


make_figure(
    'pct_ev',
    "ExEvEs as % of all days — Europe by land cover",
    "Percentage (%)", (0, 30),
    "exeves_europe_pct.png"
)

make_figure(
    'pct_q95',
    "% of q95 days falling during ExEvEs — Europe by land cover",
    "Percentage (%)", (0, 100),
    "exeves_europe_q95.png"
)

print("\nAll done.")
