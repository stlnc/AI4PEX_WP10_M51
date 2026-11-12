"""Merge global annual FLUXCOM ET files, cropped to European extent, 2001-2020."""
import glob
import xarray as xr
import pyreadr
import numpy as np

MASKS_FILE = "masks.rda"
OUT        = "ET_2001_2020_025_daily_eur.nc"

# European lat/lon bounds from masks
masks = pyreadr.read_r(MASKS_FILE)['masks']
eur   = masks[
    masks['ipcc_continent'].isin(['EUROPE', 'EUROPE-AFRICA']) &
    (masks['land_mask'] == 'land')
]
EPS    = 1e-3
lat_min = eur.lat.min() - 0.5
lat_max = eur.lat.max() + 0.5
lon_min = eur.lon.min() - 0.5
lon_max = eur.lon.max() + 0.5
print(f"European extent: lat [{lat_min:.3f}, {lat_max:.3f}]  lon [{lon_min:.3f}, {lon_max:.3f}]")

# Reference lat/lon from GLEAM (exact coordinates to assign to FLUXCOM)
gl = xr.open_dataset("gleam_e_mm_land_198001_202412_025_daily.nc")
ref_lats = gl['E'].sel(lat=slice(lat_max + EPS, lat_min - EPS)).lat.values
ref_lons = gl['E'].sel(lon=slice(lon_min - EPS, lon_max + EPS)).lon.values
gl.close()
print(f"Reference grid: {len(ref_lats)} lat × {len(ref_lons)} lon")

et_files = sorted(glob.glob("ET_????_025_daily.nc"))
# Keep only 2001-2020
et_files = [f for f in et_files if any(str(y) in f for y in range(2001, 2021))]
print(f"Files to merge: {len(et_files)}")

chunks, land_frac = [], None
for f in et_files:
    print(f"  {f} ...")
    ds = xr.open_dataset(f)
    # lat descending, lon ascending — slice with EPS for ULP tolerance
    cropped = ds.sel(
        lat=slice(lat_max + EPS, lat_min - EPS),
        lon=slice(lon_min - EPS, lon_max + EPS),
    ).assign_coords(lat=ref_lats, lon=ref_lons)
    chunks.append(cropped[["ET"]])
    if land_frac is None:
        land_frac = cropped["land_fraction"]

print("Concatenating ...")
merged = xr.concat(chunks, dim="time")
merged["land_fraction"] = land_frac

print(f"Writing {OUT} ...")
merged.to_netcdf(OUT)
print(f"Done. Shape: {merged['ET'].shape}")
