import glob
import xarray as xr

GLEAM = "gleam_e_mm_med_198001_202412_025_daily.nc"
OUT = "ET_2001_2021_025_daily_med.nc"

gleam = xr.open_dataset(GLEAM)
lon_min, lon_max = round(float(gleam.lon.min()), 3), round(float(gleam.lon.max()), 3)
lat_min, lat_max = round(float(gleam.lat.min()), 3), round(float(gleam.lat.max()), 3)
gleam.close()

# add a small epsilon to lat_max/lon_max so the slice includes boundary points
# that may be stored as e.g. 44.87500000000001 due to floating-point precision
EPS = 1e-4

et_files = sorted(glob.glob("ET_????_025_daily.nc"))
print(f"Found {len(et_files)} ET files. Extent: lon [{lon_min}, {lon_max}], lat [{lat_min}, {lat_max}]")

chunks = []
for f in et_files:
    print(f"  Cropping {f} ...")
    ds = xr.open_dataset(f)
    # ET lat is descending, so slice largest first
    cropped = ds.sel(
            lon=slice(lon_min - EPS, lon_max + EPS),
            lat=slice(lat_max + EPS, lat_min - EPS),
        )
    chunks.append(cropped[["ET"]])
    land_frac = cropped["land_fraction"]

print("Merging along time ...")
merged = xr.concat(chunks, dim="time")
merged["land_fraction"] = land_frac  # static 2D field, save once

print(f"Writing {OUT} ...")
merged.to_netcdf(OUT)
print("Done.")
