"""Extract the 2001-2020 Mediterranean FLUXCOM data from the European dataset."""

import numpy as np
import pandas as pd
import xarray as xr

from _paths import input_path, output_path

GLEAM = "gleam_e_mm_med_198001_202412_025_daily.nc"
FLUXCOM = "ET_2001_2020_025_daily_eur.nc"
OUT = "ET_2001_2020_025_daily_med.nc"
START, END = "2001-01-01", "2020-12-31"
EPS = 1e-4


def main():
    with xr.open_dataset(input_path(GLEAM)) as gleam, xr.open_dataset(input_path(FLUXCOM)) as et:
        lat = gleam.lat.values
        lon = gleam.lon.values
        cropped = et.sel(
            time=slice(START, END),
            lat=slice(float(lat.max()) + EPS, float(lat.min()) - EPS),
            lon=slice(float(lon.min()) - EPS, float(lon.max()) + EPS),
        )
        if (cropped.sizes["lat"] != len(lat) or cropped.sizes["lon"] != len(lon)
                or not np.allclose(cropped.lat.values, lat, rtol=0, atol=EPS)
                or not np.allclose(cropped.lon.values, lon, rtol=0, atol=EPS)):
            raise ValueError("European FLUXCOM and Mediterranean GLEAM grids do not match")
        expected = pd.date_range(START, END, freq="D")
        if not pd.DatetimeIndex(cropped.time.values).equals(expected):
            raise ValueError("European FLUXCOM must contain every day in 2001-2020 exactly once")
        destination = output_path(OUT)
        if destination.resolve() == input_path(FLUXCOM).resolve():
            raise ValueError("The output must not overwrite the European FLUXCOM source")
        cropped[["ET", "land_fraction"]].to_netcdf(destination)
        print(f"Saved {destination}")


if __name__ == "__main__":
    main()
