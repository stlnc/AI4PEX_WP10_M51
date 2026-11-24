"""Crop locally available GLEAM and MSWEP files to the European FLUXCOM grid."""

import numpy as np
import pandas as pd
import xarray as xr

from _paths import input_path, output_path

START, END = "2001-01-01", "2020-12-31"
FLUXCOM = "ET_2001_2020_025_daily_eur.nc"
SOURCES = (
    ("gleam_e_mm_land_198001_202412_025_daily.nc", "E", "gleam_e_mm_eur_2001_2020_025_daily.nc"),
    ("mswep-v2-8_tp_mm_land_197901_202012_025_daily.nc", "mask", "mswep-v2-8_tp_mm_eur_2001_2020_025_daily.nc"),
)
EPS = 1e-4


def main():
    destinations = [output_path(filename) for _, _, filename in SOURCES]
    for destination in destinations:
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite {destination}")
    with xr.open_dataset(input_path(FLUXCOM)) as reference:
        lat, lon = reference.lat.values, reference.lon.values
        expected = pd.date_range(START, END, freq="D")
        if not pd.DatetimeIndex(reference.time.values).equals(expected):
            raise ValueError("European FLUXCOM must contain every day in 2001-2020 exactly once")

        for (source, variable, _), destination in zip(SOURCES, destinations):
            with xr.open_dataset(input_path(source)) as dataset:
                region = dataset[[variable]].sel(
                    time=slice(START, END),
                    lat=slice(float(lat.max()) + EPS, float(lat.min()) - EPS),
                    lon=slice(float(lon.min()) - EPS, float(lon.max()) + EPS),
                )
                if (region.sizes["lat"] != len(lat) or region.sizes["lon"] != len(lon)
                        or not np.allclose(region.lat.values, lat, rtol=0, atol=EPS)
                        or not np.allclose(region.lon.values, lon, rtol=0, atol=EPS)):
                    raise ValueError(f"{source} is not aligned with the European FLUXCOM grid")
                if not pd.DatetimeIndex(region.time.values).equals(expected):
                    raise ValueError(f"{source} must contain every day in 2001-2020 exactly once")
                region.to_netcdf(destination)
                print(f"Saved {destination}")


if __name__ == "__main__":
    main()
