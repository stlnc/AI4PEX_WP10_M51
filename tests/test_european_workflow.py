import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pyreadr
import pytest
import xarray as xr

CODE = Path(__file__).resolve().parents[1] / "code"


@pytest.fixture(scope="module")
def european_inputs(tmp_path_factory):
    root = tmp_path_factory.mktemp("europe")
    rng = np.random.default_rng(42)
    dates = pd.DatetimeIndex(
        [pd.Timestamp("2000-12-31"), *pd.date_range("2001-01-01", "2003-12-31"), pd.Timestamp("2021-01-01")]
    )
    lat = np.arange(13, -1, -1) * 0.25
    lon = np.arange(14) * 0.25
    coords = {"time": dates, "lat": lat, "lon": lon}
    classes = ["Forests", "Croplands", "Savannas", "Grasslands", "Shrublands", "Barren"]
    masks = pd.DataFrame([
        {
            "lat": lat[i], "lon": lon[j], "ipcc_continent": "EUROPE", "land_mask": "land",
            "land_cover_short_class": (
                "Forests" if (i in (2, 11) or j in (2, 11))
                and classes[((i - 2) * 10 + j - 2) % len(classes)] == "Grasslands"
                else classes[((i - 2) * 10 + j - 2) % len(classes)]
            ),
        }
        for i in range(2, 12) for j in range(2, 12)
    ])
    et = rng.random((len(dates), len(lat), len(lon)), dtype=np.float32) + 0.1
    rain = rng.random(et.shape, dtype=np.float32) + 0.1
    flux = rng.random(et.shape, dtype=np.float32) + 0.1
    for outside_value in (1, 1000000):
        data = root / str(outside_value)
        data.mkdir()
        pyreadr.write_rdata(str(data / "masks.rda"), masks, df_name="masks")
        for filename, variable, values in (
            ("gleam_e_mm_eur_2001_2020_025_daily.nc", "E", et),
            ("mswep-v2-8_tp_mm_eur_2001_2020_025_daily.nc", "mask", rain),
        ):
            updated = values.copy()
            updated[[0, -1]] = outside_value
            xr.Dataset({variable: (("time", "lat", "lon"), updated)}, coords=coords).to_netcdf(data / filename)
        xr.Dataset(
            {"ET": (("time", "lat", "lon"), flux), "land_fraction": (("lat", "lon"), np.ones((14, 14)))},
            coords=coords,
        ).sel(time=slice("2001-01-01", "2020-12-31")).to_netcdf(
            data / "ET_2001_2020_025_daily_eur.nc",
            encoding={"ET": {"chunksizes": (400, 7, 7), "zlib": True}},
        )
    return root


SCRIPTS = [
    ("09_exeves_europe_lc.py", "exeves_europe_q95.png"),
    ("10_exeves_europe_q95_map.py", "exeves_europe_q95_map.png"),
    ("12_exeves_europe_q95_map_lc.py", "exeves_europe_q95_map_lc.png"),
    ("14_exeves_europe_prec_quantile_scatter_grasslands.py", "exeves_europe_prec_quantile_scatter_grasslands.png"),
    ("15_exeves_europe_prec_quantile_scatter_alllc.py", "exeves_europe_prec_quantile_scatter_alllc.png"),
    ("16_exeves_europe_prec_lag_alllc.py", "exeves_europe_prec_lag_alllc.png"),
    ("17_exeves_europe_exeve_config_scatter_alllc.py", "exeves_europe_exeve_config_scatter_alllc.png"),
    ("18_exeves_europe_jaccard_prec_alllc.py", "exeves_europe_jaccard_prec_alllc.png"),
    ("19_exeves_europe_jaccard_exeve_config_alllc.py", "exeves_europe_jaccard_exeve_config_alllc.png"),
    ("20_exeves_europe_jaccard_vs_freq_alllc.py", "exeves_europe_jaccard_vs_freq_alllc.png"),
    ("21_exeves_europe_freq_vs_prec_config_alllc.py", "exeves_europe_freq_vs_prec_config_alllc.png"),
]


def test_native_chunk_seasonal_read_preserves_values(european_inputs):
    source = european_inputs / "1" / "ET_2001_2020_025_daily_eur.nc"
    with xr.open_dataset(source) as plain, xr.open_dataset(source, chunks={}) as chunked:
        months = plain.time.dt.month.isin([4, 5, 6, 7, 8, 9, 10])
        expected = plain.ET.where(months, drop=True).values
        actual = chunked.ET.where(months, drop=True).compute().values
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(("script", "figure"), SCRIPTS)
def test_european_analysis_ignores_outside_period(european_inputs, tmp_path, script, figure):
    results = []
    for value in (1, 1000000):
        output = tmp_path / str(value)
        env = os.environ.copy()
        env.update(
            EXEVES_DATA_DIR=str(european_inputs / str(value)), EXEVES_OUTPUT_DIR=str(output),
            MPLBACKEND="Agg", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
        )
        result = subprocess.run(
            [sys.executable, str(CODE / script)], env=env, capture_output=True, text=True, timeout=180,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "1980–2020" not in result.stdout
        assert "S1:" not in result.stdout
        assert (output / figure).stat().st_size > 0
        results.append((output / figure).read_bytes())
    assert results[0] == results[1]
