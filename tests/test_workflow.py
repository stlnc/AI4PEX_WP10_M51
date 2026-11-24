import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import xarray as xr

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))

from _analysis import detect_exeves
from _alignment import validate_alignment
from _paths import annual_et_files, input_path, output_path


def test_detection_identifies_a_run_with_an_extreme_day():
    times = pd.DatetimeIndex([f"{year}-01-01" for year in range(2001, 2011)])
    data = np.column_stack(([0] * 7 + [1, 1, 3], range(10))).astype("float32")

    detected = detect_exeves(data, times)

    assert detected[:, 0].tolist() == [False] * 7 + [True] * 3
    assert detected[:, 1].tolist() == [False] * 8 + [True] * 2


def test_alignment_rejects_misaligned_time_or_grid():
    dataset = xr.Dataset(coords={"time": pd.date_range("2001-01-01", periods=2), "lat": [1.0, 0.0], "lon": [0.0, 1.0]})
    validate_alignment(dataset, dataset.assign_coords(lon=dataset.lon + 1e-8))
    with pytest.raises(ValueError, match="time coordinates"):
        validate_alignment(dataset, dataset.assign_coords(time=pd.date_range("2001-01-02", periods=2)))
    with pytest.raises(ValueError, match="lon coordinates"):
        validate_alignment(dataset, dataset.assign_coords(lon=dataset.lon + 0.25))


def test_paths_keep_input_and_output_separate(tmp_path, monkeypatch):
    data = tmp_path / "input"
    output = tmp_path / "output"
    monkeypatch.setenv("EXEVES_DATA_DIR", str(data))
    monkeypatch.setenv("EXEVES_OUTPUT_DIR", str(output))

    assert input_path("file.nc") == data / "file.nc"
    assert output_path("figure.png") == output / "figure.png"
    assert output.is_dir()
    assert not data.exists()


def test_downloader_help_does_not_require_authentication():
    result = subprocess.run(
        [sys.executable, str(CODE / "01_download_xbase_from_icos.py"), "--help"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--outdir" in result.stdout
    assert "(default: 2020)" in result.stdout


def test_missing_annual_files_are_reported(tmp_path, monkeypatch):
    monkeypatch.setenv("EXEVES_DATA_DIR", str(tmp_path))
    (tmp_path / "ET_2001_025_daily.nc").touch()

    with pytest.raises(FileNotFoundError, match="ET_2002_025_daily.nc"):
        annual_et_files(2001, 2002)


def test_regional_crop_reads_input_and_writes_output(tmp_path):
    data = tmp_path / "input"
    data.mkdir()
    output = tmp_path / "output"
    coords = {"time": pd.to_datetime(["2001-01-01"]), "lat": [1.0, 0.0], "lon": [0.0, 1.0]}
    xr.Dataset({"E": (("time", "lat", "lon"), np.ones((1, 2, 2)))}, coords=coords).to_netcdf(
        data / "gleam_e_mm_med_198001_202412_025_daily.nc"
    )
    times = pd.date_range("2000-12-31", "2021-01-01")
    european_coords = {"time": times, "lat": [2.0, 1.0, 0.0, -1.0], "lon": [-1.0, 0.0, 1.0, 2.0]}
    values = np.broadcast_to(np.arange(16).reshape(1, 4, 4), (len(times), 4, 4)).copy()
    xr.Dataset(
        {
            "ET": (("time", "lat", "lon"), values),
            "land_fraction": (("lat", "lon"), np.ones((4, 4))),
        },
        coords=european_coords,
    ).to_netcdf(data / "ET_2001_2020_025_daily_eur.nc")
    env = os.environ.copy()
    env.update(EXEVES_DATA_DIR=str(data), EXEVES_OUTPUT_DIR=str(output))

    subprocess.run([sys.executable, str(CODE / "02_crop_and_merge_ET.py")], env=env, check=True)

    result = output / "ET_2001_2020_025_daily_med.nc"
    with xr.open_dataset(result) as ds:
        np.testing.assert_array_equal(ds.time.values, pd.date_range("2001-01-01", "2020-12-31").values)
        np.testing.assert_array_equal(ds["ET"].values, values[1:-1, 1:3, 1:3])
        np.testing.assert_array_equal(ds["land_fraction"].values, np.ones((2, 2)))
    assert not (data / result.name).exists()

    source = data / "ET_2001_2020_025_daily_eur.nc"
    with xr.open_dataset(source) as ds:
        incomplete = ds.isel(time=slice(None, -2)).load()
    incomplete.to_netcdf(source, mode="w")
    failed = subprocess.run(
        [sys.executable, str(CODE / "02_crop_and_merge_ET.py")],
        env=env, capture_output=True, text=True,
    )
    assert failed.returncode != 0
    assert "every day in 2001-2020" in failed.stderr


def test_european_subsets_match_reference_grid_and_period(tmp_path):
    data = tmp_path / "input"
    data.mkdir()
    output = tmp_path / "output"
    times = pd.date_range("2000-12-31", "2021-01-01")
    coords = {"time": times, "lat": [2.0, 1.0, 0.0, -1.0], "lon": [-1.0, 0.0, 1.0, 2.0]}
    reference = xr.Dataset(coords={"time": times[1:-1], "lat": [1.0, 0.0], "lon": [0.0, 1.0]})
    reference.to_netcdf(data / "ET_2001_2020_025_daily_eur.nc")
    values = np.broadcast_to(np.arange(16).reshape(1, 4, 4), (len(times), 4, 4)).copy()
    sources = (
        ("gleam_e_mm_land_198001_202412_025_daily.nc", "E", "gleam_e_mm_eur_2001_2020_025_daily.nc"),
        ("mswep-v2-8_tp_mm_land_197901_202012_025_daily.nc", "mask", "mswep-v2-8_tp_mm_eur_2001_2020_025_daily.nc"),
    )
    for source, variable, _ in sources:
        xr.Dataset({variable: (("time", "lat", "lon"), values)}, coords=coords).to_netcdf(data / source)
    env = os.environ.copy()
    env.update(EXEVES_DATA_DIR=str(data), EXEVES_OUTPUT_DIR=str(output))
    subprocess.run([sys.executable, str(CODE / "08b_prepare_europe_inputs.py")], env=env, check=True)
    for _, variable, filename in sources:
        with xr.open_dataset(output / filename) as ds:
            np.testing.assert_array_equal(ds.time.values, times[1:-1].values)
            np.testing.assert_array_equal(ds[variable].values, values[1:-1, 1:3, 1:3])
    failed = subprocess.run(
        [sys.executable, str(CODE / "08b_prepare_europe_inputs.py")],
        env=env, capture_output=True, text=True,
    )
    assert failed.returncode != 0
    assert "Refusing to overwrite" in failed.stderr


def test_exeves_analysis_produces_figures_from_synthetic_data(tmp_path):
    data = tmp_path / "input"
    data.mkdir()
    output = tmp_path / "output"
    times = pd.date_range("1980-01-01", "2003-12-31")
    coords = {"time": times, "lat": [2.0, 1.0, 0.0], "lon": [0.0, 1.0, 2.0, 3.0]}
    rng = np.random.default_rng(42)
    shape = (len(times), 3, 4)
    sources = (
        ("gleam_e_mm_med_198001_202412_025_daily.nc", "E"),
        ("mswep-v2-8_tp_mm_med_197901_202012_025_daily.nc", "mask"),
    )
    for filename, variable in sources:
        xr.Dataset({variable: (("time", "lat", "lon"), rng.random(shape))}, coords=coords).to_netcdf(data / filename)
    flux = xr.Dataset(
        {
            "ET": (("time", "lat", "lon"), rng.random(shape)),
            "land_fraction": (("lat", "lon"), np.ones((3, 4))),
        },
        coords=coords,
    )
    flux.sel(time=slice("2001-01-01", None)).to_netcdf(data / "ET_2001_2020_025_daily_med.nc")
    env = os.environ.copy()
    env.update(EXEVES_DATA_DIR=str(data), EXEVES_OUTPUT_DIR=str(output), MPLBACKEND="Agg")

    first = subprocess.run(
        [sys.executable, str(CODE / "04_exeves_analysis.py")], env=env, check=True, capture_output=True, text=True,
    )
    assert "1980–2020" not in first.stdout
    figures = ("exeves_q95_density.png", "exeves_q95_map.png", "exeves_pct_density.png", "exeves_lag.png")
    for name in figures:
        assert (output / name).stat().st_size > 0

    for filename, variable in sources:
        path = data / filename
        with xr.open_dataset(path) as ds:
            changed = ds.load()
        changed[variable] = changed[variable].where(changed.time >= np.datetime64("2001-01-01"), 1000000)
        changed.to_netcdf(path, mode="w")
    second_output = tmp_path / "output2"
    env["EXEVES_OUTPUT_DIR"] = str(second_output)
    subprocess.run([sys.executable, str(CODE / "04_exeves_analysis.py")], env=env, check=True)
    for name in figures:
        assert (output / name).read_bytes() == (second_output / name).read_bytes()
