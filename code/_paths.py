"""Resolve input and output files without changing the working directory."""

import os
from pathlib import Path


def input_path(name):
    return Path(os.environ.get("EXEVES_DATA_DIR", ".")).resolve() / name


def output_path(name):
    directory = Path(os.environ.get("EXEVES_OUTPUT_DIR", ".")).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return directory / name


def annual_et_files(start_year, end_year):
    files = [input_path(f"ET_{year}_025_daily.nc") for year in range(start_year, end_year + 1)]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing annual FLUXCOM ET files:\n" + "\n".join(missing))
    return files
