# AI4PEX WP10: Extreme Evaporation Events

Analysis scripts and figures exploring extreme evaporation events (ExEvEs)
and their relationship to precipitation, using GLEAM, FLUXCOM and MSWEP data.

## Contents

- `code/`: Python analysis, plotting and data preparation scripts.
- `figures/`: existing PNG outputs from the original working folder.
- `docs/data.md`: input filenames, provenance notes and data handling.
- `data/external/`: instructions for data kept outside Git.

The original data files are not included. Several exceed GitHub's 100 MB
per-file limit, and the full collection is very large. Check the original
providers' access, citation and redistribution terms before sharing data.

## Running an analysis

The scripts use filenames relative to the **current working directory** and
write their output there. Run them from a working directory containing the
required inputs; inspect the input constants at the top of each script first.
For example, in PowerShell, replace the two paths with your own:

```powershell
$repo = '\\wsl.localhost\Ubuntu\home\stlnc\AI4PEX_WP10_M51'
$work = '\\wsl.localhost\Ubuntu\home\stlnc\ai4pex_wp10\exeves\data\fluxcom'
Set-Location -LiteralPath $work
python (Join-Path $repo 'code\exeves_analysis.py')
```

This example writes figures in `$work` and may overwrite existing outputs.
Run in a separate working directory if you need to preserve them. See
`docs/data.md` for the expected input names and important limitations.

The scripts import Python packages including NumPy, pandas, xarray,
Matplotlib, SciPy, pyreadr, joblib and, for the ICOS downloader, icoscp-core.
Some xarray operations use Dask and require a compatible NetCDF backend.
An exact, tested environment has not yet been recorded.

## Before publication

Document dataset versions, source URLs, licenses, citations, mask provenance,
and the software environment. Verify that every script and figure may be
redistributed. The figures are existing outputs, not freshly regenerated here.
