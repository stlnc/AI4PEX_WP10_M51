# Data and workflow notes

The source working folder contains the following files. They are deliberately
excluded from this repository; acquire or create them separately, subject to
their providers' terms. The filename alone does not establish exact product
version, license or processing history.

| File | Description |
| --- | --- |
| `ET_2001_2020_025_daily.nc` | FLUXCOM daily ET, 2001-2020 |
| `ET_2001_2020_025_daily_eur.nc` | European FLUXCOM ET subset |
| `ET_2001_2021_025_daily_med.nc` | Mediterranean FLUXCOM ET subset |
| `ET_2021_025_daily.nc` | FLUXCOM daily ET, 2021 |
| `gleam_e_mm_land_198001_202412_025_daily.nc` | GLEAM land ET |
| `gleam_e_mm_med_198001_202412_025_daily.nc` | Mediterranean GLEAM ET subset |
| `mswep-v2-8_tp_mm_land_197901_202012_025_daily.nc` | MSWEP land precipitation |
| `mswep-v2-8_tp_mm_med_197901_202012_025_daily.nc` | Mediterranean MSWEP precipitation subset |
| `masks.rda` | Land-cover/region masks used by several scripts |

`code/download_xbase_from_icos.py` contains ICOS Carbon Portal links for
FLUXCOM-X-BASE products and may require ICOS authentication. Read its help
and check the data provider's conditions before downloading.
`code/crop_and_merge_ET.py` and `code/merge_fluxcom_eur.py` expect annual
FLUXCOM files and generate subsets; their required inputs are specified in
the scripts.

The exact source and processing steps for the other datasets and `masks.rda`
have not yet been established in this repository. Add verified URLs, versions,
licenses, citations, and instructions to recreate derived files when available.

## Working directory

Scripts generally open bare filenames from the working directory and save
figures there. Place the input files needed by a particular script together
in a local working directory, or adjust that script's input paths. Outputs may
overwrite files with the same names. The checked-in `figures/` directory is a
snapshot; scripts do not automatically update it.

Some analyses open global daily datasets and may require substantial disk
space and memory. Not every script has been validated against every dataset.
