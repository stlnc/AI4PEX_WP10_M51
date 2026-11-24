# Workflow and script order

Numbers follow original last-modified timestamps, not proven creation dates
or execution order. Each script can have different required inputs; inspect
its input constants and methodology before running it.

| Number | Purpose |
| --- | --- |
| 01 | Download FLUXCOM-X-BASE annual products from ICOS |
| 02 | Crop Mediterranean ET from the deposited European FLUXCOM file |
| 03 | Compare Mediterranean GLEAM and FLUXCOM ET |
| 04 | Analyze ExEvEs and precipitation in the Mediterranean |
| 05 | Compare GLEAM and FLUXCOM ExEvEs |
| 06 | Map FLUXCOM ExEvEs and heavy precipitation |
| 07 | Plot a randomly selected cell with fixed seed 42 |
| 08 | Optional legacy route: crop and merge annual ET for Europe |
| 08b | Optional: regenerate European GLEAM and MSWEP subsets from local global files |
| 09-10 | European land-cover distributions and maps |
| 11 | Historical global grassland analysis, archived in `code/legacy/` |
| 12 | European heavy-precipitation map by land cover |
| 13 | Historical global land-cover analysis, archived in `code/legacy/` |
| 14-16 | European precipitation threshold and lag analyses |
| 17-21 | European sensitivity and product-agreement analyses |

The active analyses use 2001–2020. Precipitation thresholds use the same
period; scripts differ in whether they use all days or April–October wet
days. The global scripts are retained for historical reference only and are
not part of this workflow. Optional step 08 still requires 2001–2020 annual
ET files and its time contents have not been validated.

Download the regional inputs from the published record and use step 02 to
prepare Mediterranean FLUXCOM; see [data.md](data.md) for commands and
checksums. Step 08b is not required when using the published European subsets.
It can regenerate them from the original local global files and refuses to
overwrite existing subsets. The active analyses do not require global inputs.

## Run all active analyses

After preparing the inputs and activating the environment, choose an absolute
path for an empty output directory. This runs each analysis in its own
subdirectory and stops if a command fails or a subdirectory already exists:

```bash
export EXEVES_DATA_DIR="/path/to/input-data"
export OUTPUT_ROOT="/path/to/empty/output"
export MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
(
  set -eu
  mkdir -p "$OUTPUT_ROOT"
  for script in code/0[3-7]_*.py code/09_*.py code/10_*.py code/12_*.py \
                code/1[4-9]_*.py code/2[01]_*.py; do
    name=$(basename "$script" .py)
    mkdir "$OUTPUT_ROOT/$name"
    EXEVES_OUTPUT_DIR="$OUTPUT_ROOT/$name" python "$script"
  done
)
```

For comparison with the documented Linux/WSL run, from the repository root:

```bash
sha256sum -c docs/clean-run-sources.sha256
(cd "$OUTPUT_ROOT" && sha256sum -c "/absolute/path/to/repo/docs/clean-run-figures.sha256")
```

The figure manifest is an exact-byte baseline for one environment. Different
fonts, graphics libraries, platforms or software versions can change PNG
bytes without necessarily changing the scientific results. Investigate
differences rather than treating a checksum mismatch alone as scientific
failure.
