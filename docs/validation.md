# Validation and remaining work

The Linux Python 3.12 environment in `requirements-linux-py312.lock` was used
for the checks below.

## Clean full-data run, 2026-11-24

- On the same Linux/WSL host, a new isolated Python 3.12.3 environment
  was installed from `requirements-linux-py312.lock`. `pip check` passed and
  all 20 tests passed. This was not a separate machine or clean Git checkout:
  the analysis used the then-uncommitted working tree.
- All six source inputs were downloaded afresh from Zenodo record 24502905
  and passed the published MD5 checks. Script 02 generated the Mediterranean
  FLUXCOM input; its SHA-256 matched the earlier local derived file.
- All 16 active analyses (03–07, 09–10, 12 and 14–21) exited successfully.
  Their 24 expected PNG files were readable and nonblank. The eight figures
  previously produced by full-data scripts 03, 04 and 12 were byte-identical.
  All 24 clean-run PNGs are included in `figures/`.
- [Source](clean-run-sources.sha256) and [figure](clean-run-figures.sha256)
  manifests identify this particular run. See [workflow.md](workflow.md)
  for the per-script output directory layout and checksum commands.
- Summed analysis wall time was about 50 minutes on this host. The largest
  individual-process peak RSS reported by `/usr/bin/time` was 9.26 GiB;
  concurrent workers can use additional memory. Downloaded inputs and the
  derived file occupied approximately 4.0 GiB, and the environment 485 MiB.
- Script 03 emitted two NumPy warnings about insufficient degrees of freedom
  in some slices. Several other scripts suppress warnings, so absence of
  warnings in their logs is not evidence that none occurred.

The input data, individual figures, process logs and environment metadata are
retained locally under `data/external/clean-reproduction-20261124/`, which is
excluded from Git. The manifests, not the large inputs or outputs, are intended
to be versioned. Independent verification on another machine and scientific
review are still needed; a successful run and PNG checksums do not establish
scientific correctness.

## 2001–2020 revision

- The complete synthetic suite passed (20 tests), including each of the 11
  active European scripts with altered data outside the analysis period.
  Their generated figures were unchanged by those alterations. Synthetic
  tests also check grid/time alignment and the Mediterranean event analysis.
- The new Mediterranean FLUXCOM data contain 7,305 daily timestamps on a
  44 × 200 grid. Their ET values and land fraction match the historical local
  Mediterranean file for 2001–2020; maximum coordinate difference was
  approximately 7.1e-15 degrees.
- The European GLEAM and MSWEP files were prepared locally on the 169 × 284
  FLUXCOM grid. Every value and coordinate matches the corresponding crop
  from the local global source for 2001–2020. Their Mediterranean overlaps
  also match the corresponding deposited Mediterranean products.
- The six source inputs and the historical Mediterranean FLUXCOM file match
  the MD5 checksums published by Zenodo record 24502905. These comparisons
  used existing local files, not a fresh download.
- Scripts 03 and 04 ran on local Mediterranean data for 2001–2020. Script 03
  reported `R=0.955`, `RMSE=0.376 mm/day`, `Bias=0.281 mm/day` and emitted
  NumPy warnings about insufficient degrees of freedom in some slices.
  Script 04 produced all four figures for the two products. These earlier
  local outputs were subsequently reproduced in the clean full-data run.
- Script 12 also ran on the full European inputs and produced its land-cover
  map for 4,280 April–October days in each product.

## Earlier checks (historical periods)

- The shared detection helpers were compared with the original functions from
  the affected scripts on synthetic data.
- Script 03 and its original version were run against the same local GLEAM
  and FLUXCOM Mediterranean inputs. All three PNG outputs had identical
  SHA-256 hashes between those two runs. Both runs reported `R=0.954`,
  `RMSE=0.377` and `Bias=0.280`; both emitted NumPy warnings about
  insufficient degrees of freedom in some slices.
- The pinned pyreadr could read the local mask table.

These checks do **not** establish independent reproduction on another
machine, scientific correctness, equivalence to historical figures, or that
the underlying datasets and mask can be recreated from original sources.
Before claiming independent end-to-end reproducibility:

1. Have an independent party obtain the exact inputs and run every required
   analysis on another machine, recording versions, warnings and resources.
2. Compare numerical results and regenerated figures with approved reference
   outputs, and investigate differences, including the intentionally fixed
   random seed in script 07.
3. Review the historical scripts' scientific choices, warning suppression,
   platform assumptions and input time/grid alignment with the project team.
4. If reproduction from original products is required, document upstream
   access, licenses, input checksums, target grid and mask preparation.
