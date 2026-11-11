# AirTail migration verification report

| Check | Result | Evidence |
|---|---|---|
| Source preservation | PASS | Source HEAD, branch, porcelain status, and recursive file inventory were identical at both endpoints. The inventory records paths, sizes and modification times, with content comparisons for the selected source/config/checkpoint file types. |
| BTHSA data contract | PASS | Complete panel and station CSV match source manifests. Train scalers, station order, coordinates, neighbors and the first/last validation windows were identical. Both paths used the same Windows raw reader at the adapter boundary. |
| QCMM, PMG-RD and SA-EVT location and parity | PASS | CPU reference comparison of 1,051 captured tensors/scalars. Maximum absolute, mean absolute and relative differences were all 0. |
| Teacher/Student separation | PASS | CPU Teacher training step, strict checkpoint load, frozen Teacher in Student update and a separate Student process were exercised. |
| History-only deployment | PASS | Prediction accepts a history tensor dictionary and static graph. The command completed using only a Student checkpoint and those tensors. |
| Full-dataset training/evaluation workflow | NOT_RUN | Teacher training, Student distillation and evaluation entry points are provided. End-to-end formal dataset runs remain unverified. |
| Official checkpoint | NOT_RUN | BTHSA files are truncated and do not match the run hashes. No substitute weights were used. |
| Main table and CUDA reproduction | NOT_RUN | The source evidence ties the architecture to the BTHSA E3 run. The dirty historical source tree and three final checkpoint/metric chains could not be reconstructed. |
| YRD data reading | PASS | Missing variable units are optional. Complete local data: 127 stations, 49,042 training windows and 6,989 validation windows; independent train scalers and validation-window comparisons match exactly. CPU Student predictions are finite and monotone. Nineteen focused tests pass; formal CUDA training remains unverified. |
| GansuAir | NOT_RUN | Raw data and aligned sidecars are absent locally. |
| Publication | NOT_RUN | Redistribution rights for the code remain to be confirmed. |

The independent package is arranged as src/airtail/{data,models,training,evaluation}, with dataset manifests, stage configurations, tests and instructions. No baseline implementations, sweep scripts, raw datasets or training artifacts belong to the package. The preserved head parameter slots maintain strict state-dict compatibility and initialization RNG consumption. Scientific comparisons use only train and validation data.

See REPRODUCIBILITY.md for exact split indices, loss and metric definitions. The separate migration records contain the old-to-new source map.

## Acceptance

| # | Item | Status |
|---|---|---|
| 1 | Original source unchanged | PASS |
| 2 | Separate sibling repository | PASS |
| 3 | Only verified final implementations and required support | FAIL | 
| 4 | Runtime independence from source repository | PASS |
| 5 | QCMM, PMG-RD, SA-EVT mapped to paper | PASS |
| 6 | Independent models and frozen Teacher | PASS |
| 7 | History-only Student deployment | PASS |
| 8 | Formal checkpoint strict load and numerical equivalence | NOT_RUN |
| 9 | Formal training and evaluation protocol across all datasets | NOT_RUN |
| 10 | Offline package installation | PASS |

Item 3 remains incomplete at the submission-wide level. The release is based on the traceable BTHSA E3 execution path. Region-specific formal run provenance for YRD and GansuAir is missing and several shared implementation switches and checkpoint compatibility keys are still present. The baseline packages and dedicated experiment variants were excluded. Item 10 was checked by building a wheel with setuptools, installing it with pip into a separate directory with dependencies preinstalled, and checking the installed package and command interface. Windows path length blocked pip's temporary wheel-build path for the direct source install in the staging directory. CUDA and a new dependency environment were outside the verified scope.

## Dataset distribution

The README provides the complete BTHSA training and evaluation commands and the expected data layout. The author-provided dataset download folder is on [Google Drive](https://drive.google.com/drive/folders/122qhlnZHPcZR_qM4r5mE2mXtDAKk4QCQ).
