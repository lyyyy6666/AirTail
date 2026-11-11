# Reproducibility

## BTHSA protocol

The released protocol uses seed 2024 for Python, NumPy, PyTorch, CUDA and training-data shuffling. The model training seed takes precedence over the dataset loader seed. Saved configurations record the effective seed, and resumption requires matching model and loader seeds. Comparing paper metrics under this seed requires fresh Teacher and Student training.

The complete 70,128-hour panel uses chronological index intervals [0, 49089), [49089, 56101), [56101, 70128). A window belongs to a split when its entire forecast target is within the split. History can precede the boundary. There are 49,042 training, 6,989 validation, and 14,004 test windows with 24 observed and 24 predicted hours. Source manifest timestamps and station ordering are preserved. Finite observed values, including zero, contribute to the training-only population mean and variance. Missing targets are masked, and normalized missing values are filled with zero. Raw wind must be finite. No window-length comparison was performed.

Teacher training uses fixed, equally weighted pinball across seven levels. Student training uses equal total weight for the bulk and tail sets, and a five-epoch linear warm-up of the row-centered, gain-scaled relation SmoothL1 objective to 0.50. Query rows 4–11 use all 12 keys, in region-major and scale-minor order. Teacher checkpoint loading is strict. Student distillation freezes the Teacher and uses it in evaluation mode. Teacher receives eight real forecast-period weather variables and Student uses historical inputs only. Checkpoints save model and optimizer states, selected epoch, RNG state, config snapshots and run metadata. Checkpoint selection uses standardized validation pinball. The configured scheduler is disabled.

The historical evaluator accumulates masked standardized pinball, per-quantile empirical coverage, crossings, and median MAE/RMSE in concentration units. The optional manuscript metrics first inverse-transform predictions and observations. Q-Risk is twice aggregate pinball divided by aggregate absolute target concentration, averaged across seven quantiles. PICP is inclusive coverage of the 0.50–0.99 interval. QIS averages interval width plus positive lower/upper misses divided by 0.50/0.01, respectively.

## Verification scope

- CPU reference comparison of 1,051 tensors and scalars, with zero maximum absolute, mean absolute and relative error. The fixture uses two samples, 24 input steps, four stations and three forecast steps. It captures trained-step losses and weights, parameter gradients, intermediate module outputs and Student evaluation metrics. The weights are initialized deterministically for this comparison.
- Six focused tests cover train-only scaling, validation target boundaries, temporal-prefix causality, PMG-RD rows, the two one-sided SA-EVT jets, strict Student state loading, the history-only signature and manuscript metrics. The SA-EVT test checks value, slope and curvature close to 0.90 in float64.
- BTHSA real validation window indices 0 and 6988 match the source preprocessing at zero difference using the shared Windows HDF5 adapter seam. Station IDs, train-fitted statistics, coordinates and neighbors match. Student forward computation on the first real validation window produces finite, monotone output of shape [1,24,228,7] with initialized weights.
- The checked local BTHSA Teacher and Student files are incomplete. Full checkpoint parity, CUDA extension parity, and all main-table test numbers remain unverified.
- YRD variable units are optional descriptive metadata; missing O3.units is retained as None without changing observations or normalization. The complete local panel loads 127 stations, 49,042 training windows and 6,989 validation windows. Training-only scaler statistics and the first/last validation windows match independent raw-data calculations exactly. An initialized Student using the CPU reference backend produces finite, monotone predictions of shape [1,24,127,7]. Only training and validation data were read; full CUDA training remains unverified. The raw GansuAir panel and aligned sidecars were unavailable locally.
- The parameter containers under the SA-EVT head's direct state-dict keys are retained to preserve recorded parameter names and initialization RNG consumption. They have no prediction forward path.

The provided manifests are the authoritative identity records for data files. A licensed data copy and valid formal checkpoints are required to reproduce the paper tables.
 
## Package verification

The six focused tests were run using Python unittest. Full-dataset training and final test-set evaluation remain unverified.

Offline installation was verified by building the wheel with `setuptools.build_meta.build_wheel`, installing it with `pip install --no-deps --no-index --target` into a separate directory, and using the preinstalled Torch, NumPy, PyYAML and h5py dependencies.

## Dataset distribution

The author-provided dataset download folder is on [Google Drive](https://drive.google.com/drive/folders/122qhlnZHPcZR_qM4r5mE2mXtDAKk4QCQ). The README lists the required archive layout and full-dataset commands. The manifests record the expected file identities.

Command-entry regression used the first real BTHSA training window and a separate first validation window across all 228 stations. Before/after Teacher and Student updates, optimizer states, training histories and validation metrics matched exactly, including 970 tensors. Older-format saved models also loaded successfully. Nine focused tests passed.

Data loading checks file availability, file sizes, station alignment and tensor dimensions. Student training saves the frozen Teacher parameters in its training file for direct comparison when resuming. Earlier Student files can still be evaluated, but restarting Student training is required when the saved file lacks those Teacher parameters.

After simplifying data loading and saved-model handling, twelve focused tests passed. A real-window comparison again matched 970 model and optimizer tensors exactly. New Student files resumed successfully, and earlier Student files retained identical validation metrics.

Seed alignment passed fifteen focused tests. The first real BTHSA training batch, with 16 samples over 228 stations, matched the sample order of a generator initialized with seed 2024.

YRD reading verification passed nineteen focused tests, including missing optional units, preserved existing units, required time units and required variables. The original source repository and dataset files were not modified.
