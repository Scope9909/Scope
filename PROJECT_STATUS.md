# Current repository status

Updated 2026-11-11 from the actual files in this directory.

## Scope

The public package retains the frozen Validation/Test replay for PEMS03,
PEMS04, PEMS07, PEMS08, and PEMS-BAY. It now also contains a focused PEMS04
training entry for the frozen seed-1 Student configuration. Training and
model selection for the other four datasets, retrieval-state selection,
encoding search, multi-seed experiments, ablations, and historical outputs
remain excluded.

## Frozen replay verified

- Entry help: the four-dataset replay and PEMS-BAY replay passed.
- Preflight: 23 immutable replay assets and 15 prepared dataset files passed
  SHA-256/path checks.
- Small tests: preflight cache, ALSC codec, calibration, and low-rank predictor
  tests each passed 3/3.
- The public layout was reduced to the runtime modules imported by replay and
  tests. Both entry help commands, the 23-asset/15-data-file preflight, and all
  four small test groups passed after that migration.
- A post-migration PEMS03 Validation smoke replay completed on CPU with 5,237
  samples (raw MAE 13.213348, RMSE 21.471072, MAPE 0.125842). Its generated
  output directory was deleted after verification.
- The final GitHub audit repeated the complete PEMS04 chain on Python 3.11,
  PyTorch 2.0.0+cu117 and an RTX 3050: Validation completed first with 3,394
  samples (raw MAE 17.850632, RMSE 30.051027, MAPE 0.119606), followed by Test
  with 3,394 samples. Raw Test metrics were MAE 17.700058, RMSE 29.428931 and
  MAPE 0.120772; calibrated Test metrics were MAE 17.551138, RMSE 29.215066
  and MAPE 0.119989, matching the frozen record.
- The final code cleanup replaced two historical calibration CLIs with the
  minimal replay calibration module, removed unused readable training
  snapshots and an unreferenced utility module, and added three calibration
  contract tests. Numerical comparison against the former calibration code
  had zero difference, and the complete PEMS04 chain was repeated unchanged.
- Validation sample counts: PEMS03 5,237; PEMS04 3,394; PEMS07 5,640;
  PEMS08 3,566; PEMS-BAY 5,209.
- Test sample counts: PEMS03 5,237; PEMS04 3,394; PEMS07 5,640;
  PEMS08 3,567; PEMS-BAY 10,419.

Calibrated Test metrics from the fresh replay:

| Dataset | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| PEMS03 | 13.972812 | 23.882069 | 0.140516 |
| PEMS04 | 17.551138 | 29.215066 | 0.119989 |
| PEMS07 | 18.511486 | 31.880945 | 0.077272 |
| PEMS08 | 12.876751 | 22.487709 | 0.084893 |
| PEMS-BAY | 1.545914 | 3.497189 | 0.034644 |

Against the retained frozen reference, the maximum absolute metric difference
was `1.18e-4` (PEMS03 calibrated RMSE). Therefore the result is recorded as
numerically reproduced within `2e-4`, not bitwise reproduced.

## PEMS04 training workflow verified

The actual saved configurations establish this sequence:

1. dense base model: 100 epochs;
2. frozen-parent low-rank retrieval adapter: 20 epochs;
3. frozen-parent exact-match adapter: 20 epochs;
4. rank-256 factorized Student with KD: at most 30 epochs, Validation early
   stopping, seed 1.

The epoch-100 base model is not the reported `17.850632` Student result.

The new `training/pems04.py` entry passed:

- help and JSON configuration parsing;
- CUDA detection on PyTorch 2.0.0+cu117 and an RTX 3050 Laptop GPU;
- SHA-256 checks for all four derived training assets;
- loader construction with 10,181 Train and 3,394 Validation samples;
- strict parent loading and parameter freezing for all four stages;
- activation-aware rank-256 initialization (retained energy `0.99793651`);
- strict Teacher prediction alignment for Train/Validation/Test;
- one GPU forward/backward batch and one Validation batch for every stage.

The smoke tests wrote no checkpoint or result file. No fresh 170-epoch
end-to-end training was run, so no newly trained metric is claimed.

## Asset correction and exclusions

The PEMS-BAY Validation selection record in the working package did not match
the frozen SHA-256. The mismatching file was backed up, and the original file
was restored from the trusted 2026-11-07 workspace only after its hash matched
the frozen value `1ae0ff6a...d1c56` exactly. No source-code hash was changed to
hide a mismatch.

Excluded files were moved to
`_preparation/recovery_20261110/excluded/` for recovery. They are ignored by
the public package. Generated replay output and temporary test files were
deleted after verification.

Four PEMS04 training assets were copied locally from the trusted workspace
only after their configured SHA-256 values matched, and were verified again
after copying. They are under the ignored `datasets/` tree and are not staged
for Git.

## Remaining limitations

- Raw/prepared datasets are not redistributable from this repository.
- Public source and benchmark-download locations for all five datasets are
  documented in `DATA.md`. Those downloads are not drop-in replacements for
  the 15 prepared files: the exact timestamp features, frozen indices, and
  scalers still have to be supplied from a trusted preparation or release
  asset.
- The four derived PEMS04 training assets total about 432 MB (about 412 MiB)
  and are not in
  Git. Three exceed GitHub's ordinary per-file size limit. A GitHub Release or
  archival download URL still needs to be published and added to
  `TRAINING.md`.
- The fixed KD Teacher prediction cache is available to the Student workflow,
  but the separate Teacher-training and retrieval-cache construction pipeline
  is not public. Therefore raw-data-only exact regeneration is not claimed.
- The five-dataset numerical replay completed before the final directory
  migration. After migration, PEMS03 Validation and the complete PEMS04
  Validation-to-Test chain were repeated. A second complete five-dataset run
  is not claimed. Asset hashes were unchanged.
- The inherited implementation has an upstream licensing limitation described
  in `LICENSE`; legal attribution is the sole intentional old-name exception.
- The frozen replay repository was pushed previously. The new training files
  and documentation changes in this status update remain local and unpushed.
