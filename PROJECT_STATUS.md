# Current repository status

Updated 2026-11-11 from the actual files in this directory.

## Scope

The public package retains only the frozen Validation/Test replay for PEMS03,
PEMS04, PEMS07, PEMS08, and PEMS-BAY. Training, model selection, encoding
search, multi-seed experiments, ablations, and historical outputs are excluded.

## Verified

- Entry help: the four-dataset replay and PEMS-BAY replay passed.
- Preflight: 23 immutable replay assets and 15 prepared dataset files passed
  SHA-256/path checks.
- Small tests: preflight cache, ALSC codec, calibration, and low-rank predictor
  tests each passed 3/3.
- The public layout was subsequently reduced to `assets/`, `configs/`,
  `replay/`, `tests/`, and the runtime modules they import. Both entry help
  commands, the 23-asset/15-data-file preflight, and all four small test
  groups passed after this path migration.
- A post-migration PEMS03 Validation smoke replay also completed on CPU with
  5,237 samples (raw MAE 13.213348, RMSE 21.471072, MAPE 0.125842). Its
  generated output directory was deleted after verification.
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

## Asset correction and exclusions

The PEMS-BAY Validation selection record in the working package did not match
the frozen SHA-256. The mismatching file was backed up, and the original file
was restored from the trusted 2026-11-07 workspace only after its hash matched
the frozen value `1ae0ff6a...d1c56` exactly. No source-code hash was changed to
hide a mismatch.

Excluded files were moved to
`_preparation/recovery_20261110/excluded/` for recovery. They are ignored by
the public package. Generated replay output was deleted after its metrics were
recorded.

## Remaining limitations

- Raw/prepared datasets are not redistributable from this repository.
- Public source and benchmark-download locations for all five datasets are
  documented in `DATA.md`. Those downloads are not drop-in replacements for
  the 15 prepared files: the exact timestamp features, frozen indices, and
  scalers still have to be supplied from a trusted preparation or release
  asset.
- The five-dataset numerical replay above completed before the final directory
  migration. After migration, PEMS03 Validation and the complete PEMS04
  Validation-to-Test chain were repeated. A second complete five-dataset run
  is not claimed. Asset hashes were unchanged.
- The inherited implementation has an upstream licensing limitation described
  in `LICENSE`; legal attribution is the sole intentional old-name exception.
- No GitHub push was performed.
