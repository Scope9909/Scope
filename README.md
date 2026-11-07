# Scope

Research code for retrieval-augmented traffic forecasting and deployment-state
compression, built on TLAST. This repository contains historical experiment
implementations as well as the frozen paper configuration. A clean-machine,
end-to-end reproduction has **not yet been certified**.

## Paper pipeline

1. Train the retrieval-augmented Teacher (Hist2Res prediction chain).
2. Compute the **uncentered Train second moment**, initialize a low-rank
   Student, and distill from a frozen Teacher.
3. Freeze the Student; select a fixed SRM node set using Validation.
4. Select Q8/exact states under Validation budgets, then encode fields with
   reversible transforms. Accept mixed precision only when its complete file
   is smaller than the exact alternative. PEMS-BAY uses exact float32 coding.
5. Fit calibration using Validation. Evaluate the frozen configuration on Test.

Test must not be used for selecting nodes, quantization, seeds, or calibration.
Existing Test replay is not an independent blind test. State-file savings are
not total-system memory savings or inference speedups.

## Installation and read-only checks

Target Python: 3.11. Install PyTorch appropriate for your CUDA/CPU environment;
the requirements retain the project's torch 2.0.0 baseline. The complete
requirements have not been tested in a newly created environment.

```bash
pip install -r requirements.txt
python tools/preflight.py
python contrib3_index_codec_20261020/test_adaptive_lossless_state_codec.py
```

Preflight does not train, evaluate Test, download, or unpickle files. A passing
file check is not a numerical reproduction certificate. Do not load pickle or
checkpoint files from untrusted sources.

## Data and assets

Raw data is not redistributed here. Place each dataset at
`datasets/PEMS03/data.npz` (similarly PEMS04, PEMS07, PEMS08, PEMSBAY).
Time/index/scaler arrays and retrieval caches also need preparation; a raw
file alone is insufficient. Preserve the paper's splits and masks.

If you already have a trusted original experiment workspace, prepare the
local replay inputs without regenerating splits or running a model:

```bash
python tools/prepare_local_inputs.py --source <original-workspace>
# Review the preview, then copy with hash verification (conflicts are rejected):
python tools/prepare_local_inputs.py --source <original-workspace> --apply
python tools/preflight.py
```

This copies raw/prepared data and legacy replay inputs locally. It does not
publish datasets, download files, or execute Test. The private inventory is
written to the Git-ignored `_preparation/local_inputs_manifest.json`.

`tools/assets_manifest.json` describes the historical download inventory;
it includes auxiliary METRLA runs, not just the five paper datasets.
`tools/final_assets_manifest.json` records the locally verified final-run
checkpoints/configuration assets separately. The old `v1.0.0` Release is not
part of the rebuilt repository, and no complete replacement asset download
has been published. See `tools/final_runs.json` for exact run identities.

The historical downloader remains in source for provenance, but it has no
currently supported Release URL and is **not a complete paper reproduction
route**. Use a trusted original workspace for local inputs until a verified
asset bundle and clean-machine recipe are published.

## Source entry points

| Stage | Entry / evidence | Important qualification |
|---|---|---|
| Teacher | `run.py`, `models/TLAST_improved.py`, original Teacher configs | Select the verified Teacher config, not `strict_neighbor_k4`, which is an exploratory Student. |
| Second moment | `compute_predictor_activation_covariance.py` | Uses Train activations; takes `--exp-dir` and `--output`. |
| Student | `run.py`; `run_lite_r256_valcal_pipeline.py` | Existing pipeline supports PEMS03/07/08, not a universal five-dataset runner. |
| SRM | `paper_results/icde_retrieval_free_pilot_20261012/` | Contains selection and state construction code, not disposable results. |
| Encoding | `contrib3_index_codec_20261020/` | Beam search and reversible codec; preserve frozen recipes. |
| Calibration | `contrib3_index_codec_20261020/export_frozen_calibrators.py` | Validation-only fitting; output directory is required. |
| Four-dataset replay | `contrib3_index_codec_20261020/eval_adaptive_lossless_test.py` | Requires frozen assets and raw/prepared data; writes a new output directory. |
| PEMS-BAY | `pemsbay_full_20261027/prepare_state.py`, `evaluate_frozen.py` | Separate exact-float state workflow; do not substitute integer Q8. |

The following are **interface examples**, not a complete from-scratch recipe.
Replace placeholders with verified inputs. Do not run Test while developing
or tuning the system.

```bash
# run.py expects a path RELATIVE TO configurations/, without that prefix:
python run.py --config_file <verified-config-relative-to-configurations> --seed 1 --run_id <new-run-id>
python compute_analog_retrieval_cache.py --help
python compute_analog_exact_match_cache.py --help
python dump_teacher_predictions.py --exp_dir <teacher-run-directory> --output <teacher-predictions.npz> --splits train,val
python compute_predictor_activation_covariance.py --exp-dir <teacher-run-directory> --output <second-moment.npz>
# Optional Validation hyperparameter sweep, not the final fixed training recipe:
python run_lite_r256_kd_sweep.py --dry-run
# Only after frozen-asset and data preflight is resolved:
python contrib3_index_codec_20261020/export_frozen_calibrators.py --output <new-validation-calibrator-directory>
python contrib3_index_codec_20261020/eval_adaptive_lossless_test.py --output <new-test-replay-directory>
```

`srm_followup_validation_20261015/run_followup.py` is an attribution/timing
study, not the SRM selector. `benchmark_runtime_code_domain.py` is a legacy
runtime comparison, not the recommended final-system reproduction entry.

## Results, names, and provenance

Keep final metrics, ablations, calibration, frozen node sets and field recipes.
Do not delete `paper_results/` wholesale: scripts read inputs from it.
Dates remain in internal paths until a tested migration is available; they
are provenance identifiers, not different methods. Historical DARC/IASC/TLAST
identifiers remain in code/configs to preserve compatibility; public name: Scope.
Local preparation backups in `_preparation/` are ignored by Git.

See `README_UPSTREAM_TLAST.md` and `LICENSE` for upstream attribution and
existing license information. Preserve the TLAST citation from that README.
