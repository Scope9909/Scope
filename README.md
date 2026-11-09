# Scope

Scope is a retrieval-augmented traffic forecasting and deployment-state
compression method. It combines a retrieval-augmented Teacher, second-moment
low-rank Student initialization and distillation, Validation-selected retrieval
state, reversible state coding, and calibration. The implementation builds on
the TLAST codebase; the Scope method and its contributions are described below.
This repository contains historical experiment implementations as well as the
frozen paper configuration. A clean-machine, end-to-end reproduction has
**not yet been certified**.

## Scope names and compatibility

**Scope is the project name, developed on the TLAST_improved codebase.**
The original TLAST baseline and improved-model experiment lineage are retained.
Start from
[`models/Scope.py`](models/Scope.py), [`scope.py`](scope.py), and the five
frozen seed-1 configurations listed below.

| Path or name | Meaning | How to use it |
|---|---|---|
| `models/Scope.py` | Current Scope model implementation | Use for official Scope workflows |
| `EXP_experiments/Scope/` | Five frozen Scope seed-1 run records | Start here for final-run verification |
| `models/TLAST.py` | Retained upstream baseline implementation | Keep for explicit historical baseline runs |
| `EXP_experiments/TLAST/` and `EXP_experiments/TLAST_improved/` | Historical experiment IDs and saved paths | Do not rename: existing configs and checkpoints may refer to them |
| `TLAST_improved` in old configs | Improved-model lineage and compatibility identifier mapped to `models/Scope.py` by `run.py` | Preserve to reproduce the original model and experiments |
| `README_UPSTREAM_TLAST.md` and `LICENSE` | Upstream attribution and license | Preserve when redistributing |

The seed-1 cleanup is a repository organization choice, **not** proof that
historical multi-seed reproduction is supported after deleting its outputs.
Do not rewrite frozen file paths or checkpoint identifiers solely to remove
the word `TLAST`.

## Start here

For the final paper configuration, first identify the frozen run in
[`tools/final_runs.json`](tools/final_runs.json), then follow the matching
module entry below. The table links to source code, not to a single command
that reproduces every experiment.

| Scope component | Main source entry |
|---|---|
| Reference Forecast Branch / Teacher | [`models/Scope.py`](models/Scope.py), [`run.py`](run.py) |
| Input-Aware Student Compression (ISC) | [`compute_predictor_activation_covariance.py`](compute_predictor_activation_covariance.py), [`run_lite_r256_valcal_pipeline.py`](run_lite_r256_valcal_pipeline.py) |
| Sparse Residual Memory (SRM) | [`paper_results/icde_retrieval_free_pilot/`](paper_results/icde_retrieval_free_pilot/) |
| Validation-Constrained State Coding (ValSC) | [`contrib3_index_codec/`](contrib3_index_codec/) |
| Post-Compression Calibration | [`export_frozen_calibrators.py`](contrib3_index_codec/export_frozen_calibrators.py) |
| Frozen evaluation | [`eval_adaptive_lossless_test.py`](contrib3_index_codec/eval_adaptive_lossless_test.py) (PEMS03/04/07/08); [`pemsbay_full/`](pemsbay_full/) (PEMS-BAY) |

The five frozen run configurations are:

| Dataset | Run configuration |
|---|---|
| PEMS03 | [`config.json`](EXP_experiments/Scope/PEMS03/Scope_PEMS03_12to12_exp10101500_s1/config.json) |
| PEMS04 | [`config.json`](EXP_experiments/Scope/PEMS04/Scope_PEMS04_12to12_exppems04_joint3_lite_r256_strictalign_s1/config.json) |
| PEMS07 | [`config.json`](EXP_experiments/Scope/PEMS07/Scope_PEMS07_12to12_exppems07_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s1/config.json) |
| PEMS08 | [`config.json`](EXP_experiments/Scope/PEMS08/Scope_PEMS08_12to12_exppems08_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s1/config.json) |
| PEMS-BAY | [`config.json`](EXP_experiments/Scope/PEMSBAY/Scope_PEMSBAY_12to12_exppemsbay712_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s1/config.json) |

Raw datasets, prepared caches, and the final frozen asset bundle are not
provided by a Git clone. See [Data and assets](#data-and-assets) before
attempting a replay, and [Source entry points](#source-entry-points) for
stage-specific commands and their limitations.

## Paper pipeline

1. Train the retrieval-augmented Teacher.
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
The frozen five-dataset replay was verified locally with Python 3.11.9 and
PyTorch 2.5.1+cu121, using trusted local assets; this is not evidence that
the pinned `torch==2.0.0` environment installs or reproduces the same results.

```bash
pip install -r requirements.txt
python tools/preflight.py
python tools/test_preflight_cache.py
python contrib3_index_codec/test_adaptive_lossless_state_codec.py
```

Preflight checks the five frozen retrieval caches against their run configurations
and Train/Validation/Test index files without materializing retrieval tensors.
It does not train, evaluate Test, download, or unpickle files. A passing static
check is not a numerical reproduction certificate. Do not load pickle or
checkpoint files from untrusted sources.

## Data and assets

Raw data is not redistributed here. Place each dataset at
`datasets/PEMS03/data.npz` (similarly PEMS04, PEMS07, PEMS08, PEMSBAY).
Time/index/scaler arrays and retrieval caches also need preparation; a raw
file alone is insufficient. Preserve the paper's splits and masks.
PEMS-BAY uses a 7:1:2 split with 36,465 Train, 5,209 Validation, and 10,419
effective Test windows; the loader may display 10,432 Test rows after padding.

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

`tools/assets_manifest.json` describes the historical download inventory
for retained datasets; it is not the five-dataset frozen paper manifest.
`tools/final_assets_manifest.json` records the locally verified final-run
checkpoints/configuration assets separately. The old `v1.0.0` Release is not
part of the rebuilt repository, and no complete replacement asset download
has been published. See `tools/final_runs.json` for exact run identities.

`tools/release_assets_manifest.json` records the verified local Scope-named
candidate bundle and five retrieval-cache files; it does **not** imply those
files have been uploaded. The bundle members use their final destination names,
and the installer verifies their hashes. Once obtained from a trusted source,
put the five cache files under
`<staging>/datasets/<dataset>/<cache-filename>` and verify/install them with:

```bash
python tools/install_frozen_assets.py --bundle <local-final-assets.zip> --cache-root <staging>
# Review the preview first. The command below rejects conflicts and never overwrites:
python tools/install_frozen_assets.py --bundle <local-final-assets.zip> --cache-root <staging> --apply
```

The bundle and caches do not contain raw data or all legacy replay inputs.
Those still require the preparation route above. Until these files have a
published download location and the data-preparation recipe is independently
tested, a Git clone alone cannot reproduce the final numbers.

The historical downloader remains in source for provenance, but it has no
currently supported Release URL and is **not a complete paper reproduction
route**. Use a trusted original workspace for local inputs until a verified
asset bundle and clean-machine recipe are published.

## Source entry points

| Stage | Entry / evidence | Important qualification |
|---|---|---|
| Scope Teacher | `run.py`, `models/Scope.py`, original Teacher configs | Select the verified Teacher config, not `strict_neighbor_k4`, which is an exploratory Student. |
| Second moment | `compute_predictor_activation_covariance.py` | Uses Train activations; takes `--exp-dir` and `--output`. |
| Student | `run.py`; `run_lite_r256_valcal_pipeline.py` | Existing pipeline supports PEMS03/07/08, not a universal five-dataset runner. |
| SRM | `paper_results/icde_retrieval_free_pilot/` | Contains selection and state construction code, not disposable results. |
| Encoding | `contrib3_index_codec/` | Beam search and reversible codec; preserve frozen recipes. |
| Calibration | `contrib3_index_codec/export_frozen_calibrators.py` | Validation-only fitting; output directory is required. |
| Four-dataset replay | `contrib3_index_codec/eval_adaptive_lossless_test.py` | Requires frozen assets and raw/prepared data; writes a new output directory. |
| PEMS-BAY | `pemsbay_full/prepare_state.py`, `evaluate_frozen.py` | Separate exact-float state workflow; do not substitute integer Q8. The frozen evaluator disables cuDNN TF32 for Validation parity. |

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
python contrib3_index_codec/export_frozen_calibrators.py --output <new-validation-calibrator-directory>
python contrib3_index_codec/eval_adaptive_lossless_test.py --output <new-test-replay-directory>
# PEMS-BAY must freeze Validation outputs before the Test replay; use the same
# NEW output directory for both commands, not the source directory.
python pemsbay_full/evaluate_frozen.py --split val --output-dir <new-pemsbay-replay-directory>
python pemsbay_full/evaluate_frozen.py --split test --output-dir <same-pemsbay-replay-directory>
```

`srm_followup_validation/run_followup.py` is an attribution/timing
study, not the SRM selector. `benchmark_runtime_code_domain.py` is a legacy
runtime comparison, not the recommended final-system reproduction entry.

## Results, names, and provenance

Keep final metrics, ablations, calibration, frozen node sets and field recipes.
Do not delete `paper_results/` wholesale: scripts read inputs from it.
Figure-generation scripts and derived figure data are omitted from this
code-focused release; the experiment evidence needed by source scripts remains.
Historical `EXP_results/` CSVs are omitted; `run.py` creates new result files
when experiments are run.
The proposed method is **Scope**. The active model module is
`models/Scope.py`, and `tools/final_runs.json` points to the five frozen
`EXP_experiments/Scope/` runs. Historical `TLAST_improved` model identifiers
are resolved to `models/Scope.py` by `run.py`; no legacy model module is needed.
The original run paths and binary `config.pkl` identifiers remain available
for previously frozen references.
The current verified local asset bundle contains Scope-named final-run metadata;
its manifest records the updated configuration hashes. Historical identifiers
inside frozen binary configurations remain unchanged for traceability. The
public source paths were normalized without regenerating numerical results.
Local preparation backups in `_preparation/` are ignored by Git.

Scope's use of the TLAST codebase is acknowledged in
`README_UPSTREAM_TLAST.md` and `LICENSE`. Preserve the TLAST paper citation
and its upstream attribution when using or redistributing this code.
