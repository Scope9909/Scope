# Scope reproducibility package

This repository contains the frozen Validation/Test replay for PEMS03,
PEMS04, PEMS07, PEMS08, and PEMS-BAY, plus a focused PEMS04 Student-training
workflow. It does not contain hyperparameter search, state/encoding selection,
multi-seed studies, ablation archives, or historical experiment outputs.

## Repository layout

```text
assets/checkpoints/   frozen checkpoint and its exact configuration
assets/states/        frozen Validation/Test retrieval states
configs/              active replay paths, calibration, and SHA-256 inventory
datasets/             prepared local inputs (not tracked in Git)
layers/, models/      model definition shared by training and inference
replay/               frozen Validation/Test replay entries
training/             scoped PEMS04 four-stage training entry and configuration
tests/                small structural and numerical-unit tests
tools/                preflight and local-input preparation utilities
utils/                dataset loader and frozen-state query runtime
```

Generated replay and training output does not belong in the repository.

## What can be reproduced

With the prepared dataset files and hash-verified frozen assets, the replay
package reproduces:

- Validation inference from each frozen checkpoint and Validation state;
- Test inference from each frozen checkpoint and Test state;
- calibration fitted on Validation and applied to Test;
- official sample counts and raw/calibrated MAE, RMSE, and MAPE.

The replay does not retrain or select anything and does not change a paper
metric. It is a reproducibility check of configurations frozen before Test,
not a new independent blind Test experiment.

The PEMS04 training entry separately exposes the actual four-stage path used
for the retained Student: 100-epoch dense base, 20-epoch low-rank adapter,
20-epoch exact-match adapter, and at most 30 epochs of rank-256 Student KD.
It selects checkpoints using Validation only and contains no Test evaluation.
The workflow requires four hash-pinned derived training assets supplied in the
[PEMS04 training-assets release](https://github.com/Scope9909/Scope/releases/tag/pems04-repro-assets-v1),
not in Git. In particular, the Teacher prediction cache is a fixed prerequisite; the
separate Teacher-training pipeline is not claimed to be public. See
[`TRAINING.md`](TRAINING.md) for the exact boundary.

## Environment and checks

Python 3.11 is recommended. The completed numerical verification used Python
3.11.0, PyTorch 2.0.0+cu117 with CUDA, NumPy 1.x, and einops 0.8.1.

```bash
pip install -r requirements.txt
python -B replay/four_datasets.py --help
python -B replay/pemsbay.py --help
python -B training/pems04.py --help
python -B tools/preflight.py
python -B tests/test_preflight.py
python -B tests/test_state_codec.py
python -B tests/test_calibration.py
python -B tests/test_factorized_lite_predictor.py
```

`tools/preflight.py` verifies the frozen replay assets and required prepared
files. The training entry performs its own stage-specific hash and dependency
checks. Neither preflight is a numerical reproduction by itself.

## Required data and assets

Public acquisition links, expected shapes, provenance, and the exact directory
layout are documented in [`DATA.md`](DATA.md). The prepared arrays are present
only in the author's local working copy and are ignored by Git; cloning this
repository does not download them.

For every dataset, provide these prepared files under `datasets/<DATASET>/`:

- `data.npz`;
- `<DATASET>_time.npz`;
- `<DATASET>_12to12_index_scaler.npz` (the public name PEMS-BAY maps to local
  ID `PEMSBAY` and uses the `_712` suffix).

Raw datasets are not redistributed. The index/scaler files must preserve the
paper splits. PEMS-BAY has 5,209 Validation and 10,419 effective Test windows;
the entry removes loader padding using the official indices.

The checkpoints, binary configurations, selected states, calibration JSON,
and PEMS-BAY selection record are immutable replay inputs. Their required
hashes are in `configs/replay_assets_manifest.json`. Do not load pickle or
checkpoint files from an untrusted source.

To copy only the 15 prepared replay files from a trusted workspace, preview
first:

```bash
python -B tools/prepare_local_inputs.py --source <trusted-workspace>
python -B tools/prepare_local_inputs.py --source <trusted-workspace> --apply
python -B tools/preflight.py
```

## Frozen replay commands

Use new output directories and complete Validation before Test.

```bash
python -B replay/four_datasets.py --split val  --output-dir outputs/four
python -B replay/pemsbay.py       --split val  --output-dir outputs/pemsbay

python -B replay/four_datasets.py --split test --output-dir outputs/four
python -B replay/pemsbay.py       --split test --output-dir outputs/pemsbay
```

The output directories contain generated predictions and reports. They may be
deleted after verification and are not frozen inputs or release assets.

## PEMS04 training commands

After supplying the four derived assets listed in `TRAINING.md`:

```bash
python -B training/pems04.py --stage all --preflight-only --device cuda:0
python -B training/pems04.py --stage all --device cuda:0
```

This does not overwrite the retained checkpoint. New checkpoints go under
`outputs/training/pems04/` and must be reported as new runs.

## Verification status

The 2026-11-10 CUDA replay completed Validation and Test for all five datasets.
All sample counts matched. Test metrics agreed with the retained reference to
a maximum absolute difference of `1.18e-4`; this is numerical, not bitwise,
agreement. Exact values and later environment limitations are recorded in
[`PROJECT_STATUS.md`](PROJECT_STATUS.md).

The PEMS04 training entry passed help, asset/hash preflight, strict checkpoint
initialization, and one-batch CUDA forward/backward smoke tests for all four
stages. A full 100 + 20 + 20 + 30 epoch run then completed. Its best new
Student checkpoint was selected at epoch 26 (Validation MAE 17.785667, RMSE
29.949529, MAPE 0.118980). This is a separate run, not the retained paper
checkpoint or a replacement for its frozen replay metric.

## Provenance and license

See `LICENSE` for the repository license, upstream research attribution, and
the licensing limitation of the inherited implementation. That notice is
preserved as required; active code, paths, configurations, and commands use
the Scope name.
