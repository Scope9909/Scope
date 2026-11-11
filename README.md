# Scope frozen replay package

This repository is the public replay package for the frozen PEMS03, PEMS04,
PEMS07, PEMS08, and PEMS-BAY results. It contains no training workflow,
hyperparameter search, state/encoding selection, multi-seed study, ablation
archive, or historical experiment output.

## Repository layout

```text
assets/checkpoints/   frozen checkpoint and its exact configuration
assets/states/        frozen Validation/Test retrieval states
configs/              active paths, calibration, and SHA-256 inventory
datasets/             prepared local inputs (not redistributed)
DATA.md               public data sources and exact local file contract
layers/, models/      model definition required for inference
replay/               the two supported replay entries and their runtime
tests/                small structural and numerical-unit tests
tools/                preflight and local-input preparation utilities
utils/                dataset loader and frozen-state query runtime
```

This combines a small research repository's clear top-level layout with an
explicit data and command contract. Generated replay output does not belong in
the repository.

## What can be reproduced

With the prepared dataset files and hash-verified frozen assets, the package
replays:

- Validation inference from each frozen checkpoint and Validation state;
- Test inference from each frozen checkpoint and Test state;
- calibration fitted on Validation and applied to Test;
- official sample counts and raw/calibrated MAE, RMSE, and MAPE.

The replay does not retrain or select anything and does not change a paper
metric. It is a reproducibility check of configurations frozen before Test,
not a new independent blind Test experiment.

## Environment and checks

Python 3.11 is recommended. The completed numerical verification used Python
3.11.0, PyTorch 2.0.0+cu117 with CUDA, NumPy 1.x, and einops 0.8.1.

```bash
pip install -r requirements.txt
python -B replay/four_datasets.py --help
python -B replay/pemsbay.py --help
python -B tools/preflight.py
python -B tests/test_preflight.py
python -B tests/test_state_codec.py
python -B tests/test_calibration.py
python -B tests/test_factorized_lite_predictor.py
```

`tools/preflight.py` verifies the active asset hashes and required prepared
files. It is not a numerical reproduction check.

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
and PEMS-BAY selection record are immutable inputs. Their required hashes are
in `configs/replay_assets_manifest.json`. Do not load pickle or checkpoint
files from an untrusted source.

To copy only the 15 prepared files from a trusted workspace, preview first:

```bash
python -B tools/prepare_local_inputs.py --source <trusted-workspace>
python -B tools/prepare_local_inputs.py --source <trusted-workspace> --apply
python -B tools/preflight.py
```

## Replay commands

Use new output directories and complete Validation before Test.

```bash
python -B replay/four_datasets.py --split val  --output-dir outputs/four
python -B replay/pemsbay.py       --split val  --output-dir outputs/pemsbay

python -B replay/four_datasets.py --split test --output-dir outputs/four
python -B replay/pemsbay.py       --split test --output-dir outputs/pemsbay
```

The output directories contain generated predictions and reports. They may be
deleted after verification and are not frozen inputs or release assets.

## Verification status

The 2026-11-10 CUDA replay completed Validation and Test for all five datasets.
All sample counts matched. Test metrics agreed with the retained reference to
a maximum absolute difference of `1.18e-4`; this is numerical, not bitwise,
agreement. Exact values and later environment limitations are recorded in
`PROJECT_STATUS.md`.

## Provenance and license

See `LICENSE` for the repository license, upstream research attribution, and
the licensing limitation of the inherited implementation. That notice is
preserved as required; active code, paths, configurations, and commands use
the Scope name.
