# PEMS04 training workflow

The public training entry is intentionally limited to the frozen seed-1
PEMS04 Student configuration. It does not run hyperparameter search, select a
retrieval representation, evaluate Test during training, or change a paper
metric.

## What the workflow trains

The stages must run in this order:

1. `base`: train the dense predictor for 100 epochs from a seeded random
   initialization;
2. `lowrank`: load the best base checkpoint, freeze it, and train only the
   650-parameter low-rank retrieval adapter for 20 epochs;
3. `exact`: load the best low-rank checkpoint, freeze it, and train only the
   12-parameter exact-match adapter for 20 epochs;
4. `student`: initialize a rank-256 factorized predictor from the exact-stage
   checkpoint, then train only that predictor for at most 30 epochs with
   `kd_alpha=0.3` and Validation early stopping.

Every checkpoint is selected by Validation loss. The training entry contains
no Test evaluation code. The frozen Validation MAE `17.850632` reported by the
replay package is the retained Student result, not the epoch-100 base model.

## Required inputs

The three prepared PEMS04 files documented in `DATA.md` are required. Four
additional derived training assets are also required:

| Local path under `datasets/PEMS04/training_assets/` | Bytes | Purpose |
|---|---:|---|
| `analog_k4.npz` | 102,562,297 | train-only K=4 retrieval for the low-rank stage |
| `analog_k4_exact_mean.npz` | 105,784,863 | retrieval plus exact-match means |
| `predictor_activation_second_moment.npz` | 2,201,748 | train-only activation statistic for factorization |
| `teacher_predictions_joint3.npz` | 221,631,252 | aligned normalized Teacher predictions for KD |

Their SHA-256 values are pinned in `training/pems04.json` and checked before
use. They are ignored by Git and are not in the repository. Download the
single [PEMS04 training-assets release archive](https://github.com/Scope9909/Scope/releases/tag/pems04-repro-assets-v1)
and extract it from the repository root:

```bash
tar -xf scope-pems04-training-assets-v1.tar
```

The archive SHA-256 is
`2757bcdbc378b105c7b39b53dcfbb1b15aed3492a852aa3e058d9fae111fb221`.
The frozen paper Student checkpoint and config are already tracked in Git at
`assets/checkpoints/PEMS04/`; they are not duplicated in the release.

The Teacher prediction cache is a fixed prerequisite. This repository does
not yet expose the separate multi-stage Teacher-training and retrieval-cache
construction pipeline. Consequently, the workflow is a faithful training
path for the published Student given frozen training assets; it is not a
claim that the exact Student can currently be regenerated from raw PEMS04
alone.

Alternatively, from a trusted local workspace, preview and copy only these
four assets:

```bash
python -B tools/prepare_training_assets.py --source <trusted-workspace>
python -B tools/prepare_training_assets.py --source <trusted-workspace> --apply
```

The copy tool verifies the source hash before copying, refuses to overwrite a
mismatching destination, and verifies the copied file again.

## Environment and commands

Use the same Python/PyTorch environment as frozen replay. A CUDA device is
strongly recommended; `--device auto` selects `cuda:0` when CUDA is available.

```bash
python -B training/pems04.py --help
python -B training/pems04.py --stage all --preflight-only --device cuda:0
python -B training/pems04.py --stage all --device cuda:0
```

Outputs are written to `outputs/training/pems04/<stage>/`. Existing stage
directories are never overwritten. Each completed stage contains the
resolved JSON configuration, Validation history, and Validation-selected
`best.pth`.

For a no-write CUDA check, run one backward pass and one Validation batch for
an individual stage:

```bash
python -B training/pems04.py --stage base --smoke-test --device cuda:0
```

For a non-base stage run in isolation, pass the immediately preceding
checkpoint explicitly:

```bash
python -B training/pems04.py --stage student --device cuda:0 \
  --init-checkpoint <exact-stage-best.pth>
```

Training a new checkpoint is not the same operation as replaying the retained
paper checkpoint. Hardware, library, and nondeterministic kernel differences
can change optimization. Report the resulting Validation metrics honestly;
do not relabel a different run as the frozen `17.850632` result.
