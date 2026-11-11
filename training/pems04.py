"""Train the four frozen PEMS04 stages used by the Scope Student.

This entry never evaluates Test.  Checkpoints are selected only by Validation
loss.  The fixed teacher-prediction and retrieval caches are explicit inputs;
they are not silently regenerated or downloaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from replay.model_runtime import build_model, unpack_dataloader_input  # noqa: E402

STAGES = ("base", "lowrank", "exact", "student")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "training" / "pems04.json"
    )
    parser.add_argument(
        "--stage", choices=STAGES + ("all",), default="all",
        help="Run one stage or the complete base-to-Student sequence.",
    )
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "outputs" / "training" / "pems04"
    )
    parser.add_argument(
        "--device", default="auto",
        help="auto, cpu, cuda, or a concrete CUDA device such as cuda:0.",
    )
    parser.add_argument(
        "--init-checkpoint", type=Path,
        help="Override the parent checkpoint for a single non-base stage.",
    )
    parser.add_argument(
        "--preflight-only", action="store_true",
        help="Verify paths, hashes, configuration, CUDA, and model construction.",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run one training batch and one Validation batch without writing files.",
    )
    return parser.parse_args()


def read_config(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported training configuration schema")
    if tuple(data["stages"]) != STAGES:
        raise ValueError(f"Expected stage order {STAGES}, got {tuple(data['stages'])}")
    return data


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"Active training paths must be repository-relative: {path}")
    resolved = (ROOT / path).resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError(f"Training path escapes the repository: {path}") from exc
    return resolved


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is false. "
            "Install a CUDA-enabled PyTorch build or use --device cpu."
        )
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device


def stage_namespace(config: dict, stage: str, device: torch.device) -> SimpleNamespace:
    values = {}
    values.update(config["dataset"])
    values.update(config["model"])
    values.update(config["training"])
    values.update(config["stages"][stage])
    values.update({
        "device": str(device),
        "cache_data_path": "datasets",
        "analog_retrieval_cache": values.get("analog_retrieval_cache", "none"),
        "teacher_predictions_path": values.get("teacher_predictions_path", "none"),
        "teacher_predictions_space": values.get("teacher_predictions_space", "normalized"),
        "teacher_predictions_strict_alignment": values.get(
            "teacher_predictions_strict_alignment", 0
        ),
        "teacher_predictions_required_splits": values.get(
            "teacher_predictions_required_splits", "train,val,test"
        ),
        "kd_alpha": values.get("kd_alpha", 0.0),
        "analog_weekly_retrieval_cache": "none",
        "analog_metric_retrieval_cache": "none",
        "analog_candidate_cache": "none",
    })
    return SimpleNamespace(**values)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def initialize_model(model: torch.nn.Module) -> None:
    preserved = set()
    for module_name, module in model.named_modules():
        if getattr(module, "preserve_initialization", False):
            for parameter_name, _ in module.named_parameters(recurse=True):
                preserved.add(".".join(filter(None, (module_name, parameter_name))))
    for name, parameter in model.named_parameters():
        if parameter.dim() > 1 and name not in preserved:
            torch.nn.init.xavier_uniform_(parameter)


def load_state(path: Path, device: torch.device) -> dict:
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # PyTorch versions before weights_only was added.
        state = torch.load(path, map_location=device)
    if not isinstance(state, dict) or not state:
        raise ValueError(f"Checkpoint is not a non-empty state dictionary: {path}")
    return state


def initialize_stage(
    model: torch.nn.Module,
    args: SimpleNamespace,
    stage: str,
    parent: Path | None,
    device: torch.device,
) -> list[torch.nn.Module]:
    if stage == "base":
        return [model]
    if parent is None or not parent.is_file():
        raise FileNotFoundError(f"Parent checkpoint is required for {stage}: {parent}")
    parent_state = load_state(parent, device)
    train_names = tuple(args.train_modules)

    if stage in ("lowrank", "exact"):
        missing, unexpected = model.load_state_dict(parent_state, strict=False)
        expected_missing = set()
        for name in train_names:
            module = getattr(model, name)
            prefix = f"{name}."
            expected_missing.update(
                prefix + key for key in module.state_dict() if prefix + key not in parent_state
            )
        if set(missing) != expected_missing or unexpected:
            raise RuntimeError(
                f"{stage} parent mismatch: expected missing={sorted(expected_missing)}, "
                f"missing={list(missing)}, unexpected={list(unexpected)}"
            )
    elif stage == "student":
        predictor = model.time_wise_predictor
        shared = {
            key: value for key, value in parent_state.items()
            if not key.startswith("time_wise_predictor.")
        }
        missing, unexpected = model.load_state_dict(shared, strict=False)
        expected_missing = {
            "time_wise_predictor." + key for key in predictor.state_dict()
        }
        if set(missing) != expected_missing or unexpected:
            raise RuntimeError(
                "Student parent mismatch: expected predictor-only missing keys; "
                f"missing={list(missing)}, unexpected={list(unexpected)}"
            )
        covariance_path = repo_path(args.predictor_activation_covariance)
        with np.load(covariance_path, allow_pickle=False) as archive:
            if "second_moment" not in archive:
                raise KeyError(f"{covariance_path} lacks second_moment")
            second_moment = np.asarray(archive["second_moment"], dtype=np.float64)
        retained = predictor.initialize_from_richer_state_dict(
            parent_state,
            prefix="time_wise_predictor.",
            activation_second_moment=second_moment,
            activation_ridge=float(args.predictor_activation_ridge),
        )
        print(f"Student rank-{args.predictor_factor_rank} retained energy: {retained:.8f}")
    else:  # pragma: no cover - guarded by argparse/config validation.
        raise ValueError(stage)

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    modules = [getattr(model, name) for name in train_names]
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    return modules


class CosineScheduler:
    """Epoch scheduler matching the historical warmup/cosine semantics."""

    def __init__(self, optimizer, epochs: int, warmup: int, warmup_init: float):
        self.optimizer = optimizer
        self.epochs = int(epochs)
        self.warmup = int(warmup)
        self.warmup_init = float(warmup_init)
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.steps = [(lr - self.warmup_init) / self.warmup for lr in self.base_lrs]
        self._set([self.warmup_init] * len(self.base_lrs))

    def _set(self, values):
        for group, value in zip(self.optimizer.param_groups, values):
            group["lr"] = value

    def step(self, epoch: int) -> None:
        if epoch < self.warmup:
            values = [self.warmup_init + epoch * step for step in self.steps]
        else:
            cycle = epoch // self.epochs
            current = epoch - self.epochs * cycle
            peak = [lr * (0.1 ** cycle) for lr in self.base_lrs]
            values = [
                0.5 * value * (1.0 + math.cos(math.pi * current / self.epochs))
                for value in peak
            ]
        self._set(values)


def valid_mask(labels: torch.Tensor) -> torch.Tensor:
    cleaned = torch.where(labels.abs() < 1e-4, torch.zeros_like(labels), labels)
    return torch.isfinite(cleaned) & cleaned.ne(0.0)


def masked_mae(prediction: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    mask = valid_mask(labels)
    error = torch.where(mask, (prediction - labels).abs(), torch.zeros_like(prediction))
    return error.sum() / mask.sum().clamp_min(1).to(error.dtype)


def make_model_input(loader_input, loader_target, device: torch.device, training: bool):
    (
        x, x_time, retrieval, exact, available,
        candidate_future, candidate_available, candidate_features,
    ) = unpack_dataloader_input(loader_input)
    y, y_time = loader_target
    model_input = [
        torch.as_tensor(x, dtype=torch.float32, device=device),
        torch.as_tensor(x_time, dtype=torch.long, device=device),
        torch.as_tensor(y_time, dtype=torch.long, device=device),
    ]
    if retrieval is not None:
        dtype = torch.float16 if training and retrieval.dtype == np.float16 else torch.float32
        model_input.append(torch.as_tensor(retrieval, dtype=dtype, device=device))
    if exact is not None:
        dtype = torch.float16 if training and exact.dtype == np.float16 else torch.float32
        model_input.extend([
            torch.as_tensor(exact, dtype=dtype, device=device),
            torch.as_tensor(available, dtype=dtype, device=device),
        ])
    if candidate_future is not None:
        model_input.extend([
            torch.as_tensor(candidate_future, dtype=torch.float32, device=device),
            torch.as_tensor(candidate_available, dtype=torch.float32, device=device),
            torch.as_tensor(candidate_features, dtype=torch.float32, device=device),
        ])
    return model_input, torch.as_tensor(y, dtype=torch.float32, device=device)


def set_train_mode(model, stage: str, train_modules: list[torch.nn.Module]) -> None:
    if stage == "base":
        model.train()
    else:
        model.eval()
        for module in train_modules:
            module.train()


def train_batch(
    model, scaler, model_input, truth, teacher, args, optimizer, stage,
    train_modules,
):
    set_train_mode(model, stage, train_modules)
    optimizer.zero_grad()
    normalized, _ = model(model_input)
    prediction = scaler.inverse_transform(normalized)
    loss = masked_mae(prediction, truth)
    kd_alpha = float(args.kd_alpha)
    if kd_alpha:
        if teacher is None:
            raise RuntimeError("Student KD requires an aligned teacher batch")
        soft = torch.as_tensor(teacher, dtype=torch.float32, device=prediction.device)
        if args.teacher_predictions_space == "normalized":
            soft = scaler.inverse_transform(soft)
        if soft.shape != prediction.shape:
            raise ValueError(f"Teacher shape {soft.shape} != prediction shape {prediction.shape}")
        loss = (1.0 - kd_alpha) * loss + kd_alpha * masked_mae(prediction, soft)
    ortho = getattr(model, "ortho_loss", None)
    if isinstance(ortho, torch.Tensor):
        loss = loss + float(args.lambda_ortho) * ortho
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.gradient_clip))
    optimizer.step()
    return float(loss.detach())


@torch.no_grad()
def validate(model, scaler, loader, device: torch.device) -> dict:
    model.eval()
    horizons = None
    for loader_input, loader_target in loader.get_iterator():
        model_input, truth = make_model_input(loader_input, loader_target, device, False)
        normalized, _ = model(model_input)
        prediction = scaler.inverse_transform(normalized)
        if horizons is None:
            count = prediction.shape[1]
            horizons = {
                key: np.zeros(count, dtype=np.float64)
                for key in ("count", "absolute", "squared", "percentage")
            }
        for index in range(prediction.shape[1]):
            label_h = truth[:, index]
            pred_h = prediction[:, index]
            mask = valid_mask(label_h)
            error = torch.where(mask, pred_h - label_h, torch.zeros_like(pred_h))
            horizons["count"][index] += float(mask.sum())
            horizons["absolute"][index] += float(error.abs().sum())
            horizons["squared"][index] += float(error.square().sum())
            percentage = torch.where(
                mask, error.abs() / label_h.abs().clamp_min(1e-12), torch.zeros_like(error)
            )
            horizons["percentage"][index] += float(percentage.sum())
    if horizons is None:
        raise RuntimeError("Validation loader yielded no batches")
    count = np.maximum(horizons["count"], 1.0)
    mae = horizons["absolute"] / count
    rmse = np.sqrt(horizons["squared"] / count)
    mape = horizons["percentage"] / count
    return {
        "loss": float(mae.mean()),
        "mae": float(mae.mean()),
        "rmse": float(rmse.mean()),
        "mape": float(mape.mean()),
        "horizon_mae": mae.tolist(),
    }


def stage_parent(
    config: dict,
    stage: str,
    override: Path | None,
    output_root: Path,
) -> Path | None:
    if stage == "base":
        if override is not None:
            raise ValueError("--init-checkpoint is not valid for the base stage")
        return None
    if override is not None:
        return override.resolve()
    configured = repo_path(config["stages"][stage]["parent_checkpoint"])
    previous = STAGES[STAGES.index(stage) - 1]
    expected_default = (
        ROOT / "outputs" / "training" / "pems04" / previous / "best.pth"
    )
    if configured != expected_default:
        raise ValueError(
            f"Configured parent chain mismatch for {stage}: "
            f"{configured} != {expected_default}"
        )
    return output_root / previous / "best.pth"


def required_assets(config: dict, stage: str) -> list[Path]:
    keys = []
    if stage == "lowrank":
        keys = ["datasets/PEMS04/training_assets/analog_k4.npz"]
    elif stage in ("exact", "student"):
        keys = ["datasets/PEMS04/training_assets/analog_k4_exact_mean.npz"]
    if stage == "student":
        keys += [
            "datasets/PEMS04/training_assets/predictor_activation_second_moment.npz",
            "datasets/PEMS04/training_assets/teacher_predictions_joint3.npz",
        ]
    return [repo_path(key) for key in keys]


def preflight(config: dict, stage: str, parent: Path | None, require_parent: bool) -> None:
    prepared = [
        repo_path("datasets/PEMS04/data.npz"),
        repo_path("datasets/PEMS04/PEMS04_time.npz"),
        repo_path("datasets/PEMS04/PEMS04_12to12_index_scaler.npz"),
    ]
    for path in prepared:
        if not path.is_file():
            raise FileNotFoundError(path)
    expected = config["training_assets"]
    for path in required_assets(config, stage):
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        key = path.relative_to(ROOT).as_posix()
        if actual != expected[key]:
            raise ValueError(f"Training asset hash mismatch: {key}: {actual}")
    if require_parent and parent is not None and not parent.is_file():
        raise FileNotFoundError(parent)


def run_stage(
    config: dict,
    stage: str,
    output_root: Path,
    device: torch.device,
    parent_override: Path | None,
    preflight_only: bool,
    smoke_test: bool,
) -> None:
    from utils import datasets_utils

    args = stage_namespace(config, stage, device)
    parent = stage_parent(config, stage, parent_override, output_root)
    preflight(config, stage, parent, require_parent=not preflight_only or parent_override is not None)
    set_seed(int(args.seed))
    logger = logging.getLogger(f"scope_training_{stage}")
    logger.handlers.clear()
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)

    loaders, nodes, scaler, _ = datasets_utils.load_dataset(args, logger)
    args.num_nodes = nodes
    model = build_model(args, device)
    initialize_model(model)
    if preflight_only and stage != "base" and parent is not None and not parent.is_file():
        print(f"{stage}: generated parent will be {parent}")
        print(f"{stage}: asset and model-construction preflight passed")
        return
    train_modules = initialize_stage(model, args, stage, parent, device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=float(args.learning_rate), weight_decay=float(args.weight_decay)
    )
    scheduler = CosineScheduler(
        optimizer, int(args.epochs), int(args.lr_warmup_epochs), float(args.lr_warmup_init)
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    active = sum(parameter.numel() for parameter in trainable)
    print(f"{stage}: device={device}, total_parameters={total}, trainable_parameters={active}")

    if preflight_only:
        print(f"{stage}: asset, checkpoint, loader, and model preflight passed")
        return

    if smoke_test:
        loaders["train_loader"].shuffle()
        loader_input, loader_target = next(loaders["train_loader"].get_iterator())
        model_input, truth = make_model_input(loader_input, loader_target, device, True)
        loss = train_batch(
            model, scaler, model_input, truth,
            loaders["train_loader"].current_teacher_batch,
            args, optimizer, stage, train_modules,
        )
        original_batches = loaders["val_loader"].num_batch
        loaders["val_loader"].num_batch = 1
        try:
            metrics = validate(model, scaler, loaders["val_loader"], device)
        finally:
            loaders["val_loader"].num_batch = original_batches
        print(json.dumps({"stage": stage, "train_loss": loss, "validation_batch": metrics}, indent=2))
        print("Smoke test wrote no checkpoint or result file.")
        return

    output = output_root / stage
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite training output: {output}")
    output.mkdir(parents=True)
    (output / "config.json").write_text(
        json.dumps(vars(args), indent=2), encoding="utf-8"
    )

    history = []
    best_loss = math.inf
    stale = 0
    if stage != "base":
        anchor = validate(model, scaler, loaders["val_loader"], device)
        best_loss = anchor["loss"]
        torch.save(model.state_dict(), output / "best.pth")
        history.append({"epoch": 0, "validation": anchor, "anchor": True})
        print(f"{stage} epoch 000 Validation MAE={anchor['mae']:.6f}")

    started = time.time()
    for epoch in range(1, int(args.epochs) + 1):
        loaders["train_loader"].shuffle()
        losses = []
        for batch_index, (loader_input, loader_target) in enumerate(
            loaders["train_loader"].get_iterator()
        ):
            model_input, truth = make_model_input(loader_input, loader_target, device, True)
            losses.append(train_batch(
                model, scaler, model_input, truth,
                loaders["train_loader"].current_teacher_batch,
                args, optimizer, stage, train_modules,
            ))
            if batch_index % int(args.print_every) == 0:
                print(f"{stage} epoch={epoch:03d} batch={batch_index:03d} loss={losses[-1]:.6f}")
        validation = validate(model, scaler, loaders["val_loader"], device)
        scheduler.step(epoch)
        row = {
            "epoch": epoch,
            "train_loss_batch_mean": float(np.mean(losses)),
            "validation": validation,
            "learning_rate_next_epoch": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"{stage} epoch={epoch:03d} Validation "
            f"MAE={validation['mae']:.6f} RMSE={validation['rmse']:.6f} "
            f"MAPE={validation['mape']:.6f}"
        )
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            stale = 0
            torch.save(model.state_dict(), output / "best.pth")
        elif epoch >= int(args.early_stop_start_epoch):
            stale += 1
        if bool(args.early_stop) and stale >= int(args.early_stop_step):
            print(f"{stage}: early stop after {epoch} epochs")
            break

    report = {
        "stage": stage,
        "selection_split": "Validation",
        "test_used": False,
        "best_validation_loss": best_loss,
        "elapsed_seconds": time.time() - started,
        "history": history,
    }
    (output / "history.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{stage}: best checkpoint written to {output / 'best.pth'}")


def main() -> None:
    cli = parse_args()
    if cli.preflight_only and cli.smoke_test:
        raise ValueError("--preflight-only and --smoke-test are mutually exclusive")
    if cli.stage == "all" and cli.init_checkpoint is not None:
        raise ValueError("--init-checkpoint is only valid with one explicit stage")
    os.chdir(ROOT)
    config = read_config(cli.config.resolve())
    device = select_device(cli.device)
    if device.type == "cuda":
        print(
            f"CUDA available: torch={torch.__version__}, runtime={torch.version.cuda}, "
            f"device={torch.cuda.get_device_name(device)}"
        )
    else:
        print(f"Using CPU with torch={torch.__version__}")
    stages = STAGES if cli.stage == "all" else (cli.stage,)
    output_root = cli.output_root.resolve()
    for stage in stages:
        run_stage(
            config, stage, output_root, device,
            cli.init_checkpoint,
            cli.preflight_only,
            cli.smoke_test,
        )


if __name__ == "__main__":
    main()
