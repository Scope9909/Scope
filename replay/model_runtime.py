"""Model construction and dataloader unpacking needed by frozen replay."""
from __future__ import annotations


def build_model(args, device):
    # Delay the model import so ``--help`` and static preflight stay lightweight.
    from models import Scope

    return Scope.Model(args).to(device)


def unpack_dataloader_input(loader_input):
    """Return traffic/time arrays and optional train-memory tensors."""
    empty_candidate = (None, None, None)
    if not isinstance(loader_input, (list, tuple)):
        raise TypeError("Dataloader model input must be a list or tuple")
    if len(loader_input) == 2:
        x, x_time = loader_input
        return (x, x_time, None, None, None) + empty_candidate
    if len(loader_input) == 3:
        x, x_time, retrieval = loader_input
        return (x, x_time, retrieval, None, None) + empty_candidate
    if len(loader_input) == 5:
        x, x_time, retrieval, exact, available = loader_input
        return (x, x_time, retrieval, exact, available) + empty_candidate
    if len(loader_input) == 8:
        return tuple(loader_input)
    raise ValueError(
        f"Expected 2, 3, 5, or 8 dataloader inputs, got {len(loader_input)}"
    )
