"""Small contracts for the scoped PEMS04 training workflow."""
from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.pems04 import (  # noqa: E402
    CosineScheduler,
    STAGES,
    masked_mae,
    stage_parent,
)


class TrainingWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(
            (ROOT / "training" / "pems04.json").read_text(encoding="utf-8")
        )

    def test_frozen_stage_contract(self):
        self.assertEqual(tuple(self.config["stages"]), STAGES)
        epochs = [self.config["stages"][name]["epochs"] for name in STAGES]
        self.assertEqual(epochs, [100, 20, 20, 30])
        self.assertEqual(self.config["stages"]["student"]["kd_alpha"], 0.3)
        self.assertEqual(
            self.config["stages"]["student"]["predictor_factor_rank"], 256
        )

    def test_active_paths_are_relative_and_hashes_are_sha256(self):
        for name, expected in self.config["training_assets"].items():
            self.assertFalse(Path(name).is_absolute())
            self.assertEqual(len(expected), 64)
            int(expected, 16)
        serialized = json.dumps(self.config)
        for drive in ("C", "D"):
            self.assertNotIn(f"{drive}:{os.sep}", serialized)

    def test_mask_and_cosine_warmup(self):
        prediction = torch.tensor([0.0, 3.0, 100.0])
        labels = torch.tensor([0.0, 1.0, 0.0])
        self.assertEqual(masked_mae(prediction, labels).item(), 2.0)
        parameter = torch.nn.Parameter(torch.ones(()))
        optimizer = torch.optim.AdamW([parameter], lr=1e-3)
        scheduler = CosineScheduler(optimizer, epochs=100, warmup=5, warmup_init=1e-6)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-6)
        scheduler.step(1)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.0002008)

    def test_custom_output_root_controls_parent_chain(self):
        custom = ROOT / "outputs" / "custom-training"
        parent = stage_parent(self.config, "exact", None, custom)
        self.assertEqual(parent, custom / "lowrank" / "best.pth")


if __name__ == "__main__":
    unittest.main()
