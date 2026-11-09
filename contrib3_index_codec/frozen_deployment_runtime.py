"""Load frozen retrieval state and fitted calibration tables for inference."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from adaptive_lossless_state_codec import load_runtime_state
from calibrate_dualscale_smooth_tod import (
    smooth_correction_for, target_time_correction_for,
)
from utils.online_residual_memory import OnlineResidualMemory


@dataclass
class FrozenAssets:
    memory: OnlineResidualMemory
    centers: np.ndarray
    smooth_residuals: np.ndarray
    target_time_residuals: np.ndarray
    smooth_shrinkage: float
    time_shrinkage: float
    slices_per_day: int

    @classmethod
    def load(cls, state_path: str | Path, calibrator_path: str | Path):
        memory = OnlineResidualMemory(load_runtime_state(state_path))
        with np.load(calibrator_path, allow_pickle=False) as archive:
            centers = np.asarray(archive["centers"], dtype=np.float64)
            residuals = np.asarray(archive["smooth_residuals"], dtype=np.float64)
            time_table = np.asarray(archive["target_time_residuals"], dtype=np.float64)
            smooth_shrinkage = float(archive["smooth_shrinkage"])
            time_shrinkage = float(archive["time_shrinkage"])
            slices_per_day = int(archive["slices_per_day"])
            num_time_bins = int(archive["num_time_bins"])
        if centers.shape != residuals.shape or centers.ndim != 3:
            raise ValueError("Invalid smooth-level calibration tables")
        if (num_time_bins < 1 or slices_per_day % num_time_bins or
                time_table.shape != (num_time_bins, centers.shape[2])):
            raise ValueError("Invalid target-time calibration table")
        return cls(memory, centers, residuals, time_table,
                   smooth_shrinkage, time_shrinkage, slices_per_day)

    def calibrate(self, prediction: np.ndarray, endpoints: np.ndarray) -> np.ndarray:
        if prediction.ndim != 3 or len(prediction) != len(endpoints):
            raise ValueError("Prediction/endpoints are not aligned")
        smooth = prediction + self.smooth_shrinkage * smooth_correction_for(
            prediction, self.centers, self.smooth_residuals)
        return smooth + self.time_shrinkage * target_time_correction_for(
            np.asarray(endpoints), prediction.shape[1],
            self.target_time_residuals, self.slices_per_day)
