"""Memory-bounded batch-1 replay of Student, gate, and conditional Teacher.

The 512 requests are the first consecutive official PEMS07 Test endpoints.
No Test label is loaded or used. Retrieval/candidate payloads are the frozen
precomputed Train-memory caches, built on demand per request. The gate and
calibration tables are frozen using Validation before this replay.
"""

import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from calibrate_dualscale_smooth_tod import (
    fit_and_apply, smooth_correction_for, target_time_correction_for,
)
from run_pems07_teacher_escalation import UtilityGate, build_features
from screen_matched_valcal_oracle import CALIBRATION_KEYS, SOURCES


def prepared_cache(path, positions, quality, std):
    with np.load(path, allow_pickle=False) as archive:
        retrieval = np.asarray(archive["retrieval_test"], dtype=np.float32)[positions]
        exact = np.asarray(archive["exact_retrieval_test"], dtype=np.float32)[positions]
        available = np.asarray(archive["exact_available_test"], dtype=np.uint8)[positions]
        if quality:
            distance = np.asarray(archive["history_distance_test"], dtype=np.float32)[positions]
            dispersion = np.asarray(archive["future_dispersion_test"], dtype=np.float32)[positions]
            distance = np.repeat((distance/(std**2))[:, None, :, None], 12, axis=1)
            dispersion = dispersion/std
            retrieval = np.concatenate((retrieval, distance, dispersion), axis=-1)
    return retrieval, exact, available


def load_engine(exp_dir, scaler, num_nodes, device):
    with (exp_dir / "config.pkl").open("rb") as stream:
        args = pickle.load(stream)
    args.device = str(device)
    args.batch_size = 1
    args.cache_data_path = "datasets"
    args.pearson_adj = None
    args.num_nodes = num_nodes
    if isinstance(getattr(args, "k_hops", "1,2"), str):
        args.k_hops = [int(x) for x in args.k_hops.split(",")]
    from run import trainer
    engine = trainer(scaler=scaler, device=device, args=args, inverse=True)
    engine.model.load_state_dict(torch.load(exp_dir/"best.pth", map_location=device), strict=True)
    engine.model.eval()
    return engine, args


def calibrator(fit_pred, fit_truth, fit_endpoints, params):
    _, _, tables = fit_and_apply(
        fit_pred, fit_truth, fit_endpoints, fit_pred[:1], fit_endpoints[:1], **params,
    )
    return tables


def apply_cal(raw_prediction, endpoint, tables, params):
    centers, residuals, time_table = tables
    pred = np.asarray(raw_prediction, dtype=np.float64)
    pred = pred + params["smooth_shrinkage"]*smooth_correction_for(pred, centers, residuals)
    pred = pred + params["time_shrinkage"]*target_time_correction_for(
        np.asarray([endpoint], dtype=np.int64), pred.shape[1], time_table,
        params["slices_per_day"],
    )
    return pred


def sample_input(loader, row, endpoint, device, teacher):
    x = loader.data[endpoint+loader.x_offsets, :, :][None].astype(np.float32)
    x = loader.scaler.transform(x)
    x_time = loader.time_data[endpoint+loader.x_offsets, :][None]
    y_time = loader.time_data[endpoint+loader.y_offsets, :][None]
    payload = [
        torch.as_tensor(x, dtype=torch.float32, device=device),
        torch.as_tensor(x_time, dtype=torch.long, device=device),
        torch.as_tensor(y_time, dtype=torch.long, device=device),
        torch.as_tensor(loader.retrieval_data[row:row+1], dtype=torch.float32, device=device),
        torch.as_tensor(loader.exact_retrieval_data[row:row+1], dtype=torch.float32, device=device),
        torch.as_tensor(loader.exact_available_data[row:row+1], dtype=torch.float32, device=device),
    ]
    if teacher:
        candidate, availability, features = loader._build_analog_candidate_batch(
            np.asarray([endpoint], dtype=np.int64), loader.analog_candidate_index_data[row:row+1],
        )
        payload.extend((
            torch.as_tensor(candidate, dtype=torch.float32, device=device),
            torch.as_tensor(availability, dtype=torch.float32, device=device),
            torch.as_tensor(features, dtype=torch.float32, device=device),
        ))
    return payload


def call_model(engine, loader, row, endpoint, device, teacher, tables, params):
    payload = sample_input(loader, row, endpoint, device, teacher)
    with torch.inference_mode():
        output = engine._forward(payload)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    prediction = output.detach().cpu().numpy()[..., 0]
    return prediction, apply_cal(prediction, endpoint, tables, params)


def main():
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    paths = SOURCES["PEMS07"]
    gate_state = torch.load(HERE/"pems07_calibrated_utility_gate_frozen.pt", map_location="cpu")
    params_json = json.loads((ROOT/paths["calibration"]).read_text(encoding="utf-8"))
    params = {key: params_json[key] for key in CALIBRATION_KEYS}
    with np.load(ROOT/"datasets/PEMS07/PEMS07_12to12_index_scaler.npz", allow_pickle=False) as z:
        test_indices = np.asarray(z["idx_test"], dtype=np.int64)
        val_indices = np.asarray(z["idx_val"], dtype=np.int64)
        mean, std = float(z["scaler_mean"]), float(z["scaler_std"])
    positions = np.arange(512, dtype=np.int64)
    endpoints = test_indices[positions]
    with np.load(ROOT/"datasets/PEMS07/data.npz", allow_pickle=False) as z:
        raw = np.asarray(z["data"][..., :1], dtype=np.float32)
    with np.load(ROOT/"datasets/PEMS07/PEMS07_time.npz", allow_pickle=False) as z:
        time_data = np.asarray(z["time"])
    from utils.datasets_utils import DataLoader_time, StandardScaler
    scaler = StandardScaler(mean=np.asarray(mean), std=np.asarray(std))
    student_cache = ROOT/"datasets/PEMS07/PEMS07_12to12_analog_k4_exact_mean.npz"
    teacher_cache = ROOT/"datasets/PEMS07/PEMS07_12to12_analog_k8_exact_mean.npz"
    sr, sx, sa = prepared_cache(student_cache, positions, False, std)
    tr, tx, ta = prepared_cache(teacher_cache, positions, True, std)
    with np.load(ROOT/"datasets/PEMS07/PEMS07_12to12_analog_k8_candidate_indices.npz", allow_pickle=False) as z:
        candidates = np.asarray(z["candidate_index_test"], dtype=np.int32)[positions]
    logger = logging.getLogger("route_benchmark")
    logger.setLevel(logging.ERROR)
    common = dict(data=raw, time_data=time_data, idx=endpoints, seq_len=12,
                  horizon=12, bs=1, logger=logger, pad_last_sample=False,
                  scaler=scaler, drop_last=False)
    student_loader = DataLoader_time(**common, retrieval_data=sr,
        exact_retrieval_data=sx, exact_available_data=sa)
    teacher_loader = DataLoader_time(**common, retrieval_data=tr,
        exact_retrieval_data=tx, exact_available_data=ta,
        analog_candidate_index_data=candidates)
    del sr, sx, sa, tr, tx, ta, candidates

    cal_length = len(val_indices)//4
    with np.load(ROOT/paths["student"], allow_pickle=False) as z:
        fit_student = np.asarray(z["val_prediction"][:cal_length], dtype=np.float64)
        fit_truth = np.asarray(z["val_truth"][:cal_length], dtype=np.float64)
    with np.load(ROOT/paths["teacher"], allow_pickle=False) as z:
        fit_teacher = np.asarray(z["teacher_pred_val"][:cal_length], dtype=np.float64)[..., 0]
        fit_teacher = fit_teacher*std+mean
    student_tables = calibrator(fit_student, fit_truth, val_indices[:cal_length], params)
    teacher_tables = calibrator(fit_teacher, fit_truth, val_indices[:cal_length], params)
    del fit_student, fit_teacher, fit_truth

    saved_argv = sys.argv
    sys.argv = [sys.argv[0]]
    student_engine, _ = load_engine(ROOT/"EXP_experiments/TLAST_improved/PEMS07/TLAST_improved_PEMS07_12to12_exppems07_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s1", scaler, raw.shape[1], device)
    teacher_engine, _ = load_engine(ROOT/"EXP_experiments/TLAST_improved/PEMS07/TLAST_improved_PEMS07_12to12_exppaper_pems07_s1_final_joint", scaler, raw.shape[1], device)
    sys.argv = saved_argv
    gates = []
    for state in gate_state["model_states"]:
        model = UtilityGate(len(gate_state["feature_mean"]))
        model.load_state_dict(state, strict=True)
        gates.append(model.eval())
    feature_mean = np.asarray(gate_state["feature_mean"])
    feature_std = np.asarray(gate_state["feature_std"])
    threshold = float(gate_state["selection_threshold_10pct"])
    with np.load(HERE/"pems07_calibrated_gate_final_test_paired_audit.npz", allow_pickle=False) as z:
        expected_calls = np.asarray(z["gate_selected"][:512], dtype=bool)
        expected_scores = np.asarray(z["gate_score"][:512], dtype=np.float64)
    with np.load(ROOT/paths["student"], allow_pickle=False) as z:
        expected_student = np.asarray(z["test_prediction"][:512], dtype=np.float32)
    with np.load(ROOT/paths["teacher"], allow_pickle=False) as z:
        expected_teacher = np.asarray(z["teacher_pred_test"][:512], dtype=np.float32)[..., 0]*std+mean

    def decision(row, calibrated_student):
        visible, _ = build_features(raw[..., 0], endpoints[row:row+1], calibrated_student)
        scaled = ((visible-feature_mean)/feature_std).astype(np.float32)
        tensor = torch.from_numpy(scaled)
        with torch.inference_mode():
            return float(np.mean([model(tensor).item() for model in gates]))

    # Parity check of both live model predictions and the five-model gate.
    student_max = teacher_max = score_max = 0.0
    for row in (0, 1, 100, 255, 511):
        raw_s, cal_s = call_model(student_engine, student_loader, row, int(endpoints[row]), device, False, student_tables, params)
        raw_t, _ = call_model(teacher_engine, teacher_loader, row, int(endpoints[row]), device, True, teacher_tables, params)
        student_max = max(student_max, float(np.max(np.abs(raw_s[0]-expected_student[row]))))
        teacher_max = max(teacher_max, float(np.max(np.abs(raw_t[0]-expected_teacher[row]))))
        score_max = max(score_max, abs(decision(row, cal_s)-expected_scores[row]))
    print("parity", student_max, teacher_max, score_max, flush=True)
    # CUDA floating-point kernel order gives <0.01 raw-unit variation here;
    # the gate score itself must remain essentially identical.
    if student_max > 0.02 or teacher_max > 0.02 or score_max > 1e-3:
        raise ValueError("Live predictions/gate scores do not match frozen caches")

    def timed(mode):
        times = []
        calls = []
        for repeat in range(3):
            tokens = 0.0
            call_flags = np.zeros(len(endpoints), dtype=bool)
            for row, endpoint in enumerate(endpoints):
                start = time.perf_counter()
                if mode == "teacher":
                    call_model(teacher_engine, teacher_loader, row, int(endpoint), device, True, teacher_tables, params)
                else:
                    _, student_cal = call_model(student_engine, student_loader, row, int(endpoint), device, False, student_tables, params)
                    if mode == "routed":
                        score = decision(row, student_cal)
                        tokens = min(51.0, tokens+0.1)
                        if score >= threshold and tokens+1e-12 >= 1.0:
                            call_model(teacher_engine, teacher_loader, row, int(endpoint), device, True, teacher_tables, params)
                            call_flags[row] = True
                            tokens -= 1.0
                times.append((time.perf_counter()-start)*1000)
            calls.append(call_flags)
        if mode == "routed" and not np.array_equal(calls[0], expected_calls):
            raise ValueError("Online gate decisions do not match frozen full-Test prefix")
        return {
            "mean_ms": float(np.mean(times)),
            "p50_ms": float(np.percentile(times, 50)),
            "p95_ms": float(np.percentile(times, 95)),
            "teacher_calls_per_pass": int(calls[0].sum()) if mode == "routed" else (len(endpoints) if mode == "teacher" else 0),
            "samples_per_pass": len(endpoints),
            "repeats": 3,
        }
    # Warm both models, gate, candidate construction, and calibration.
    for row in range(16):
        endpoint = int(endpoints[row])
        _, cal_s = call_model(student_engine, student_loader, row, endpoint, device, False, student_tables, params)
        decision(row, cal_s)
        call_model(teacher_engine, teacher_loader, row, endpoint, device, True, teacher_tables, params)
    results = {mode: timed(mode) for mode in ("student", "teacher", "routed")}
    report = {
        "protocol": "PEMS07_same_process_batch1_first512_Test_requests_frozen_gate_3_repeats",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "test_labels_read": False,
        "request_scope": "first 512 consecutive official Test endpoints; no result-based selection",
        "retrieval": "frozen precomputed Train-memory cache lookup; teacher candidate features built on demand",
        "timed": "input slicing + cache payload construction + CPU/GPU transfer + model forward + calibration + CPU gate + conditional Teacher",
        "parity_max_abs": {"student": student_max, "teacher": teacher_max, "gate_score": score_max},
        "results": results,
        "caveat": "Sequential single-process replay on 512 requests, not full-Test production/concurrent throughput; model/cache loading excluded; 3 repeats share a process.",
    }
    out = HERE/"pems07_integrated_route_latency_512.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
