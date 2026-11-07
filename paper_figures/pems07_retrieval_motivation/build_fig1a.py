"""Build a PEMS07 validation-only retrieval-utility figure from frozen caches.

No Test labels, calibrated outputs, random data, or SRM-masked retrieval outputs
are used. Node IDs are zero-based array-column indices.
"""
from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA = ROOT / "datasets/PEMS07/data.npz"
TIME = ROOT / "datasets/PEMS07/PEMS07_time.npz"
INDEX = ROOT / "datasets/PEMS07/PEMS07_12to12_index_scaler.npz"
CACHE = ROOT / "datasets/PEMS07/PEMS07_12to12_analog_k4_exact_mean.npz"
EXP = ROOT / ("EXP_experiments/TLAST_improved/PEMS07/"
              "TLAST_improved_PEMS07_12to12_exppems07_lite_r256_"
              "kdfix3_a30_lr1e4_fixedteacher_s1")
BASE = ROOT / ("paper_results/icde_retrieval_free_pilot_20261012/"
               "pems07_node_budget_s1_no_memory/no_memory_validation_predictions.npz")
RETR = ROOT / ("paper_results/pems07_lite_r256_kdfix3_a30_lr1e4_"
               "fixedteacher_s1_network_only/selected_predictions.npz")
PROV = ROOT / ("paper_results/icde_retrieval_free_pilot_20261012/"
               "pems07_node_budget_s1_validation_screen.json")
SRM = ROOT / "cold_start_streaming_20261014/pems07_sparse_cold_state.npz"
START = datetime(2017, 5, 1)  # utils/datasets_utils.py, meta_info_dict
P = Q = 12
N = 883


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_checked():
    for p in (DATA, TIME, INDEX, CACHE, EXP/"config.pkl", EXP/"best.pth",
              BASE, RETR, PROV, SRM):
        if not p.is_file():
            raise FileNotFoundError(p)
    with (EXP/"config.pkl").open("rb") as f:
        cfg = pickle.load(f)
    wanted = {"dataset_name": "PEMS07", "input_length": P,
              "predict_length": Q, "analog_retrieval_k": 4,
              "use_analog_lowrank_retrieval_residual": 1,
              "use_analog_exact_match_residual": 1,
              "use_analog_confidence_aware_retrieval_residual": 0}
    for key, value in wanted.items():
        if getattr(cfg, key, None) != value:
            raise ValueError(f"Checkpoint configuration differs: {key}")
    if Path(cfg.analog_retrieval_cache).as_posix() != CACHE.relative_to(ROOT).as_posix():
        raise ValueError("Checkpoint references another retrieval cache")
    with np.load(INDEX, allow_pickle=False) as z:
        train, val, test = (np.array(z[f"idx_{s}"], dtype=np.int64)
                            for s in ("train", "val", "test"))
    if (len(train), len(val), len(test)) != (16921, 5640, 5640):
        raise ValueError("Unexpected project 12-to-12 split")
    if not (np.all(np.diff(train) == 1) and np.all(np.diff(val) == 1)
            and train[-1] < val[0] and val[-1] < test[0]):
        raise ValueError("Split endpoints are not chronological")
    with np.load(CACHE, allow_pickle=False) as z:
        metadata = {"k": 4, "input_length": P, "predict_length": Q,
                    "min_valid_history": 6, "calendar_mode": "tod",
                    "history_distance_metric": "masked_mse",
                    "candidate_selection_strategy": "nearest",
                    "exact_match_rule": "masked_mse_eq_0",
                    "exact_min_valid_history": 6,
                    "train_leave_one_out": 1}
        for key, value in metadata.items():
            if z[key].item() != value:
                raise ValueError(f"Retrieval cache differs: {key}")
        if not (np.array_equal(z["idx_train"], train)
                and np.array_equal(z["idx_val"], val)):
            raise ValueError("Retrieval cache indices differ")
    prov = json.loads(PROV.read_text(encoding="utf-8"))
    ps = prov["prediction_sources"]
    prefix = int(.55 * len(val))
    if (Path(ps["no_memory"]).resolve() != BASE.resolve()
            or Path(ps["full_memory"]).resolve() != RETR.resolve()
            or prov["folds"][0]["fit_samples"] != prefix):
        raise ValueError("Archived same-checkpoint cache provenance differs")
    with np.load(RETR, allow_pickle=False) as z:
        er = np.array(z["val_endpoints"], dtype=np.int64)
        retr = np.array(z["val_prediction"][:prefix], copy=True)
        truth = np.array(z["val_truth"][:prefix], copy=True)
    with np.load(BASE, allow_pickle=False) as z:
        eb = np.array(z["val_endpoints"], dtype=np.int64)
        base = np.array(z["val_prediction"][:prefix], copy=True)
        base_truth = np.array(z["val_truth"][:prefix], copy=True)
    if not (np.array_equal(er, val) and np.array_equal(eb, val)
            and np.array_equal(base_truth, truth)):
        raise ValueError("Base and retrieval endpoints/labels do not match")
    for name, arr in (("base", base), ("retrieval", retr), ("truth", truth)):
        if arr.shape != (prefix, Q, N) or arr.dtype != np.float32 or not np.isfinite(arr).all():
            raise ValueError(f"Unexpected {name} array")
    with np.load(TIME, allow_pickle=False) as z:
        time = np.array(z["time"], copy=True)
    frames = np.arange(28224)
    if (time.shape != (28224, 2) or time.dtype != np.int64
            or not np.array_equal(time[:, 0], frames % 288)
            or not np.array_equal(time[:, 1], (frames // 288) % 7)):
        raise ValueError("TOD/DOW features differ from five-minute project calendar")
    # Ground truth is checked against raw traffic channel 0 for *every* plotted
    # selection-prefix sample; the time archive is never used as flow data.
    with np.load(DATA, allow_pickle=False) as z:
        raw = z["data"]
    if raw.shape != (28224, N, 3) or raw.dtype != np.float64:
        raise ValueError("Unexpected PEMS07 traffic array")
    for i in range(0, prefix, 96):
        j = min(i + 96, prefix)
        indices = val[i:j, None] + np.arange(1, Q+1)[None, :]
        if not np.array_equal(raw[indices, :, 0].astype(np.float32), truth[i:j]):
            raise ValueError("Cached truth differs from raw traffic flow")
    # Audit a strong dataset peculiarity before choosing an illustrative node:
    # exact equality at the same five-minute slot one week (2016 frames) ago.
    week = 7 * 288
    first_target = int(val[0] + 1)
    last_target_exclusive = int(val[prefix-1] + Q + 1)
    weekly_repeat = np.mean(
        raw[first_target:last_target_exclusive, :, 0] ==
        raw[first_target-week:last_target_exclusive-week, :, 0], axis=0)
    del raw, base_truth
    with np.load(SRM, allow_pickle=False) as z:
        srm = np.array(z["node_ids"], dtype=np.int64)
    if len(np.unique(srm)) != len(srm) or np.any(srm < 0) or np.any(srm >= N):
        raise ValueError("Invalid frozen SRM node set")
    return base, retr, truth, val, prefix, srm, weekly_repeat


def calculate_gain(base, retr, truth):
    bsum = np.zeros(N, np.float64)
    rsum = np.zeros(N, np.float64)
    count = np.zeros(N, np.int64)
    for i in range(0, len(truth), 128):
        j = min(i+128, len(truth))
        y = truth[i:j].astype(np.float64)
        b = base[i:j].astype(np.float64)
        r = retr[i:j].astype(np.float64)
        mask = np.rint(y) > 0
        count += mask.sum((0, 1))
        bsum += np.where(mask, np.abs(b-y), 0).sum((0, 1))
        rsum += np.where(mask, np.abs(r-y), 0).sum((0, 1))
    if np.any(count == 0):
        raise ValueError("At least one PEMS07 node has no valid prefix target")
    bmae, rmae = bsum/count, rsum/count
    return bmae, rmae, bmae-rmae, count


def nearest(ids, gain, target):
    return int(ids[np.lexsort((ids, np.abs(gain[ids]-target)))[0]])


def choose_nodes(gain, base_mae, weekly_repeat):
    positive, negative = np.flatnonzero(gain > 0), np.flatnonzero(gain < 0)
    if not len(positive) or not len(negative):
        raise ValueError("Observed gains do not support all three panel classes")
    p90, n10 = np.quantile(gain[positive], .9), np.quantile(gain[negative], .1)
    # A must not owe its striking appearance to an almost exact weekly copy.
    # A >=10% relative MAE reduction makes the high-utility curve visible
    # without selecting an extreme single window. Both filters use Validation.
    a_pool = positive[(weekly_repeat[positive] < .95)
                      & (gain[positive] / base_mae[positive] >= .10)]
    if not len(a_pool):
        raise ValueError("No non-duplicated, materially improved high-gain node")
    a = nearest(a_pool, gain, p90)
    c = nearest(negative, gain, n10)
    b = nearest(np.setdiff1d(np.arange(N), [a, c]), gain, 0)
    return [a, b, c], {"positive_p90": float(p90), "negative_p10": float(n10),
                       "a_pool": "positive nodes with <95% exact seven-day "
                                 "repetition and >=10% relative MAE reduction",
                       "a_pool_size": int(len(a_pool))}


def choose_window(nodes, gain, base, retr, truth):
    y, b, r = (arr[:, :, nodes].astype(np.float64)
               for arr in (truth, base, retr))
    mask = np.rint(y) > 0
    count = mask.sum(axis=1)
    bmae = np.divide(np.where(mask, np.abs(b-y), 0).sum(axis=1), count,
                     out=np.full_like(count, np.nan, dtype=float), where=count > 0)
    rmae = np.divide(np.where(mask, np.abs(r-y), 0).sum(axis=1), count,
                     out=np.full_like(count, np.nan, dtype=float), where=count > 0)
    wg = bmae-rmae
    full = np.all(count == Q, axis=1)
    signs = full & (wg[:, 0] > 0) & (wg[:, 2] < 0)
    if not np.any(signs):
        raise ValueError("No shared complete window preserves A/C gain signs")
    gap = np.abs(wg-gain[nodes][None, :])
    scale = np.maximum(np.median(gap[full], axis=0), .01)
    for tail in (.10, .05, 0):
        central = signs.copy()
        # Exclude unusually easy or difficult 12-step windows as well as
        # extreme window gains; otherwise a close gain could conceal highly
        # unrepresentative absolute prediction errors.
        for values in (wg, bmae, rmae):
            lo, hi = (np.quantile(values[full], p, axis=0)
                      for p in (tail, 1-tail))
            central &= np.all((values >= lo) & (values <= hi), axis=1)
        candidates = np.flatnonzero(central)
        if len(candidates):
            standardized = gap[candidates] / scale[None, :]
            score = standardized.max(axis=1) + .1*standardized.mean(axis=1)
            chosen = int(candidates[np.lexsort((candidates, score))[0]])
            return chosen, {
                "valid_common_candidates": int(signs.sum()),
                "selected_range_percentiles": [100*tail, 100*(1-tail)],
                "rule": "same sample; all 12 targets valid; A positive and C "
                        "negative window gains; window gain, base MAE, and "
                        "retrieval MAE in their non-extreme ranges; minimize "
                        "standardized distance to three node-wise gains",
                "base_mae": bmae[chosen].tolist(),
                "retrieval_mae": rmae[chosen].tolist(),
                "gain": wg[chosen].tolist(),
                "valid_count": count[chosen].tolist(),
            }
    raise AssertionError("A sign-consistent pool existed")


class Vectors:
    """Simple matching vector primitives in top-origin SVG and PDF coordinates."""
    def __init__(self, svg_path, pdf_path, w, h):
        self.w, self.h, self.svg_path = w, h, svg_path
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w/72:.3f}in" '
            f'height="{h/72:.3f}in" viewBox="0 0 {w:.3f} {h:.3f}">',
            '<title>Heterogeneous retrieval utility across PEMS07 nodes</title>',
            f'<rect x="0" y="0" width="{w}" height="{h}" fill="#ffffff"/>',
        ]
        self.pdf = canvas.Canvas(str(pdf_path), pagesize=(w, h))
        self.pdf.setTitle("Heterogeneous Retrieval Utility across PEMS07 Nodes")

    def line(self, x1, y1, x2, y2, color="#333333", width=.6, dash=None):
        extra = f' stroke-dasharray="{dash[0]},{dash[1]}"' if dash else ""
        self.parts.append(f'<line x1="{x1:.3f}" y1="{y1:.3f}" '
                          f'x2="{x2:.3f}" y2="{y2:.3f}" stroke="{color}" '
                          f'stroke-width="{width}"{extra}/>')
        self.pdf.setStrokeColor(HexColor(color)); self.pdf.setLineWidth(width)
        self.pdf.setDash(*dash) if dash else self.pdf.setDash()
        self.pdf.line(x1, self.h-y1, x2, self.h-y2); self.pdf.setDash()

    def poly(self, points, color, width, dash=None):
        coords = " ".join(f"{x:.3f},{y:.3f}" for x, y in points)
        extra = f' stroke-dasharray="{dash[0]},{dash[1]}"' if dash else ""
        self.parts.append(f'<polyline points="{coords}" fill="none" '
                          f'stroke="{color}" stroke-width="{width}" '
                          f'stroke-linecap="round" stroke-linejoin="round"{extra}/>')
        self.pdf.setStrokeColor(HexColor(color)); self.pdf.setLineWidth(width)
        self.pdf.setDash(*dash) if dash else self.pdf.setDash()
        path = self.pdf.beginPath()
        path.moveTo(points[0][0], self.h-points[0][1])
        for x, y in points[1:]: path.lineTo(x, self.h-y)
        self.pdf.drawPath(path); self.pdf.setDash()

    def text(self, x, y, value, size=7.5, bold=False, anchor="start", rotate=False):
        font = "TNR-Bold" if bold else "TNR"
        transform = f' transform="rotate(-90 {x:.3f} {y:.3f})"' if rotate else ""
        self.parts.append(f'<text x="{x:.3f}" y="{y:.3f}" '
                          f'font-family="Times New Roman" font-size="{size}" '
                          f'font-weight="{700 if bold else 400}" fill="#111111" '
                          f'text-anchor="{anchor}"{transform}>{html.escape(str(value))}</text>')
        self.pdf.saveState(); self.pdf.setFont(font, size)
        if rotate:
            self.pdf.translate(x, self.h-y); self.pdf.rotate(90); xx, yy = 0, 0
        else:
            xx, yy = x, self.h-y
        if anchor == "middle": self.pdf.drawCentredString(xx, yy, str(value))
        elif anchor == "end": self.pdf.drawRightString(xx, yy, str(value))
        else: self.pdf.drawString(xx, yy, str(value))
        self.pdf.restoreState()

    def save(self):
        self.parts.append("</svg>")
        self.svg_path.write_text("\n".join(self.parts)+"\n", encoding="utf-8")
        self.pdf.showPage(); self.pdf.save()


def ticks_for(values):
    lo, hi = float(np.min(values)), float(np.max(values))
    pad = max((hi-lo)*.09, 1.)
    lo, hi = max(0., lo-pad), hi+pad
    raw = (hi-lo)/3
    unit = 10**math.floor(math.log10(raw))
    step = next(v*unit for v in (1, 2, 2.5, 5, 10) if v*unit >= raw)
    first, last = math.floor(lo/step)*step, math.ceil(hi/step)*step
    return first, last, np.arange(max(0., first), last+step/2, step)


def draw_figure(nodes, sample, gain, base, retr, truth):
    pdfmetrics.registerFont(TTFont("TNR", r"C:\Windows\Fonts\times.ttf"))
    pdfmetrics.registerFont(TTFont("TNR-Bold", r"C:\Windows\Fonts\timesbd.ttf"))
    w, h = 7.16*72, 2.85*72
    d = Vectors(OUT/"fig1a_pems07_three_nodes.svg",
                OUT/"fig1a_pems07_three_nodes.pdf", w, h)
    black, blue, green = "#111111", "#2860A4", "#147A55"
    for x, label, color, dash in ((70, "Ground Truth", black, None),
                                   (220, "Base Prediction", blue, (4, 2)),
                                   (371, "Retrieval Prediction", green, None)):
        d.line(x, 13, x+20, 13, color, 1.5, dash)
        d.text(x+25, 15.5, label, 8)
    heads = ("(a) High retrieval gain", "(b) Near-zero retrieval gain",
             "(c) Negative retrieval gain")
    scales = []
    for j, node in enumerate(nodes):
        left = 8+j*169
        x0, x1, y0, y1 = left+42, left+157, 65, 163
        d.text(left+84, 34, heads[j], 8.5, True, "middle")
        gain_label = f"{gain[node]:+.5f}" if abs(gain[node]) < .001 else f"{gain[node]:+.3f}"
        d.text(left+84, 48, f"Node {node}  |  Val. gain {gain_label}",
               7.7, anchor="middle")
        curves = (truth[sample, :, node], base[sample, :, node], retr[sample, :, node])
        low, high, ticks = ticks_for(np.concatenate(curves))
        scales.append([low, high])
        def x_at(step): return x0+(step-1)*(x1-x0)/(Q-1)
        def y_at(v): return y1-(float(v)-low)*(y1-y0)/(high-low)
        for tick in ticks:
            yy = y_at(tick)
            d.line(x0, yy, x1, yy, "#D9DEE3", .35)
            label = f"{tick:.0f}" if abs(tick) >= 10 or tick == int(tick) else f"{tick:.1f}"
            d.text(x0-4, yy+2.3, label, 7, anchor="end")
        d.line(x0, y0, x0, y1); d.line(x0, y1, x1, y1)
        for step in (1, 4, 8, 12):
            xx = x_at(step)
            d.line(xx, y1, xx, y1+2.5, width=.5)
            d.text(xx, y1+12, str(step), 7, anchor="middle")
        for curve, color, lw, dash in ((curves[0], black, 2., None),
                                       (curves[1], blue, 1.7, (4, 2)),
                                       (curves[2], green, 1.7, None)):
            d.poly([(x_at(step), y_at(curve[step-1]))
                    for step in range(1, Q+1)], color, lw, dash)
        d.text(left+99, 192, "Forecast horizon (5-min steps)", 7.7,
               anchor="middle")
        d.text(left+7, 114, "Traffic flow", 7.7, anchor="middle", rotate=True)
    d.save()
    svg = (OUT/"fig1a_pems07_three_nodes.svg").read_text(encoding="utf-8")
    if "<image" in svg or svg.count("<polyline") != 9 or "<text" not in svg:
        raise ValueError("SVG is not editable vector curves and text")
    return scales


def csv_file(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    base, retr, truth, endpoints, prefix, srm, weekly_repeat = load_checked()
    bmae, rmae, gain, count = calculate_gain(base, retr, truth)
    nodes, node_rule = choose_nodes(gain, bmae, weekly_repeat)
    sample, window = choose_window(nodes, gain, base, retr, truth)
    csv_file(OUT/"node_gain.csv", [
        {"node_id": n, "base_mae": f"{bmae[n]:.10f}",
         "retrieval_mae": f"{rmae[n]:.10f}",
         "gain": f"{gain[n]:.10f}", "valid_count": int(count[n]),
         "seven_day_exact_repeat_rate": f"{weekly_repeat[n]:.10f}"}
        for n in range(N)])
    roles = ("A_high", "B_near_zero", "C_negative")
    selected = [
        {"role": role, "node_id": n, "base_mae": f"{bmae[n]:.10f}",
         "retrieval_mae": f"{rmae[n]:.10f}", "gain": f"{gain[n]:.10f}",
         "valid_count": int(count[n]), "srm_selected": bool(n in srm),
         "seven_day_exact_repeat_rate": f"{weekly_repeat[n]:.10f}"}
        for role, n in zip(roles, nodes)]
    csv_file(OUT/"selected_three_nodes.csv", selected)
    endpoint = int(endpoints[sample])
    start = START+timedelta(minutes=5*(endpoint+1))
    end = START+timedelta(minutes=5*(endpoint+Q))
    windows = [
        {"role": role, "node_id": n, "val_sample_index_0based": sample,
         "input_endpoint_index_0based": endpoint,
         "forecast_start": str(start), "forecast_end": str(end),
         "window_base_mae": f"{window['base_mae'][j]:.10f}",
         "window_retrieval_mae": f"{window['retrieval_mae'][j]:.10f}",
         "window_gain": f"{window['gain'][j]:.10f}",
         "window_valid_count": int(window['valid_count'][j]),
         "node_validation_gain": f"{gain[n]:.10f}"}
        for j, (role, n) in enumerate(zip(roles, nodes))]
    csv_file(OUT/"selected_windows.csv", windows)
    figure_data = {
        "node_id": np.asarray(nodes, np.int32),
        "val_sample_index_0based": np.full(3, sample, np.int32),
        "input_endpoint_index_0based": np.full(3, endpoint, np.int32),
        "forecast_horizon": np.arange(1, Q+1, dtype=np.int32),
        "target_frame_index": np.tile(endpoint+np.arange(1, Q+1), (3, 1)),
        "truth": np.stack([truth[sample, :, n] for n in nodes]),
        "base_prediction": np.stack([base[sample, :, n] for n in nodes]),
        "retrieval_prediction": np.stack([retr[sample, :, n] for n in nodes]),
        "node_gain": gain[nodes], "node_base_mae": bmae[nodes],
        "node_retrieval_mae": rmae[nodes], "window_gain": window["gain"],
    }
    np.savez_compressed(OUT/"figure_data.npz", **figure_data)
    with np.load(OUT/"figure_data.npz", allow_pickle=False) as z:
        for k, original in figure_data.items():
            if not np.array_equal(z[k], original):
                raise ValueError(f"Figure data failed round trip: {k}")
    scales = draw_figure(nodes, sample, gain, base, retr, truth)
    pos, neg = int(np.sum(gain > 0)), int(np.sum(gain < 0))
    quantiles = {str(p): float(np.quantile(gain, p))
                 for p in (.10, .25, .75, .90)}
    audit = {
        "dataset": "PEMS07", "checkpoint_seed": 1,
        "validation_prefix_count": prefix,
        "validation_total_count": len(endpoints),
        "valid_nodes": int(np.sum(count > 0)),
        "positive_nodes": pos, "positive_fraction": pos/N,
        "negative_nodes": neg, "negative_fraction": neg/N,
        "zero_gain_nodes": int(np.sum(gain == 0)),
        "gain_min": float(np.min(gain)), "gain_median": float(np.median(gain)),
        "gain_max": float(np.max(gain)), "gain_quantiles": quantiles,
        "selection_rule": node_rule, "window_selection": window,
        "selected_nodes": selected, "selected_windows": windows,
        "srm_node_count": len(srm), "panel_y_scales": scales,
        "nodes_with_weekly_repeat_ge_95pct": int(np.sum(weekly_repeat >= .95)),
        "srm_nodes_with_weekly_repeat_ge_95pct":
            int(np.sum(weekly_repeat[srm] >= .95)),
        "srm_min_weekly_repeat": float(np.min(weekly_repeat[srm])),
        "source_sha256": {str(p.relative_to(ROOT)): digest(p)
                          for p in (DATA, TIME, INDEX, BASE, RETR, EXP/"best.pth", SRM)},
        "new_inference_run": False, "calibration_used": False,
        "test_labels_read": False,
    }
    (OUT/"audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    caption = (
        "Heterogeneous retrieval utility across PEMS07 nodes. Using the same "
        "frozen Student, uncalibrated base and train-only retrieval predictions "
        "are compared with ground truth on the first 55\\% of Validation. "
        "Node-wise gain is the difference between masked MAEs over all 12 "
        "forecast horizons; each panel shows one representative 12-step "
        "window. The illustrated high-gain node excludes near-exact weekly "
        "repeats; positive and negative gains indicate lower and higher "
        "prediction error after retrieval, respectively."
    )
    readme = [
        "# PEMS07 retrieval-utility motivation figure", "",
        "This is a real-data, uncalibrated, Validation-only seed-1 diagnostic. "
        "The paper's later seed-2 selection and Test result are separate.", "",
        "## Sources and protocol", "",
        f"- Raw flow: `{DATA.relative_to(ROOT)}` (`data`, 28224 x 883 x 3, "
        "float64). Only channel 0 is used as traffic flow.",
        f"- Time features: `{TIME.relative_to(ROOT)}` (`time`, 28224 x 2, "
        "int64). These are TOD/DOW, **not flow**. Absolute timestamps use "
        "the project metadata origin 2017-05-01 00:00:00 and 5-minute steps.",
        f"- Indices: `{INDEX.relative_to(ROOT)}`; 16921 Train, 5640 Validation, "
        f"5640 Test endpoints. P=Q=12; only the first {prefix} Validation "
        "queries are evaluated here.",
        f"- Frozen Student: `{(EXP/'best.pth').relative_to(ROOT)}`.",
        f"- Uncalibrated Base: `{BASE.relative_to(ROOT)}`; uncalibrated "
        f"all-node Retrieval: `{RETR.relative_to(ROOT)}`. Their archived "
        f"same-checkpoint provenance is `{PROV.relative_to(ROOT)}`.",
        f"- Train-only retrieval source: `{CACHE.relative_to(ROOT)}`. HAR uses "
        "the same 5-minute TOD slot, Top-4 nearest Train histories, masked "
        "mean squared distance and at least six jointly nonzero positions. "
        "Exact matching independently searches the entire same-TOD Train "
        "group for six jointly valid, exactly matching history positions.",
        f"- Frozen SRM IDs: `{SRM.relative_to(ROOT)}`; used only for the "
        "membership annotation, not to mask retrieval predictions.", "",
        "All cached labels were matched exactly to raw flow channel 0 at "
        "endpoint + horizons 1-12. Base and Retrieval archives have exactly "
        "the same Validation endpoints and labels. Prediction values are in "
        "raw traffic-flow units.", "",
        "## Gain and node selection", "",
        "`gain = masked_MAE(Base) - masked_MAE(Retrieval)`, with the same "
        "`round(truth)>0` mask and all 12 horizons. Node IDs are zero-based "
        "array-column indices, not external sensor identifiers.", "",
        f"Valid nodes: {N}; positive: {pos} ({pos/N:.2%}); negative: {neg} "
        f"({neg/N:.2%}). Gain min/median/max: {np.min(gain):.6f} / "
        f"{np.median(gain):.6f} / {np.max(gain):.6f}.",
        "Gain quantiles 10/25/75/90%: " + " / ".join(
            f"{quantiles[str(p)]:.6f}" for p in (.10, .25, .75, .90)) + ".", "",
        "The raw data contain a near-exact weekly-repeat subset: "
        f"{int(np.sum(weekly_repeat >= .95))} nodes repeat at least 95% of "
        "their first-55% Validation frames from seven days earlier; "
        f"all {len(srm)} frozen SRM nodes meet this threshold. This is "
        "a data characteristic, not evidence of target leakage by itself. "
        "It is reported because near-perfect retrieval curves from that "
        "subset would overstate the generality of the illustration.", "",
        "A is the positive node nearest the positive-class "
        f"90th percentile ({node_rule['positive_p90']:.6f}) among nodes with "
        "under 95% seven-day exact repetition and at least 10% relative "
        f"MAE reduction ({node_rule['a_pool_size']} candidates); B has the "
        "smallest absolute gain excluding A/C; C is nearest the "
        f"negative-class 10th percentile ({node_rule['negative_p10']:.6f}). "
        "Ties use the smaller node ID. The screen changes only which real "
        "node is displayed; no gain or curve value is changed. SRM "
        "membership remains a separate property.", "",
        f"Window selection: {window['rule']}. Shared eligible windows: "
        f"{window['valid_common_candidates']}; chosen sample: {sample}.", "",
        "| Role | Node ID | Base MAE | Retrieval MAE | Node gain | SRM? | "
        "Validation sample | Forecast start | Window gain |",
        "|---|---:|---:|---:|---:|:---:|---:|---|---:|",
    ]
    for row, win in zip(selected, windows):
        readme.append(f"| {row['role']} | {row['node_id']} | "
                      f"{float(row['base_mae']):.4f} | "
                      f"{float(row['retrieval_mae']):.4f} | "
                      f"{float(row['gain']):+.5f}" if abs(float(row['gain'])) < .001 else f"{float(row['gain']):+.4f}"
                      f" | "
                      f"{'yes' if row['srm_selected'] else 'no'} | "
                      f"{sample} | {win['forecast_start']} | "
                      f"{float(win['window_gain']):+.4f} |")
    readme.extend([
        "", "## Files and reproducibility", "",
        "`node_gain.csv` contains all 883 rows. `selected_three_nodes.csv` and "
        "`selected_windows.csv` retain node- and window-level quantities "
        "separately. `figure_data.npz` stores the exact plotted arrays and "
        "indices. `audit.json` stores hashes and all-node statistics. "
        "The SVG contains nine real vector polylines plus editable text, "
        "with no bitmap embedding. The PDF uses embedded Times New Roman; "
        "the PNG is rendered from it at 300 dpi. Run `build_fig1a.py` from "
        "any working directory using a Python with NumPy and ReportLab.", "",
        "Suggested IEEE caption:", "", caption, "",
        "The absolute date follows repository metadata. The time NPZ stores "
        "only TOD/DOW, and the figure does not imply that the seed-2 frozen "
        "system has identical node-wise gains.", "",
    ])
    (OUT/"README.md").write_text("\n".join(readme), encoding="utf-8")
    print(json.dumps({"summary": {k: audit[k] for k in
                                 ("valid_nodes", "positive_nodes", "negative_nodes",
                                  "gain_min", "gain_median", "gain_max", "gain_quantiles")},
                      "nodes": selected, "windows": windows,
                      "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
