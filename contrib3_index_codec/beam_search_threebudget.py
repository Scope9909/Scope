"""Validation-only beam search for three-budget online-state coding.

This driver deliberately keeps the existing ``select_retrieval_aware.py``
implementation as the source of truth for dataset/model loading, codec
construction, MAE evaluation, and retrieval-stability auditing.  It never
loads Test targets and writes a new report instead of touching greedy JSON.
"""
import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument("dataset", choices=("PEMS03", "PEMS04", "PEMS07", "PEMS08"))
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--gate", type=float, default=0.02)
ap.add_argument("--topk-gate", type=float, default=0.005)
ap.add_argument("--exact-gate", type=float, default=0.01)
ap.add_argument("--strong-lossless-bytes", type=int, required=True)
ap.add_argument("--max-candidates", type=int, default=8)
ap.add_argument("--beam-width", type=int, default=3)
ap.add_argument("--max-depth", type=int, default=4)
ap.add_argument("--probe-batches", type=int, default=1000)
ap.add_argument("--output", type=Path, required=True)
cli = ap.parse_args()

# Import the established implementation without executing its main().
select_path = HERE / "select_retrieval_aware.py"
sys.argv = [str(select_path), cli.dataset, "--seed", str(cli.seed),
            "--gate", str(cli.gate), "--topk-gate", str(cli.topk_gate),
            "--exact-gate", str(cli.exact_gate),
            "--probe-batches", str(cli.probe_batches)]
spec = importlib.util.spec_from_file_location("select_retrieval_aware", select_path)
sel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sel)
sys.argv = [__file__]


def evaluate_state(nodes, val_path, loader, endpoints, scaler, model, device,
                   codec, base_memory, base_path, baseline_mae, temp, cache):
    path = temp / ("state_" + ("_".join(map(str, nodes)) or "zero") + ".npz")
    size = codec.make_mixed_state(val_path, path, list(nodes), pack_index=True)
    if size >= cache.get("best_bytes", 1 << 62):
        return None
    memory = codec.load_memory(path)
    predicted, target = sel.runner.collect(
        loader, endpoints, model, scaler, memory, device, 32)
    delta = sel.runner.metrics(predicted, target)["mae"] - baseline_mae
    # Re-open both states for stability: collect() advances cold-start state.
    stability = sel.retrieval_change(
        codec.load_memory(base_path), codec.load_memory(path), cache["batches"])
    passed = (delta <= cli.gate and
              stability["topk_set_flip_rate"] <= cli.topk_gate and
              stability["exact_hit_flip_rate"] <= cli.exact_gate and
              size < cli.strong_lossless_bytes)
    return dict(nodes=list(nodes), bytes=int(size), saving=int(cache["base_bytes"] - size),
                val_delta=float(delta), stability=stability, passed=bool(passed),
                deployment_viable=bool(size < cli.strong_lossless_bytes), path=str(path))


def main():
    val_path, loader, endpoints, scaler, model, device = sel.setup()
    with tempfile.TemporaryDirectory(prefix="beam_threebudget_") as directory:
        temp = Path(directory)
        base_path = temp / "lossless.npz"
        base_bytes = sel.codec.make_mixed_state(val_path, base_path, [], pack_index=True)
        base_memory = sel.codec.load_memory(base_path)
        batches = sel.probe_inputs(loader, scaler, base_memory, cli.probe_batches)
        prediction, truth = sel.runner.collect(
            loader, endpoints, model, scaler, sel.codec.load_memory(base_path), device, 32)
        baseline_mae = sel.runner.metrics(prediction, truth)["mae"]
        # Candidate pool is byte-first, matching the existing greedy protocol.
        candidates = []
        node_count = base_memory.train_raw.shape[1]
        for node in range(node_count):
            path = temp / (f"single_{node}.npz")
            size = sel.codec.make_mixed_state(val_path, path, [node], pack_index=True)
            if size < base_bytes:
                candidates.append((base_bytes - size, node))
        candidates.sort(key=lambda x: (-x[0], x[1]))
        candidates = candidates[:cli.max_candidates]
        cache = {"base_bytes": base_bytes, "best_bytes": 1 << 62, "batches": batches}
        beam = [dict(nodes=[], bytes=base_bytes, val_delta=0.0, passed=False)]
        all_states = []
        for depth in range(1, cli.max_depth + 1):
            expanded = []
            for parent in beam:
                for _, node in candidates:
                    if node in parent["nodes"]:
                        continue
                    nodes = sorted(parent["nodes"] + [node])
                    row = evaluate_state(nodes, val_path, loader, endpoints, scaler,
                                         model, device, sel.codec, base_memory,
                                         base_path, baseline_mae, temp, cache)
                    if row is not None:
                        expanded.append(row)
            feasible = [r for r in expanded if r["passed"]]
            if feasible:
                cache["best_bytes"] = min(r["bytes"] for r in feasible)
            # Keep lower bytes first, then lower error; preserve feasible states.
            expanded.sort(key=lambda r: (not r["passed"], r["bytes"], r["val_delta"]))
            beam = expanded[:cli.beam_width]
            all_states.extend([{k: v for k, v in r.items() if k != "path"} for r in expanded])
            print(f"depth={depth} expanded={len(expanded)} kept={len(beam)} "
                  f"feasible={len(feasible)}", flush=True)
            if not beam:
                break
        feasible = [r for r in all_states if r["passed"]]
        best = min(feasible, key=lambda r: (r["bytes"], r["val_delta"])) if feasible else None
        result = dict(dataset=cli.dataset, seed=cli.seed,
                      scope="Validation only; no Test labels used",
                      search="beam", beam_width=cli.beam_width,
                      max_depth=cli.max_depth, max_candidates=cli.max_candidates,
                      gate=cli.gate, topk_gate=cli.topk_gate,
                      exact_gate=cli.exact_gate,
                      strong_lossless_bytes=cli.strong_lossless_bytes,
                      retrieval_k=int(base_memory.k), base_bytes=int(base_bytes),
                      baseline_val_mae=float(baseline_mae),
                      candidate_nodes=[n for _, n in candidates], best=best,
                      states=all_states, test_used=False)
        cli.output.parent.mkdir(parents=True, exist_ok=True)
        cli.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in result.items() if k not in ("states",)}, indent=2))
        print(f"Report: {cli.output}")


if __name__ == "__main__":
    main()
