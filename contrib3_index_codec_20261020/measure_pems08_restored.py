"""Measure the validation-selected PEMS08 Teacher cache as real .fcz files."""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import container


BENCH = HERE / "packages" / "bench"
rule = json.loads((HERE / "frozen_rule_v1.json").read_text(encoding="utf-8"))
plan = dict(rule["selection"]["PEMS08"]["plan_keep"])
for field in rule["backoff_order"][:3]:
    ladder = rule["ladders"][field]
    rung = next(i for i, spec in enumerate(ladder) if spec == plan[field])
    plan[field] = ladder[rung + 1]

results = {}
for split in ("val", "test"):
    raw_path = BENCH / f"PEMS08__raw__{split}.fcz"
    raw_blobs, raw_header = container.read_package(raw_path)
    fields = container.decode_fields(raw_blobs, raw_header["layout"])
    blobs, layout = container.encode_fields(fields, plan)
    out = BENCH / f"PEMS08__teacher_restored__{split}.fcz"
    container.write_package(out, blobs, dict(
        kind="retrieval_cache", dataset="PEMS08", split=split,
        method="teacher_restored", plan=plan, layout=layout,
        selection="validation_only_restore_3"))
    check_blobs, check_header = container.read_package(out)
    decoded = container.decode_fields(check_blobs, check_header["layout"])
    if any(decoded[name].shape != fields[name].shape for name in fields):
        raise ValueError(f"Shape mismatch after decoding {out}")
    baseline, restored = raw_path.stat().st_size, out.stat().st_size
    results[split] = dict(raw_bytes=baseline, restored_bytes=restored,
                          saving_fraction=1 - restored / baseline,
                          path=str(out))

print(json.dumps(dict(plan=plan, results=results), indent=2))
