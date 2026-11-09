"""Audit stability metric dimensions without reading Test data or labels."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ONLINE = HERE / "packages" / "online"


def metric_counts(base_retrieval, coded_retrieval, base_avail, coded_avail,
                  base_topk, coded_topk):
    if base_retrieval.shape != coded_retrieval.shape:
        raise ValueError("retrieval shapes differ")
    if base_avail.shape != coded_avail.shape or base_avail.shape != base_retrieval.shape:
        raise ValueError("availability denominator must match retrieval elements")
    if base_topk.shape != coded_topk.shape or base_topk.ndim != 3:
        raise ValueError("top-k arrays must be (query,node,k)")
    base_sets = np.sort(base_topk, axis=2)
    coded_sets = np.sort(coded_topk, axis=2)
    return dict(
        exact_flips=int(np.count_nonzero(base_avail != coded_avail)),
        exact_count=int(base_avail.size),
        topk_flips=int(np.any(base_sets != coded_sets, axis=2).sum()),
        topk_count=int(base_sets.shape[0] * base_sets.shape[1]),
    )


def main():
    # Hand-checkable case: ordering-only change is not a set flip; one actual
    # set change and one availability element change are counted.
    shape = (2, 3, 2, 1)
    base_value = np.zeros(shape, dtype=np.float32)
    coded_value = base_value.copy()
    base_avail = np.zeros(shape, dtype=np.uint8)
    coded_avail = base_avail.copy(); coded_avail[1, 2, 1, 0] = 1
    base_topk = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
    coded_topk = np.array([[[1, 0], [2, 8]], [[4, 5], [6, 7]]])
    artificial = metric_counts(base_value, coded_value, base_avail, coded_avail,
                               base_topk, coded_topk)
    assert artificial == dict(exact_flips=1, exact_count=12,
                              topk_flips=1, topk_count=4)

    frozen = []
    for dataset in ("pems03", "pems04", "pems07"):
        path = ONLINE / f"{dataset}_s1_beam_threebudget.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        best = data["best"]
        stability = best["stability"]
        exact_flips = int(stability["exact_hit_flips"])
        exact_rate = float(stability["exact_hit_flip_rate"])
        topk_count = int(stability["topk_set_count"])
        topk_flips = int(stability["topk_set_flips"])
        # The frozen task is 12-step forecasting.  A top-k set is counted per
        # query/node, while availability is counted per query/horizon/node.
        exact_count = topk_count * 12
        assert np.isclose(topk_flips / topk_count,
                          float(stability["topk_set_flip_rate"]), rtol=0, atol=1e-15)
        if exact_flips:
            assert np.isclose(exact_flips / exact_count, exact_rate, rtol=0, atol=1e-12)
        frozen.append(dict(dataset=data["dataset"], nodes=best["nodes"],
                           retrieval_k=data["retrieval_k"],
                           topk_set_flips=topk_flips, topk_set_count=topk_count,
                           topk_set_flip_rate=topk_flips / topk_count,
                           exact_flips=exact_flips, exact_element_count=exact_count,
                           exact_flip_rate=exact_rate,
                           source=str(path.resolve())))
    report = dict(scope="Validation stability denominator audit",
                  test_used=False, labels_used=False,
                  conclusion=("availability and retrieval arrays share (query,horizon,node,1); "
                              "the old retrieved_count variable is therefore the exact element denominator"),
                  artificial_case=artificial, frozen_records=frozen,
                  pems08_final="exact fallback: both flip rates are zero by identity")
    target = ONLINE / "stability_denominator_audit.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
