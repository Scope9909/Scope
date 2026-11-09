"""Build a fixed PEMS04 four-neighbor graph from Train observations only."""
from pathlib import Path

import numpy as np

from contrib3_neighbor_path.screen_neighbor_signal import graph_from_train


def main():
    root = Path(__file__).resolve().parents[1]
    with np.load(root / "datasets/PEMS04/PEMS04_12to12_index_scaler.npz") as split:
        cutoff = int(split["idx_val"].min())
        assert int(split["idx_train"].max()) + 1 == cutoff
    with np.load(root / "datasets/PEMS04/data.npz") as archive:
        flow = np.asarray(archive["data"][:cutoff, :, 0], dtype=np.float32)
    neighbors, _, correlation, random_correlation = graph_from_train(
        flow, cutoff, k=4, seed=41
    )
    assert neighbors.shape == (flow.shape[1], 4)
    assert np.all(neighbors != np.arange(flow.shape[1])[:, None])
    target = Path(__file__).with_name("pems04_train_only_neighbor_k4.npz")
    np.savez_compressed(
        target,
        neighbors=neighbors,
        train_observation_end_exclusive=cutoff,
        mean_delta_correlation=correlation,
        random_mean_delta_correlation=random_correlation,
    )
    print(
        f"{target}: cutoff={cutoff} nodes={flow.shape[1]} "
        f"mean_corr={correlation:.4f} random_corr={random_correlation:.4f}"
    )


if __name__ == "__main__":
    main()
