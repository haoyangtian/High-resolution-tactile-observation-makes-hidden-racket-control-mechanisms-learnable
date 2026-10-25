"""Small component demonstration using only synthetic inputs, not study results."""
import json
import torch
import numpy as np
from tactile_components.posture.pose_model import GNNSuite
from tactile_components.spatial.graph_topology import build_knn_adj
from tactile_components.temporal.temporal_attention import PersonSpecificTemporalAttention


def main():
    torch.manual_seed(17)  # Demonstration seed, not a manuscript experiment seed.
    coords = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.], [2., 1.]], dtype=np.float32)
    adjacency = build_knn_adj(coords, k=2)
    model = GNNSuite(n_nodes=len(coords), hidden=16, dropout=0.).eval()
    with torch.no_grad():
        prediction = model(torch.randn(2, 7, len(coords), 1), adjacency)
        attention = PersonSpecificTemporalAttention(1, 3, n_persons=2)(torch.tensor([0, 1]))
    print(json.dumps({
        "data": "synthetic_only",
        "pose_output_shape": list(prediction.shape),
        "finite_pose_output": bool(torch.isfinite(prediction).all()),
        "attention_sums": attention["weights"].sum(-1).tolist(),
        "manuscript_results_reproduced": False,
    }, indent=2))


if __name__ == "__main__":
    main()
