"""Evaluate saved Q predictions and paired value iteration on held-out 2-D maps."""

from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from q_transformer.evaluate import evaluate_q_transformer
from robot_2d.utils_2d import mdp_from_description


if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parent / "saved_results/q_transformer"
    evaluate_q_transformer(
        output_dir / "q_data_2d.npz", output_dir / "checkpoint", mdp_from_description,
        n_maps=10, seed=42, validation_fraction=0.1, tolerance=1e-8,
        n_workers=2, cpu_threads=4,
    )
