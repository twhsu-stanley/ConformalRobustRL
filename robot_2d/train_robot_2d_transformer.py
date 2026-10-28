"""Train a Q-transformer using the previously saved 2-D dataset."""

import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(
    *, epochs=100, batch_size=4, learning_rate=1e-5, seed=42,
    model_name="bert-base-uncased",
):
    output_dir = Path(__file__).resolve().parent / "saved_results/q_transformer"
    dataset_path = output_dir / "q_data.npz"
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}. "
            "Run python -m robot_2d.generate_robot_2d_q_data first."
        )
    from q_transformer.train import train_q_transformer

    return train_q_transformer(
        dataset_path, output_dir / "checkpoint", model_name=model_name,
        epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, seed=seed,
    )


if __name__ == "__main__":
    main()
