"""Train a text-only Q-transformer from the saved 3-D Q-table data."""

import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


if __name__ == "__main__":
    from q_transformer.train import train_q_transformer
    from robot_3d.robot_3d_motion_mdp import Robot3DMotionMDP
    import torch

    output_dir = Path(__file__).resolve().parent / "saved_results/q_transformer"
    dataset_path = output_dir / "q_data_3d.npz"
    checkpoint_path = output_dir / "checkpoint"
    model_name = "bert-base-uncased"
    epochs = 100
    batch_size = 4  # Map/R/C prompts per GPU.
    learning_rate = 1e-5
    head_learning_rate = 3e-4
    position_learning_rate = 2e-5
    validation_fraction = 0.1
    seed = 42
    max_length = 2048
    width = 128
    device = None
    gradient_accumulation_steps = 1
    warmup_ratio = 0.05
    patience = 10

    try:
        train_q_transformer(
            dataset_path, checkpoint_path, Robot3DMotionMDP, model_name=model_name,
            epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
            head_learning_rate=head_learning_rate, position_learning_rate=position_learning_rate,
            validation_fraction=validation_fraction, seed=seed, max_length=max_length,
            width=width, device=device, gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=warmup_ratio, patience=patience,
        )
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
