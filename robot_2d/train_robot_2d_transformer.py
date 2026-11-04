"""Train a Q-transformer using the previously saved 2-D dataset."""

import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parent / "saved_results/q_transformer"
    dataset_path = output_dir / "q_data_2d.npz"
    checkpoint_path = output_dir / "checkpoint"
    model_name = "bert-base-uncased"
    epochs = 100
    batch_size = 4
    learning_rate = 1e-5
    validation_fraction = 0.1
    seed = 42
    max_length = 512
    device = None
    training_options = None
    head_learning_rate = None
    position_learning_rate = None
    split_by_map = False
    padding = "max_length"
    callbacks = None
    resume_from_checkpoint = None
    restart_lr_schedule = False

    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}. "
            "Run python -m robot_2d.generate_robot_2d_q_data first."
        )
    from q_transformer.train import train_q_transformer

    history = train_q_transformer(
        dataset_path, checkpoint_path, model_name=model_name,
        epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
        validation_fraction=validation_fraction, seed=seed, max_length=max_length, device=device,
        training_options=training_options,
        head_learning_rate=head_learning_rate, position_learning_rate=position_learning_rate,
        split_by_map=split_by_map, padding=padding, callbacks=callbacks,
        resume_from_checkpoint=resume_from_checkpoint, restart_lr_schedule=restart_lr_schedule,
    )
