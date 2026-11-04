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
    split_by_map = True
    padding = "max_length"
    callbacks = None
    gradient_accumulation_steps = 1
    warmup_ratio = 0.05
    patience = 5
    resume_from_checkpoint = None
    restart_lr_schedule = False

    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}. "
            "Run python -m robot_2d.generate_robot_2d_q_data first."
        )
    import torch
    from transformers import EarlyStoppingCallback

    from q_transformer.train import train_q_transformer

    options = {
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "warmup_ratio": warmup_ratio,
        "eval_strategy": "epoch", "save_total_limit": 5,
        "load_best_model_at_end": True, "metric_for_best_model": "rmse",
        "greater_is_better": False, "eval_accumulation_steps": 1,
        "logging_first_step": True,
        "bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "tf32": True if torch.cuda.is_available()
                and torch.cuda.get_device_capability()[0] >= 8 else None,
    }
    options.update(training_options or {})
    if callbacks is None:
        callbacks = [EarlyStoppingCallback(early_stopping_patience=patience,
                                           early_stopping_threshold=1e-5)]

    history = train_q_transformer(
        dataset_path, checkpoint_path, model_name=model_name,
        epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
        validation_fraction=validation_fraction, seed=seed, max_length=max_length, device=device,
        training_options=options,
        head_learning_rate=head_learning_rate, position_learning_rate=position_learning_rate,
        split_by_map=split_by_map, padding=padding, callbacks=callbacks,
        resume_from_checkpoint=resume_from_checkpoint, restart_lr_schedule=restart_lr_schedule,
    )
