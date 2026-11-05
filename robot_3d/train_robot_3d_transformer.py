"""Train the 3-D Q-transformer using the shared trainer and saved dataset."""

import sys
from pathlib import Path

import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parent / "saved_results/q_transformer"
    dataset_path = output_dir / "q_data_3d.npz"
    checkpoint_path = output_dir / "checkpoint"
    model_name = "bert-base-uncased"
    epochs = 100
    batch_size = 16
    learning_rate = 1e-5
    validation_fraction = 0.1
    seed = 42
    max_length = 1536
    device = None
    head_learning_rate = 5e-5
    position_learning_rate = 2.5e-5
    split_by_map = True
    padding = "longest"
    gradient_accumulation_steps = 1
    warmup_ratio = 0.05
    patience = 5

    # For resuming from a checkpoint, set resume_from_checkpoint to the path of the checkpoint file and restart_lr_schedule to True.
    #resume_from_checkpoint = checkpoint_path / "checkpoint-50640"
    #restart_lr_schedule = True
    #
    # For a fresh run, use resume_from_checkpoint=None and restart_lr_schedule=False. 
    resume_from_checkpoint = None
    restart_lr_schedule = False

    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}. "
            "Run python -m robot_3d.generate_robot_3d_q_data first."
        )
    import torch
    from transformers import EarlyStoppingCallback

    from q_transformer.model import extend_bert_positions
    from q_transformer.train import train_q_transformer

    with np.load(dataset_path, allow_pickle=False) as data:
        if any("0 = free space, 1 = obstacle." not in text for text in data["descriptions"]):
            raise ValueError(
                "The dataset must use the full-grid prompt. "
                "Run python -m robot_3d.generate_robot_3d_q_data again."
            )

    training_options = {
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
    callbacks = [EarlyStoppingCallback(early_stopping_patience=patience,
                                       early_stopping_threshold=1e-5)]

    history = train_q_transformer(
        dataset_path, checkpoint_path, model_name=model_name,
        epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
        validation_fraction=validation_fraction, seed=seed, max_length=max_length, device=device,
        prepare_encoder=extend_bert_positions, padding=padding, truncation=False,
        head_learning_rate=head_learning_rate, position_learning_rate=position_learning_rate,
        split_by_map=split_by_map, callbacks=callbacks, training_options=training_options,
        resume_from_checkpoint=resume_from_checkpoint, restart_lr_schedule=restart_lr_schedule,
    )
