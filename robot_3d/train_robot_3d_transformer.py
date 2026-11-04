"""Train the 3-D Q-transformer using the shared trainer and saved dataset."""

import sys
from pathlib import Path

import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(
    *, epochs=100, batch_size=16, learning_rate=2e-5, seed=42,
    model_name="bert-base-uncased", max_length=1536,
    head_learning_rate=1e-4, position_learning_rate=5e-5,
    gradient_accumulation_steps=1, warmup_ratio=0.05, patience=5,
    resume_from_checkpoint=None, restart_lr_schedule=False,
):
    if not isinstance(max_length, (int, np.integer)) or max_length < 1:
        raise ValueError("max_length must be a positive integer.")
    output_dir = Path(__file__).resolve().parent / "saved_results/q_transformer"
    dataset_path = output_dir / "q_data_3d.npz"
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}. "
            "Run python -m robot_3d.generate_robot_3d_q_data first."
        )
    import torch
    from transformers import AutoTokenizer, EarlyStoppingCallback

    from q_transformer.data import format_prompt
    from q_transformer.train import train_q_transformer
    from robot_3d.transformer import extend_bert_positions, validate_prompt_length

    tokenizer = AutoTokenizer.from_pretrained(resume_from_checkpoint or model_name)
    with np.load(dataset_path, allow_pickle=False) as data:
        descriptions, indices, levels, radii = (
            data["descriptions"], data["map_index"], data["R"], data["C"],
        )
        for index, R, C in zip(indices, levels, radii):
            validate_prompt_length(tokenizer, format_prompt(descriptions[index], R, C), max_length)
        if any("0 = free space, 1 = obstacle." not in text for text in descriptions):
            raise ValueError(
                "The dataset must use the full-grid prompt. "
                "Run python -m robot_3d.generate_robot_3d_q_data again."
            )

    return train_q_transformer(
        dataset_path, output_dir / "checkpoint", model_name=model_name,
        epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, seed=seed,
        max_length=max_length, prepare_encoder=extend_bert_positions, padding="longest",
        head_learning_rate=head_learning_rate, position_learning_rate=position_learning_rate,
        split_by_map=True,
        resume_from_checkpoint=resume_from_checkpoint, restart_lr_schedule=restart_lr_schedule,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience,
                                         early_stopping_threshold=1e-5)],
        training_options={
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "warmup_ratio": warmup_ratio,
            "eval_strategy": "epoch", "save_total_limit": 5,
            "load_best_model_at_end": True, "metric_for_best_model": "rmse",
            "greater_is_better": False, "eval_accumulation_steps": 1,
            "logging_first_step": True,
            "bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            "tf32": True if torch.cuda.is_available()
                    and torch.cuda.get_device_capability()[0] >= 8 else None,
        },
    )


if __name__ == "__main__":
    checkpoint_dir = Path(__file__).resolve().parent / "saved_results/q_transformer/checkpoint"
    main(
        epochs=100, learning_rate=1e-5, head_learning_rate=5e-5, position_learning_rate=2.5e-5,
        resume_from_checkpoint=checkpoint_dir / "checkpoint-50640", restart_lr_schedule=True,
    )
