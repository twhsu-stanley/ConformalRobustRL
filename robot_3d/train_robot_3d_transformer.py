"""Train a Q-transformer using the previously saved 3-D dataset."""

import sys
from pathlib import Path

import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(
    *, epochs=100, batch_size=4, learning_rate=1e-5, seed=42,
    model_name="bert-base-uncased", max_length=512,
):
    if not isinstance(max_length, (int, np.integer)) or max_length < 1:
        raise ValueError("max_length must be a positive integer.")
    output_dir = Path(__file__).resolve().parent / "saved_results/q_transformer"
    dataset_path = output_dir / "q_data.npz"
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}. "
            "Run python -m robot_3d.generate_robot_3d_q_data first."
        )
    from transformers import AutoConfig, AutoTokenizer

    from q_transformer.data import format_prompt
    from q_transformer.train import train_q_transformer
    from robot_3d.transformer import validate_prompt_length

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    limit = min(max_length, config.max_position_embeddings, tokenizer.model_max_length)
    with np.load(dataset_path, allow_pickle=False) as data:
        descriptions, indices, levels, radii = (
            data["descriptions"], data["map_index"], data["R"], data["C"],
        )
        for index, R, C in zip(indices, levels, radii):
            prompt = format_prompt(descriptions[index], R, C)
            validate_prompt_length(tokenizer, prompt, limit)

    return train_q_transformer(
        dataset_path, output_dir / "checkpoint", model_name=model_name,
        epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, seed=seed,
        max_length=max_length,
    )


if __name__ == "__main__":
    main()
