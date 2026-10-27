"""Supervised Q regression with map-disjoint validation and one best checkpoint."""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer

from q_transformer.data import format_prompt, split_by_map
from q_transformer.model import QValueTransformer, tokenize_prompts


def train_q_transformer(
    dataset_path, checkpoint_path, *, model_name="google-bert/bert-base-uncased",
    epochs=30, batch_size=8, learning_rate=2e-5, validation_fraction=0.2, seed=7,
    max_length=512, device=None,
):
    """Fine-tune the encoder and regression head using Huber loss on numeric targets.

    model_name can also be a local pretrained encoder directory. R and C are
    supplied only in the input; the regression target is the Q-table alone.
    """
    for name, value in (("epochs", epochs), ("batch_size", batch_size), ("max_length", max_length)):
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite.")
    torch.manual_seed(seed)
    with np.load(dataset_path, allow_pickle=False) as data:
        map_indices = data["map_index"]
        if len(map_indices) != len(data["R"]) or len(map_indices) != len(data["C"]):
            raise ValueError("Each training record must contain a map index, R, and C.")
        prompts = [
            format_prompt(data["descriptions"][index], R, C)
            for index, R, C in zip(map_indices, data["R"], data["C"])
        ]
        targets = np.asarray(data["Q"], dtype=np.float32)
    if (
        targets.ndim != 3 or targets.shape[0] != len(prompts)
        or min(targets.shape[1:]) < 1 or not np.all(np.isfinite(targets))
    ):
        raise ValueError("Q targets must be a finite (n_examples, n_states, n_actions) array.")
    n_states, n_actions = targets.shape[1:]
    train_indices, validation_indices = split_by_map(map_indices, validation_fraction, seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name)
    max_length = min(max_length, encoder.config.max_position_embeddings, tokenizer.model_max_length)
    encoded = tokenize_prompts(tokenizer, prompts, max_length)
    dataset = TensorDataset(
        encoded["input_ids"], encoded["attention_mask"], torch.from_numpy(targets),
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, train_indices), batch_size=batch_size,
        shuffle=True, generator=generator,
    )
    validation_loader = DataLoader(
        torch.utils.data.Subset(dataset, validation_indices), batch_size=batch_size,
    )
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = QValueTransformer(encoder, n_states, n_actions).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_function = nn.SmoothL1Loss()
    history = []
    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for input_ids, attention_mask, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            predictions = model(input_ids.to(device), attention_mask.to(device))
            loss = loss_function(predictions, labels.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError("Transformer training produced a nonfinite loss.")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(labels)
        model.eval()
        validation_loss, absolute_error, max_error = 0.0, 0.0, 0.0
        with torch.inference_mode():
            for input_ids, attention_mask, labels in validation_loader:
                predictions = model(input_ids.to(device), attention_mask.to(device))
                labels = labels.to(device)
                validation_loss += loss_function(predictions, labels).item() * len(labels)
                errors = (predictions - labels).abs()
                absolute_error += errors.sum().item()
                max_error = max(max_error, errors.max().item())
        validation_loss /= len(validation_indices)
        if not np.isfinite(validation_loss):
            raise RuntimeError("Transformer validation produced a nonfinite loss.")
        metrics = {
            "epoch": epoch, "training_loss": total_loss / len(train_indices),
            "validation_loss": validation_loss,
            "validation_mae": absolute_error / (len(validation_indices) * targets[0].size),
            "validation_max_error": max_error,
        }
        history.append(metrics)
        if validation_loss < best_loss:
            best_loss = validation_loss
            training = {
                **metrics, "seed": int(seed), "learning_rate": float(learning_rate),
                "training_maps": np.unique(map_indices[train_indices]).tolist(),
                "validation_maps": np.unique(map_indices[validation_indices]).tolist(),
            }
            model.save_checkpoint(checkpoint_path, tokenizer, int(max_length), training)
        print(
            f"Epoch {epoch}/{epochs}: train={metrics['training_loss']:.6g}, "
            f"validation={validation_loss:.6g}, max error={max_error:.6g}",
        )
    return history
