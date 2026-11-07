"""Joint training of a fresh pretrained BERT encoder and a shared cell decoder."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, EarlyStoppingCallback
from transformers import Trainer, TrainingArguments, set_seed

from q_transformer.data import format_prompt, split_samples
from q_transformer.model import QValueTransformer, extend_bert_positions, tokenize_prompts


class _MapDataset(Dataset):
    """One map per item, with all its R/C labels and no duplicated BERT inputs."""

    def __init__(self, path, mdp_from_description, tokenizer, max_length, show_progress):
        with np.load(path, allow_pickle=False) as data:
            descriptions, map_indices = data["descriptions"], data["map_index"]
            levels, radii = data["R"], data["C"]
            targets = np.asarray(data["Q"], dtype=np.float32)
        if (
            targets.ndim != 3 or targets.shape[0] != len(map_indices)
            or len(levels) != len(map_indices) or len(radii) != len(map_indices)
            or not np.all(np.isfinite(targets))
        ):
            raise ValueError("Every record must have a map index, R, C, and a finite Q-table.")
        if not np.issubdtype(map_indices.dtype, np.integer) or np.any(
            (map_indices < 0) | (map_indices >= len(descriptions))
        ):
            raise ValueError("Map indices must refer to saved descriptions.")
        if np.any(~np.isfinite(levels)) or np.any((levels < 0) | (levels > 1)):
            raise ValueError("Every R must be a probability in [0, 1].")
        self.maps = np.unique(map_indices)
        if len(self.maps) < 2:
            raise ValueError("Training and validation require at least two distinct maps.")
        self.n_states, self.n_actions = targets.shape[1:]
        self.targets = torch.from_numpy(targets)
        self.levels = torch.as_tensor(levels, dtype=torch.float32)
        self.rows, self.features, self.radius_indices, self.terminal = [], [], [], []
        prompts = [format_prompt(str(descriptions[index])) for index in self.maps]
        self.encoded = tokenize_prompts(tokenizer, prompts, max_length, self.n_states)
        conditions = None
        for map_index in tqdm(self.maps, desc="Preparing map features", disable=not show_progress):
            rows = np.flatnonzero(map_indices == map_index)
            current = list(zip(levels[rows].tolist(), radii[rows].tolist()))
            if len(set(current)) != len(current) or (
                conditions is not None and current != conditions
            ):
                raise ValueError("All maps must contain the same ordered, distinct R/C conditions.")
            conditions = current
            description = str(descriptions[map_index])
            mdp = mdp_from_description(description)
            if mdp.q_transformer_context() != description:
                raise ValueError("The reconstructed MDP must reproduce its saved description.")
            if (mdp.n_state, mdp.action_space.n) != (self.n_states, self.n_actions):
                raise ValueError("Every map must match the Q-label dimensions.")
            if np.any(targets[rows][:, mdp.terminal_mask] != 0):
                raise ValueError("Saved terminal Q-values must be exactly zero.")
            unique_C, inverse = np.unique(radii[rows], return_inverse=True)
            features = mdp.q_transformer_features(np.zeros(len(unique_C)), unique_C)
            self.rows.append(torch.from_numpy(rows))
            self.features.append(torch.from_numpy(features))
            self.radius_indices.append(torch.from_numpy(inverse))
            self.terminal.append(torch.from_numpy(mdp.terminal_mask.copy()))
        self.numeric_width = self.features[0].shape[-1]
        self.R_column = 2 * len(mdp.grid_shape)
        self.n_conditions = len(conditions)

    def __len__(self):
        return len(self.maps)

    def __getitem__(self, index):
        rows = self.rows[index]
        features = self.features[index][self.radius_indices[index]]
        features[..., self.R_column] = self.levels[rows, None]
        return {
            **{key: value[index] for key, value in self.encoded.items()},
            "features": features, "terminal_mask": self.terminal[index],
            "labels": self.targets[rows],
        }


class _QTrainer(Trainer):
    """Save the cell decoder in the same format used by the Algorithm 1 predictor."""

    def _save(self, output_dir=None, state_dict=None):
        training = {
            "epoch": self.state.epoch, "global_step": self.state.global_step,
            "seed": self.args.seed, "learning_rate": self.args.learning_rate,
            "best_metric": self.state.best_metric,
            "metric_for_best_model": self.args.metric_for_best_model,
        }
        self.model.save_checkpoint(
            output_dir or self.args.output_dir, self.processing_class,
            self.processing_class.model_max_length, training,
        )

    def _load_best_model(self):
        path = Path(self.state.best_model_checkpoint) / "model.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(checkpoint["state_dict"])


def _q_error_summaries(logits, labels):
    """Gather sufficient statistics per map, preserving nonterminal entry counts."""
    targets, terminal = labels
    errors = logits.float() - targets.float()
    active = (~terminal[:, None, :, None]).expand_as(errors)
    absolute = errors.abs()
    maximum = absolute.amax(dim=(-1, -2))
    return torch.stack((
        (errors.square() * active).sum(dim=(1, 2, 3)),
        (absolute * active).sum(dim=(1, 2, 3)), active.sum(dim=(1, 2, 3)),
        torch.full_like(maximum[:, 0], errors[0].numel()), maximum.sum(1),
        torch.full_like(maximum[:, 0], errors.shape[1]), maximum.amax(1),
    ), dim=1)


def _q_metrics(prediction):
    errors = prediction.predictions.astype(np.float64)
    total = errors.sum(axis=0)
    return {
        "rmse": float(np.sqrt(total[0] / total[2])),
        "whole_rmse": float(np.sqrt(total[0] / total[3])),
        "mae": float(total[1] / total[2]), "mean_max_error": float(total[4] / total[5]),
        "max_error": float(errors[:, 6].max()),
    }


def _optimizer_groups(model, trainer, head_learning_rate, position_learning_rate):
    """Apply separate peak rates, sharing one cosine schedule and decay exclusions."""
    decay_parameters = trainer.get_decay_parameter_names(model)
    groups = {}
    for name, parameter in model.named_parameters():
        rate = trainer.args.learning_rate
        if not name.startswith("encoder."):
            rate = head_learning_rate
        elif name == "encoder.embeddings.position_embeddings.weight":
            rate = rate if position_learning_rate is None else position_learning_rate
        decay = trainer.args.weight_decay if name in decay_parameters else 0.0
        group = groups.setdefault((rate, decay), {"params": [], "lr": rate, "weight_decay": decay})
        group["params"].append(parameter)
    return list(groups.values())


def train_q_transformer(
    dataset_path, checkpoint_path, mdp_from_description, *, model_name="bert-base-uncased",
    epochs=100, batch_size=4, learning_rate=1e-5, head_learning_rate=3e-4,
    position_learning_rate=None, validation_fraction=0.1, seed=42, max_length=512,
    width=128, device=None, gradient_accumulation_steps=1, warmup_ratio=0.05, patience=10,
    training_options=None,
):
    """Start fresh from pretrained BERT; one batch item contains a map's R/C cases.

    Both encoder and decoder learn jointly. Existing Q-transformer checkpoints are
    never loaded. The map-level split is seeded; targets and map generation remain
    unchanged. CUDA uses BF16 when supported, otherwise FP16; CPU uses FP32.
    training_options overrides native TrainingArguments without duplicating example code.
    """
    for name, value in (
        ("epochs", epochs), ("batch_size", batch_size), ("max_length", max_length),
        ("width", width), ("gradient_accumulation_steps", gradient_accumulation_steps),
    ):
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    for name, rate in (
        ("learning_rate", learning_rate), ("head_learning_rate", head_learning_rate),
        ("position_learning_rate", position_learning_rate),
    ):
        if rate is not None and (not np.isfinite(rate) or rate <= 0):
            raise ValueError(f"{name} must be positive and finite.")
    if not 0 <= warmup_ratio < 1 or not isinstance(patience, int) or patience < 1:
        raise ValueError("Use warmup_ratio in [0, 1) and a positive early-stopping patience.")
    if not Path(dataset_path).is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}.")
    set_seed(seed)
    use_cpu = not torch.cuda.is_available() or (
        device is not None and torch.device(device).type == "cpu"
    )
    if not use_cpu and device is not None:
        torch.cuda.set_device(device)
    bf16 = not use_cpu and torch.cuda.is_bf16_supported()
    options = dict(
        output_dir=str(checkpoint_path), save_strategy="epoch", eval_strategy="epoch",
        num_train_epochs=epochs, per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size, learning_rate=learning_rate, weight_decay=0.01,
        gradient_accumulation_steps=gradient_accumulation_steps, warmup_ratio=warmup_ratio,
        max_grad_norm=1.0, optim="adamw_torch", lr_scheduler_type="cosine",
        use_cpu=use_cpu, bf16=bf16, fp16=not use_cpu and not bf16,
        tf32=True if not use_cpu and torch.cuda.get_device_capability()[0] >= 8 else None,
        load_best_model_at_end=True, metric_for_best_model="rmse", greater_is_better=False,
        save_total_limit=5, eval_accumulation_steps=1, logging_steps=50,
        logging_first_step=True, seed=seed, report_to="none",
        label_names=["labels", "terminal_mask"],
    )
    options.update(training_options or {})
    if options.get("bf16"):
        options["fp16"] = False
    if options["use_cpu"]:
        options.update(fp16=False, bf16=False, tf32=None)
    arguments = TrainingArguments(**options)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
    if max_length > encoder.config.max_position_embeddings:
        extend_bert_positions(encoder, max_length)
    tokenizer.model_max_length = max_length
    dataset = _MapDataset(
        dataset_path, mdp_from_description, tokenizer, max_length,
        show_progress=arguments.process_index == 0,
    )
    training, validation = split_samples(len(dataset), validation_fraction, seed)
    model = QValueTransformer(
        encoder, dataset.n_states, dataset.n_actions, dataset.numeric_width, width,
    )
    trainer = _QTrainer(
        model=model, args=arguments, train_dataset=Subset(dataset, training),
        eval_dataset=Subset(dataset, validation), processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=patience, early_stopping_threshold=1e-5,
        )], compute_metrics=_q_metrics, preprocess_logits_for_metrics=_q_error_summaries,
    )
    trainer.optimizer = torch.optim.AdamW(
        _optimizer_groups(model, trainer, head_learning_rate, position_learning_rate),
        betas=(arguments.adam_beta1, arguments.adam_beta2), eps=arguments.adam_epsilon,
    )
    if trainer.is_world_process_zero():
        precision = "BF16" if arguments.bf16 else "FP16" if arguments.fp16 else "FP32"
        batch = arguments.train_batch_size * arguments.world_size
        batch *= arguments.gradient_accumulation_steps
        print(f"Training on {arguments.device}; configured precision={precision}.")
        print(f"Effective batch={batch} maps, each with {dataset.n_conditions} R/C cases.")
        print(f"Training={len(training)} maps; validation={len(validation)} maps; "
              f"Q-table targets={len(dataset.targets)}.")
        print(f"Token length={dataset.encoded['input_ids'].shape[1]}; capacity={max_length}.")
        print("Fresh BERT initialization; trainable cell decoder; warmup then cosine decay.")
    trainer.train()
    trainer.save_model(str(checkpoint_path))
    return trainer.state.log_history
