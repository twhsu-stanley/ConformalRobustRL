"""Supervised Q regression using a transformer-based model."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments, set_seed

from q_transformer.data import format_prompt, split_samples
from q_transformer.model import QValueTransformer, tokenize_prompts


class _QTrainer(Trainer):
    """Keep epoch and final checkpoints compatible with the warm-start predictor."""

    def _save(self, output_dir=None, state_dict=None):
        training = {
            "epoch": self.state.epoch, "global_step": self.state.global_step,
            "seed": self.args.seed, "learning_rate": self.args.learning_rate,
        }
        if self.args.load_best_model_at_end:
            training.update({
                "best_global_step": getattr(self.state, "best_global_step", None),
                "best_metric": self.state.best_metric,
                "metric_for_best_model": self.args.metric_for_best_model,
            })
        self.model.save_checkpoint(
            output_dir or self.args.output_dir, self.processing_class,
            self.processing_class.model_max_length, training,
        )

    def _load_best_model(self):
        path = Path(self.state.best_model_checkpoint) / "model.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(checkpoint["state_dict"])


def _q_error_summaries(logits, labels):
    """Gather three errors per Q-table instead of every predicted entry."""
    errors = (logits.float() - labels.float()).flatten(start_dim=1)
    return torch.stack((errors.square().mean(1), errors.abs().mean(1), errors.abs().amax(1)), dim=1)


def _q_metrics(prediction):
    errors = prediction.predictions
    return {"rmse": float(np.sqrt(errors[:, 0].mean())), "mae": float(errors[:, 1].mean()),
            "mean_max_error": float(errors[:, 2].mean())}


def _optimizer_groups(model, trainer, head_learning_rate, position_learning_rate):
    """Use module-specific rates while preserving the trainer's weight-decay exclusions."""
    decay_parameters = trainer.get_decay_parameter_names(model)
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        rate = trainer.args.learning_rate
        if name.startswith("reg_head.") and head_learning_rate is not None:
            rate = head_learning_rate
        elif name == "encoder.embeddings.position_embeddings.weight":
            rate = rate if position_learning_rate is None else position_learning_rate
        decay = trainer.args.weight_decay if name in decay_parameters else 0.0
        group = groups.setdefault((rate, decay), {"params": [], "lr": rate, "weight_decay": decay})
        group["params"].append(parameter)
    return list(groups.values())


def train_q_transformer(
    dataset_path, checkpoint_path, *, model_name="bert-base-uncased",
    epochs=100, batch_size=4, learning_rate=1e-5, validation_fraction=0.1, seed=42,
    max_length=512, device=None,
    training_options=None, head_learning_rate=None, position_learning_rate=None,
    split_by_map=False, prepare_encoder=None, padding="max_length", callbacks=None,
):
    """Fine-tune the encoder and regression head using Huber loss on numeric targets.

    model_name can also be a local pretrained encoder directory. R and C are
    supplied only in the input; the regression target is the Q-table alone.
    CUDA training uses FP16; unavailable CUDA falls back to CPU with FP32.
    training_options overrides native TrainingArguments. Other optional settings
    enable held-out maps, encoder preparation, and module-specific learning rates.
    The defaults preserve the original training behavior.
    """
    for name, value in (("epochs", epochs), ("batch_size", batch_size), ("max_length", max_length)):
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite.")
    for name, rate in (
        ("head_learning_rate", head_learning_rate),
        ("position_learning_rate", position_learning_rate),
    ):
        if rate is not None and (not np.isfinite(rate) or rate <= 0):
            raise ValueError(f"{name} must be positive and finite, or None.")
    if padding not in ("max_length", "longest"):
        raise ValueError("padding must be 'max_length' or 'longest'.")
    set_seed(seed)
    with np.load(dataset_path, allow_pickle=False) as data:
        descriptions, map_indices, levels, radii = (
            data["descriptions"], data["map_index"], data["R"], data["C"],
        )
        if len(map_indices) != len(levels) or len(map_indices) != len(radii):
            raise ValueError("Each training record must contain a map index, R, and C.")
        prompts = [
            format_prompt(descriptions[index], R, C)
            for index, R, C in zip(map_indices, levels, radii)
        ]
        targets = np.asarray(data["Q"], dtype=np.float32)
    if (
        targets.ndim != 3 or targets.shape[0] != len(prompts)
        or min(targets.shape[1:]) < 1 or not np.all(np.isfinite(targets))
    ):
        raise ValueError("Q targets must be a finite (n_examples, n_states, n_actions) array.")
    n_states, n_actions = targets.shape[1:]
    if split_by_map:
        maps = np.unique(map_indices)
        train_maps, validation_maps = split_samples(len(maps), validation_fraction, seed)
        train_indices = np.flatnonzero(np.isin(map_indices, maps[train_maps]))
        validation_indices = np.flatnonzero(np.isin(map_indices, maps[validation_maps]))
    else:
        train_indices, validation_indices = split_samples(len(prompts), validation_fraction, seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name)
    if prepare_encoder is not None:
        prepare_encoder(encoder, max_length)
        tokenizer.model_max_length = max_length
    max_length = min(max_length, encoder.config.max_position_embeddings, tokenizer.model_max_length)
    tokenizer.model_max_length = int(max_length)
    if padding == "max_length":
        encoded = tokenize_prompts(tokenizer, prompts, max_length)
    else:
        encoded = tokenizer(
            prompts, padding=True, truncation=False, return_tensors="pt",
            return_token_type_ids=False, verbose=False,
            pad_to_multiple_of=8 if max_length % 8 == 0 else None,
        )
        if encoded["input_ids"].shape[1] > max_length:
            raise ValueError("A complete prompt exceeds the encoder's token limit.")
    dataset = [
        {"input_ids": encoded["input_ids"][i], "attention_mask": encoded["attention_mask"][i],
         "labels": torch.from_numpy(targets[i])}
        for i in range(len(prompts))
    ]
    use_cpu = not torch.cuda.is_available() or (
        device is not None and torch.device(device).type == "cpu"
    )
    if not use_cpu and device is not None and torch.device(device).type == "cuda":
        torch.cuda.set_device(device)
    options = dict(
        output_dir=str(checkpoint_path), save_strategy="epoch", num_train_epochs=epochs,
        per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
        logging_steps=50, learning_rate=learning_rate, weight_decay=0.01,
        fp16=not use_cpu, use_cpu=use_cpu, seed=seed,
        optim="adamw_torch", lr_scheduler_type="linear", report_to="none",
    )
    options.update(training_options or {})
    if options.get("bf16", False):
        options["fp16"] = False
    if options["use_cpu"]:
        options.update(fp16=False, bf16=False, tf32=None)
    arguments = TrainingArguments(**options)
    metrics = {}
    if arguments.eval_strategy != "no":
        metrics = {
            "compute_metrics": _q_metrics, "preprocess_logits_for_metrics": _q_error_summaries,
        }
    trainer = _QTrainer(
        model=QValueTransformer(encoder, n_states, n_actions), args=arguments,
        train_dataset=Subset(dataset, train_indices),
        eval_dataset=Subset(dataset, validation_indices), processing_class=tokenizer,
        callbacks=callbacks, **metrics,
    )
    if head_learning_rate is not None or position_learning_rate is not None:
        trainer.optimizer = torch.optim.AdamW(
            _optimizer_groups(trainer.model, trainer, head_learning_rate, position_learning_rate),
            betas=(arguments.adam_beta1, arguments.adam_beta2), eps=arguments.adam_epsilon,
        )
    if trainer.is_world_process_zero():
        precision = "BF16" if arguments.bf16 else "FP16" if arguments.fp16 else "FP32"
        batch = arguments.train_batch_size * arguments.world_size
        batch *= arguments.gradient_accumulation_steps
        print(f"Training on {arguments.device} ({precision}); effective batch size={batch}.")
        print(f"Training samples={len(train_indices)}; "
              f"validation samples={len(validation_indices)}.")
        print(f"Padded training length={encoded['input_ids'].shape[1]}; capacity={max_length}.")
    trainer.train()
    trainer.save_model(str(checkpoint_path))
    return trainer.state.log_history
