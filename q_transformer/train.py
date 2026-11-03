"""Supervised Q regression using a transformer-based model."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset
from transformers import AutoConfig, AutoModel, AutoTokenizer, Trainer, TrainingArguments, set_seed
from transformers import get_linear_schedule_with_warmup
from transformers.trainer_callback import TrainerState

from q_transformer.data import format_prompt, split_samples
from q_transformer.model import QValueTransformer, tokenize_prompts


class _QTrainer(Trainer):
    """Keep epoch and final checkpoints compatible with the warm-start predictor."""

    def __init__(self, *args, restart_lr_schedule=False, **kwargs):
        self.restart_lr_schedule = restart_lr_schedule
        self._resume_path = None
        self._resume_state = None
        super().__init__(*args, **kwargs)

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

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        self._resume_path = Path(resume_from_checkpoint).resolve()
        self._resume_state = TrainerState.load_from_json(self._resume_path / "trainer_state.json")
        checkpoint = torch.load(
            self._resume_path / "model.pt", map_location="cpu", weights_only=True,
        )
        model = self.model if model is None else model
        if tuple(checkpoint["Q_shape"]) != (model.n_states, model.n_actions):
            raise ValueError("The checkpoint Q-table shape does not match the dataset.")
        if checkpoint["max_length"] != self.processing_class.model_max_length:
            raise ValueError("Resume with the checkpoint's original token limit.")
        if checkpoint["training"]["seed"] != self.args.seed:
            raise ValueError("Resume with the original seed to preserve the validation split.")
        if self._resume_state.train_batch_size != self.args.train_batch_size:
            raise ValueError("Resume with the original per-process training batch size.")
        rng_name = "rng_state.pth" if self.args.world_size == 1 else (
            f"rng_state_{self.args.process_index}.pth"
        )
        rng_files = list(self._resume_path.glob("rng_state*.pth"))
        if len(rng_files) != self.args.world_size or not (self._resume_path / rng_name).is_file():
            raise ValueError("Resume with the original number of training processes.")
        model.load_state_dict(checkpoint["state_dict"])

    def create_scheduler(self, num_training_steps, optimizer=None):
        if self._resume_path is None:
            return super().create_scheduler(num_training_steps, optimizer)
        if num_training_steps <= self._resume_state.global_step:
            raise ValueError("The total training budget must exceed the saved global step.")
        if self.lr_scheduler is None:
            saved = torch.load(self._resume_path / "scheduler.pt", weights_only=True)
            if self.restart_lr_schedule:
                start_step = self._resume_state.global_step
                steps = num_training_steps - start_step
                warmup_steps = self.args.get_warmup_steps(steps)
            else:
                start_step = saved.get("start_step", 0)
                steps = saved.get("num_training_steps", self._resume_state.max_steps)
                warmup_steps = saved.get("num_warmup_steps", self.args.get_warmup_steps(steps))
                if num_training_steps != start_step + steps:
                    raise ValueError(
                        "Set restart_lr_schedule=True when extending the training budget."
                    )
            optimizer = self.optimizer if optimizer is None else optimizer
            self.lr_scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, steps)
            self.lr_scheduler.start_step = start_step
            self.lr_scheduler.num_training_steps = steps
            self.lr_scheduler.num_warmup_steps = warmup_steps
            self._created_lr_scheduler = True
        return self.lr_scheduler

    def _load_optimizer_and_scheduler(self, checkpoint):
        if checkpoint is None or not self.restart_lr_schedule:
            return super()._load_optimizer_and_scheduler(checkpoint)
        schedule = self.lr_scheduler.state_dict()
        super()._load_optimizer_and_scheduler(checkpoint)
        self.lr_scheduler.load_state_dict(schedule)
        for group, peak, rate in zip(
            self.optimizer.param_groups, schedule["base_lrs"], schedule["_last_lr"],
        ):
            group.update(initial_lr=peak, lr=rate)

    def _load_callback_state(self):
        super()._load_callback_state()
        if self.state.best_model_checkpoint is not None:
            name = Path(self.state.best_model_checkpoint.replace("\\", "/")).name
            local_best = self._resume_path.parent / name
            if (local_best / "model.pt").is_file():
                self.state.best_model_checkpoint = str(local_best)
            elif self.args.load_best_model_at_end:
                if not (Path(self.state.best_model_checkpoint) / "model.pt").is_file():
                    raise FileNotFoundError("The saved best-model checkpoint is missing.")
        if self.restart_lr_schedule:
            self.control.should_training_stop = False
            for callback in self.callback_handler.callbacks:
                if hasattr(callback, "early_stopping_patience_counter"):
                    callback.early_stopping_patience_counter = 0


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
    resume_from_checkpoint=None, restart_lr_schedule=False,
):
    """Fine-tune the encoder and regression head using Huber loss on numeric targets.

    model_name can also be a local pretrained encoder directory. R and C are
    supplied only in the input; the regression target is the Q-table alone.
    CUDA training uses FP16; unavailable CUDA falls back to CPU with FP32.
    training_options overrides native TrainingArguments. Other optional settings
    enable held-out maps, encoder preparation, and module-specific learning rates.
    The defaults preserve the original training behavior.
    epochs is the total limit, including epochs already completed in a checkpoint.
    Restarting the linear schedule applies the supplied rates to the remaining steps.
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
    if restart_lr_schedule and resume_from_checkpoint is None:
        raise ValueError("Restarting the learning-rate schedule requires a checkpoint.")
    if resume_from_checkpoint is not None:
        resume_from_checkpoint = Path(resume_from_checkpoint).resolve()
        for name in ("model.pt", "optimizer.pt", "scheduler.pt", "trainer_state.json"):
            if not (resume_from_checkpoint / name).is_file():
                raise FileNotFoundError(f"Missing checkpoint file: {resume_from_checkpoint / name}")
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
    if resume_from_checkpoint is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        encoder = AutoModel.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(resume_from_checkpoint, local_files_only=True)
        config = AutoConfig.from_pretrained(resume_from_checkpoint, local_files_only=True)
        encoder = AutoModel.from_config(config)
    if prepare_encoder is not None and resume_from_checkpoint is None:
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
    if resume_from_checkpoint is not None:
        options["restore_callback_states_from_checkpoint"] = not restart_lr_schedule
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
        restart_lr_schedule=restart_lr_schedule,
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
        if resume_from_checkpoint is not None:
            schedule = "restarted" if restart_lr_schedule else "restored"
            print(f"Resuming {resume_from_checkpoint}; total epoch limit={epochs}; "
                  f"learning-rate schedule {schedule}.")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(str(checkpoint_path))
    return trainer.state.log_history
