"""Supervised Q regression using a transformer-based model."""

import numpy as np
import torch
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
        self.model.save_checkpoint(
            output_dir or self.args.output_dir, self.processing_class,
            self.processing_class.model_max_length, training,
        )


def train_q_transformer(
    dataset_path, checkpoint_path, *, model_name="bert-base-uncased",
    epochs=100, batch_size=4, learning_rate=1e-5, validation_fraction=0.1, seed=42,
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
    set_seed(seed)
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
    train_indices, validation_indices = split_samples(len(prompts), validation_fraction, seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name)
    max_length = min(max_length, encoder.config.max_position_embeddings, tokenizer.model_max_length)
    tokenizer.model_max_length = int(max_length)
    encoded = tokenize_prompts(tokenizer, prompts, max_length)
    dataset = [
        {"input_ids": encoded["input_ids"][i], "attention_mask": encoded["attention_mask"][i],
         "labels": torch.from_numpy(targets[i])}
        for i in range(len(prompts))
    ]
    use_cpu = device is not None and torch.device(device).type == "cpu"
    if device is not None and torch.device(device).type == "cuda":
        torch.cuda.set_device(device)
    arguments = TrainingArguments(
        output_dir=str(checkpoint_path), save_strategy="epoch", num_train_epochs=epochs,
        per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
        logging_steps=50, learning_rate=learning_rate, weight_decay=0.01,
        fp16=torch.cuda.is_available() and not use_cpu, use_cpu=use_cpu, seed=seed,
        optim="adamw_torch", lr_scheduler_type="linear", report_to="none",
    )
    trainer = _QTrainer(
        model=QValueTransformer(encoder, n_states, n_actions), args=arguments,
        train_dataset=[dataset[i] for i in train_indices],
        eval_dataset=[dataset[i] for i in validation_indices], processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(checkpoint_path))
    return trainer.state.log_history
