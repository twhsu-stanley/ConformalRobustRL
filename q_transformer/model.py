"""BERT-style Q-table regression and a frozen, callable warm-start predictor."""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from q_transformer.data import format_prompt


def tokenize_prompts(tokenizer, prompts, max_length):
    """Use the original fixed-length padding and truncation for training and inference."""
    encoded = tokenizer(
        prompts, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt",
    )
    return {key: encoded[key] for key in ("input_ids", "attention_mask")}


class QValueTransformer(nn.Module):
    """Pretrained encoder, masked mean pooling, and a Q-only linear regression head."""

    def __init__(self, encoder, n_states, n_actions):
        super().__init__()
        self.encoder = encoder
        self.n_states = n_states
        self.n_actions = n_actions
        self.reg_head = nn.Linear(encoder.config.hidden_size, n_states * n_actions)
        nn.init.uniform_(self.reg_head.weight, a=-1, b=1)

    def forward(self, input_ids, attention_mask, labels=None):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        Q = self.reg_head(pooled).reshape(-1, self.n_states, self.n_actions)
        if labels is None:
            return Q
        return {"loss": nn.functional.smooth_l1_loss(Q, labels), "logits": Q}

    def save_checkpoint(self, path, tokenizer, max_length, training):
        """Store weights once, with the tokenizer, encoder config, and Q-table shape."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(path)
        self.encoder.config.save_pretrained(path)
        torch.save({
            "state_dict": {key: value.detach().cpu() for key, value in self.state_dict().items()},
            "Q_shape": (self.n_states, self.n_actions),
            "max_length": max_length, "training": training,
        }, path / "model.pt")


class QTransformerPredictor:
    """Load once, then call predictor(mdp, R_used, C_used) before Algorithm 1."""

    def __init__(self, checkpoint_path, *, device=None):
        path = Path(checkpoint_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
        self.max_length = checkpoint["max_length"]
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        config = AutoConfig.from_pretrained(path, local_files_only=True)
        self.model = QValueTransformer(AutoModel.from_config(config), *checkpoint["Q_shape"])
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device).eval()
        self.model.requires_grad_(False)

    def __call__(self, mdp, R_used, C_used):
        base = mdp.unwrapped
        if (len(base.states), mdp.action_space.n) != (self.model.n_states, self.model.n_actions):
            raise ValueError("The checkpoint Q-table shape does not match the MDP.")
        prompt = format_prompt(base.q_transformer_context(), R_used, C_used)
        encoded = tokenize_prompts(self.tokenizer, [prompt], self.max_length)
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.inference_mode():
            Q = self.model(**encoded)[0].cpu().numpy().astype(float)
        if not np.all(np.isfinite(Q)):
            raise ValueError("The transformer predicted a nonfinite Q-table.")
        Q[mdp.unwrapped.terminal_mask] = 0.0
        return Q
