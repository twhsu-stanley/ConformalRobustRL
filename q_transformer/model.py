"""Text-only Q regression with a shared cell-wise prediction head."""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from q_transformer.data import format_prompt, grid_cell_spans


def extend_bert_positions(encoder, max_length):
    """Keep pretrained positions and randomly initialize additional trainable vectors."""
    if encoder.config.model_type != "bert" or encoder.config.position_embedding_type != "absolute":
        raise ValueError("Extending the context requires BERT with absolute positional embeddings.")
    embeddings = encoder.embeddings
    original = embeddings.position_embeddings
    if max_length <= original.num_embeddings:
        return
    expanded = nn.Embedding(
        max_length, original.embedding_dim, device=original.weight.device,
        dtype=original.weight.dtype,
    )
    with torch.no_grad():
        expanded.weight.normal_(mean=0, std=encoder.config.initializer_range)
        expanded.weight[:original.num_embeddings].copy_(original.weight)
    embeddings.position_embeddings = expanded
    position_ids = torch.arange(max_length, device=original.weight.device).expand((1, -1))
    embeddings.register_buffer("position_ids", position_ids, persistent=False)
    embeddings.register_buffer("token_type_ids", torch.zeros_like(position_ids), persistent=False)
    encoder.config.max_position_embeddings = max_length


def tokenize_prompts(tokenizer, prompts, max_length, n_states):
    """Align every grid entry with a token; never silently truncate a map."""
    if not tokenizer.is_fast:
        raise ValueError("Cell alignment requires a fast tokenizer with character offsets.")
    encoded = tokenizer(
        prompts, padding="longest", truncation=False, return_offsets_mapping=True,
        return_token_type_ids=False, pad_to_multiple_of=8 if max_length % 8 == 0 else None,
        verbose=False,
    )
    if any(len(ids) > max_length for ids in encoded["input_ids"]):
        raise ValueError("A complete prompt exceeds the encoder's token limit.")
    positions = []
    for prompt, offsets in zip(prompts, encoded.pop("offset_mapping")):
        spans = grid_cell_spans(prompt)
        lookup = {tuple(span): i for i, span in enumerate(offsets) if span[0] < span[1]}
        if len(spans) != n_states or any(span not in lookup for span in spans):
            raise ValueError("Every state must have exactly one binary grid token in row order.")
        positions.append([lookup[span] for span in spans])
    return {
        "input_ids": torch.tensor(encoded["input_ids"]),
        "attention_mask": torch.tensor(encoded["attention_mask"]),
        "cell_positions": torch.tensor(positions),
    }


class QValueTransformer(nn.Module):
    """Predict Q-values directly from each cell's contextual BERT representation."""

    def __init__(self, encoder, n_states, n_actions, width=128):
        super().__init__()
        self.encoder = encoder
        self.n_states, self.n_actions = n_states, n_actions
        self.width = width
        self.normalize = nn.LayerNorm(encoder.config.hidden_size, elementwise_affine=False)
        self.context = nn.Linear(encoder.config.hidden_size, width)
        self.q_head = nn.Sequential(
            nn.GELU(), nn.Linear(width, width), nn.GELU(), nn.Linear(width, n_actions),
        )

    def forward(self, input_ids, attention_mask, cell_positions, terminal_mask, labels=None):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        indices = cell_positions[..., None].expand(-1, -1, hidden.shape[-1])
        cells = self.normalize(hidden.gather(1, indices))
        Q = self.q_head(self.context(cells))
        Q = Q.masked_fill(terminal_mask[..., None], 0.0)
        if labels is None:
            return Q
        errors = nn.functional.smooth_l1_loss(Q.float(), labels.float(), reduction="none")
        active = ~terminal_mask[..., None]
        count = active.sum() * Q.shape[-1]
        loss = (errors * active).sum() / count.clamp(min=1)
        return {"loss": loss, "logits": Q}

    def save_checkpoint(self, path, tokenizer, max_length, training):
        """Store the encoder, Q head, and metadata needed by the predictor."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(path)
        self.encoder.config.save_pretrained(path)
        torch.save({
            "architecture": "text_cell_q", "head_width": self.width,
            "Q_shape": (self.n_states, self.n_actions),
            "state_dict": {key: value.detach().cpu() for key, value in self.state_dict().items()},
            "max_length": max_length, "training": training,
        }, path / "model.pt")


class QTransformerPredictor:
    """Encode the known task as text and return an Algorithm 1 initial Q-table."""

    def __init__(self, checkpoint_path, *, device=None):
        path = Path(checkpoint_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
        if checkpoint.get("architecture") != "text_cell_q":
            raise ValueError("Train a Q-only cell-head checkpoint; old models are incompatible.")
        self.max_length = checkpoint["max_length"]
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        config = AutoConfig.from_pretrained(path, local_files_only=True)
        self.model = QValueTransformer(
            AutoModel.from_config(config, add_pooling_layer=False), *checkpoint["Q_shape"],
            checkpoint["head_width"],
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device).eval().requires_grad_(False)

    def __call__(self, mdp, R_used, C_used):
        base = mdp.unwrapped
        if (len(base.states), mdp.action_space.n) != (self.model.n_states, self.model.n_actions):
            raise ValueError("The checkpoint Q-table shape does not match the MDP.")
        encoded = tokenize_prompts(
            self.tokenizer, [format_prompt(base, R_used, C_used)],
            self.max_length, self.model.n_states,
        )
        encoded["terminal_mask"] = torch.from_numpy(base.terminal_mask)[None]
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.inference_mode():
            Q = self.model(**encoded)[0].cpu().numpy().astype(float)
        if not np.all(np.isfinite(Q)):
            raise ValueError("The transformer predicted a nonfinite Q-table.")
        return Q
