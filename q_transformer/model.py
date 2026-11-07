"""Cell-wise BERT Q regression and the callable Algorithm 1 initializer."""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from q_transformer.data import format_prompt


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
    marker = "0 = free space, 1 = obstacle.\n\n"
    for prompt, offsets in zip(prompts, encoded.pop("offset_mapping")):
        start = prompt.index(marker) + len(marker)
        end = prompt.index("Start at", start)
        cells = [i for i, (left, right) in enumerate(offsets) if start <= left < right <= end]
        entries = [prompt[offsets[i][0]:offsets[i][1]] for i in cells]
        if len(cells) != n_states or any(value not in ("0", "1") for value in entries):
            raise ValueError("Every state must have exactly one binary grid token in row order.")
        positions.append(cells)
    return {
        "input_ids": torch.tensor(encoded["input_ids"]),
        "attention_mask": torch.tensor(encoded["attention_mask"]),
        "cell_positions": torch.tensor(positions),
    }


class QValueTransformer(nn.Module):
    """One trainable BERT embedding per cell and a shared numerical-feature decoder."""

    def __init__(self, encoder, n_states, n_actions, numeric_width, width=128):
        super().__init__()
        self.encoder = encoder
        self.n_states, self.n_actions = n_states, n_actions
        self.numeric_width, self.width = numeric_width, width
        self.normalize = nn.LayerNorm(encoder.config.hidden_size, elementwise_affine=False)
        self.context = nn.Linear(encoder.config.hidden_size, width)
        self.numeric = nn.Linear(numeric_width, width)
        self.decoder = nn.Sequential(
            nn.GELU(), nn.Linear(width, width), nn.GELU(), nn.Linear(width, n_actions),
        )

    def forward(self, input_ids, attention_mask, cell_positions, features, terminal_mask,
                labels=None):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        indices = cell_positions[..., None].expand(-1, -1, hidden.shape[-1])
        cells = hidden.gather(1, indices)
        context = self.context(self.normalize(cells))[:, None]
        Q = self.decoder(context + self.numeric(features))
        Q = Q.masked_fill(terminal_mask[:, None, :, None], 0.0)
        if labels is None:
            return Q
        errors = nn.functional.smooth_l1_loss(Q.float(), labels.float(), reduction="none")
        active = ~terminal_mask[:, None, :, None]
        count = active.sum() * Q.shape[1] * Q.shape[-1]
        loss = (errors * active).sum() / count.clamp(min=1)
        return {"loss": loss, "logits": Q}

    def save_checkpoint(self, path, tokenizer, max_length, training):
        """Store one model plus the metadata needed to reconstruct its decoder."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(path)
        self.encoder.config.save_pretrained(path)
        torch.save({
            "architecture": "cell_decoder", "numeric_width": self.numeric_width,
            "decoder_width": self.width, "Q_shape": (self.n_states, self.n_actions),
            "state_dict": {key: value.detach().cpu() for key, value in self.state_dict().items()},
            "max_length": max_length, "training": training,
        }, path / "model.pt")


class QTransformerPredictor:
    """Load the new cell decoder once and return a numerical Q-table for Algorithm 1."""

    def __init__(self, checkpoint_path, *, device=None):
        path = Path(checkpoint_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
        if checkpoint.get("architecture") != "cell_decoder":
            raise ValueError("Use a new cell-decoder checkpoint; old models are incompatible.")
        self.max_length = checkpoint["max_length"]
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        config = AutoConfig.from_pretrained(path, local_files_only=True)
        self.model = QValueTransformer(
            AutoModel.from_config(config, add_pooling_layer=False), *checkpoint["Q_shape"],
            checkpoint["numeric_width"], checkpoint["decoder_width"],
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device).eval().requires_grad_(False)

    def __call__(self, mdp, R_used, C_used):
        base = mdp.unwrapped
        if (len(base.states), mdp.action_space.n) != (self.model.n_states, self.model.n_actions):
            raise ValueError("The checkpoint Q-table shape does not match the MDP.")
        encoded = tokenize_prompts(
            self.tokenizer, [format_prompt(base.q_transformer_context())],
            self.max_length, self.model.n_states,
        )
        features = base.q_transformer_features(R_used, C_used)
        if features.shape != (1, self.model.n_states, self.model.numeric_width):
            raise ValueError("The MDP numerical features do not match the checkpoint.")
        encoded["features"] = torch.from_numpy(features)[None]
        encoded["terminal_mask"] = torch.from_numpy(base.terminal_mask)[None]
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.inference_mode():
            Q = self.model(**encoded)[0, 0].cpu().numpy().astype(float)
        if not np.all(np.isfinite(Q)):
            raise ValueError("The transformer predicted a nonfinite Q-table.")
        return Q
