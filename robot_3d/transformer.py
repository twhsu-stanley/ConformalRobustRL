"""Extend pretrained BERT positions and validate complete 3-D transformer inputs."""

import torch

from q_transformer.data import format_prompt
from q_transformer.model import QTransformerPredictor as SharedQTransformerPredictor


def extend_bert_positions(encoder, max_length):
    """Keep all pretrained weights; initialize only the additional position vectors."""
    if encoder.config.model_type != "bert" or encoder.config.position_embedding_type != "absolute":
        raise ValueError("Extending the context requires BERT with absolute positional embeddings.")
    embeddings = encoder.embeddings
    original = embeddings.position_embeddings
    if max_length <= original.num_embeddings:
        return
    expanded = torch.nn.Embedding(
        max_length, original.embedding_dim, device=original.weight.device,
        dtype=original.weight.dtype,
    )
    with torch.no_grad():
        expanded.weight.normal_(mean=0, std=encoder.config.initializer_range)
        expanded.weight[:original.num_embeddings].copy_(original.weight)
    expanded.weight.requires_grad_(original.weight.requires_grad)
    embeddings.position_embeddings = expanded
    position_ids = torch.arange(max_length, device=original.weight.device).expand((1, -1))
    embeddings.register_buffer("position_ids", position_ids, persistent=False)
    embeddings.register_buffer("token_type_ids", torch.zeros_like(position_ids), persistent=False)
    encoder.config.max_position_embeddings = max_length


def validate_prompt_length(tokenizer, prompt, max_length):
    """Reject truncation before training or inference; include special tokens in the count."""
    length = len(tokenizer(prompt, truncation=False, verbose=False)["input_ids"])
    if length > max_length:
        raise ValueError(
            f"The complete 3-D prompt needs {length} tokens, but the model allows {max_length}. "
            "Reduce the obstacle description or use an encoder with a longer context."
        )


class QTransformerPredictor(SharedQTransformerPredictor):
    """Keep the shared frozen predictor and validate every complete 3-D input."""

    def __call__(self, mdp, R_used, C_used):
        prompt = format_prompt(mdp.unwrapped.q_transformer_context(), R_used, C_used)
        validate_prompt_length(self.tokenizer, prompt, self.max_length)
        return super().__call__(mdp, R_used, C_used)
