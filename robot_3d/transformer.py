"""Use shared transformer inference while rejecting incomplete 3-D prompts."""

from q_transformer.data import format_prompt
from q_transformer.model import QTransformerPredictor as SharedQTransformerPredictor


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
