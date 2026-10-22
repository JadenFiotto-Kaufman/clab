"""Writes documents/artifacts/sae_gemma_scope.safetensors and documents/artifacts/sae_saelens.safetensors, the
published sparse autoencoders sae_published_jumprelu.json and sae_published_topk.json load: random 16 -> 32
dictionaries for the tiny Llama's layer_output at layer 0, drawn from seeded generators and laid out as the two
trainers publish them, with no clab header. Gemma Scope's holds a JumpReLU `threshold` (and its encoder does not
subtract `b_dec`); SAELens's does not. Fixtures: their mechanics are real and their latents mean nothing. Run
from the corpus root: `uv run python tests/fixtures/published_sae.py`. They are written as clab writes a bundle,
their headers' keys sorted: the same bytes on every run."""

import torch

from clab.output import write


if __name__ == "__main__":
    g = torch.Generator().manual_seed(1)
    write("documents/artifacts/sae_gemma_scope.safetensors", {
        "W_enc": torch.randn(16, 32, generator=g) / 4, "W_dec": torch.randn(32, 16, generator=g) / 4,
        "b_enc": torch.randn(32, generator=g) / 100, "b_dec": torch.randn(16, generator=g) / 100,
        "threshold": torch.rand(32, generator=g) / 50})
    g = torch.Generator().manual_seed(2)
    write("documents/artifacts/sae_saelens.safetensors", {
        "W_enc": torch.randn(16, 32, generator=g) / 4, "W_dec": torch.randn(32, 16, generator=g) / 4,
        "b_enc": torch.zeros(32), "b_dec": torch.randn(16, generator=g) / 100})
