"""Writes documents/artifacts/sae.safetensors, the sparse autoencoder sae_feature_ablation.json loads: a random
16 -> 32 dictionary for the tiny Llama's layer_output at layer 0, drawn from a seeded generator. A fixture: its
mechanics are real and its latents mean nothing. Run from the corpus root: `uv run python tests/fixtures/sae.py`.
It is written as clab writes a bundle, its header's keys sorted: the same bytes on every run."""

import torch

from clab.output.dump import write

HEADER = {"format": "clab/1", "kind": "sae", "model.key": "hf-internal-testing/tiny-random-LlamaForCausalLM",
          "model.revision": "9fb191250dd56d0ba7ec9785a025ed29c03d5998", "model.dtype": "float32",
          "component": "layer_output", "layer": "0", "width": "16", "document": "sae", "activation": "relu",
          "centered": "true"}

if __name__ == "__main__":
    g = torch.Generator().manual_seed(0)
    tensors = {"W_enc": torch.randn(16, 32, generator=g) / 4, "b_enc": torch.zeros(32),
               "W_dec": torch.randn(32, 16, generator=g) / 4, "b_dec": torch.randn(16, generator=g) / 100}
    write("documents/artifacts/sae.safetensors", tensors, HEADER)
