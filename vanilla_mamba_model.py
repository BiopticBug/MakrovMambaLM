"""A parameter-matched vanilla Mamba language-model baseline."""

import torch
from torch import Tensor, nn

from mamba_block import MambaBlock


class VanillaMambaLM(nn.Module):
    """Stack pre-norm residual Mamba blocks without Markov conditioning."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 66,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_layers: int = 4,
        max_seq_len: int = 512,
        use_positional_embedding: bool = True,
    ) -> None:
        super().__init__()
        if vocab_size <= 0 or num_layers <= 0 or max_seq_len <= 0:
            raise ValueError("vocab_size, num_layers, and max_seq_len must be positive")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.use_positional_embedding = use_positional_embedding

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # Mamba recurrence is order-aware, but learned positions remain useful
        # as an explicitly switchable ablation matching the hybrid model.
        self.position_embedding = (
            nn.Embedding(max_seq_len, d_model)
            if use_positional_embedding
            else None
        )
        self.norms = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(num_layers))
        self.layers = nn.ModuleList(
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            for _ in range(num_layers)
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: Tensor) -> dict[str, Tensor]:
        """Return vocabulary logits using only the vanilla Mamba pathway."""
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must have shape (batch, seq_len), got {tuple(input_ids.shape)}"
            )
        if input_ids.dtype not in (torch.int64, torch.int32):
            raise TypeError("input_ids must contain integer token ids")

        batch_size, seq_len = input_ids.shape
        if seq_len == 0 or seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len must be between 1 and {self.max_seq_len}, got {seq_len}"
            )

        hidden = self.token_embedding(input_ids)
        if self.position_embedding is not None:
            positions = torch.arange(seq_len, device=input_ids.device)
            hidden = hidden + self.position_embedding(positions).unsqueeze(0)

        for norm, mamba_block in zip(self.norms, self.layers):
            # Match the hybrid model's stable pre-norm residual architecture.
            hidden = hidden + mamba_block(norm(hidden))

        return {"logits": self.output_head(self.final_norm(hidden))}


if __name__ == "__main__":
    model = VanillaMambaLM(vocab_size=50)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"VanillaMambaLM parameter count: {parameter_count}")
