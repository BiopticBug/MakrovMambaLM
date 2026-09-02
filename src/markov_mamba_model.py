"""A stacked Markov-Mamba language-model-shaped network."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from markov_mamba_block import MarkovMambaBlock


class MarkovMambaLM(nn.Module):
    """Stack pre-norm residual Markov-Mamba blocks behind a token embedding."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_markov_states: int = 8,
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
        # Mamba recurrence is already order-aware, so positions are not strictly
        # required. Keeping learned positions makes this an explicit ablation.
        self.position_embedding = (
            nn.Embedding(max_seq_len, d_model)
            if use_positional_embedding
            else None
        )

        self.norms = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(num_layers))
        self.layers = nn.ModuleList(
            MarkovMambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                num_markov_states=num_markov_states,
            )
            for _ in range(num_layers)
        )

        self.final_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: Tensor) -> dict[str, Tensor | list[Tensor]]:
        """Return vocabulary logits and every layer's state distributions."""
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

        all_state_distributions = []
        for norm, layer in zip(self.norms, self.layers):
            # Pre-normalization keeps each residual branch well-scaled, while
            # the identity path improves optimization stability as depth grows.
            layer_result = layer(norm(hidden))
            hidden = hidden + layer_result["output"]
            all_state_distributions.append(layer_result["state_distributions"])

        logits = self.output_head(self.final_norm(hidden))
        return {
            "logits": logits,
            "all_state_distributions": all_state_distributions,
        }


if __name__ == "__main__":
    torch.manual_seed(0)

    batch_size, seq_len, vocab_size = 4, 64, 1000
    model = MarkovMambaLM(
        vocab_size=vocab_size,
        d_model=64,
        num_layers=4,
        num_markov_states=8,
    )
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    result = model(input_ids)
    logits = result["logits"]
    all_state_distributions = result["all_state_distributions"]

    print(f"Logits shape:                {tuple(logits.shape)}")
    assert logits.shape == (batch_size, seq_len, vocab_size)
    print(f"State-distribution layers:   {len(all_state_distributions)}")
    assert len(all_state_distributions) == 4

    # Position t predicts the token at position t+1.
    shifted_logits = logits[:, :-1, :].reshape(-1, vocab_size)
    shifted_targets = input_ids[:, 1:].reshape(-1)
    loss = F.cross_entropy(shifted_logits, shifted_targets)
    loss.backward()

    missing_gradients = []
    for parameter_name, parameter in model.named_parameters():
        if parameter.grad is None:
            print(f"Missing gradient: {parameter_name}")
            missing_gradients.append(parameter_name)
    assert not missing_gradients, f"Parameters without gradients: {missing_gradients}"

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Loss:                         {loss.item():.6f}")
    print(f"Total parameter count:       {parameter_count}")
    print("All parameters received gradients: yes")
    print()
    print("Model summary")
    print("-------------")
    print(f"Model name:          MarkovMambaLM")
    print(f"Number of layers:    {len(model.layers)}")
    print(f"d_model:             {model.d_model}")
    print(f"Markov states:       {model.layers[0].markov_tracker.num_states}")
    print(f"Total parameters:    {parameter_count}")
    print(f"Final loss:          {loss.item():.6f}")
