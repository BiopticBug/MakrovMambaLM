"""A differentiable, input-conditioned Markov state tracker."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class MarkovStateTracker(nn.Module):
    """Track a soft discrete state sequence using an input-conditioned transition."""

    def __init__(
        self,
        input_dim: int,
        num_states: int = 8,
        state_embed_dim: int = 128,
        gumbel_tau: float = 1.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or num_states <= 0 or state_embed_dim <= 0:
            raise ValueError("All dimensions must be positive")
        if gumbel_tau <= 0:
            raise ValueError("gumbel_tau must be positive")

        self.input_dim = input_dim
        self.num_states = num_states
        self.state_embed_dim = state_embed_dim
        # This remains a plain attribute so a training loop can anneal it directly.
        self.gumbel_tau = gumbel_tau

        # Conditioning on both terms makes this a Markov transition model: the
        # next state depends on the current content and the previous state.
        self.transition_layer = nn.Linear(input_dim + num_states, num_states)
        self.state_embedding = nn.Embedding(num_states, state_embed_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return state embeddings and soft state assignments for every token."""
        if x.ndim != 3 or x.shape[-1] != self.input_dim:
            raise ValueError(
                f"x must have shape (batch, seq_len, {self.input_dim}), got {tuple(x.shape)}"
            )
        if x.shape[1] == 0:
            raise ValueError("x must contain at least one timestep")

        batch_size, seq_len, _ = x.shape

        # A uniform prior treats every state symmetrically at t=0. The first
        # token's content can therefore determine the initial transition.
        previous_state = x.new_full(
            (batch_size, self.num_states), 1.0 / self.num_states
        )
        state_distributions = []
        state_embeddings = []

        for timestep in range(seq_len):
            token = x[:, timestep, :]
            transition_input = torch.cat((token, previous_state), dim=-1)
            logits = self.transition_layer(transition_input)

            # Gumbel-Softmax gives a sample-like state assignment while retaining
            # gradients for end-to-end training with the Mamba backbone. A plain
            # softmax is deterministic during evaluation; hard argmax would stop
            # gradients from the state tracker reaching the transition weights.
            if self.training:
                current_state = F.gumbel_softmax(
                    logits, tau=self.gumbel_tau, hard=False, dim=-1
                )
            else:
                current_state = F.softmax(logits, dim=-1)

            embedding = current_state @ self.state_embedding.weight
            state_distributions.append(current_state)
            state_embeddings.append(embedding)
            previous_state = current_state

        return (
            torch.stack(state_embeddings, dim=1),
            torch.stack(state_distributions, dim=1),
        )


if __name__ == "__main__":
    torch.manual_seed(0)

    tracker = MarkovStateTracker(
        input_dim=64,
        num_states=8,
        state_embed_dim=128,
    )
    inputs = torch.randn(2, 32, 64, requires_grad=True)
    embeddings, distributions = tracker(inputs)

    assert embeddings.shape == (2, 32, 128)
    assert distributions.shape == (2, 32, 8)
    assert torch.allclose(
        distributions.sum(dim=-1),
        torch.ones(2, 32),
        atol=1e-5,
    )

    embeddings.sum().backward()
    transition_grad_norm = tracker.transition_layer.weight.grad.norm().item()
    assert transition_grad_norm > 0, (
        "The transition scoring layer did not receive a nonzero gradient"
    )

    state_sequence = distributions[0].argmax(dim=-1)[:10].tolist()
    print(f"State embeddings shape:     {tuple(embeddings.shape)}")
    print(f"State distributions shape:  {tuple(distributions.shape)}")
    print(f"Transition weight grad norm: {transition_grad_norm:.6f}")
    print(f"First sample, first 10 states: {state_sequence}")
