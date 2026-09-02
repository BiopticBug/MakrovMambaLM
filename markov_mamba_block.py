"""A hybrid Markov state tracker and selective Mamba block."""

import torch
from torch import Tensor, nn

from mamba_block import MambaBlock
from markov_state import MarkovStateTracker


class MarkovMambaBlock(nn.Module):
    """Inject differentiable Markov state embeddings into a Mamba block."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_markov_states: int = 8,
        gumbel_tau: float = 1.0,
    ) -> None:
        super().__init__()

        self.mamba = MambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.markov_tracker = MarkovStateTracker(
            input_dim=d_model,
            num_states=num_markov_states,
            state_embed_dim=self.mamba.d_inner,
            gumbel_tau=gumbel_tau,
        )

        # The state embedding must equal d_inner: MambaBlock.extra_bias is added
        # elementwise to its post-convolution SSM branch. Keeping this assertion
        # close to construction catches the most common integration mismatch.
        assert self.markov_tracker.state_embed_dim == self.mamba.d_inner

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Return Mamba output plus state data for later analysis or losses."""
        state_embeddings, state_distributions = self.markov_tracker(x)

        # Injection after convolution lets the Markov signal condition the local
        # features that produce delta, B, and C. Before convolution it would be
        # mixed by the temporal filter; after the whole block it could not affect
        # the selective dynamics, so this location keeps the conditioning signal
        # both temporally contextualized and directly visible to the SSM.
        output = self.mamba(x, extra_bias=state_embeddings)

        return {
            "output": output,
            "state_distributions": state_distributions,
            "state_embeddings": state_embeddings,
        }


if __name__ == "__main__":
    batch_size, seq_len, d_model = 2, 32, 64
    block = MarkovMambaBlock(
        d_model=d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        num_markov_states=8,
    )
    parameter_count = sum(parameter.numel() for parameter in block.parameters())

    torch.manual_seed(0)
    inputs = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    result = block(inputs)
    output = result["output"]
    state_distributions = result["state_distributions"]

    print(f"Output shape:                {tuple(output.shape)}")
    assert output.shape == inputs.shape
    print(f"State distributions shape:   {tuple(state_distributions.shape)}")

    loss = output.sum() + state_distributions.sum()
    loss.backward()

    A_log_grad_norm = block.mamba.A_log.grad.norm().item()
    transition_grad_norm = block.markov_tracker.transition_layer.weight.grad.norm().item()
    in_proj_grad_norm = block.mamba.in_proj.weight.grad.norm().item()
    assert A_log_grad_norm > 0
    assert transition_grad_norm > 0
    assert in_proj_grad_norm > 0

    print(f"A_log gradient norm:         {A_log_grad_norm:.6f}")
    print(f"Transition weight grad norm: {transition_grad_norm:.6f}")
    print(f"Mamba in_proj grad norm:     {in_proj_grad_norm:.6f}")
    print(f"Total parameter count:       {parameter_count}")

    for pass_index in range(3):
        torch.manual_seed(pass_index + 1)
        block.zero_grad()
        repeated_inputs = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        repeated_result = block(repeated_inputs)
        repeated_loss = (
            repeated_result["output"].sum()
            + repeated_result["state_distributions"].sum()
        )
        repeated_loss.backward()
        assert repeated_result["output"].shape == repeated_inputs.shape
        print(f"Repeated pass {pass_index + 1}: passed")
