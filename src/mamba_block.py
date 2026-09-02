"""A small, transparent PyTorch implementation of a selective Mamba block."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class MambaBlock(nn.Module):
    """Minimal Mamba-style block with an explicit sequential selective scan."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        if d_model <= 0 or d_state <= 0 or d_conv <= 0 or expand <= 0:
            raise ValueError("All dimensions and expand must be positive")

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = expand * d_model

        # One projection supplies the SSM input and the separate gate signal.
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)

        # Left padding plus truncation in forward() makes this convolution causal.
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        # These projections make the continuous-time parameters input-dependent.
        self.delta_proj = nn.Linear(self.d_inner, self.d_inner)
        self.B_proj = nn.Linear(self.d_inner, d_state)
        self.C_proj = nn.Linear(self.d_inner, d_state)

        # A diagonal, HiPPO-inspired set of increasing negative decay rates.
        # Storing log(-A) lets the parameter remain stable after optimization:
        # A = -exp(A_log) is always strictly negative.
        A_init = -torch.arange(1, d_state + 1, dtype=torch.float32)
        A_init = A_init.unsqueeze(0).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(-A_init))

        # D provides the learned input skip connection from the SSM branch.
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model)

    def _selective_scan(
        self,
        x: Tensor,
        delta: Tensor,
        B: Tensor,
        C: Tensor,
    ) -> Tensor:
        """Run the recurrence one timestep at a time for clarity."""
        batch_size, seq_len, _ = x.shape
        A = -torch.exp(self.A_log)
        hidden = x.new_zeros(batch_size, self.d_inner, self.d_state)
        outputs = []

        for timestep in range(seq_len):
            x_t = x[:, timestep, :]
            delta_t = delta[:, timestep, :]
            B_t = B[:, timestep, :]
            C_t = C[:, timestep, :]

            decay = torch.exp(delta_t.unsqueeze(-1) * A)
            hidden = decay * hidden + (
                delta_t * x_t
            ).unsqueeze(-1) * B_t.unsqueeze(1)
            y_t = torch.sum(C_t.unsqueeze(1) * hidden, dim=-1)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)

    def forward(self, x: Tensor, extra_bias: Tensor | None = None) -> Tensor:
        """Process ``x`` with optional post-convolution SSM bias injection."""
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(
                f"x must have shape (batch, seq_len, {self.d_model}), got {tuple(x.shape)}"
            )

        ssm_input, gate = self.in_proj(x).chunk(2, dim=-1)
        convolved = self.conv1d(ssm_input.transpose(1, 2))
        ssm_input = convolved[..., : x.shape[1]].transpose(1, 2)

        if extra_bias is not None:
            expected_shape = (x.shape[0], x.shape[1], self.d_inner)
            if extra_bias.shape != expected_shape:
                raise ValueError(
                    f"extra_bias must have shape {expected_shape}, got {tuple(extra_bias.shape)}"
                )
            ssm_input = ssm_input + extra_bias

        ssm_input = F.silu(ssm_input)
        delta = F.softplus(self.delta_proj(ssm_input))
        B = self.B_proj(ssm_input)
        C = self.C_proj(ssm_input)

        y = self._selective_scan(ssm_input, delta, B, C)
        y = y + ssm_input * self.D
        return self.out_proj(y * F.silu(gate))


if __name__ == "__main__":
    torch.manual_seed(0)

    block = MambaBlock(d_model=64, d_state=16, d_conv=4, expand=2)
    inputs = torch.randn(2, 32, 64, requires_grad=True)
    outputs = block(inputs)

    assert outputs.shape == inputs.shape
    outputs.sum().backward()

    A_log_grad_norm = block.A_log.grad.norm().item()
    assert A_log_grad_norm > 0, "A_log did not receive a nonzero gradient"

    print(f"Input shape:  {tuple(inputs.shape)}")
    print(f"Output shape: {tuple(outputs.shape)}")
    print(f"A_log gradient norm: {A_log_grad_norm:.6f}")
