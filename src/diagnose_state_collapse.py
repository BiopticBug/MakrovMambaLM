"""Diagnose whether Markov state assignments collapse during training."""

import math

import torch
from torch.nn import functional as F

from markov_mamba_model import MarkovMambaLM
from synthetic_tasks import generate_associative_recall_task


torch.manual_seed(42)

vocab_size = 50
seq_len = 16
num_pairs = 1
d_model = 64
num_layers = 4
num_markov_states = 8
batch_size = 32
num_train_steps = 500
eval_every = 50
learning_rate = 1e-3
gradient_clip_norm = 1.0
device = "cuda" if torch.cuda.is_available() else "cpu"
max_entropy = math.log(num_markov_states)


def state_statistics(
    all_state_distributions: list[torch.Tensor],
) -> tuple[float, float, list[tuple[float, float]]]:
    """Return global entropy, global dominant fraction, and per-layer metrics."""
    layer_statistics = []
    all_distributions = []
    all_assignments = []
    for distributions in all_state_distributions:
        entropy = -(
            distributions.clamp_min(1e-12) * distributions.clamp_min(1e-12).log()
        ).sum(dim=-1)
        assignments = distributions.argmax(dim=-1)
        counts = torch.bincount(
            assignments.reshape(-1), minlength=num_markov_states
        )
        layer_statistics.append(
            (float(entropy.mean().item()), float(counts.max().item() / assignments.numel()))
        )
        all_distributions.append(distributions.reshape(-1, num_markov_states))
        all_assignments.append(assignments.reshape(-1))

    combined_distributions = torch.cat(all_distributions, dim=0)
    combined_assignments = torch.cat(all_assignments, dim=0)
    combined_entropy = -(
        combined_distributions.clamp_min(1e-12)
        * combined_distributions.clamp_min(1e-12).log()
    ).sum(dim=-1)
    combined_counts = torch.bincount(
        combined_assignments, minlength=num_markov_states
    )
    dominant_fraction = float(
        combined_counts.max().item() / combined_assignments.numel()
    )
    return float(combined_entropy.mean().item()), dominant_fraction, layer_statistics


def main() -> None:
    model = MarkovMambaLM(
        vocab_size=vocab_size,
        d_model=d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        num_markov_states=num_markov_states,
        num_layers=num_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print(f"Device: {device}")
    print("Markov state collapse diagnosis")
    print(
        f"Configuration: seq_len={seq_len}, num_pairs={num_pairs}, "
        f"steps={num_train_steps}, learning_rate={learning_rate}, "
        f"clip_norm={gradient_clip_norm}"
    )
    print(f"Maximum possible state entropy: ln({num_markov_states}) = {max_entropy:.3f}")

    final_statistics = None
    model.train()
    for step in range(1, num_train_steps + 1):
        input_ids, target_ids = generate_associative_recall_task(
            batch_size, seq_len, vocab_size, num_pairs
        )
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        result = model(input_ids)
        logits = result["logits"]
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            target_ids.reshape(-1),
            ignore_index=-100,
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=gradient_clip_norm
        )
        optimizer.step()

        if step % eval_every == 0:
            with torch.no_grad():
                entropy, dominant_fraction, final_statistics = state_statistics(
                    result["all_state_distributions"]
                )
            print(
                f"Step {step:3d} | loss: {loss.item():.3f} "
                f"| avg_entropy: {entropy:.3f} "
                f"| dominant_state_fraction: {dominant_fraction:.3f}"
            )

    # Use a fresh diagnostic batch after training for the final verdict.
    torch.manual_seed(43)
    heldout_input_ids, _ = generate_associative_recall_task(
        batch_size, seq_len, vocab_size, num_pairs
    )
    with torch.no_grad():
        final_result = model(heldout_input_ids.to(device))
        final_entropy, final_dominant_fraction, final_statistics = state_statistics(
            final_result["all_state_distributions"]
        )

    print("\nFinal state-collapse evaluation")
    print(f"Average entropy:           {final_entropy:.6f}")
    print(f"Dominant-state fraction:   {final_dominant_fraction:.6f}")

    if final_entropy > 1.8 and final_dominant_fraction < 0.3:
        print("STATE TRACKER IS DIVERSE - not collapsed, issue is elsewhere")
    elif final_entropy < 0.5 or final_dominant_fraction > 0.5:
        print(
            "STATE TRACKER HAS COLLAPSED - this explains the generalization "
            "failure, needs fixing"
        )
    else:
        print(
            "STATE TRACKER IS PARTIALLY DIVERSE - inconclusive, print the full "
            "per-layer breakdown of entropy and dominant-state fraction for "
            "manual inspection"
        )

    print("\nFinal per-layer breakdown")
    for layer_index, (entropy, dominant_fraction) in enumerate(final_statistics, 1):
        print(
            f"Layer {layer_index}: entropy={entropy:.6f}, "
            f"dominant_state_fraction={dominant_fraction:.6f}"
        )


if __name__ == "__main__":
    main()
