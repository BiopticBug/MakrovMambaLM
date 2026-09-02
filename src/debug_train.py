"""Debug training stability on a simplified associative-recall task."""

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
num_train_steps = 300
eval_every = 20
learning_rate = 3e-4
gradient_clip_norm = 1.0
device = "cuda" if torch.cuda.is_available() else "cpu"


def compute_accuracy(logits: torch.Tensor, target_ids: torch.Tensor) -> float | None:
    """Compute accuracy only at target positions that are not masked."""
    supervised = target_ids != -100
    total = int(supervised.sum().item())
    if total == 0:
        return None
    predictions = logits.argmax(dim=-1)
    correct = (predictions[supervised] == target_ids[supervised]).sum().item()
    return float(correct / total)


def total_gradient_norm(model: torch.nn.Module) -> float:
    """Return the L2 norm across all currently populated parameter gradients."""
    squared_norm = torch.zeros((), device=device)
    for parameter in model.parameters():
        if parameter.grad is not None:
            squared_norm = squared_norm + parameter.grad.detach().pow(2).sum()
    return float(squared_norm.sqrt().item())


def print_alignment_probe(
    model: MarkovMambaLM,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
) -> None:
    """Show the old shifted index beside the corrected aligned index."""
    model.eval()
    with torch.no_grad():
        logits = model(input_ids)["logits"]
    supervised_positions = (target_ids[0] != -100).nonzero(as_tuple=False)
    if supervised_positions.numel() == 0:
        raise ValueError("Alignment probe requires a supervised target")
    position = int(supervised_positions[0].item())
    target = int(target_ids[0, position].item())
    aligned_prediction = int(logits[0, position].argmax().item())
    shifted_prediction = (
        int(logits[0, position - 1].argmax().item()) if position > 0 else None
    )
    print("\nAlignment probe (first debug sample)")
    print(f"Supervised target position: {position}")
    print(f"Target token at that position: {target}")
    print(
        "Before fix, shifted prediction from position "
        f"{position - 1}: {shifted_prediction}"
    )
    print(
        "After fix, aligned prediction from position "
        f"{position}: {aligned_prediction}"
    )
    print("Loss/accuracy now use logits[:, position] with target_ids[:, position].")


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
    history: list[tuple[int, float, float | None, float]] = []

    # This probe makes the position convention visible before optimization.
    torch.manual_seed(123)
    probe_input_ids, probe_target_ids = generate_associative_recall_task(
        1, seq_len, vocab_size, num_pairs
    )
    print_alignment_probe(
        model,
        probe_input_ids.to(device),
        probe_target_ids.to(device),
    )
    torch.manual_seed(42)

    print(f"Device: {device}")
    print("Simplified associative recall debug training")
    print(
        f"Configuration: seq_len={seq_len}, num_pairs={num_pairs}, "
        f"steps={num_train_steps}, clip_norm={gradient_clip_norm}"
    )

    model.train()
    for step in range(1, num_train_steps + 1):
        input_ids, target_ids = generate_associative_recall_task(
            batch_size, seq_len, vocab_size, num_pairs
        )
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        logits = model(input_ids)["logits"]
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            target_ids.reshape(-1),
            ignore_index=-100,
        )
        optimizer.zero_grad()
        loss.backward()

        # Measure the unconstrained norm first, then clip before the update.
        pre_clip_gradient_norm = total_gradient_norm(model)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=gradient_clip_norm
        )
        optimizer.step()

        train_loss = float(loss.item())
        train_accuracy = compute_accuracy(logits.detach(), target_ids)
        history.append(
            (step, train_loss, train_accuracy, pre_clip_gradient_norm)
        )

        if step % eval_every == 0 or step == num_train_steps:
            accuracy_text = (
                f"{train_accuracy:.3f}" if train_accuracy is not None else "n/a"
            )
            print(
                f"Step {step:3d} | loss: {train_loss:.3f} "
                f"| train_acc: {accuracy_text} "
                f"| pre_clip_grad_norm: {pre_clip_gradient_norm:.3f}"
            )

    torch.manual_seed(43)
    heldout_input_ids, heldout_target_ids = generate_associative_recall_task(
        128, seq_len, vocab_size, num_pairs
    )
    heldout_input_ids = heldout_input_ids.to(device)
    heldout_target_ids = heldout_target_ids.to(device)
    model.eval()
    with torch.no_grad():
        heldout_logits = model(heldout_input_ids)["logits"]
        heldout_loss = F.cross_entropy(
            heldout_logits.reshape(-1, vocab_size),
            heldout_target_ids.reshape(-1),
            ignore_index=-100,
        )
        heldout_accuracy = compute_accuracy(heldout_logits, heldout_target_ids)

    print("\nFinal held-out evaluation")
    print(f"Held-out loss:     {heldout_loss.item():.6f}")
    print(f"Held-out accuracy: {heldout_accuracy:.6f}")
    if heldout_accuracy > 0.8:
        print(f"Result: LEARNED (final accuracy > 0.8)")
    else:
        print(
            f"Result: DID NOT LEARN (final accuracy: {heldout_accuracy:.6f})"
        )


if __name__ == "__main__":
    main()
