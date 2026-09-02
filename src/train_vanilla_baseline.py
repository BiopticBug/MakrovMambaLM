"""Train the vanilla Mamba baseline on synthetic associative recall."""

import json

import torch
from torch.nn import functional as F

from synthetic_tasks import generate_associative_recall_task
from vanilla_mamba_model import VanillaMambaLM


torch.manual_seed(42)

vocab_size = 50
seq_len = 16
num_pairs = 1
d_model = 66
num_layers = 4
num_markov_states = 8
batch_size = 32
num_train_steps = 3000
eval_every = 100
learning_rate = 1e-3
gradient_clip_norm = 1.0
device = "cuda" if torch.cuda.is_available() else "cpu"
history_path = "training_history_vanilla_mamba_recall.json"


def compute_accuracy(logits: torch.Tensor, target_ids: torch.Tensor) -> float | None:
    """Return accuracy at aligned, non-masked target positions."""
    predictions = logits.argmax(dim=-1)
    supervised = target_ids != -100
    total = int(supervised.sum().item())
    if total == 0:
        return None
    correct = (predictions[supervised] == target_ids[supervised]).sum().item()
    return float(correct / total)


def evaluate(
    model: VanillaMambaLM,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
) -> tuple[float, float | None]:
    """Compute masked loss and aligned accuracy without tracking gradients."""
    with torch.no_grad():
        logits = model(input_ids)["logits"]
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            target_ids.reshape(-1),
            ignore_index=-100,
        )
        accuracy = compute_accuracy(logits, target_ids)
    return float(loss.item()), accuracy


def main() -> None:
    model = VanillaMambaLM(
        vocab_size=vocab_size,
        d_model=d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        num_layers=num_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history: list[tuple[int, float, float | None]] = []

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Device: {device}")
    print("Training vanilla Mamba associative recall baseline")
    print(f"Parameter count: {parameter_count}")
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
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=gradient_clip_norm
        )
        optimizer.step()

        train_loss = float(loss.item())
        train_accuracy = compute_accuracy(logits.detach(), target_ids)
        history.append((step, train_loss, train_accuracy))
        if step % eval_every == 0 or step == num_train_steps:
            accuracy_text = (
                f"{train_accuracy:.3f}" if train_accuracy is not None else "n/a"
            )
            print(
                f"Step {step:4d} | loss: {train_loss:.3f} "
                f"| train_acc: {accuracy_text}"
            )

    torch.manual_seed(43)
    heldout_input_ids, heldout_target_ids = generate_associative_recall_task(
        256, seq_len, vocab_size, num_pairs
    )
    model.eval()
    heldout_loss, heldout_accuracy = evaluate(
        model,
        heldout_input_ids.to(device),
        heldout_target_ids.to(device),
    )

    with open(history_path, "w", encoding="utf-8") as history_file:
        json.dump(
            [
                {"step": step, "loss": loss_value, "accuracy": accuracy}
                for step, loss_value, accuracy in history
            ],
            history_file,
            indent=2,
        )

    final_train_accuracy = history[-1][2]
    print("\nFinal held-out evaluation")
    print(f"Held-out loss:              {heldout_loss:.6f}")
    print(
        f"Held-out accuracy:          {heldout_accuracy:.6f}"
        if heldout_accuracy is not None
        else "Held-out accuracy:          n/a"
    )
    print("\nTraining summary")
    print("----------------")
    print("Task:                       associative recall")
    print(f"Model:                      VanillaMambaLM")
    print(f"Total training steps:      {num_train_steps}")
    print(f"Final training accuracy:   {final_train_accuracy:.6f}")
    print(
        f"Final held-out accuracy:   {heldout_accuracy:.6f}"
        if heldout_accuracy is not None
        else "Final held-out accuracy:   n/a"
    )
    print(f"Total parameter count:     {parameter_count}")
    print(f"History saved to:          {history_path}")


if __name__ == "__main__":
    main()
