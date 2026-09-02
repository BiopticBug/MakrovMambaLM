"""Compare Markov-Mamba and vanilla Mamba on synthetic copying."""

import torch
from torch.nn import functional as F

from markov_mamba_model import MarkovMambaLM
from synthetic_tasks import generate_copying_task
from vanilla_mamba_model import VanillaMambaLM


torch.manual_seed(42)

vocab_size = 50
seq_len = 32
num_tokens_to_copy = 4
batch_size = 32
num_train_steps = 2000
eval_every = 100
learning_rate = 1e-3
gradient_clip_norm = 1.0
device = "cuda" if torch.cuda.is_available() else "cpu"


def compute_accuracy(logits: torch.Tensor, target_ids: torch.Tensor) -> float | None:
    """Compute accuracy at the same positions as the aligned targets."""
    supervised = target_ids != -100
    total = int(supervised.sum().item())
    if total == 0:
        return None
    predictions = logits.argmax(dim=-1)
    return float((predictions[supervised] == target_ids[supervised]).float().mean())


def train_model(model: torch.nn.Module, model_name: str) -> dict[str, float | int]:
    """Train one model on fresh batches and evaluate it on fresh held-out data."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    print(f"\n=== {model_name} ===")
    print(f"Parameter count: {parameter_count}")
    model.train()
    final_train_accuracy = None
    for step in range(1, num_train_steps + 1):
        input_ids, target_ids = generate_copying_task(
            batch_size, seq_len, vocab_size, num_tokens_to_copy
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

        final_train_accuracy = compute_accuracy(logits.detach(), target_ids)
        if step % eval_every == 0 or step == num_train_steps:
            accuracy_text = (
                f"{final_train_accuracy:.3f}"
                if final_train_accuracy is not None
                else "n/a"
            )
            print(
                f"Step {step:4d} | loss: {loss.item():.3f} "
                f"| train_acc: {accuracy_text}"
            )

    torch.manual_seed(100 if model_name == "MarkovMambaLM" else 101)
    heldout_input_ids, heldout_target_ids = generate_copying_task(
        256, seq_len, vocab_size, num_tokens_to_copy
    )
    model.eval()
    with torch.no_grad():
        heldout_logits = model(heldout_input_ids.to(device))["logits"]
        heldout_loss = F.cross_entropy(
            heldout_logits.reshape(-1, vocab_size),
            heldout_target_ids.to(device).reshape(-1),
            ignore_index=-100,
        )
        heldout_accuracy = compute_accuracy(
            heldout_logits, heldout_target_ids.to(device)
        )

    print("Final held-out evaluation")
    print(f"Held-out loss:     {heldout_loss.item():.6f}")
    print(f"Held-out accuracy: {heldout_accuracy:.6f}")
    return {
        "final_train_accuracy": final_train_accuracy,
        "heldout_accuracy": heldout_accuracy,
        "parameter_count": parameter_count,
    }


def main() -> None:
    markov_results = train_model(
        MarkovMambaLM(
            vocab_size=vocab_size,
            d_model=64,
            d_state=16,
            d_conv=4,
            expand=2,
            num_markov_states=8,
            num_layers=4,
        ),
        "MarkovMambaLM",
    )
    vanilla_results = train_model(
        VanillaMambaLM(
            vocab_size=vocab_size,
            d_model=66,
            d_state=16,
            d_conv=4,
            expand=2,
            num_layers=4,
        ),
        "VanillaMambaLM",
    )

    print("\nCopying-task comparison")
    print("-----------------------")
    print(
        f"{'Model':<18} | {'Final train acc':>15} | "
        f"{'Final held-out acc':>18} | {'Parameters':>10}"
    )
    print("-" * 72)
    for model_name, results in (
        ("MarkovMambaLM", markov_results),
        ("VanillaMambaLM", vanilla_results),
    ):
        print(
            f"{model_name:<18} | {results['final_train_accuracy']:>15.6f} | "
            f"{results['heldout_accuracy']:>18.6f} | "
            f"{results['parameter_count']:>10}"
        )


if __name__ == "__main__":
    main()
