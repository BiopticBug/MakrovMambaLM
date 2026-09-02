"""Long-run copying experiment on vanilla Mamba for phase-transition comparison."""

import json
import time

import torch
from torch.nn import functional as F

from synthetic_tasks import generate_copying_task
from vanilla_mamba_model import VanillaMambaLM


torch.manual_seed(42)

vocab_size = 50
seq_len = 32
num_tokens_to_copy = 4
d_model = 66
num_layers = 4
batch_size = 32
num_train_steps = 15000
eval_every = 50
learning_rate = 1e-3
gradient_clip_norm = 1.0
benchmark_steps = 20
device = "cuda" if torch.cuda.is_available() else "cpu"
history_path = "long_run_copying_history_vanilla.json"


def compute_accuracy(logits: torch.Tensor, target_ids: torch.Tensor) -> float:
    """Compute copying accuracy at aligned, non-masked target positions."""
    supervised = target_ids != -100
    predictions = logits.argmax(dim=-1)
    return float((predictions[supervised] == target_ids[supervised]).float().mean())


def synchronize_device() -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def run_step(
    model: VanillaMambaLM,
    optimizer: torch.optim.Optimizer,
) -> tuple[float, float]:
    """Run one fresh-batch optimization step and return loss and accuracy."""
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
    return loss.item(), compute_accuracy(logits.detach(), target_ids)


def benchmark_step_time() -> float:
    """Measure average optimization-step time on the actual experiment setup."""
    benchmark_model = VanillaMambaLM(
        vocab_size=vocab_size,
        d_model=d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        num_layers=num_layers,
    ).to(device)
    benchmark_optimizer = torch.optim.AdamW(
        benchmark_model.parameters(), lr=learning_rate
    )
    benchmark_model.train()
    synchronize_device()
    start = time.perf_counter()
    for _ in range(benchmark_steps):
        run_step(benchmark_model, benchmark_optimizer)
    synchronize_device()
    return (time.perf_counter() - start) / benchmark_steps


def detect_phase_transition(history: list[dict[str, float | int]]) -> int | None:
    """Find a rapid logged rise from below 0.1 to above 0.5.

    A transition can span several logging intervals, so inspect a short
    500-step window rather than requiring one adjacent pair to cross both
    thresholds.
    """
    window_size = 500
    for current_index, current in enumerate(history):
        if current["accuracy"] <= 0.5:
            continue
        window_start = max(0, current_index - window_size // eval_every)
        if any(
            history[index]["accuracy"] < 0.1
            for index in range(window_start, current_index)
        ):
            return int(current["step"])
    return None


def main() -> None:
    step_seconds = benchmark_step_time()
    estimated_seconds = step_seconds * num_train_steps
    print(f"Device: {device}")
    print(
        f"Benchmark: {benchmark_steps} steps at "
        f"{step_seconds:.4f} seconds/step"
    )
    print(
        f"Estimated runtime for {num_train_steps} steps: "
        f"{estimated_seconds:.1f} seconds ({estimated_seconds / 60:.1f} minutes)"
    )

    # Reset both initialization and data RNG streams after timing calibration.
    torch.manual_seed(42)
    model = VanillaMambaLM(
        vocab_size=vocab_size,
        d_model=d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        num_layers=num_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()

    print("\nLong-run copying training (VanillaMambaLM)")
    print(
        f"Configuration: seq_len={seq_len}, num_tokens_to_copy={num_tokens_to_copy}, "
        f"batch_size={batch_size}, steps={num_train_steps}, "
        f"learning_rate={learning_rate}, clip_norm={gradient_clip_norm}"
    )
    history: list[dict[str, float | int]] = []
    for step in range(1, num_train_steps + 1):
        loss, accuracy = run_step(model, optimizer)
        if step % eval_every == 0 or step == num_train_steps:
            record = {"step": step, "loss": loss, "accuracy": accuracy}
            history.append(record)
            print(
                f"Step {step:5d} | loss: {loss:.3f} "
                f"| train_acc: {accuracy:.3f}"
            )

    torch.manual_seed(43)
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

    with open(history_path, "w", encoding="utf-8") as history_file:
        json.dump(history, history_file, indent=2)

    transition_step = detect_phase_transition(history)
    print("\nPhase-transition analysis")
    if transition_step is None:
        print("NO PHASE TRANSITION OBSERVED within 15000 steps.")
    else:
        print(f"PHASE TRANSITION DETECTED at approximately step {transition_step}")

    print("\nFinal held-out evaluation")
    print(f"Held-out loss:     {heldout_loss.item():.6f}")
    print(f"Held-out accuracy: {heldout_accuracy:.6f}")
    print(f"History saved to:  {history_path}")


if __name__ == "__main__":
    main()
