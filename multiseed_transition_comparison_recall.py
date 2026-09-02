"""Multi-seed phase-transition comparison for associative recall using Markov vs vanilla Mamba."""

import json
import statistics
import time
from typing import Any

import torch
from torch.nn import functional as F

from markov_mamba_model import MarkovMambaLM
from synthetic_tasks import generate_associative_recall_task
from vanilla_mamba_model import VanillaMambaLM


SEEDS = [0, 1, 2, 3, 4]
BATCH_SIZE = 32
SEQ_LEN = 32
NUM_PAIRS = 3
VOCAB_SIZE = 50
LEARNING_RATE = 1e-3
GRADIENT_CLIP_NORM = 1.0
NUM_TRAIN_STEPS = 25000
EVAL_EVERY = 50
EARLY_STOP_STREAK = 5
EARLY_STOP_STEPS = EARLY_STOP_STREAK * EVAL_EVERY
PER_STEP_SECONDS_LOW = 0.22
PER_STEP_SECONDS_HIGH = 0.27
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_PATH = "multiseed_transition_results_recall.json"


def compute_accuracy(logits: torch.Tensor, target_ids: torch.Tensor) -> float:
    """Compute recall accuracy at aligned, non-masked target positions."""
    supervised = target_ids != -100
    predictions = logits.argmax(dim=-1)
    return float((predictions[supervised] == target_ids[supervised]).float().mean())


def synchronize_device() -> None:
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def detect_phase_transition(history: list[dict[str, float | int]]) -> int | None:
    """Find a rapid logged rise from below 0.1 to above 0.5."""
    window_size = 500
    for current_index, current in enumerate(history):
        if current["accuracy"] <= 0.5:
            continue
        window_start = max(0, current_index - window_size // EVAL_EVERY)
        if any(
            history[index]["accuracy"] < 0.1
            for index in range(window_start, current_index)
        ):
            return int(current["step"])
    return None


def run_step(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> tuple[float, float]:
    """Run one fresh-batch optimization step and return loss and accuracy."""
    input_ids, target_ids = generate_associative_recall_task(
        BATCH_SIZE, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS
    )
    input_ids = input_ids.to(DEVICE)
    target_ids = target_ids.to(DEVICE)

    logits = model(input_ids)["logits"]
    loss = F.cross_entropy(
        logits.reshape(-1, VOCAB_SIZE),
        target_ids.reshape(-1),
        ignore_index=-100,
    )
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_NORM)
    optimizer.step()
    return loss.item(), compute_accuracy(logits.detach(), target_ids)


def run_single_model_experiment(
    model_name: str,
    model_cls: type[torch.nn.Module],
    seed: int,
) -> dict[str, Any]:
    """Train one model on the associative recall task and return its transition estimate."""
    print(f"\n=== {model_name} seed {seed} ===")
    torch.manual_seed(seed)
    if model_name == "MarkovMamba":
        model = model_cls(
            vocab_size=VOCAB_SIZE,
            d_model=64,
            d_state=16,
            d_conv=4,
            expand=2,
            num_markov_states=8,
            num_layers=4,
        ).to(DEVICE)
    else:
        model = model_cls(
            vocab_size=VOCAB_SIZE,
            d_model=66,
            d_state=16,
            d_conv=4,
            expand=2,
            num_layers=4,
        ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    model.train()

    torch.manual_seed(seed)
    history: list[dict[str, float | int]] = []
    perfect_streak = 0
    transition_step: int | None = None
    transition_reason = "NO TRANSITION"
    start = time.perf_counter()

    for step in range(1, NUM_TRAIN_STEPS + 1):
        loss, accuracy = run_step(model, optimizer)
        if step % EVAL_EVERY == 0 or step == NUM_TRAIN_STEPS:
            record = {"step": step, "loss": loss, "accuracy": accuracy}
            history.append(record)
            print(f"Step {step:5d} | loss: {loss:.3f} | train_acc: {accuracy:.3f}")

            if accuracy >= 0.99:
                perfect_streak += 1
            else:
                perfect_streak = 0

            if perfect_streak >= EARLY_STOP_STREAK and transition_step is None:
                transition_step = step
                transition_reason = "EARLY_STOP_PERFECT_STREAK"
                print(
                    f"EARLY STOP at step {step}: {EARLY_STOP_STREAK} consecutive "
                    f"checkpoints >= 0.99 (transition step {transition_step})"
                )
                break

    if transition_step is None:
        transition_step = detect_phase_transition(history)
        if transition_step is not None:
            transition_reason = "PHASE_DETECTION"

    if transition_step is None:
        transition_reason = "NO TRANSITION"

    torch.manual_seed(seed + 100)
    heldout_input_ids, heldout_target_ids = generate_associative_recall_task(
        256, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS
    )
    model.eval()
    with torch.no_grad():
        heldout_logits = model(heldout_input_ids.to(DEVICE))["logits"]
        heldout_loss = F.cross_entropy(
            heldout_logits.reshape(-1, VOCAB_SIZE),
            heldout_target_ids.to(DEVICE).reshape(-1),
            ignore_index=-100,
        )
        heldout_accuracy = compute_accuracy(
            heldout_logits, heldout_target_ids.to(DEVICE)
        )

    elapsed = time.perf_counter() - start
    result = {
        "seed": seed,
        "model": model_name,
        "transition_step": transition_step,
        "transition_reason": transition_reason,
        "heldout_loss": float(heldout_loss.item()),
        "heldout_accuracy": float(heldout_accuracy),
        "elapsed_seconds": elapsed,
        "history": history,
    }
    print(
        f"Transition step: {transition_step if transition_step is not None else 'NO TRANSITION'}"
    )
    print(f"Held-out loss:     {heldout_loss.item():.6f}")
    print(f"Held-out accuracy: {heldout_accuracy:.6f}")
    print(f"Elapsed time:      {elapsed:.1f} seconds")
    return result


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute mean/std summary statistics for transition steps."""
    summary: dict[str, Any] = {}
    for model_name in ["MarkovMamba", "VanillaMamba"]:
        steps = [
            entry["transition_step"]
            for entry in results
            if entry["model"] == model_name and entry["transition_step"] is not None
        ]
        if not steps:
            summary[model_name] = {
                "mean": None,
                "std": None,
                "count": 0,
            }
            continue
        summary[model_name] = {
            "mean": float(statistics.mean(steps)),
            "std": float(statistics.pstdev(steps)) if len(steps) > 1 else 0.0,
            "count": len(steps),
        }

    diffs = []
    for seed in SEEDS:
        markov = next(
            (entry["transition_step"] for entry in results if entry["model"] == "MarkovMamba" and entry["seed"] == seed),
            None,
        )
        vanilla = next(
            (entry["transition_step"] for entry in results if entry["model"] == "VanillaMamba" and entry["seed"] == seed),
            None,
        )
        if markov is not None and vanilla is not None:
            diffs.append(markov - vanilla)

    summary["difference"] = {
        "mean": float(statistics.mean(diffs)) if diffs else None,
        "std": float(statistics.pstdev(diffs)) if len(diffs) > 1 else 0.0 if diffs else None,
        "count": len(diffs),
    }
    return summary


def main() -> None:
    total_runs = len(SEEDS) * 2
    low_bound_seconds = total_runs * NUM_TRAIN_STEPS * PER_STEP_SECONDS_LOW
    high_bound_seconds = total_runs * NUM_TRAIN_STEPS * PER_STEP_SECONDS_HIGH
    print("Multi-seed associative recall phase-transition comparison")
    print(f"Seeds: {SEEDS}")
    print(f"Runs: {total_runs} total (5 seeds × 2 models)")
    print(f"Task: generate_associative_recall_task with seq_len={SEQ_LEN}, num_pairs={NUM_PAIRS}, vocab_size={VOCAB_SIZE}")
    print(f"Step budget per run: {NUM_TRAIN_STEPS}, eval_every={EVAL_EVERY}, early_stop_streak={EARLY_STOP_STREAK}")
    print(f"Per-step timing observed previously: {PER_STEP_SECONDS_LOW:.2f}-{PER_STEP_SECONDS_HIGH:.2f} sec/step")
    print(
        f"Estimated total runtime upper bound: "
        f"{low_bound_seconds:.1f}-{high_bound_seconds:.1f} seconds "
        f"({(low_bound_seconds/60):.1f}-{(high_bound_seconds/60):.1f} minutes, "
        f"{(low_bound_seconds/3600):.2f}-{(high_bound_seconds/3600):.2f} hours)"
    )
    print("Early stopping may substantially shorten the total runtime if a model reaches the perfect-accuracy plateau early.")

    all_results: list[dict[str, Any]] = []
    for seed in SEEDS:
        markov_result = run_single_model_experiment("MarkovMamba", MarkovMambaLM, seed)
        vanilla_result = run_single_model_experiment("VanillaMamba", VanillaMambaLM, seed)
        all_results.extend([markov_result, vanilla_result])

    print("\nSummary table")
    print("seed | MarkovMamba transition | VanillaMamba transition | difference")
    print("-----|-----------------------|------------------------|----------")
    for seed in SEEDS:
        markov = next(
            (entry["transition_step"] for entry in all_results if entry["model"] == "MarkovMamba" and entry["seed"] == seed),
            None,
        )
        vanilla = next(
            (entry["transition_step"] for entry in all_results if entry["model"] == "VanillaMamba" and entry["seed"] == seed),
            None,
        )
        diff = None if markov is None or vanilla is None else markov - vanilla
        print(
            f"{seed:4d} | {markov if markov is not None else 'NO TRANSITION':>20} | "
            f"{vanilla if vanilla is not None else 'NO TRANSITION':>20} | "
            f"{diff if diff is not None else 'N/A':>10}"
        )

    summary = summarize_results(all_results)
    print("\nAggregate statistics")
    print(f"MarkovMamba mean transition step: {summary['MarkovMamba']['mean'] if summary['MarkovMamba']['mean'] is not None else 'N/A'}")
    print(f"MarkovMamba std transition step: {summary['MarkovMamba']['std'] if summary['MarkovMamba']['std'] is not None else 'N/A'}")
    print(f"VanillaMamba mean transition step: {summary['VanillaMamba']['mean'] if summary['VanillaMamba']['mean'] is not None else 'N/A'}")
    print(f"VanillaMamba std transition step: {summary['VanillaMamba']['std'] if summary['VanillaMamba']['std'] is not None else 'N/A'}")
    print(f"Mean difference (Markov - Vanilla): {summary['difference']['mean'] if summary['difference']['mean'] is not None else 'N/A'}")
    print(f"Difference std: {summary['difference']['std'] if summary['difference']['std'] is not None else 'N/A'}")

    with open(RESULTS_PATH, "w", encoding="utf-8") as outfile:
        json.dump(all_results, outfile, indent=2)

    print(f"\nAll raw results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()