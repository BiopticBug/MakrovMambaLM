"""Interpretability analysis for the delayed-copying task."""

import statistics
from typing import Iterable

import torch
from sklearn.metrics import normalized_mutual_info_score
from torch.nn import functional as F

from markov_mamba_model import MarkovMambaLM
from synthetic_tasks import generate_copying_task


SEED = 42
BATCH_SIZE = 32
ANALYSIS_BATCH_SIZE = 256
SEQ_LEN = 32
NUM_TOKENS_TO_COPY = 4
VOCAB_SIZE = 50
NUM_LAYERS = 4
NUM_MARKOV_STATES = 8
LEARNING_RATE = 1e-3
GRADIENT_CLIP_NORM = 1.0
MAX_TRAIN_STEPS = 15_000
EVAL_EVERY = 50
EARLY_STOP_STREAK = 5
EARLY_STOP_ACCURACY = 0.90
MIN_ANALYSIS_ACCURACY = 0.80
NUM_SHUFFLES = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    supervised = targets != -100
    predictions = logits.argmax(dim=-1)
    return float((predictions[supervised] == targets[supervised]).float().mean())


def generate_copying_phase_labels(
    batch_size: int, seq_len: int, num_tokens_to_copy: int
) -> torch.Tensor:
    """Build phase labels from the same task geometry used by generate_copying_task."""
    trigger_position = seq_len - num_tokens_to_copy - 1
    labels = torch.full((batch_size, seq_len), 1, dtype=torch.long)
    labels[:, :num_tokens_to_copy] = 0
    labels[:, trigger_position] = 2
    labels[:, trigger_position + 1 :] = 3
    return labels


def train_model() -> tuple[MarkovMambaLM, float, bool]:
    torch.manual_seed(SEED)
    model = MarkovMambaLM(
        vocab_size=VOCAB_SIZE,
        d_model=64,
        d_state=16,
        d_conv=4,
        expand=2,
        num_markov_states=NUM_MARKOV_STATES,
        num_layers=NUM_LAYERS,
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    perfect_streak = 0
    final_accuracy = 0.0
    reached_early_stop = False
    model.train()

    print(f"Device: {DEVICE}")
    print(
        f"Training: seq_len={SEQ_LEN}, num_tokens_to_copy={NUM_TOKENS_TO_COPY}, "
        f"steps={MAX_TRAIN_STEPS}, early_stop={EARLY_STOP_ACCURACY:.2f} x "
        f"{EARLY_STOP_STREAK} checkpoints"
    )

    for step in range(1, MAX_TRAIN_STEPS + 1):
        input_ids, target_ids = generate_copying_task(
            BATCH_SIZE, SEQ_LEN, VOCAB_SIZE, NUM_TOKENS_TO_COPY
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
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=GRADIENT_CLIP_NORM
        )
        optimizer.step()

        if step % EVAL_EVERY == 0 or step == MAX_TRAIN_STEPS:
            accuracy = compute_accuracy(logits.detach(), target_ids)
            final_accuracy = accuracy
            print(
                f"Step {step:5d} | loss: {loss.item():.3f} | train_acc: {accuracy:.3f}"
            )
            if accuracy >= EARLY_STOP_ACCURACY:
                perfect_streak += 1
            else:
                perfect_streak = 0
            if perfect_streak >= EARLY_STOP_STREAK:
                print(
                    f"EARLY STOP at step {step}: {EARLY_STOP_STREAK} consecutive checkpoints "
                    f"at or above {EARLY_STOP_ACCURACY:.2f} train_acc"
                )
                reached_early_stop = True
                break

    return model, final_accuracy, reached_early_stop


def contingency_table(states: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    table = torch.zeros(NUM_MARKOV_STATES, 4, dtype=torch.long)
    state_ids = states.reshape(-1).cpu()
    phase_ids = labels.reshape(-1).cpu()
    table.index_put_((state_ids, phase_ids), torch.ones_like(state_ids), accumulate=True)
    return table


def metrics_for_layer(states: torch.Tensor, labels: torch.Tensor) -> tuple[float, float, torch.Tensor]:
    flat_states = states.reshape(-1).cpu().numpy()
    flat_labels = labels.reshape(-1).cpu().numpy()
    nmi = float(normalized_mutual_info_score(flat_states, flat_labels))
    table = contingency_table(states, labels)
    purity = 0.0
    total = flat_states.size
    for state_index in range(NUM_MARKOV_STATES):
        row = table[state_index]
        phase_count = row.max().item()
        if phase_count > 0:
            purity += phase_count
    purity = float(purity / total) if total else 0.0
    return nmi, purity, table


def shuffled_metrics(states: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    flat_states = states.reshape(-1)
    shuffled = flat_states[torch.randperm(flat_states.numel())].reshape(states.shape)
    nmi, purity, _ = metrics_for_layer(shuffled, labels)
    return nmi, purity


def position_only_states(labels: torch.Tensor) -> torch.Tensor:
    """Deterministic position-only lookup with fixed bucket assignment by position alone."""
    if NUM_MARKOV_STATES < 4:
        raise ValueError("Position control requires at least four state buckets")
    seq_len = labels.shape[1]
    trigger_position = seq_len - NUM_TOKENS_TO_COPY - 1
    bucket_for_position = torch.full((seq_len,), 1, dtype=torch.long)
    bucket_for_position[:NUM_TOKENS_TO_COPY] = 0
    bucket_for_position[trigger_position] = 2
    bucket_for_position[trigger_position + 1 :] = 3
    buckets = bucket_for_position.unsqueeze(0).expand(labels.shape[0], -1)
    return buckets.clone()


def optimal_capacity_limited_position_states(labels: torch.Tensor) -> torch.Tensor:
    """Optimal deterministic 8-state quantizer of position alone.

    In this synthetic task, each absolute position deterministically belongs to a single
    phase (source, waiting, trigger, output). The optimal NMI-maximizing discrete code is
    therefore to collapse all positions sharing the same phase into the same bucket, while
    leaving the remaining capacity unused. This is the correct 8-state ceiling for a
    position-only encoding: it captures the maximum information that pure position can ever
    contribute under the same capacity limit as the trained Markov states, without invoking
    any content-dependent information. Because the task is deterministic by position, this
    greedy/phase-collapse construction reaches the global optimum for NMI.
    """
    seq_len = labels.shape[1]
    if seq_len == 0:
        return labels.clone()

    # Use the phase label observed at each position as the optimization target.
    # Since phase is deterministic by position, the best bucket assignment is to merge
    # positions with the same phase into the same bucket.
    position_phase = labels[0]
    bucket_for_position = torch.full((seq_len,), -1, dtype=torch.long)

    # Reserve four buckets for the four phases. The remaining four states are unused,
    # which is still globally optimal under the 8-state capacity ceiling.
    phase_to_bucket = {0: 0, 1: 1, 2: 2, 3: 3}
    for pos_idx in range(seq_len):
        bucket_for_position[pos_idx] = phase_to_bucket[int(position_phase[pos_idx].item())]

    buckets = bucket_for_position.unsqueeze(0).expand(labels.shape[0], -1)
    return buckets.clone()


def print_visualization(labels: torch.Tensor, layer_states: list[torch.Tensor]) -> None:
    sample_labels = labels[0].tolist()
    print("\nSingle-sequence visualization (sample 0)")
    print("True phase: " + " ".join(str(value) for value in sample_labels))
    for layer_index, states in enumerate(layer_states, 1):
        state_values = states[0].tolist()
        print(f"Layer {layer_index:2d} state: " + " ".join(str(value) for value in state_values))


def build_summary_table(
    real_metrics: list[tuple[float, float]],
    baseline_samples: list[list[tuple[float, float]]],
    position_nmi_values: list[float],
    optimal_position_nmi_values: list[float],
) -> None:
    print("\nSummary: shuffled baseline vs optimal 8-bucket position ceiling vs unconstrained position-only reference vs trained model")
    print("layer | shuffle baseline (mean +/- std) | optimal-8-bucket position ceiling | position-only NMI | real NMI | verdict")
    print("-" * 170)
    for layer_index, ((real_nmi, _), samples, position_nmi, optimal_nmi) in enumerate(
        zip(real_metrics, baseline_samples, position_nmi_values, optimal_position_nmi_values), 1
    ):
        nmi_values = [value[0] for value in samples]
        nmi_mean = statistics.mean(nmi_values)
        nmi_std = statistics.pstdev(nmi_values)
        if real_nmi > optimal_nmi * 1.15:
            verdict = "ALIGNMENT EXCEEDS PURE POSITION CEILING - evidence of content-dependent state organization beyond position alone"
        elif abs(real_nmi - optimal_nmi) <= 0.15 * max(optimal_nmi, 1e-9):
            verdict = "ALIGNMENT CONSISTENT WITH OPTIMAL POSITION ENCODING - no evidence of content-dependent information beyond capacity-limited position"
        elif real_nmi < optimal_nmi * 0.85:
            verdict = "STATES UNDERUTILIZE AVAILABLE CAPACITY - alignment is real but suboptimal, possibly noisy training or insufficient Markov temperature annealing"
        else:
            verdict = "ALIGNMENT CONSISTENT WITH OPTIMAL POSITION ENCODING - no evidence of content-dependent information beyond capacity-limited position"
        print(
            f"{layer_index:5d} | {nmi_mean:.6f} +/- {nmi_std:.6f} | "
            f"{optimal_nmi:.6f} | {position_nmi:.6f} | {real_nmi:.6f} | {verdict}"
        )


def main() -> None:
    print("MarkovMamba copying-task interpretability analysis")
    print(f"Phase labels: 0=source, 1=waiting, 2=trigger, 3=output; seq_len={SEQ_LEN}")

    model, final_train_accuracy, reached_early_stop = train_model()
    print(f"\nFinal train accuracy: {final_train_accuracy:.3f}")
    print(f"Training reached >= 0.80 train_acc: {final_train_accuracy >= MIN_ANALYSIS_ACCURACY}")
    if final_train_accuracy < MIN_ANALYSIS_ACCURACY:
        print(
            "WARNING: model failed to reach 0.80 train_acc within the training budget. "
            "STOPPING analysis; unsolved model states are not interpretable evidence."
        )
        return
    if reached_early_stop:
        print("Model converged under the 0.90 early-stop rule.")
    else:
        print("Training reached the 0.80 threshold without the 0.90 early-stop streak.")

    torch.manual_seed(SEED + 1)
    analysis_input_ids, analysis_target_ids = generate_copying_task(
        ANALYSIS_BATCH_SIZE, SEQ_LEN, VOCAB_SIZE, NUM_TOKENS_TO_COPY
    )
    phase_labels = generate_copying_phase_labels(
        ANALYSIS_BATCH_SIZE, SEQ_LEN, NUM_TOKENS_TO_COPY
    )
    model.eval()
    with torch.no_grad():
        result = model(analysis_input_ids.to(DEVICE))

    layer_states = [
        distribution.argmax(dim=-1).cpu()
        for distribution in result["all_state_distributions"]
    ]
    labels = phase_labels.cpu()
    positional_states = position_only_states(labels)
    optimal_position_states = optimal_capacity_limited_position_states(labels)
    position_nmi_values: list[float] = []
    optimal_position_nmi_values: list[float] = []
    position_purity_values: list[float] = []
    real_metrics: list[tuple[float, float]] = []
    baseline_samples: list[list[tuple[float, float]]] = [[] for _ in layer_states]

    print("\nFresh analysis batch: same copying template, new random content per sample")
    print("This is a fresh batch from the same generator, so it is the explicit content-randomized control for the final analysis.")

    print("\nPer-layer contingency tables (rows=state, columns=phase 0/1/2/3)")
    for layer_index, states in enumerate(layer_states, 1):
        mutual_information, purity, table = metrics_for_layer(states, labels)
        real_metrics.append((mutual_information, purity))
        position_nmi, position_purity, _ = metrics_for_layer(positional_states, labels)
        position_nmi_values.append(position_nmi)
        optimal_position_nmi, _, _ = metrics_for_layer(optimal_position_states, labels)
        optimal_position_nmi_values.append(optimal_position_nmi)
        position_purity_values.append(position_purity)
        print(f"\nLayer {layer_index}")
        print("state | source waiting trigger output")
        for state_index, row in enumerate(table.tolist()):
            print(f"{state_index:5d} | " + " ".join(f"{value:7d}" for value in row))

    torch.manual_seed(SEED + 2)
    for _ in range(NUM_SHUFFLES):
        for layer_index, states in enumerate(layer_states):
            baseline_samples[layer_index].append(shuffled_metrics(states, labels))

    print_visualization(labels, layer_states)
    build_summary_table(real_metrics, baseline_samples, position_nmi_values, optimal_position_nmi_values)

    print("\nPosition-only control details")
    for layer_index, position_nmi in enumerate(position_nmi_values, 1):
        print(f"Layer {layer_index}: position-only NMI = {position_nmi:.6f}")
    print("\nOptimal 8-bucket position-quantizer ceiling")
    for layer_index, optimal_nmi in enumerate(optimal_position_nmi_values, 1):
        print(f"Layer {layer_index}: optimal 8-bucket position NMI = {optimal_nmi:.6f}")

    print("\nLayer verdicts")
    for layer_index, ((real_nmi, _), samples, position_nmi, optimal_nmi) in enumerate(
        zip(real_metrics, baseline_samples, position_nmi_values, optimal_position_nmi_values), 1
    ):
        nmi_values = [value[0] for value in samples]
        nmi_mean = statistics.mean(nmi_values)
        nmi_std = statistics.pstdev(nmi_values)
        if real_nmi > optimal_nmi * 1.15:
            verdict = "ALIGNMENT EXCEEDS PURE POSITION CEILING - evidence of content-dependent state organization beyond position alone"
        elif abs(real_nmi - optimal_nmi) <= 0.15 * max(optimal_nmi, 1e-9):
            verdict = "ALIGNMENT CONSISTENT WITH OPTIMAL POSITION ENCODING - no evidence of content-dependent information beyond capacity-limited position"
        elif real_nmi < optimal_nmi * 0.85:
            verdict = "STATES UNDERUTILIZE AVAILABLE CAPACITY - alignment is real but suboptimal, possibly noisy training or insufficient Markov temperature annealing"
        else:
            verdict = "ALIGNMENT CONSISTENT WITH OPTIMAL POSITION ENCODING - no evidence of content-dependent information beyond capacity-limited position"
        print(
            f"Layer {layer_index}: {verdict} "
            f"(real NMI={real_nmi:.6f}, optimal 8-bucket position NMI={optimal_nmi:.6f}, "
            f"position-only NMI={position_nmi:.6f}, shuffle mean={nmi_mean:.6f} +/- {nmi_std:.6f})"
        )


if __name__ == "__main__":
    main()
