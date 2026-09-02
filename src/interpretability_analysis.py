"""Analyze whether Markov states align with associative-recall task phases."""

import statistics
import torch
from sklearn.metrics import normalized_mutual_info_score
from torch.nn import functional as F

from markov_mamba_model import MarkovMambaLM
from synthetic_tasks import generate_associative_recall_task_with_labels


SEED = 42
BATCH_SIZE = 32
ANALYSIS_BATCH_SIZE = 500
SEQ_LEN = 32
NUM_PAIRS = 3
VOCAB_SIZE = 50
NUM_LAYERS = 4
NUM_MARKOV_STATES = 8
LEARNING_RATE = 1e-3
GRADIENT_CLIP_NORM = 1.0
NUM_TRAIN_STEPS = 40_000
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
        f"Training: seq_len={SEQ_LEN}, num_pairs={NUM_PAIRS}, "
        f"steps={NUM_TRAIN_STEPS}, early_stop={EARLY_STOP_ACCURACY:.2f} x "
        f"{EARLY_STOP_STREAK} checkpoints"
    )

    for step in range(1, NUM_TRAIN_STEPS + 1):
        input_ids, targets, _ = generate_associative_recall_task_with_labels(
            BATCH_SIZE, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS
        )
        input_ids = input_ids.to(DEVICE)
        targets = targets.to(DEVICE)
        logits = model(input_ids)["logits"]
        loss = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE),
            targets.reshape(-1),
            ignore_index=-100,
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=GRADIENT_CLIP_NORM
        )
        optimizer.step()

        if step % EVAL_EVERY == 0 or step == NUM_TRAIN_STEPS:
            accuracy = compute_accuracy(logits.detach(), targets)
            final_accuracy = accuracy
            print(f"Step {step:5d} | loss: {loss.item():.3f} | train_acc: {accuracy:.3f}")
            perfect_streak = perfect_streak + 1 if accuracy >= EARLY_STOP_ACCURACY else 0
            if perfect_streak >= EARLY_STOP_STREAK:
                print(f"EARLY STOP at step {step}: {EARLY_STOP_STREAK} consecutive checkpoints")
                reached_early_stop = True
                break
    return model, final_accuracy, reached_early_stop


def weighted_purity(states: torch.Tensor, labels: torch.Tensor) -> float:
    """Return state-frequency-weighted purity, including unused states as zero."""
    state_ids = states.reshape(-1)
    phase_ids = labels.reshape(-1)
    counts = torch.zeros(NUM_MARKOV_STATES, 4, dtype=torch.long)
    counts.index_put_((state_ids.cpu(), phase_ids.cpu()), torch.ones_like(state_ids.cpu()), accumulate=True)
    total = counts.sum().item()
    return float(counts.max(dim=1).values.sum().item() / total) if total else 0.0


def contingency_table(states: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    table = torch.zeros(NUM_MARKOV_STATES, 4, dtype=torch.long)
    state_ids = states.reshape(-1).cpu()
    phase_ids = labels.reshape(-1).cpu()
    table.index_put_((state_ids, phase_ids), torch.ones_like(state_ids), accumulate=True)
    return table


def metrics_for_layer(states: torch.Tensor, labels: torch.Tensor) -> tuple[float, float, torch.Tensor]:
    flat_states = states.reshape(-1).cpu().numpy()
    flat_labels = labels.reshape(-1).cpu().numpy()
    mutual_information = float(normalized_mutual_info_score(flat_states, flat_labels))
    return mutual_information, weighted_purity(states, labels), contingency_table(states, labels)


def shuffled_metrics(states: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    shuffled = states.reshape(-1).clone()
    shuffled = shuffled[torch.randperm(shuffled.numel())].reshape(states.shape)
    mutual_information, purity, _ = metrics_for_layer(shuffled, labels)
    return mutual_information, purity


def position_only_states(labels: torch.Tensor) -> torch.Tensor:
    """Map each absolute position to a fixed state bucket, ignoring content.

    The task phase is deterministic by position, so mapping phase ids 0 through
    3 to the first four state buckets gives the maximum position-only alignment.
    The same lookup is used for every sample and layer; extra state buckets are
    intentionally unused.
    """
    if NUM_MARKOV_STATES < 4:
        raise ValueError("Position control requires at least four state buckets")
    return labels.clamp_max(NUM_MARKOV_STATES - 1).clone()


def print_visualization(labels: torch.Tensor, layer_states: list[torch.Tensor]) -> None:
    sample_labels = labels[0].tolist()
    print("\nSingle-sequence visualization (sample 0)")
    print("True phase: " + " ".join(str(value) for value in sample_labels))
    for layer_index, states in enumerate(layer_states, 1):
        print(f"Layer {layer_index:2d} state: " + " ".join(str(value) for value in states[0].tolist()))


def main() -> None:
    print("MarkovMamba associative-recall phase interpretability analysis")
    print(f"Phase labels: 0=encoding, 1=waiting, 2=query, 3=retrieval; seq_len={SEQ_LEN}")
    model, final_train_accuracy, reached_early_stop = train_model()
    print(f"\nTraining reached >= 0.80 train_acc: {final_train_accuracy >= MIN_ANALYSIS_ACCURACY}")
    if final_train_accuracy < MIN_ANALYSIS_ACCURACY:
        print(
            "WARNING: model failed to reach 0.80 train_acc within the budget. "
            "STOPPING analysis; unsolved model states are not interpretable evidence."
        )
        return
    if not reached_early_stop:
        print("Training reached the analysis threshold without the 0.90 early-stop streak.")

    # This is a fresh batch: new key-value pairs share the same structural
    # template but randomize content independently of every training batch.
    torch.manual_seed(SEED + 1)
    input_ids, _, phase_labels = generate_associative_recall_task_with_labels(
        ANALYSIS_BATCH_SIZE, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS
    )
    model.eval()
    with torch.no_grad():
        result = model(input_ids.to(DEVICE))
    labels = phase_labels.cpu()
    layer_states = [distribution.argmax(dim=-1).cpu() for distribution in result["all_state_distributions"]]
    positional_states = position_only_states(labels)
    position_nmi, position_purity, _ = metrics_for_layer(positional_states, labels)
    print("Fresh analysis batch: randomized key-value content with identical positions/structure")

    print("\nPer-layer contingency tables (rows=state, columns=phase 0/1/2/3)")
    real_metrics: list[tuple[float, float]] = []
    for layer_index, states in enumerate(layer_states, 1):
        mutual_information, purity, table = metrics_for_layer(states, labels)
        real_metrics.append((mutual_information, purity))
        print(f"\nLayer {layer_index}")
        print("state | encoding waiting query retrieval")
        for state_index, row in enumerate(table.tolist()):
            print(f"{state_index:5d} | " + " ".join(f"{value:8d}" for value in row))

    print_visualization(labels, layer_states)
    baseline_metrics: list[list[tuple[float, float]]] = [[] for _ in layer_states]
    torch.manual_seed(SEED + 2)
    for _ in range(NUM_SHUFFLES):
        for layer_index, states in enumerate(layer_states):
            baseline_metrics[layer_index].append(shuffled_metrics(states, labels))

    print("\nSummary: shuffle baseline vs position-only control vs trained model")
    print("layer | shuffle NMI mean +/- std | position-only NMI | real NMI | real purity")
    for layer_index, ((real_nmi, real_purity), samples) in enumerate(zip(real_metrics, baseline_metrics), 1):
        nmi_values = [sample[0] for sample in samples]
        purity_values = [sample[1] for sample in samples]
        nmi_mean, nmi_std = statistics.mean(nmi_values), statistics.pstdev(nmi_values)
        purity_mean, purity_std = statistics.mean(purity_values), statistics.pstdev(purity_values)
        close_to_position = real_nmi >= position_nmi * 0.90
        above_position = real_nmi > position_nmi + 2 * nmi_std
        close_to_shuffle = real_nmi <= nmi_mean + 2 * nmi_std
        if close_to_position:
            verdict = "ALIGNMENT EXPLAINED BY POSITION ALONE - not evidence of content-dependent interpretability"
        elif above_position:
            verdict = "ALIGNMENT EXCEEDS POSITIONAL BASELINE - evidence of content-dependent state organization"
        elif close_to_shuffle:
            verdict = "NO MEANINGFUL ALIGNMENT DETECTED"
        else:
            verdict = "NO MEANINGFUL ALIGNMENT DETECTED"
        print(f"{layer_index:5d} | {nmi_mean:.6f} +/- {nmi_std:.6f} | {position_nmi:.6f} | {real_nmi:.6f} | {real_purity:.6f}")
        print(f"        shuffle purity: {purity_mean:.6f} +/- {purity_std:.6f}; position-only purity: {position_purity:.6f}")
        print(f"Layer {layer_index} verdict: {verdict}")


if __name__ == "__main__":
    main()