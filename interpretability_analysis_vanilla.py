"""Interpretability analysis for the vanilla Mamba baseline on the delayed-copying task.

This mirrors the discrete-state analysis in interpretability_analysis_copying.py, but
replaces the Markov state distributions with cluster assignments computed from the
continuous hidden state output of each Mamba block. We use the pre-residual block output
(the value returned by ``mamba_block(norm(hidden))`` before adding the residual), because
that is the simplest internal representation available in the existing VanillaMambaLM
forward pass without changing the model implementation itself.
"""

import statistics

import torch
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from torch.nn import functional as F

from synthetic_tasks import generate_copying_task
from vanilla_mamba_model import VanillaMambaLM

SEED = 42
BATCH_SIZE = 32
ANALYSIS_BATCH_SIZE = 500
SEQ_LEN = 32
NUM_TOKENS_TO_COPY = 4
VOCAB_SIZE = 50
NUM_LAYERS = 4
NUM_CLUSTERS = 8
LEARNING_RATE = 1e-3
GRADIENT_CLIP_NORM = 1.0
MAX_TRAIN_STEPS = 15_000
EVAL_EVERY = 50
EARLY_STOP_STREAK = 5
EARLY_STOP_ACCURACY = 0.90
MIN_ANALYSIS_ACCURACY = 0.80
NUM_SHUFFLES = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MARKOV_REFERENCE_NMI = {
    1: 0.294749,
    2: 0.373203,
    3: 0.552615,
    4: 0.209445,
}


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    supervised = targets != -100
    predictions = logits.argmax(dim=-1)
    return float((predictions[supervised] == targets[supervised]).float().mean())


def generate_copying_phase_labels(
    batch_size: int, seq_len: int, num_tokens_to_copy: int
) -> torch.Tensor:
    """Build phase labels from the same task geometry used by the copying task."""
    trigger_position = seq_len - num_tokens_to_copy - 1
    labels = torch.full((batch_size, seq_len), 1, dtype=torch.long)
    labels[:, :num_tokens_to_copy] = 0
    labels[:, trigger_position] = 2
    labels[:, trigger_position + 1 :] = 3
    return labels


def train_model() -> tuple[VanillaMambaLM, float, bool]:
    torch.manual_seed(SEED)
    model = VanillaMambaLM(
        vocab_size=VOCAB_SIZE,
        d_model=64,
        d_state=16,
        d_conv=4,
        expand=2,
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


def collect_layer_hidden_states(
    model: VanillaMambaLM, input_ids: torch.Tensor
) -> list[torch.Tensor]:
    """Return the pre-residual output of each Mamba block for each layer."""
    hidden = model.token_embedding(input_ids)
    if model.position_embedding is not None:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        hidden = hidden + model.position_embedding(positions).unsqueeze(0)

    layer_outputs: list[torch.Tensor] = []
    for norm, block in zip(model.norms, model.layers):
        block_output = block(norm(hidden))
        layer_outputs.append(block_output.detach().cpu())
        hidden = hidden + block_output
    return layer_outputs


def cluster_layer_states(layer_hidden: torch.Tensor) -> torch.Tensor:
    """K-means discretize a layer's continuous hidden states into 8 buckets."""
    batch_size, seq_len, hidden_dim = layer_hidden.shape
    flat_hidden = layer_hidden.reshape(-1, hidden_dim).numpy()
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=0, n_init=20, max_iter=500)
    cluster_ids = kmeans.fit_predict(flat_hidden)
    return torch.tensor(cluster_ids.reshape(batch_size, seq_len), dtype=torch.long)


def clustered_nmi_for_layer(states: torch.Tensor, labels: torch.Tensor) -> float:
    flat_states = states.reshape(-1).numpy()
    flat_labels = labels.reshape(-1).numpy()
    return float(normalized_mutual_info_score(flat_states, flat_labels))


def shuffled_clustered_metrics(states: torch.Tensor, labels: torch.Tensor) -> float:
    flat_states = states.reshape(-1)
    shuffled = flat_states[torch.randperm(flat_states.numel())].reshape(states.shape)
    return clustered_nmi_for_layer(shuffled, labels)


def print_summary_table(
    vanilla_nmi_values: list[float],
    shuffle_means: list[float],
    shuffle_stds: list[float],
) -> None:
    print("\nSummary: shuffled baseline vs vanilla Mamba clustered hidden states vs MarkovMambaLM discrete states")
    print("layer | shuffle NMI mean +/- std | vanilla clustered-state NMI | MarkovMambaLM NMI | verdict")
    print("-" * 170)
    for layer_index, (vanilla_nmi, shuffle_mean, shuffle_std) in enumerate(
        zip(vanilla_nmi_values, shuffle_means, shuffle_stds), 1
    ):
        markov_nmi = MARKOV_REFERENCE_NMI[layer_index]
        if vanilla_nmi >= markov_nmi:
            verdict = "VANILLA MAMBA SHOWS EQUIVALENT OR BETTER ALIGNMENT - discrete states add accessibility, not new information"
        else:
            verdict = "MARKOV STATES SHOW STRONGER ALIGNMENT - discrete mechanism may be surfacing/creating information vanilla Mamba doesn't organize as clearly"
        print(
            f"{layer_index:5d} | {shuffle_mean:.6f} +/- {shuffle_std:.6f} | "
            f"{vanilla_nmi:.6f} | {markov_nmi:.6f} | {verdict}"
        )


def main() -> None:
    print("Vanilla Mamba copying-task interpretability analysis")
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
    analysis_input_ids, _ = generate_copying_task(
        ANALYSIS_BATCH_SIZE, SEQ_LEN, VOCAB_SIZE, NUM_TOKENS_TO_COPY
    )
    phase_labels = generate_copying_phase_labels(
        ANALYSIS_BATCH_SIZE, SEQ_LEN, NUM_TOKENS_TO_COPY
    )
    analysis_input_ids = analysis_input_ids.to(DEVICE)
    model.eval()

    with torch.no_grad():
        layer_hidden = collect_layer_hidden_states(model, analysis_input_ids)

    vanilla_nmi_values: list[float] = []
    shuffle_means: list[float] = []
    shuffle_stds: list[float] = []

    print("\nPer-layer NMI on clustered continuous hidden states")
    for layer_index, hidden in enumerate(layer_hidden, 1):
        clustered_states = cluster_layer_states(hidden)
        nmi = clustered_nmi_for_layer(clustered_states, phase_labels)
        vanilla_nmi_values.append(nmi)
        print(f"Layer {layer_index}: clustered-state NMI = {nmi:.6f}")

        shuffled_values = [
            shuffled_clustered_metrics(clustered_states, phase_labels)
            for _ in range(NUM_SHUFFLES)
        ]
        shuffle_means.append(statistics.mean(shuffled_values))
        shuffle_stds.append(statistics.pstdev(shuffled_values))

    print_summary_table(vanilla_nmi_values, shuffle_means, shuffle_stds)

    print("\nLayer verdicts")
    for layer_index, vanilla_nmi in enumerate(vanilla_nmi_values, 1):
        markov_nmi = MARKOV_REFERENCE_NMI[layer_index]
        if vanilla_nmi >= markov_nmi:
            verdict = "VANILLA MAMBA SHOWS EQUIVALENT OR BETTER ALIGNMENT - discrete states add accessibility, not new information"
        else:
            verdict = "MARKOV STATES SHOW STRONGER ALIGNMENT - discrete mechanism may be surfacing/creating information vanilla Mamba doesn't organize as clearly"
        print(
            f"Layer {layer_index}: {verdict} "
            f"(vanilla clustered-state NMI={vanilla_nmi:.6f}, markov NMI={markov_nmi:.6f}, "
            f"shuffle mean={shuffle_means[layer_index - 1]:.6f} +/- {shuffle_stds[layer_index - 1]:.6f})"
        )


if __name__ == "__main__":
    main()
