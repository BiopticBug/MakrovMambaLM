"""Check whether MarkovMambaLM can memorize one fixed recall batch."""

import torch
from torch.nn import functional as F

from markov_mamba_model import MarkovMambaLM
from synthetic_tasks import generate_associative_recall_task


vocab_size = 50
seq_len = 16
num_pairs = 1
d_model = 64
num_layers = 4
num_markov_states = 8
batch_size = 32
num_train_steps = 1000
log_every = 50
learning_rate = 1e-3
gradient_clip_norm = 1.0
device = "cuda" if torch.cuda.is_available() else "cpu"


def compute_accuracy(logits: torch.Tensor, target_ids: torch.Tensor) -> float:
    """Compute accuracy at supervised positions without shifting either tensor."""
    supervised = target_ids != -100
    predictions = logits.argmax(dim=-1)
    return float((predictions[supervised] == target_ids[supervised]).float().mean())


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

    # Seed immediately before generation so this exact fixed batch is reproducible.
    torch.manual_seed(42)
    input_ids, target_ids = generate_associative_recall_task(
        batch_size=batch_size,
        seq_len=seq_len,
        vocab_size=vocab_size,
        num_pairs=num_pairs,
    )
    input_ids = input_ids.to(device)
    target_ids = target_ids.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    print(f"Device: {device}")
    print("Overfitting one fixed associative-recall batch")
    print(
        f"Configuration: batch_size={batch_size}, seq_len={seq_len}, "
        f"num_pairs={num_pairs}, steps={num_train_steps}, "
        f"learning_rate={learning_rate}, clip_norm={gradient_clip_norm}"
    )

    model.train()
    for step in range(1, num_train_steps + 1):
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

        if step % log_every == 0 or step == 1:
            train_accuracy = compute_accuracy(logits.detach(), target_ids)
            print(
                f"Step {step:4d} | loss: {loss.item():.6f} "
                f"| train_acc: {train_accuracy:.3f}"
            )

    model.eval()
    with torch.no_grad():
        final_logits = model(input_ids)["logits"]
        final_loss = F.cross_entropy(
            final_logits.reshape(-1, vocab_size),
            target_ids.reshape(-1),
            ignore_index=-100,
        )
        final_accuracy = compute_accuracy(final_logits, target_ids)

    print("\nFinal fixed-batch result")
    print(f"Final loss:     {final_loss.item():.6f}")
    print(f"Final accuracy: {final_accuracy:.6f}")

    if final_accuracy > 0.95:
        print(
            "MODEL CAN MEMORIZE - architecture has sufficient capacity, issue "
            "is likely generalization/hyperparameters, not a structural bug"
        )
    else:
        print(
            "MODEL CANNOT MEMORIZE EVEN ONE FIXED BATCH - this points to a "
            "structural/architectural bug, not a training or generalization issue"
        )
        supervised_positions = (target_ids[0] != -100).nonzero(as_tuple=False)
        position = int(supervised_positions[0].item())
        print(f"\nTop-5 predictions at supervised position {position}")
        probabilities = F.softmax(final_logits[:, position, :], dim=-1)
        top_probabilities, top_tokens = probabilities.topk(5, dim=-1)
        for sample_index in range(3):
            tokens = top_tokens[sample_index].tolist()
            probs = [round(value, 6) for value in top_probabilities[sample_index].tolist()]
            target = int(target_ids[sample_index, position].item())
            print(
                f"Sample {sample_index}: target={target}, "
                f"top_tokens={tokens}, probabilities={probs}"
            )


if __name__ == "__main__":
    main()
