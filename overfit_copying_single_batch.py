"""Check whether MarkovMambaLM can memorize one copying batch."""

import torch
from torch.nn import functional as F

from markov_mamba_model import MarkovMambaLM
from synthetic_tasks import generate_copying_task


vocab_size = 50
seq_len = 32
num_tokens_to_copy = 4
batch_size = 32
num_train_steps = 1000
log_every = 50
learning_rate = 1e-3
gradient_clip_norm = 1.0
device = "cuda" if torch.cuda.is_available() else "cpu"


def compute_accuracy(logits: torch.Tensor, target_ids: torch.Tensor) -> float:
    """Compute accuracy at the exact non-masked target positions."""
    supervised = target_ids != -100
    predictions = logits.argmax(dim=-1)
    return float((predictions[supervised] == target_ids[supervised]).float().mean())


def main() -> None:
    model = MarkovMambaLM(
        vocab_size=vocab_size,
        d_model=64,
        d_state=16,
        d_conv=4,
        expand=2,
        num_markov_states=8,
        num_layers=4,
    ).to(device)

    # Seed immediately before generation to make this exact fixed batch stable.
    torch.manual_seed(42)
    input_ids, target_ids = generate_copying_task(
        batch_size=batch_size,
        seq_len=seq_len,
        vocab_size=vocab_size,
        num_tokens_to_copy=num_tokens_to_copy,
    )
    supervised_positions = (target_ids[0] != -100).nonzero(as_tuple=False).flatten()
    print(f"Device: {device}")
    print("Fixed copying batch inspection")
    print(f"First sample input_ids:  {input_ids[0].tolist()}")
    print(f"First sample target_ids: {target_ids[0].tolist()}")
    print(f"Supervised positions:    {supervised_positions.tolist()}")
    print(
        "Target tokens at supervised positions: "
        f"{target_ids[0, supervised_positions].tolist()}"
    )

    input_ids = input_ids.to(device)
    target_ids = target_ids.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    print("\nOverfitting one fixed copying batch")
    print(
        f"Configuration: batch_size={batch_size}, seq_len={seq_len}, "
        f"num_tokens_to_copy={num_tokens_to_copy}, steps={num_train_steps}, "
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
            "MODEL CAN MEMORIZE COPYING TASK - bug is likely in the fresh-batch "
            "comparison script, not the task or model"
        )
    else:
        print(
            "MODEL CANNOT MEMORIZE EVEN ONE FIXED COPYING BATCH - there is a "
            "structural bug, likely in how logits/targets align for this "
            "specific task's multi-position target structure"
        )
        positions = supervised_positions.to(device)
        predictions = final_logits[:, positions].argmax(dim=-1)
        for sample_index in range(2):
            print(
                f"Sample {sample_index} predicted tokens: "
                f"{predictions[sample_index].tolist()}"
            )
            print(
                f"Sample {sample_index} target tokens:    "
                f"{target_ids[sample_index, positions].tolist()}"
            )


if __name__ == "__main__":
    main()
