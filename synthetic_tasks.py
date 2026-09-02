"""Synthetic long-range memory and selective-recall benchmarks."""

import torch
from torch import Tensor


IGNORE_INDEX = -100


def _validate_common(batch_size: int, seq_len: int, vocab_size: int) -> None:
    if batch_size <= 0 or seq_len <= 0:
        raise ValueError("batch_size and seq_len must be positive")
    if vocab_size < 4:
        raise ValueError("vocab_size must be at least 4")


def _validate_copy_length(seq_len: int, num_tokens_to_copy: int) -> None:
    # Source tokens, one trigger, and blank answer slots must fit in the sequence.
    if num_tokens_to_copy <= 0:
        raise ValueError("num_tokens_to_copy must be positive")
    if seq_len < 2 * num_tokens_to_copy + 1:
        raise ValueError(
            "seq_len must fit source tokens, a trigger, and answer slots"
        )


def generate_copying_task(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    num_tokens_to_copy: int,
) -> tuple[Tensor, Tensor]:
    """Generate a delayed-copy task for testing long-range sequence memory.

    The model must retain a token sequence across filler and then reproduce it
    after a recall trigger. This isolates ordered long-range recall, a useful
    diagnostic for whether a sequence model preserves information over many
    recurrent steps. ``vocab_size - 1`` is reserved as the blank/filler token
    and ``vocab_size - 2`` as the recall trigger, so ordinary data tokens use
    ids in ``[0, vocab_size - 2)``.
    """
    _validate_common(batch_size, seq_len, vocab_size)
    _validate_copy_length(seq_len, num_tokens_to_copy)

    blank_id = vocab_size - 1
    trigger_id = vocab_size - 2
    trigger_position = seq_len - num_tokens_to_copy - 1
    input_ids = torch.full((batch_size, seq_len), blank_id, dtype=torch.long)
    target_ids = torch.full(
        (batch_size, seq_len), IGNORE_INDEX, dtype=torch.long
    )

    copied_tokens = torch.randint(
        0, trigger_id, (batch_size, num_tokens_to_copy), dtype=torch.long
    )
    input_ids[:, :num_tokens_to_copy] = copied_tokens
    input_ids[:, trigger_position] = trigger_id
    answer_positions = torch.arange(
        trigger_position + 1, seq_len, dtype=torch.long
    )
    target_ids[:, answer_positions] = copied_tokens
    return input_ids, target_ids


def generate_associative_recall_task(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    num_pairs: int,
) -> tuple[Tensor, Tensor]:
    """Generate key-value recall examples for content-addressable memory.

    Randomly ordered key-value pairs are presented first, followed by filler,
    a trigger, a query key, and a blank answer slot. The target is stored at
    the answer slot immediately after the query key. The model must retrieve
    the value associated with that specific key rather than merely repeat
    recent context, directly testing long-range associative memory. The two
    highest token ids remain available as the blank and trigger conventions;
    keys and values use the ordinary range ``[0, vocab_size - 2)``.
    """
    _validate_common(batch_size, seq_len, vocab_size)
    if num_pairs <= 0:
        raise ValueError("num_pairs must be positive")
    minimum_length = 2 * num_pairs + 4  # pairs + one filler + trigger + query + answer
    if seq_len < minimum_length:
        raise ValueError(
            f"seq_len must be at least {minimum_length} for {num_pairs} pairs "
            "and one filler token"
        )

    blank_id = vocab_size - 1
    trigger_id = vocab_size - 2
    if num_pairs > trigger_id:
        raise ValueError(
            "num_pairs must be no greater than the number of ordinary token ids"
        )
    trigger_position = seq_len - 3
    query_position = seq_len - 2
    answer_position = seq_len - 1
    input_ids = torch.full((batch_size, seq_len), blank_id, dtype=torch.long)
    target_ids = torch.full(
        (batch_size, seq_len), IGNORE_INDEX, dtype=torch.long
    )

    # Sample without replacement per row so each key has exactly one value.
    keys = torch.stack(
        [torch.randperm(trigger_id)[:num_pairs] for _ in range(batch_size)]
    )
    values = torch.randint(
        0, trigger_id, (batch_size, num_pairs), dtype=torch.long
    )
    # Shuffle pair order independently per sample so the relevant pair is not
    # always in a fixed location, while keeping each key next to its value.
    pair_order = torch.argsort(torch.rand(batch_size, num_pairs), dim=1)
    shuffled_keys = torch.gather(keys, 1, pair_order)
    shuffled_values = torch.gather(values, 1, pair_order)
    pair_tokens = torch.stack((shuffled_keys, shuffled_values), dim=-1)
    input_ids[:, : 2 * num_pairs] = pair_tokens.reshape(batch_size, -1)

    filler_start = 2 * num_pairs
    input_ids[:, filler_start:trigger_position] = blank_id
    input_ids[:, trigger_position] = trigger_id
    query_pair = torch.randint(0, num_pairs, (batch_size,))
    input_ids[:, query_position] = shuffled_keys.gather(
        1, query_pair.unsqueeze(1)
    ).squeeze(1)
    input_ids[:, answer_position] = blank_id
    target_ids[:, answer_position] = shuffled_values.gather(
        1, query_pair.unsqueeze(1)
    ).squeeze(1)
    return input_ids, target_ids


def generate_associative_recall_task_with_labels(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    num_pairs: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Generate associative recall examples with ground-truth phase labels.

    The returned ``phase_labels`` has shape ``(batch_size, seq_len)`` and uses
    phase ids 0 through 3 for encoding, waiting, query, and retrieval. Encoding
    labels cover the key-value pair positions, waiting labels cover the filler
    positions between the pairs and trigger, the query label covers the query
    key, and the retrieval label covers the answer position. Labels are built
    from the same structural boundaries as the task generator, so they cannot
    drift from the generated example layout.
    """
    _validate_common(batch_size, seq_len, vocab_size)
    if num_pairs <= 0:
        raise ValueError("num_pairs must be positive")
    minimum_length = 2 * num_pairs + 4
    if seq_len < minimum_length:
        raise ValueError(
            f"seq_len must be at least {minimum_length} for {num_pairs} pairs "
            "and one filler token"
        )

    input_ids, target_ids = generate_associative_recall_task(
        batch_size, seq_len, vocab_size, num_pairs
    )
    trigger_position = seq_len - 3
    query_position = seq_len - 2
    answer_position = seq_len - 1
    phase_labels = torch.full(
        (batch_size, seq_len), 1, dtype=torch.long
    )
    phase_labels[:, : 2 * num_pairs] = 0
    phase_labels[:, query_position] = 2
    phase_labels[:, answer_position] = 3
    return input_ids, target_ids, phase_labels


def generate_selective_copying_task(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    num_tokens_to_copy: int,
) -> tuple[Tensor, Tensor]:
    """Generate selective copying examples with interleaved distractors.

    Relevant tokens are placed at fixed even positions in a noisy context, then
    must be reproduced after a recall trigger. The reproducible every-other
    position rule makes relevance depend on the learned selective mechanism,
    while the distractors test whether irrelevant inputs can be ignored. This
    mirrors selective-copying diagnostics for input-dependent state updates.

    As in :func:`generate_copying_task`, ``vocab_size - 1`` is the blank/filler
    id and ``vocab_size - 2`` is the trigger id; ordinary tokens use lower ids.
    """
    _validate_common(batch_size, seq_len, vocab_size)
    _validate_copy_length(seq_len, num_tokens_to_copy)

    blank_id = vocab_size - 1
    trigger_id = vocab_size - 2
    trigger_position = seq_len - num_tokens_to_copy - 1
    context_len = trigger_position
    context_positions = (
        torch.arange(num_tokens_to_copy) * context_len // num_tokens_to_copy
    )
    input_ids = torch.randint(
        0, trigger_id, (batch_size, seq_len), dtype=torch.long
    )
    target_ids = torch.full(
        (batch_size, seq_len), IGNORE_INDEX, dtype=torch.long
    )

    copied_tokens = torch.randint(
        0, trigger_id, (batch_size, num_tokens_to_copy), dtype=torch.long
    )
    input_ids[:, context_positions] = copied_tokens
    input_ids[:, trigger_position] = trigger_id
    input_ids[:, trigger_position + 1 :] = blank_id
    answer_positions = torch.arange(
        trigger_position + 1, seq_len, dtype=torch.long
    )
    target_ids[:, answer_positions] = copied_tokens
    return input_ids, target_ids


def _print_task(name: str, input_ids: Tensor, target_ids: Tensor) -> None:
    print(f"\n{name}")
    print(f"input_ids shape:  {tuple(input_ids.shape)}")
    print(f"target_ids shape: {tuple(target_ids.shape)}")
    print(f"first input_ids:  {input_ids[0].tolist()}")
    print(f"first target_ids: {target_ids[0].tolist()}")
    print(
        "first sample supervised positions: "
        f"{int((target_ids[0] != IGNORE_INDEX).sum())}"
    )


def _print_associative_verification(
    input_ids: Tensor, target_ids: Tensor, num_pairs: int
) -> None:
    first_input = input_ids[0]
    first_target = target_ids[0]
    pair_keys = first_input[: 2 * num_pairs : 2]
    pair_values = first_input[1 : 2 * num_pairs : 2]
    queried_key = int(first_input[-2])
    target_value = int(first_target[-1])
    matching_pair = (pair_keys == queried_key).nonzero(as_tuple=False).item()
    expected_value = int(pair_values[matching_pair])
    print(f"queried key: {queried_key}")
    print(f"target value: {target_value}")
    print(f"matching pair value: {expected_value}")
    assert target_value == expected_value


if __name__ == "__main__":
    torch.manual_seed(0)
    batch_size, seq_len, vocab_size, task_size = 4, 64, 50, 5

    tasks = (
        (
            "Copying task",
            generate_copying_task(
                batch_size, seq_len, vocab_size, task_size
            ),
        ),
        (
            "Associative recall task",
            generate_associative_recall_task(
                batch_size, seq_len, vocab_size, task_size
            ),
        ),
        (
            "Selective copying task",
            generate_selective_copying_task(
                batch_size, seq_len, vocab_size, task_size
            ),
        ),
    )

    for task_name, (input_ids, target_ids) in tasks:
        _print_task(task_name, input_ids, target_ids)
        assert input_ids.shape == target_ids.shape
        assert torch.all((target_ids != IGNORE_INDEX).sum(dim=1) > 0)
        if task_name == "Associative recall task":
            _print_associative_verification(input_ids, target_ids, task_size)

    print("\nAll task shape and supervision assertions passed.")
