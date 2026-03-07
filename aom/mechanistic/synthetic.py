from __future__ import annotations

from typing import Optional, Set

import torch


def build_allowed_ids(vocab_size: int, exclude_token_ids: Optional[Set[int]]) -> Optional[torch.Tensor]:
    if not exclude_token_ids:
        return None
    if vocab_size < 1:
        raise ValueError("vocab_size must be >= 1")
    excluded = set(int(x) for x in exclude_token_ids)
    allowed = [i for i in range(int(vocab_size)) if i not in excluded]
    if not allowed:
        raise ValueError("exclude_token_ids removes all tokens from the vocabulary")
    return torch.tensor(allowed, dtype=torch.long)


def generate_induction_batch(
    *,
    vocab_size: int,
    base_len: int,
    repeats: int,
    batch_size: int,
    seed: int,
    exclude_token_ids: Optional[Set[int]] = None,
    allowed_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Generate a batch of repeated random token sequences: X | X | ... (repeats times).
    If allowed_ids is provided, sampling is restricted to that token list.

    Output shape: [batch_size, base_len * repeats]
    """
    if base_len < 2:
        raise ValueError("base_len must be >= 2")
    if repeats < 2:
        raise ValueError("repeats must be >= 2")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if vocab_size < 1:
        raise ValueError("vocab_size must be >= 1")

    g = torch.Generator()
    g.manual_seed(int(seed))

    allowed = allowed_ids
    if allowed is None:
        allowed = build_allowed_ids(vocab_size, exclude_token_ids)
    if allowed is None:
        base = torch.randint(0, int(vocab_size), (int(batch_size), int(base_len)), generator=g, dtype=torch.long)
    else:
        if allowed.dtype != torch.long:
            allowed = allowed.to(dtype=torch.long)
        if allowed.numel() < 1:
            raise ValueError("allowed_ids must contain at least one token id")
        idx = torch.randint(0, int(allowed.numel()), (int(batch_size), int(base_len)), generator=g, dtype=torch.long)
        base = allowed[idx]
    return base.repeat(1, int(repeats))


def permute_first_half(
    input_ids: torch.Tensor,
    *,
    base_len: int,
    seed: int,
    resample_identity: bool = True,
) -> torch.Tensor:
    """
    Permute the first half of a repeated sequence batch: perm(X) | X.

    Permutations are applied per batch element and seeded deterministically.
    """
    if input_ids.ndim != 2:
        raise ValueError("input_ids must be rank-2 [batch, seq_len]")
    if base_len < 2:
        raise ValueError("base_len must be >= 2")
    if input_ids.size(1) < 2 * base_len:
        raise ValueError("input_ids length is shorter than 2 * base_len")

    bsz = int(input_ids.size(0))
    out = input_ids.clone()
    g = torch.Generator()
    g.manual_seed(int(seed))
    for i in range(bsz):
        perm = torch.randperm(int(base_len), generator=g, dtype=torch.long)
        if resample_identity:
            identity = torch.arange(int(base_len), dtype=torch.long)
            if torch.equal(perm, identity):
                perm = torch.randperm(int(base_len), generator=g, dtype=torch.long)
        out[i, :base_len] = out[i, :base_len][perm]
    return out
