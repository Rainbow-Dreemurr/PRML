"""Synthetic sequence-to-sequence datasets for positional encoding experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .constants import BOS_ID, EOS_ID, FIRST_SYMBOL_ID, PAD_ID


@dataclass(frozen=True)
class Batch:
    src: torch.Tensor
    tgt_in: torch.Tensor
    tgt_out: torch.Tensor


class ReverseSequenceDataset(Dataset):
    """Variable-length sequence reversal task.

    Source: random symbols followed by EOS.
    Target input: BOS followed by the reversed random symbols.
    Target output: reversed random symbols followed by EOS.

    The task is useful for positional encoding ablations because the output is
    not a function of the input multiset. A model without positional encodings
    sees self-attention inputs as orderless evidence and cannot reliably choose
    the correct reverse order.
    """

    def __init__(
        self,
        num_samples: int,
        min_len: int,
        max_len: int,
        vocab_size: int,
        seed: int,
    ):
        if vocab_size <= FIRST_SYMBOL_ID:
            raise ValueError("vocab_size must leave room for non-special symbols")
        rng = np.random.default_rng(seed)
        self.samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for _ in range(num_samples):
            length = int(rng.integers(min_len, max_len + 1))
            core = rng.integers(FIRST_SYMBOL_ID, vocab_size, size=length, dtype=np.int64)
            reversed_core = core[::-1].copy()
            src = torch.tensor(np.concatenate([core, [EOS_ID]]), dtype=torch.long)
            tgt_in = torch.tensor(np.concatenate([[BOS_ID], reversed_core]), dtype=torch.long)
            tgt_out = torch.tensor(np.concatenate([reversed_core, [EOS_ID]]), dtype=torch.long)
            self.samples.append((src, tgt_in, tgt_out))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.samples[idx]


def collate_reverse_batch(items: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Batch:
    src, tgt_in, tgt_out = zip(*items)
    return Batch(
        src=pad_sequence(src, batch_first=True, padding_value=PAD_ID),
        tgt_in=pad_sequence(tgt_in, batch_first=True, padding_value=PAD_ID),
        tgt_out=pad_sequence(tgt_out, batch_first=True, padding_value=PAD_ID),
    )
