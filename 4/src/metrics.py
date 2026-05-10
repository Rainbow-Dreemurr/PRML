"""Metrics and decoding helpers for sequence experiments."""

from __future__ import annotations

import torch

from .constants import BOS_ID, EOS_ID, PAD_ID


def token_accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    mask = target != PAD_ID
    if mask.sum().item() == 0:
        return 0.0
    return (pred[mask] == target[mask]).float().mean().item()


def trim_sequence(seq: list[int]) -> list[int]:
    trimmed: list[int] = []
    for token in seq:
        if token == PAD_ID:
            break
        trimmed.append(token)
        if token == EOS_ID:
            break
    return trimmed


def exact_match_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_cpu = pred.detach().cpu().tolist()
    target_cpu = target.detach().cpu().tolist()
    matches = 0
    for pred_seq, target_seq in zip(pred_cpu, target_cpu):
        if trim_sequence(pred_seq) == trim_sequence(target_seq):
            matches += 1
    return matches / max(1, len(target_cpu))


@torch.no_grad()
def greedy_decode(model: torch.nn.Module, src: torch.Tensor, max_len: int) -> torch.Tensor:
    model.eval()
    generated = torch.full((src.size(0), 1), BOS_ID, dtype=torch.long, device=src.device)
    finished = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
    memory, src_mask = model.encode(src)
    for _ in range(max_len):
        decoded = model.decode(generated, memory, src_mask)
        logits = model.generator(decoded[:, -1])
        next_token = logits.argmax(dim=-1)
        next_token = torch.where(finished, torch.full_like(next_token, PAD_ID), next_token)
        generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
        finished = finished | (next_token == EOS_ID)
        if finished.all():
            break
    return generated[:, 1:]
