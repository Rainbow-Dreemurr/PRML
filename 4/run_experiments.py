"""Train and evaluate Transformer positional-encoding ablations with PyTorch.

Default usage:

    D:\\Anaconda\\envs\\pytorch\\python.exe run_experiments.py

The script runs a faithful encoder-decoder Transformer implementation on a
controlled sequence reversal task and writes CSV logs, figures, predictions,
checkpoints, and a Chinese report.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.constants import PAD_ID
from src.data import ReverseSequenceDataset, collate_reverse_batch
from src.metrics import exact_match_accuracy, greedy_decode, token_accuracy, trim_sequence
from src.transformer import TransformerSeq2Seq


@dataclass
class ExperimentConfig:
    vocab_size: int = 32
    train_min_len: int = 4
    train_max_len: int = 12
    ood_min_len: int = 13
    ood_max_len: int = 18
    train_samples: int = 12000
    val_samples: int = 2000
    test_samples: int = 2000
    d_model: int = 128
    d_ff: int = 512
    n_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    batch_size: int = 128
    epochs: int = 14
    warmup_steps: int = 400
    label_smoothing: float = 0.1
    grad_clip: float = 1.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def noam_lr(step: int, d_model: int, warmup_steps: int) -> float:
    step = max(step, 1)
    return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))


def make_loader(
    dataset: ReverseSequenceDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        collate_fn=collate_reverse_batch,
        pin_memory=torch.cuda.is_available(),
    )


def make_datasets(cfg: ExperimentConfig, seed: int) -> dict[str, ReverseSequenceDataset]:
    return {
        "train": ReverseSequenceDataset(
            cfg.train_samples,
            cfg.train_min_len,
            cfg.train_max_len,
            cfg.vocab_size,
            seed=10_000 + seed,
        ),
        "val": ReverseSequenceDataset(
            cfg.val_samples,
            cfg.train_min_len,
            cfg.train_max_len,
            cfg.vocab_size,
            seed=20_000 + seed,
        ),
        "test_iid": ReverseSequenceDataset(
            cfg.test_samples,
            cfg.train_min_len,
            cfg.train_max_len,
            cfg.vocab_size,
            seed=30_000 + seed,
        ),
        "test_ood": ReverseSequenceDataset(
            cfg.test_samples,
            cfg.ood_min_len,
            cfg.ood_max_len,
            cfg.vocab_size,
            seed=40_000 + seed,
        ),
    }


@torch.no_grad()
def evaluate_teacher_forced(
    model: TransformerSeq2Seq,
    loader: DataLoader,
    device: torch.device,
    label_smoothing: float,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    token_acc_numer = 0.0
    for batch in loader:
        src = batch.src.to(device)
        tgt_in = batch.tgt_in.to(device)
        tgt_out = batch.tgt_out.to(device)
        logits = model(src, tgt_in)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt_out.reshape(-1),
            ignore_index=PAD_ID,
            label_smoothing=label_smoothing,
            reduction="sum",
        )
        non_pad = (tgt_out != PAD_ID).sum().item()
        total_loss += loss.item()
        total_tokens += non_pad
        token_acc_numer += token_accuracy(logits, tgt_out) * non_pad
    return {
        "loss": total_loss / max(1, total_tokens),
        "token_acc": token_acc_numer / max(1, total_tokens),
    }


@torch.no_grad()
def evaluate_greedy(
    model: TransformerSeq2Seq,
    loader: DataLoader,
    device: torch.device,
    max_decode_len: int,
    limit_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    exact_total = 0.0
    count = 0
    for batch_idx, batch in enumerate(loader):
        if limit_batches is not None and batch_idx >= limit_batches:
            break
        src = batch.src.to(device)
        tgt_out = batch.tgt_out.to(device)
        pred = greedy_decode(model, src, max_decode_len)
        batch_size = src.size(0)
        exact_total += exact_match_accuracy(pred, tgt_out) * batch_size
        count += batch_size
    return {"exact_match": exact_total / max(1, count)}


@torch.no_grad()
def collect_predictions(
    model: TransformerSeq2Seq,
    dataset: ReverseSequenceDataset,
    device: torch.device,
    max_decode_len: int,
    n_examples: int = 8,
) -> list[dict[str, list[int]]]:
    model.eval()
    loader = make_loader(dataset, batch_size=n_examples, shuffle=False, seed=0)
    batch = next(iter(loader))
    src = batch.src.to(device)
    pred = greedy_decode(model, src, max_decode_len).cpu().tolist()
    rows: list[dict[str, list[int]]] = []
    for i in range(min(n_examples, len(pred))):
        rows.append(
            {
                "src": trim_sequence(batch.src[i].tolist()),
                "target": trim_sequence(batch.tgt_out[i].tolist()),
                "prediction": trim_sequence(pred[i]),
            }
        )
    return rows


def train_one(
    pe_kind: str,
    seed: int,
    cfg: ExperimentConfig,
    out_dir: Path,
    device: torch.device,
    use_residual: bool = True,
) -> tuple[list[dict[str, float | int | str]], dict[str, float | int | str], list[dict[str, list[int]]]]:
    set_seed(seed)
    datasets = make_datasets(cfg, seed)
    loaders = {
        name: make_loader(ds, cfg.batch_size, shuffle=(name == "train"), seed=seed)
        for name, ds in datasets.items()
    }
    max_model_len = cfg.ood_max_len + 2
    model = TransformerSeq2Seq(
        vocab_size=cfg.vocab_size,
        max_len=max_model_len,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        num_layers=cfg.num_layers,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        pe_kind=pe_kind,
        use_residual=use_residual,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    global_step = 0
    best_val_exact = -1.0
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict[str, float | int | str]] = []
    start_time = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        train_tokens = 0
        for batch in loaders["train"]:
            src = batch.src.to(device)
            tgt_in = batch.tgt_in.to(device)
            tgt_out = batch.tgt_out.to(device)
            logits = model(src, tgt_in)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tgt_out.reshape(-1),
                ignore_index=PAD_ID,
                label_smoothing=cfg.label_smoothing,
                reduction="sum",
            )
            non_pad = (tgt_out != PAD_ID).sum()
            normalized_loss = loss / non_pad.clamp_min(1)

            optimizer.zero_grad(set_to_none=True)
            normalized_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            global_step += 1
            lr = noam_lr(global_step, cfg.d_model, cfg.warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()

            train_loss += loss.item()
            train_tokens += int(non_pad.item())

        val_teacher = evaluate_teacher_forced(model, loaders["val"], device, cfg.label_smoothing)
        val_greedy = evaluate_greedy(
            model,
            loaders["val"],
            device,
            max_decode_len=cfg.train_max_len + 1,
            limit_batches=max(1, math.ceil(512 / cfg.batch_size)),
        )
        row = {
            "pe_kind": pe_kind,
            "use_residual": str(use_residual),
            "seed": seed,
            "epoch": epoch,
            "step": global_step,
            "lr": noam_lr(global_step, cfg.d_model, cfg.warmup_steps),
            "train_loss": train_loss / max(1, train_tokens),
            "val_loss": val_teacher["loss"],
            "val_token_acc": val_teacher["token_acc"],
            "val_exact_match_subset": val_greedy["exact_match"],
            "elapsed_sec": time.time() - start_time,
        }
        history.append(row)
        print(
            f"[{pe_kind} seed={seed}] epoch {epoch:02d}/{cfg.epochs} "
            f"train_loss={row['train_loss']:.4f} val_loss={row['val_loss']:.4f} "
            f"val_tok={row['val_token_acc']:.3f} val_exact={row['val_exact_match_subset']:.3f}"
        )
        if val_greedy["exact_match"] > best_val_exact:
            best_val_exact = val_greedy["exact_match"]
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    final_teacher_iid = evaluate_teacher_forced(model, loaders["test_iid"], device, cfg.label_smoothing)
    final_teacher_ood = evaluate_teacher_forced(model, loaders["test_ood"], device, cfg.label_smoothing)
    final_greedy_iid = evaluate_greedy(model, loaders["test_iid"], device, max_decode_len=cfg.train_max_len + 1)
    final_greedy_ood = evaluate_greedy(model, loaders["test_ood"], device, max_decode_len=cfg.ood_max_len + 1)
    summary = {
        "pe_kind": pe_kind,
        "use_residual": str(use_residual),
        "seed": seed,
        "best_val_exact_subset": best_val_exact,
        "test_iid_loss": final_teacher_iid["loss"],
        "test_iid_token_acc": final_teacher_iid["token_acc"],
        "test_iid_exact_match": final_greedy_iid["exact_match"],
        "test_ood_loss": final_teacher_ood["loss"],
        "test_ood_token_acc": final_teacher_ood["token_acc"],
        "test_ood_exact_match": final_greedy_ood["exact_match"],
        "elapsed_sec": time.time() - start_time,
        "device": str(device),
    }

    residual_suffix = "residual" if use_residual else "no_residual"
    checkpoint_path = out_dir / "checkpoints" / f"transformer_{pe_kind}_{residual_suffix}_seed{seed}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(cfg),
            "pe_kind": pe_kind,
            "seed": seed,
            "use_residual": use_residual,
        },
        checkpoint_path,
    )
    predictions = collect_predictions(model, datasets["test_iid"], device, cfg.train_max_len + 1)
    return history, summary, predictions


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_predictions(path: Path) -> dict[str, list[dict[str, list[int]]]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_rows(rows: list[dict[str, float | int | str]], use_residual_default: bool = True) -> list[dict[str, float | int | str]]:
    normalized = []
    for row in rows:
        item = dict(row)
        item.setdefault("use_residual", str(use_residual_default))
        normalized.append(item)
    return normalized


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def variant_label(row: dict[str, float | int | str]) -> str:
    pe_kind = str(row["pe_kind"])
    use_residual = str(row.get("use_residual", "True")).lower() == "true"
    return pe_kind if use_residual else f"{pe_kind}_no_residual"


def aggregate_summary(summary: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    grouped: dict[str, list[dict[str, float | int | str]]] = {}
    for row in summary:
        grouped.setdefault(variant_label(row), []).append(row)
    rows: list[dict[str, float | int | str]] = []
    metric_names = [
        "best_val_exact_subset",
        "test_iid_token_acc",
        "test_iid_exact_match",
        "test_ood_token_acc",
        "test_ood_exact_match",
    ]
    for label, items in sorted(grouped.items()):
        result: dict[str, float | int | str] = {
            "variant": label,
            "n_seeds": len(items),
            "seeds": " ".join(str(item["seed"]) for item in items),
        }
        for metric in metric_names:
            values = np.array([float(item[metric]) for item in items], dtype=float)
            result[f"{metric}_mean"] = float(values.mean())
            result[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(result)
    return rows


def plot_outputs(out_dir: Path, history: list[dict[str, float | int | str]], summary: list[dict[str, float | int | str]]) -> None:
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8.4, 5.0))
    grouped: dict[str, list[dict[str, float | int | str]]] = {}
    for row in history:
        grouped.setdefault(variant_label(row), []).append(row)
    for pe_kind, rows in grouped.items():
        by_epoch: dict[int, list[float]] = {}
        for item in rows:
            by_epoch.setdefault(int(item["epoch"]), []).append(float(item["val_exact_match_subset"]))
        epochs = sorted(by_epoch)
        means = [float(np.mean(by_epoch[epoch])) for epoch in epochs]
        stds = [float(np.std(by_epoch[epoch], ddof=1)) if len(by_epoch[epoch]) > 1 else 0.0 for epoch in epochs]
        plt.plot(
            epochs,
            means,
            marker="o",
            linewidth=1.8,
            label=pe_kind,
        )
        plt.fill_between(epochs, np.array(means) - np.array(stds), np.array(means) + np.array(stds), alpha=0.12)
    plt.xlabel("Epoch")
    plt.ylabel("Validation exact match")
    plt.ylim(-0.02, 1.02)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.title("Validation sequence accuracy by positional encoding")
    plt.tight_layout()
    plt.savefig(figures_dir / "validation_exact_match.png", dpi=180)
    plt.close()

    aggregated = aggregate_summary(summary)
    write_csv(out_dir / "summary_by_variant.csv", aggregated)
    labels = [str(row["variant"]) for row in aggregated]
    iid = [float(row["test_iid_exact_match_mean"]) for row in aggregated]
    ood = [float(row["test_ood_exact_match_mean"]) for row in aggregated]
    iid_std = [float(row["test_iid_exact_match_std"]) for row in aggregated]
    ood_std = [float(row["test_ood_exact_match_std"]) for row in aggregated]
    x = np.arange(len(labels))
    width = 0.36
    plt.figure(figsize=(8.2, 5.0))
    plt.bar(x - width / 2, iid, width, yerr=iid_std, capsize=4, label="IID length")
    plt.bar(x + width / 2, ood, width, yerr=ood_std, capsize=4, label="Longer OOD length")
    plt.xticks(x, labels)
    plt.ylabel("Greedy exact match")
    plt.ylim(-0.02, 1.02)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.title("Final sequence accuracy")
    for xpos, value in zip(x - width / 2, iid):
        plt.text(xpos, value + 0.015, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    for xpos, value in zip(x + width / 2, ood):
        plt.text(xpos, value + 0.015, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(figures_dir / "final_exact_match.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("outputs_transformer"))
    parser.add_argument("--pe-kinds", nargs="+", default=["none", "linear", "learned", "sinusoidal"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Skip finished pe_kind/seed/residual runs in out-dir.")
    parser.add_argument("--no-residual", action="store_true", help="Disable residual connections for ablation.")
    parser.add_argument("--quick", action="store_true", help="Small smoke run for debugging only.")
    args = parser.parse_args()

    cfg = ExperimentConfig()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.train_samples is not None:
        cfg.train_samples = args.train_samples
    if args.quick:
        cfg.train_samples = 512
        cfg.val_samples = 256
        cfg.test_samples = 256
        cfg.epochs = 2
        cfg.d_model = 64
        cfg.d_ff = 128
        cfg.batch_size = 64

    device = torch.device(args.device)
    use_residual = not args.no_residual
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": asdict(cfg),
                "pe_kinds": args.pe_kinds,
                "seeds": args.seeds,
                "use_residual": use_residual,
                "device": str(device),
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    all_history: list[dict[str, float | int | str]] = []
    summaries: list[dict[str, float | int | str]] = []
    all_predictions: dict[str, list[dict[str, list[int]]]] = {}
    if args.resume:
        all_history = normalize_rows(read_csv(args.out_dir / "history.csv"), use_residual_default=use_residual)
        summaries = normalize_rows(read_csv(args.out_dir / "results.csv"), use_residual_default=use_residual)
        all_predictions = read_predictions(args.out_dir / "predictions.json")
        print(f"Resume mode: loaded {len(summaries)} summaries and {len(all_history)} history rows.")

    completed = {
        (str(row["pe_kind"]), int(row["seed"]), str(row.get("use_residual", str(use_residual))).lower())
        for row in summaries
    }

    for pe_kind in args.pe_kinds:
        for seed in args.seeds:
            key = (pe_kind, seed, str(use_residual).lower())
            if args.resume and key in completed:
                print(f"Skipping completed run pe={pe_kind} seed={seed} use_residual={use_residual}")
                continue
            history, summary, predictions = train_one(pe_kind, seed, cfg, args.out_dir, device, use_residual)
            all_history.extend(history)
            summaries.append(summary)
            suffix = "residual" if use_residual else "no_residual"
            all_predictions[f"{pe_kind}_{suffix}_seed{seed}"] = predictions
            completed.add(key)
            write_csv(args.out_dir / "history.csv", all_history)
            write_csv(args.out_dir / "results.csv", summaries)
            with (args.out_dir / "predictions.json").open("w", encoding="utf-8") as f:
                json.dump(all_predictions, f, indent=2, ensure_ascii=False)
            plot_outputs(args.out_dir, all_history, summaries)

    print(f"Results written to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
