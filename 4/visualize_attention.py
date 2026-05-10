"""Create a cross-attention heatmap for a trained Transformer checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.constants import EOS_ID
from src.data import ReverseSequenceDataset
from src.transformer import TransformerSeq2Seq


def choose_checkpoint(out_dir: Path, pe_kind: str, seed: int) -> Path:
    candidates = [
        out_dir / "checkpoints" / f"transformer_{pe_kind}_residual_seed{seed}.pt",
        out_dir / "checkpoints" / f"transformer_{pe_kind}_seed{seed}.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No checkpoint found for {pe_kind} seed={seed} in {out_dir}")


def trim(seq: torch.Tensor) -> list[int]:
    values = seq.detach().cpu().tolist()
    if EOS_ID in values:
        return values[: values.index(EOS_ID) + 1]
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("outputs_transformer"))
    parser.add_argument("--pe-kind", default="sinusoidal")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint_path = choose_checkpoint(args.out_dir, args.pe_kind, args.seed)
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    cfg = checkpoint["config"]
    device = torch.device(args.device)
    model = TransformerSeq2Seq(
        vocab_size=cfg["vocab_size"],
        max_len=cfg["ood_max_len"] + 2,
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        num_layers=cfg["num_layers"],
        d_ff=cfg["d_ff"],
        dropout=cfg["dropout"],
        pe_kind=checkpoint["pe_kind"],
        use_residual=bool(checkpoint.get("use_residual", True)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dataset = ReverseSequenceDataset(
        num_samples=max(16, args.sample_index + 1),
        min_len=cfg["train_min_len"],
        max_len=cfg["train_max_len"],
        vocab_size=cfg["vocab_size"],
        seed=30_000 + args.seed,
    )
    src, tgt_in, tgt_out = dataset[args.sample_index]
    with torch.no_grad():
        _ = model(src.unsqueeze(0).to(device), tgt_in.unsqueeze(0).to(device))
    attention = model.decoder_layers[-1].cross_attn.last_attention
    if attention is None:
        raise RuntimeError("No cross-attention weights were captured.")

    src_tokens = trim(src)
    target_tokens = trim(tgt_out)
    matrix = attention[0, :, : len(target_tokens), : len(src_tokens)].mean(dim=0).detach().cpu().tolist()

    figures_dir = args.out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    json_path = figures_dir / f"{args.pe_kind}_cross_attention_seed{args.seed}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": str(checkpoint_path),
                "src": src_tokens,
                "target": target_tokens,
                "attention": matrix,
            },
            f,
            indent=2,
        )

    fig_w = max(7.0, 0.55 * len(src_tokens))
    fig_h = max(5.2, 0.42 * len(target_tokens))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    image = ax.imshow(matrix, cmap="viridis", aspect="auto", vmin=0.0)
    ax.set_xticks(range(len(src_tokens)))
    ax.set_yticks(range(len(target_tokens)))
    ax.set_xticklabels([str(x) for x in src_tokens])
    ax.set_yticklabels([str(y) for y in target_tokens])
    ax.set_xlabel("Source token")
    ax.set_ylabel("Decoded target token")
    ax.set_title("Decoder cross-attention, averaged over heads")

    source_core_len = len(src_tokens) - 1
    expected_x = list(reversed(range(source_core_len))) + [source_core_len]
    expected_y = list(range(len(target_tokens)))
    ax.scatter(expected_x, expected_y, s=70, facecolors="none", edgecolors="white", linewidths=1.5)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Attention weight")
    fig.tight_layout()
    png_path = figures_dir / f"{args.pe_kind}_cross_attention_seed{args.seed}.png"
    fig.savefig(png_path, dpi=190)
    plt.close(fig)
    print(f"Wrote {png_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
