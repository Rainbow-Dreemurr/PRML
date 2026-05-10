"""Create cross-experiment summary tables and residual-ablation plots."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    main_dir = Path("outputs_transformer")
    no_residual_dir = Path("outputs_no_residual")
    pe_rows = read_csv(main_dir / "summary_by_variant.csv")
    no_residual_rows = read_csv(no_residual_dir / "summary_by_variant.csv")

    combined = pe_rows + no_residual_rows
    write_csv(main_dir / "all_summary_by_variant.csv", combined)

    residual_rows = [
        next(row for row in pe_rows if row["variant"] == "sinusoidal"),
        next(row for row in no_residual_rows if row["variant"] == "sinusoidal_no_residual"),
    ]
    renamed = []
    for row in residual_rows:
        item = dict(row)
        item["variant"] = (
            "sinusoidal_with_residual"
            if item["variant"] == "sinusoidal"
            else "sinusoidal_no_residual"
        )
        renamed.append(item)
    write_csv(main_dir / "residual_ablation_summary.csv", renamed)

    labels = ["with residual", "no residual"]
    iid = [float(row["test_iid_exact_match_mean"]) for row in renamed]
    ood = [float(row["test_ood_exact_match_mean"]) for row in renamed]
    iid_std = [float(row["test_iid_exact_match_std"]) for row in renamed]
    ood_std = [float(row["test_ood_exact_match_std"]) for row in renamed]
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.bar(x - width / 2, iid, width, yerr=iid_std, capsize=5, label="IID length")
    ax.bar(x + width / 2, ood, width, yerr=ood_std, capsize=5, label="Longer OOD length")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel("Greedy exact match")
    ax.set_title("Residual connection ablation, sinusoidal PE")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    for xpos, value in zip(x - width / 2, iid):
        ax.text(xpos, value + 0.02, f"{value:.3f}", ha="center", va="bottom")
    for xpos, value in zip(x + width / 2, ood):
        ax.text(xpos, value + 0.02, f"{value:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    out_path = main_dir / "figures" / "residual_ablation.png"
    fig.savefig(out_path, dpi=190)
    plt.close(fig)
    print(f"Wrote {main_dir / 'all_summary_by_variant.csv'}")
    print(f"Wrote {main_dir / 'residual_ablation_summary.csv'}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
