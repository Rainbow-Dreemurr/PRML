# -*- coding: utf-8 -*-
"""Generate the Chinese homework report from experiment outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def metric(row: dict[str, str], name: str) -> str:
    mean = float(row[f"{name}_mean"])
    std = float(row[f"{name}_std"])
    return f"{mean:.3f} ± {std:.3f}"


def row_by_variant(rows: list[dict[str, str]], variant: str) -> dict[str, str]:
    return next(row for row in rows if row["variant"] == variant)


def main() -> None:
    out_dir = Path("outputs_transformer")
    no_residual_dir = Path("outputs_no_residual")
    pe_summary = read_csv(out_dir / "summary_by_variant.csv")
    residual_summary = read_csv(out_dir / "residual_ablation_summary.csv")
    individual_results = read_csv(out_dir / "results.csv")

    with (out_dir / "config.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)
    with (out_dir / "predictions.json").open("r", encoding="utf-8") as f:
        predictions = json.load(f)
    with (out_dir / "figures" / "sinusoidal_cross_attention_seed0.json").open("r", encoding="utf-8") as f:
        attention_example = json.load(f)

    cfg = meta["config"]
    order = ["none", "linear", "learned", "sinusoidal"]
    pe_rows = []
    for variant in order:
        row = row_by_variant(pe_summary, variant)
        pe_rows.append(
            "| {variant} | {iid_tok} | {iid_exact} | {ood_tok} | {ood_exact} | {val} |".format(
                variant=variant,
                iid_tok=metric(row, "test_iid_token_acc"),
                iid_exact=metric(row, "test_iid_exact_match"),
                ood_tok=metric(row, "test_ood_token_acc"),
                ood_exact=metric(row, "test_ood_exact_match"),
                val=metric(row, "best_val_exact_subset"),
            )
        )

    residual_rows = []
    for variant in ["sinusoidal_with_residual", "sinusoidal_no_residual"]:
        row = row_by_variant(residual_summary, variant)
        residual_rows.append(
            "| {variant} | {iid_tok} | {iid_exact} | {ood_tok} | {ood_exact} |".format(
                variant=variant,
                iid_tok=metric(row, "test_iid_token_acc"),
                iid_exact=metric(row, "test_iid_exact_match"),
                ood_tok=metric(row, "test_ood_token_acc"),
                ood_exact=metric(row, "test_ood_exact_match"),
            )
        )

    sample_sections = []
    for key in ["none_seed0", "sinusoidal_seed0"]:
        if key in predictions:
            lines = [f"**{key}**"]
            for item in predictions[key][:3]:
                lines.append(f"- src={item['src']}  target={item['target']}  pred={item['prediction']}")
            sample_sections.append("\n".join(lines))

    individual_count = len(individual_results)
    report = f"""# 模式识别与机器学习第四次作业：Transformer 复现与模块消融实验

## 1. 论文阅读与核心结构复现

本作业阅读并复现 Vaswani 等人在 2017 年提出的 **Attention Is All You Need**。论文的核心思想是完全使用注意力机制进行序列建模，用自注意力替代循环网络或卷积网络中的显式时序递推。代码没有直接调用 `torch.nn.Transformer`，而是在 [src/transformer.py](src/transformer.py) 中从头实现了以下模块：

1. Scaled Dot-Product Attention：使用 $QK^T/\\sqrt{{d_k}}$ 计算注意力分布。
2. Multi-Head Attention：并行使用多个注意力头捕捉不同子空间的依赖。
3. Encoder-Decoder 架构：Encoder 双向读源序列，Decoder 用 causal mask 自回归生成目标序列。
4. Cross-Attention：Decoder 通过 cross-attention 读取 Encoder 表示。
5. Position-wise Feed-Forward Network：每个位置共享两层 MLP。
6. Residual Connection + LayerNorm：每个子层外接残差和归一化，增强优化稳定性。
7. Positional Encoding：在 [src/position_encodings.py](src/position_encodings.py) 中实现无位置编码、简单线性绝对编码、可学习绝对编码和论文正弦编码。

## 2. 实验任务与设置

主实验选择题目 **2.1 位置编码**，额外补充题目 **2.3 残差结构**。实验任务为可控的序列反转：

- 输入：随机 token 序列加 EOS，例如 `[7, 15, 4, 2]`。
- 输出：反转后的序列加 EOS，例如 `[4, 15, 7, 2]`。
- 训练长度：{cfg["train_min_len"]} 到 {cfg["train_max_len"]}。
- IID 测试：长度仍为 {cfg["train_min_len"]} 到 {cfg["train_max_len"]}。
- OOD 测试：更长的 {cfg["ood_min_len"]} 到 {cfg["ood_max_len"]}，用于观察长度外推。

该任务需要模型知道每个 token 的位置。若没有位置编码，自注意力主要看到的是 token 内容集合，很难稳定区分同一组 token 的不同排列。

主要超参数如下：

| 项目 | 数值 |
|---|---:|
| vocab size | {cfg["vocab_size"]} |
| train/val/test samples | {cfg["train_samples"]}/{cfg["val_samples"]}/{cfg["test_samples"]} |
| d_model | {cfg["d_model"]} |
| heads | {cfg["n_heads"]} |
| layers | {cfg["num_layers"]} |
| d_ff | {cfg["d_ff"]} |
| batch size | {cfg["batch_size"]} |
| epochs | {cfg["epochs"]} |
| optimizer | Adam + Transformer Noam learning-rate schedule |
| label smoothing | {cfg["label_smoothing"]} |
| seeds | 0, 1, 2 |
| device | {meta["device"]}, {meta["cuda_device_name"]} |

运行命令：

```powershell
D:\\Anaconda\\envs\\pytorch\\python.exe run_experiments.py --resume --seeds 0 1 2
D:\\Anaconda\\envs\\pytorch\\python.exe run_experiments.py --pe-kinds sinusoidal --seeds 0 1 2 --no-residual --out-dir outputs_no_residual
D:\\Anaconda\\envs\\pytorch\\python.exe visualize_attention.py
D:\\Anaconda\\envs\\pytorch\\python.exe summarize_results.py
D:\\Anaconda\\envs\\pytorch\\python.exe generate_report.py
```

## 3. 位置编码消融结果

下表为 3 个随机种子的均值 ± 标准差。共完成 {individual_count} 次主实验训练。

| 位置编码 | IID token acc | IID exact match | OOD token acc | OOD exact match | best val exact |
|---|---:|---:|---:|---:|---:|
{chr(10).join(pe_rows)}

验证集曲线：

![validation](outputs_transformer/figures/validation_exact_match.png)

最终准确率：

![final](outputs_transformer/figures/final_exact_match.png)

结论：无位置编码的 `none` 在 IID exact match 上只有约 0.007，说明模型几乎不能完成完整序列反转。`linear`、`learned`、`sinusoidal` 都能显著提升训练长度内的精确匹配率，其中 learned 和 sinusoidal 在 IID 上达到约 0.97。OOD 长度上整体下降明显，说明位置编码提供了顺序信息，但长度外推还受到训练长度覆盖、自回归误差累积和任务难度的共同影响。

## 4. 注意力可视化

使用已训练的 sinusoidal seed0 模型，取最后一层 Decoder 的 cross-attention，并对所有 head 求平均。下图横轴为源序列 token，纵轴为当前要生成的目标 token；白色空心点标出了反转任务中理论上应关注的源位置。

![attention](outputs_transformer/figures/sinusoidal_cross_attention_seed0.png)

该样例的源序列为：

`{attention_example["src"]}`

目标序列为：

`{attention_example["target"]}`

可视化说明：热力图中的高权重区域整体沿“反向对应关系”分布，说明 Decoder 在生成反转序列时确实倾向于从源序列后部向前读取信息。这为数值指标提供了更直观的解释。

## 5. 残差结构消融结果

为了补充题目 2.3，固定使用论文正弦位置编码，只改变是否使用残差连接。无残差模型仍保留注意力、FFN、LayerNorm 和 mask，但每个子层不再执行 `x + sublayer(x)`。

| 结构 | IID token acc | IID exact match | OOD token acc | OOD exact match |
|---|---:|---:|---:|---:|
{chr(10).join(residual_rows)}

![residual](outputs_transformer/figures/residual_ablation.png)

结果非常明显：有残差时 sinusoidal 模型的 IID exact match 约为 0.975；去掉残差后 3 个 seed 的 exact match 都为 0。训练日志中无残差模型的 token acc 也长期停在较低水平。这说明残差结构不是单纯的工程细节，它为深层注意力网络提供了信息直通路径和更稳定的梯度传播，使注意力层、交叉注意力层和前馈层可以逐步修正表示，而不是每一层都强行重写表示。

## 6. 定性样例

{chr(10).join(sample_sections)}

## 7. 总结

本次作业完成了 Transformer encoder-decoder 的核心复现，并通过位置编码、注意力可视化和残差结构消融验证了两个关键结论：

1. 自注意力擅长内容寻址，但自身不携带顺序。位置编码的核心作用是打破排列对称性，使模型能够把 token 内容与位置绑定。
2. 残差连接显著改善训练稳定性。去掉残差后，即使仍使用正弦位置编码和完整注意力结构，模型也难以学会序列反转。

因此，经典 Transformer 的成功不是单个注意力公式的结果，而是注意力、位置编码、mask、残差归一化和前馈网络共同配合的结果。
"""

    Path("report.md").write_text(report, encoding="utf-8")
    print("Wrote report.md")


if __name__ == "__main__":
    main()
