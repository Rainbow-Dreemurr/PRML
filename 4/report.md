# 模式识别与机器学习第四次作业：Transformer 复现与模块消融实验

## 1. 论文阅读与核心结构复现

本作业阅读并复现 Vaswani 等人在 2017 年提出的 **Attention Is All You Need**。论文的核心思想是完全使用注意力机制进行序列建模，用自注意力替代循环网络或卷积网络中的显式时序递推。代码没有直接调用 `torch.nn.Transformer`，而是在 [src/transformer.py](src/transformer.py) 中从头实现了以下模块：

1. Scaled Dot-Product Attention：使用 $QK^T/\sqrt{d_k}$ 计算注意力分布。
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
- 训练长度：4 到 12。
- IID 测试：长度仍为 4 到 12。
- OOD 测试：更长的 13 到 18，用于观察长度外推。

该任务需要模型知道每个 token 的位置。若没有位置编码，自注意力主要看到的是 token 内容集合，很难稳定区分同一组 token 的不同排列。

主要超参数如下：

| 项目 | 数值 |
|---|---:|
| vocab size | 32 |
| train/val/test samples | 12000/2000/2000 |
| d_model | 128 |
| heads | 4 |
| layers | 2 |
| d_ff | 512 |
| batch size | 128 |
| epochs | 14 |
| optimizer | Adam + Transformer Noam learning-rate schedule |
| label smoothing | 0.1 |
| seeds | 0, 1, 2 |
| device | cuda, NVIDIA RTX 1000 Ada Generation Laptop GPU |

运行命令：

```powershell
D:\Anaconda\envs\pytorch\python.exe run_experiments.py --resume --seeds 0 1 2
D:\Anaconda\envs\pytorch\python.exe run_experiments.py --pe-kinds sinusoidal --seeds 0 1 2 --no-residual --out-dir outputs_no_residual
D:\Anaconda\envs\pytorch\python.exe visualize_attention.py
D:\Anaconda\envs\pytorch\python.exe summarize_results.py
D:\Anaconda\envs\pytorch\python.exe generate_report.py
```

## 3. 位置编码消融结果

下表为 3 个随机种子的均值 ± 标准差。共完成 12 次主实验训练。

| 位置编码 | IID token acc | IID exact match | OOD token acc | OOD exact match | best val exact |
|---|---:|---:|---:|---:|---:|
| none | 0.415 ± 0.017 | 0.007 ± 0.002 | 0.264 ± 0.025 | 0.000 ± 0.000 | 0.010 ± 0.002 |
| linear | 0.988 ± 0.002 | 0.897 ± 0.020 | 0.648 ± 0.103 | 0.181 ± 0.090 | 0.909 ± 0.013 |
| learned | 0.997 ± 0.002 | 0.969 ± 0.015 | 0.357 ± 0.021 | 0.000 ± 0.000 | 0.970 ± 0.010 |
| sinusoidal | 0.997 ± 0.000 | 0.975 ± 0.003 | 0.431 ± 0.016 | 0.098 ± 0.047 | 0.971 ± 0.002 |

验证集曲线：

![validation](outputs_transformer/figures/validation_exact_match.png)

最终准确率：

![final](outputs_transformer/figures/final_exact_match.png)

结论：无位置编码的 `none` 在 IID exact match 上只有约 0.007，说明模型几乎不能完成完整序列反转。`linear`、`learned`、`sinusoidal` 都能显著提升训练长度内的精确匹配率，其中 learned 和 sinusoidal 在 IID 上达到约 0.97。OOD 长度上整体下降明显，说明位置编码提供了顺序信息，但长度外推还受到训练长度覆盖、自回归误差累积和任务难度的共同影响。

## 4. 注意力可视化

使用已训练的 sinusoidal seed0 模型，取最后一层 Decoder 的 cross-attention，并对所有 head 求平均。下图横轴为源序列 token，纵轴为当前要生成的目标 token；白色空心点标出了反转任务中理论上应关注的源位置。

![attention](outputs_transformer/figures/sinusoidal_cross_attention_seed0.png)

该样例的源序列为：

`[11, 22, 20, 21, 6, 30, 14, 27, 8, 11, 14, 21, 2]`

目标序列为：

`[21, 14, 11, 8, 27, 14, 30, 6, 21, 20, 22, 11, 2]`

可视化说明：热力图中的高权重区域整体沿“反向对应关系”分布，说明 Decoder 在生成反转序列时确实倾向于从源序列后部向前读取信息。这为数值指标提供了更直观的解释。

## 5. 残差结构消融结果

为了补充题目 2.3，固定使用论文正弦位置编码，只改变是否使用残差连接。无残差模型仍保留注意力、FFN、LayerNorm 和 mask，但每个子层不再执行 `x + sublayer(x)`。

| 结构 | IID token acc | IID exact match | OOD token acc | OOD exact match |
|---|---:|---:|---:|---:|
| sinusoidal_with_residual | 0.997 ± 0.000 | 0.975 ± 0.003 | 0.431 ± 0.016 | 0.098 ± 0.047 |
| sinusoidal_no_residual | 0.169 ± 0.003 | 0.000 ± 0.000 | 0.127 ± 0.002 | 0.000 ± 0.000 |

![residual](outputs_transformer/figures/residual_ablation.png)

结果非常明显：有残差时 sinusoidal 模型的 IID exact match 约为 0.975；去掉残差后 3 个 seed 的 exact match 都为 0。训练日志中无残差模型的 token acc 也长期停在较低水平。这说明残差结构不是单纯的工程细节，它为深层注意力网络提供了信息直通路径和更稳定的梯度传播，使注意力层、交叉注意力层和前馈层可以逐步修正表示，而不是每一层都强行重写表示。

## 6. 定性样例

**none_seed0**
- src=[11, 22, 20, 21, 6, 30, 14, 27, 8, 11, 14, 21, 2]  target=[21, 14, 11, 8, 27, 14, 30, 6, 21, 20, 22, 11, 2]  pred=[14, 21, 11, 22, 8, 21, 6, 30, 21, 27, 2]
- src=[11, 11, 20, 27, 19, 14, 28, 20, 10, 2]  target=[10, 20, 28, 14, 19, 27, 20, 11, 11, 2]  pred=[11, 20, 14, 28, 19, 27, 10, 20, 11, 2]
- src=[8, 17, 7, 29, 10, 2]  target=[10, 29, 7, 17, 8, 2]  pred=[10, 29, 17, 7, 8, 2]
**sinusoidal_seed0**
- src=[11, 22, 20, 21, 6, 30, 14, 27, 8, 11, 14, 21, 2]  target=[21, 14, 11, 8, 27, 14, 30, 6, 21, 20, 22, 11, 2]  pred=[21, 14, 11, 8, 27, 14, 30, 6, 21, 20, 22, 11, 2]
- src=[11, 11, 20, 27, 19, 14, 28, 20, 10, 2]  target=[10, 20, 28, 14, 19, 27, 20, 11, 11, 2]  pred=[10, 20, 28, 14, 19, 27, 20, 11, 11, 2]
- src=[8, 17, 7, 29, 10, 2]  target=[10, 29, 7, 17, 8, 2]  pred=[10, 29, 7, 17, 8, 2]

## 7. 总结

本次作业完成了 Transformer encoder-decoder 的核心复现，并通过位置编码、注意力可视化和残差结构消融验证了两个关键结论：

1. 自注意力擅长内容寻址，但自身不携带顺序。位置编码的核心作用是打破排列对称性，使模型能够把 token 内容与位置绑定。
2. 残差连接显著改善训练稳定性。去掉残差后，即使仍使用正弦位置编码和完整注意力结构，模型也难以学会序列反转。

因此，经典 Transformer 的成功不是单个注意力公式的结果，而是注意力、位置编码、mask、残差归一化和前馈网络共同配合的结果。
