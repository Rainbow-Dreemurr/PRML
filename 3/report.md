# 模式识别与机器学习第三次作业：基于 LSTM 的空气质量多变量预测

## 1. 任务目标

本作业使用 Kaggle Air Quality 数据集，研究如何用 LSTM 同时利用污染物历史值和气象变量，预测下一小时的 PM2.5 浓度。预测问题定义为：

> 给定过去 24 小时的污染物浓度、露点、温度、气压、风向、累计风速、累计降雪小时数、累计降雨小时数等多变量序列，预测下一小时的 PM2.5 浓度。

数据集主文件为 `LSTM-Multivariate_pollution.csv`，共 43,800 条小时级记录，时间范围为 2010-01-02 00:00:00 到 2014-12-31 23:00:00。

## 2. 数据处理

原始字段包括：

- `date`: 时间戳
- `pollution`: PM2.5 浓度，也是预测目标
- `dew`: 露点
- `temp`: 温度
- `press`: 气压
- `wnd_dir`: 风向
- `wnd_spd`: 累计风速
- `snow`: 累计降雪小时数
- `rain`: 累计降雨小时数

预处理步骤如下：

1. 按 `date` 升序排列，保持时间序列的因果顺序。
2. 将风向 `wnd_dir` 做 one-hot 编码，得到 `wind_NE`、`wind_NW`、`wind_SE`、`wind_cv`。
3. 从时间戳中提取小时和月份，并用正余弦编码表示周期性：
   - `hour_sin`, `hour_cos`
   - `month_sin`, `month_cos`
4. 使用训练集拟合 Min-Max 缩放器，再应用到验证集和测试集，避免未来信息泄漏。
5. 构造监督学习样本：每个样本输入过去 24 小时的多变量序列，标签为下一小时 PM2.5。

最终输入特征数为 15，分别为：

`pollution, dew, temp, press, wnd_spd, snow, rain, hour_sin, hour_cos, month_sin, month_cos, wind_NE, wind_NW, wind_SE, wind_cv`

## 3. 实验设计

时间序列预测不能随机划分样本，否则训练集可能看到测试集之后的信息。因此，本实验采用严格的时间顺序划分：

| 集合 | 样本数 | 时间范围说明 |
|---|---:|---|
| 训练集 | 30,635 | 约前 70% 时间段 |
| 验证集 | 6,571 | 中间 15% 时间段 |
| 测试集 | 6,570 | 最后 15% 时间段 |

训练集结束边界为 2013-07-02 11:00:00，验证集结束边界为 2014-04-02 06:00:00。

对比模型包括：

- Persistence baseline：直接使用上一小时 PM2.5 作为下一小时预测。
- Ridge regression：将过去 24 小时多变量历史展平成向量，训练线性岭回归。
- PyTorch LSTM：用 LSTM 学习多变量序列的非线性时间依赖。

## 4. LSTM 模型结构

最终 LSTM 配置如下：

| 参数 | 设置 |
|---|---:|
| lookback | 24 小时 |
| input size | 15 |
| hidden size | 64 |
| LSTM layers | 1 |
| dropout | 0 |
| optimizer | Adam |
| learning rate | 0.001 |
| weight decay | 0.00001 |
| batch size | 256 |
| epochs | 20 |
| device | CUDA |
| loss | MSE |

模型采用 many-to-one 结构：输入 24 个时间步，取最后一个时间步的隐藏状态，通过全连接层输出下一小时 PM2.5 的缩放值。训练时监控验证集 MSE，并保存验证集表现最好的模型参数用于测试。

## 5. 实验结果

测试集指标如下：

| 模型 | MAE | RMSE | R2 |
|---|---:|---:|---:|
| Persistence, previous hour | 11.457 | 22.635 | 0.917 |
| Ridge regression, flattened history | 12.016 | 21.968 | 0.922 |
| PyTorch LSTM, multivariate history | 11.976 | 21.925 | 0.922 |

从结果可以看到：

1. Persistence baseline 已经很强，因为小时级 PM2.5 有明显连续性，上一小时浓度通常是下一小时浓度的强预测因子。
2. Ridge regression 利用 24 小时多变量历史后，RMSE 从 22.635 降到 21.968，说明天气变量和更长历史窗口确实提供了额外信息。
3. LSTM 的 RMSE 为 21.925，是三个模型中最低的；R2 也最高，为 0.922。提升幅度不大，但说明 LSTM 学到了一部分线性模型难以完全表示的非线性时序模式。
4. LSTM 的 MAE 略高于 Persistence baseline，说明其平均绝对误差没有全面压低上一小时基线；但 RMSE 更低，表示它对部分较大误差的控制更好。

训练曲线见 `results/training_curve.png`，测试集前 500 个时间点的预测曲线见 `results/forecast_test.png`。

## 6. 结论

本实验完整实现了多变量空气质量预测流程：数据清洗、时间特征构造、风向编码、时序窗口转换、基线模型、PyTorch LSTM、指标评估和可视化。实验结果表明，对于小时级 PM2.5 预测，简单的上一小时基线已经非常强；引入多变量历史能够进一步降低 RMSE，而 LSTM 在本实验配置下取得最低 RMSE 和最高 R2，验证了 LSTM 对多变量时间序列预测任务的适用性。

未来可继续尝试：

- 使用更长 lookback，例如 48 或 72 小时。
- 调整 LSTM 层数、hidden size 和 dropout。
- 加入早停、学习率调度器或 Huber loss。
- 对极端污染峰值单独分析，提升高污染时段预测能力。

## 7. 复现命令

```powershell
conda run -n pytorch python src/air_quality_lstm.py --backend torch --device cuda --epochs 20 --hidden-size 64 --batch-size 256 --learning-rate 0.001 --weight-decay 0.00001
```

说明：当前 `pytorch` 环境中 PyTorch 2.0.0 与 NumPy 2.0.2 存在 ABI 提示，运行时会输出 NumPy 兼容性警告。代码已避免使用 `torch.from_numpy`，实验可以正常完成。若希望消除该提示，可将该环境中的 NumPy 降到 1.x 版本或升级 PyTorch 到支持 NumPy 2.x 的版本。
