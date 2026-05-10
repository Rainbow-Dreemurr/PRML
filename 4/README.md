# PRML Homework 4: Attention Is All You Need

This workspace contains a PyTorch reproduction of the core Transformer encoder-decoder architecture from *Attention Is All You Need* and an automated positional-encoding ablation experiment.

## Reproduce

Use the local `pytorch` conda environment:

```powershell
D:\Anaconda\envs\pytorch\python.exe run_experiments.py
D:\Anaconda\envs\pytorch\python.exe run_experiments.py --resume --seeds 0 1 2
D:\Anaconda\envs\pytorch\python.exe run_experiments.py --pe-kinds sinusoidal --seeds 0 1 2 --no-residual --out-dir outputs_no_residual
D:\Anaconda\envs\pytorch\python.exe visualize_attention.py
D:\Anaconda\envs\pytorch\python.exe summarize_results.py
D:\Anaconda\envs\pytorch\python.exe generate_report.py
```

The experiment runs on CUDA when available and compares:

- `none`: no positional encoding.
- `linear`: fixed simple absolute positional encoding.
- `learned`: learned absolute position embeddings.
- `sinusoidal`: the paper's sinusoidal positional encoding.

## Key Files

- `attention_is_all_you_need.pdf`: the assigned paper.
- `src/transformer.py`: from-scratch Multi-Head Attention, Encoder, Decoder, masks, residual paths, and FFN.
- `src/position_encodings.py`: all positional encoding variants.
- `src/data.py`: sequence reversal dataset.
- `run_experiments.py`: training/evaluation automation.
- `visualize_attention.py`: writes a decoder cross-attention heatmap.
- `summarize_results.py`: combines multi-seed and residual-ablation summaries.
- `generate_report.py`: writes `report.md` after experiments finish.
- `outputs_transformer/results.csv`: final metric table.
- `outputs_transformer/summary_by_variant.csv`: 3-seed mean/std for positional encodings.
- `outputs_transformer/residual_ablation_summary.csv`: residual vs no-residual summary.
- `outputs_transformer/figures/`: generated plots.
- `outputs_no_residual/`: residual-ablation checkpoints and logs.
