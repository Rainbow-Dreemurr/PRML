# Air Quality PM2.5 Forecasting with LSTM

This project completes the third PRML homework: forecasting next-hour PM2.5
from prior multivariate air-quality and weather observations.

## Files

- `LSTM-Multivariate_pollution.csv`: hourly Beijing air-quality dataset from 2010-01-02 to 2014-12-31.
- `pollution_test_data1.csv`: additional sample/test CSV shipped with the Kaggle dataset.
- `src/air_quality_lstm.py`: full preprocessing, baseline, LSTM training, evaluation, and plotting pipeline.
- `results/metrics.csv`: final test-set MAE, RMSE, and R2.
- `results/training_history.csv`: LSTM scaled MSE per epoch.
- `results/training_curve.png`: training and validation loss curve.
- `results/forecast_test.png`: first 500 chronological test predictions.
- `report.md`: homework report in Chinese.

## Reproduce

Use the local Conda environment named `pytorch`:

```powershell
conda run -n pytorch python src/air_quality_lstm.py --backend torch --device cuda --epochs 20 --hidden-size 64 --batch-size 256 --learning-rate 0.001 --weight-decay 0.00001
```

If CUDA is unavailable, use `--device cpu`.

The script does not require pandas. It uses standard-library CSV parsing,
NumPy, scikit-learn, matplotlib, and PyTorch. In the current `pytorch`
environment, PyTorch 2.0.0 is paired with NumPy 2.0.2, so PyTorch prints a
NumPy ABI warning. The script avoids `torch.from_numpy`, and the experiment
still completes normally.

## Final Result

Chronological test set:

| Model | MAE | RMSE | R2 |
|---|---:|---:|---:|
| Persistence (previous hour) | 11.457 | 22.635 | 0.917 |
| Ridge regression (flattened history) | 12.016 | 21.968 | 0.922 |
| PyTorch LSTM (multivariate history) | 11.976 | 21.925 | 0.922 |

The PyTorch LSTM obtains the lowest RMSE and the highest R2 among the tested
models, showing a small but measurable benefit from learning nonlinear
temporal patterns over the multivariate 24-hour history.
