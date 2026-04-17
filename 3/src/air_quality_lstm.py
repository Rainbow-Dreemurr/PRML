"""Air quality forecasting with a small NumPy LSTM.

The script is intentionally self-contained so the homework can be reproduced
without TensorFlow or PyTorch. It trains chronological baselines and a
single-layer LSTM to forecast next-hour PM2.5 from prior multivariate hours.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


RNG_SEED = 42
FEATURE_COLUMNS = [
    "pollution",
    "dew",
    "temp",
    "press",
    "wnd_spd",
    "snow",
    "rain",
]


@dataclass
class MinMaxScaler:
    minimum: np.ndarray
    maximum: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "MinMaxScaler":
        minimum = values.min(axis=0)
        maximum = values.max(axis=0)
        return cls(minimum=minimum, maximum=maximum)

    def transform(self, values: np.ndarray) -> np.ndarray:
        scale = np.where(self.maximum - self.minimum == 0, 1.0, self.maximum - self.minimum)
        return (values - self.minimum) / scale

    def inverse_pollution(self, values: np.ndarray, pollution_index: int = 0) -> np.ndarray:
        return values * (self.maximum[pollution_index] - self.minimum[pollution_index]) + self.minimum[pollution_index]


def load_and_engineer(csv_path: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            date = datetime.fromisoformat(row["date"])
            rows.append(
                {
                    "date": date,
                    "pollution": float(row["pollution"]),
                    "dew": float(row["dew"]),
                    "temp": float(row["temp"]),
                    "press": float(row["press"]),
                    "wnd_spd": float(row["wnd_spd"]),
                    "snow": float(row["snow"]),
                    "rain": float(row["rain"]),
                    "wnd_dir": row["wnd_dir"],
                }
            )
    rows.sort(key=lambda item: item["date"])
    wind_dirs = sorted({row["wnd_dir"] for row in rows})
    feature_names = FEATURE_COLUMNS + ["hour_sin", "hour_cos", "month_sin", "month_cos"] + [
        f"wind_{direction}" for direction in wind_dirs
    ]
    values = []
    dates = []
    for row in rows:
        date = row["date"]
        hour = date.hour
        month = date.month
        base = [row[name] for name in FEATURE_COLUMNS]
        time_features = [
            math.sin(2 * math.pi * hour / 24),
            math.cos(2 * math.pi * hour / 24),
            math.sin(2 * math.pi * month / 12),
            math.cos(2 * math.pi * month / 12),
        ]
        wind_features = [1.0 if row["wnd_dir"] == direction else 0.0 for direction in wind_dirs]
        values.append(base + time_features + wind_features)
        dates.append(date)
    return np.asarray(dates, dtype=object), feature_names, np.asarray(values, dtype=np.float32)


def make_supervised(values: np.ndarray, dates: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, target_dates = [], [], []
    for i in range(lookback, len(values)):
        xs.append(values[i - lookback : i])
        ys.append(values[i, 0])
        target_dates.append(dates[i])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32).reshape(-1, 1), np.asarray(target_dates)


def chronological_splits(n_rows: int, lookback: int) -> tuple[int, int, int, int]:
    train_row_end = int(n_rows * 0.70)
    val_row_end = int(n_rows * 0.85)
    train_seq_end = train_row_end - lookback
    val_seq_end = val_row_end - lookback
    if train_seq_end <= 0 or val_seq_end <= train_seq_end:
        raise ValueError("Dataset is too short for the requested lookback.")
    return train_row_end, val_row_end, train_seq_end, val_seq_end


def evaluate_predictions(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | str]:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "model": name,
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(rmse),
        "R2": float(r2_score(y_true, y_pred)),
    }


def write_csv(path: Path, rows: list[dict[str, float | int | str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_metrics(metrics: list[dict[str, float | str]]) -> None:
    print(f"{'model':42s} {'MAE':>10s} {'RMSE':>10s} {'R2':>10s}")
    for row in metrics:
        print(f"{str(row['model']):42s} {row['MAE']:10.3f} {row['RMSE']:10.3f} {row['R2']:10.3f}")


class NumpyLSTMRegressor:
    """A compact many-to-one LSTM trained with Adam and truncated BPTT."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 32,
        learning_rate: float = 0.003,
        seed: int = RNG_SEED,
        grad_clip: float = 1.0,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        limit = 1.0 / math.sqrt(n_features + hidden_size)
        self.W = rng.uniform(-limit, limit, size=(n_features + hidden_size, 4 * hidden_size)).astype(np.float32)
        self.b = np.zeros((4 * hidden_size,), dtype=np.float32)
        self.b[hidden_size : 2 * hidden_size] = 1.0
        self.Wy = rng.uniform(-limit, limit, size=(hidden_size, 1)).astype(np.float32)
        self.by = np.zeros((1,), dtype=np.float32)
        self._adam_m = {name: np.zeros_like(value) for name, value in self.params().items()}
        self._adam_v = {name: np.zeros_like(value) for name, value in self.params().items()}
        self._step = 0

    def params(self) -> dict[str, np.ndarray]:
        return {"W": self.W, "b": self.b, "Wy": self.Wy, "by": self.by}

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -40, 40)
        return 1.0 / (1.0 + np.exp(-x))

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, list[dict[str, np.ndarray]]]:
        batch_size, steps, _ = x.shape
        h = np.zeros((batch_size, self.hidden_size), dtype=np.float32)
        c = np.zeros_like(h)
        cache: list[dict[str, np.ndarray]] = []
        for t in range(steps):
            z = np.concatenate([x[:, t, :], h], axis=1)
            gates = z @ self.W + self.b
            i = self._sigmoid(gates[:, : self.hidden_size])
            f = self._sigmoid(gates[:, self.hidden_size : 2 * self.hidden_size])
            o = self._sigmoid(gates[:, 2 * self.hidden_size : 3 * self.hidden_size])
            g = np.tanh(gates[:, 3 * self.hidden_size :])
            c_prev = c
            h_prev = h
            c = f * c + i * g
            h = o * np.tanh(c)
            cache.append({"z": z, "i": i, "f": f, "o": o, "g": g, "c": c, "c_prev": c_prev, "h_prev": h_prev})
        y_hat = h @ self.Wy + self.by
        return y_hat, cache

    def predict(self, x: np.ndarray, batch_size: int = 512) -> np.ndarray:
        preds = []
        for start in range(0, len(x), batch_size):
            y_hat, _ = self._forward(x[start : start + batch_size])
            preds.append(y_hat)
        return np.vstack(preds)

    def _backward(self, dy: np.ndarray, cache: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        grads = {name: np.zeros_like(value) for name, value in self.params().items()}
        last_h = cache[-1]["o"] * np.tanh(cache[-1]["c"])
        grads["Wy"] = last_h.T @ dy
        grads["by"] = dy.sum(axis=0)

        dh = dy @ self.Wy.T
        dc_next = np.zeros_like(dh)
        for t in reversed(range(len(cache))):
            item = cache[t]
            i, f, o, g, c = item["i"], item["f"], item["o"], item["g"], item["c"]
            tanh_c = np.tanh(c)
            do = dh * tanh_c
            dc = dh * o * (1 - tanh_c**2) + dc_next
            df = dc * item["c_prev"]
            di = dc * g
            dg = dc * i
            dc_next = dc * f

            gates_grad = np.concatenate(
                [
                    di * i * (1 - i),
                    df * f * (1 - f),
                    do * o * (1 - o),
                    dg * (1 - g**2),
                ],
                axis=1,
            )
            grads["W"] += item["z"].T @ gates_grad
            grads["b"] += gates_grad.sum(axis=0)
            dz = gates_grad @ self.W.T
            dh = dz[:, self.n_features :] + 0.0 * item["h_prev"]

        for key in grads:
            grads[key] = np.clip(grads[key], -self.grad_clip, self.grad_clip)
        return grads

    def _adam_update(self, grads: dict[str, np.ndarray]) -> None:
        self._step += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for name, param in self.params().items():
            self._adam_m[name] = beta1 * self._adam_m[name] + (1 - beta1) * grads[name]
            self._adam_v[name] = beta2 * self._adam_v[name] + (1 - beta2) * (grads[name] ** 2)
            m_hat = self._adam_m[name] / (1 - beta1**self._step)
            v_hat = self._adam_v[name] / (1 - beta2**self._step)
            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 8,
        batch_size: int = 128,
    ) -> list[dict[str, float | int]]:
        rng = np.random.default_rng(RNG_SEED)
        history = []
        for epoch in range(1, epochs + 1):
            order = rng.permutation(len(x_train))
            batch_losses = []
            for start in range(0, len(order), batch_size):
                idx = order[start : start + batch_size]
                x_batch = x_train[idx]
                y_batch = y_train[idx]
                y_hat, cache = self._forward(x_batch)
                error = y_hat - y_batch
                batch_losses.append(float(np.mean(error**2)))
                dy = (2.0 / len(x_batch)) * error
                grads = self._backward(dy, cache)
                self._adam_update(grads)
            train_mse = float(np.mean(batch_losses))
            val_pred = self.predict(x_val)
            val_mse = float(np.mean((val_pred - y_val) ** 2))
            history.append({"epoch": epoch, "train_mse_scaled": train_mse, "val_mse_scaled": val_mse})
            print(f"epoch {epoch:02d} | train_mse_scaled={train_mse:.5f} | val_mse_scaled={val_mse:.5f}", flush=True)
        return history


def train_torch_lstm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[dict[str, float | int]], str]:
    warnings.filterwarnings("ignore", message="Failed to initialize NumPy.*")
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    np.random.seed(RNG_SEED)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    class TorchLSTMRegressor(nn.Module):
        def __init__(self, n_features: int, hidden_size: int, num_layers: int, dropout: float) -> None:
            super().__init__()
            effective_dropout = dropout if num_layers > 1 else 0.0
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=effective_dropout,
            )
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            output, _ = self.lstm(x)
            return self.head(output[:, -1, :])

    # This avoids torch.from_numpy because the provided PyTorch 2.0
    # environment may be paired with NumPy 2.x, which disables that bridge.
    train_ds = TensorDataset(
        torch.tensor(x_train.tolist(), dtype=torch.float32),
        torch.tensor(y_train.tolist(), dtype=torch.float32),
    )
    val_x = torch.tensor(x_val.tolist(), dtype=torch.float32, device=device)
    val_y = torch.tensor(y_val.tolist(), dtype=torch.float32, device=device)
    test_x = torch.tensor(x_test.tolist(), dtype=torch.float32, device=device)

    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    model = TorchLSTMRegressor(
        n_features=x_train.shape[-1],
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    history = []
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            val_mse = float(loss_fn(model(val_x), val_y).detach().cpu())
        train_mse = float(np.mean(losses))
        history.append({"epoch": epoch, "train_mse_scaled": train_mse, "val_mse_scaled": val_mse})
        if val_mse < best_val:
            best_val = val_mse
            best_state = copy.deepcopy(model.state_dict())
        print(f"epoch {epoch:02d} | train_mse_scaled={train_mse:.5f} | val_mse_scaled={val_mse:.5f}", flush=True)

    model.load_state_dict(best_state)
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(test_x), args.batch_size * 4):
            preds.extend(model(test_x[start : start + args.batch_size * 4]).detach().cpu().view(-1).tolist())
    return np.asarray(preds, dtype=np.float32).reshape(-1, 1), history, str(device)


def plot_series(dates: np.ndarray, y_true: np.ndarray, predictions: dict[str, np.ndarray], output: Path, limit: int = 500) -> None:
    plt.figure(figsize=(13, 5))
    sl = slice(0, min(limit, len(y_true)))
    plt.plot(dates[sl], y_true[sl], label="Actual", linewidth=1.7)
    for name, pred in predictions.items():
        plt.plot(dates[sl], pred[sl], label=name, linewidth=1.2, alpha=0.85)
    plt.title("Next-hour PM2.5 Forecast on Chronological Test Set")
    plt.xlabel("Date")
    plt.ylabel("PM2.5 concentration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def plot_history(history: list[dict[str, float | int]], output: Path) -> None:
    plt.figure(figsize=(8, 4.5))
    epochs = [item["epoch"] for item in history]
    train_mse = [item["train_mse_scaled"] for item in history]
    val_mse = [item["val_mse_scaled"] for item in history]
    plt.plot(epochs, train_mse, marker="o", label="Train")
    plt.plot(epochs, val_mse, marker="o", label="Validation")
    plt.title("LSTM Training Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Scaled MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def run_experiment(args: argparse.Namespace) -> None:
    root = Path(args.root)
    results_dir = root / "results"
    results_dir.mkdir(exist_ok=True)

    dates, feature_names, raw_values = load_and_engineer(root / args.data)
    n_rows = len(raw_values)
    train_row_end, val_row_end, train_seq_end, val_seq_end = chronological_splits(n_rows, args.lookback)
    scaler = MinMaxScaler.fit(raw_values[:train_row_end])
    scaled_values = scaler.transform(raw_values).astype(np.float32)
    x, y_scaled, target_dates = make_supervised(scaled_values, dates, args.lookback)

    x_train, y_train = x[:train_seq_end], y_scaled[:train_seq_end]
    x_val, y_val = x[train_seq_end:val_seq_end], y_scaled[train_seq_end:val_seq_end]
    x_test, y_test_scaled = x[val_seq_end:], y_scaled[val_seq_end:]
    test_dates = target_dates[val_seq_end:]
    y_test = scaler.inverse_pollution(y_test_scaled).reshape(-1)

    last_pollution_scaled = x_test[:, -1, 0]
    persistence_pred = scaler.inverse_pollution(last_pollution_scaled).reshape(-1)

    ridge = Ridge(alpha=args.ridge_alpha)
    ridge.fit(x_train.reshape(len(x_train), -1), y_train.reshape(-1))
    ridge_pred_scaled = ridge.predict(x_test.reshape(len(x_test), -1)).reshape(-1, 1)
    ridge_pred = scaler.inverse_pollution(ridge_pred_scaled).reshape(-1)

    backend = args.backend
    if backend == "auto":
        try:
            import torch  # noqa: F401

            backend = "torch"
        except ImportError:
            backend = "numpy"

    if backend == "torch":
        lstm_pred_scaled, history, device_name = train_torch_lstm(x_train, y_train, x_val, y_val, x_test, args)
        lstm_name = "PyTorch LSTM (multivariate history)"
    elif backend == "numpy":
        device_name = "cpu"
        lstm = NumpyLSTMRegressor(
            n_features=x_train.shape[-1],
            hidden_size=args.hidden_size,
            learning_rate=args.learning_rate,
            seed=RNG_SEED,
        )
        history = lstm.fit(x_train, y_train, x_val, y_val, epochs=args.epochs, batch_size=args.batch_size)
        lstm_pred_scaled = lstm.predict(x_test)
        lstm_name = "NumPy LSTM (multivariate history)"
    else:
        raise ValueError("--backend must be one of: auto, torch, numpy")

    lstm_pred = scaler.inverse_pollution(lstm_pred_scaled).reshape(-1)

    metrics = [
        evaluate_predictions("Persistence (previous hour)", y_test, persistence_pred),
        evaluate_predictions("Ridge regression (flattened history)", y_test, ridge_pred),
        evaluate_predictions(lstm_name, y_test, lstm_pred),
    ]
    write_csv(results_dir / "metrics.csv", metrics, ["model", "MAE", "RMSE", "R2"])

    with (results_dir / "experiment_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "rows": n_rows,
                "date_start": str(dates[0]),
                "date_end": str(dates[-1]),
                "lookback_hours": args.lookback,
                "feature_names": feature_names,
                "train_sequences": len(x_train),
                "validation_sequences": len(x_val),
                "test_sequences": len(x_test),
                "split_dates": {
                    "train_end_exclusive": str(dates[train_row_end]),
                    "validation_end_exclusive": str(dates[val_row_end]),
                },
                "lstm": {
                    "backend": backend,
                    "device": device_name,
                    "hidden_size": args.hidden_size,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                },
            },
            f,
            indent=2,
        )

    write_csv(results_dir / "training_history.csv", history, ["epoch", "train_mse_scaled", "val_mse_scaled"])
    plot_series(
        test_dates,
        y_test,
        {
            "Persistence": persistence_pred,
            "Ridge": ridge_pred,
            "LSTM": lstm_pred,
        },
        results_dir / "forecast_test.png",
    )
    plot_history(history, results_dir / "training_curve.png")

    print("\nMetrics on chronological test set:")
    print_metrics(metrics)
    print(f"\nSaved results to {results_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate next-hour PM2.5 forecasting models.")
    parser.add_argument("--root", default=".", help="Project root directory.")
    parser.add_argument("--data", default="LSTM-Multivariate_pollution.csv", help="Input CSV file.")
    parser.add_argument("--lookback", type=int, default=24, help="Number of prior hours used as model input.")
    parser.add_argument("--epochs", type=int, default=8, help="LSTM training epochs.")
    parser.add_argument("--hidden-size", type=int, default=32, help="LSTM hidden state size.")
    parser.add_argument("--num-layers", type=int, default=1, help="Number of stacked LSTM layers for PyTorch backend.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout between LSTM layers when num_layers > 1.")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=0.003, help="Adam learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Adam weight decay for PyTorch backend.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm.")
    parser.add_argument("--backend", choices=["auto", "torch", "numpy"], default="auto", help="LSTM backend.")
    parser.add_argument("--device", default="auto", help="PyTorch device: auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--ridge-alpha", type=float, default=1.0, help="Ridge baseline regularization.")
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
