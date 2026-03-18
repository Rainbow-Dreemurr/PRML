from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
DATA_FILENAME = "Data4Regression.xlsx"


@dataclass
class Dataset:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    source_path: Path


def find_data_file() -> Path:
    local = ROOT / DATA_FILENAME
    if local.exists():
        return local

    desktop = Path.home() / "Desktop"
    matches = list(desktop.rglob(DATA_FILENAME))
    if not matches:
        raise FileNotFoundError(f"Could not find {DATA_FILENAME} under {desktop}.")
    return matches[0]


def load_dataset() -> Dataset:
    path = find_data_file()
    train_df = pd.read_excel(path, sheet_name="Training Data")
    test_df = pd.read_excel(path, sheet_name="Test Data")
    return Dataset(
        x_train=train_df["x"].to_numpy(dtype=float),
        y_train=train_df["y_complex"].to_numpy(dtype=float),
        x_test=test_df["x_new"].to_numpy(dtype=float),
        y_test=test_df["y_new_complex"].to_numpy(dtype=float),
        source_path=path,
    )


def design_matrix_linear(x: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones_like(x), x])


def design_matrix_poly(x: np.ndarray, degree: int) -> np.ndarray:
    return np.column_stack([x ** power for power in range(degree + 1)])


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def least_squares_fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    X = design_matrix_linear(x)
    return np.linalg.solve(X.T @ X, X.T @ y)


def gradient_descent_fit(
    x: np.ndarray,
    y: np.ndarray,
    max_iter: int = 5000,
    tol: float = 1e-10,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    X = design_matrix_linear(x)
    n_samples = X.shape[0]
    hessian = (2.0 / n_samples) * (X.T @ X)
    lipschitz = float(np.linalg.eigvalsh(hessian).max())
    learning_rate = 0.9 / lipschitz

    w = np.zeros(X.shape[1], dtype=float)
    history: list[dict[str, float]] = []

    for step in range(max_iter):
        residual = X @ w - y
        train_mse = float(np.mean(residual**2))
        history.append({"iteration": step, "train_mse": train_mse})
        gradient = (2.0 / n_samples) * (X.T @ residual)
        next_w = w - learning_rate * gradient
        if np.linalg.norm(next_w - w) < tol:
            w = next_w
            break
        w = next_w

    final_residual = X @ w - y
    history.append({"iteration": len(history), "train_mse": float(np.mean(final_residual**2))})
    return w, history


def newton_fit(
    x: np.ndarray,
    y: np.ndarray,
    max_iter: int = 5,
    tol: float = 1e-12,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    X = design_matrix_linear(x)
    n_samples = X.shape[0]
    hessian = (2.0 / n_samples) * (X.T @ X)
    w = np.zeros(X.shape[1], dtype=float)
    history: list[dict[str, float]] = []

    for step in range(max_iter):
        residual = X @ w - y
        train_mse = float(np.mean(residual**2))
        history.append({"iteration": step, "train_mse": train_mse})
        gradient = (2.0 / n_samples) * (X.T @ residual)
        delta = np.linalg.solve(hessian, gradient)
        next_w = w - delta
        if np.linalg.norm(next_w - w) < tol:
            w = next_w
            break
        w = next_w

    final_residual = X @ w - y
    history.append({"iteration": len(history), "train_mse": float(np.mean(final_residual**2))})
    return w, history


def predict_linear(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return design_matrix_linear(x) @ weights


def k_fold_indices(n_samples: int, n_splits: int = 5, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    folds = np.array_split(indices, n_splits)
    split_indices: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in folds:
        valid_idx = fold
        train_idx = np.setdiff1d(indices, valid_idx)
        split_indices.append((train_idx, valid_idx))
    return split_indices


def ridge_closed_form(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    reg = np.eye(X.shape[1], dtype=float) * lam
    reg[0, 0] = 0.0
    return np.linalg.solve(X.T @ X + reg, X.T @ y)


def rbf_design_matrix(x: np.ndarray, centers: np.ndarray, sigma: float) -> np.ndarray:
    sq_dist = (x[:, None] - centers[None, :]) ** 2
    basis = np.exp(-sq_dist / (2.0 * sigma**2))
    return np.column_stack([np.ones_like(x), basis])


def select_rbf_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[dict[str, float], pd.DataFrame, np.ndarray]:
    center_counts = [5, 8, 10, 12, 15, 20, 25, 30]
    sigmas = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
    lambdas = [1e-6, 1e-4, 1e-2, 1e-1, 1.0]
    cv_splits = k_fold_indices(len(x_train), n_splits=5, seed=0)
    rows: list[dict[str, float]] = []
    best: dict[str, float] | None = None

    for center_count in center_counts:
        centers = np.linspace(x_train.min(), x_train.max(), center_count)
        for sigma in sigmas:
            for lam in lambdas:
                fold_errors = []
                for train_idx, valid_idx in cv_splits:
                    X_fit = rbf_design_matrix(x_train[train_idx], centers, sigma)
                    X_valid = rbf_design_matrix(x_train[valid_idx], centers, sigma)
                    weights = ridge_closed_form(X_fit, y_train[train_idx], lam)
                    pred_valid = X_valid @ weights
                    fold_errors.append(mse(y_train[valid_idx], pred_valid))

                cv_mse = float(np.mean(fold_errors))
                X_full = rbf_design_matrix(x_train, centers, sigma)
                weights = ridge_closed_form(X_full, y_train, lam)
                train_mse = mse(y_train, X_full @ weights)
                test_mse = mse(y_test, rbf_design_matrix(x_test, centers, sigma) @ weights)
                row = {
                    "center_count": float(center_count),
                    "sigma": float(sigma),
                    "lambda": float(lam),
                    "cv_mse": cv_mse,
                    "train_mse": train_mse,
                    "test_mse": test_mse,
                }
                rows.append(row)

                if best is None or cv_mse < best["cv_mse"]:
                    best = row

    assert best is not None
    best_center_count = int(best["center_count"])
    best_sigma = float(best["sigma"])
    best_lambda = float(best["lambda"])
    best_centers = np.linspace(x_train.min(), x_train.max(), best_center_count)
    best_weights = ridge_closed_form(rbf_design_matrix(x_train, best_centers, best_sigma), y_train, best_lambda)
    results_df = pd.DataFrame(rows).sort_values(["cv_mse", "test_mse"]).reset_index(drop=True)
    return best, results_df, best_weights


def save_curve_plot(
    dataset: Dataset,
    linear_weights: dict[str, np.ndarray],
    rbf_centers: np.ndarray,
    rbf_sigma: float,
    rbf_weights: np.ndarray,
) -> None:
    x_grid = np.linspace(
        min(dataset.x_train.min(), dataset.x_test.min()),
        max(dataset.x_train.max(), dataset.x_test.max()),
        400,
    )
    plt.figure(figsize=(10, 6))
    plt.scatter(dataset.x_train, dataset.y_train, s=25, label="Training data", alpha=0.75)
    plt.scatter(dataset.x_test, dataset.y_test, s=25, label="Test data", alpha=0.75)
    for name, weights in linear_weights.items():
        plt.plot(x_grid, predict_linear(x_grid, weights), linewidth=2, label=f"Linear ({name})")
    plt.plot(x_grid, rbf_design_matrix(x_grid, rbf_centers, rbf_sigma) @ rbf_weights, linewidth=3, label="RBF ridge")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Regression fits on training and test data")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "fit_comparison.png", dpi=200)
    plt.close()


def save_optimization_plot(gd_history: list[dict[str, float]], newton_history: list[dict[str, float]]) -> None:
    gd_df = pd.DataFrame(gd_history)
    newton_df = pd.DataFrame(newton_history)
    plt.figure(figsize=(10, 6))
    plt.plot(gd_df["iteration"], gd_df["train_mse"], label="Gradient descent", linewidth=2)
    plt.plot(newton_df["iteration"], newton_df["train_mse"], marker="o", label="Newton method", linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Training MSE")
    plt.title("Optimization behavior of GD and Newton method")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "optimization_curves.png", dpi=200)
    plt.close()


def save_model_selection_plot(model_selection_df: pd.DataFrame) -> None:
    best_per_center = (
        model_selection_df.sort_values(["center_count", "cv_mse"])
        .groupby("center_count", as_index=False)
        .first()
        .sort_values("center_count")
    )
    plt.figure(figsize=(10, 6))
    plt.plot(best_per_center["center_count"], best_per_center["cv_mse"], marker="o", label="Best CV MSE")
    plt.plot(best_per_center["center_count"], best_per_center["test_mse"], marker="s", label="Test MSE")
    plt.xlabel("Number of RBF centers")
    plt.ylabel("MSE")
    plt.title("RBF model selection by center count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "rbf_selection.png", dpi=200)
    plt.close()


def write_report(
    dataset: Dataset,
    metrics_df: pd.DataFrame,
    best_rbf: dict[str, float],
    gd_history: list[dict[str, float]],
    newton_history: list[dict[str, float]],
) -> None:
    linear_rows = metrics_df[metrics_df["model"].isin(["Least Squares", "Gradient Descent", "Newton Method"])]
    nonlinear_row = metrics_df[metrics_df["model"] == "RBF Ridge"].iloc[0]
    gd_iters = int(pd.DataFrame(gd_history)["iteration"].max())
    newton_iters = int(pd.DataFrame(newton_history)["iteration"].max())

    report = f"""# 模式识别与机器学习第一次作业实验报告

## 1. 实验目标

给定 `Data4Regression.xlsx` 中的训练集与测试集，完成以下任务：

1. 使用最小二乘法、梯度下降法（GD）和牛顿法对数据做线性拟合，并观察训练误差与测试误差。
2. 在线性模型拟合效果不理想的前提下，寻找更合适的非线性模型，给出模型选择原因、实验结果与分析。

## 2. 数据说明

- 数据源：`{dataset.source_path}`
- 训练集样本数：`{len(dataset.x_train)}`
- 测试集样本数：`{len(dataset.x_test)}`
- 输入变量：一维连续变量 `x`
- 输出变量：目标值 `y`

从散点图可以看出，数据整体呈现明显的弯曲波动趋势，而不是近似直线关系，因此线性模型预计只能提供较粗糙的拟合结果。

## 3. 实验方法

### 3.1 线性模型

统一采用线性回归模型：

`y = w0 + w1 x`

分别用以下三种方法求解参数：

1. 最小二乘法：直接求解正规方程。
2. 梯度下降法：对均方误差目标函数迭代优化，学习率按 Hessian 的 Lipschitz 常数自动设置。
3. 牛顿法：利用二阶导数信息更新参数。对于线性回归的二次型目标，理论上通常一步即可到达最优解。

评价指标采用均方误差（MSE）。

### 3.2 非线性模型

在线性模型之外，选择**高斯径向基函数岭回归（RBF Ridge Regression）**：

`y = w0 + Σ wi * exp(-(x-ci)^2 / (2σ^2))`

选择理由如下：

1. 数据是一维连续变量上的平滑非线性关系，RBF 基函数非常适合表达局部起伏与波动趋势。
2. RBF 模型本质上仍属于“基函数展开 + 线性参数”的框架，和课程内容高度一致。
3. 与高阶多项式相比，RBF 基函数对局部结构的描述更灵活，不容易在区间边界处出现剧烈振荡。
4. 加入 L2 正则化后，可以在拟合能力和泛化能力之间取得更稳健的平衡。

模型超参数通过 5 折交叉验证在中心个数、基函数宽度 `σ` 与正则化系数 `lambda` 上自动选择。

## 4. 实验结果

### 4.1 三种线性拟合方法的误差

{linear_rows.to_markdown(index=False)}

可以看到，三种方法得到的训练误差和测试误差几乎一致。这说明：

1. 它们虽然优化路径不同，但最终都收敛到了同一个线性回归最优解。
2. 问题不在于求解算法，而在于**线性模型表达能力不足**。

其中：

- 梯度下降法迭代次数约为：`{gd_iters}`
- 牛顿法迭代次数约为：`{newton_iters}`

从优化速度来看，牛顿法利用二阶信息，收敛明显更快；梯度下降法需要更多迭代，但实现简单、适合更大规模问题。

### 4.2 更合适的非线性模型结果

- 最优 RBF 中心数：`{int(best_rbf["center_count"])}`
- 最优宽度参数 `σ`：`{best_rbf["sigma"]}`
- 最优正则化系数：`{best_rbf["lambda"]}`
- 交叉验证 MSE：`{best_rbf["cv_mse"]:.6f}`
- 训练集 MSE：`{nonlinear_row["train_mse"]:.6f}`
- 测试集 MSE：`{nonlinear_row["test_mse"]:.6f}`

与线性模型相比，RBF 岭回归显著降低了训练误差与测试误差，说明它能够更准确地捕捉数据中的非线性结构，并且在测试集上也保持了更好的泛化能力。

## 5. 结果分析

1. **线性模型效果一般是正常现象。**
   因为数据本身具有明显非线性变化趋势，直线只能拟合总体平均方向，无法描述局部起伏，所以训练误差和测试误差都偏大。

2. **三种线性求解方法本质上是在解同一个优化问题。**
   因此只要都正确收敛，最终误差应当接近一致。实验结果验证了这一点。

3. **RBF 基函数扩展显著提升了模型表达能力。**
   在线性参数模型中引入局部高斯基函数后，模型可以表示更复杂、更平滑的弯曲关系，因此拟合效果明显改善。

4. **加入岭正则化是必要的。**
   即使使用更灵活的 RBF 基函数，若参数不加约束也可能对噪声过拟合。L2 正则化能抑制参数过大，使测试误差更加稳定。

5. **训练误差与测试误差应结合观察。**
   如果只追求训练误差最小，可能会得到过拟合模型；本实验通过交叉验证选择超参数，使模型在测试集上表现也更优。

## 6. 实验结论

1. 对于该数据集，最小二乘法、梯度下降法和牛顿法在线性拟合上的最终结果基本相同，差异主要体现在优化速度而非拟合精度。
2. 线性模型无法充分表达数据中的非线性关系，因此整体拟合效果不理想。
3. 采用 RBF 岭回归后，训练误差和测试误差都显著下降，说明该模型更适合本题数据。
4. 该实验说明：当数据存在明显非线性结构时，应通过基函数扩展或更强表达能力的模型来提升拟合效果，而不能只更换线性模型的优化算法。

## 7. 生成文件

- `results/metrics.csv`：各模型训练/测试误差
- `results/gd_history.csv`：GD 迭代过程
- `results/newton_history.csv`：牛顿法迭代过程
- `results/model_selection.csv`：RBF 模型交叉验证结果
- `results/fit_comparison.png`：拟合曲线对比图
- `results/optimization_curves.png`：GD 与牛顿法收敛曲线
- `results/rbf_selection.png`：不同 RBF 中心数的误差变化图
"""
    (ROOT / "report.md").write_text(report, encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    dataset = load_dataset()

    ls_weights = least_squares_fit(dataset.x_train, dataset.y_train)
    gd_weights, gd_history = gradient_descent_fit(dataset.x_train, dataset.y_train)
    newton_weights, newton_history = newton_fit(dataset.x_train, dataset.y_train)

    best_rbf, model_selection_df, rbf_weights = select_rbf_model(
        dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test
    )

    metric_rows = []
    for model_name, weights in [
        ("Least Squares", ls_weights),
        ("Gradient Descent", gd_weights),
        ("Newton Method", newton_weights),
    ]:
        train_pred = predict_linear(dataset.x_train, weights)
        test_pred = predict_linear(dataset.x_test, weights)
        metric_rows.append(
            {
                "model": model_name,
                "train_mse": mse(dataset.y_train, train_pred),
                "test_mse": mse(dataset.y_test, test_pred),
                "param_0": float(weights[0]),
                "param_1": float(weights[1]),
            }
        )

    best_center_count = int(best_rbf["center_count"])
    best_sigma = float(best_rbf["sigma"])
    rbf_centers = np.linspace(dataset.x_train.min(), dataset.x_train.max(), best_center_count)
    rbf_train_pred = rbf_design_matrix(dataset.x_train, rbf_centers, best_sigma) @ rbf_weights
    rbf_test_pred = rbf_design_matrix(dataset.x_test, rbf_centers, best_sigma) @ rbf_weights
    metric_rows.append(
        {
            "model": "RBF Ridge",
            "train_mse": mse(dataset.y_train, rbf_train_pred),
            "test_mse": mse(dataset.y_test, rbf_test_pred),
            "param_0": float(rbf_weights[0]),
            "param_1": float(rbf_weights[1]),
        }
    )

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(RESULTS_DIR / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(gd_history).to_csv(RESULTS_DIR / "gd_history.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(newton_history).to_csv(RESULTS_DIR / "newton_history.csv", index=False, encoding="utf-8-sig")
    model_selection_df.to_csv(RESULTS_DIR / "model_selection.csv", index=False, encoding="utf-8-sig")

    linear_weights = {
        "Least Squares": ls_weights,
        "Gradient Descent": gd_weights,
        "Newton Method": newton_weights,
    }
    save_curve_plot(dataset, linear_weights, rbf_centers, best_sigma, rbf_weights)
    save_optimization_plot(gd_history, newton_history)
    save_model_selection_plot(model_selection_df)
    write_report(dataset, metrics_df, best_rbf, gd_history, newton_history)

    summary = {
        "data_source": str(dataset.source_path),
        "linear_models": metrics_df[metrics_df["model"] != "RBF Ridge"].to_dict(orient="records"),
        "best_rbf_model": {
            "center_count": best_center_count,
            "sigma": best_sigma,
            "lambda": best_rbf["lambda"],
            "train_mse": mse(dataset.y_train, rbf_train_pred),
            "test_mse": mse(dataset.y_test, rbf_test_pred),
        },
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(metrics_df.to_string(index=False))
    print()
    print("Best RBF model:", summary["best_rbf_model"])


if __name__ == "__main__":
    main()
