from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
TRAIN_SAMPLES = 1000
TEST_SAMPLES = 500
NOISE = 0.2
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


@dataclass
class ExperimentResult:
    model_name: str
    best_cv_accuracy: float
    test_accuracy: float
    test_precision: float
    test_recall: float
    test_f1: float
    best_params: dict[str, Any]
    classification_summary: str
    confusion_matrix_array: np.ndarray


def make_moons_3d(
    total_samples: int = 1000,
    noise: float = 0.1,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a balanced 3D two-moons style dataset."""
    if total_samples % 2 != 0:
        raise ValueError("total_samples must be even so the classes stay balanced.")

    rng = np.random.default_rng(random_state)
    half = total_samples // 2
    t = np.linspace(0, np.pi, half)

    x0 = np.cos(t)
    y0 = np.sin(t)
    z0 = 0.8 * np.sin(2 * t)
    class0 = np.column_stack([x0, y0, z0])

    x1 = 1 - np.cos(t)
    y1 = 0.5 - np.sin(t)
    z1 = -0.8 * np.sin(2 * t)
    class1 = np.column_stack([x1, y1, z1])

    X = np.vstack([class0, class1])
    y = np.hstack([np.zeros(half, dtype=int), np.ones(half, dtype=int)])
    X += rng.normal(scale=noise, size=X.shape)

    return X, y


def build_experiments() -> list[tuple[str, Any, dict[str, list[Any]]]]:
    return [
        (
            "Decision Tree",
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {
                "criterion": ["gini", "entropy"],
                "max_depth": [None, 3, 5, 8, 12],
                "min_samples_leaf": [1, 3, 5, 10],
            },
        ),
        (
            "AdaBoost + Decision Tree",
            AdaBoostClassifier(
                estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
                algorithm="SAMME",
                random_state=RANDOM_STATE,
            ),
            {
                "estimator__max_depth": [1, 2, 3],
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.3, 0.5, 1.0],
            },
        ),
        (
            "SVM (linear kernel)",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("svc", SVC(kernel="linear", random_state=RANDOM_STATE)),
                ]
            ),
            {"svc__C": [0.1, 1, 10, 100]},
        ),
        (
            "SVM (RBF kernel)",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("svc", SVC(kernel="rbf", random_state=RANDOM_STATE)),
                ]
            ),
            {
                "svc__C": [0.1, 1, 10, 100],
                "svc__gamma": ["scale", 0.1, 1, 5],
            },
        ),
        (
            "SVM (polynomial kernel)",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("svc", SVC(kernel="poly", random_state=RANDOM_STATE)),
                ]
            ),
            {
                "svc__C": [0.1, 1, 10],
                "svc__degree": [2, 3, 4],
                "svc__gamma": ["scale", 0.5, 1],
            },
        ),
        (
            "SVM (sigmoid kernel)",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("svc", SVC(kernel="sigmoid", random_state=RANDOM_STATE)),
                ]
            ),
            {
                "svc__C": [0.1, 1, 10],
                "svc__gamma": ["scale", 0.1, 1],
            },
        ),
    ]


def evaluate_model(
    model_name: str,
    estimator: Any,
    param_grid: dict[str, list[Any]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> ExperimentResult:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        estimator=clone(estimator),
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=1,
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    return ExperimentResult(
        model_name=model_name,
        best_cv_accuracy=grid.best_score_,
        test_accuracy=accuracy_score(y_test, y_pred),
        test_precision=precision_score(y_test, y_pred),
        test_recall=recall_score(y_test, y_pred),
        test_f1=f1_score(y_test, y_pred),
        best_params=grid.best_params_,
        classification_summary=classification_report(y_test, y_pred, digits=4),
        confusion_matrix_array=confusion_matrix(y_test, y_pred),
    )


def save_dataset_plot(X: np.ndarray, y: np.ndarray, output_path: Path, title: str) -> None:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap="viridis", s=20, alpha=0.8)
    legend = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_accuracy_plot(results_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df = results_df.sort_values("test_accuracy", ascending=False)
    ax.bar(plot_df["model_name"], plot_df["test_accuracy"], color="#4C72B0")
    ax.set_ylabel("Test Accuracy")
    ax.set_ylim(0.5, 1.05)
    ax.set_title("Model Comparison on 3D Moons Test Set")
    ax.tick_params(axis="x", rotation=25)

    for idx, value in enumerate(plot_df["test_accuracy"]):
        ax.text(idx, value + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def create_results_dataframe(results: list[ExperimentResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        rows.append(
            {
                "model_name": result.model_name,
                "best_cv_accuracy": result.best_cv_accuracy,
                "test_accuracy": result.test_accuracy,
                "test_precision": result.test_precision,
                "test_recall": result.test_recall,
                "test_f1": result.test_f1,
                "best_params": result.best_params,
            }
        )
    return pd.DataFrame(rows)


def build_analysis(results_df: pd.DataFrame) -> list[str]:
    sorted_df = results_df.sort_values("test_accuracy", ascending=False).reset_index(drop=True)
    best_row = sorted_df.iloc[0]
    worst_row = sorted_df.iloc[-1]

    lines = [
        "## Result Discussion",
        f"- The best test performer is **{best_row['model_name']}** with accuracy {best_row['test_accuracy']:.4f}.",
        f"- The weakest test performer is **{worst_row['model_name']}** with accuracy {worst_row['test_accuracy']:.4f}.",
        "- This dataset is a noisy, curved and non-linearly separable 3D moon structure, so models that can capture smooth non-linear boundaries usually have an advantage.",
        "- A single decision tree can carve highly irregular boundaries, but it is sensitive to local noise and axis-aligned splits are not a natural fit for crescent-shaped manifolds.",
        "- AdaBoost improves over one tree because multiple weak trees can focus on difficult regions, reducing bias while keeping enough flexibility to follow the moon-shaped boundary.",
        "- SVM performance depends strongly on the kernel. Linear SVM is limited to one global hyperplane, so it cannot model the moon geometry well.",
        "- Non-linear SVM kernels such as RBF and polynomial can map the data into richer feature spaces, which better matches the curved class structure in this task.",
        "- The sigmoid kernel often behaves less stably on this kind of problem, so it may underperform compared with RBF or polynomial kernels.",
    ]
    return lines


def write_report(
    report_path: Path,
    results: list[ExperimentResult],
    results_df: pd.DataFrame,
    train_plot_path: Path,
    test_plot_path: Path,
    accuracy_plot_path: Path,
) -> None:
    lines = [
        "# Pattern Recognition and Machine Learning Homework 2",
        "",
        "## Experimental Setup",
        f"- Training set: {TRAIN_SAMPLES} samples, balanced across C0 and C1.",
        f"- Test set: {TEST_SAMPLES} samples, balanced across C0 and C1, generated from the same distribution.",
        f"- Data noise standard deviation: {NOISE}.",
        "- Model selection uses 5-fold stratified cross validation on the training set.",
        "- Compared models: Decision Tree, AdaBoost + Decision Tree, and SVM with linear, RBF, polynomial, and sigmoid kernels.",
        "",
        "## Aggregate Results",
        "",
        results_df.sort_values("test_accuracy", ascending=False).to_markdown(index=False),
        "",
        "## Figures",
        f"- Training set plot: `{train_plot_path.name}`",
        f"- Test set plot: `{test_plot_path.name}`",
        f"- Accuracy comparison plot: `{accuracy_plot_path.name}`",
        "",
    ]

    for result in results:
        lines.extend(
            [
                f"## {result.model_name}",
                f"- Best cross-validation accuracy: {result.best_cv_accuracy:.4f}",
                f"- Test accuracy: {result.test_accuracy:.4f}",
                f"- Test precision: {result.test_precision:.4f}",
                f"- Test recall: {result.test_recall:.4f}",
                f"- Test F1-score: {result.test_f1:.4f}",
                f"- Best parameters: `{result.best_params}`",
                "",
                "### Classification Report",
                "```text",
                result.classification_summary.rstrip(),
                "```",
                "### Confusion Matrix",
                "```text",
                np.array2string(result.confusion_matrix_array),
                "```",
                "",
            ]
        )

    lines.extend(build_analysis(results_df))
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    X_train, y_train = make_moons_3d(
        total_samples=TRAIN_SAMPLES,
        noise=NOISE,
        random_state=RANDOM_STATE,
    )
    X_test, y_test = make_moons_3d(
        total_samples=TEST_SAMPLES,
        noise=NOISE,
        random_state=RANDOM_STATE + 1,
    )

    train_plot_path = OUTPUT_DIR / "train_dataset.png"
    test_plot_path = OUTPUT_DIR / "test_dataset.png"
    accuracy_plot_path = OUTPUT_DIR / "model_accuracy.png"
    report_path = OUTPUT_DIR / "homework_report.md"
    csv_path = OUTPUT_DIR / "model_results.csv"

    save_dataset_plot(X_train, y_train, train_plot_path, "3D Moons Training Set")
    save_dataset_plot(X_test, y_test, test_plot_path, "3D Moons Test Set")

    results = []
    for model_name, estimator, param_grid in build_experiments():
        print(f"Running experiment: {model_name}")
        result = evaluate_model(model_name, estimator, param_grid, X_train, y_train, X_test, y_test)
        results.append(result)
        print(
            f"  CV accuracy={result.best_cv_accuracy:.4f}, "
            f"test accuracy={result.test_accuracy:.4f}, "
            f"best params={result.best_params}"
        )

    results_df = create_results_dataframe(results)
    results_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    save_accuracy_plot(results_df, accuracy_plot_path)
    write_report(report_path, results, results_df, train_plot_path, test_plot_path, accuracy_plot_path)

    print(f"\nSaved report to: {report_path}")
    print(f"Saved detailed metrics to: {csv_path}")
    print(f"Saved figures to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
