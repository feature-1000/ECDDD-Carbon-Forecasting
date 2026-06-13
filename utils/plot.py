"""Plot generation for paper tables and figures."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from utils.common import ensure_dir


def _prepare_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_drift_detection(
    series: Iterable[float],
    drift_points: Iterable[int],
    output_path: str | Path,
    *,
    true_drifts: Iterable[int] | None = None,
    title: str = "Drift Detection",
) -> Path:
    plt = _prepare_matplotlib()
    values = np.asarray(series, dtype=float)
    out = Path(output_path)
    ensure_dir(out.parent)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(values, color="#1f77b4", linewidth=1.2, label="Series")
    for idx, point in enumerate(drift_points):
        ax.axvline(int(point), color="#d62728", linestyle="--", linewidth=1.0, label="Detected" if idx == 0 else None)
    if true_drifts is not None:
        for idx, point in enumerate(true_drifts):
            ax.axvline(int(point), color="#111111", linestyle=":", linewidth=1.0, label="True" if idx == 0 else None)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def plot_forecast(
    y_true: Iterable[float],
    predictions: dict[str, Iterable[float]],
    output_path: str | Path,
    *,
    title: str = "Forecasting Results",
) -> Path:
    plt = _prepare_matplotlib()
    out = Path(output_path)
    ensure_dir(out.parent)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(np.asarray(y_true, dtype=float), color="#111111", linewidth=1.6, label="Actual")
    for name, pred in predictions.items():
        ax.plot(np.asarray(pred, dtype=float), linewidth=1.0, alpha=0.85, label=name)
    ax.set_title(title)
    ax.set_xlabel("Test sequence")
    ax.set_ylabel("CO2")
    ax.legend(loc="best", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def plot_metric_bars(
    metrics_df: pd.DataFrame,
    output_path: str | Path,
    metrics: tuple[str, ...] = ("RMSE", "MAPE", "R2"),
    *,
    title: str = "Model Metrics",
) -> Path:
    plt = _prepare_matplotlib()
    out = Path(output_path)
    ensure_dir(out.parent)
    available = [metric for metric in metrics if metric in metrics_df.columns]
    if not available:
        return out
    plot_df = metrics_df.copy()
    if "Dataset" in plot_df.columns:
        plot_df["Model"] = plot_df["Dataset"].astype(str) + " / " + plot_df["Model"].astype(str)
    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 4))
    if len(available) == 1:
        axes = [axes]
    for ax, metric in zip(axes, available):
        sorted_df = plot_df.sort_values(metric, ascending=(metric != "R2"))
        ax.bar(sorted_df["Model"], sorted_df[metric], color="#4c78a8")
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def plot_sensitivity(
    sensitivity_df: pd.DataFrame,
    output_path: str | Path,
    *,
    x_col: str,
    y_cols: tuple[str, ...] = ("RMSE", "MAPE"),
    title: str = "Sensitivity Analysis",
) -> Path:
    plt = _prepare_matplotlib()
    out = Path(output_path)
    ensure_dir(out.parent)
    fig, ax = plt.subplots(figsize=(7, 4))
    if "Dataset" in sensitivity_df.columns:
        for dataset, group in sensitivity_df.groupby("Dataset", sort=False):
            for y_col in y_cols:
                if y_col in group.columns:
                    ax.plot(group[x_col], group[y_col], marker="o", label=f"{dataset} {y_col}")
    else:
        for y_col in y_cols:
            if y_col in sensitivity_df.columns:
                ax.plot(sensitivity_df[x_col], sensitivity_df[y_col], marker="o", label=y_col)
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out
