"""Forecasting and drift detection metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


def _as_1d(values: Iterable[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("Metric input is empty.")
    return arr


def mae(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true_arr = _as_1d(y_true)
    y_pred_arr = _as_1d(y_pred)
    return float(np.mean(np.abs(y_true_arr - y_pred_arr)))


def mse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true_arr = _as_1d(y_true)
    y_pred_arr = _as_1d(y_pred)
    return float(np.mean((y_true_arr - y_pred_arr) ** 2))


def rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def mape(y_true: Iterable[float], y_pred: Iterable[float], eps: float = 1e-8) -> float:
    y_true_arr = _as_1d(y_true)
    y_pred_arr = _as_1d(y_pred)
    denom = np.maximum(np.abs(y_true_arr), eps)
    return float(np.mean(np.abs((y_true_arr - y_pred_arr) / denom)) * 100.0)


def smape(y_true: Iterable[float], y_pred: Iterable[float], eps: float = 1e-8) -> float:
    y_true_arr = _as_1d(y_true)
    y_pred_arr = _as_1d(y_pred)
    denom = np.maximum(np.abs(y_true_arr) + np.abs(y_pred_arr), eps)
    return float(np.mean(2.0 * np.abs(y_pred_arr - y_true_arr) / denom) * 100.0)


def r2_score(y_true: Iterable[float], y_pred: Iterable[float], eps: float = 1e-12) -> float:
    y_true_arr = _as_1d(y_true)
    y_pred_arr = _as_1d(y_pred)
    ss_res = np.sum((y_true_arr - y_pred_arr) ** 2)
    ss_tot = np.sum((y_true_arr - np.mean(y_true_arr)) ** 2)
    return float(1.0 - ss_res / max(ss_tot, eps))


def forecasting_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    """Return the metrics used by the manuscript and common extras."""

    return {
        "MAE": mae(y_true, y_pred),
        "MSE": mse(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "SMAPE": smape(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


@dataclass(frozen=True)
class DriftMetrics:
    detection_delay: float
    detection_position_offset: float
    false_alarms: int
    missed_detections: int
    matched_detections: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "Detection Delay": self.detection_delay,
            "Detection Position Offset": self.detection_position_offset,
            "False Alarms": self.false_alarms,
            "Miss Detection Numbers": self.missed_detections,
            "Matched Detections": self.matched_detections,
        }


def drift_detection_metrics(
    true_drifts: Iterable[int],
    detected_drifts: Iterable[int],
    tolerance: int = 100,
) -> DriftMetrics:
    """Match detected drift points to true drift points and compute paper metrics."""

    truths = sorted(int(x) for x in true_drifts)
    detections = sorted(int(x) for x in detected_drifts)
    used: set[int] = set()
    delays: list[float] = []
    offsets: list[float] = []

    for truth in truths:
        # Match each true drift to the nearest unused detection within a
        # tolerance window, then count unmatched detections as false alarms.
        candidates = [
            (idx, det)
            for idx, det in enumerate(detections)
            if idx not in used and abs(det - truth) <= tolerance
        ]
        if not candidates:
            continue
        idx, det = min(candidates, key=lambda item: (abs(item[1] - truth), item[1] < truth))
        used.add(idx)
        offsets.append(abs(det - truth))
        delays.append(max(0, det - truth))

    false_alarms = len(detections) - len(used)
    missed = len(truths) - len(offsets)

    return DriftMetrics(
        detection_delay=float(np.mean(delays)) if delays else float("nan"),
        detection_position_offset=float(np.mean(offsets)) if offsets else float("nan"),
        false_alarms=false_alarms,
        missed_detections=missed,
        matched_detections=len(offsets),
    )


def binary_detection_metrics(true_labels: Iterable[int], pred_labels: Iterable[int]) -> dict[str, float]:
    """Return accuracy, precision, recall, and F1 for binary drift labels."""

    y_true = np.asarray(list(true_labels), dtype=int).reshape(-1)
    y_pred = np.asarray(list(pred_labels), dtype=int).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError("true_labels and pred_labels must have the same shape.")

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = (tp + tn) / max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "Accuracy": float(accuracy),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
    }


def pairwise_wilcoxon_tests(
    y_true: Iterable[float],
    predictions: dict[str, Iterable[float]],
    reference_model: str | None = None,
) -> pd.DataFrame:
    """Pair-wise Wilcoxon signed-rank tests on absolute forecasting errors."""

    try:
        from scipy.stats import wilcoxon
    except Exception as exc:
        raise ImportError("scipy is required for Wilcoxon statistical tests.") from exc

    y = _as_1d(y_true)
    names = list(predictions)
    rows: list[dict[str, float | str]] = []
    pairs: list[tuple[str, str]] = []
    if reference_model and reference_model in predictions:
        pairs = [(name, reference_model) for name in names if name != reference_model]
    else:
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                pairs.append((left, right))

    for left, right in pairs:
        # Use paired absolute errors so both models are compared on exactly the
        # same test timestamps.
        err_left = np.abs(y - _as_1d(predictions[left]))
        err_right = np.abs(y - _as_1d(predictions[right]))
        n = min(len(err_left), len(err_right))
        if np.allclose(err_left[:n], err_right[:n]):
            stat, p_value = 0.0, 1.0
        else:
            stat, p_value = wilcoxon(err_left[:n], err_right[:n], zero_method="wilcox", alternative="two-sided")
        rows.append(
            {
                "Model A": left,
                "Model B": right,
                "Wilcoxon Statistic": float(stat),
                "p-value": float(p_value),
                "Significant (p<0.05)": bool(p_value < 0.05),
            }
        )
    return pd.DataFrame(rows)


def friedman_test(y_true: Iterable[float], predictions: dict[str, Iterable[float]]) -> pd.DataFrame:
    """Run a Friedman test across all model absolute-error series."""

    try:
        from scipy.stats import friedmanchisquare
    except Exception as exc:
        raise ImportError("scipy is required for Friedman statistical tests.") from exc

    y = _as_1d(y_true)
    names = list(predictions)
    errors = [np.abs(y - _as_1d(predictions[name])) for name in names]
    n = min(len(err) for err in errors)
    stat, p_value = friedmanchisquare(*[err[:n] for err in errors])
    return pd.DataFrame(
        [
            {
                "Models": ", ".join(names),
                "Friedman Statistic": float(stat),
                "p-value": float(p_value),
                "Significant (p<0.05)": bool(p_value < 0.05),
            }
        ]
    )
