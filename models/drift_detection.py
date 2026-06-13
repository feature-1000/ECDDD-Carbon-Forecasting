"""ECDDD drift detector and baseline detectors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

import numpy as np

from models.ceemdan import ceemdan_decompose


DetectorName = Literal["ecddd", "fedd", "elm"]


@dataclass
class DetectionResult:
    drift_points: list[int]
    scores: np.ndarray
    statistic_series: np.ndarray
    threshold: float
    detector: str
    metadata: dict[str, object] = field(default_factory=dict)


def approximate_entropy(series: Iterable[float], m: int = 2, r: float | None = None) -> float:
    """Compute approximate entropy (ApEn) for a 1D sequence."""

    x = np.asarray(series, dtype=float).reshape(-1)
    n = len(x)
    if n <= m + 1:
        return 0.0
    if r is None:
        r = 0.2 * float(np.std(x))
    r = max(float(r), 1e-12)

    def phi(dim: int) -> float:
        # Count template matches in reconstructed phase space. The logarithmic
        # average is the core ApEn statistic used by ECDDD.
        vectors = np.array([x[i : i + dim] for i in range(n - dim + 1)])
        distances = np.max(np.abs(vectors[:, None, :] - vectors[None, :, :]), axis=2)
        counts = np.mean(distances <= r, axis=1)
        return float(np.mean(np.log(np.maximum(counts, 1e-12))))

    return phi(m) - phi(m + 1)


def _glr_mean_variance(values: np.ndarray, min_segment: int = 5) -> tuple[float, int]:
    """Gaussian GLR statistic for an unknown mean/variance shift."""

    x = np.asarray(values, dtype=float).reshape(-1)
    n = len(x)
    if n < min_segment * 2:
        return 0.0, 0

    var_all = float(np.var(x) + 1e-12)
    best_score = 0.0
    best_k = min_segment
    for k in range(min_segment, n - min_segment + 1):
        # The GLR statistic compares one Gaussian segment against a two-segment
        # alternative, allowing both mean and variance to shift.
        left = x[:k]
        right = x[k:]
        var_left = float(np.var(left) + 1e-12)
        var_right = float(np.var(right) + 1e-12)
        score = n * np.log(var_all) - k * np.log(var_left) - (n - k) * np.log(var_right)
        if score > best_score:
            best_score = float(score)
            best_k = k
    bartlett = max(1.0, 1.0 + 11.0 / (6.0 * max(n, 1)))
    return best_score / bartlett, best_k


def _chi2_threshold(alpha: float = 0.01, df: int = 2) -> float:
    try:
        from scipy.stats import chi2

        return float(chi2.ppf(1.0 - alpha, df=df))
    except Exception:
        # df=2 chi-square quantile approximation: P(X > x)=exp(-x/2).
        return float(-2.0 * np.log(max(alpha, 1e-12)))


@dataclass
class ECDDDDetector:
    """Entropy-CEEMDAN Data Drift Detection from the manuscript."""

    window_size: int = 30
    apen_m: int = 2
    apen_r_factor: float = 0.2
    alpha: float = 0.01
    threshold: float | None = None
    min_history: int = 12
    min_segment: int = 5
    refractory: int = 30
    ceemdan_trials: int = 100
    ceemdan_noise_width: float = 0.2
    ceemdan_max_imfs: int | None = 4
    seed: int = 42

    def entropy_statistic(self, window: np.ndarray) -> float:
        # ECDDD first decomposes the local window, then monitors the complexity
        # of IMFs instead of raw observations.
        decomposition = ceemdan_decompose(
            window,
            trials=self.ceemdan_trials,
            noise_width=self.ceemdan_noise_width,
            max_imfs=self.ceemdan_max_imfs,
            seed=self.seed,
            allow_fallback=True,
        )
        entropies = [
            approximate_entropy(imf, m=self.apen_m, r=self.apen_r_factor * np.std(imf))
            for imf in decomposition.imfs
        ]
        if not entropies:
            return approximate_entropy(window, m=self.apen_m, r=self.apen_r_factor * np.std(window))
        return float(np.mean(entropies))

    def detect(self, series: Iterable[float]) -> DetectionResult:
        arr = np.asarray(series, dtype=float).reshape(-1)
        if len(arr) < self.window_size + self.min_history:
            return DetectionResult([], np.array([]), np.array([]), self.threshold or 0.0, "ECDDD")

        threshold = self.threshold if self.threshold is not None else _chi2_threshold(self.alpha, df=2)
        entropy_values: list[float] = []
        scores = np.zeros(len(arr), dtype=float)
        drifts: list[int] = []
        history_start_index = self.window_size - 1
        last_drift = -self.refractory

        for end in range(self.window_size, len(arr) + 1):
            point_index = end - 1
            window = arr[end - self.window_size : end]
            entropy_values.append(self.entropy_statistic(window))
            history = np.asarray(entropy_values, dtype=float)
            if len(history) < self.min_history:
                continue
            score, split = _glr_mean_variance(history, min_segment=self.min_segment)
            scores[point_index] = score
            if score > threshold and point_index - last_drift >= self.refractory:
                # After detecting drift, the SPC monitor restarts from the
                # estimated split point, following the streaming procedure.
                drift = history_start_index + split
                drifts.append(int(drift))
                last_drift = point_index
                keep_from = max(0, split)
                entropy_values = entropy_values[keep_from:]
                history_start_index = drift

        return DetectionResult(
            drift_points=drifts,
            scores=scores,
            statistic_series=np.asarray(entropy_values, dtype=float),
            threshold=float(threshold),
            detector="ECDDD",
            metadata={"window_size": self.window_size, "alpha": self.alpha},
        )


@dataclass
class FEDDDetector:
    """Lightweight entropy baseline detector.

    This baseline tracks approximate entropy on the raw sliding window and uses
    the same GLR/SPC monitor, without CEEMDAN feature extraction.
    """

    window_size: int = 30
    apen_m: int = 2
    apen_r_factor: float = 0.2
    alpha: float = 0.01
    threshold: float | None = None
    min_history: int = 12
    min_segment: int = 5
    refractory: int = 30

    def detect(self, series: Iterable[float]) -> DetectionResult:
        arr = np.asarray(series, dtype=float).reshape(-1)
        threshold = self.threshold if self.threshold is not None else _chi2_threshold(self.alpha, df=2)
        entropy_values: list[float] = []
        scores = np.zeros(len(arr), dtype=float)
        drifts: list[int] = []
        history_start_index = self.window_size - 1
        last_drift = -self.refractory

        for end in range(self.window_size, len(arr) + 1):
            point_index = end - 1
            window = arr[end - self.window_size : end]
            entropy_values.append(approximate_entropy(window, self.apen_m, self.apen_r_factor * np.std(window)))
            if len(entropy_values) < self.min_history:
                continue
            score, split = _glr_mean_variance(np.asarray(entropy_values), min_segment=self.min_segment)
            scores[point_index] = score
            if score > threshold and point_index - last_drift >= self.refractory:
                drift = history_start_index + split
                drifts.append(int(drift))
                last_drift = point_index
                entropy_values = entropy_values[max(0, split) :]
                history_start_index = drift

        return DetectionResult(drifts, scores, np.asarray(entropy_values), float(threshold), "FEDD")


@dataclass
class ELMDriftDetector:
    """Extreme Learning Machine residual baseline for unsupervised streams."""

    window_size: int = 30
    hidden_size: int = 50
    activation: str = "tanh"
    alpha: float = 0.01
    threshold_scale: float = 3.0
    seed: int = 42
    refractory: int = 30

    def _activate(self, x: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return np.maximum(x, 0.0)
        return np.tanh(x)

    def detect(self, series: Iterable[float]) -> DetectionResult:
        arr = np.asarray(series, dtype=float).reshape(-1)
        if len(arr) < self.window_size * 3:
            return DetectionResult([], np.array([]), np.array([]), 0.0, "ELM")

        rng = np.random.default_rng(self.seed)
        weights = rng.normal(0.0, 1.0, size=(self.window_size, self.hidden_size))
        bias = rng.normal(0.0, 0.5, size=(self.hidden_size,))

        # ELM uses the early stream as a reference concept, then flags large
        # one-step prediction residuals as potential distribution changes.
        train_end = max(self.window_size + 10, int(len(arr) * 0.25))
        x_train = np.stack([arr[i : i + self.window_size] for i in range(train_end - self.window_size)])
        y_train = arr[self.window_size:train_end]
        hidden = self._activate(x_train @ weights + bias)
        beta = np.linalg.pinv(hidden) @ y_train

        residuals = np.zeros(len(arr), dtype=float)
        train_pred = hidden @ beta
        baseline_resid = np.abs(y_train - train_pred)
        center = float(np.mean(baseline_resid))
        spread = float(np.std(baseline_resid) + 1e-12)
        threshold = center + self.threshold_scale * spread

        drifts: list[int] = []
        last_drift = -self.refractory
        for idx in range(train_end, len(arr)):
            x = arr[idx - self.window_size : idx].reshape(1, -1)
            pred = float(self._activate(x @ weights + bias) @ beta)
            residual = abs(arr[idx] - pred)
            residuals[idx] = residual
            if residual > threshold and idx - last_drift >= self.refractory:
                drifts.append(idx)
                last_drift = idx

        return DetectionResult(drifts, residuals, residuals, float(threshold), "ELM")


def build_detector(name: DetectorName, **kwargs) -> ECDDDDetector | FEDDDetector | ELMDriftDetector:
    key = name.lower()
    if key == "ecddd":
        return ECDDDDetector(**kwargs)
    if key == "fedd":
        return FEDDDetector(**kwargs)
    if key == "elm":
        return ELMDriftDetector(**kwargs)
    raise ValueError(f"Unknown detector: {name}")
