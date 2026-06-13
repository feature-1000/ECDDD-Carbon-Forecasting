"""CEEMDAN decomposition and IMF frequency classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class DecompositionResult:
    imfs: np.ndarray
    residue: np.ndarray
    method: str

    @property
    def components(self) -> np.ndarray:
        if self.residue.size == 0:
            return self.imfs
        return np.vstack([self.imfs, self.residue.reshape(1, -1)])


@dataclass
class IMFClassification:
    high_indices: list[int]
    low_indices: list[int]
    residue_index: int | None
    ttest_table: list[dict[str, float | int | str]]


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    window = max(3, int(window) | 1)
    kernel = np.ones(window, dtype=float) / window
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def fallback_multiscale_decompose(series: Iterable[float], levels: int = 6) -> DecompositionResult:
    """A deterministic fallback when PyEMD/EMD-signal is unavailable.

    It is not a replacement for CEEMDAN in the paper. It exists so that the
    scripts remain runnable in minimal environments and still exercise the full
    pipeline.
    """

    arr = np.asarray(series, dtype=float).reshape(-1)
    residual = arr.copy()
    imfs: list[np.ndarray] = []
    for level in range(levels):
        # Repeated smoothing produces coarse-to-fine residuals. This keeps the
        # full pipeline testable even before CEEMDAN dependencies are installed.
        window = min(max(5, 2 ** (level + 2) + 1), max(5, len(arr) // 2 * 2 - 1))
        smooth = _moving_average(residual, window)
        imfs.append(residual - smooth)
        residual = smooth
        if np.std(imfs[-1]) < 1e-10:
            break
    return DecompositionResult(imfs=np.asarray(imfs), residue=residual, method="moving_average_fallback")


def ceemdan_decompose(
    series: Iterable[float],
    trials: int = 1000,
    noise_width: float = 0.2,
    max_imfs: int | None = None,
    max_iter: int = 1000,
    seed: int = 42,
    allow_fallback: bool = True,
) -> DecompositionResult:
    """Run CEEMDAN using PyEMD/EMD-signal, with an explicit fallback."""

    arr = np.asarray(series, dtype=float).reshape(-1)
    if len(arr) < 8:
        raise ValueError("CEEMDAN needs at least 8 observations.")

    try:
        from PyEMD import CEEMDAN

        # EMD-signal exposes CEEMDAN through PyEMD. We keep all paper-level
        # decomposition parameters configurable from config.yaml.
        ceemdan = CEEMDAN(trials=trials, epsilon=noise_width)
        ceemdan.noise_seed(seed)
        if hasattr(ceemdan, "MAX_ITERATION"):
            ceemdan.MAX_ITERATION = max_iter
        imfs = ceemdan.ceemdan(arr, max_imf=max_imfs or -1)
        residue = arr - np.sum(imfs, axis=0)
        return DecompositionResult(imfs=np.asarray(imfs, dtype=float), residue=residue, method="ceemdan")
    except Exception:
        if not allow_fallback:
            raise
        levels = max_imfs or min(6, max(1, len(arr) // 20))
        return fallback_multiscale_decompose(arr, levels=levels)


def classify_imfs_ttest(
    imfs: np.ndarray,
    residue: np.ndarray | None = None,
    alpha: float = 0.05,
) -> IMFClassification:
    """Classify IMFs using the manuscript's one-sample t-test rule.

    Once the first IMF significantly differs from zero, it and all later IMFs
    are treated as low-frequency because CEEMDAN orders IMFs from high to low
    frequency.
    """

    try:
        from scipy import stats
    except Exception as exc:
        raise ImportError("scipy is required for IMF t-test classification.") from exc

    arr = np.asarray(imfs, dtype=float)
    if arr.ndim != 2:
        raise ValueError("imfs must be a 2D array shaped [n_imfs, n_samples].")

    table: list[dict[str, float | int | str]] = []
    first_low: int | None = None
    for idx, imf in enumerate(arr):
        # CEEMDAN orders IMFs from high to low frequency. Once the first IMF
        # mean differs significantly from zero, later IMFs are treated as low
        # frequency as described in the manuscript.
        test = stats.ttest_1samp(imf, popmean=0.0, nan_policy="omit")
        p_value = float(test.pvalue) if np.isfinite(test.pvalue) else 1.0
        mean_diff = float(np.nanmean(imf))
        std = float(np.nanstd(imf, ddof=1))
        se = std / np.sqrt(max(len(imf), 1))
        ci_low = mean_diff - 1.96 * se
        ci_high = mean_diff + 1.96 * se
        is_low = p_value < alpha
        if is_low and first_low is None:
            first_low = idx
        table.append(
            {
                "component": f"IMF{idx + 1}",
                "t_statistic": float(test.statistic) if np.isfinite(test.statistic) else 0.0,
                "degrees_of_freedom": int(len(imf) - 1),
                "p_value": p_value,
                "average_discrepancy": mean_diff,
                "ci_lower": float(ci_low),
                "ci_upper": float(ci_high),
            }
        )

    if first_low is None:
        high_indices = list(range(len(arr)))
        low_indices: list[int] = []
    else:
        high_indices = list(range(first_low))
        low_indices = list(range(first_low, len(arr)))

    residue_index = len(arr) if residue is not None else None
    if residue is not None:
        table.append(
            {
                "component": "RES",
                "t_statistic": 0.0,
                "degrees_of_freedom": int(len(residue) - 1),
                "p_value": 1.0,
                "average_discrepancy": float(np.mean(residue)),
                "ci_lower": float(np.min(residue)),
                "ci_upper": float(np.max(residue)),
            }
        )

    return IMFClassification(
        high_indices=high_indices,
        low_indices=low_indices,
        residue_index=residue_index,
        ttest_table=table,
    )


def reconstruct_frequency_components(
    decomposition: DecompositionResult,
    classification: IMFClassification,
) -> dict[str, np.ndarray]:
    """Return high-frequency IMFs, aggregated LF component, and residue."""

    imfs = decomposition.imfs
    components: dict[str, np.ndarray] = {}
    for idx in classification.high_indices:
        components[f"HF_IMF{idx + 1}"] = imfs[idx]
    if classification.low_indices:
        components["LF"] = np.sum(imfs[classification.low_indices], axis=0)
    if decomposition.residue.size:
        components["RES"] = decomposition.residue
    return components
