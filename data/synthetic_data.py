"""Synthetic AR sequence generation for drift detection experiments.

The manuscript evaluates ECDDD on three groups of autoregressive sequences.
This module keeps those coefficient/sigma settings in one place and exposes
small helpers that can also be used as fallback data for smoke tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


DriftMode = Literal["abrupt", "gradual"]


@dataclass(frozen=True)
class ARConcept:
    coeffs: tuple[float, ...]
    sigma: float


PAPER_AR_GROUPS: dict[str, tuple[ARConcept, ...]] = {
    "group1": (
        ARConcept((0.9, -0.2, 0.8, -0.5), 0.5),
        ARConcept((-0.3, 1.4, 0.4, -0.5), 1.5),
        ARConcept((1.5, -0.4, 0.3, 0.2), 2.5),
        ARConcept((-0.1, 1.4, 0.4, -0.7), 3.5),
    ),
    "group2": (
        ARConcept((1.1, -0.6, 0.8, 0.5, -0.1, 0.3), 0.5),
        ARConcept((-0.1, 1.2, 0.4, 0.3, -0.2, -0.6), 1.5),
        ARConcept((1.2, -0.4, -0.3, 0.7, -0.6, 0.4), 2.5),
        ARConcept((-0.1, 1.1, 0.5, 0.2, -0.2, -0.5), 3.5),
    ),
    "group3": (
        ARConcept((0.5, 0.5), 0.5),
        ARConcept((1.5, 0.5), 1.5),
        ARConcept((0.9, -0.2, 0.8, -0.5), 2.5),
        ARConcept((0.9, 0.8, -0.6, 0.2, -0.5, -0.2, 0.4), 3.5),
    ),
}


def _stable_step(value: float) -> float:
    """Keep synthetic AR sequences bounded while preserving drift shape."""

    return float(np.tanh(value / 8.0) * 8.0)


def generate_ar_series(
    concepts: tuple[ARConcept, ...],
    segment_length: int = 500,
    drift_mode: DriftMode = "abrupt",
    transition_length: int = 80,
    burn_in: int = 50,
    seed: int | None = None,
) -> tuple[np.ndarray, list[int]]:
    """Generate one AR time series with three drift points.

    Parameters follow the manuscript table. Sudden drift switches immediately
    between concepts; gradual drift linearly blends AR coefficients and sigma.
    """

    rng = np.random.default_rng(seed)
    max_order = max(len(c.coeffs) for c in concepts)
    output_length = segment_length * len(concepts)
    total_length = output_length + burn_in
    values = np.zeros(total_length + max_order, dtype=float)
    values[:max_order] = rng.normal(0.0, 1.0, size=max_order)

    drift_points = [segment_length * i for i in range(1, len(concepts))]

    for t in range(max_order, total_length + max_order):
        pos = t - max_order
        output_pos = max(0, pos - burn_in)
        concept_idx = min(output_pos // segment_length, len(concepts) - 1)
        concept = concepts[concept_idx]

        coeffs = np.asarray(concept.coeffs, dtype=float)
        sigma = concept.sigma
        if drift_mode == "gradual" and concept_idx > 0:
            # Gradual drift linearly blends the previous and current concepts
            # during the transition window, matching the benchmark setting.
            local_pos = output_pos - concept_idx * segment_length
            if local_pos < transition_length:
                prev = concepts[concept_idx - 1]
                alpha = local_pos / max(transition_length, 1)
                max_len = max(len(prev.coeffs), len(concept.coeffs))
                prev_coeffs = np.pad(prev.coeffs, (0, max_len - len(prev.coeffs)))
                next_coeffs = np.pad(concept.coeffs, (0, max_len - len(concept.coeffs)))
                coeffs = (1.0 - alpha) * prev_coeffs + alpha * next_coeffs
                sigma = (1.0 - alpha) * prev.sigma + alpha * concept.sigma

        history = values[t - len(coeffs) : t][::-1]
        noise = rng.normal(0.0, sigma)
        values[t] = _stable_step(float(np.dot(coeffs, history) + noise))

    series = values[max_order + burn_in :]
    return series, drift_points


def generate_paper_group(
    group: str,
    length: int = 2000,
    drift_mode: DriftMode = "abrupt",
    transition_length: int = 80,
    seed: int | None = None,
) -> tuple[np.ndarray, list[int]]:
    """Generate a synthetic series for one manuscript AR group."""

    key = group.lower()
    if key not in PAPER_AR_GROUPS:
        raise KeyError(f"Unknown group {group!r}. Expected one of {sorted(PAPER_AR_GROUPS)}.")
    segment_length = max(length // len(PAPER_AR_GROUPS[key]), 20)
    return generate_ar_series(
        PAPER_AR_GROUPS[key],
        segment_length=segment_length,
        drift_mode=drift_mode,
        transition_length=transition_length,
        seed=seed,
    )


def generate_synthetic_dataset(
    groups: tuple[str, ...] = ("group1", "group2", "group3"),
    runs_per_group: int = 30,
    length: int = 2000,
    gradual_ratio: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate repeated synthetic sequences for drift detector evaluation."""

    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    for group in groups:
        for run in range(runs_per_group):
            mode: DriftMode = "gradual" if run / max(runs_per_group, 1) >= 1 - gradual_ratio else "abrupt"
            series, drifts = generate_paper_group(
                group,
                length=length,
                drift_mode=mode,
                seed=int(rng.integers(0, 2**32 - 1)),
            )
            for idx, value in enumerate(series):
                rows.append(
                    {
                        "group": group,
                        "run": run,
                        "mode": mode,
                        "time": idx,
                        "value": float(value),
                        "is_drift": int(idx in drifts),
                        "drift_points": ",".join(map(str, drifts)),
                    }
                )
    return pd.DataFrame(rows)


def generate_carbon_like_series(length: int = 1734, seed: int = 42) -> np.ndarray:
    """Create a smooth, seasonal carbon-like series for local smoke tests."""

    rng = np.random.default_rng(seed)
    t = np.arange(length)
    # This fallback is only for smoke tests when the real CSV is unavailable.
    # It combines trend, weekly/annual seasonality, and structural changes.
    trend = 0.0015 * t
    annual = 1.2 * np.sin(2 * np.pi * t / 365.0)
    weekly = 0.25 * np.sin(2 * np.pi * t / 7.0)
    structural = np.where(t > int(length * 0.42), -0.8 + 0.001 * (t - int(length * 0.42)), 0.0)
    recovery = np.where(t > int(length * 0.65), 0.9 * (1 - np.exp(-(t - int(length * 0.65)) / 120.0)), 0.0)
    noise = rng.normal(0.0, 0.2, size=length)
    series = 12.5 + trend + annual + weekly + structural + recovery + noise
    return series.astype(float)
