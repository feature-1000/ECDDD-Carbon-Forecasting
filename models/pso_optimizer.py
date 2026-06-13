"""PSO hyperparameter optimization utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from models.backbones import train_forecaster
from utils.metrics import rmse


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    low: float
    high: float
    kind: str = "float"
    log_scale: bool = False

    def decode(self, value: float) -> int | float:
        clipped = float(np.clip(value, self.low, self.high))
        if self.log_scale:
            decoded = 10 ** clipped
        else:
            decoded = clipped
        if self.kind == "int":
            return int(round(decoded))
        return float(decoded)


DEFAULT_SEARCH_SPACE = (
    ParameterSpec("hidden_size", 16, 128, "int"),
    ParameterSpec("num_layers", 1, 3, "int"),
    ParameterSpec("dropout", 0.0, 0.5, "float"),
    ParameterSpec("learning_rate", -4, -2, "float", log_scale=True),
    ParameterSpec("weight_decay", -6, -2, "float", log_scale=True),
)


@dataclass
class PSOResult:
    best_params: dict[str, int | float]
    best_score: float
    history: list[float]
    method: str


def build_search_space(config_space: dict[str, Any] | None = None) -> tuple[ParameterSpec, ...]:
    """Build PSO parameter specs from config.yaml bounds."""

    if not config_space:
        return DEFAULT_SEARCH_SPACE
    specs: list[ParameterSpec] = []
    for name, bounds in config_space.items():
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            continue
        low, high = float(bounds[0]), float(bounds[1])
        kind = "int" if name in {"hidden_size", "num_layers", "epochs", "batch_size"} else "float"
        log_scale = name in {"learning_rate", "weight_decay"} and low > 0 and high > 0
        if log_scale:
            low, high = np.log10(low), np.log10(high)
        specs.append(ParameterSpec(name, low, high, kind=kind, log_scale=log_scale))
    return tuple(specs) if specs else DEFAULT_SEARCH_SPACE


def decode_position(position: np.ndarray, specs: tuple[ParameterSpec, ...]) -> dict[str, int | float]:
    return {spec.name: spec.decode(float(value)) for spec, value in zip(specs, position)}


def _evaluate_params(
    params: dict[str, int | float],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    model_name: str,
    fixed_params: dict[str, Any],
) -> float:
    merged = dict(fixed_params)
    merged.update(params)
    try:
        # Fitness is validation RMSE. The final experiment retrains with the
        # selected hyperparameters before testing.
        forecaster = train_forecaster(model_name, x_train, y_train, x_val, y_val, **merged)
        pred = forecaster.predict(x_val)
        score = rmse(y_val, pred)
        return float(score)
    except Exception:
        return float("inf")


def _fallback_pso(
    objective: Callable[[np.ndarray], float],
    bounds: tuple[np.ndarray, np.ndarray],
    particles: int = 10,
    iterations: int = 10,
    inertia_max: float = 0.9,
    inertia_min: float = 0.4,
    c1: float = 2.0,
    c2: float = 2.0,
    seed: int = 42,
) -> tuple[float, np.ndarray, list[float]]:
    """Manuscript PSO implementation with a global-best topology."""

    rng = np.random.default_rng(seed)
    lower, upper = bounds
    dimensions = len(lower)
    positions = rng.uniform(lower, upper, size=(particles, dimensions))
    velocities = rng.normal(0.0, 0.1, size=(particles, dimensions))
    personal_best = positions.copy()
    personal_scores = np.asarray([objective(pos) for pos in positions], dtype=float)
    best_idx = int(np.nanargmin(personal_scores))
    global_best = personal_best[best_idx].copy()
    global_score = float(personal_scores[best_idx])
    history = [global_score]

    for iteration in range(iterations):
        # Standard global-best PSO update with the manuscript's linearly
        # decreasing inertia schedule.
        inertia = inertia_max - ((inertia_max - inertia_min) * iteration / max(iterations - 1, 1))
        r1 = rng.random(size=(particles, dimensions))
        r2 = rng.random(size=(particles, dimensions))
        velocities = (
            inertia * velocities
            + c1 * r1 * (personal_best - positions)
            + c2 * r2 * (global_best - positions)
        )
        positions = np.clip(positions + velocities, lower, upper)
        scores = np.asarray([objective(pos) for pos in positions], dtype=float)
        improved = scores < personal_scores
        personal_scores[improved] = scores[improved]
        personal_best[improved] = positions[improved]
        best_idx = int(np.nanargmin(personal_scores))
        if personal_scores[best_idx] < global_score:
            global_score = float(personal_scores[best_idx])
            global_best = personal_best[best_idx].copy()
        history.append(global_score)
    return global_score, global_best, history


def optimize_hyperparameters(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    model_name: str = "bilstm",
    search_space: dict[str, Any] | None = None,
    particles: int = 10,
    iterations: int = 10,
    inertia_max: float = 0.9,
    inertia_min: float = 0.4,
    c1: float = 2.0,
    c2: float = 2.0,
    seed: int = 42,
    fixed_params: dict[str, Any] | None = None,
) -> PSOResult:
    """Optimize forecasting hyperparameters with PSO."""

    specs = build_search_space(search_space)
    lower = np.asarray([spec.low for spec in specs], dtype=float)
    upper = np.asarray([spec.high for spec in specs], dtype=float)
    fixed = dict(fixed_params or {})
    # Defaults are used only when the caller does not pass manuscript settings.
    fixed.setdefault("epochs", 30)
    fixed.setdefault("patience", 5)

    cache: dict[tuple[tuple[str, float | int], ...], float] = {}

    def objective_single(position: np.ndarray) -> float:
        params = decode_position(position, specs)
        key = tuple(sorted(params.items()))
        if key not in cache:
            cache[key] = _evaluate_params(params, x_train, y_train, x_val, y_val, model_name, fixed)
        return cache[key]

    best_score, best_position, history = _fallback_pso(
        objective_single,
        bounds=(lower, upper),
        particles=int(particles),
        iterations=int(iterations),
        inertia_max=float(inertia_max),
        inertia_min=float(inertia_min),
        c1=float(c1),
        c2=float(c2),
        seed=seed,
    )
    method = "manuscript_pso"

    best_params = decode_position(np.asarray(best_position, dtype=float), specs)
    return PSOResult(best_params=best_params, best_score=float(best_score), history=history, method=method)
