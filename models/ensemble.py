"""CEEMDAN-PSO-BiLSTM ensemble model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from data.data_loader import (
    MinMaxScaler1D,
    create_supervised_sequences,
    resolve_chronological_split_indices,
    train_test_split_sequences,
)
from models.backbones import FittedForecaster, train_forecaster
from models.ceemdan import ceemdan_decompose, classify_imfs_ttest, reconstruct_frequency_components
from models.drift_detection import ECDDDDetector
from models.pso_optimizer import PSOResult, optimize_hyperparameters
from utils.metrics import forecasting_metrics


@dataclass
class ForecastResult:
    name: str
    y_true: np.ndarray
    y_pred: np.ndarray
    metrics: dict[str, float]
    fitted_models: dict[str, FittedForecaster] = field(default_factory=dict)
    pso_results: dict[str, PSOResult] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TemporalSplitLayout:
    """Raw-index layout shared by direct, decomposed, and segmented models."""

    total_samples: int
    train_val_samples: int
    train_samples: int
    val_samples: int
    train_val_observation_end: int


def _base_model_params(config: dict[str, Any], device: str | None = None, model_name: str | None = None) -> dict[str, Any]:
    model_cfg = dict(config.get("model", {}))
    params = {
        "hidden_size": int(model_cfg.get("hidden_size", 64)),
        "num_layers": int(model_cfg.get("num_layers", 1)),
        "dropout": float(model_cfg.get("dropout", 0.0)),
        "n_heads": model_cfg.get("n_heads"),
        "moving_average_window": model_cfg.get("moving_average_window"),
        "learning_rate": float(model_cfg.get("learning_rate", 1e-3)),
        "weight_decay": float(model_cfg.get("weight_decay", 0.0)),
        "scheduler": model_cfg.get("scheduler"),
        "epochs": int(model_cfg.get("epochs", 100)),
        "batch_size": int(model_cfg.get("batch_size", 32)),
        "patience": int(model_cfg.get("patience", 10)),
    }
    if model_name:
        overrides = dict(config.get("model_overrides", {}).get(model_name.lower(), {}))
        params.update(overrides)
        for int_key in ("hidden_size", "num_layers", "batch_size", "epochs", "patience", "n_heads", "moving_average_window"):
            if params.get(int_key) is not None:
                params[int_key] = int(params[int_key])
        for float_key in ("dropout", "learning_rate", "weight_decay"):
            if params.get(float_key) is not None:
                params[float_key] = float(params[float_key])
    if device:
        params["device"] = device
    params["seed"] = int(config.get("project", {}).get("seed", 42))
    return params


def _split_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    data_cfg = config.get("data", {})
    split = {
        "window_size": int(data_cfg.get("window_size", 5)),
        "horizon": int(data_cfg.get("horizon", 1)),
        "train_ratio": float(data_cfg.get("train_ratio", 0.7)),
        "validation_ratio": float(data_cfg.get("validation_ratio", 0.1)),
        "normalize": bool(data_cfg.get("normalize", True)),
    }
    return split


def _combine_train_validation(split: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    x_train = np.concatenate([split["x_train"], split["x_val"]], axis=0)
    y_train = np.concatenate([split["y_train"], split["y_val"]], axis=0)
    return x_train, y_train


def _split_layout(series_length: int, split_cfg: dict[str, Any]) -> TemporalSplitLayout:
    """Mirror data_loader's chronological split while exposing raw cut points."""

    window_size = int(split_cfg.get("window_size", 5))
    horizon = int(split_cfg.get("horizon", 1))
    total_samples = series_length - window_size - horizon + 1
    if total_samples <= 0:
        raise ValueError(
            f"Series length {series_length} is too short for window_size={window_size}, horizon={horizon}."
        )
    raw_train_end, raw_val_end = resolve_chronological_split_indices(
        series_length,
        train_ratio=float(split_cfg.get("train_ratio", 0.7)),
        validation_ratio=split_cfg.get("validation_ratio", 0.1),
    )
    target_indices = np.arange(window_size + horizon - 1, series_length, dtype=int)
    train_samples = int(np.sum(target_indices < raw_train_end))
    train_val_samples = int(np.sum(target_indices < raw_val_end))
    if train_samples < 1 or train_val_samples <= train_samples or train_val_samples >= total_samples:
        raise ValueError("Chronological 7:1:2 split produced an empty train, validation, or test partition.")
    val_samples = train_val_samples - train_samples
    # Last train/validation target index plus one. CEEMDAN is only allowed to
    # see observations up to this point in the strict forecasting protocol.
    train_val_observation_end = raw_val_end
    return TemporalSplitLayout(
        total_samples=total_samples,
        train_val_samples=train_val_samples,
        train_samples=train_samples,
        val_samples=val_samples,
        train_val_observation_end=train_val_observation_end,
    )


def _train_val_component_split(
    component_values: np.ndarray,
    layout: TemporalSplitLayout,
    split_cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create component train/validation arrays without touching test labels."""

    x, y = create_supervised_sequences(
        component_values,
        window_size=int(split_cfg.get("window_size", 5)),
        horizon=int(split_cfg.get("horizon", 1)),
    )
    if len(x) < layout.train_val_samples:
        raise ValueError("Component sequence is shorter than the configured train/validation layout.")
    x_tv = x[: layout.train_val_samples]
    y_tv = y[: layout.train_val_samples]
    return (
        x_tv[: layout.train_samples],
        y_tv[: layout.train_samples],
        x_tv[layout.train_samples :],
        y_tv[layout.train_samples :],
        x_tv,
        y_tv,
    )


def _test_target_indices(layout: TemporalSplitLayout, split_cfg: dict[str, Any]) -> list[int]:
    """Return raw target indices for the chronological test samples."""

    window_size = int(split_cfg.get("window_size", 5))
    horizon = int(split_cfg.get("horizon", 1))
    return [
        int(sample_idx + window_size + horizon - 1)
        for sample_idx in range(layout.train_val_samples, layout.total_samples)
    ]


def _reconstruct_with_fixed_schema(decomposition, classification) -> dict[str, np.ndarray]:
    """Rebuild rolling CEEMDAN components using the train-period IMF schema.

    CEEMDAN may return a slightly different number of IMFs on a shorter rolling
    history. Missing expected IMFs are represented as zeros so every fitted
    component model receives a stable input channel during causal inference.
    """

    imfs = np.asarray(decomposition.imfs, dtype=float)
    n = imfs.shape[1] if imfs.ndim == 2 and imfs.size else len(decomposition.residue)
    zeros = np.zeros(n, dtype=float)
    components: dict[str, np.ndarray] = {}
    for idx in classification.high_indices:
        components[f"HF_IMF{idx + 1}"] = imfs[idx] if idx < len(imfs) else zeros.copy()
    if classification.low_indices:
        available = [imfs[idx] for idx in classification.low_indices if idx < len(imfs)]
        components["LF"] = np.sum(available, axis=0) if available else zeros.copy()
    if classification.residue_index is not None:
        components["RES"] = decomposition.residue if decomposition.residue.size else zeros.copy()
    return components


def _ceemdan_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    ce_cfg = config.get("ceemdan", {})
    return {
        "trials": int(ce_cfg.get("trials", 1000)),
        "noise_width": float(ce_cfg.get("noise_width", 0.2)),
        "max_imfs": ce_cfg.get("max_imfs"),
        "max_iter": int(ce_cfg.get("max_iter", 1000)),
        "seed": int(config.get("project", {}).get("seed", 42)),
    }


def fit_direct_forecaster(
    series: Iterable[float],
    config: dict[str, Any],
    *,
    model_name: str = "bilstm",
    use_pso: bool = False,
    device: str | None = None,
) -> ForecastResult:
    """Fit a direct rolling-window forecaster on the configured series."""

    raw = np.asarray(series, dtype=float).reshape(-1)
    split_cfg = _split_kwargs(config)
    layout = _split_layout(len(raw), split_cfg)
    split = train_test_split_sequences(raw, **split_cfg)
    params = _base_model_params(config, device=device, model_name=model_name)
    params["seed"] = int(config.get("project", {}).get("seed", 42))
    pso_result: PSOResult | None = None
    if use_pso:
        # PSO tunes on the validation slice only. The held-out test set remains
        # untouched until the final metric computation below.
        pso_cfg = config.get("pso", {})
        pso_result = optimize_hyperparameters(
            split["x_train"],
            split["y_train"],
            split["x_val"],
            split["y_val"],
            model_name=model_name,
            search_space=pso_cfg.get("search_space"),
            particles=int(pso_cfg.get("particles", 8)),
            iterations=int(pso_cfg.get("iterations", 8)),
            inertia_max=float(pso_cfg.get("inertia_max", 0.9)),
            inertia_min=float(pso_cfg.get("inertia_min", 0.4)),
            c1=float(pso_cfg.get("c1", 2.0)),
            c2=float(pso_cfg.get("c2", 2.0)),
            seed=int(config.get("project", {}).get("seed", 42)),
            fixed_params=params,
        )
        params.update(pso_result.best_params)

    x_fit, y_fit = _combine_train_validation(split)
    # After hyperparameter selection, retrain on train+validation and evaluate
    # exactly once on the chronological test set.
    forecaster = train_forecaster(model_name, x_fit, y_fit, None, None, **params)
    y_pred_scaled = forecaster.predict(split["x_test"])
    scaler: MinMaxScaler1D = split["scaler"]
    y_true = scaler.inverse_transform(split["y_test"])
    y_pred = scaler.inverse_transform(y_pred_scaled)
    name = f"PSO-{model_name.upper()}" if use_pso else model_name.upper()
    return ForecastResult(
        name=name,
        y_true=y_true,
        y_pred=y_pred,
        metrics=forecasting_metrics(y_true, y_pred),
        fitted_models={model_name: forecaster},
        pso_results={model_name: pso_result} if pso_result else {},
        metadata={
            "forecasting_protocol": "direct",
            "split": {k: len(split[k]) for k in ("x_train", "x_val", "x_test")},
            "test_target_indices": _test_target_indices(layout, split_cfg),
        },
    )


def _fit_ceemdan_ensemble_full_series(
    series: Iterable[float],
    config: dict[str, Any],
    *,
    backbone: str = "bilstm",
    use_pso: bool = False,
    device: str | None = None,
) -> ForecastResult:
    """Legacy CEEMDAN protocol that decomposes the configured dataset at once."""

    split_cfg = _split_kwargs(config)
    raw = np.asarray(series, dtype=float).reshape(-1)
    layout = _split_layout(len(raw), split_cfg)
    normalize = bool(split_cfg.pop("normalize"))
    if normalize:
        # The scaler is train-period only, but this legacy protocol still runs
        # CEEMDAN on the full scaled series. Keep it only for reproducing older
        # manuscript tables that used whole-series decomposition.
        base_split = train_test_split_sequences(raw, normalize=True, **split_cfg)
        scaler: MinMaxScaler1D = base_split["scaler"]
        scaled = base_split["scaled_series"]
    else:
        scaler = MinMaxScaler1D()
        scaled = raw.copy()

    decomposition = ceemdan_decompose(
        scaled,
        **_ceemdan_kwargs(config),
    )
    classification = classify_imfs_ttest(
        decomposition.imfs,
        residue=decomposition.residue,
        alpha=float(config.get("ceemdan", {}).get("alpha", 0.05)),
    )
    components = reconstruct_frequency_components(decomposition, classification)

    component_predictions: list[np.ndarray] = []
    fitted: dict[str, FittedForecaster] = {}
    pso_results: dict[str, PSOResult] = {}
    y_true_scaled: np.ndarray | None = None

    for component_name, component_values in components.items():
        # High-frequency IMFs are modeled separately, while low-frequency IMFs
        # are aggregated into LF by reconstruct_frequency_components().
        split = train_test_split_sequences(
            component_values,
            normalize=False,
            **split_cfg,
        )
        params = _base_model_params(config, device=device, model_name=backbone)
        if use_pso:
            pso_cfg = config.get("pso", {})
            pso_result = optimize_hyperparameters(
                split["x_train"],
                split["y_train"],
                split["x_val"],
                split["y_val"],
                model_name=backbone,
                search_space=pso_cfg.get("search_space"),
                particles=int(pso_cfg.get("particles", 8)),
                iterations=int(pso_cfg.get("iterations", 8)),
                inertia_max=float(pso_cfg.get("inertia_max", 0.9)),
                inertia_min=float(pso_cfg.get("inertia_min", 0.4)),
                c1=float(pso_cfg.get("c1", 2.0)),
                c2=float(pso_cfg.get("c2", 2.0)),
                seed=int(config.get("project", {}).get("seed", 42)),
                fixed_params=params,
            )
            params.update(pso_result.best_params)
            pso_results[component_name] = pso_result
        x_fit, y_fit = _combine_train_validation(split)
        forecaster = train_forecaster(backbone, x_fit, y_fit, None, None, **params)
        fitted[component_name] = forecaster
        component_predictions.append(forecaster.predict(split["x_test"]))
        if y_true_scaled is None:
            original_split = train_test_split_sequences(scaled, normalize=False, **split_cfg)
            y_true_scaled = original_split["y_test"]

    y_pred_scaled = np.sum(component_predictions, axis=0)
    y_true = scaler.inverse_transform(y_true_scaled if y_true_scaled is not None else np.zeros_like(y_pred_scaled))
    y_pred = scaler.inverse_transform(y_pred_scaled)
    name = f"CEEMDAN-PSO-{backbone.upper()}" if use_pso else f"CEEMDAN-{backbone.upper()}"
    return ForecastResult(
        name=name,
        y_true=y_true,
        y_pred=y_pred,
        metrics=forecasting_metrics(y_true, y_pred),
        fitted_models=fitted,
        pso_results=pso_results,
        metadata={
            "forecasting_protocol": "full_series",
            "leakage_note": "Whole-series CEEMDAN is retained only for legacy manuscript reproduction.",
            "decomposition_method": decomposition.method,
            "components": list(components),
            "high_indices": classification.high_indices,
            "low_indices": classification.low_indices,
            "ttest_table": classification.ttest_table,
            "test_target_indices": _test_target_indices(layout, split_cfg),
        },
    )


def _fit_ceemdan_ensemble_causal(
    series: Iterable[float],
    config: dict[str, Any],
    *,
    backbone: str = "bilstm",
    use_pso: bool = False,
    device: str | None = None,
) -> ForecastResult:
    """Fit CEEMDAN models using a causal expanding-window test protocol.

    Training and validation components are decomposed only from the
    train/validation observation span. At test time, each prediction origin is
    decomposed using observations available up to that origin, never the target
    value or later test observations.
    """

    split_cfg = _split_kwargs(config)
    raw = np.asarray(series, dtype=float).reshape(-1)
    layout = _split_layout(len(raw), split_cfg)
    normalize = bool(split_cfg.pop("normalize"))
    if normalize:
        base_split = train_test_split_sequences(raw, normalize=True, **split_cfg)
        scaler: MinMaxScaler1D = base_split["scaler"]
        scaled = base_split["scaled_series"]
    else:
        scaler = MinMaxScaler1D()
        scaled = raw.copy()

    train_val_series = scaled[: layout.train_val_observation_end]
    decomposition = ceemdan_decompose(train_val_series, **_ceemdan_kwargs(config))
    classification = classify_imfs_ttest(
        decomposition.imfs,
        residue=decomposition.residue,
        alpha=float(config.get("ceemdan", {}).get("alpha", 0.05)),
    )
    components = _reconstruct_with_fixed_schema(decomposition, classification)

    fitted: dict[str, FittedForecaster] = {}
    pso_results: dict[str, PSOResult] = {}
    for component_name, component_values in components.items():
        x_train, y_train, x_val, y_val, x_tv, y_tv = _train_val_component_split(
            component_values,
            layout,
            split_cfg,
        )
        params = _base_model_params(config, device=device, model_name=backbone)
        if use_pso:
            pso_cfg = config.get("pso", {})
            pso_result = optimize_hyperparameters(
                x_train,
                y_train,
                x_val,
                y_val,
                model_name=backbone,
                search_space=pso_cfg.get("search_space"),
                particles=int(pso_cfg.get("particles", 8)),
                iterations=int(pso_cfg.get("iterations", 8)),
                inertia_max=float(pso_cfg.get("inertia_max", 0.9)),
                inertia_min=float(pso_cfg.get("inertia_min", 0.4)),
                c1=float(pso_cfg.get("c1", 2.0)),
                c2=float(pso_cfg.get("c2", 2.0)),
                seed=int(config.get("project", {}).get("seed", 42)),
                fixed_params=params,
            )
            params.update(pso_result.best_params)
            pso_results[component_name] = pso_result
        fitted[component_name] = train_forecaster(backbone, x_tv, y_tv, None, None, **params)

    data_split = train_test_split_sequences(scaled, normalize=False, **split_cfg)
    y_true_scaled = np.asarray(data_split["y_test"], dtype=float)
    window_size = int(split_cfg.get("window_size", 5))
    horizon = int(split_cfg.get("horizon", 1))
    y_pred_scaled: list[float] = []
    expected_components = list(components)

    for sample_idx in range(layout.train_val_samples, layout.total_samples):
        # sample_idx defines the rolling supervised sample. Its input window
        # ends before the target at sample_idx + window_size + horizon - 1.
        observed_end = sample_idx + window_size
        history = scaled[:observed_end]
        rolling_decomposition = ceemdan_decompose(history, **_ceemdan_kwargs(config))
        rolling_components = _reconstruct_with_fixed_schema(rolling_decomposition, classification)
        prediction = 0.0
        for component_name in expected_components:
            values = rolling_components.get(component_name)
            if values is None or len(values) < window_size:
                x_component = np.zeros((1, window_size, 1), dtype=float)
            else:
                x_component = np.asarray(values[-window_size:], dtype=float).reshape(1, window_size, 1)
            prediction += float(fitted[component_name].predict(x_component)[0])
        y_pred_scaled.append(prediction)

    y_pred_scaled_arr = np.asarray(y_pred_scaled, dtype=float)
    if len(y_pred_scaled_arr) != len(y_true_scaled):
        raise RuntimeError(
            "Causal CEEMDAN produced a prediction count that does not match the chronological test split."
        )

    y_true = scaler.inverse_transform(y_true_scaled)
    y_pred = scaler.inverse_transform(y_pred_scaled_arr)
    name = f"CEEMDAN-PSO-{backbone.upper()}" if use_pso else f"CEEMDAN-{backbone.upper()}"
    return ForecastResult(
        name=name,
        y_true=y_true,
        y_pred=y_pred,
        metrics=forecasting_metrics(y_true, y_pred),
        fitted_models=fitted,
        pso_results=pso_results,
        metadata={
            "forecasting_protocol": "causal_expanding",
            "leakage_note": "Each test prediction decomposes only observations available before the forecast target.",
            "decomposition_method": decomposition.method,
            "components": expected_components,
            "high_indices": classification.high_indices,
            "low_indices": classification.low_indices,
            "ttest_table": classification.ttest_table,
            "train_val_observation_end": layout.train_val_observation_end,
            "test_predictions": len(y_pred_scaled_arr),
            "test_target_indices": _test_target_indices(layout, split_cfg),
            "horizon": horizon,
        },
    )


def fit_ceemdan_ensemble(
    series: Iterable[float],
    config: dict[str, Any],
    *,
    backbone: str = "bilstm",
    use_pso: bool = False,
    device: str | None = None,
) -> ForecastResult:
    """Fit CEEMDAN component models and sum component forecasts."""

    protocol = str(config.get("ceemdan", {}).get("forecasting_protocol", "causal_expanding")).lower()
    if protocol in {"full_series", "legacy_full_series", "paper_full_series"}:
        return _fit_ceemdan_ensemble_full_series(
            series,
            config,
            backbone=backbone,
            use_pso=use_pso,
            device=device,
        )
    if protocol not in {"causal", "causal_expanding", "strict"}:
        raise ValueError(
            "ceemdan.forecasting_protocol must be 'causal_expanding' or 'full_series'."
        )
    return _fit_ceemdan_ensemble_causal(
        series,
        config,
        backbone=backbone,
        use_pso=use_pso,
        device=device,
    )


def _ecddd_detector_from_config(config: dict[str, Any]) -> ECDDDDetector:
    ec_cfg = dict(config.get("drift_detection", {}).get("ecddd", {}))
    return ECDDDDetector(
        window_size=int(ec_cfg.get("window_size", 30)),
        alpha=float(ec_cfg.get("alpha", 0.01)),
        threshold=ec_cfg.get("threshold"),
        refractory=int(ec_cfg.get("refractory", 30)),
        ceemdan_trials=int(ec_cfg.get("ceemdan_trials", 100)),
        ceemdan_noise_width=float(ec_cfg.get("ceemdan_noise_width", 0.2)),
        ceemdan_max_imfs=ec_cfg.get("ceemdan_max_imfs", 4),
        seed=int(config.get("project", {}).get("seed", 42)),
    )


def ecddd_segments(series: Iterable[float], config: dict[str, Any]) -> list[dict[str, Any]]:
    """Detect ECDDD drift points and return usable chronological segments."""

    arr = np.asarray(series, dtype=float).reshape(-1)
    detector = _ecddd_detector_from_config(config)
    result = detector.detect(arr)
    min_length = int(config.get("drift_detection", {}).get("min_segment_length", 80))
    margin = int(config.get("drift_detection", {}).get("segment_margin", 60))
    cuts = [0] + [point for point in result.drift_points if margin < point < len(arr) - margin] + [len(arr)]
    segments = []
    for idx, (start, end) in enumerate(zip(cuts[:-1], cuts[1:]), start=1):
        if end - start >= min_length:
            segments.append(
                {
                    "name": f"segment_{idx}",
                    "start": int(start),
                    "end": int(end),
                    "values": arr[start:end],
                }
            )
    if not segments:
        segments = [{"name": "segment_1", "start": 0, "end": len(arr), "values": arr}]
    return segments


def fit_ecddd_segmented_forecaster(
    series: Iterable[float],
    config: dict[str, Any],
    *,
    backbone: str = "bilstm",
    use_ceemdan: bool = True,
    use_pso: bool = True,
    result_name: str | None = None,
    device: str | None = None,
) -> ForecastResult:
    """Fit the full ECDDD segmented forecasting framework."""

    predictions: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    target_indices: list[int] = []
    segment_results: list[tuple[dict[str, Any], ForecastResult]] = []
    for segment in ecddd_segments(series, config):
        if use_ceemdan:
            result = fit_ceemdan_ensemble(
                segment["values"],
                config,
                backbone=backbone,
                use_pso=use_pso,
                device=device,
            )
        else:
            result = fit_direct_forecaster(
                segment["values"],
                config,
                model_name=backbone,
                use_pso=use_pso,
                device=device,
            )
        predictions.append(result.y_pred)
        truths.append(result.y_true)
        target_indices.extend(
            int(segment["start"]) + int(local_idx)
            for local_idx in result.metadata.get("test_target_indices", [])
        )
        segment_results.append((segment, result))

    y_true = np.concatenate(truths)
    y_pred = np.concatenate(predictions)
    if result_name is not None:
        name = result_name
    elif use_ceemdan and use_pso and backbone.lower() == "bilstm":
        name = "ECDDD-CEEMDAN-PSO-BILSTM"
    else:
        prefix = "ECDDD-"
        middle = "CEEMDAN-" if use_ceemdan else ""
        opt = "PSO-" if use_pso else ""
        name = f"{prefix}{middle}{opt}{backbone.upper()}"

    fitted = {
        f"{segment['name']}_{component_name}": model
        for segment, result in segment_results
        for component_name, model in result.fitted_models.items()
    }
    pso_payload = {
        f"{segment['name']}_{component_name}": pso
        for segment, result in segment_results
        for component_name, pso in result.pso_results.items()
    }
    return ForecastResult(
        name=name,
        y_true=y_true,
        y_pred=y_pred,
        metrics=forecasting_metrics(y_true, y_pred),
        fitted_models=fitted,
        pso_results=pso_payload,
        metadata={
            "forecasting_protocol": "ecddd_segmented",
            "segments": [
                {"name": segment["name"], "start": segment["start"], "end": segment["end"]}
                for segment, _ in segment_results
            ],
            "segment_count": len(segment_results),
            "component_protocols": [result.metadata.get("forecasting_protocol") for _, result in segment_results],
            "test_target_indices": target_indices,
        },
    )


def fit_paper_model(
    model_key: str,
    series: Iterable[float],
    config: dict[str, Any],
    *,
    device: str | None = None,
) -> ForecastResult:
    """Dispatch the five paper models plus common extended variants."""

    key = model_key.lower()
    if key in {"ecddd_ceemdan_pso_bilstm", "ecddd-ceemdan-pso-bilstm"}:
        return fit_ecddd_segmented_forecaster(
            series,
            config,
            backbone="bilstm",
            use_ceemdan=True,
            use_pso=True,
            device=device,
        )
    if key == "pso_bilstm":
        return fit_direct_forecaster(series, config, model_name="bilstm", use_pso=True, device=device)
    if key == "ceemdan_bilstm":
        return fit_ceemdan_ensemble(series, config, backbone="bilstm", use_pso=False, device=device)
    if key == "ceemdan_pso_bilstm":
        return fit_ceemdan_ensemble(series, config, backbone="bilstm", use_pso=True, device=device)
    return fit_direct_forecaster(series, config, model_name=key, use_pso=False, device=device)
