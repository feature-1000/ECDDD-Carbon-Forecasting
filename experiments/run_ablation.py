"""Run ablation and sensitivity experiments controlled by config.yaml."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.data_loader import load_series_from_config
from models.ensemble import ForecastResult, fit_ceemdan_ensemble, fit_direct_forecaster, fit_ecddd_segmented_forecaster
from utils.common import load_config, resolve_device, resolve_output_dir, set_seed
from utils.plot import plot_metric_bars, plot_sensitivity


def quick_config(config: dict) -> dict:
    cfg = copy.deepcopy(config)
    cfg["model"].update({"epochs": 3, "patience": 2, "hidden_size": 16, "num_layers": 1})
    cfg["pso"].update({"particles": 2, "iterations": 2})
    cfg["ceemdan"].update({"trials": 5, "max_imfs": 3})
    cfg["sensitivity"]["ecddd_window_sizes"] = cfg["sensitivity"].get("ecddd_window_sizes", [10, 20])[:2]
    cfg["sensitivity"]["num_layers"] = cfg["sensitivity"].get("num_layers", [1, 2])[:2]
    return cfg


def manuscript_datasets(config: dict) -> list[tuple[str, np.ndarray]]:
    """Return the unsegmented series and the configured ECDDD segments."""

    series_data = load_series_from_config(config)
    segments = config.get("data", {}).get("experiment_segments") or []
    if series_data.dates is None or not segments:
        return [("Unsegmented", series_data.values)]
    datasets: list[tuple[str, np.ndarray]] = []
    for segment in segments:
        start = pd.to_datetime(segment.get("start"))
        end = pd.to_datetime(segment.get("end"))
        mask = (series_data.dates >= start) & (series_data.dates <= end)
        values = series_data.values[mask.to_numpy()]
        if len(values) >= 80:
            datasets.append((str(segment.get("name", f"{start.date()}-{end.date()}")), values))
    return datasets or [("Unsegmented", series_data.values)]


def fit_variant(series: np.ndarray, config: dict, variant_name: str, variant_cfg: dict, device: str) -> ForecastResult:
    """Fit one ablation variant from config.yaml."""

    use_ceemdan = bool(variant_cfg.get("use_ceemdan", False))
    use_pso = bool(variant_cfg.get("use_pso", False))
    backbone = str(variant_cfg.get("backbone", "bilstm"))
    if use_ceemdan:
        result = fit_ceemdan_ensemble(series, config, backbone=backbone, use_pso=use_pso, device=device)
    else:
        result = fit_direct_forecaster(series, config, model_name=backbone, use_pso=use_pso, device=device)
    result.name = variant_name
    return result


def fit_drift_segmented_variant(
    series: np.ndarray,
    config: dict,
    variant_name: str,
    variant_cfg: dict,
    device: str,
) -> ForecastResult:
    return fit_ecddd_segmented_forecaster(
        series,
        config,
        backbone=str(variant_cfg.get("backbone", "bilstm")),
        use_ceemdan=bool(variant_cfg.get("use_ceemdan", False)),
        use_pso=bool(variant_cfg.get("use_pso", False)),
        result_name=variant_name,
        device=device,
    )


def run_ablation(config: dict, device: str) -> pd.DataFrame:
    rows = []
    for dataset_name, values in manuscript_datasets(config):
        for name, variant in config.get("ablation", {}).get("variants", {}).items():
            print(f"Running ablation variant {name} on {dataset_name}...")
            try:
                if variant.get("use_drift_segments", False):
                    result = fit_drift_segmented_variant(values, config, name, variant, device)
                else:
                    result = fit_variant(values, config, name, variant, device)
                row = {
                    "Dataset": dataset_name,
                    "Variant": name,
                    **result.metrics,
                    "Segments": result.metadata.get("segment_count", result.metadata.get("segments", 1)),
                    "Protocol": result.metadata.get("forecasting_protocol", "direct"),
                }
                rows.append(row)
            except Exception as exc:
                rows.append({"Dataset": dataset_name, "Variant": name, "Error": str(exc)})
                print(f"[WARN] {dataset_name}/{name} failed: {exc}")
    return pd.DataFrame(rows)


def run_sensitivity(config: dict, device: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run manuscript sensitivity studies for ECDDD window and BiLSTM depth."""

    window_rows = []
    for dataset_name, values in manuscript_datasets(config):
        for window_size in config.get("sensitivity", {}).get("ecddd_window_sizes", [30]):
            cfg = copy.deepcopy(config)
            cfg["drift_detection"]["ecddd"]["window_size"] = int(window_size)
            print(f"Running sensitivity ecddd_window_size={window_size} on {dataset_name}...")
            try:
                result = fit_ecddd_segmented_forecaster(
                    values,
                    cfg,
                    backbone="bilstm",
                    use_ceemdan=True,
                    use_pso=True,
                    device=device,
                )
                window_rows.append({"Dataset": dataset_name, "ecddd_window_size": window_size, **result.metrics})
            except Exception as exc:
                window_rows.append({"Dataset": dataset_name, "ecddd_window_size": window_size, "Error": str(exc)})

    layer_rows = []
    for dataset_name, values in manuscript_datasets(config):
        for num_layers in config.get("sensitivity", {}).get("num_layers", [1]):
            cfg = copy.deepcopy(config)
            cfg["model"]["num_layers"] = int(num_layers)
            print(f"Running sensitivity num_layers={num_layers} on {dataset_name}...")
            try:
                result = fit_ecddd_segmented_forecaster(
                    values,
                    cfg,
                    backbone="bilstm",
                    use_ceemdan=True,
                    use_pso=True,
                    device=device,
                )
                layer_rows.append({"Dataset": dataset_name, "num_layers": num_layers, **result.metrics})
            except Exception as exc:
                layer_rows.append({"Dataset": dataset_name, "num_layers": num_layers, "Error": str(exc)})
    return pd.DataFrame(window_rows), pd.DataFrame(layer_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.quick:
        config = quick_config(config)
    set_seed(int(config.get("project", {}).get("seed", 42)))
    device = resolve_device(config.get("project", {}).get("device", "auto"))
    result_dir = resolve_output_dir(config, "results")
    figure_dir = resolve_output_dir(config, "figures")

    ablation_df = run_ablation(config, device)
    ablation_df.to_csv(result_dir / "ablation_results.csv", index=False)
    if {"Variant", "RMSE", "MAPE", "R2"}.issubset(ablation_df.columns):
        plot_metric_bars(ablation_df.rename(columns={"Variant": "Model"}), figure_dir / "ablation_metrics.png")

    window_df, layer_df = run_sensitivity(config, device)
    window_df.to_csv(result_dir / "sensitivity_ecddd_window_size.csv", index=False)
    layer_df.to_csv(result_dir / "sensitivity_num_layers.csv", index=False)
    if {"ecddd_window_size", "RMSE"}.issubset(window_df.columns):
        plot_sensitivity(window_df, figure_dir / "sensitivity_ecddd_window_size.png", x_col="ecddd_window_size")
    if {"num_layers", "RMSE"}.issubset(layer_df.columns):
        plot_sensitivity(layer_df, figure_dir / "sensitivity_num_layers.png", x_col="num_layers")

    print(f"Saved ablation and sensitivity results to {result_dir}")
    print(ablation_df.to_string(index=False))


if __name__ == "__main__":
    main()
