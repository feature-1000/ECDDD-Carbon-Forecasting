"""Run main forecasting and baseline comparison experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.data_loader import load_series_from_config, resolve_chronological_split_indices, train_test_split_sequences
from models.ensemble import ForecastResult, fit_direct_forecaster, fit_paper_model
from utils.common import load_config, resolve_device, resolve_output_dir, save_json, set_seed
from utils.metrics import friedman_test, pairwise_wilcoxon_tests
from utils.plot import plot_forecast, plot_metric_bars


def apply_quick_settings(config: dict) -> dict:
    cfg = dict(config)
    cfg["model"] = dict(cfg.get("model", {}))
    cfg["model"].update({"epochs": 3, "patience": 2, "hidden_size": 16, "num_layers": 1, "batch_size": 32})
    cfg["pso"] = dict(cfg.get("pso", {}))
    cfg["pso"].update({"particles": 2, "iterations": 2})
    cfg["ceemdan"] = dict(cfg.get("ceemdan", {}))
    cfg["ceemdan"].update({"trials": 5, "max_imfs": 3})
    cfg["model_overrides"] = dict(cfg.get("model_overrides", {}))
    cfg["model_overrides"].update(
        {
            "dlinear": {"moving_average_window": 5, "learning_rate": 0.001},
            "itransformer": {"hidden_size": 16, "n_heads": 1, "learning_rate": 0.001},
            "mlp_mixer": {"hidden_size": 16, "learning_rate": 0.001},
        }
    )
    cfg["baselines"] = ["dlinear", "itransformer", "mlp_mixer"]
    cfg["paper_models"] = ["lstm", "bilstm"]
    return cfg


def result_to_row(result: ForecastResult, label: str | None = None) -> dict[str, object]:
    row: dict[str, object] = {"Model": label or result.name}
    row.update(result.metrics)
    row["Best Epoch"] = ",".join(
        f"{name}:{model.history.best_epoch}" for name, model in result.fitted_models.items() if model.history
    )
    row["Components"] = ",".join(result.metadata.get("components", [])) if result.metadata.get("components") else ""
    row["Protocol"] = result.metadata.get("forecasting_protocol", "direct")
    row["Segments"] = result.metadata.get("segment_count", 1)
    return row


def dataset_slices(config: dict) -> list[tuple[str, object, object | None]]:
    """Build the full-series and manuscript segment datasets."""

    series_data = load_series_from_config(config)
    segments = config.get("data", {}).get("experiment_segments") or []
    datasets = []
    if series_data.dates is None or not segments:
        return [("Unsegmented", series_data.values, None)]
    for segment in segments:
        start = pd.to_datetime(segment.get("start"))
        end = pd.to_datetime(segment.get("end"))
        mask = (series_data.dates >= start) & (series_data.dates <= end)
        values = series_data.values[mask.to_numpy()]
        dates = series_data.dates[mask].reset_index(drop=True)
        if len(values) >= 80:
            datasets.append((str(segment.get("name", f"{start.date()}-{end.date()}")), values, dates))
    return datasets or [("Unsegmented", series_data.values, series_data.dates)]


def split_summary(dataset_name: str, values, dates, config: dict) -> dict[str, object]:
    """Record both raw-date and supervised-window partition details."""

    data_cfg = config.get("data", {})
    train_ratio = float(data_cfg.get("train_ratio", 0.7))
    validation_ratio = float(data_cfg.get("validation_ratio", 0.1))
    n = len(values)
    train_end, validation_end = resolve_chronological_split_indices(
        n,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
    )
    row = {
        "Dataset": dataset_name,
        "Total Count": n,
        "Train Count": train_end,
        "Validation Count": validation_end - train_end,
        "Test Count": n - validation_end,
        "Train Ratio": train_ratio,
        "Validation Ratio": validation_ratio,
        "Test Ratio": float(data_cfg.get("test_ratio", 1.0 - train_ratio - validation_ratio)),
        "Split Protocol": "chronological_7_1_2",
    }
    supervised = train_test_split_sequences(
        values,
        window_size=int(data_cfg.get("window_size", 5)),
        horizon=int(data_cfg.get("horizon", 1)),
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        normalize=bool(data_cfg.get("normalize", True)),
    )
    row.update(
        {
            "Window Size": int(data_cfg.get("window_size", 5)),
            "Horizon": int(data_cfg.get("horizon", 1)),
            "Supervised Train Windows": len(supervised["x_train"]),
            "Supervised Validation Windows": len(supervised["x_val"]),
            "Supervised Test Windows": len(supervised["x_test"]),
        }
    )
    if dates is not None and len(dates) == n:
        row.update(
            {
                "Full Start": str(dates.iloc[0].date()),
                "Full End": str(dates.iloc[-1].date()),
                "Train Start": str(dates.iloc[0].date()),
                "Train End": str(dates.iloc[train_end - 1].date()),
                "Validation Start": str(dates.iloc[train_end].date()),
                "Validation End": str(dates.iloc[validation_end - 1].date()),
                "Test Start": str(dates.iloc[validation_end].date()),
                "Test End": str(dates.iloc[-1].date()),
            }
        )
    return row


def run_models(config: dict, model_keys: list[str], device: str, series_values) -> list[ForecastResult]:
    results: list[ForecastResult] = []
    seen: set[str] = set()
    for key in model_keys:
        key_lower = key.lower()
        if key_lower in seen:
            continue
        seen.add(key_lower)
        print(f"Running {key_lower}...")
        try:
            if key_lower in {
                "pso_bilstm",
                "ceemdan_bilstm",
                "ceemdan_pso_bilstm",
                "ecddd_ceemdan_pso_bilstm",
            }:
                result = fit_paper_model(key_lower, series_values, config, device=device)
            else:
                result = fit_direct_forecaster(series_values, config, model_name=key_lower, use_pso=False, device=device)
            results.append(result)
        except Exception as exc:
            print(f"[WARN] {key_lower} failed: {exc}")
    return results


def apply_method_references(metrics_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Map internal model keys to table-ready method names."""

    refs = config.get("baseline_refs", {})
    renamed = metrics_df.copy()

    def with_ref(name: str) -> str:
        key = name.lower().replace("-", "_")
        return str(refs.get(key, refs.get(name.lower(), name)))

    renamed["Model"] = renamed["Model"].map(with_ref)
    return renamed


def write_manuscript_markdown(metrics_df: pd.DataFrame, path: Path, config: dict) -> None:
    """Create a Markdown table with arrows and bold best values."""

    table = apply_method_references(metrics_df, config)
    lines = ["| Dataset | Model | RMSE ↓ | MAPE ↓ | R² ↑ |", "|---|---|---:|---:|---:|"]
    for dataset, group in table.groupby("Dataset", sort=False):
        best_rmse = group["RMSE"].min()
        best_mape = group["MAPE"].min()
        best_r2 = group["R2"].max()
        for _, row in group.iterrows():
            rmse = f"{row['RMSE']:.6f}"
            mape = f"{row['MAPE']:.6f}"
            r2 = f"{row['R2']:.6f}"
            if row["RMSE"] == best_rmse:
                rmse = f"**{rmse}**"
            if row["MAPE"] == best_mape:
                mape = f"**{mape}**"
            if row["R2"] == best_r2:
                r2 = f"**{r2}**"
            lines.append(f"| {dataset} | {row['Model']} | {rmse} | {mape} | {r2} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_outputs(config: dict, grouped_results: dict[str, list[ForecastResult]], partition_rows: list[dict[str, object]]) -> None:
    result_dir = resolve_output_dir(config, "results")
    figure_dir = resolve_output_dir(config, "figures")

    # Store machine-readable metrics and a table-ready version for manuscripts.
    metric_rows = []
    for dataset_name, results in grouped_results.items():
        for result in results:
            row = {"Dataset": dataset_name}
            row.update(result_to_row(result))
            metric_rows.append(row)
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(result_dir / "forecast_metrics.csv", index=False)
    manuscript = apply_method_references(metrics_df, config).rename(columns={"RMSE": "RMSE ↓", "MAPE": "MAPE ↓", "R2": "R² ↑"})
    manuscript.to_csv(result_dir / "forecast_metrics_manuscript_format.csv", index=False)
    write_manuscript_markdown(metrics_df, result_dir / "forecast_metrics_manuscript_table.md", config)
    pd.DataFrame(partition_rows).to_csv(result_dir / "dataset_partition_summary.csv", index=False)

    # Keep all test-set predictions so statistical tests and plots can be
    # reproduced without retraining.
    prediction_rows = []
    for dataset_name, results in grouped_results.items():
        for result in results:
            target_indices = result.metadata.get("test_target_indices") or list(range(len(result.y_true)))
            for step, (actual, predicted) in enumerate(zip(result.y_true, result.y_pred)):
                prediction_rows.append(
                    {
                        "Dataset": dataset_name,
                        "Model": result.name,
                        "Step": step,
                        "Target Index": target_indices[step] if step < len(target_indices) else step,
                        "Actual": actual,
                        "Prediction": predicted,
                    }
                )
        plot_forecast(
            results[0].y_true,
            {result.name: result.y_pred for result in results},
            figure_dir / f"forecast_predictions_{dataset_name.replace('/', '_').replace(':', '_')}.png",
            title=f"Forecasting Results - {dataset_name}",
        )
    pd.DataFrame(prediction_rows).to_csv(result_dir / "forecast_predictions.csv", index=False)

    # Save tuning metadata separately from metrics. This makes fair-comparison
    # settings explicit for readers inspecting the repository.
    pso_payload = {
        f"{dataset_name}/{result.name}": {name: pso.best_params for name, pso in result.pso_results.items()}
        for dataset_name, results in grouped_results.items()
        for result in results
        if result.pso_results
    }
    save_json(pso_payload, result_dir / "pso_best_params.json")
    save_json(
        {
            "model_defaults": config.get("model", {}),
            "model_overrides": config.get("model_overrides", {}),
            "hyperparameter_tuning": config.get("hyperparameter_tuning", {}),
            "pso_search_space": config.get("pso", {}).get("search_space", {}),
            "ceemdan_protocol": config.get("ceemdan", {}).get("forecasting_protocol", "causal_expanding"),
            "baseline_refs": config.get("baseline_refs", {}),
            "fairness_note": (
                "All methods are run on the same chronological 7:1:2 train/validation/test protocol. "
                "PSO validation fitness uses only the validation period, and the held-out test period is evaluated once. "
                "Paired statistical tests are emitted only for models with matching test lengths."
            ),
        },
        result_dir / "hyperparameter_protocol.json",
    )

    # Statistical tests operate on paired absolute errors from the same test
    # timestamps, which is appropriate for chronological forecasting outputs.
    preferred_reference = "ECDDD-CEEMDAN-PSO-BILSTM"
    wilcoxon_frames = []
    friedman_frames = []
    for dataset_name, results in grouped_results.items():
        # Paired statistical tests are valid only when models share exactly the
        # same target timestamps. ECDDD segmented models can evaluate different
        # held-out points, so compare only identical target-index groups.
        by_target_index: dict[tuple[int, ...], list[ForecastResult]] = {}
        for result in results:
            target_indices = tuple(int(idx) for idx in result.metadata.get("test_target_indices", []))
            if not target_indices:
                target_indices = tuple(range(len(result.y_true)))
            by_target_index.setdefault(target_indices, []).append(result)
        for target_indices, same_length_results in by_target_index.items():
            if len(same_length_results) < 2:
                continue
            prediction_map = {result.name: result.y_pred for result in same_length_results}
            ref = preferred_reference if preferred_reference in prediction_map else same_length_results[-1].name
            wdf = pairwise_wilcoxon_tests(same_length_results[0].y_true, prediction_map, reference_model=ref)
            wdf.insert(0, "Dataset", dataset_name)
            wdf.insert(1, "Test Length", len(target_indices))
            wilcoxon_frames.append(wdf)
            if len(prediction_map) >= 3:
                fdf = friedman_test(same_length_results[0].y_true, prediction_map)
                fdf.insert(0, "Dataset", dataset_name)
                fdf.insert(1, "Test Length", len(target_indices))
                friedman_frames.append(fdf)
    if wilcoxon_frames:
        pd.concat(wilcoxon_frames, ignore_index=True).to_csv(result_dir / "wilcoxon_tests.csv", index=False)
    if friedman_frames:
        pd.concat(friedman_frames, ignore_index=True).to_csv(result_dir / "friedman_test.csv", index=False)

    plot_metric_bars(metrics_df, figure_dir / "forecast_metrics.png")
    print(f"Saved forecast results to {result_dir}")
    print(metrics_df[["Dataset", "Model", "RMSE", "MAPE", "R2"]].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--quick", action="store_true", help="Run a short smoke-test version.")
    parser.add_argument(
        "--models",
        nargs="*",
        help="Optional explicit model list, e.g. lstm bilstm dlinear itransformer mlp_mixer ceemdan_pso_bilstm.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.quick:
        config = apply_quick_settings(config)
    set_seed(int(config.get("project", {}).get("seed", 42)))
    device = resolve_device(config.get("project", {}).get("device", "auto"))

    model_keys = args.models or list(config.get("paper_models", [])) + list(config.get("baselines", []))
    grouped_results = {}
    partition_rows = []
    for dataset_name, values, dates in dataset_slices(config):
        print(f"=== Dataset: {dataset_name}, n={len(values)} ===")
        partition_rows.append(split_summary(dataset_name, values, dates, config))
        grouped_results[dataset_name] = run_models(config, model_keys, device=device, series_values=values)
    grouped_results = {name: results for name, results in grouped_results.items() if results}
    if not grouped_results:
        raise RuntimeError("No forecasting model finished successfully.")
    save_outputs(config, grouped_results, partition_rows)


if __name__ == "__main__":
    main()
