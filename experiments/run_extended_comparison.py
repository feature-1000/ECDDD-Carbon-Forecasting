"""Run the second-revision extended comparison corresponding to Fig. 11."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Callable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.data_loader import load_series_from_config
from models.ensemble import ForecastResult, fit_decomposition_ensemble, fit_direct_forecaster
from utils.common import load_config, resolve_device, resolve_output_dir, save_json, set_seed
from utils.plot import plot_metric_bars


def display_name(value: str) -> str:
    return str(value).upper().replace("_", "-")


def quick_config(config: dict) -> dict:
    """Shrink every expensive loop for smoke tests without changing logic."""

    cfg = copy.deepcopy(config)
    cfg["model"].update({"epochs": 2, "patience": 1, "hidden_size": 16, "num_layers": 1, "batch_size": 32})
    cfg["pso"].update({"particles": 2, "iterations": 1, "search_space": {"hidden_size": [8, 16]}})
    cfg["ceemdan"].update({"trials": 5, "max_imfs": 2})
    cfg.setdefault("vmd", {}).update({"modes": 2})
    cfg.setdefault("metaheuristics", {}).setdefault("ga", {}).update({"population": 4, "generations": 1})
    cfg.setdefault("metaheuristics", {}).setdefault("gwo", {}).update({"wolves": 3, "iterations": 1})
    cfg["model_overrides"] = dict(cfg.get("model_overrides", {}))
    for name in ("lstm", "gru", "bilstm"):
        cfg["model_overrides"][name] = {"hidden_size": 16, "learning_rate": 0.001, "dropout": 0.1}
    cfg["model_overrides"]["cnn_lstm"] = {
        "hidden_size": 16,
        "cnn_channels": 8,
        "cnn_kernel_size": 3,
        "learning_rate": 0.001,
        "dropout": 0.1,
    }
    return cfg


def result_to_row(
    result: ForecastResult,
    *,
    group: str,
    configuration: str,
    backbone: str,
    decomposition: str = "",
    optimizer: str = "",
) -> dict[str, object]:
    row: dict[str, object] = {
        "Dataset": "Unsegmented Power",
        "Group": group,
        "Configuration": configuration,
        "Model": result.name,
        "Backbone": display_name(backbone),
        "Decomposition": display_name(decomposition) if decomposition else "",
        "Optimizer": display_name(optimizer) if optimizer else "",
        "Protocol": result.metadata.get("forecasting_protocol", "direct"),
        "Split Protocol": "chronological_7_1_2",
        "Runtime Decomposition": result.metadata.get("decomposition_method", ""),
        "Configured Decomposition": result.metadata.get("decomposition_method_configured", decomposition),
    }
    row.update(result.metrics)
    if result.pso_results:
        row["Optimizer Method"] = ",".join(sorted(opt.method for opt in result.pso_results.values()))
        row["Validation Best Score"] = min(float(opt.best_score) for opt in result.pso_results.values())
        row["Best Params"] = json.dumps(
            {name: opt.best_params for name, opt in result.pso_results.items()},
            ensure_ascii=False,
            sort_keys=True,
        )
    return row


def error_row(
    *,
    group: str,
    configuration: str,
    backbone: str,
    decomposition: str = "",
    optimizer: str = "",
    error: Exception,
) -> dict[str, object]:
    return {
        "Dataset": "Unsegmented Power",
        "Group": group,
        "Configuration": configuration,
        "Model": configuration,
        "Backbone": display_name(backbone),
        "Decomposition": display_name(decomposition) if decomposition else "",
        "Optimizer": display_name(optimizer) if optimizer else "",
        "Split Protocol": "chronological_7_1_2",
        "Error": str(error),
    }


def append_result(
    rows: list[dict[str, object]],
    factory: Callable[[], ForecastResult],
    *,
    group: str,
    configuration: str,
    backbone: str,
    decomposition: str = "",
    optimizer: str = "",
) -> None:
    try:
        result = factory()
        rows.append(
            result_to_row(
                result,
                group=group,
                configuration=configuration,
                backbone=backbone,
                decomposition=decomposition,
                optimizer=optimizer,
            )
        )
    except Exception as exc:
        rows.append(
            error_row(
                group=group,
                configuration=configuration,
                backbone=backbone,
                decomposition=decomposition,
                optimizer=optimizer,
                error=exc,
            )
        )
        print(f"[WARN] {group}/{configuration} failed: {exc}")


def run_decomposition_block(series, config: dict, device: str) -> list[dict[str, object]]:
    """Compare EMD, EEMD, VMD, and CEEMDAN under the same forecasting protocol."""

    rows: list[dict[str, object]] = []
    ext_cfg = config.get("extended_comparison", {})
    methods = ext_cfg.get("decomposition_methods", ["emd", "eemd", "vmd", "ceemdan"])
    backbones = ext_cfg.get("decomposition_backbones", ext_cfg.get("recurrent_backbones", ["bilstm"]))
    for method in methods:
        for backbone in backbones:
            configuration = f"{display_name(method)}-{display_name(backbone)}"
            print(f"Running extended decomposition comparison: {configuration}")
            append_result(
                rows,
                lambda method=method, backbone=backbone: fit_decomposition_ensemble(
                    series,
                    config,
                    decomposition_method=method,
                    backbone=backbone,
                    use_pso=False,
                    device=device,
                ),
                group="decomposition_methods",
                configuration=configuration,
                backbone=backbone,
                decomposition=method,
            )
    return rows


def run_optimizer_block(series, config: dict, device: str) -> list[dict[str, object]]:
    """Compare GA, GWO, and PSO using validation-only hyperparameter selection."""

    rows: list[dict[str, object]] = []
    ext_cfg = config.get("extended_comparison", {})
    optimizers = ext_cfg.get("optimization_methods", ["ga", "gwo", "pso"])
    backbones = ext_cfg.get("optimization_backbones", ext_cfg.get("recurrent_backbones", ["lstm", "gru", "bilstm", "cnn_lstm"]))
    for optimizer in optimizers:
        for backbone in backbones:
            configuration = f"{display_name(optimizer)}-{display_name(backbone)}"
            print(f"Running extended optimizer comparison: {configuration}")
            append_result(
                rows,
                lambda optimizer=optimizer, backbone=backbone: fit_direct_forecaster(
                    series,
                    config,
                    model_name=backbone,
                    use_pso=True,
                    optimizer=optimizer,
                    device=device,
                ),
                group="optimization_algorithms",
                configuration=configuration,
                backbone=backbone,
                optimizer=optimizer,
            )
    return rows


def run_configuration_block(series, config: dict, device: str) -> list[dict[str, object]]:
    """Compare Raw, CEEMDAN, PSO, and CEEMDAN+PSO with recurrent backbones."""

    rows: list[dict[str, object]] = []
    ext_cfg = config.get("extended_comparison", {})
    configurations = ext_cfg.get("configurations", ["raw", "ceemdan", "pso", "ceemdan_pso"])
    backbones = ext_cfg.get("recurrent_backbones", ["lstm", "gru", "bilstm", "cnn_lstm"])
    for configuration_key in configurations:
        key = str(configuration_key).lower().replace("-", "_")
        for backbone in backbones:
            configuration = f"{display_name(configuration_key)}-{display_name(backbone)}"
            print(f"Running extended configuration comparison: {configuration}")
            if key == "raw":
                append_result(
                    rows,
                    lambda backbone=backbone: fit_direct_forecaster(
                        series,
                        config,
                        model_name=backbone,
                        use_pso=False,
                        device=device,
                    ),
                    group="configuration_matrix",
                    configuration=configuration,
                    backbone=backbone,
                )
            elif key == "ceemdan":
                append_result(
                    rows,
                    lambda backbone=backbone: fit_decomposition_ensemble(
                        series,
                        config,
                        decomposition_method="ceemdan",
                        backbone=backbone,
                        use_pso=False,
                        device=device,
                    ),
                    group="configuration_matrix",
                    configuration=configuration,
                    backbone=backbone,
                    decomposition="ceemdan",
                )
            elif key == "pso":
                append_result(
                    rows,
                    lambda backbone=backbone: fit_direct_forecaster(
                        series,
                        config,
                        model_name=backbone,
                        use_pso=True,
                        optimizer="pso",
                        device=device,
                    ),
                    group="configuration_matrix",
                    configuration=configuration,
                    backbone=backbone,
                    optimizer="pso",
                )
            elif key in {"ceemdan_pso", "ceemdan+pso"}:
                append_result(
                    rows,
                    lambda backbone=backbone: fit_decomposition_ensemble(
                        series,
                        config,
                        decomposition_method="ceemdan",
                        backbone=backbone,
                        use_pso=True,
                        optimizer="pso",
                        device=device,
                    ),
                    group="configuration_matrix",
                    configuration=configuration,
                    backbone=backbone,
                    decomposition="ceemdan",
                    optimizer="pso",
                )
            else:
                rows.append(
                    error_row(
                        group="configuration_matrix",
                        configuration=configuration,
                        backbone=backbone,
                        error=ValueError(f"Unknown configuration {configuration_key!r}."),
                    )
                )
    return rows


def save_outputs(rows: list[dict[str, object]], config: dict) -> None:
    result_dir = resolve_output_dir(config, "results")
    figure_dir = resolve_output_dir(config, "figures")
    df = pd.DataFrame(rows)
    df.to_csv(result_dir / "extended_comparison_results.csv", index=False)
    save_json(
        {
            "dataset": "unsegmented China Power series",
            "split_protocol": "chronological_7_1_2",
            "forecasting_protocol": config.get("ceemdan", {}).get("forecasting_protocol", "causal_expanding"),
            "normalization": "training-period scaler only",
            "groups": config.get("extended_comparison", {}),
            "optimizer_search_space": config.get("pso", {}).get("search_space", {}),
            "metaheuristics": config.get("metaheuristics", {}),
        },
        result_dir / "extended_comparison_protocol.json",
    )
    if {"Model", "RMSE", "MAPE", "R2"}.issubset(df.columns):
        ok = df[df.get("Error").isna()] if "Error" in df.columns else df
        if not ok.empty:
            plot_df = ok.copy()
            plot_df["Model"] = plot_df["Group"].astype(str) + " / " + plot_df["Model"].astype(str)
            plot_metric_bars(
                plot_df,
                figure_dir / "extended_comparison_fig11.png",
                title="Extended Comparison for Fig. 11",
            )
    print(f"Saved extended comparison results to {result_dir / 'extended_comparison_results.csv'}")
    print(df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--groups",
        nargs="*",
        choices=["decomposition", "optimization", "configuration"],
        help="Optional subset of Fig. 11 groups to run.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.quick:
        config = quick_config(config)
    set_seed(int(config.get("project", {}).get("seed", 42)))
    device = resolve_device(config.get("project", {}).get("device", "auto"))
    series = load_series_from_config(config).values

    groups = set(args.groups or ["decomposition", "optimization", "configuration"])
    rows: list[dict[str, object]] = []
    if "decomposition" in groups:
        rows.extend(run_decomposition_block(series, config, device))
    if "optimization" in groups:
        rows.extend(run_optimizer_block(series, config, device))
    if "configuration" in groups:
        rows.extend(run_configuration_block(series, config, device))
    save_outputs(rows, config)


if __name__ == "__main__":
    main()
