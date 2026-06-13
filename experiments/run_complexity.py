"""Run model complexity evaluation."""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.data_loader import load_series_from_config, train_test_split_sequences
from models.ensemble import fit_direct_forecaster, fit_paper_model
from utils.common import load_config, resolve_device, resolve_output_dir, set_seed
from utils.complexity import complexity_report, ensemble_complexity_report


def quick_config(config: dict) -> dict:
    cfg = copy.deepcopy(config)
    cfg["model"].update({"epochs": 2, "patience": 1, "hidden_size": 16, "num_layers": 1})
    cfg["pso"].update({"particles": 2, "iterations": 1})
    cfg["ceemdan"].update({"trials": 5, "max_imfs": 2})
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
    cfg["complexity"].update({"repeats": 10})
    return cfg


def sample_x_from_config(series, config: dict):
    """Build representative test windows for inference profiling."""

    data_cfg = config.get("data", {})
    split = train_test_split_sequences(
        series,
        window_size=int(data_cfg.get("window_size", 5)),
        horizon=int(data_cfg.get("horizon", 1)),
        train_ratio=float(data_cfg.get("train_ratio", 0.7)),
        validation_ratio=float(data_cfg.get("validation_ratio", 0.1)),
        normalize=bool(data_cfg.get("normalize", True)),
    )
    return split["x_test"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--models", nargs="*")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.quick:
        config = quick_config(config)
    set_seed(int(config.get("project", {}).get("seed", 42)))
    device = resolve_device(config.get("project", {}).get("device", "auto"))
    result_dir = resolve_output_dir(config, "results")

    series = load_series_from_config(config).values
    complexity_cfg = config.get("complexity", {})
    sample_x = sample_x_from_config(series, config)
    model_keys = args.models or list(complexity_cfg.get("models", [])) or list(config.get("paper_models", [])) + list(
        config.get("baselines", [])
    )
    rows = []
    seen = set()

    for key in model_keys:
        key_lower = key.lower()
        if key_lower in seen:
            continue
        seen.add(key_lower)
        print(f"Profiling {key_lower}...")
        try:
            # Complexity is measured after fitting each model so parameter
            # counts and prediction paths match the actual experiment code.
            train_start = time.perf_counter()
            if key_lower in {
                "pso_bilstm",
                "ceemdan_bilstm",
                "ceemdan_pso_bilstm",
                "ecddd_ceemdan_pso_bilstm",
            }:
                result = fit_paper_model(key_lower, series, config, device=device)
            else:
                result = fit_direct_forecaster(series, config, model_name=key_lower, use_pso=False, device=device)
            training_time = time.perf_counter() - train_start
            if len(result.fitted_models) == 1:
                report = complexity_report(
                    next(iter(result.fitted_models.values())),
                    sample_x,
                    repeats=int(complexity_cfg.get("repeats", 100)),
                    batch_size=int(complexity_cfg.get("batch_size", 1)),
                    profile_gpu_memory=bool(complexity_cfg.get("profile_gpu_memory", True)),
                )
            else:
                report = ensemble_complexity_report(
                    result.fitted_models,
                    sample_x,
                    repeats=int(complexity_cfg.get("repeats", 100)),
                    batch_size=int(complexity_cfg.get("batch_size", 1)),
                    profile_gpu_memory=bool(complexity_cfg.get("profile_gpu_memory", True)),
                )
            rows.append(
                {
                    "Model": result.name,
                    "Complexity Scope": "training_pipeline_and_fitted_forecaster_inference",
                    "Training Time (s)": training_time,
                    **report.as_dict(),
                    **result.metrics,
                }
            )
        except Exception as exc:
            rows.append({"Model": key_lower, "Error": str(exc)})
            print(f"[WARN] {key_lower} failed: {exc}")

    df = pd.DataFrame(rows)
    df.to_csv(result_dir / "complexity_results.csv", index=False)
    print(f"Saved complexity results to {result_dir / 'complexity_results.csv'}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
