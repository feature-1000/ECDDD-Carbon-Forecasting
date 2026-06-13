"""Run drift detection experiments.

Outputs:
  outputs/results/drift_detection_raw.csv
  outputs/results/drift_detection_summary.csv
  outputs/results/power_drift_points.csv
  outputs/results/sector_drift_points.csv
  outputs/figures/*_drift_detection.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.data_loader import load_series_from_config
from data.synthetic_data import generate_paper_group
from models.drift_detection import ECDDDDetector, ELMDriftDetector, FEDDDetector
from utils.common import load_config, resolve_output_dir, set_seed
from utils.metrics import drift_detection_metrics
from utils.plot import plot_drift_detection


def build_detectors(config: dict, quick: bool = False):
    """Instantiate ECDDD and lightweight comparison detectors."""

    drift_cfg = config.get("drift_detection", {})
    ecddd_cfg = dict(drift_cfg.get("ecddd", {}))
    seed = int(config.get("project", {}).get("seed", 42))
    if quick:
        ecddd_cfg["ceemdan_trials"] = min(int(ecddd_cfg.get("ceemdan_trials", 100)), 10)
        ecddd_cfg["ceemdan_max_imfs"] = 3
    common = {
        "window_size": int(ecddd_cfg.get("window_size", 30)),
        "alpha": float(ecddd_cfg.get("alpha", 0.01)),
        "threshold": ecddd_cfg.get("threshold"),
        "refractory": int(ecddd_cfg.get("refractory", 30)),
    }
    return {
        "ECDDD": ECDDDDetector(
            **common,
            ceemdan_trials=int(ecddd_cfg.get("ceemdan_trials", 100)),
            ceemdan_noise_width=float(ecddd_cfg.get("ceemdan_noise_width", 0.2)),
            ceemdan_max_imfs=ecddd_cfg.get("ceemdan_max_imfs", 4),
            seed=seed,
        ),
        "FEDD": FEDDDetector(**common),
        "ELM": ELMDriftDetector(window_size=int(common["window_size"]), seed=seed),
    }


def run_synthetic(config: dict, quick: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate detectors on the three AR drift groups from the manuscript."""

    syn_cfg = config.get("synthetic", {})
    groups = ("group1", "group2", "group3")
    runs = 3 if quick else 30
    length = 600 if quick else int(syn_cfg.get("sequence_length", 2000))
    rows: list[dict[str, object]] = []

    for group in groups:
        for run in range(runs):
            mode = "gradual" if run >= runs // 2 else "abrupt"
            series, true_drifts = generate_paper_group(
                group,
                length=length,
                drift_mode=mode,
                seed=int(config.get("project", {}).get("seed", 42)) + run,
            )
            for name, detector in build_detectors(config, quick=quick).items():
                result = detector.detect(series)
                metrics = drift_detection_metrics(true_drifts, result.drift_points, tolerance=max(60, length // 10))
                row = {
                    "Group": group,
                    "Run": run,
                    "Mode": mode,
                    "Framework": name,
                    "True Drifts": ",".join(map(str, true_drifts)),
                    "Detected Drifts": ",".join(map(str, result.drift_points)),
                }
                row.update(metrics.as_dict())
                rows.append(row)

    raw = pd.DataFrame(rows)
    summary = (
        raw.groupby(["Group", "Framework"], as_index=False)
        .agg(
            {
                "Detection Delay": ["mean", "std"],
                "Detection Position Offset": ["mean", "std"],
                "False Alarms": ["mean", "std"],
                "Miss Detection Numbers": ["mean", "std"],
            }
        )
        .round(4)
    )
    summary.columns = [" ".join(col).strip() if isinstance(col, tuple) else col for col in summary.columns]
    return raw, summary


def run_real_sector(config: dict, sector: str, quick: bool = False) -> tuple[pd.DataFrame, object]:
    """Run ECDDD on one real Carbon Monitor sector."""

    sector_config = dict(config)
    sector_config["data"] = dict(config.get("data", {}))
    sector_config["data"]["sector"] = sector
    series_data = load_series_from_config(sector_config)
    detector = build_detectors(config, quick=quick)["ECDDD"]
    result = detector.detect(series_data.values)
    dates = []
    if series_data.dates is not None:
        for point in result.drift_points:
            if 0 <= point < len(series_data.dates):
                dates.append(str(series_data.dates.iloc[point].date()))
            else:
                dates.append("")
    else:
        dates = [""] * len(result.drift_points)
    return pd.DataFrame(
        {
            "Sector": sector,
            "Index": result.drift_points,
            "Date": dates,
            "Detector": result.detector,
            "Threshold": result.threshold,
        }
    ), series_data


def run_real_sectors(config: dict, quick: bool = False) -> pd.DataFrame:
    """Run sector-level drift detection and save one figure per sector."""

    sectors = config.get("data", {}).get("compare_sectors") or [config.get("data", {}).get("sector", "Power")]
    rows = []
    figure_dir = resolve_output_dir(config, "figures")
    for sector in sectors:
        points, series_data = run_real_sector(config, str(sector), quick=quick)
        rows.append(points)
        plot_drift_detection(
            series_data.values,
            points["Index"].tolist(),
            figure_dir / f"{str(sector).lower().replace(' ', '_')}_drift_detection.png",
            title=f"{config.get('data', {}).get('country', 'China')} {sector} Drift Detection",
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--quick", action="store_true", help="Run a short smoke-test version.")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.get("project", {}).get("seed", 42)))
    result_dir = resolve_output_dir(config, "results")

    raw, summary = run_synthetic(config, quick=args.quick)
    raw.to_csv(result_dir / "drift_detection_raw.csv", index=False)
    summary.to_csv(result_dir / "drift_detection_summary.csv", index=False)

    sector_points = run_real_sectors(config, quick=args.quick)
    sector_points.to_csv(result_dir / "sector_drift_points.csv", index=False)
    power_points = sector_points[sector_points["Sector"] == config.get("data", {}).get("sector", "Power")]
    power_points.to_csv(result_dir / "power_drift_points.csv", index=False)
    print(f"Saved drift results to {result_dir}")
    print(summary.to_string(index=False))
    print(sector_points.to_string(index=False))


if __name__ == "__main__":
    main()
