"""Data loading and preprocessing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from data.synthetic_data import generate_carbon_like_series


@dataclass
class SeriesData:
    values: np.ndarray
    dates: pd.Series | None = None
    frame: pd.DataFrame | None = None
    target_column: str = "value"


class MinMaxScaler1D:
    """Small 1D scaler that avoids adding another abstraction dependency."""

    def __init__(self) -> None:
        self.min_: float = 0.0
        self.max_: float = 1.0

    def fit(self, values: Iterable[float]) -> "MinMaxScaler1D":
        arr = np.asarray(values, dtype=float)
        self.min_ = float(np.nanmin(arr))
        self.max_ = float(np.nanmax(arr))
        return self

    def transform(self, values: Iterable[float]) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        scale = self.max_ - self.min_
        if abs(scale) < 1e-12:
            return np.zeros_like(arr, dtype=float)
        return (arr - self.min_) / scale

    def inverse_transform(self, values: Iterable[float]) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        return arr * (self.max_ - self.min_) + self.min_

    def fit_transform(self, values: Iterable[float]) -> np.ndarray:
        return self.fit(values).transform(values)


def infer_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    normalized = {str(col).strip().lower(): str(col) for col in columns}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in normalized:
            return normalized[key]
    for col in columns:
        col_lower = str(col).lower()
        if any(candidate.lower() in col_lower for candidate in candidates):
            return str(col)
    return None


def load_carbon_csv(
    path: str | Path,
    target_column: str | None = None,
    date_column: str | None = None,
    country_column: str | None = None,
    country: str | None = None,
    sector_column: str | None = None,
    sector: str | None = None,
) -> SeriesData:
    """Load Carbon Monitor-style CSV data with forgiving column detection."""

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Data file is empty: {csv_path}")

    # The public Carbon Monitor file is a long table, but the loader also
    # accepts slightly different column names to make reproduction less brittle.
    date_column = date_column or infer_column(df.columns, ("date", "time", "day", "timestamp"))
    country_column = country_column or infer_column(df.columns, ("country", "nation", "region"))
    sector_column = sector_column or infer_column(df.columns, ("sector", "department", "category"))
    target_column = target_column or infer_column(
        df.columns,
        ("carbon_emission", "emission", "co2", "co2_emission", "value", "power"),
    )
    if target_column is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError("Could not infer a numeric target column.")
        target_column = numeric_cols[-1]

    if country and country_column and country_column in df.columns:
        # Empirical experiments in the manuscript use China; keeping this as a
        # filter makes the same loader usable for other countries later.
        mask = df[country_column].astype(str).str.lower() == country.lower()
        if mask.any():
            df = df.loc[mask].copy()
        else:
            raise ValueError(f"No rows found for country={country!r} in {csv_path}.")

    if sector and sector_column and sector_column in df.columns:
        # The main paper reports Power-sector forecasting, while drift
        # generalization can reuse this branch for Industry and Residential.
        mask = df[sector_column].astype(str).str.lower() == sector.lower()
        if mask.any():
            df = df.loc[mask].copy()
        elif sector in df.columns:
            target_column = sector
        else:
            raise ValueError(f"No rows found for sector={sector!r} in {csv_path}.")

    if date_column and date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
        df = df.sort_values(date_column)
        dates = df[date_column].reset_index(drop=True)
    else:
        dates = None

    values = pd.to_numeric(df[target_column], errors="coerce").interpolate().bfill().ffill()
    values = values.to_numpy(dtype=float)
    return SeriesData(values=values, dates=dates, frame=df.reset_index(drop=True), target_column=target_column)


def create_supervised_sequences(
    series: Iterable[float],
    window_size: int = 5,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a 1D series into rolling supervised samples."""

    arr = np.asarray(series, dtype=float).reshape(-1)
    if window_size <= 0 or horizon <= 0:
        raise ValueError("window_size and horizon must be positive.")
    total = len(arr) - window_size - horizon + 1
    if total <= 0:
        raise ValueError(
            f"Series length {len(arr)} is too short for window_size={window_size}, horizon={horizon}."
        )
    x = np.stack([arr[i : i + window_size] for i in range(total)])
    y = np.asarray([arr[i + window_size + horizon - 1] for i in range(total)], dtype=float)
    return x[..., None], y


def resolve_chronological_split_indices(
    n_samples: int,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.1,
) -> tuple[int, int]:
    """Return chronological train and validation end indices.

    The revised manuscript uses an absolute 7:1:2 split: 70% training,
    10% validation, and 20% held-out testing.
    """

    n = int(n_samples)
    if n < 3:
        raise ValueError("Need at least three supervised samples to split data.")
    train_ratio = float(train_ratio)
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")

    validation_ratio = float(validation_ratio)
    if validation_ratio <= 0.0 or train_ratio + validation_ratio >= 1.0:
        raise ValueError("For 7:1:2-style splitting, train_ratio + validation_ratio must be less than 1.")
    train_end = max(1, min(n - 2, int(n * train_ratio)))
    val_size = max(1, int(n * validation_ratio))
    val_end = max(train_end + 1, min(n - 1, train_end + val_size))
    return train_end, val_end


def chronological_split(
    x: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.1,
    target_indices: np.ndarray | None = None,
    n_observations: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split windows chronologically.

    The manuscript protocol is train:validation:test = 7:1:2. All boundaries
    are chronological. When target_indices are provided, windows are assigned
    by their raw target timestamp so the split matches the original series.
    """

    if target_indices is not None:
        targets = np.asarray(target_indices, dtype=int).reshape(-1)
        if len(targets) != len(x):
            raise ValueError("target_indices must have the same length as x and y.")
        raw_count = int(n_observations) if n_observations is not None else int(targets[-1] + 1)
        train_end, val_end = resolve_chronological_split_indices(
            raw_count,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
        )
        train_mask = targets < train_end
        val_mask = (targets >= train_end) & (targets < val_end)
        test_mask = targets >= val_end
        return (
            x[train_mask],
            y[train_mask],
            x[val_mask],
            y[val_mask],
            x[test_mask],
            y[test_mask],
        )

    train_end, val_end = resolve_chronological_split_indices(
        len(x),
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
    )
    return (
        x[:train_end],
        y[:train_end],
        x[train_end:val_end],
        y[train_end:val_end],
        x[val_end:],
        y[val_end:],
    )


def train_test_split_sequences(
    series: Iterable[float],
    window_size: int = 5,
    horizon: int = 1,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.1,
    normalize: bool = True,
) -> dict[str, object]:
    """Scale a series, create windows, and return chronological splits."""

    raw = np.asarray(series, dtype=float).reshape(-1)
    total_samples = len(raw) - window_size - horizon + 1
    if total_samples <= 0:
        raise ValueError(
            f"Series length {len(raw)} is too short for window_size={window_size}, horizon={horizon}."
        )
    # Split boundaries are defined on the original time axis. This matches the
    # manuscript's 7:1:2 protocol and avoids future values in preprocessing.
    raw_train_end, raw_val_end = resolve_chronological_split_indices(
        len(raw),
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
    )
    scaler = MinMaxScaler1D()
    if normalize:
        # The scaler sees only observations in the training period.
        # Validation/test windows are transformed with the training scaler.
        scaler.fit(raw[:raw_train_end])
        scaled = scaler.transform(raw)
    else:
        scaled = raw.copy()
    x, y = create_supervised_sequences(scaled, window_size=window_size, horizon=horizon)
    target_indices = np.arange(window_size + horizon - 1, len(raw), dtype=int)
    x_train, y_train, x_val, y_val, x_test, y_test = chronological_split(
        x,
        y,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        target_indices=target_indices,
        n_observations=len(raw),
    )
    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
        "train_target_indices": target_indices[target_indices < raw_train_end],
        "validation_target_indices": target_indices[
            (target_indices >= raw_train_end) & (target_indices < raw_val_end)
        ],
        "test_target_indices": target_indices[target_indices >= raw_val_end],
        "scaler": scaler,
        "scaled_series": scaled,
        "raw_series": raw,
    }


def load_series_from_config(config: Mapping[str, Any]) -> SeriesData:
    """Load the configured China-sector Carbon Monitor series.

    If the configured CSV is unavailable, a carbon-like synthetic series is
    returned so that experiment scripts still run for smoke testing.
    """

    data_cfg = config.get("data", {}) if isinstance(config, Mapping) else {}
    csv_path = data_cfg.get("csv_path")
    if csv_path:
        from utils.common import project_path

        csv_path_obj = Path(str(csv_path))
        if not csv_path_obj.is_absolute():
            csv_path_obj = project_path(csv_path_obj)
    else:
        csv_path_obj = None
    if csv_path_obj and csv_path_obj.exists():
        return load_carbon_csv(
            csv_path_obj,
            target_column=data_cfg.get("target_column"),
            date_column=data_cfg.get("date_column"),
            country_column=data_cfg.get("country_column"),
            country=data_cfg.get("country"),
            sector_column=data_cfg.get("sector_column"),
            sector=data_cfg.get("sector"),
        )
    values = generate_carbon_like_series(seed=int(config.get("project", {}).get("seed", 42)))
    return SeriesData(values=values, dates=None, frame=None, target_column=str(data_cfg.get("target_column", "co2")))


def make_torch_loaders(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    batch_size: int = 32,
    shuffle: bool = True,
):
    """Create PyTorch DataLoaders from numpy arrays."""

    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:
        raise ImportError("PyTorch is required for neural forecasting models.") from exc

    train_ds = TensorDataset(
        torch.as_tensor(x_train, dtype=torch.float32),
        torch.as_tensor(y_train.reshape(-1, 1), dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    val_loader = None
    if x_val is not None and y_val is not None and len(x_val):
        val_ds = TensorDataset(
            torch.as_tensor(x_val, dtype=torch.float32),
            torch.as_tensor(y_val.reshape(-1, 1), dtype=torch.float32),
        )
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
