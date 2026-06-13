"""Model parameter count, inference time, and GPU memory profiling."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ComplexityReport:
    parameters_m: float
    inference_time_s_per_seq: float
    gpu_memory_gb: float
    framework: str

    def as_dict(self) -> dict[str, float | str]:
        return {
            "Parameters (M)": self.parameters_m,
            "Inference Time (ms/seq)": self.inference_time_s_per_seq * 1000.0,
            "GPU Memory (GB)": self.gpu_memory_gb,
            "Framework": self.framework,
        }


def count_torch_parameters(model: Any) -> int:
    return int(sum(p.numel() for p in model.parameters() if getattr(p, "requires_grad", False)))


def count_parameters(forecaster: Any) -> int:
    if getattr(forecaster, "framework", "") == "torch":
        return count_torch_parameters(forecaster.model)
    return 0


def measure_inference_time(forecaster: Any, sample_x: np.ndarray, repeats: int = 100, batch_size: int = 1) -> float:
    """Measure average inference time per sequence."""

    x = np.asarray(sample_x, dtype=np.float32)
    if x.ndim == 2:
        x = x[None, ...]
    if len(x) == 0:
        raise ValueError("sample_x is empty.")
    n = min(max(batch_size, 1), len(x))
    batch = x[:n]

    # Warm up kernels/caches before timing, especially for CUDA models.
    for _ in range(5):
        forecaster.predict(batch, batch_size=n)
    start = time.perf_counter()
    for _ in range(max(repeats, 1)):
        forecaster.predict(batch, batch_size=n)
    elapsed = time.perf_counter() - start
    return float(elapsed / max(repeats * n, 1))


def measure_gpu_memory(forecaster: Any, sample_x: np.ndarray, batch_size: int = 1) -> float:
    """Profile peak CUDA memory in GB when available."""

    if getattr(forecaster, "framework", "") != "torch" or getattr(forecaster, "device", "cpu") != "cuda":
        return 0.0
    try:
        import torch

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        x = np.asarray(sample_x, dtype=np.float32)
        forecaster.predict(x[:batch_size], batch_size=batch_size)
        return float(torch.cuda.max_memory_allocated() / (1024**3))
    except Exception:
        return 0.0


def complexity_report(
    forecaster: Any,
    sample_x: np.ndarray,
    *,
    repeats: int = 100,
    batch_size: int = 1,
    profile_gpu_memory: bool = True,
) -> ComplexityReport:
    """Return inference complexity for an already fitted forecasting model.

    This helper intentionally measures the deployed prediction path only.
    `experiments.run_complexity` wraps model fitting separately to report the
    manuscript's training-time column, including CEEMDAN, ECDDD, and PSO costs.
    """

    params = count_parameters(forecaster)
    inference = measure_inference_time(forecaster, sample_x, repeats=repeats, batch_size=batch_size)
    gpu_mem = measure_gpu_memory(forecaster, sample_x, batch_size=batch_size) if profile_gpu_memory else 0.0
    return ComplexityReport(
        parameters_m=float(params / 1_000_000),
        inference_time_s_per_seq=inference,
        gpu_memory_gb=gpu_mem,
        framework=str(getattr(forecaster, "framework", "unknown")),
    )


def ensemble_complexity_report(
    fitted_models: dict[str, Any],
    sample_x: np.ndarray,
    *,
    repeats: int = 100,
    batch_size: int = 1,
    profile_gpu_memory: bool = True,
) -> ComplexityReport:
    """Aggregate complexity across component models in a CEEMDAN ensemble."""

    if not fitted_models:
        return ComplexityReport(0.0, 0.0, 0.0, "ensemble")
    reports = [
        complexity_report(
            model,
            sample_x,
            repeats=repeats,
            batch_size=batch_size,
            profile_gpu_memory=profile_gpu_memory,
        )
        for model in fitted_models.values()
    ]
    return ComplexityReport(
        parameters_m=float(sum(r.parameters_m for r in reports)),
        inference_time_s_per_seq=float(sum(r.inference_time_s_per_seq for r in reports)),
        gpu_memory_gb=float(sum(r.gpu_memory_gb for r in reports)),
        framework="ensemble",
    )
