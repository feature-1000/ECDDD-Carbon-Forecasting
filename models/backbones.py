"""Forecasting backbones used by the manuscript comparison experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from data.data_loader import make_torch_loaders
from utils.common import set_seed


TorchBackbone = Literal["lstm", "bilstm", "itransformer", "dlinear", "mlp_mixer"]

TORCH_BACKBONES = {"lstm", "bilstm", "itransformer", "dlinear", "mlp_mixer"}


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_epoch: int = 0


@dataclass
class FittedForecaster:
    name: str
    model: Any
    framework: str
    params: dict[str, Any]
    history: TrainHistory | None = None
    device: str = "cpu"

    def predict(self, x: np.ndarray, batch_size: int = 256) -> np.ndarray:
        return predict_forecaster(self, x, batch_size=batch_size)


def _import_torch():
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        raise ImportError("PyTorch is required for neural forecasting backbones.") from exc
    return torch, nn


def _last_step_output(output):
    return output[:, -1, :]


def build_torch_model(
    name: str,
    input_size: int = 1,
    window_size: int = 5,
    hidden_size: int = 64,
    num_layers: int = 1,
    dropout: float = 0.0,
    n_heads: int | None = None,
    moving_average_window: int | None = None,
) -> Any:
    """Build a PyTorch forecasting model."""

    torch, nn = _import_torch()
    key = name.lower()

    def valid_head_count() -> int:
        requested = int(n_heads or 0)
        candidates = (requested, 8, 4, 2, 1) if requested > 0 else (8, 4, 2, 1)
        for candidate in candidates:
            if candidate > 0 and hidden_size % candidate == 0:
                return candidate
        return 1

    class RecurrentForecaster(nn.Module):
        def __init__(self, bidirectional: bool = False) -> None:
            super().__init__()
            recurrent_dropout = dropout if num_layers > 1 else 0.0
            self.rnn = nn.LSTM(
                input_size,
                hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=recurrent_dropout,
                bidirectional=bidirectional,
            )
            out_size = hidden_size * (2 if bidirectional else 1)
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(out_size, 1))

        def forward(self, x):
            output, _ = self.rnn(x)
            return self.head(_last_step_output(output))

    class ITransformerForecaster(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            nhead = valid_head_count()
            self.value_projection = nn.Linear(window_size, hidden_size)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=nhead,
                dim_feedforward=max(hidden_size * 2, 32),
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(num_layers, 1))
            self.head = nn.Sequential(nn.LayerNorm(hidden_size * input_size), nn.Linear(hidden_size * input_size, 1))

        def forward(self, x):
            # iTransformer inverts the tokenization: each variable is treated as
            # a token and its temporal history is projected as the feature
            # vector. For the univariate carbon-emission series this reduces to
            # one variate token, while keeping the manuscript's model identity.
            y = x.transpose(1, 2)
            y = self.value_projection(y)
            y = self.encoder(y)
            return self.head(y.flatten(start_dim=1))

    class DLinearForecaster(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            ma_window = int(moving_average_window or window_size)
            self.ma_window = max(1, min(ma_window, window_size))
            self.trend = nn.Linear(window_size, 1)
            self.seasonal = nn.Linear(window_size, 1)

        def forward(self, x):
            y = x.squeeze(-1)
            # DLinear decomposes the input into a moving-average trend and a
            # residual seasonal term before applying separate linear heads.
            if self.ma_window <= 1:
                trend = y
            else:
                left = self.ma_window // 2
                right = self.ma_window - 1 - left
                padded = nn.functional.pad(y.unsqueeze(1), (left, right), mode="replicate")
                trend = nn.functional.avg_pool1d(padded, kernel_size=self.ma_window, stride=1).squeeze(1)
            seasonal = y - trend
            return self.trend(trend) + self.seasonal(seasonal)

    class MixerBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.token_norm = nn.LayerNorm(hidden_size)
            self.token_mixer = nn.Sequential(
                nn.Linear(window_size, window_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(window_size, window_size),
            )
            self.channel_norm = nn.LayerNorm(hidden_size)
            self.channel_mixer = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 2, hidden_size),
            )

        def forward(self, x):
            # Token mixing operates across time steps; channel mixing operates
            # across hidden features, matching the MLP-Mixer baseline idea.
            tokens = self.token_norm(x).transpose(1, 2)
            y = x + self.token_mixer(tokens).transpose(1, 2)
            return y + self.channel_mixer(self.channel_norm(y))

    class MLPMixerForecaster(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = nn.Linear(input_size, hidden_size)
            self.blocks = nn.Sequential(*[MixerBlock() for _ in range(max(num_layers, 1))])
            self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Flatten(), nn.Linear(window_size * hidden_size, 1))

        def forward(self, x):
            y = self.projection(x)
            y = self.blocks(y)
            return self.head(y)

    if key == "lstm":
        return RecurrentForecaster(bidirectional=False)
    if key == "bilstm":
        return RecurrentForecaster(bidirectional=True)
    if key == "itransformer":
        return ITransformerForecaster()
    if key == "dlinear":
        return DLinearForecaster()
    if key == "mlp_mixer":
        return MLPMixerForecaster()
    raise ValueError(f"Unknown torch backbone: {name}")


def train_torch_forecaster(
    name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    *,
    hidden_size: int = 64,
    num_layers: int = 1,
    dropout: float = 0.0,
    n_heads: int | None = None,
    moving_average_window: int | None = None,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    scheduler: str | None = None,
    epochs: int = 100,
    batch_size: int = 32,
    patience: int = 10,
    device: str = "cpu",
    seed: int = 42,
) -> FittedForecaster:
    """Train a neural forecaster with early stopping."""

    set_seed(seed)
    torch, nn = _import_torch()
    x_train = np.asarray(x_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32).reshape(-1)
    x_val_arr = None if x_val is None else np.asarray(x_val, dtype=np.float32)
    y_val_arr = None if y_val is None else np.asarray(y_val, dtype=np.float32).reshape(-1)

    model = build_torch_model(
        name,
        input_size=x_train.shape[-1],
        window_size=x_train.shape[1],
        hidden_size=int(hidden_size),
        num_layers=int(num_layers),
        dropout=float(dropout),
        n_heads=n_heads,
        moving_average_window=moving_average_window,
    ).to(device)
    train_loader, val_loader = make_torch_loaders(
        x_train,
        y_train,
        x_val_arr,
        y_val_arr,
        batch_size=batch_size,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    lr_scheduler = None
    if scheduler and str(scheduler).lower() in {"cosine", "cosine_annealing", "cosineannealing"}:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(epochs), 1))
    criterion = nn.MSELoss()
    history = TrainHistory()
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val = float("inf")
    stale_epochs = 0

    for epoch in range(int(epochs)):
        # Early stopping monitors validation loss when provided. During final
        # fitting, callers may pass no validation set and train on train+val.
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        train_loss = float(np.mean(losses)) if losses else float("nan")
        history.train_loss.append(train_loss)

        model.eval()
        if val_loader is None:
            val_loss = train_loss
        else:
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    val_losses.append(float(criterion(model(xb), yb).detach().cpu()))
            val_loss = float(np.mean(val_losses)) if val_losses else train_loss
        history.val_loss.append(val_loss)

        if val_loss < best_val - 1e-10:
            best_val = val_loss
            history.best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if patience > 0 and stale_epochs >= patience:
            break
        if lr_scheduler is not None:
            lr_scheduler.step()

    model.load_state_dict(best_state)
    params = {
        "hidden_size": int(hidden_size),
        "num_layers": int(num_layers),
        "dropout": float(dropout),
        "n_heads": n_heads,
        "moving_average_window": moving_average_window,
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "scheduler": scheduler,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "patience": int(patience),
    }
    return FittedForecaster(name=name, model=model, framework="torch", params=params, history=history, device=device)


def train_forecaster(
    name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    **params: Any,
) -> FittedForecaster:
    """Train any supported forecasting backbone."""

    key = name.lower()
    if key in TORCH_BACKBONES:
        return train_torch_forecaster(key, x_train, y_train, x_val, y_val, **params)
    raise ValueError(f"Unknown forecaster {name!r}. Supported: {sorted(TORCH_BACKBONES)}")


def predict_forecaster(forecaster: FittedForecaster, x: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Predict with a fitted PyTorch forecaster."""

    x_arr = np.asarray(x, dtype=np.float32)
    torch, _ = _import_torch()
    model = forecaster.model
    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x_arr), batch_size):
            xb = torch.as_tensor(x_arr[start : start + batch_size], dtype=torch.float32).to(forecaster.device)
            pred = model(xb).detach().cpu().numpy().reshape(-1)
            preds.append(pred)
    return np.concatenate(preds) if preds else np.array([], dtype=float)


def supported_backbones() -> list[str]:
    return sorted(TORCH_BACKBONES)
