#!/usr/bin/env python3
"""PyTorch temporal residual model for BBB dual-brain inference."""
from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

WINDOW_SIZE = 10
INPUT_DIM = 10
NUM_HEADS = 2
KEY_DIM = 16
D_MODEL = NUM_HEADS * KEY_DIM
DROPOUT = 0.30
RANDOM_STATE = 42


@dataclass(frozen=True)
class TransformerTrainConfig:
    epochs: int = 120
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 15


class TemporalResidualTransformer(nn.Module):
    """10D -> 32D -> 2x16 attention -> masked GAP -> residual."""

    def __init__(
        self,
        *,
        input_dim: int = INPUT_DIM,
        num_heads: int = NUM_HEADS,
        key_dim: int = KEY_DIM,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_heads = int(num_heads)
        self.key_dim = int(key_dim)
        self.d_model = self.num_heads * self.key_dim

        self.input_projection = nn.Linear(self.input_dim, self.d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.dropout = nn.Dropout(float(dropout))
        self.output = nn.Linear(self.d_model, 1)

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("x must have shape (batch, window, features)")
        if valid_mask.ndim != 2:
            raise ValueError("valid_mask must have shape (batch, window)")

        h = self.input_projection(x)
        key_padding_mask = ~valid_mask.bool()
        attended, _ = self.attention(
            h,
            h,
            h,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        attended = self.dropout(attended)

        mask = valid_mask.to(attended.dtype).unsqueeze(-1)
        summed = torch.sum(attended * mask, dim=1)
        denom = torch.clamp(torch.sum(mask, dim=1), min=1.0)
        pooled = summed / denom
        return self.output(pooled).squeeze(-1)


def set_deterministic_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def build_sequence_windows(
    features_10d: np.ndarray,
    shoe_ids: Sequence[str],
    *,
    window_size: int = WINDOW_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(features_10d, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != INPUT_DIM:
        raise ValueError(f"features_10d must have shape (N, {INPUT_DIM})")
    if len(x) != len(shoe_ids):
        raise ValueError("features and shoe_ids must align")

    windows = np.zeros((len(x), window_size, INPUT_DIM), dtype=np.float32)
    masks = np.zeros((len(x), window_size), dtype=np.bool_)
    history: dict[str, list[np.ndarray]] = {}

    for i, (vector, shoe_id) in enumerate(zip(x, shoe_ids)):
        key = str(shoe_id)
        queue = history.setdefault(key, [])
        queue.append(np.asarray(vector, dtype=np.float32).copy())
        if len(queue) > window_size:
            del queue[0 : len(queue) - window_size]

        start = window_size - len(queue)
        windows[i, start:, :] = np.stack(queue, axis=0)
        masks[i, start:] = True

    return windows, masks


def predict_transformer(
    model: TemporalResidualTransformer,
    windows: np.ndarray,
    valid_masks: np.ndarray,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x = torch.as_tensor(windows, dtype=torch.float32)
        m = torch.as_tensor(valid_masks, dtype=torch.bool)
        values = model(x, m)
    return values.detach().cpu().numpy().astype(np.float32)


def train_transformer(
    windows: np.ndarray,
    valid_masks: np.ndarray,
    residual_targets: np.ndarray,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    *,
    config: TransformerTrainConfig | None = None,
) -> TemporalResidualTransformer:
    cfg = config or TransformerTrainConfig()
    set_deterministic_seed(RANDOM_STATE)

    x = np.asarray(windows, dtype=np.float32)
    m = np.asarray(valid_masks, dtype=np.bool_)
    y = np.asarray(residual_targets, dtype=np.float32).reshape(-1)
    train_mask = np.asarray(train_mask, dtype=np.bool_)
    validation_mask = np.asarray(validation_mask, dtype=np.bool_)

    if not (len(x) == len(m) == len(y) == len(train_mask) == len(validation_mask)):
        raise ValueError("training arrays must align")
    if not np.any(train_mask):
        raise ValueError("empty Transformer training split")
    if not np.any(validation_mask):
        raise ValueError("empty Transformer validation split")

    model = TemporalResidualTransformer()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    loss_fn = nn.MSELoss()

    train_dataset = TensorDataset(
        torch.as_tensor(x[train_mask], dtype=torch.float32),
        torch.as_tensor(m[train_mask], dtype=torch.bool),
        torch.as_tensor(y[train_mask], dtype=torch.float32),
    )
    generator = torch.Generator()
    generator.manual_seed(RANDOM_STATE)
    loader = DataLoader(
        train_dataset,
        batch_size=max(1, int(cfg.batch_size)),
        shuffle=True,
        generator=generator,
    )

    x_val = torch.as_tensor(x[validation_mask], dtype=torch.float32)
    m_val = torch.as_tensor(m[validation_mask], dtype=torch.bool)
    y_val = torch.as_tensor(y[validation_mask], dtype=torch.float32)

    best_state = copy.deepcopy(model.state_dict())
    best_loss = math.inf
    stale_epochs = 0

    for _ in range(max(1, int(cfg.epochs))):
        model.train()
        for batch_x, batch_m, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch_x, batch_m)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val, m_val)
            val_loss = float(loss_fn(val_pred, y_val).item())

        if val_loss < best_loss - 1e-7:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= max(1, int(cfg.patience)):
                break

    model.load_state_dict(best_state)
    model.eval()
    return model


def export_transformer_payload(
    model: TemporalResidualTransformer,
) -> dict[str, Any]:
    state = model.state_dict()

    def tensor(name: str) -> list[Any]:
        return state[name].detach().cpu().numpy().astype(np.float32).tolist()

    return {
        "trained": True,
        "window_size": WINDOW_SIZE,
        "input_dim": INPUT_DIM,
        "num_heads": NUM_HEADS,
        "key_dim": KEY_DIM,
        "d_model": D_MODEL,
        "dropout": DROPOUT,
        "padding": "left_zero_padding_with_valid_mask",
        "weights": {
            "input_projection_weight": tensor("input_projection.weight"),
            "input_projection_bias": tensor("input_projection.bias"),
            "in_proj_weight": tensor("attention.in_proj_weight"),
            "in_proj_bias": tensor("attention.in_proj_bias"),
            "out_proj_weight": tensor("attention.out_proj.weight"),
            "out_proj_bias": tensor("attention.out_proj.bias"),
            "dense_weight": tensor("output.weight"),
            "dense_bias": tensor("output.bias"),
        },
    }


class TemporalWindowBuffer:
    """Online current-shoe 10D queue with left-zero padding."""

    def __init__(self, window_size: int = WINDOW_SIZE) -> None:
        self.window_size = int(window_size)
        self.rows: list[np.ndarray] = []

    def reset(self) -> None:
        self.rows.clear()

    def push(self, feature_10d: Sequence[float]) -> None:
        vector = np.asarray(feature_10d, dtype=np.float32).reshape(-1)
        if vector.shape[0] != INPUT_DIM:
            raise ValueError(f"feature_10d must contain {INPUT_DIM} values")
        self.rows.append(vector.copy())
        if len(self.rows) > self.window_size:
            self.rows = self.rows[-self.window_size :]

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        window = np.zeros((1, self.window_size, INPUT_DIM), dtype=np.float32)
        mask = np.zeros((1, self.window_size), dtype=np.bool_)
        if self.rows:
            start = self.window_size - len(self.rows)
            window[0, start:, :] = np.stack(self.rows, axis=0)
            mask[0, start:] = True
        return window, mask
