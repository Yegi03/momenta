"""
src/models/temporal.py
Temporal aggregation for MOMENTA with optional timestamp-aware Transformer.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class TimestampPE(nn.Module):
    """Continuous timestamp positional encoding."""

    def __init__(self, d_model: int, n_freqs: int = 16):
        super().__init__()
        log_freqs = torch.linspace(
            math.log(1 / 3600),
            math.log(1 / 86400 / 365),
            n_freqs,
        )
        self.register_buffer("log_freqs", log_freqs)
        self.proj = nn.Linear(2 * n_freqs, d_model)

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        ts_min = timestamps.min()
        ts_max = timestamps.max()
        ts_norm = (timestamps - ts_min) / (ts_max - ts_min + 1e-8)
        freqs = torch.exp(self.log_freqs).to(timestamps.device)
        angles = ts_norm.unsqueeze(-1) * freqs.unsqueeze(0)
        pe_raw = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.proj(pe_raw)


class TemporalAggregator(nn.Module):
    """
    Stage A: per-window attention aggregation with drift and momentum.
    Stage B (optional): Transformer over the window sequence with timestamp PE.
    """

    def __init__(
        self,
        d: int,
        decay_alpha: float = 0.5,
        momentum_beta: float = 0.9,
        use_transformer: bool = True,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d = d
        self.alpha = decay_alpha
        self.beta = momentum_beta
        self.use_transformer = use_transformer

        # Per-window attention + linear head
        self.W_Q = nn.Linear(d, d, bias=False)
        self.W_K = nn.Linear(d, d, bias=False)
        self.W_V = nn.Linear(d, d, bias=False)
        self.classifier = nn.Linear(2 * d + 1, 1)

        if use_transformer:
            self.t_event_proj = nn.Linear(2 * d + 1, d)
            self.ts_pe = TimestampPE(d_model=d)
            self.input_ln = nn.LayerNorm(d)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d,
                nhead=n_heads,
                dim_feedforward=4 * d,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.transformer_classifier = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(
        self,
        window_features: torch.Tensor,  # (B, T, d)
        timestamps: torch.Tensor,  # (B, T)
        prev_L: Optional[torch.Tensor] = None,
        prev_M: Optional[torch.Tensor] = None,
        pad_mask: Optional[torch.Tensor] = None,  # (B, T) True for padding
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, d = window_features.shape
        device = window_features.device

        if prev_L is None:
            prev_L = torch.zeros(B, d, device=device, dtype=window_features.dtype)
        if prev_M is None:
            prev_M = torch.zeros(B, 1, device=device, dtype=window_features.dtype)

        if pad_mask is None:
            pad_mask = torch.zeros((B, T), dtype=torch.bool, device=device)

        valid = ~pad_mask  # (B, T)
        valid_f = valid.float()
        valid_count = valid_f.sum(dim=1, keepdim=True).clamp(min=1.0)  # (B, 1)

        # Ignore padded positions when computing the window max timestamp.
        t_max = timestamps.masked_fill(pad_mask, -1e18).max(dim=1, keepdim=True).values
        dt = (t_max - timestamps).clamp(min=0.0) / 86400.0
        lam = torch.exp(-self.alpha * dt)
        lam = lam.masked_fill(pad_mask, 0.0)

        # Query is the mean over non-padded window features.
        q = (window_features * valid_f.unsqueeze(-1)).sum(dim=1, keepdim=True) / valid_count.unsqueeze(-1)
        Q = self.W_Q(q)
        K = self.W_K(window_features)
        V = self.W_V(window_features)
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d)

        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask.unsqueeze(1), -1e9)
        attn = torch.softmax(scores, dim=-1).squeeze(1)  # (B, T)

        w = lam * attn
        w = w / (w.sum(dim=1, keepdim=True) + 1e-8)
        L_t = torch.bmm(w.unsqueeze(1), V).squeeze(1)

        delta_t = L_t - prev_L
        M_t = self.beta * prev_M + (1.0 - self.beta) * delta_t.norm(dim=1, keepdim=True)

        T_event = torch.cat([L_t, delta_t, M_t], dim=-1)
        p = self.classifier(T_event).squeeze(-1)
        return p, T_event, L_t, M_t

    def encode_sequence(
        self,
        t_events: torch.Tensor,  # (W, 2d+1)
        win_timestamps: torch.Tensor,  # (W,)
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.use_transformer or t_events.shape[0] < 2:
            return None, None

        x = self.t_event_proj(t_events)
        pe = self.ts_pe(win_timestamps)
        x = self.input_ln(x + pe)

        hidden = self.temporal_transformer(x.unsqueeze(0)).squeeze(0)  # (W, d)
        logits = self.transformer_classifier(hidden).squeeze(-1)  # (W,)
        return logits, hidden


class TemporalWindowCollator:
    """Group a sequence into overlapping windows."""

    def __init__(self, window_size: int = 8, stride: Optional[int] = None):
        self.T = window_size
        self.stride = stride if stride is not None else window_size // 2
        self.last_starts: List[int] = []

    def __call__(
        self, features: torch.Tensor, timestamps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N, d = features.shape
        T, S = self.T, self.stride

        starts = list(range(0, max(N - T + 1, 1), S))
        # Ensure the final posts are covered by at least one window.
        last_start = max(N - T, 0)
        if starts[-1] != last_start:
            starts.append(last_start)
        self.last_starts = starts
        W = len(starts)

        windows = torch.zeros(W, T, d, dtype=features.dtype, device=features.device)
        ts_windows = torch.zeros(W, T, dtype=timestamps.dtype, device=timestamps.device)
        pad_mask = torch.ones(W, T, dtype=torch.bool, device=features.device)

        for wi, start in enumerate(starts):
            end = min(start + T, N)
            size = end - start
            windows[wi, :size] = features[start:end]
            ts_windows[wi, :size] = timestamps[start:end]
            pad_mask[wi, :size] = False

        return windows, ts_windows, pad_mask


def run_temporal_sequence(
    aggregator: TemporalAggregator,
    windows: torch.Tensor,
    ts_windows: torch.Tensor,
    pad_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    W = windows.shape[0]
    probs, tevents = [], []
    L_prev = M_prev = None

    for t in range(W):
        feats = windows[t].unsqueeze(0)
        ts = ts_windows[t].unsqueeze(0)
        mask = pad_mask[t]

        valid = (~mask).float().unsqueeze(-1)
        feats = feats * valid.unsqueeze(0)

        p, T_event, L_prev, M_prev = aggregator(feats, ts, L_prev, M_prev, pad_mask=mask.unsqueeze(0))
        probs.append(p)
        tevents.append(T_event)

    all_probs = torch.cat(probs, dim=0)
    all_tevents = torch.cat(tevents, dim=0)

    seq_hidden: List[torch.Tensor] = []
    if aggregator.use_transformer and W >= 2:
        win_ts = ts_windows.mean(dim=-1)
        tr_logits, hidden = aggregator.encode_sequence(all_tevents, win_ts)
        if hidden is not None:
            all_probs = tr_logits
            seq_hidden = [hidden[i] for i in range(W)]

    return all_probs, all_tevents, seq_hidden

