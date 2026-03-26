#!/usr/bin/env python3
"""
LOSO evaluation entrypoint (main project layout).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]  # main/
sys.path.insert(0, str(ROOT))

from src.data.dataset import build_dataloaders
from src.models.momenta import MOMENTA
from src.models.temporal import TemporalAggregator, TemporalWindowCollator

DATASETS = ["fakeddit", "mmcovar", "weibo", "xfacta"]
CKPT_PREFIX = "checkpoints/best_loso_held_"


def load_model(ckpt_path: str, device: torch.device, bert_name: str, clip_name: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    momenta_model = MOMENTA(bert_name=bert_name, clip_name=clip_name, d=256, K=4, n_heads=4, dropout=0.0, n_domains=4).to(device)
    momenta_model.load_state_dict(ckpt.get("momenta", {}), strict=False)
    momenta_model.eval()

    temporal_agg = TemporalAggregator(d=256, use_transformer=False).to(device)
    temporal_agg.load_state_dict(ckpt.get("temporal_agg", {}), strict=False)
    temporal_agg.eval()
    return momenta_model, temporal_agg


@torch.no_grad()
def run_inference(momenta_model, temporal_agg, collator, loader, device, threshold: float = 0.5):
    all_scores, all_preds, all_labels = [], [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        pixels = batch["pixel_values"].to(device)
        labels = batch["label"]
        ts = batch["timestamp"].to(device)

        P_i, *_ = momenta_model(input_ids, attn_mask, pixels)
        order = ts.argsort()
        P_sorted = P_i[order]
        ts_sort = ts[order]
        windows, ts_w, pad_mask = collator(P_sorted, ts_sort)
        W = windows.shape[0]

        L_prev = M_prev = None
        win_logits = torch.zeros(W, device=device)
        for w in range(W):
            valid = (~pad_mask[w]).float().unsqueeze(-1)
            feat_w = (windows[w] * valid).unsqueeze(0)
            ts_w_ = ts_w[w].unsqueeze(0)
            p_w, _, L_t, M_t = temporal_agg(
                feat_w,
                ts_w_,
                L_prev,
                M_prev,
                pad_mask=pad_mask[w].unsqueeze(0),
            )
            win_logits[w] = p_w.squeeze(0)
            L_prev, M_prev = L_t, M_t

        scores = torch.sigmoid(win_logits).tolist()
        preds = [int(s >= threshold) for s in scores]
        lbl = int(labels[0].item())

        all_scores.append(float(np.mean(scores)))
        all_preds.append(int(np.mean(preds) >= 0.5))
        all_labels.append(lbl)

    all_scores = [0.5 if (s != s or abs(s) == float("inf")) else s for s in all_scores]
    return all_scores, all_preds, all_labels


def find_best_threshold(momenta_model, temporal_agg, collator, val_loader, device):
    scores, _, labels = run_inference(momenta_model, temporal_agg, collator, val_loader, device)
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.3, 0.75, 0.02):
        preds = [int(s >= t) for s in scores]
        f1 = f1_score(labels, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def compute_metrics(scores, preds, labels):
    labels = np.array(labels)
    preds = np.array(preds)
    scores = np.array(scores)
    n_cls = len(np.unique(labels))
    auc = roc_auc_score(labels, scores) if n_cls > 1 else 0.5
    return {
        "acc": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, zero_division=0),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "auc": auc,
        "mcc": matthews_corrcoef(labels, preds),
        "prec": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
    }


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--bert_name", type=str, default="xlm-roberta-large")
    p.add_argument("--clip_name", type=str, default="openai/clip-vit-large-patch14")
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.bert_name)
    collator = TemporalWindowCollator(window_size=8, stride=4)
    loaders = build_dataloaders(dataset_names=DATASETS, tokenizer=tokenizer, batch_size=1, num_workers=2, max_text_len=128)

    rows = []
    for held_ds in DATASETS:
        ckpt_path = ROOT / f"{CKPT_PREFIX}{held_ds}.pt"
        if not ckpt_path.exists():
            print(f"[{held_ds}] checkpoint not found: {ckpt_path}")
            continue
        momenta_model, temporal_agg = load_model(str(ckpt_path), device, args.bert_name, args.clip_name)
        best_t, _ = find_best_threshold(momenta_model, temporal_agg, collator, loaders[held_ds]["val"], device)
        scores, preds, labels = run_inference(momenta_model, temporal_agg, collator, loaders[held_ds]["test"], device, threshold=best_t)
        m = compute_metrics(scores, preds, labels)
        rows.append(m)
        print(f"held_out={held_ds:<8}  macro_f1={m['macro_f1']:.3f}  f1={m['f1']:.3f}  auc={m['auc']:.3f}  acc={m['acc']:.3f}  thr={best_t:.2f}")

    if rows:
        avg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0].keys()}
        print(f"\nLOSO average: macro_f1={avg['macro_f1']:.3f}  f1={avg['f1']:.3f}  auc={avg['auc']:.3f}  acc={avg['acc']:.3f}")


if __name__ == "__main__":
    main()

