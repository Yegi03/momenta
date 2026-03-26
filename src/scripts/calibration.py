#!/usr/bin/env python3
"""
MC Dropout calibration analysis (main project layout).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]  # main/
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer

from src.data.dataset import build_dataloaders
from src.models.momenta import MOMENTA
from src.models.temporal import TemporalAggregator, TemporalWindowCollator


def mc_predict(momenta_model, temporal_agg, collator, loader, device, n_passes: int = 30, max_samples: Optional[int] = None):
    momenta_model.train()
    temporal_agg.train()

    all_conf, all_unc, all_labels = [], [], []
    n_done = 0
    with torch.no_grad():
        for batch in loader:
            if max_samples is not None and n_done >= max_samples:
                break
            n_done += 1
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            pixels = batch["pixel_values"].to(device)
            labels = batch["label"].cpu().numpy()
            ts = batch["timestamp"].to(device)

            pass_probs = []
            for _ in range(n_passes):
                P_i, *_ = momenta_model(input_ids, attn_mask, pixels)
                order = ts.argsort()
                P_sorted = P_i[order]
                ts_sort = ts[order]
                windows, ts_w, pad_mask = collator(P_sorted, ts_sort)
                W = windows.shape[0]

                win_logits = torch.zeros(W, device=device)
                L_prev = M_prev = None
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

                prob = torch.sigmoid(win_logits[-1]).item()
                prob = max(1e-6, min(1 - 1e-6, prob))
                pass_probs.append(prob)

            probs_arr = np.array(pass_probs)
            all_conf.append(float(probs_arr.mean()))
            all_unc.append(float(probs_arr.std()))
            all_labels.append(int(labels[-1]) if len(labels) else 0)

    return np.array(all_conf), np.array(all_unc), np.array(all_labels)


def compute_ece(confidences, labels, n_bins: int = 10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = len(confidences)
    bin_data = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            bin_data.append((0.5 * (lo + hi), 0.0, 0))
            continue
        acc = (labels[mask] == (confidences[mask] >= 0.5).astype(int)).mean()
        avg_c = confidences[mask].mean()
        n_bin = mask.sum()
        ece += (n_bin / n_total) * abs(acc - avg_c)
        bin_data.append((avg_c, acc, n_bin))
    return float(ece), bin_data


def plot_reliability(bin_data, ece: float, ds_name: str, out_path: Path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    avgs = [b[0] for b in bin_data if b[2] > 0]
    accs = [b[1] for b in bin_data if b[2] > 0]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.plot(avgs, accs, marker="o", color="#377eb8")
    ax.set_title(f"Reliability Diagram — {ds_name} (ECE={ece:.3f})")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--datasets", nargs="+", default=["fakeddit", "mmcovar", "weibo", "xfacta"])
    p.add_argument("--mc_passes", type=int, default=30)
    p.add_argument("--max_test_samples", type=int, default=400)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)

    bert_name = "xlm-roberta-large"
    clip_name = "openai/clip-vit-large-patch14"

    momenta_model = MOMENTA(bert_name=bert_name, clip_name=clip_name, d=256, K=4, n_heads=4, dropout=0.2, n_domains=4).to(device)
    momenta_model.load_state_dict(ckpt.get("momenta", {}), strict=False)

    temporal_agg = TemporalAggregator(d=256, use_transformer=False).to(device)
    temporal_agg.load_state_dict(ckpt.get("temporal_agg", {}), strict=False)

    tokenizer = AutoTokenizer.from_pretrained(bert_name)
    loaders = build_dataloaders(dataset_names=args.datasets, tokenizer=tokenizer, batch_size=1, num_workers=2, max_text_len=128)
    collator = TemporalWindowCollator(window_size=8, stride=4)

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    for ds in args.datasets:
        conf, unc, labels = mc_predict(
            momenta_model, temporal_agg, collator, loaders[ds]["test"], device, n_passes=args.mc_passes, max_samples=args.max_test_samples
        )
        ece, bin_data = compute_ece(conf, labels, n_bins=10)
        print(f"[{ds}] ECE={ece:.4f}  mean_conf={conf.mean():.3f}  mean_unc={unc.mean():.3f}  n={len(conf)}")
        plot_reliability(bin_data, ece, ds, out_dir / f"calibration_{ds}.png")


if __name__ == "__main__":
    main()

