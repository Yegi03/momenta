#!/usr/bin/env python3
"""
t-SNE visualization of MOMENTA's embedding space (main project layout).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]  # main/
sys.path.insert(0, str(ROOT))

from src.data.dataset import build_dataloaders
from src.models.momenta import MOMENTA

DATASET_NAMES = ["fakeddit", "mmcovar", "weibo", "xfacta"]
DATASET_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#ff7f00"]
CLASS_COLORS = ["#2166ac", "#d73027"]
CLASS_LABELS = ["Real", "Fake"]


def extract_embeddings(checkpoint: str, device: torch.device, max_per_ds: int = 300):
    ckpt = torch.load(checkpoint, map_location=device)

    bert_name = "xlm-roberta-large"
    clip_name = "openai/clip-vit-large-patch14"

    momenta_model = MOMENTA(bert_name=bert_name, clip_name=clip_name, d=256, K=4, n_heads=4, dropout=0.0, n_domains=4).to(device)
    momenta_model.load_state_dict(ckpt.get("momenta", {}), strict=False)
    momenta_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(bert_name)
    loaders = build_dataloaders(dataset_names=DATASET_NAMES, tokenizer=tokenizer, batch_size=32, num_workers=2, max_text_len=128)

    all_emb, all_labels, all_domains = [], [], []
    with torch.no_grad():
        for ds_idx, ds in enumerate(DATASET_NAMES):
            test_loader = loaders[ds]["test"]
            count = 0
            for batch in test_loader:
                if count >= max_per_ds:
                    break
                input_ids = batch["input_ids"].to(device)
                attn_mask = batch["attention_mask"].to(device)
                pixels = batch["pixel_values"].to(device)
                labels = batch["label"]

                P_i, *_ = momenta_model(input_ids, attn_mask, pixels)
                emb = F.normalize(P_i.float(), dim=-1).cpu().numpy()

                all_emb.append(emb)
                all_labels.append(labels.numpy())
                all_domains.append(np.full(len(labels), ds_idx, dtype=np.int32))
                count += len(labels)
                if count >= max_per_ds:
                    break

    embeddings = np.concatenate(all_emb, axis=0)
    labels_arr = np.concatenate(all_labels, axis=0)
    domains_arr = np.concatenate(all_domains, axis=0)
    return embeddings, labels_arr, domains_arr


def plot_tsne(embeddings, labels, domains, out_dir: Path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plots.")
        return

    tsne = TSNE(n_components=2, perplexity=40, n_iter=1000, random_state=42, init="pca", learning_rate="auto")
    coords = tsne.fit_transform(embeddings)
    x, y = coords[:, 0], coords[:, 1]

    fig, ax = plt.subplots(figsize=(8, 6))
    for ds_idx, (ds_name, color) in enumerate(zip(DATASET_NAMES, DATASET_COLORS)):
        mask = domains == ds_idx
        ax.scatter(x[mask], y[mask], c=color, s=8, alpha=0.6, label=ds_name)
    ax.legend(markerscale=3, fontsize=10)
    ax.set_title("MOMENTA Feature Space — colored by Dataset")
    ax.axis("off")
    fig.savefig(out_dir / "tsne_by_domain.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    for cls_val, (cls_name, color) in enumerate(zip(CLASS_LABELS, CLASS_COLORS)):
        mask = labels == cls_val
        ax.scatter(x[mask], y[mask], c=color, s=8, alpha=0.6, label=cls_name)
    ax.legend(markerscale=3, fontsize=10)
    ax.set_title("MOMENTA Feature Space — colored by Class")
    ax.axis("off")
    fig.savefig(out_dir / "tsne_by_class.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=str(ROOT / "checkpoints" / "best_main.pt"))
    p.add_argument("--out_dir", type=str, default=str(ROOT / "results"))
    p.add_argument("--max_per_ds", type=int, default=400)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    embeddings, labels, domains = extract_embeddings(args.checkpoint, device, max_per_ds=args.max_per_ds)
    plot_tsne(embeddings, labels, domains, out_dir)
    print(f"[viz_tsne] saved plots to {out_dir}")


if __name__ == "__main__":
    main()

