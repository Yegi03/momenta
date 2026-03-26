#!/usr/bin/env python3
"""
Training entrypoint for MOMENTA (main project layout).

Run from `main/`:
  python -m src.scripts.train --datasets fakeddit mmcovar weibo xfacta --equalize --batches_per_ds 150
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

ROOT = Path(__file__).resolve().parents[2]  # main/
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer

from src.data.dataset import build_dataloaders
from src.models.momenta import MOMENTA
from src.models.temporal import TemporalAggregator, TemporalWindowCollator
from src.models.loss import MOMENTALoss, PrototypeMemoryBank, compute_class_weights


def parse_args():
    p = argparse.ArgumentParser(description="Train MOMENTA (main)")

    # Data
    p.add_argument("--datasets", nargs="+", default=["fakeddit", "mmcovar", "weibo", "xfacta"])
    p.add_argument(
        "--fakeddit_train_frac",
        type=float,
        default=None,
        help="Use this fraction of Fakeddit train only (e.g. 0.1). Val/test unchanged. Seed 42.",
    )
    p.add_argument("--max_text_len", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--num_workers", type=int, default=4)

    # Model
    p.add_argument("--d", type=int, default=256)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--window_size", type=int, default=8)
    p.add_argument("--window_stride", type=int, default=4)
    p.add_argument("--freeze_backbones", action="store_true")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--bert_name", type=str, default="")
    p.add_argument("--clip_name", type=str, default="")
    p.add_argument("--model_cache", type=str, default=str(ROOT / "deployment_cache" / "models"))

    # Training
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--backbone_lr", type=float, default=3e-6)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--amp", action="store_true")

    # Loss weights / options
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--focal_gamma", type=float, default=2.5)
    p.add_argument("--lambda_align", type=float, default=0.05)
    p.add_argument("--lambda_tc", type=float, default=0.01)
    p.add_argument("--lambda_match", type=float, default=0.1)
    p.add_argument("--lambda_contrast", type=float, default=0.05)
    p.add_argument("--lambda_reg", type=float, default=0.0)
    p.add_argument("--lambda_domain", type=float, default=0.03)
    p.add_argument("--lambda_rdrop", type=float, default=0.5)
    p.add_argument("--lambda_proto", type=float, default=0.05)
    p.add_argument("--lambda_proto_memory", type=float, default=0.10)
    # R-Drop is enabled by default; use --no_rdrop to disable.
    p.add_argument("--rdrop", action="store_true", default=True)
    p.add_argument("--no_rdrop", dest="rdrop", action="store_false")
    p.add_argument("--tau", type=float, default=0.2)
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument("--use_transformer", action="store_true", default=True)
    p.add_argument("--no_transformer", dest="use_transformer", action="store_false")
    p.add_argument("--lambda_tc_lstm", type=float, default=0.01)

    # Balanced sampling
    p.add_argument("--equalize", action="store_true")
    p.add_argument("--batches_per_ds", type=int, default=0)

    # Fine-tuning
    p.add_argument("--resume_from", type=str, default="")
    p.add_argument("--run_tag", type=str, default="main")
    p.add_argument("--n_domains", type=int, default=4)

    # Output
    p.add_argument("--out_dir", type=str, default=str(ROOT / "checkpoints"))
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_every", type=int, default=2)

    return p.parse_args()


def setup_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_metrics(preds: List[int], labels: List[int], scores: Optional[List[float]] = None) -> Dict:
    n = len(preds)
    if n == 0:
        return dict(acc=0, prec=0, recall=0, f1=0, macro_f1=0, spec=0, auc=0.5, mcc=0)

    tp = fp = fn = tn = 0
    for pr, y in zip(preds, labels):
        if pr == 1 and y == 1:
            tp += 1
        elif pr == 1 and y == 0:
            fp += 1
        elif pr == 0 and y == 1:
            fn += 1
        else:
            tn += 1

    acc = (tp + tn) / n
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * prec * recall / (prec + recall) if (prec + recall) else 0.0
    prec0 = tn / (tn + fn) if (tn + fn) else 0.0
    rec0 = tn / (tn + fp) if (tn + fp) else 0.0
    f1_0 = 2 * prec0 * rec0 / (prec0 + rec0) if (prec0 + rec0) else 0.0
    macro_f1 = (f1 + f1_0) / 2.0
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom else 0.0

    auc = 0.5
    if scores is not None and len(set(labels)) > 1:
        try:
            from sklearn.metrics import roc_auc_score

            auc = float(roc_auc_score(labels, scores))
        except Exception:
            pass

    return dict(acc=acc, prec=prec, recall=recall, f1=f1, macro_f1=macro_f1, spec=spec, auc=auc, mcc=mcc)


def fmt_metrics(m: Dict) -> str:
    return (
        f"acc={m['acc']:.3f}  f1={m['f1']:.3f}  macro_f1={m['macro_f1']:.3f}  "
        f"auc={m['auc']:.3f}  mcc={m['mcc']:.3f}  prec={m['prec']:.3f}  rec={m['recall']:.3f}"
    )


def make_scheduler(optimizer, warmup_steps: int, total_steps: int, eta_min_ratio: float = 0.05) -> LambdaLR:
    def _lr(step: int):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return eta_min_ratio + (1.0 - eta_min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, _lr)


def grl_alpha(epoch: int, total_epochs: int) -> float:
    p = min(epoch / max(total_epochs // 2, 1), 1.0)
    return 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0


def build_models(args, device: torch.device, bert_name: str, clip_name: str):
    momenta = MOMENTA(
        d=args.d,
        K=args.K,
        n_heads=args.n_heads,
        bert_name=bert_name,
        clip_name=clip_name,
        freeze_backbones=args.freeze_backbones,
        dropout=args.dropout,
        n_domains=args.n_domains,
    ).to(device)
    temporal_agg = TemporalAggregator(
        d=args.d,
        use_transformer=args.use_transformer,
    ).to(device)
    collator = TemporalWindowCollator(window_size=args.window_size, stride=args.window_stride)
    return momenta, temporal_agg, collator


def build_optimizer(args, momenta, temporal_agg, total_steps: int):
    backbone_names = {"bert", "clip"}

    def is_backbone(name: str) -> bool:
        return any(name.startswith(b + ".") or ("." + b + ".") in name for b in backbone_names)

    def no_decay(name: str) -> bool:
        return "bias" in name or "norm" in name.lower()

    bb_wd, bb_nd, new_wd, new_nd = [], [], [], []
    for name, p in (list(momenta.named_parameters()) + list(temporal_agg.named_parameters())):
        if not p.requires_grad:
            continue
        if is_backbone(name):
            (bb_nd if no_decay(name) else bb_wd).append(p)
        else:
            (new_nd if no_decay(name) else new_wd).append(p)

    optimizer = AdamW(
        [
            {"params": bb_wd, "lr": args.backbone_lr, "weight_decay": args.weight_decay},
            {"params": bb_nd, "lr": args.backbone_lr, "weight_decay": 0.0},
            {"params": new_wd, "lr": args.lr, "weight_decay": args.weight_decay},
            {"params": new_nd, "lr": args.lr, "weight_decay": 0.0},
        ]
    )
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = make_scheduler(optimizer, warmup_steps, total_steps)
    return optimizer, scheduler, new_wd


def _aggregate_windows_to_posts(
    win_logits: torch.Tensor,
    spans: List[Tuple[int, int]],
    labels: torch.Tensor,
    order: torch.Tensor,
    threshold: float = 0.5,
) -> Tuple[List[int], List[int], List[float]]:
    """
    Aggregate overlapping window outputs into post-level outputs.
    - score: mean sigmoid probability across windows containing a post
    - class: majority vote across window-level binary decisions
    """
    N = int(labels.numel())
    if N == 0:
        return [], [], []

    probs = torch.sigmoid(win_logits).detach().float()
    post_probs: List[List[float]] = [[] for _ in range(N)]
    post_preds: List[List[int]] = [[] for _ in range(N)]

    for w, (start, end) in enumerate(spans):
        p = float(probs[w].item())
        pred = int(p >= threshold)
        for idx in range(start, end):
            post_probs[idx].append(p)
            post_preds[idx].append(pred)

    # Fallback for any uncovered post (should be rare after tail window fix)
    if spans:
        nearest_prob = float(probs[-1].item())
        nearest_pred = int(nearest_prob >= threshold)
    else:
        nearest_prob = 0.5
        nearest_pred = int(nearest_prob >= threshold)

    score_sorted = torch.empty(N, dtype=torch.float32, device=labels.device)
    pred_sorted = torch.empty(N, dtype=torch.long, device=labels.device)
    for i in range(N):
        if post_probs[i]:
            mean_prob = sum(post_probs[i]) / len(post_probs[i])
            vote = sum(post_preds[i]) / len(post_preds[i])
            score_sorted[i] = mean_prob
            pred_sorted[i] = int(vote >= 0.5)
        else:
            score_sorted[i] = nearest_prob
            pred_sorted[i] = nearest_pred

    # Restore original (unsorted) batch order.
    inv_order = torch.empty_like(order)
    inv_order[order] = torch.arange(order.numel(), device=order.device)
    score_orig = score_sorted[inv_order].tolist()
    pred_orig = pred_sorted[inv_order].tolist()
    labels_orig = labels.long().tolist()
    return pred_orig, labels_orig, score_orig


def forward_batch(
    batch: Dict,
    momenta: MOMENTA,
    temporal_agg: TemporalAggregator,
    collator: TemporalWindowCollator,
    loss_fn: MOMENTALoss,
    class_weights: torch.Tensor,
    device: torch.device,
    weight_params: List,
    use_amp: bool = False,
    use_rdrop: bool = False,
    proto_memory: Optional[PrototypeMemoryBank] = None,
) -> Tuple[torch.Tensor, Dict, List[int], List[int], List[float], float]:
    input_ids = batch["input_ids"].to(device)
    attn_mask = batch["attention_mask"].to(device)
    pixels = batch["pixel_values"].to(device)
    labels = batch["label"].to(device)
    y_match = batch["y_match"].to(device)
    timestamps = batch["timestamp"].to(device)
    domain_labels = batch["dataset_id"].to(device)

    batch_cw = class_weights.to(device)

    def _run_once():
        with autocast("cuda", enabled=use_amp):
            P_i, A_i, p_match, T_proj, I_proj, dom_logits = momenta(input_ids, attn_mask, pixels)

            order = timestamps.argsort()
            P_sorted = P_i[order]
            ts_sorted = timestamps[order]
            lab_sorted = labels[order]

            windows, ts_w, pad_mask = collator(P_sorted, ts_sorted)
            W = windows.shape[0]
            win_labels = torch.zeros(W, dtype=torch.long, device=device)
            win_logits = torch.zeros(W, device=device)
            L_t_list: List[torch.Tensor] = []
            t_event_list: List[torch.Tensor] = []
            L_prev = M_prev = None

            spans: List[Tuple[int, int]] = []
            for w in range(W):
                valid = (~pad_mask[w]).float().unsqueeze(-1)
                feat_w = (windows[w] * valid).unsqueeze(0)
                ts_w_ = ts_w[w].unsqueeze(0)
                p_w, t_event_w, L_t_w, M_t_w = temporal_agg(
                    feat_w,
                    ts_w_,
                    L_prev,
                    M_prev,
                    pad_mask=pad_mask[w].unsqueeze(0),
                )
                win_logits[w] = p_w.squeeze(0)
                L_t_list.append(L_t_w.squeeze(0))
                t_event_list.append(t_event_w.squeeze(0))
                L_prev, M_prev = L_t_w, M_t_w
                start = collator.last_starts[w] if hasattr(collator, "last_starts") else (collator.stride * w)
                end = min(start + collator.T, len(P_sorted))
                win_labels[w] = lab_sorted[end - 1]
                spans.append((start, end))

            seq_hidden: List[torch.Tensor] = []
            if getattr(temporal_agg, "use_transformer", False) and W >= 2:
                tevents_stacked = torch.stack(t_event_list, dim=0)
                valid_ts = (~pad_mask).float()
                win_ts = (ts_w * valid_ts).sum(dim=-1) / valid_ts.sum(dim=-1).clamp(min=1.0)
                tr_logits, tr_hidden = temporal_agg.encode_sequence(tevents_stacked, win_ts)
                if tr_logits is not None:
                    win_logits = tr_logits
                if tr_hidden is not None:
                    seq_hidden = [tr_hidden[i] for i in range(W)]

        return (
            win_logits,
            win_labels,
            P_i,
            A_i,
            p_match,
            T_proj,
            I_proj,
            L_t_list,
            dom_logits,
            seq_hidden,
            spans,
            order,
            labels,
        )

    def _safe(t: torch.Tensor) -> torch.Tensor:
        return t.float().nan_to_num(nan=0.0, posinf=20.0, neginf=-20.0)

    (
        win_logits,
        win_labels,
        P_i,
        A_i,
        p_match,
        T_proj,
        I_proj,
        L_t_list,
        dom_logits,
        seq_hidden,
        spans,
        order,
        labels_orig_tensor,
    ) = _run_once()
    win_logits2 = None
    if use_rdrop:
        win_logits2_raw, *_ = _run_once()
        win_logits2 = _safe(win_logits2_raw)

    total_loss, terms = loss_fn(
        p=_safe(win_logits),
        y=win_labels,
        A_i=_safe(A_i),
        p_match=_safe(p_match),
        y_match=y_match,
        T_proj=_safe(T_proj),
        I_proj=_safe(I_proj),
        L_t_list=[_safe(t) for t in L_t_list],
        class_weights=batch_cw,
        model_params=weight_params,
        p2=win_logits2,
        domain_logits=_safe(dom_logits),
        domain_labels=domain_labels,
        embeddings=_safe(P_i),
        post_labels=labels,
        lstm_hidden_states=[_safe(h) for h in seq_hidden] if seq_hidden else None,
        proto_memory=proto_memory,
    )

    post_preds, post_labels, post_scores = _aggregate_windows_to_posts(
        win_logits=win_logits,
        spans=spans,
        labels=labels_orig_tensor,
        order=order,
        threshold=0.5,
    )
    uncertainty = float(win_logits.std()) if win_logits.numel() > 1 else 0.0
    return total_loss, terms, post_preds, post_labels, post_scores, uncertainty


@torch.no_grad()
def evaluate(
    loader,
    momenta,
    temporal_agg,
    collator,
    loss_fn,
    class_weights,
    device,
    weight_params,
    use_amp: bool = False,
    threshold: float = 0.5,
    return_raw: bool = False,
):
    momenta.eval()
    temporal_agg.eval()

    all_preds, all_labels, all_scores = [], [], []
    total_loss, n_batches = 0.0, 0

    for batch in loader:
        loss, _, preds, labels, scores, _ = forward_batch(
            batch,
            momenta,
            temporal_agg,
            collator,
            loss_fn,
            class_weights,
            device,
            weight_params,
            use_amp=use_amp,
            use_rdrop=False,
        )
        total_loss += loss.item()
        all_preds += preds
        all_labels += labels
        all_scores += scores
        n_batches += 1

    all_scores = [0.5 if (s != s or s == float("inf") or s == float("-inf")) else s for s in all_scores]
    if threshold != 0.5:
        all_preds = [1 if s > threshold else 0 for s in all_scores]

    m = compute_metrics(all_preds, all_labels, all_scores)
    m["loss"] = total_loss / max(n_batches, 1)
    m["threshold"] = threshold
    if return_raw:
        return m, all_scores, all_labels
    return m


class InfiniteLoader:
    def __init__(self, loader):
        self.loader = loader
        self._iter = iter(loader)

    def __next__(self):
        try:
            return next(self._iter)
        except StopIteration:
            self._iter = iter(self.loader)
            return next(self._iter)


def make_equalized_epoch(loaders_dict: Dict, train_datasets: List[str], batches_per_ds: int):
    sizes = [len(loaders_dict[ds]["train"]) for ds in train_datasets]
    if batches_per_ds <= 0:
        bpd = int(math.exp(sum(math.log(max(s, 1)) for s in sizes) / len(sizes)))
    else:
        bpd = batches_per_ds
    bpd = max(bpd, 50)
    print(f"[equalize] batches_per_ds={bpd}  total_batches_per_epoch={bpd * len(train_datasets)}")

    inf = {ds: InfiniteLoader(loaders_dict[ds]["train"]) for ds in train_datasets}
    for _ in range(bpd):
        for ds in train_datasets:
            yield next(inf[ds])


def run_training(args, train_datasets, eval_datasets, loaders, class_weights, device, bert_name, clip_name, out_dir: Path):
    loss_fn = MOMENTALoss(
        lambda_align=args.lambda_align,
        lambda_TC=args.lambda_tc,
        lambda_match=args.lambda_match,
        lambda_contrast=args.lambda_contrast,
        lambda_reg=args.lambda_reg,
        lambda_domain=args.lambda_domain,
        lambda_rdrop=args.lambda_rdrop if args.rdrop else 0.0,
        lambda_proto=args.lambda_proto,
        lambda_proto_memory=args.lambda_proto_memory,
        lambda_tc_lstm=args.lambda_tc_lstm,
        tau=args.tau,
        gamma=args.gamma,
        label_smoothing=args.label_smoothing,
        focal_gamma=args.focal_gamma,
    )

    momenta, temporal_agg, collator = build_models(args, device, bert_name, clip_name)

    if args.resume_from:
        ckpt_path = Path(args.resume_from)
        if not ckpt_path.is_absolute():
            ckpt_path = out_dir / ckpt_path
        if ckpt_path.exists():
            ckpt = torch.load(str(ckpt_path), map_location=device)
            momenta.load_state_dict(ckpt.get("momenta", {}), strict=False)
            temporal_agg.load_state_dict(ckpt.get("temporal_agg", {}), strict=False)
            print(f"[resume] loaded {ckpt_path}")

    if args.equalize:
        total_batches = max(args.batches_per_ds, 50) * len(train_datasets)
    else:
        total_batches = sum(len(loaders[ds]["train"]) for ds in train_datasets)

    total_steps = (total_batches * args.epochs) // max(args.grad_accum, 1)
    optimizer, scheduler, weight_params = build_optimizer(args, momenta, temporal_agg, total_steps)

    use_amp = args.amp and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp) if use_amp else GradScaler(enabled=False)

    proto_memory = PrototypeMemoryBank(n_datasets=args.n_domains, n_classes=2, dim=args.d, momentum=0.99).to(device)

    best_avg_f1 = 0.0
    best_metrics = {}
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        momenta.train()
        temporal_agg.train()
        momenta.grl.alpha = grl_alpha(epoch, args.epochs)

        epoch_preds, epoch_labels, epoch_scores = [], [], []
        epoch_loss = 0.0
        batch_idx = 0
        t0 = time.time()
        optimizer.zero_grad()

        if args.equalize:
            batch_source = make_equalized_epoch(loaders, train_datasets, args.batches_per_ds)
        else:
            iters = [iter(loaders[ds]["train"]) for ds in train_datasets]
            def _drain_all():
                active = list(iters)
                while active:
                    nxt = []
                    for it in active:
                        try:
                            yield next(it)
                            nxt.append(it)
                        except StopIteration:
                            pass
                    active = nxt
            batch_source = _drain_all()

        for batch in batch_source:
            loss, terms, preds, lbls, scores, _unc = forward_batch(
                batch,
                momenta,
                temporal_agg,
                collator,
                loss_fn,
                class_weights,
                device,
                weight_params,
                use_amp=use_amp,
                use_rdrop=args.rdrop,
                proto_memory=proto_memory,
            )

            scaler.scale(loss / args.grad_accum).backward()
            epoch_loss += loss.item()
            epoch_preds += preds
            epoch_labels += lbls
            epoch_scores += scores
            batch_idx += 1

            if batch_idx % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                all_p = list(momenta.parameters()) + list(temporal_agg.parameters())
                torch.nn.utils.clip_grad_norm_(all_p, args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            if batch_idx % args.log_every == 0:
                m = compute_metrics(epoch_preds[-args.log_every :], epoch_labels[-args.log_every :], epoch_scores[-args.log_every :])
                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"  Ep{epoch} step{batch_idx} loss={terms['L_total']:.3f} "
                    f"acc={m['acc']:.3f} macro_f1={m['macro_f1']:.3f} auc={m['auc']:.3f} "
                    f"grl_alpha={momenta.grl.alpha:.2f} lr={lr_now:.2e} {time.time()-t0:.0f}s",
                    flush=True,
                )

        train_m = compute_metrics(epoch_preds, epoch_labels, epoch_scores)
        train_m["loss"] = epoch_loss / max(batch_idx, 1)
        print(f"\nEpoch {epoch}/{args.epochs} train: {fmt_metrics(train_m)}")

        val_f1s = []
        for ds in eval_datasets:
            vm = evaluate(loaders[ds]["val"], momenta, temporal_agg, collator, loss_fn, class_weights, device, weight_params)
            val_f1s.append(vm["macro_f1"])
            print(f"  [{ds}/val] {fmt_metrics(vm)}")

        avg_val_f1 = sum(val_f1s) / len(val_f1s) if val_f1s else 0.0
        print(f"  [avg_val_macro_f1={avg_val_f1:.4f} patience={no_improve}/{args.patience}]", flush=True)

        if avg_val_f1 > best_avg_f1:
            best_avg_f1 = avg_val_f1
            best_metrics = {"epoch": epoch, "avg_val_macro_f1": avg_val_f1}
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "run_tag": args.run_tag,
                    "momenta": momenta.state_dict(),
                    "temporal_agg": temporal_agg.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "metrics": best_metrics,
                    "args": vars(args),
                },
                out_dir / f"best_{args.run_tag}.pt",
            )
            print(f"  *** New best avg macro_F1: {best_avg_f1:.4f} → saved best_{args.run_tag}.pt")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"  Early stop: no improvement for {args.patience} epochs.")
                break

    return best_metrics


def main():
    args = parse_args()
    device = setup_device()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve backbone names (prefer local cache paths if present)
    cache = Path(args.model_cache)
    xlmr_path = cache / "xlm-roberta-large"
    clip_path = cache / "clip-vit-l14"
    bert_name = str(xlmr_path) if xlmr_path.exists() and args.bert_name == "" else (args.bert_name or "xlm-roberta-large")
    clip_name = str(clip_path) if clip_path.exists() and args.clip_name == "" else (args.clip_name or "openai/clip-vit-large-patch14")

    tokenizer = AutoTokenizer.from_pretrained(bert_name)

    loaders = build_dataloaders(
        dataset_names=args.datasets,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_text_len=args.max_text_len,
        num_workers=args.num_workers,
        fakeddit_train_frac=getattr(args, "fakeddit_train_frac", None),
    )

    all_train_labels = []
    for ds in args.datasets:
        all_train_labels.extend(r["label"] for r in loaders[ds]["train"].dataset._records)
    class_weights = compute_class_weights(torch.tensor(all_train_labels, dtype=torch.long))
    print(f"[class_weights] {class_weights.tolist()}")

    run_training(
        args=args,
        train_datasets=args.datasets,
        eval_datasets=args.datasets,
        loaders=loaders,
        class_weights=class_weights,
        device=device,
        bert_name=bert_name,
        clip_name=clip_name,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()

