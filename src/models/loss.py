"""
src/models/loss.py
MOMENTA total loss + EMA Prototype Memory Bank.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeMemoryBank(nn.Module):
    """
    EMA Prototype Memory Bank for cross-dataset class alignment.
    Stores prototypes for (dataset_id, class) and computes global class prototypes.
    """

    def __init__(self, n_datasets: int, n_classes: int = 2, dim: int = 256, momentum: float = 0.99):
        super().__init__()
        self.n_datasets = n_datasets
        self.n_classes = n_classes
        self.momentum = momentum
        self.register_buffer("prototypes", torch.zeros(n_datasets, n_classes, dim))
        self.register_buffer("initialized", torch.zeros(n_datasets, n_classes, dtype=torch.bool))

    @torch.no_grad()
    def update(self, embeddings: torch.Tensor, labels: torch.Tensor, dataset_ids: torch.Tensor) -> None:
        emb_norm = F.normalize(embeddings.float(), dim=-1)
        for ds in dataset_ids.unique():
            ds_mask = dataset_ids == ds
            for cls in range(self.n_classes):
                cls_mask = ds_mask & (labels.long() == cls)
                if cls_mask.sum() < 1:
                    continue
                batch_proto = emb_norm[cls_mask].mean(0)
                ds_i = int(ds.item())
                if not self.initialized[ds_i, cls]:
                    self.prototypes[ds_i, cls] = batch_proto
                    self.initialized[ds_i, cls] = True
                else:
                    self.prototypes[ds_i, cls] = (
                        self.momentum * self.prototypes[ds_i, cls] + (1.0 - self.momentum) * batch_proto
                    )

    def global_prototypes(self) -> torch.Tensor:
        protos = F.normalize(self.prototypes, dim=-1)
        init = self.initialized.float()
        numer = (protos * init.unsqueeze(-1)).sum(0)
        denom = init.sum(0).clamp(min=1.0).unsqueeze(-1)
        return F.normalize(numer / denom, dim=-1)

    def n_initialized(self) -> int:
        return int(self.initialized.sum().item())

    def is_ready(self) -> bool:
        return self.n_initialized() >= 2


class MOMENTALoss(nn.Module):
    def __init__(
        self,
        lambda_align: float = 0.1,
        lambda_TC: float = 0.1,
        lambda_match: float = 0.1,
        lambda_contrast: float = 0.1,
        lambda_reg: float = 1e-4,
        lambda_domain: float = 0.1,
        lambda_rdrop: float = 0.5,
        lambda_proto: float = 0.05,
        lambda_proto_memory: float = 0.10,
        lambda_tc_lstm: float = 0.01,
        tau: float = 0.07,
        gamma: float = 0.9,
        label_smoothing: float = 0.05,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.lambda_align = lambda_align
        self.lambda_TC = lambda_TC
        self.lambda_match = lambda_match
        self.lambda_contrast = lambda_contrast
        self.lambda_reg = lambda_reg
        self.lambda_domain = lambda_domain
        self.lambda_rdrop = lambda_rdrop
        self.lambda_proto = lambda_proto
        self.lambda_proto_memory = lambda_proto_memory
        self.lambda_tc_lstm = lambda_tc_lstm
        self.tau = tau
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.focal_gamma = focal_gamma

    def forward(
        self,
        p: torch.Tensor,
        y: torch.Tensor,
        A_i: torch.Tensor,
        p_match: torch.Tensor,
        y_match: torch.Tensor,
        T_proj: torch.Tensor,
        I_proj: torch.Tensor,
        L_t_list: List[torch.Tensor],
        class_weights: Optional[torch.Tensor] = None,
        model_params: Optional[List[torch.Tensor]] = None,
        p2: Optional[torch.Tensor] = None,
        domain_logits: Optional[torch.Tensor] = None,
        domain_labels: Optional[torch.Tensor] = None,
        embeddings: Optional[torch.Tensor] = None,
        post_labels: Optional[torch.Tensor] = None,
        lstm_hidden_states: Optional[List[torch.Tensor]] = None,
        proto_memory: Optional[PrototypeMemoryBank] = None,
    ):
        y = y.float()
        y_match = y_match.float()

        L_CE = self._l_ce(p, y, class_weights)
        L_align = self._l_align(A_i)
        L_TC = self._l_tc(L_t_list, ref=p)
        L_match = self._l_match(p_match, y_match)
        L_contrast = self._l_contrast(T_proj, I_proj)

        L_reg = self._l_reg(model_params, ref=p) if self.lambda_reg > 0.0 else p.new_zeros(())
        L_rdrop = self._l_rdrop(p, p2) if (p2 is not None and self.lambda_rdrop > 0.0) else p.new_zeros(())
        L_domain = (
            self._l_domain(domain_logits, domain_labels)
            if (domain_logits is not None and domain_labels is not None and self.lambda_domain > 0.0)
            else p.new_zeros(())
        )

        L_proto = (
            self._l_proto(embeddings, post_labels)
            if (embeddings is not None and post_labels is not None and self.lambda_proto > 0.0)
            else p.new_zeros(())
        )

        if (
            proto_memory is not None
            and embeddings is not None
            and post_labels is not None
            and domain_labels is not None
        ):
            proto_memory.update(embeddings.detach(), post_labels.detach(), domain_labels.detach())

        L_proto_mem = (
            self._l_proto_memory(embeddings, post_labels, proto_memory)
            if (
                proto_memory is not None
                and embeddings is not None
                and post_labels is not None
                and self.lambda_proto_memory > 0.0
                and proto_memory.is_ready()
            )
            else p.new_zeros(())
        )

        L_tc_lstm = (
            self._l_tc_sequence(lstm_hidden_states)
            if (lstm_hidden_states is not None and len(lstm_hidden_states) >= 2 and self.lambda_tc_lstm > 0.0)
            else p.new_zeros(())
        )

        total = (
            L_CE
            + self.lambda_align * L_align
            + self.lambda_TC * L_TC
            + self.lambda_match * L_match
            + self.lambda_contrast * L_contrast
            + self.lambda_reg * L_reg
            + self.lambda_rdrop * L_rdrop
            + self.lambda_domain * L_domain
            + self.lambda_proto * L_proto
            + self.lambda_proto_memory * L_proto_mem
            + self.lambda_tc_lstm * L_tc_lstm
        )

        terms = {
            "L_CE": L_CE.item(),
            "L_align": L_align.item(),
            "L_TC": L_TC.item(),
            "L_match": L_match.item(),
            "L_contrast": L_contrast.item(),
            "L_reg": L_reg.item(),
            "L_rdrop": L_rdrop.item(),
            "L_domain": L_domain.item(),
            "L_proto": L_proto.item(),
            "L_proto_mem": L_proto_mem.item(),
            "L_tc_lstm": L_tc_lstm.item(),
            "L_total": total.item(),
        }
        return total, terms

    def _l_ce(self, p: torch.Tensor, y: torch.Tensor, class_weights: Optional[torch.Tensor]) -> torch.Tensor:
        y_hard = y.long()
        if self.label_smoothing > 0.0:
            y_soft = y * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        else:
            y_soft = y.float()

        bce = F.binary_cross_entropy_with_logits(p.float(), y_soft.float(), reduction="none")
        if self.focal_gamma > 0.0:
            # Focal factor on positive-class probability: (1 - p_pos)^{gamma_foc}.
            p_pos = torch.sigmoid(p.float())
            focal_w = (1.0 - p_pos) ** self.focal_gamma
            bce = focal_w * bce
        if class_weights is not None:
            bce = bce * class_weights[y_hard].float()
        return bce.sum()

    def _l_align(self, A_i: torch.Tensor) -> torch.Tensor:
        return (1.0 - A_i).sum()

    def _l_tc(self, L_t_list: List[torch.Tensor], ref: Optional[torch.Tensor] = None) -> torch.Tensor:
        _ref = L_t_list[0] if L_t_list else ref
        if len(L_t_list) < 2:
            return _ref.new_zeros(()) if _ref is not None else torch.zeros(())
        tc = L_t_list[0].new_zeros(())
        for t in range(1, len(L_t_list)):
            diff = L_t_list[t] - L_t_list[t - 1]
            sq_norm = (diff**2).sum(dim=-1).mean()
            weight = self.gamma**t
            tc = tc + weight * sq_norm
        return tc

    def _l_match(self, p_match: torch.Tensor, y_match: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(p_match.float(), y_match.float(), reduction="sum")

    def _l_contrast(self, T_proj: torch.Tensor, I_proj: torch.Tensor) -> torch.Tensor:
        T_n = F.normalize(T_proj.float(), dim=-1)
        I_n = F.normalize(I_proj.float(), dim=-1)
        sim = torch.mm(T_n, I_n.t()).clamp(-1.0, 1.0)
        logits = sim / self.tau
        B = T_proj.shape[0]
        targets = torch.arange(B, device=T_proj.device)

        beta = 1.0 / (self.tau * 2.0)
        mask_neg = ~torch.eye(B, dtype=torch.bool, device=T_proj.device)
        hard_w = torch.zeros_like(logits)
        hard_w[mask_neg] = torch.softmax((beta * sim[mask_neg]).view(B, B - 1), dim=-1).view(-1)
        logits = logits + 0.1 * hard_w

        l_ti = F.cross_entropy(logits, targets, reduction="sum")
        l_it = F.cross_entropy(logits.t(), targets, reduction="sum")
        return (l_ti + l_it) / 2.0

    def _l_proto(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        emb = F.normalize(embeddings.float(), dim=-1)
        loss = emb.new_zeros(())
        n_terms = 0
        for cls_val in (0, 1):
            mask = labels.long() == cls_val
            if mask.sum() < 2:
                continue
            cls_emb = emb[mask]
            centroid = cls_emb.mean(0, keepdim=True).detach()
            cos_sim = (cls_emb * centroid).sum(-1)
            loss = loss + (1.0 - cos_sim).mean()
            n_terms += 1
        return loss / max(n_terms, 1)

    def _l_proto_memory(
        self, embeddings: torch.Tensor, labels: torch.Tensor, memory_bank: PrototypeMemoryBank
    ) -> torch.Tensor:
        margin = 0.1
        emb = F.normalize(embeddings.float(), dim=-1)
        g_protos = memory_bank.global_prototypes().to(emb.device)
        loss = emb.new_zeros(())
        n_terms = 0
        for cls_val in (0, 1):
            mask = labels.long() == cls_val
            if mask.sum() == 0:
                continue
            cls_emb = emb[mask]
            pos_proto = g_protos[cls_val].unsqueeze(0)
            neg_proto = g_protos[1 - cls_val].unsqueeze(0)
            attract = (1.0 - (cls_emb * pos_proto).sum(-1)).mean()
            repel = (cls_emb * neg_proto).sum(-1).clamp(min=margin) - margin
            repel = repel[repel > 0].mean() if (repel > 0).any() else emb.new_zeros(())
            loss = loss + attract + repel
            n_terms += 1
        return loss / max(n_terms, 1)

    def _l_rdrop(self, logits1: torch.Tensor, logits2: torch.Tensor) -> torch.Tensor:
        if logits2 is None:
            return logits1.new_zeros(())
        p1 = torch.sigmoid(logits1.float()).clamp(1e-6, 1 - 1e-6)
        p2 = torch.sigmoid(logits2.float()).clamp(1e-6, 1 - 1e-6)
        kl12 = p1 * (p1.log() - p2.log()) + (1 - p1) * ((1 - p1).log() - (1 - p2).log())
        kl21 = p2 * (p2.log() - p1.log()) + (1 - p2) * ((1 - p2).log() - (1 - p1).log())
        return (kl12 + kl21).mean() / 2.0

    def _l_domain(self, domain_logits: torch.Tensor, domain_labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(domain_logits.float(), domain_labels)

    def _l_tc_sequence(self, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        if len(hidden_states) < 2:
            return hidden_states[0].new_zeros(())
        h_curr = torch.stack(hidden_states[1:], dim=0)
        h_prev = torch.stack(hidden_states[:-1], dim=0)
        diff_sq = (h_curr - h_prev).pow(2).sum(dim=-1)
        cos_sim = F.cosine_similarity(h_curr, h_prev, dim=-1).clamp(min=0.0)
        return (diff_sq * cos_sim).mean()

    def _l_reg(self, model_params: Optional[List[torch.Tensor]], ref: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not model_params:
            return ref.new_zeros(()) if ref is not None else torch.zeros(())
        return sum(p.norm(2) ** 2 for p in model_params)


def compute_class_weights(labels: torch.Tensor) -> torch.Tensor:
    n_total = float(len(labels))
    n_classes = 2.0
    weights = torch.zeros(2, dtype=torch.float32)
    for c in range(2):
        n_c = (labels == c).sum().float()
        weights[c] = n_total / (n_classes * n_c.clamp(min=1.0))
    return weights

