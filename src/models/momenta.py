"""
src/models/momenta.py
MOMENTA model — multimodal encoder and fusion (text + image).

Includes:
- Mixture-of-Experts per modality (MoE)
- Bidirectional co-attention
- Gated fusion
- Semantic inconsistency head
- Cross-modal discrepancy branch (gated residual)
- Domain-adversarial head (GRL)
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ════════════════════════════════════════════════════════════════════════════
# ─── Gradient Reversal Layer (for Domain Adversarial Training) ───────────────

class _GRL(torch.autograd.Function):
    """Gradient reversal: forward is identity, backward negates and scales."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return -ctx.alpha * grad, None


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha  # updated externally each epoch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GRL.apply(x, self.alpha)


# ════════════════════════════════════════════════════════════════════════════
# ─── Building blocks ────────────────────────────────────────────────────────

class ExpertFFN(nn.Module):
    """Single expert: 2-layer FFN with GELU activation."""

    def __init__(self, d: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, expansion * d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * d, d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MoELayer(nn.Module):
    """
    Mixture-of-Experts for one modality.

      M = Σ_k  α_k · E_k(x)
      α = softmax( W_gate · x )
    """

    def __init__(self, d: int, K: int = 4, dropout: float = 0.1):
        super().__init__()
        self.experts = nn.ModuleList([ExpertFFN(d, dropout=dropout) for _ in range(K)])
        self.gate = nn.Linear(d, K)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = torch.softmax(self.gate(x), dim=-1)  # (B, K)
        stack = torch.stack([e(x) for e in self.experts], dim=1)  # (B, K, d)
        return (alpha.unsqueeze(-1) * stack).sum(dim=1)  # (B, d)


# ════════════════════════════════════════════════════════════════════════════
# ─── Main MOMENTA model ─────────────────────────────────────────────────────

class MOMENTA(nn.Module):
    """
    Multimodal encoder + fusion + auxiliary heads.

    forward() returns:
      P_i     : (B, d)  fused post representation
      A_i     : (B,)    cosine alignment score
      p_match : (B,)    inconsistency logit
      T_proj  : (B, d)  projected text embedding  (for contrastive)
      I_proj  : (B, d)  projected image embedding (for contrastive)
      domain_logits : (B, n_domains)
    """

    def __init__(
        self,
        d: int = 256,
        K: int = 4,
        n_heads: int = 4,
        bert_name: str = "xlm-roberta-large",
        clip_name: str = "openai/clip-vit-large-patch14",
        bert_dim: int = 1024,
        clip_dim: int = 768,
        freeze_backbones: bool = False,
        dropout: float = 0.3,
        n_domains: int = 4,
    ):
        super().__init__()
        assert d % n_heads == 0, f"d ({d}) must be divisible by n_heads ({n_heads})"
        self.d = d

        from transformers import AutoModel, CLIPModel

        self.bert = AutoModel.from_pretrained(bert_name)
        self.clip = CLIPModel.from_pretrained(clip_name)

        if freeze_backbones:
            for p in self.bert.parameters():
                p.requires_grad = False
            for p in self.clip.parameters():
                p.requires_grad = False

        # Projections + LN
        self.W_text = nn.Linear(bert_dim, d)
        self.W_img = nn.Linear(clip_dim, d)
        self.ln_T_proj = nn.LayerNorm(d)
        self.ln_I_proj = nn.LayerNorm(d)

        # Intra-modal MHSA
        self.mhsa_text = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.mhsa_img = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ln_mhsa_t = nn.LayerNorm(d)
        self.ln_mhsa_i = nn.LayerNorm(d)

        # MoE
        self.moe_text = MoELayer(d, K, dropout=dropout)
        self.moe_img = MoELayer(d, K, dropout=dropout)

        # Co-attention (shared projections)
        self.W_Q = nn.Linear(d, d, bias=False)
        self.W_K = nn.Linear(d, d, bias=False)
        self.W_V = nn.Linear(d, d, bias=False)
        self.ln_co_t = nn.LayerNorm(d)
        self.ln_co_i = nn.LayerNorm(d)

        # Fusion gate
        self.W_g = nn.Linear(2 * d, d)

        # Inconsistency head
        self.match_head = nn.Linear(2 * d, 1)

        # Discrepancy branch
        self.W_disc = nn.Linear(2 * d, d)
        self.ln_disc = nn.LayerNorm(d)
        self.disc_gate = nn.Parameter(torch.zeros(1))  # scalar

        # Domain adversarial
        self.grl = GradientReversalLayer(alpha=0.0)
        self.domain_classifier = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d // 2, n_domains),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # ── Text: encode + project ───────────────────────────────────────────
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        T_seq = self.ln_T_proj(self.W_text(bert_out))  # (B, L, d)
        T_proj = T_seq[:, 0]  # CLS (B, d)

        # ── Image: encode + project ──────────────────────────────────────────
        clip_out = self.clip.get_image_features(pixel_values=pixel_values)
        if isinstance(clip_out, torch.Tensor):
            clip_feat = clip_out
        elif hasattr(clip_out, "image_embeds"):
            clip_feat = clip_out.image_embeds
        else:
            clip_feat = clip_out.pooler_output
        I_proj = self.ln_I_proj(self.W_img(clip_feat))  # (B, d)

        # ── Intra-modal MHSA ────────────────────────────────────────────────
        pad_mask = attention_mask == 0
        T_attn, _ = self.mhsa_text(T_seq, T_seq, T_seq, key_padding_mask=pad_mask)
        T_tilde = self.ln_mhsa_t(T_seq + T_attn)[:, 0]  # CLS

        I_seq = I_proj.unsqueeze(1)
        I_attn, _ = self.mhsa_img(I_seq, I_seq, I_seq)
        I_tilde = self.ln_mhsa_i(I_seq + I_attn).squeeze(1)

        # ── MoE ─────────────────────────────────────────────────────────────
        M_text = self.moe_text(T_tilde)
        M_img = self.moe_img(I_tilde)

        # ── Bidirectional co-attention ──────────────────────────────────────
        Q_t = self.W_Q(M_text).unsqueeze(1)
        K_i = self.W_K(M_img).unsqueeze(1)
        V_i = self.W_V(M_img).unsqueeze(1)
        score_ti = torch.bmm(Q_t, K_i.transpose(1, 2)) / math.sqrt(self.d)
        C_TI = torch.bmm(torch.softmax(score_ti, dim=-1), V_i).squeeze(1)

        Q_i = self.W_Q(M_img).unsqueeze(1)
        K_t = self.W_K(M_text).unsqueeze(1)
        V_t = self.W_V(M_text).unsqueeze(1)
        score_it = torch.bmm(Q_i, K_t.transpose(1, 2)) / math.sqrt(self.d)
        C_IT = torch.bmm(torch.softmax(score_it, dim=-1), V_t).squeeze(1)

        M_prime_text = self.ln_co_t(M_text + C_TI)
        M_prime_img = self.ln_co_i(M_img + C_IT)

        # ── Fusion + alignment ──────────────────────────────────────────────
        concat = torch.cat([M_prime_text, M_prime_img], dim=-1)
        g_i = torch.sigmoid(self.W_g(concat))
        P_i = g_i * M_prime_text + (1.0 - g_i) * M_prime_img
        A_i = F.cosine_similarity(M_prime_text, M_prime_img, dim=-1)

        # ── Discrepancy gated residual ──────────────────────────────────────
        disc_abs = torch.abs(T_proj - I_proj)
        disc_prod = T_proj * I_proj
        disc_feat = torch.cat([disc_abs, disc_prod], dim=-1)
        disc_out = self.ln_disc(self.W_disc(disc_feat))
        gate = torch.tanh(self.disc_gate)
        P_i = P_i + gate * disc_out

        # ── Inconsistency head ──────────────────────────────────────────────
        p_match = self.match_head(concat).squeeze(-1)

        # ── Domain adversarial ──────────────────────────────────────────────
        domain_logits = self.domain_classifier(self.grl(P_i))

        return P_i, A_i, p_match, T_proj, I_proj, domain_logits

