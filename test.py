# fast_bigbird_content_router.py
# BigBird-FastRouter: O(n) content-aware token selection for HF BigBird
# - Locals: straight top-k inside a sliding window (no MMR, no candidate gathers)
# - Globals: per-head shared top-1 per bucket using cheap salience (||K|| + boundary deltas)
# - Teleports: tiny uniform anchors along the sequence (quantile positions)
# - Preserves O(n); avoids large [BH,T,Kc,d] tensors; MPS-safe
#
# Usage:
#   python fast_bigbird_content_router.py
#
# Notes:
# - Patches BigBird's SelfAttention classes in-place (both dense and block-sparse variants).
# - Prints one-time token-selection stats (heads, k, g, t, M, softmax ratio).
# - Defaults tuned for IMDB-length (T≈896) on Mac MPS; adjust RouterConfig if needed.

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    TrainerCallback,
    set_seed,
)
from transformers.trainer_utils import EvalPrediction

# -------------------------------
# Import HF BigBird SelfAttention
# -------------------------------
try:
    from transformers.models.big_bird.modeling_big_bird import (
        BigBirdSelfAttention,
        BigBirdBlockSparseAttention,
    )
except Exception as e:
    raise ImportError(
        "Transformers with BigBird is required. Install: pip install transformers"
    ) from e

# -------------------------------
# Router Config
# -------------------------------
@dataclass
class RouterConfig:
    # Scope (encoder-only for BigBird)
    patch_encoder_only: bool = True

    # Window
    fragment_size: int = 128

    # Softmax density → k (locals)
    r_target_softmax: float = 0.16
    min_k: int = 48
    max_k: int = 64

    # Globals per head (shared across all queries in a layer)
    globals_per_head: int = 4

    # Teleports per head (shared)
    teleports_per_head: int = 2

    # Dense fallback
    dense_fallback_under: int = 512

    # Salience mixing (per-head shared selection)
    sal_a_keynorm: float = 1.0
    sal_b_boundary: float = 0.25

    # Positional prior for locals
    alpha_pos_prior: float = 0.10

    # Misc / Debug
    log_once_pairs: bool = True
    debug_collect: bool = False


router_cfg = RouterConfig(
    fragment_size=128,
    r_target_softmax=0.16,
    min_k=56,        # slightly higher local mass
    max_k=64,
    globals_per_head=4,
    teleports_per_head=2,
    dense_fallback_under=512,
    sal_a_keynorm=1.0,
    sal_b_boundary=0.25,
    alpha_pos_prior=0.12,
    log_once_pairs=True,
    debug_collect=False,
)

# -------------------------------
# Runtime (share globals across layers in a forward)
# -------------------------------
class RouterRuntime:
    def __init__(self, num_heads: int, cfg: RouterConfig):
        self.num_heads = num_heads
        self.cfg = cfg
        self._active = False
        self._globals_cache: Dict[int, List[torch.Tensor]] = {}
        self._teleports_cache: Dict[int, List[torch.Tensor]] = {}
        self._last_layer_src_len: Dict[int, int] = {}

    def begin_forward(self):
        self._active = True
        self._globals_cache.clear()
        self._teleports_cache.clear()
        self._last_layer_src_len.clear()

    def end_forward(self):
        self._active = False
        self._globals_cache.clear()
        self._teleports_cache.clear()
        self._last_layer_src_len.clear()

    def get_shared(self, layer_idx: Optional[int], src_len: int):
        if (not self._active) or (layer_idx is None):
            return None, None
        if self._last_layer_src_len.get(layer_idx, None) == src_len:
            return self._globals_cache.get(layer_idx), self._teleports_cache.get(layer_idx)
        return None, None

    def store_shared(self, layer_idx: Optional[int], src_len: int,
                     globals_per_head: List[torch.Tensor],
                     teleports_per_head: List[torch.Tensor]):
        if (not self._active) or (layer_idx is None):
            return
        self._globals_cache[layer_idx] = globals_per_head
        self._teleports_cache[layer_idx] = teleports_per_head
        self._last_layer_src_len[layer_idx] = src_len

# -------------------------------
# Helpers
# -------------------------------
def F_normalize_safe(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True).clamp_min(eps))

def positional_prior(indices: torch.Tensor, center_positions: torch.Tensor, tau: float = 8.0) -> torch.Tensor:
    dist = (indices - center_positions.unsqueeze(1)).abs().float()
    return torch.exp(-dist / max(tau, 1e-6))

def build_indices_encoder(seq_len: int, frag: int, device: torch.device) -> torch.Tensor:
    half = frag // 2
    t = torch.arange(seq_len, device=device)
    starts = torch.clamp(t - half, 0, max(0, seq_len - frag))
    return starts.unsqueeze(1) + torch.arange(frag, device=device).unsqueeze(0)   # [T,Fw]

def sliding_window_view_seq(x: torch.Tensor, Fw: int) -> torch.Tensor:
    # x: [BH, T, d] -> [BH, T, Fw, d]
    BH, T, D = x.shape
    left = Fw // 2
    right = Fw - 1 - left
    xpad = F.pad(x, (0, 0, left, right), mode="replicate")
    s0, sT, sD = xpad.stride()
    return xpad.as_strided(size=(BH, T, Fw, D), stride=(s0, sT, sT, sD))

def flat_batched_gather_kv(K_or_V: torch.Tensor, abs_idx: torch.Tensor) -> torch.Tensor:
    """
    K_or_V: [BH, Tk, d], abs_idx: [BH, Tq, M] -> [BH, Tq, M, d]
    """
    BH, Tk, d = K_or_V.shape
    _, Tq, M = abs_idx.shape
    base = (torch.arange(BH, device=K_or_V.device) * Tk).view(BH, 1, 1)
    flat_idx = (abs_idx + base).reshape(-1)
    flat = K_or_V.reshape(BH * Tk, d)
    out = flat.index_select(0, flat_idx)
    return out.view(BH, Tq, M, d)

# -------------------------------
# Globals selection: cheap salience + bucketed top-1
# -------------------------------
def compute_salience_per_head(K: torch.Tensor, H: int, B: int, T: int, d: int,
                              a: float, b: float) -> torch.Tensor:
    """
    K: [BH, T, d] → reshape to [H,B,T,d]
    s(h,t) = a * mean_b ||K[h,b,t,:]||_2 + b * mean_b ||(K[h,b,t,:] - K[h,b,t-1,:])||_2
    Returns s: [H,T]
    """
    KHB = K.view(H, B, T, d)
    kn = torch.linalg.vector_norm(KHB, dim=-1).mean(dim=1)  # [H,T]
    # boundary deltas (prepend first as previous)
    dK = KHB - torch.cat([KHB[:, :, :1, :], KHB[:, :, :-1, :]], dim=2)
    dn = torch.linalg.vector_norm(dK, dim=-1).mean(dim=1)   # [H,T]
    return a * kn + b * dn

def bucketed_top1_indices(s: torch.Tensor, g: int) -> List[torch.Tensor]:
    """
    s: [H,T] → pick exactly 1 index per ~equal bucket (g buckets).
    Returns a list of length H; each tensor is [g] Long indices.
    """
    H, T = s.shape
    if g <= 0:
        return [torch.empty(0, dtype=torch.long, device=s.device) for _ in range(H)]
    # Compute bucket boundaries (as evenly as possible)
    # For each bucket j, start = floor(j*T/g), end = floor((j+1)*T/g)
    idxs = []
    for h in range(H):
        picks = []
        for j in range(g):
            start = (j * T) // g
            end = ((j + 1) * T) // g
            if end <= start:
                continue
            seg = s[h, start:end]                      # [len]
            pos = torch.argmax(seg)                    # scalar
            picks.append(start + pos)
        if len(picks) == 0:
            idxs.append(torch.empty(0, dtype=torch.long, device=s.device))
        else:
            idxs.append(torch.stack(picks))
    return idxs  # list of [g]

def quantile_anchors(T: int, t: int, device: torch.device) -> torch.Tensor:
    """
    t evenly spaced anchors in [0, T-1], excluding endpoints if t>=2.
    Returns [t] Long.
    """
    if t <= 0:
        return torch.empty(0, dtype=torch.long, device=device)
    if t == 1:
        return torch.tensor([T // 2], dtype=torch.long, device=device)
    # Use open interval (0, T-1) to avoid duplicating CLS/EOS
    xs = torch.linspace(0, T - 1, steps=t + 2, device=device)[1:-1]
    return torch.round(xs).long()

# -------------------------------
# Fast content-aware SelfAttention for BigBird
# -------------------------------
class FastBigBirdSelfAttention(nn.Module):
    """
    Drop-in replacement for HF BigBirdSelfAttention:
      - keeps Q/K/V weights
      - local window top-k + shared globals + shared teleports + CLS/EOS
      - O(n) overall
    """
    def __init__(self, base_attn: BigBirdSelfAttention, cfg: RouterConfig, runtime: RouterRuntime):
        super().__init__()
        self.num_heads = base_attn.num_attention_heads
        self.head_dim = base_attn.attention_head_size
        self.all_head_size = self.num_heads * self.head_dim
        self.layer_idx = getattr(base_attn, "layer_idx", None)

        # Copy projections
        self.hidden_size = int(getattr(base_attn.query, "in_features", self.all_head_size))
        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key   = nn.Linear(self.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.hidden_size, self.all_head_size)
        self.query.load_state_dict(base_attn.query.state_dict())
        self.key.load_state_dict(base_attn.key.state_dict())
        self.value.load_state_dict(base_attn.value.state_dict())

        # Dropout
        if hasattr(base_attn, "dropout") and isinstance(base_attn.dropout, nn.Dropout):
            self.drop_p = float(base_attn.dropout.p)
        else:
            self.drop_p = 0.1
        self.dropout = nn.Dropout(self.drop_p)

        # Router machinery
        self.cfg = cfg
        self.runtime = runtime
        self._printed_cfg = False
        self._logged_pairs = False

        # cache for indices/prior
        self._idx_cache: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}

    def _transpose_for_scores(self, x: torch.Tensor, bsz: int, seq_len: int) -> torch.Tensor:
        return x.view(bsz, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()

    def _get_idx_and_prior(self, T: int, Fw: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (T, Fw)
        cached = self._idx_cache.get(key)
        if cached is not None:
            return cached
        idx_win = build_indices_encoder(T, Fw, device)  # [T,Fw]
        centers = torch.arange(T, device=device).long()
        prior = positional_prior(idx_win, centers, tau=max(Fw / 4, 1.0))  # [T,Fw]
        self._idx_cache[key] = (idx_win, prior)
        return idx_win, prior

    def _mask_to_window(self, attention_mask: torch.Tensor, bsz: int, T: int, Fw: int, idx_win: torch.Tensor) -> torch.Tensor:
        """
        attention_mask → boolean [B,T,T] allowed, then gather to [B,T,Fw]
        """
        src_len = T
        if attention_mask.dtype == torch.bool:
            if attention_mask.dim() == 2:           # [B,T]
                am_small = attention_mask.unsqueeze(1).expand(bsz, T, src_len)
            elif attention_mask.dim() == 4:         # [B,1,1,T]
                am_small = attention_mask.squeeze(1).expand(bsz, T, src_len)
            else:                                   # assume [B,T,T]
                am_small = attention_mask
        else:
            if attention_mask.dim() == 4:
                am_small = torch.isfinite(attention_mask).squeeze(1).expand(bsz, T, src_len)
            elif attention_mask.dim() == 2:
                am_small = (attention_mask > 0).unsqueeze(1).expand(bsz, T, src_len)
            else:
                am_small = torch.isfinite(attention_mask)
        idx_btf = idx_win.unsqueeze(0).expand(bsz, -1, -1)     # [B,T,Fw]
        mask_win = torch.gather(am_small, -1, idx_btf)         # [B,T,Fw]
        return mask_win

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs,
    ):
        bsz, T, _ = hidden_states.size()
        device = hidden_states.device

        # Projections
        Q = self._transpose_for_scores(self.query(hidden_states), bsz, T)  # [B,H,T,d]
        K = self._transpose_for_scores(self.key(hidden_states),   bsz, T)
        V = self._transpose_for_scores(self.value(hidden_states), bsz, T)

        Q = Q.reshape(bsz * self.num_heads, T, self.head_dim)   # [BH,T,d]
        K = K.reshape(bsz * self.num_heads, T, self.head_dim)
        V = V.reshape(bsz * self.num_heads, T, self.head_dim)
        BH = Q.shape[0]
        d = self.head_dim

        # Dense fallback (short sequences): exact normalized dot-product attention
        if T <= int(self.cfg.dense_fallback_under):
            qn = F_normalize_safe(Q, dim=-1)
            kn = F_normalize_safe(K, dim=-1)
            scores = torch.matmul(qn, kn.transpose(-1, -2))  # [BH,T,T]
            if attention_mask is not None:
                if attention_mask.dtype == torch.bool:
                    if attention_mask.dim() == 2:
                        am_small = attention_mask.unsqueeze(1).expand(bsz, T, T)
                    elif attention_mask.dim() == 4:
                        am_small = attention_mask.squeeze(1).expand(bsz, T, T)
                    else:
                        am_small = attention_mask
                else:
                    if attention_mask.dim() == 4:
                        am_small = torch.isfinite(attention_mask).squeeze(1).expand(bsz, T, T)
                    elif attention_mask.dim() == 2:
                        am_small = (attention_mask > 0).unsqueeze(1).expand(bsz, T, T)
                    else:
                        am_small = torch.isfinite(attention_mask)
                am_bool = am_small.unsqueeze(1).expand(bsz, self.num_heads, T, T).reshape(BH, T, T)
                scores = scores.masked_fill(~am_bool, torch.finfo(scores.dtype).min)
            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            ctx = torch.bmm(attn, V)  # [BH,T,d]
            ctx = ctx.view(bsz, self.num_heads, T, d).permute(0, 2, 1, 3).reshape(bsz, T, self.all_head_size)
            attn_w = attn.view(bsz, self.num_heads, T, T) if output_attentions else None
            return (ctx, attn_w)

        # --- Sparse content-aware path ---
        Fw = min(int(self.cfg.fragment_size), T)
        idx_win, prior_win = self._get_idx_and_prior(T, Fw, device)  # [T,Fw], [T,Fw]

        # Local window normalized scores
        with torch.no_grad():
            K_win = sliding_window_view_seq(K, Fw=Fw)                          # [BH,T,Fw,d]
            qn = F_normalize_safe(Q, dim=-1).unsqueeze(2)                      # [BH,T,1,d]
            kn = F_normalize_safe(K_win, dim=-1).transpose(-1, -2)             # [BH,T,d,Fw]
            scores_win = torch.matmul(qn, kn).squeeze(2)                        # [BH,T,Fw]

            # Blend positional prior
            scores_win = scores_win + float(self.cfg.alpha_pos_prior) * prior_win.unsqueeze(0)

            # Apply mask → window if provided
            if attention_mask is not None:
                mask_win = self._mask_to_window(attention_mask, bsz, T, Fw, idx_win)  # [B,T,Fw]
                mask_win_bh = mask_win.unsqueeze(1).expand(bsz, self.num_heads, T, Fw).reshape(BH, T, Fw)
                scores_win = scores_win.masked_fill(~mask_win_bh, torch.finfo(scores_win.dtype).min)

            # Decide locals budget k_here (same across queries for stability)
            g_cfg = int(self.cfg.globals_per_head)
            t_cfg = int(self.cfg.teleports_per_head)
            M_target = max(g_cfg + t_cfg + 2, int(np.ceil(self.cfg.r_target_softmax * T)))
            k_here = max(self.cfg.min_k, min(self.cfg.max_k, M_target - (g_cfg + t_cfg + 2)))
            k_here = min(k_here, Fw)

            if self.cfg.log_once_pairs and not self._printed_cfg:
                print(f"[FastRouter] heads={self.num_heads} Fw={Fw} k={k_here} "
                      f"g={g_cfg} t={t_cfg}  M~{k_here+g_cfg+t_cfg+2}")
                self._printed_cfg = True

            # Pure top-k inside the window (NO MMR)
            # sel_idx: [BH,T,k_here] are positions within the window [0..Fw-1]
            _, sel_pos = torch.topk(scores_win, k=k_here, dim=-1, largest=True, sorted=False)
            abs_idx_loc = torch.gather(idx_win.unsqueeze(0).expand(BH, -1, -1), -1, sel_pos)  # [BH,T,k_here]

            # ---- Shared globals/teleports per head (reused by all queries) ----
            H = self.num_heads
            B = bsz
            # Try to reuse between layers in one forward
            cached_globals, cached_teleports = self.runtime.get_shared(self.layer_idx, T)
            if cached_globals is None or cached_teleports is None:
                # Salience per head (cheap)
                s = compute_salience_per_head(
                    K, H, B, T, d, a=float(self.cfg.sal_a_keynorm), b=float(self.cfg.sal_b_boundary)
                )  # [H,T]
                globals_per_head = bucketed_top1_indices(s, g=g_cfg)           # list of H tensors [g]
                # Teleports: deterministic quantile anchors (distinct from CLS/EOS)
                teleports_per_head = [quantile_anchors(T, t_cfg, device=K.device) for _ in range(H)]
                self.runtime.store_shared(self.layer_idx, T, globals_per_head, teleports_per_head)
            else:
                globals_per_head = cached_globals
                teleports_per_head = cached_teleports

            # --- Always include CLS/EOS (per batch)
            if attention_mask is not None and attention_mask.dim() == 2:
                lens = attention_mask.long().sum(dim=1).clamp(min=1)  # [B]
                eos_idx_b = (lens - 1)
            else:
                eos_idx_b = torch.full((bsz,), T - 1, device=device, dtype=torch.long)
            cls_idx_b = torch.zeros(bsz, dtype=torch.long, device=device)

            cls_exp = cls_idx_b.view(bsz, 1).unsqueeze(1).expand(bsz, T, 1)  # [B,T,1]
            eos_exp = eos_idx_b.view(bsz, 1).unsqueeze(1).expand(bsz, T, 1)  # [B,T,1]

            # ---- Build final absolute indices per head: [B,T,M]
            abs_idx_list = []
            for h in range(H):
                # locals slice for this head's batch range:
                loc_h = abs_idx_loc[h * bsz:(h + 1) * bsz]                      # [B,T,k]
                # expand shared globals/teleports to [B,T,*]
                g = globals_per_head[h]
                g_exp = g.view(1, 1, -1).expand(bsz, T, -1)                     # [B,T,g]
                te = teleports_per_head[h]
                te_exp = te.view(1, 1, -1).expand(bsz, T, -1) if te.numel() > 0 else te.view(1,1,-1).expand(bsz,T,0)
                parts = [loc_h, g_exp, cls_exp, eos_exp]
                if te_exp.size(-1) > 0:
                    parts.append(te_exp)
                abs_idx_h = torch.cat(parts, dim=-1)                             # [B,T,M]
                abs_idx_list.append(abs_idx_h)

            # Stack heads into [BH,T,M] with equal M
            M = abs_idx_list[0].size(-1)
            for i in range(1, len(abs_idx_list)):
                if abs_idx_list[i].size(-1) != M:
                    diff = M - abs_idx_list[i].size(-1)
                    pad = abs_idx_list[i][..., :1].expand(-1, -1, diff)
                    abs_idx_list[i] = torch.cat([abs_idx_list[i], pad], dim=-1)
            abs_idx = torch.cat(abs_idx_list, dim=0)  # [BH,T,M]

            if self.cfg.log_once_pairs and not self._logged_pairs:
                M_now = abs_idx.size(-1)
                comps_sparse = BH * T * M_now
                comps_full = BH * T * T
                ratio = comps_sparse / max(1, comps_full)
                print(f"[softmax] sparse={comps_sparse:,} | full={comps_full:,} | ratio={ratio:.4f}")
                self._logged_pairs = True

        # Final attention over selected tokens
        K_sel = flat_batched_gather_kv(K, abs_idx)  # [BH,T,M,d]
        V_sel = flat_batched_gather_kv(V, abs_idx)  # [BH,T,M,d]
        scores_sel = (Q.unsqueeze(2) * K_sel).sum(-1)  # [BH,T,M]

        # Mask disallowed selected positions (if any)
        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                if attention_mask.dim() == 2:
                    am_small = attention_mask.unsqueeze(1).expand(bsz, T, T)
                elif attention_mask.dim() == 4:
                    am_small = attention_mask.squeeze(1).expand(bsz, T, T)
                else:
                    am_small = attention_mask
            else:
                if attention_mask.dim() == 4:
                    am_small = torch.isfinite(attention_mask).squeeze(1).expand(bsz, T, T)
                elif attention_mask.dim() == 2:
                    am_small = (attention_mask > 0).unsqueeze(1).expand(bsz, T, T)
                else:
                    am_small = torch.isfinite(attention_mask)
            abs_idx_hb = abs_idx.view(self.num_heads, bsz, T, -1)
            allowed_chunks = []
            for h in range(self.num_heads):
                allowed_h = torch.gather(am_small, -1, abs_idx_hb[h]).bool()  # [B,T,M]
                allowed_chunks.append(allowed_h)
            allowed = torch.cat(allowed_chunks, dim=0)  # [BH,T,M]
            scores_sel = scores_sel.masked_fill(~allowed, torch.finfo(scores_sel.dtype).min)

        attn_probs = F.softmax(scores_sel, dim=-1)    # [BH,T,M]
        attn_probs = self.dropout(attn_probs)

        BH_T = BH * T
        ctx = torch.bmm(attn_probs.reshape(BH_T, 1, -1), V_sel.reshape(BH_T, -1, d)).reshape(BH, T, d)
        ctx = ctx.view(bsz, self.num_heads, T, d).permute(0, 2, 1, 3).contiguous().view(bsz, T, self.all_head_size)

        attn_weights_reshaped = attn_probs.view(bsz, self.num_heads, T, -1) if output_attentions else None
        return (ctx, attn_weights_reshaped)

# Block-sparse variant wrapper (accepts the same args as HF; ignores band masks)
class FastBigBirdBlockSparseAttention(FastBigBirdSelfAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        band_mask: Optional[torch.Tensor] = None,
        from_mask: Optional[torch.Tensor] = None,
        to_mask: Optional[torch.Tensor] = None,
        from_blocked_mask: Optional[torch.Tensor] = None,
        to_blocked_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs,
    ):
        # derive a simple [B,T] mask if provided
        attention_mask = to_mask if to_mask is not None else from_mask
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            head_mask=None,
            output_attentions=output_attentions,
        )

# -------------------------------
# Patching utility
# -------------------------------
def patch_bigbird_with_fast_router(model: nn.Module, cfg: RouterConfig) -> RouterRuntime:
    # find num_heads
    num_heads = None
    for mod in model.modules():
        if isinstance(mod, (BigBirdSelfAttention, BigBirdBlockSparseAttention)):
            num_heads = getattr(mod, "num_attention_heads", None)
            break
    if num_heads is None:
        num_heads = 12

    runtime = RouterRuntime(num_heads=num_heads, cfg=cfg)

    # one forward = one begin/end
    def _begin_hook(_m, _in):
        runtime.begin_forward()
    def _end_hook(_m, _in, _out):
        runtime.end_forward()
    try:
        model.register_forward_pre_hook(_begin_hook)
        model.register_forward_hook(_end_hook)
    except Exception:
        pass

    # replace attentions
    def _recurse(module: nn.Module, layer_idx_start=[0]):
        for name, child in list(module.named_children()):
            if isinstance(child, BigBirdSelfAttention):
                setattr(child, "layer_idx", layer_idx_start[0]); layer_idx_start[0] += 1
                setattr(module, name, FastBigBirdSelfAttention(child, cfg, runtime))
            elif isinstance(child, BigBirdBlockSparseAttention):
                setattr(child, "layer_idx", layer_idx_start[0]); layer_idx_start[0] += 1
                setattr(module, name, FastBigBirdBlockSparseAttention(child, cfg, runtime))
            else:
                _recurse(child, layer_idx_start)

    _recurse(model)
    replaced_self = sum(1 for m in model.modules() if isinstance(m, FastBigBirdSelfAttention))
    replaced_block = sum(1 for m in model.modules() if isinstance(m, FastBigBirdBlockSparseAttention))
    total = replaced_self + replaced_block
    print(f"[patch] FastRouter installed: self={replaced_self}, block_sparse={replaced_block}, total={total}")
    return runtime

# -------------------------------
# Dataset / metrics (IMDB)
# -------------------------------
@dataclass
class TrainConfig:
    bigbird_name: str = "google/bigbird-roberta-base"
    seed: int = 42
    epochs: int = 3
    per_device_train_bs: int = 2
    per_device_eval_bs: int = 2
    grad_accum_steps: int = 8
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    max_length: int = 896
    train_samples: int = 2000
    eval_samples: int = 1000

train_cfg = TrainConfig()

def compute_metrics(eval_pred):
    if isinstance(eval_pred, EvalPrediction):
        preds, labels = eval_pred.predictions, eval_pred.label_ids
    else:
        preds, labels = eval_pred
    return {"accuracy": accuracy_score(labels, preds), "f1": f1_score(labels, preds)}

def build_imdb_dataset(tokenizer, fixed_length: int):
    ds = load_dataset("imdb")
    if train_cfg.train_samples:
        ds["train"] = ds["train"].shuffle(seed=train_cfg.seed).select(range(train_cfg.train_samples))
    if train_cfg.eval_samples:
        ds["test"] = ds["test"].shuffle(seed=train_cfg.seed).select(range(train_cfg.eval_samples))

    def tok_fn(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=fixed_length)

    ds = ds.map(tok_fn, batched=True, remove_columns=["text"])
    ds = ds.rename_column("label", "labels")
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return {"train": ds["train"], "validation": ds["test"]}

# -------------------------------
# Device flags / Training args
# -------------------------------
def device_flags():
    use_cuda = torch.cuda.is_available()
    use_mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    fp16 = False
    bf16 = False
    torch_compile = False
    if use_cuda:
        fp16 = True
        bf16 = torch.cuda.is_bf16_supported()
        torch_compile = False
    return fp16, bf16, torch_compile, use_mps

def make_args(out_dir: str) -> TrainingArguments:
    fp16, bf16, torch_compile, use_mps = device_flags()
    eval_accum = 1 if use_mps else 8
    return TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=train_cfg.epochs,
        per_device_train_batch_size=train_cfg.per_device_train_bs,
        per_device_eval_batch_size=train_cfg.per_device_eval_bs,
        gradient_accumulation_steps=train_cfg.grad_accum_steps,
        learning_rate=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
        warmup_ratio=train_cfg.warmup_ratio,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        logging_strategy="steps",
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
        fp16=fp16,
        bf16=bf16,
        remove_unused_columns=False,
        dataloader_num_workers=0,       # macOS best
        dataloader_pin_memory=False,    # macOS best
        gradient_checkpointing=False,   # macOS best
        torch_compile=torch_compile,
        optim="adamw_torch",
        eval_accumulation_steps=eval_accum,
    )

# -------------------------------
# Annealing callbacks (optional)
# -------------------------------
class RouterAnnealCallback(TrainerCallback):
    def __init__(self, root_model, start_frac=0.20, end_frac=0.15, warmup_steps=150,
                 start_max_k=80, end_max_k=64):
        self.root_model = root_model
        self.start_frac = float(start_frac)
        self.end_frac = float(end_frac)
        self.warmup_steps = int(warmup_steps)
        self.start_max_k = int(start_max_k)
        self.end_max_k = int(end_max_k)

    def _iters(self):
        for m in self.root_model.modules():
            if isinstance(m, FastBigBirdSelfAttention):
                yield m

    def on_step_begin(self, args, state, control, **kwargs):
        step = state.global_step
        if step <= self.warmup_steps:
            t = step / max(1, self.warmup_steps)
            frac = self.start_frac + t * (self.end_frac - self.start_frac)
            max_k = int(round(self.start_max_k + t * (self.end_max_k - self.start_max_k)))
            for m in self._iters():
                m.cfg.r_target_softmax = float(frac)
                m.cfg.max_k = int(max_k)

class FirstEpochRouterDensity(TrainerCallback):
    def __init__(self, root_model, first_r=0.18, first_max_k=72, rest_r=0.15, rest_max_k=64):
        self.root_model = root_model
        self.first_r = float(first_r); self.first_max_k = int(first_max_k)
        self.rest_r  = float(rest_r);  self.rest_max_k  = int(rest_max_k)
        self._epoch_idx = -1

    def _iters(self):
        for m in self.root_model.modules():
            if isinstance(m, FastBigBirdSelfAttention):
                yield m

    def _set(self, r, k):
        for m in self._iters():
            m.cfg.r_target_softmax = float(r)
            m.cfg.max_k = int(k)

    def on_train_begin(self, args, state, control, **kwargs):
        self._set(self.first_r, self.first_max_k)

    def on_epoch_begin(self, args, state, control, **kwargs):
        self._epoch_idx += 1
        if self._epoch_idx == 0:
            self._set(self.first_r, self.first_max_k)
        else:
            self._set(self.rest_r, self.rest_max_k)

    def on_evaluate(self, args, state, control, **kwargs):
        self._set(self.rest_r, self.rest_max_k)

# Simple EMA for stability (device/dtype-safe)
class EMACallback(TrainerCallback):
    def __init__(self, model, decay=0.999):
        self.model = model; self.decay = float(decay)
        self.shadow = {}; self._backup = {}

    @torch.no_grad()
    def on_train_begin(self, args, state, control, **kwargs):
        self.shadow.clear()
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.detach().clone().to(device=p.device, dtype=p.dtype)

    @torch.no_grad()
    def on_step_end(self, args, state, control, **kwargs):
        for n, p in self.model.named_parameters():
            if not p.requires_grad: continue
            s = self.shadow.get(n)
            if (s is None) or (s.shape != p.data.shape) or (s.device != p.device) or (s.dtype != p.data.dtype):
                s = p.detach().clone().to(device=p.device, dtype=p.data.dtype)
                self.shadow[n] = s
            s.mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    @torch.no_grad()
    def _swap_to_shadow(self):
        self._backup = {}
        for n, p in self.model.named_parameters():
            if not p.requires_grad: continue
            self._backup[n] = p.data.detach().clone()
            p.data.copy_(self.shadow[n])

    @torch.no_grad()
    def _swap_back(self):
        for n, p in self.model.named_parameters():
            if not p.requires_grad: continue
            p.data.copy_(self._backup[n])

    def on_evaluate(self, args, state, control, **kwargs):
        self._swap_to_shadow()

    def on_evaluate_end(self, args, state, control, **kwargs):
        self._swap_back()

    def on_save(self, args, state, control, **kwargs):
        if not self._backup:
            self._swap_to_shadow()

    def on_save_end(self, args, state, control, **kwargs):
        if self._backup:
            self._swap_back()

# -------------------------------
# Training / evaluation
# -------------------------------
def load_bigbird_tok(model_name: str):
    try:
        from transformers import BigBirdTokenizer
        return BigBirdTokenizer.from_pretrained(model_name)
    except Exception as e_slow:
        print(f"[BigBird] slow tokenizer failed: {e_slow}\nTrying fast tokenizer...", flush=True)
        return AutoTokenizer.from_pretrained(model_name, use_fast=True)

def summarize_token_selection_configs(cfg: RouterConfig, bigbird_model_name: str):
    print("\n=== Token-Selection Configs ===")
    print(
        "BigBird-FastRouter (ours): "
        f"Fw={cfg.fragment_size}, k∈[{cfg.min_k},{cfg.max_k}] (target={cfg.r_target_softmax:.2f}·T), "
        f"globals/head={cfg.globals_per_head}, teleports/head={cfg.teleports_per_head}"
    )
    bb_cfg = AutoConfig.from_pretrained(bigbird_model_name)
    bb_attn_type = getattr(bb_cfg, "attention_type", None)
    bb_block = getattr(bb_cfg, "block_size", None)
    bb_rand = getattr(bb_cfg, "num_random_blocks", None)
    bb_heads = getattr(bb_cfg, "num_attention_heads", None)
    print(
        "BigBird (HF): "
        f"attention_type={bb_attn_type}, block_size={bb_block}, random_blocks={bb_rand}, heads={bb_heads}, "
        "globals=via attention_mask"
    )
    print("================================\n", flush=True)

def train_and_eval_bigbird_fast(tokenizer, ds):
    model = AutoModelForSequenceClassification.from_pretrained(
        train_cfg.bigbird_name,
        num_labels=2,
    )
    runtime = patch_bigbird_with_fast_router(model, router_cfg)

    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=64)
    args = make_args("./bigbird-fast-out")

    router_epoch0 = FirstEpochRouterDensity(model, first_r=0.18, first_max_k=72, rest_r=0.15, rest_max_k=64)
    router_within = RouterAnnealCallback(model, start_frac=0.20, end_frac=0.15, warmup_steps=150,
                                         start_max_k=80, end_max_k=64)
    ema_cb = EMACallback(model)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=lambda logits, labels: (logits[0] if isinstance(logits, (tuple, list)) else logits).argmax(dim=-1),
        callbacks=[router_epoch0, router_within, ema_cb],
    )

    print("Training BigBird-FastRouter ...", flush=True)
    train_res = trainer.train()
    print(train_res.metrics)

    print("Evaluating BigBird-FastRouter ...", flush=True)
    eval_res = trainer.evaluate()
    print(eval_res)
    return eval_res

def train_and_eval_bigbird_baseline(tokenizer, ds):
    model = AutoModelForSequenceClassification.from_pretrained(
        train_cfg.bigbird_name,
        num_labels=2,
    )
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=64)
    args = make_args("./bigbird-baseline-out")

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=lambda logits, labels: (logits[0] if isinstance(logits, (tuple, list)) else logits).argmax(dim=-1),
    )

    print("Training BigBird baseline (HF block-sparse) ...", flush=True)
    train_res = trainer.train()
    print(train_res.metrics)

    print("Evaluating BigBird baseline ...", flush=True)
    eval_res = trainer.evaluate()
    print(eval_res)
    return eval_res

def main():
    set_seed(train_cfg.seed)

    summarize_token_selection_configs(router_cfg, train_cfg.bigbird_name)

    fixed_len = (train_cfg.max_length + 63) // 64 * 64
    tok = load_bigbird_tok(train_cfg.bigbird_name)
    ds = build_imdb_dataset(tok, fixed_length=fixed_len)

    ca = train_and_eval_bigbird_fast(tok, ds)
    base = train_and_eval_bigbird_baseline(tok, ds)

    print("\n==== Summary ====")
    print("[BigBird-FastRouter] ", ca)
    print("[BigBird baseline]   ", base)

if __name__ == "__main__":
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    main()
