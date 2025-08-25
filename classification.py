# classification.py
# BiggerBird (BART patch + training) with MPS-safe and speed optimizations:
#   - Dense fallback for short seqs
#   - Locals: blocked-MMR (topK* prefilter + 1–2 diversity steps)
#   - No huge expands: flat batched gather for K/V, sliding-window view for locals
#   - Window index/prior tiny LRU cache
#   - Globals/teleports adaptive to length (O(n))
#   - MPS-friendly eval: logits -> argmax on device + eval_accumulation_steps=1 on MPS
#   - Startup print: side-by-side token-selection configs (BiggerBird vs BigBird)

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
import os
from transformers.modeling_outputs import SequenceClassifierOutput

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
# Router / Model patch
# -------------------------------

try:
    from transformers.models.bart.modeling_bart import BartAttention
except Exception as e:
    raise ImportError("Transformers with BART is required. Install: pip install transformers") from e


@dataclass
class RouterConfig:
    # Scope
    patch_encoder_only: bool = True

    # Candidate construction
    fragment_size: int = 64      # window size Fw — clamped to src_len
    k_per_query: int = 24         # locals picked from the window
    globals_per_head: int = 6    # g per head

    # Softmax target fraction (controls k on short sequences)
    r_target_softmax: float = 0.12
    min_k: int = 48
    max_k: int = 48

    # Teleports
    teleports_per_head: int = 3
    teleport_bias_frac: float = 0.4

    # Utility shaping for globals (facility-location proxy)
    w_mean: float = 1.0
    w_max: float = 0.6
    w_topk: float = 0.4
    w_std: float  = 0.2
    topk_frac: float = 0.2
    keynorm_exponent: float = 0.0

    # Scoring blend / locals
    alpha_pos_prior: float = 0.15
    gamma_diversity: float = 0.25   # diversity penalty

    # Globals (prefilter + prototypes)
    top_u: int = 16                 # per-head prefilter size U
    proto_count: int = 24          # query prototypes p (<= Tq)

    # Blocked-MMR params
    mmr_prefilter_mult: float = 3   # candidates = min(Fw, ceil(mult*k))
    mmr_diversity_steps: int = 7      # number of diversity rounds (≈MMR)

    # Amortization
    share_stride_layers: int = 2      # reuse globals across adjacent layers

    # Dense fallback threshold
    dense_fallback_under: int = 512   # if src_len <= this → use dense attention (super().forward)

    # Misc / Debug
    random_selection: bool = False
    debug_collect: bool = False
    log_once_pairs: bool = True

default_router_config = RouterConfig()

router_cfg = RouterConfig(
    fragment_size=128,          # slightly tighter window → cleaner local top-k
    r_target_softmax=0.16,      # ensures k hits max_k at 896 tokens
    min_k=56,
    max_k=64,                   # locals per query (main quality driver)
    globals_per_head=6,
    teleports_per_head=4,       # a tiny bump helps long-range without much cost
    teleport_bias_frac=0.75,

    top_u=32,
    proto_count=48,

    mmr_prefilter_mult=3.0,
    mmr_diversity_steps=2,      # ↓ from 7 → less over-diversification, higher precision
    gamma_diversity=0.16,       # moderate penalty works best with steps=2

    alpha_pos_prior=0.12,       # restore a useful locality bias for IMDB
    share_stride_layers=2,

    dense_fallback_under=512,
    random_selection=False,
    debug_collect=False,
    log_once_pairs=True,
)





class RouterRuntime:
    def __init__(self, num_heads: int, cfg: RouterConfig):
        self.num_heads = num_heads
        self.cfg = cfg
        self._active = False
        self._globals_cache: Dict[int, List[torch.Tensor]] = {}
        self._last_layer_src_len: Dict[int, int] = {}

    def begin_forward(self):
        self._active = True
        self._globals_cache.clear()
        self._last_layer_src_len.clear()

    def end_forward(self):
        self._active = False
        self._globals_cache.clear()
        self._last_layer_src_len.clear()

    def maybe_get_shared_globals(self, layer_idx: Optional[int], src_len: int) -> Optional[List[torch.Tensor]]:
        if (not self._active) or (layer_idx is None):
            return None
        s = max(1, int(self.cfg.share_stride_layers))
        if s <= 1:
            return None
        if layer_idx % s == 1:
            prev = layer_idx - 1
            if prev in self._globals_cache and self._last_layer_src_len.get(prev, -1) == src_len:
                return self._globals_cache[prev]
        return None

    def store_globals(self, layer_idx: Optional[int], src_len: int, globals_per_head: List[torch.Tensor]):
        if (not self._active) or (layer_idx is None):
            return
        self._globals_cache[layer_idx] = globals_per_head
        self._last_layer_src_len[layer_idx] = src_len


# --------- small helpers ---------
def F_normalize_safe(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True).clamp_min(eps))

def positional_prior(indices: torch.Tensor, center_positions: torch.Tensor, tau: float = 8.0) -> torch.Tensor:
    dist = (indices - center_positions.unsqueeze(1)).abs().float()
    return torch.exp(-dist / max(tau, 1e-6))

def build_indices_encoder(seq_len: int, frag: int, device: torch.device) -> torch.Tensor:
    half = frag // 2
    t = torch.arange(seq_len, device=device)
    starts = torch.clamp(t - half, 0, max(0, seq_len - frag))
    return starts.unsqueeze(1) + torch.arange(frag, device=device).unsqueeze(0)

def build_indices_decoder_causal(seq_len: int, frag: int, device: torch.device) -> torch.Tensor:
    t = torch.arange(seq_len, device=device)
    starts = torch.clamp(t - (frag - 1), 0, max(0, seq_len - frag))
    return starts.unsqueeze(1) + torch.arange(frag, device=device).unsqueeze(0)

def build_indices_cross(tgt_len: int, src_len: int, frag: int, device: torch.device) -> torch.Tensor:
    centers = torch.zeros(max(1, tgt_len), device=device, dtype=torch.long) if tgt_len <= 1 \
        else torch.round(torch.linspace(0, src_len - 1, steps=tgt_len, device=device)).long()
    half = frag // 2
    starts = torch.clamp(centers - half, 0, max(0, src_len - frag))
    return starts.unsqueeze(1) + torch.arange(frag, device=device).unsqueeze(0)

def sliding_window_view_seq(x: torch.Tensor, Fw: int, causal: bool) -> torch.Tensor:
    # x: [BH, T, d] -> [BH, T, Fw, d]
    BH, T, D = x.shape
    if causal:
        left, right = Fw - 1, 0
    else:
        left = Fw // 2
        right = Fw - 1 - left
    xpad = F.pad(x, (0, 0, left, right), mode="replicate")  # [BH, T+left+right, d]
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


# --------- selection modules ---------
class GlobalChooser:
    def __init__(self, cfg: RouterConfig):
        self.cfg = cfg

    def _proto_idx(self, Tq: int, p: int, device) -> torch.Tensor:
        if p <= 0 or p >= Tq:
            return torch.arange(Tq, device=device)
        return torch.round(torch.linspace(0, Tq - 1, steps=p, device=device)).long()

    def choose_packed(self, key_states: torch.Tensor, q_summary: torch.Tensor,
                    bsz: int, num_heads: int, g_eff: int) -> List[torch.Tensor]:
        """
        key_states: [BH, Tk, d]
        q_summary:  either [H, Tq, d]  (legacy, batch-averaged)
                    or     [H, B, Tq, d] (per-example)
                    Caller should migrate to [H,B,Tq,d].
        returns: list length H, each [B, g_eff] LongTensor (absolute positions)
        """
        device = key_states.device
        H = num_heads
        BH, Tk, d = key_states.shape

        # Reshape keys per head and per example
        KHB = key_states.view(H, bsz, Tk, d)                       # [H,B,Tk,d]
        Kbar = F_normalize_safe(KHB, dim=-1)                       # [H,B,Tk,d]

        # --- Normalize & shape q_summary to [H,B,Tq,d] ---
        if q_summary.dim() == 3:                                   # [H,Tq,d] -> broadcast over batch
            H_, Tq, d_ = q_summary.shape
            assert H_ == H and d_ == d, f"q_summary dims mismatch: got {q_summary.shape}, expected H={H}, d={d}"
            q_summary = q_summary.unsqueeze(1).expand(H, bsz, Tq, d)  # [H,B,Tq,d]
        elif q_summary.dim() == 4:                                 # [H,B,Tq,d]
            H_, B_, Tq, d_ = q_summary.shape
            assert H_ == H and B_ == bsz and d_ == d, \
                f"q_summary dims mismatch: got {q_summary.shape}, expected [H={H}, B={bsz}, Tq, d={d}]"
        else:
            raise ValueError(f"q_summary must be 3D or 4D, got shape {q_summary.shape}")

        # Prototypes per example
        p = max(1, int(self.cfg.proto_count))
        idxp = self._proto_idx(Tq, p, device)                      # [p']
        Qp = q_summary.index_select(2, idxp)                       # [H,B,p',d]
        Qp = F_normalize_safe(Qp, dim=-1)

        # Similarity per token to prototypes: [H,B,Tk,p']
        S = torch.relu(torch.einsum('hbkd,hbpd->hbkp', Kbar, Qp))

        # Prefilter score per token: [H,B,Tk]
        mean = S.mean(-1)
        mx   = S.max(-1).values
        kq   = max(1, int(round(S.size(-1) * self.cfg.topk_frac)))
        topk_mean = torch.topk(S, k=kq, dim=-1).values.mean(-1)
        std  = S.std(-1)
        u_full = (self.cfg.w_mean * mean + self.cfg.w_max * mx
                + self.cfg.w_topk * topk_mean + self.cfg.w_std * std)   # [H,B,Tk]

        # Top-U prefilter per (h,b)
        U = int(min(max(1, int(self.cfg.top_u)), Tk))
        _, top_idx = torch.topk(u_full, k=U, dim=-1)                       # [H,B,U]
        idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, -1, S.size(-1))
        S_sub = torch.gather(S, dim=2, index=idx_exp)                      # [H,B,U,p']

        # Greedy facility-location per (h,b)
        p_eff = S_sub.size(-1)
        m = torch.zeros(H, bsz, p_eff, device=device, dtype=S_sub.dtype)   # running max coverage
        blocked = torch.zeros(H, bsz, U, device=device, dtype=torch.bool)
        chosen_local = torch.zeros(H, bsz, g_eff, device=device, dtype=torch.long)

        steps = min(g_eff, U)
        for r in range(steps):
            gains = torch.relu(S_sub - m.unsqueeze(2)).sum(dim=-1)         # [H,B,U]
            gains = gains.masked_fill(blocked, -1e9)
            j = gains.argmax(dim=-1)                                       # [H,B] (indices in 0..U-1)
            chosen_local[:, :, r] = j
            blocked.scatter_(2, j.unsqueeze(-1), True)
            # m = max(m, S_sub[..., j, :])  (gather version, vectorized)
            sel = S_sub.gather(2, j.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, p_eff)).squeeze(2)  # [H,B,p']
            m = torch.maximum(m, sel)

        chosen_abs = torch.gather(top_idx, dim=2, index=chosen_local)      # [H,B,g_eff]
        return [chosen_abs[h] for h in range(H)]                           # list of [B,g_eff]




class WindowSelector:
    def __init__(self, cfg: RouterConfig):
        self.cfg = cfg

    def select_blocked_mmr(self, scores_win: torch.Tensor, prior_win: torch.Tensor,
                           keys_win: Optional[torch.Tensor],
                           k_target: int) -> torch.Tensor:
        """
        Blocked-MMR:
          1) TopK* prefilter over blended scores (K* = min(Fw, ceil(mult*k)))
          2) 1–2 diversity rounds on the candidate set (vectorized)
        """
        BH, Tq, Fw = scores_win.shape
        k = min(k_target, Fw)
        if self.cfg.random_selection:
            return torch.randint(low=0, high=Fw, size=(BH, Tq, k), device=scores_win.device)

        blended = scores_win + self.cfg.alpha_pos_prior * (
            prior_win if prior_win.dim() == 3 else prior_win.unsqueeze(0)
        )

        # Prefilter: Kc = min(Fw, ceil(mult*k))
        Kc = int(min(Fw, max(k, int(np.ceil(self.cfg.mmr_prefilter_mult * k)))))
        # topK over Fw → candidate positions (per BH,Tq)
        cand_vals, cand_idx = torch.topk(blended, k=Kc, dim=-1, largest=True)  # [BH,Tq,Kc], [BH,Tq,Kc]

        if keys_win is None or self.cfg.gamma_diversity == 0.0 or k == 1:
            # No diversity: just take top-k of candidates
            top_vals, top_pos = torch.topk(cand_vals, k=k, dim=-1, largest=True)  # pos within candidates
            return torch.gather(cand_idx, -1, top_pos)

        # Diversity on candidate set
        kv = F_normalize_safe(keys_win, dim=-1)  # [BH,Tq,Fw,d]
        # gather candidate key vectors: [BH,Tq,Kc,d]
        cand_keys = torch.gather(kv, -2, cand_idx.unsqueeze(-1).expand(-1, -1, -1, kv.size(-1)))

        # initialize
        sel = torch.zeros(BH, Tq, k, dtype=torch.long, device=scores_win.device)
        remaining_scores = cand_vals.clone()  # [BH,Tq,Kc]
        selected_vecs = None

        steps = min(k, 1 + max(0, int(self.cfg.mmr_diversity_steps)))  # 1..(1+diversity_steps)
        for r in range(steps):
            j = remaining_scores.argmax(dim=-1)               # [BH,Tq] (index within candidate set)
            sel[:, :, r] = torch.gather(cand_idx, -1, j.unsqueeze(-1)).squeeze(-1)  # absolute positions
            # update remaining_scores with diversity penalty for next round
            if r < steps - 1:
                # pull selected vectors: [BH,Tq,d]
                sel_vec = torch.gather(cand_keys, -2, j.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, cand_keys.size(-1))).squeeze(-2)
                # cosine to all candidates: [BH,Tq,Kc]
                cos = (cand_keys * sel_vec.unsqueeze(-2)).sum(dim=-1).clamp(min=0)
                remaining_scores = remaining_scores - self.cfg.gamma_diversity * cos
                # mask the selected candidate to avoid re-picking
                remaining_scores.scatter_(-1, j.unsqueeze(-1), -1e9)

        # If k > steps, fill the rest by score within candidates (fast topk on masked scores)
        if steps < k:
            fill_k = k - steps
            # take the top-(k-steps) from remaining_scores
            fill_vals, fill_pos = torch.topk(remaining_scores, k=fill_k, dim=-1, largest=True)
            fill_abs = torch.gather(cand_idx, -1, fill_pos)
            sel[:, :, steps:] = fill_abs
        return sel


class _IdxCache:
    """Tiny LRU for (idx_win, prior_win)."""
    def __init__(self, cap: int = 8):
        self.cap = cap
        self.cache: Dict[Tuple[int, int, int, str], Tuple[torch.Tensor, torch.Tensor]] = {}
        self.usage: List[Tuple[int, int, int, str]] = []

    def get(self, key):
        if key in self.cache:
            # move to end (most recent)
            if key in self.usage:
                self.usage.remove(key)
            self.usage.append(key)
            return self.cache[key]
        return None

    def put(self, key, value):
        if key in self.cache:
            self.cache[key] = value
            if key in self.usage:
                self.usage.remove(key)
            self.usage.append(key)
            return
        if len(self.cache) >= self.cap:
            old = self.usage.pop(0)
            self.cache.pop(old, None)
        self.cache[key] = value
        self.usage.append(key)


class BiggerBird(BartAttention):
    def __init__(self, base_attn: BartAttention, cfg: RouterConfig, runtime: RouterRuntime):
        self.drop_p = base_attn.dropout.p if isinstance(base_attn.dropout, nn.Dropout) else float(base_attn.dropout)
        super().__init__(
            embed_dim=base_attn.embed_dim,
            num_heads=base_attn.num_heads,
            dropout=self.drop_p,
            is_decoder=base_attn.is_decoder,
            bias=base_attn.k_proj.bias is not None,
            is_causal=getattr(base_attn, "is_causal", False),
            layer_idx=getattr(base_attn, "layer_idx", None),
        )
        # copy weights
        self.q_proj.load_state_dict(base_attn.q_proj.state_dict())
        self.k_proj.load_state_dict(base_attn.k_proj.state_dict())
        self.v_proj.load_state_dict(base_attn.v_proj.state_dict())
        self.out_proj.load_state_dict(base_attn.out_proj.state_dict())

        self.cfg = cfg
        self.runtime = runtime
        self.global_chooser = GlobalChooser(cfg)
        self.window_selector = WindowSelector(cfg)
        self._printed_cfg = False
        self._logged_pairs = False
        self._idx_cache = _IdxCache(cap=8)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _get_idx_and_prior(self, tgt_len: int, src_len: int, Fw: int, device, mode: str):
        key = (tgt_len, src_len, Fw, mode)
        cached = self._idx_cache.get(key)
        if cached is not None:
            return cached
        if mode == "cross":
            idx_win = build_indices_cross(tgt_len, src_len, Fw, device)
            centers = torch.round(torch.linspace(0, src_len - 1, steps=tgt_len, device=device)).long()
        else:
            if mode == "decoder":
                idx_win = build_indices_decoder_causal(tgt_len, Fw, device)
            else:
                idx_win = build_indices_encoder(tgt_len, Fw, device)
            centers = torch.arange(tgt_len, device=device).long().clamp(0, src_len - 1)
        prior = positional_prior(idx_win, centers, tau=max(Fw / 4, 1.0))
        self._idx_cache.put(key, (idx_win, prior))
        return idx_win, prior

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        # ignore extras
        _ = kwargs.pop("cache_position", None)
        _ = kwargs.pop("position_bias", None)
        _ = kwargs.pop("alibi_bias", None)

        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()
        device = hidden_states.device

        # projections
        query_states = self.q_proj(hidden_states) * (self.head_dim ** -0.5)
        if is_cross_attention:
            key_states_full = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states_full = self._shape(self.v_proj(key_value_states), -1, bsz)
        else:
            key_states_full = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states_full = self._shape(self.v_proj(hidden_states), -1, bsz)

        BH = bsz * self.num_heads
        Q = self._shape(query_states, tgt_len, bsz).reshape(BH, tgt_len, self.head_dim)
        K = key_states_full.reshape(BH, -1, self.head_dim)   # [BH, Tk, d]
        V = value_states_full.reshape(BH, -1, self.head_dim)
        src_len = K.size(1)

        # ---- Dense fallback for short sequences
        if src_len <= int(self.cfg.dense_fallback_under):
            return super().forward(
                hidden_states=hidden_states,
                key_value_states=key_value_states,
                past_key_value=past_key_value,
                attention_mask=attention_mask,
                layer_head_mask=layer_head_mask,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )

        # Window indices + prior (cached)
        Fw = min(int(self.cfg.fragment_size), src_len)
        mode = "cross" if is_cross_attention else ("decoder" if self.is_decoder else "encoder")
        idx_win, prior_win = self._get_idx_and_prior(tgt_len, src_len, Fw, device, mode)


        # ---- LOCALS: sliding K_win (no huge expand)
        with torch.no_grad():
            if is_cross_attention:
                idx_expanded = idx_win.view(1, tgt_len, Fw, 1).expand(BH, tgt_len, Fw, self.head_dim)
                K_win = torch.gather(K.unsqueeze(1).expand(BH, tgt_len, src_len, self.head_dim), 2, idx_expanded)
            else:
                K_win = sliding_window_view_seq(K, Fw=Fw, causal=bool(self.is_decoder))  # [BH,Tq,Fw,d]

            # scores for locals
            Qn = F_normalize_safe(Q, dim=-1).unsqueeze(2)          # [BH,Tq,1,d]
            Kn = F_normalize_safe(K_win, dim=-1).transpose(-1, -2) # [BH,Tq,d,Fw]
            scores_win = torch.matmul(Qn, Kn).squeeze(2)           # [BH,Tq,Fw]


            if self.cfg.log_once_pairs and not self._printed_cfg:
                print(f"[QuAttn] heads={self.num_heads} F={Fw} "
                      f"k={self.cfg.k_per_query} (min..max={self.cfg.min_k}..{self.cfg.max_k}, target={self.cfg.r_target_softmax:.2f}) "
                      f"g={self.cfg.globals_per_head} t={self.cfg.teleports_per_head} "
                      f"(locals=blocked-MMR, globals=facility-location via prototypes p={self.cfg.proto_count}) "
                      f"is_decoder={self.is_decoder}")
                self._printed_cfg = True

            # ---- window selection (blocked MMR)
            Tk = K.shape[1]
            g_cfg = int(self.cfg.globals_per_head)
            t_cfg = max(0, int(self.cfg.teleports_per_head))

            # Adaptive g/t (keep O(n), reduce overhead on shorts; mild growth on very long)
            if Tk < 768:
                g_eff = max(1, (g_cfg + 1) // 2)
                t_eff = 0 if Tk < 384 else min(1, t_cfg)
            elif Tk > 2048:
                g_eff = min(g_cfg + 1, g_cfg + (1 if self.num_heads <= 16 else 0))
                t_eff = t_cfg
            else:
                g_eff, t_eff = g_cfg, t_cfg

            # Desired M; clamp k to [min_k, max_k] and ≤ Fw
            M_desired = max(g_eff + t_eff + 1, int(self.cfg.r_target_softmax * Tk))
            k_here = max(self.cfg.min_k, min(self.cfg.max_k, M_desired - (g_eff + t_eff)))
            k_here = min(k_here, Fw)

            # After: scores_win = ...  # [BH, Tq, Fw]
            if attention_mask is not None:
                # Normalize to a boolean "allowed" mask shaped [B, Tq, Tk]
                if attention_mask.dtype == torch.bool:
                    if attention_mask.dim() == 2:               # [B, Tk]
                        am_small = attention_mask.unsqueeze(1).expand(bsz, tgt_len, src_len)
                    elif attention_mask.dim() == 4:             # [B, 1, Tq, Tk]
                        am_small = attention_mask.squeeze(1)    # -> [B, Tq, Tk]
                    else:                                       # assume [B, Tq, Tk]
                        am_small = attention_mask
                else:
                    # additive mask: 0 allowed, -inf disallowed (Bart uses additive)
                    if attention_mask.dim() == 4:               # [B, 1, Tq, Tk]
                        am_small = torch.isfinite(attention_mask).squeeze(1)  # True=allowed
                    elif attention_mask.dim() == 2:             # [B, Tk]
                        am_small = (attention_mask > 0).unsqueeze(1).expand(bsz, tgt_len, src_len)
                    else:                                       # assume [B, Tq, Tk]
                        am_small = torch.isfinite(attention_mask)

                # Gather into the window: idx_win [Tq, Fw] -> [B, Tq, Fw]
                idx_btf = idx_win.unsqueeze(0).expand(bsz, -1, -1)
                mask_win = torch.gather(am_small, -1, idx_btf)  # [B, Tq, Fw]

                # Expand to BH and apply
                mask_win_bh = mask_win.unsqueeze(1).expand(bsz, self.num_heads, tgt_len, Fw).reshape(BH, tgt_len, Fw)
                scores_win = scores_win.masked_fill(~mask_win_bh, torch.finfo(scores_win.dtype).min)


            sel_idx = self.window_selector.select_blocked_mmr(
                scores_win=scores_win, prior_win=prior_win, keys_win=K_win, k_target=k_here
            )  # [BH,Tq,k]
            k = sel_idx.size(-1)

            # Absolute indices for locals: gather from idx_win (small)
            abs_idx_win = torch.gather(idx_win.unsqueeze(0).expand(BH, -1, -1), -1, sel_idx)  # [BH,Tq,k]

            # ---- globals per head (facility-location; amortized) ----
            H = self.num_heads
            Qh = Q.view(H, bsz, tgt_len, self.head_dim)                # [H,B,Tq,d]
            q_summary = F_normalize_safe(Qh, dim=-1)       # [H,Tq,d]

            globals_per_head = self.runtime.maybe_get_shared_globals(getattr(self, "layer_idx", None), src_len)
            if globals_per_head is None or len(globals_per_head) != H or globals_per_head[0].numel() < g_eff:
                globals_per_head = self.global_chooser.choose_packed(
                    key_states=K, q_summary=q_summary, bsz=bsz, num_heads=H, g_eff=g_eff
                )
                self.runtime.store_globals(getattr(self, "layer_idx", None), src_len, globals_per_head)

            # ---- teleports per head (prototype-biased + uniform), O(H*Tk*p) ----
            t = max(0, int(t_eff))
            tele_list: List[torch.Tensor] = []
            if t > 0:
                # Keys averaged across batch (per-head)
                K_head = K.view(H, bsz, src_len, self.head_dim).mean(dim=1)   # [H,Tk,d]
                Kbar   = F_normalize_safe(K_head, dim=-1)                     # [H,Tk,d]

                # Reduce q_summary back to per-head for teleports only
                # q_summary is [H,B,Tq,d] after your change; fall back if still 3D
                if q_summary.dim() == 4:
                    q_long = F_normalize_safe(q_summary.mean(dim=1), dim=-1)  # [H,Tq,d]
                else:
                    q_long = q_summary                                        # [H,Tq,d]

                p = max(1, int(self.cfg.proto_count))
                idxp = self.global_chooser._proto_idx(tgt_len, p, device)     # [p']
                Qp   = q_long.index_select(1, idxp)                           # [H,p',d]
                Qp   = F_normalize_safe(Qp, dim=-1)

                # Similarity to long-range prototypes: [H,Tk,p']
                S_long = torch.relu(torch.einsum('hkd,hpd->hkp', Kbar, Qp))
                u_long = S_long.mean(dim=-1)                                   # [H,Tk]

                tele_biased  = int(round(t * float(self.cfg.teleport_bias_frac)))
                tele_uniform = t - tele_biased
                for h in range(H):
                    picks = []
                    if tele_biased > 0:
                        picks.append(torch.topk(u_long[h], k=min(tele_biased, src_len)).indices)
                    if tele_uniform > 0:
                        picks.append(torch.randperm(src_len, device=device)[:tele_uniform])
                    tele = torch.cat(picks)[:t] if picks else torch.empty(0, dtype=torch.long, device=device)
                    tele_list.append(tele)
            else:
                tele_list = [torch.empty(0, dtype=torch.long, device=device) for _ in range(H)]

    
            # --- always-global anchors (CLS and EOS) ---
            if attention_mask is not None and attention_mask.dim() == 2:  # [B,T]
                lens = attention_mask.long().sum(dim=1).clamp(min=1)
                eos_idx_b = (lens - 1)  # last non-pad per batch
            else:
                eos_idx_b = torch.full((bsz,), tgt_len - 1, device=device, dtype=torch.long)
            cls_idx_b = torch.zeros(bsz, dtype=torch.long, device=device)

            cls_exp = cls_idx_b.view(bsz, 1).unsqueeze(1).expand(bsz, tgt_len, 1)  # [B,Tq,1]
            eos_exp = eos_idx_b.view(bsz, 1).unsqueeze(1).expand(bsz, tgt_len, 1)  # [B,Tq,1]

            # ---- Build final absolute index tensor per head: [B,Tq,M] ----
            abs_idx_list = []
            for h in range(H):
                y = globals_per_head[h].to(device=device, dtype=torch.long)  # [g_eff]
                y_exp = y.view(1, 1, -1).expand(bsz, tgt_len, -1)            # [B,Tq,g_eff]
                parts = [abs_idx_win[h*bsz:(h+1)*bsz], y_exp, cls_exp, eos_exp]
                te = tele_list[h]
                if te.numel() > 0:
                    te_exp = te.view(1, 1, -1).expand(bsz, tgt_len, -1)
                    parts.append(te_exp)
                abs_idx_h = torch.cat(parts, dim=-1)                         # [B,Tq,M]
                abs_idx_list.append(abs_idx_h)
            
            # Ensure all heads share same M
            M = abs_idx_list[0].size(-1)
            for i in range(1, len(abs_idx_list)):
                if abs_idx_list[i].size(-1) != M:
                    diff = M - abs_idx_list[i].size(-1)
                    pad = abs_idx_list[i][..., :1].expand(-1, -1, diff)
                    abs_idx_list[i] = torch.cat([abs_idx_list[i], pad], dim=-1)

            abs_idx = torch.cat(abs_idx_list, dim=0)  # [BH,Tq,M]

            if self.cfg.log_once_pairs and not hasattr(self, "_printed_M"):
                M = abs_idx.size(-1)
                print(f"[router] Tk={K.shape[1]} Fw={Fw} k_here={k_here} g_eff={g_eff} t_eff={t_eff} M={M}")
                self._printed_M = True

        # ---- Final attention over selected tokens: flat batched gather ----
        K_sel = flat_batched_gather_kv(K, abs_idx)  # [BH,Tq,M,d]
        V_sel = flat_batched_gather_kv(V, abs_idx)  # [BH,Tq,M,d]

        # Per-query dot product over d
        scores_sel = (Q.unsqueeze(2) * K_sel).sum(-1)  # [BH,Tq,M]

        # Softmax-comparison ratio (one-time log)
        if self.cfg.log_once_pairs and not self._logged_pairs:
            BH_local = BH
            Tk_local = K.shape[1]
            M_now = abs_idx.size(-1)
            comps_sparse = BH_local * tgt_len * M_now
            comps_full_unmasked = BH_local * tgt_len * Tk_local
            ratio_softmax = comps_sparse / max(1, comps_full_unmasked)
            print(f"[softmax] sparse={comps_sparse:,} | full(unmasked)={comps_full_unmasked:,} ratio={ratio_softmax:.4f}")
            self._logged_pairs = True

        # Apply attention mask (head-sliced gather)
        if attention_mask is not None:
            am_bool = attention_mask if attention_mask.dtype == torch.bool else (attention_mask > -1e-8)
            if am_bool.dim() == 3:   # [B,Tq,Tk] -> [B,1,Tq,Tk]
                am_bool = am_bool.unsqueeze(1)
            if am_bool.dim() == 2:   # [B,Tk] -> [B,1,1,Tk]
                am_bool = am_bool[:, None, None, :]
            abs_idx_hb = abs_idx.view(self.num_heads, bsz, tgt_len, M)
            allowed_chunks = []
            for h in range(self.num_heads):
                am_small = am_bool.expand(bsz, 1, tgt_len, src_len)   # view
                allowed_h = torch.gather(am_small, -1, abs_idx_hb[h].unsqueeze(1)).squeeze(1)  # [B,Tq,M]
                allowed_chunks.append(allowed_h)
            allowed = torch.cat(allowed_chunks, dim=0)                # [BH,Tq,M]
            scores_sel = scores_sel.masked_fill(~allowed, torch.finfo(scores_sel.dtype).min)

        # Softmax + dropout + output
        attn_probs = F.softmax(scores_sel, dim=-1)            # [BH,Tq,M]
        attn_probs = F.dropout(attn_probs, p=self.drop_p, training=self.training)

        BH_Tq = BH * tgt_len
        out = torch.bmm(
            attn_probs.reshape(BH_Tq, 1, M),                  # [BH*Tq,1,M]
            V_sel.reshape(BH_Tq, M, self.head_dim)            # [BH*Tq,M,d]
        ).reshape(BH, tgt_len, self.head_dim)                 # [BH,Tq,d]

        attn_output = out.view(bsz, self.num_heads, tgt_len, self.head_dim) \
                        .transpose(1, 2).reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        present_key_value = (key_states_full, value_states_full) if use_cache else None
        attn_weights_reshaped = attn_probs.view(bsz, self.num_heads, tgt_len, abs_idx.size(-1)) if output_attentions else None

        # Optional debug stats
        if self.cfg.debug_collect:
            mean_dist = 0.0
            if k > 0:
                local_abs = abs_idx[:, :, :k]
                centers = torch.arange(tgt_len, device=device).long().clamp(0, src_len - 1) if not is_cross_attention \
                          else torch.round(torch.linspace(0, src_len - 1, steps=tgt_len, device=device)).long()
                mean_dist = (local_abs - centers.view(1, -1, 1)).abs().float().mean().item()
            jacc = 0.0
            # globals_per_head used above is list[Tensor]
            # (skip jacc if empty)
            self._debug_meta = {"M": abs_idx.size(-1), "mean_local_distance": mean_dist, "global_jaccard": jacc}

        if use_cache:
            return (attn_output, attn_weights_reshaped, present_key_value)
        else:
            return (attn_output, attn_weights_reshaped)


def patch_bart_with_biggerbird(model: nn.Module, cfg: RouterConfig) -> RouterRuntime:
    num_heads = None
    for mod in model.modules():
        if isinstance(mod, BartAttention) and (not getattr(mod, "is_decoder", False)):
            num_heads = getattr(mod, "num_heads", None)
            if num_heads is not None:
                break
    if num_heads is None:
        num_heads = 12  # fallback

    runtime = RouterRuntime(num_heads=num_heads, cfg=cfg)

    def _recurse(module: nn.Module):
        for name, child in list(module.named_children()):
            if isinstance(child, BartAttention):
                if cfg.patch_encoder_only and getattr(child, "is_decoder", False):
                    continue
                setattr(module, name, BiggerBird(child, cfg, runtime))
            else:
                _recurse(child)
    _recurse(model)
    return runtime

class AttnPool(nn.Module):
    """
    Single-head additive attention pooling:
      scores_t = w^T tanh(W x_t)  -> softmax over T  ->  sum_t a_t x_t
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj  = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x:    [B, T, D]
        mask: [B, T] (1 for real tokens, 0 for pad)
        """
        h = torch.tanh(self.proj(x))               # [B,T,D]
        s = self.score(h).squeeze(-1)              # [B,T]
        # mask pads to -inf before softmax (MPS-safe)
        s = s.masked_fill(~mask.bool(), torch.finfo(s.dtype).min)
        a = torch.softmax(s, dim=-1)               # [B,T]
        pooled = torch.bmm(a.unsqueeze(1), x).squeeze(1)  # [B,D]
        return pooled

class BiggerBirdBartForSequenceClassification(nn.Module):
    def __init__(self, base_model: nn.Module, cfg: RouterConfig = default_router_config):
        super().__init__()
        self.model = base_model
        self.runtime = patch_bart_with_biggerbird(self.model, cfg)

        # hidden size for BART is usually config.d_model (not hidden_size)
        hidden_size = getattr(self.model.config, "hidden_size", None) or getattr(self.model.config, "d_model")
        self.attn_pool = AttnPool(hidden_size)

    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.model, "gradient_checkpointing_enable"):
            return self.model.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        if hasattr(self.model, "gradient_checkpointing_disable"):
            return self.model.gradient_checkpointing_disable()

    @property
    def supports_gradient_checkpointing(self):
        return getattr(self.model, "supports_gradient_checkpointing", True)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        token_type_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
        **kwargs,
    ):
        # patch lifetime
        self.runtime.begin_forward()
        try:
            # don't let HF pass extras we don't use
            kwargs.pop("token_type_ids", None)
            kwargs.pop("num_items_in_batch", None)

            # run the BART backbone (patched with BiggerBird) to get last hidden states
            outputs = self.model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
            last_hidden = outputs.last_hidden_state   # [B, T, D]

            # build a mask if none provided
            if attention_mask is None:
                if input_ids is not None and self.model.config.pad_token_id is not None:
                    attention_mask = (input_ids != self.model.config.pad_token_id).long()
                else:
                    attention_mask = torch.ones(last_hidden.size()[:2], device=last_hidden.device, dtype=torch.long)

            # Attention pooling            
            mask_bool = attention_mask.bool()  # shape [B, T]
            pooled = self.attn_pool(last_hidden, mask_bool)   # <-- pass the mask

            # same classification head as BartForSequenceClassification
            logits = self.model.classification_head(pooled)

            # same classification head as BartForSequenceClassification
            logits = self.model.classification_head(pooled)

            # ---- compute loss for Trainer ----
            loss = None
            if labels is not None:
                if labels.dtype != torch.long:
                    labels = labels.long()
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.model.config.num_labels),
                    labels.view(-1),
                )

            # return HF-style output
            if return_dict:
                return SequenceClassifierOutput(loss=loss, logits=logits)
            else:
                output = (logits,)
                return ((loss,) + output) if loss is not None else output
        finally:
            self.runtime.end_forward()

    @property
    def config(self):
        return self.model.config

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, cfg: RouterConfig = default_router_config):
        from transformers import AutoModelForSequenceClassification
        base = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        return cls(base, cfg)


# -------------------------------
# Training / evaluation
# -------------------------------

@dataclass
class TrainConfig:
    bart_name: str = "facebook/bart-base"
    bigbird_name: str = "google/bigbird-roberta-base"

    seed: int = 42
    epochs: int = 3

    per_device_train_bs: int = 2
    per_device_eval_bs: int = 2
    grad_accum_steps: int = 8

    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    max_length: int = 768

    train_samples: int = 2000 #6k for overnight
    eval_samples: int = 1000

    show_debug_meta: bool = True

train_cfg = TrainConfig()

# Router defaults (tuned for speed while keeping quality)
from transformers import TrainerCallback

class RouterAnnealCallback(TrainerCallback):
    def __init__(self, model, start_frac=0.16, end_frac=0.12, warmup_steps=80,
                 start_max_k=28, end_max_k=24):
        self.model = model
        self.start_frac = start_frac
        self.end_frac = end_frac
        self.warmup_steps = warmup_steps
        self.start_max_k = start_max_k
        self.end_max_k = end_max_k

    def on_step_begin(self, args, state, control, **kwargs):
        step = state.global_step
        if step <= self.warmup_steps:
            t = step / max(1, self.warmup_steps)
            frac = self.start_frac + t * (self.end_frac - self.start_frac)
            max_k = int(round(self.start_max_k + t * (self.end_max_k - self.start_max_k)))
            # update all BiggerBird modules
            for m in self.model.model.modules():
                if isinstance(m, BiggerBird):
                    m.cfg.r_target_softmax = frac
                    m.cfg.max_k = max_k
# --- Put this near your other callbacks ---
from transformers import TrainerCallback

def _set_router_density(model, r, max_k):
    for m in model.model.modules():
        if isinstance(m, BiggerBird):
            m.cfg.r_target_softmax = float(r)
            m.cfg.max_k = int(max_k)

class FirstEpochRouterDensity(TrainerCallback):
    """
    Epoch 0: r_target_softmax=first_r, max_k=first_max_k
    Epoch 1+: r_target_softmax=rest_r,  max_k=rest_max_k
    """
    def __init__(self, model, first_r=0.14, first_max_k=28, rest_r=0.12, rest_max_k=24):
        self.model = model
        self.first_r = first_r
        self.first_max_k = first_max_k
        self.rest_r = rest_r
        self.rest_max_k = rest_max_k
        self._epoch_idx = -1

    def on_train_begin(self, args, state, control, **kwargs):
        # Ensure we start dense before the first forward
        _set_router_density(self.model, self.first_r, self.first_max_k)

    def on_epoch_begin(self, args, state, control, **kwargs):
        self._epoch_idx += 1
        if self._epoch_idx == 0:
            _set_router_density(self.model, self.first_r, self.first_max_k)
        else:
            _set_router_density(self.model, self.rest_r, self.rest_max_k)

    def on_evaluate(self, args, state, control, **kwargs):
        # Make sure eval happens at the "rest" density
        _set_router_density(self.model, self.rest_r, self.rest_max_k)

# --- EMA that is device/dtype safe (MPS-friendly) ---
class EMACallback(TrainerCallback):
    def __init__(self, model, decay=0.999):
        self.model = model                # pass the *HF* model, e.g. model.model
        self.decay = float(decay)
        self.shadow = {}
        self._backup = {}

    @torch.no_grad()
    def on_train_begin(self, args, state, control, **kwargs):
        # Create shadows on the current param devices/dtypes
        self.shadow.clear()
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.detach().clone().to(device=p.device, dtype=p.dtype)

    @torch.no_grad()
    def on_step_end(self, args, state, control, **kwargs):
        # Keep shadow in sync with param device/dtype and update EMA
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            s = self.shadow.get(n)
            if (s is None) or (s.shape != p.data.shape) or (s.device != p.device) or (s.dtype != p.data.dtype):
                s = p.detach().clone().to(device=p.device, dtype=p.dtype)
                self.shadow[n] = s
            s.mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    @torch.no_grad()
    def _swap_to_shadow(self):
        self._backup = {}
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            self._backup[n] = p.data.detach().clone()
            p.data.copy_(self.shadow[n])  # already correct device/dtype

    @torch.no_grad()
    def _swap_back(self):
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            p.data.copy_(self._backup[n])

    # Use EMA weights for eval/save, then swap back
    def on_evaluate(self, args, state, control, **kwargs):
        self._swap_to_shadow()

    def on_evaluate_end(self, args, state, control, **kwargs):
        self._swap_back()

    def on_save(self, args, state, control, **kwargs):
        # ensure checkpoints save EMA weights if save happens outside evaluate()
        if not self._backup:
            self._swap_to_shadow()

    def on_save_end(self, args, state, control, **kwargs):
        if self._backup:
            self._swap_back()


def compute_metrics(eval_pred):
    # With preprocess_logits_for_metrics, predictions are class indices (np array)
    if isinstance(eval_pred, EvalPrediction):
        preds, labels = eval_pred.predictions, eval_pred.label_ids
    else:
        preds, labels = eval_pred
    return {"accuracy": accuracy_score(labels, preds), "f1": f1_score(labels, preds)}

def build_imdb_dataset(tokenizer, max_length: int, fixed_length: int = None):
    ds = load_dataset("imdb")
    if train_cfg.train_samples:
        ds["train"] = ds["train"].shuffle(seed=train_cfg.seed).select(range(train_cfg.train_samples))
    if train_cfg.eval_samples:
        ds["test"] = ds["test"].shuffle(seed=train_cfg.seed).select(range(train_cfg.eval_samples))

    def tok_fn(batch):
        if fixed_length is not None:
            return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=fixed_length)
        else:
            return tokenizer(batch["text"], truncation=True, max_length=max_length)

    ds = ds.map(tok_fn, batched=True, remove_columns=["text"])
    ds = ds.rename_column("label", "labels")
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return {"train": ds["train"], "validation": ds["test"]}

def device_flags():
    use_cuda = torch.cuda.is_available()
    use_mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    # On MPS: FP32, no compile, no checkpoint, workers=0
    fp16 = False
    bf16 = False
    torch_compile = False
    pin_mem = False
    if use_cuda:
        fp16 = True
        bf16 = torch.cuda.is_bf16_supported()
        torch_compile = False
        pin_mem = True
    return fp16, bf16, torch_compile, pin_mem, use_mps

def make_args(out_dir: str) -> TrainingArguments:
    fp16, bf16, torch_compile, pin_mem, use_mps = device_flags()
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
        lr_scheduler_type="cosine",        # smoother for short runs
        max_grad_norm=1.0,                 # optional: tighter clip if spikes persist
        logging_strategy="steps",
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
        fp16=fp16,
        bf16=bf16,
        remove_unused_columns=False,
        dataloader_num_workers=0,       # macOS best: 0
        dataloader_pin_memory=False,    # False on MPS/CPU
        gradient_checkpointing=False,   # off on MPS
        torch_compile=torch_compile,    # off on MPS
        optim="adamw_torch",
        eval_accumulation_steps=eval_accum,
        #label_smoothing_factor=0.05,
    )

def load_bigbird_tok(model_name: str):
    try:
        from transformers import BigBirdTokenizer
        return BigBirdTokenizer.from_pretrained(model_name)  # stable slow tokenizer
    except Exception as e_slow:
        print(f"[BigBird] slow tokenizer failed: {e_slow}\nTrying fast tokenizer...", flush=True)
        return AutoTokenizer.from_pretrained(model_name, use_fast=True)

def summarize_token_selection_configs(router_cfg: RouterConfig, bigbird_model_name: str):
    print("\n=== Token-Selection Configs ===")
    # BiggerBird
    print(
        "BiggerBird (ours): "
        f"Fw={router_cfg.fragment_size}, k e [{router_cfg.min_k},{router_cfg.max_k}] "
        f"(target={router_cfg.r_target_softmax:.2f}·Tk), globals/head={router_cfg.globals_per_head}, "
        f"teleports/head={router_cfg.teleports_per_head}, prototypes p={router_cfg.proto_count}, prefilter U={router_cfg.top_u}"
    )
    # BigBird (from config fields)
    bb_cfg = AutoConfig.from_pretrained(bigbird_model_name)
    bb_attn_type = getattr(bb_cfg, "attention_type", None)
    bb_block = getattr(bb_cfg, "block_size", None)
    bb_rand = getattr(bb_cfg, "num_random_blocks", None)
    bb_heads = getattr(bb_cfg, "num_attention_heads", None)
    print(
        "BigBird (HF): "
        f"attention_type={bb_attn_type}, block_size={bb_block}, random_blocks={bb_rand}, heads={bb_heads}, "
        "globals=dynamic (via attention_mask)"
    )
    print("================================\n", flush=True)

# -------------------------------
# Trainers
# -------------------------------
def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    return logits.argmax(dim=-1)

def train_and_eval_biggerbird(tokenizer, ds):
    from transformers import AutoModelForSequenceClassification
    base = AutoModelForSequenceClassification.from_pretrained(train_cfg.bart_name, num_labels=2)
    # right after loading base
    base.config.classifier_dropout = 0.1

    model = BiggerBirdBartForSequenceClassification(base_model=base, cfg=router_cfg)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=64)
    args = make_args("./biggerbird-out")

    # ---- Replace these two instantiations in train_and_eval_biggerbird(...) ----

    anneal_epoch0 = FirstEpochRouterDensity(
        model,
        first_r=0.18, first_max_k=72,   # epoch 0: a bit denser to stabilize training
        rest_r=0.15,  rest_max_k=64,    # epochs 1+: target regime
    )

    anneal_within = RouterAnnealCallback(
        model,
        start_frac=0.20, end_frac=0.15, # r schedule just needs to be ≥ ~0.12 so k hits max_k
        warmup_steps=150,               # spread over ~first 150 steps
        start_max_k=80, end_max_k=64,   # within-epoch warmup -> target k
    )


    ema_cb = EMACallback(model)
    cb = TrainerCallback()
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[cb],
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    print("Training BiggerBird-BART (sparse windows + facility-location globals) ...", flush=True)
    train_res = trainer.train()
    print(train_res.metrics)

    print("Evaluating BiggerBird-BART...", flush=True)
    eval_res = trainer.evaluate()
    print(eval_res)

    if train_cfg.show_debug_meta:
        for n, m in model.model.named_modules():
            if isinstance(m, BiggerBird) and not m.is_decoder and hasattr(m, "_debug_meta"):
                print("[debug_meta]", n, m._debug_meta)
                break
    return eval_res

def train_and_eval_bigbird(tokenizer, ds):
    model = AutoModelForSequenceClassification.from_pretrained(
        train_cfg.bigbird_name,
        num_labels=2,
    )
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=64)
    args = make_args("./bigbird-out")

    base_trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    print("Training BigBird baseline (sparse) ...", flush=True)
    train_res = base_trainer.train()
    print(train_res.metrics)

    print("Evaluating BigBird...", flush=True)
    eval_res = base_trainer.evaluate()
    print(eval_res)
    return eval_res

def main():
    set_seed(train_cfg.seed)

    # Print token-selection configs up front
    summarize_token_selection_configs(router_cfg, train_cfg.bigbird_name)

    # Force BigBird to fixed len ≥ 704 and multiple of 64 (avoid dense fallback in HF BigBird)
    train_cfg.max_length = 896
    min_len = 704
    fixed_len = max(min_len, train_cfg.max_length)
    fixed_len = (fixed_len + 63) // 64 * 64  # align to 64

    # BART tokenizer
    bart_tok = AutoTokenizer.from_pretrained(train_cfg.bart_name, use_fast=True)
    ds_bart = build_imdb_dataset(bart_tok, train_cfg.max_length, fixed_length=fixed_len)

    # BigBird tokenizer (prefer slow)
    bigbird_tok = load_bigbird_tok(train_cfg.bigbird_name)
    ds_bigbird = build_imdb_dataset(bigbird_tok, train_cfg.max_length, fixed_length=fixed_len)

    # Train + eval
    bbird_eval = train_and_eval_biggerbird(bart_tok, ds_bart)
    bigbird_eval = train_and_eval_bigbird(bigbird_tok, ds_bigbird)

    print("\n==== Summary ====")
    print("[BiggerBird-BART] ", bbird_eval)
    print("[BigBird]         ", bigbird_eval)

if __name__ == "__main__":
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    main()
