# biggerbird_bart.py
# BiggerBird for BART with:
#   • Globals: packed QUBO solved by NEAL (default) or Tabu (toggle)
#   • Windows: greedy k-selection w/ cosine diversity
#   • Teleports: biased + uniform mix, per head
#   • Amortization: reuse global selections across adjacent encoder layers
#
# Vectorized QUBO build (NumPy + integer labels) for 3–6× faster BQM construction.

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from transformers.models.bart.modeling_bart import BartAttention
except Exception as e:
    raise ImportError("Transformers with BART is required. Install: pip install transformers") from e

# ===============================
# Config
# ===============================

@dataclass
class RouterConfig:
    # Scope
    patch_encoder_only: bool = True

    # Candidate construction
    fragment_size: int = 16      # F (window size) — clamped to src_len
    k_per_query: int = 6         # k (picked from the window per query)
    globals_per_head: int = 4    # g (per head, per layer)

    # RouterConfig
    r_target_softmax: float = 0.08  # try ~8% on short seqs
    min_k: int = 6                  # never go below this
    max_k: int = 14                 # hard cap keeps O(n)

    # Teleports
    teleports_per_head: int = 2
    teleport_bias_frac: float = 0.5

    # --- Utility shaping (query coverage) ---
    w_mean: float = 1.0
    w_max: float = 0.6
    w_topk: float = 0.4
    w_std: float  = 0.2
    topk_frac: float = 0.2
    keynorm_exponent: float = 0.0

    # Scoring blend / windows
    alpha_pos_prior: float = 0.2
    gamma_diversity: float = 0.2   # window selection diversity

    # Globals (prefilter + prototypes)
    top_u: int = 12                # Top-U prefilter per head before selection
    proto_count: int = 32          # downsample queries to p prototypes per head (<= Tq)

    # Amortization
    share_stride_layers: int = 2

    # Misc / Debug
    random_selection: bool = False   # windows only
    debug_collect: bool = False
    log_once_pairs: bool = True



default_router_config = RouterConfig()


# ===============================
# Runtime cache for amortization
# ===============================

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


# ===============================
# Index builders
# ===============================

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

def positional_prior(indices: torch.Tensor, center_positions: torch.Tensor, tau: float = 8.0) -> torch.Tensor:
    dist = (indices - center_positions.unsqueeze(1)).abs().float()
    return torch.exp(-dist / max(tau, 1e-6))

def F_normalize_safe(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True).clamp_min(eps))


# ===============================
# Globals chooser (QUBO + sampler) — VECTORIZED
# ===============================

# ===============================
# Globals chooser (facility-location greedy)
# ===============================

class GlobalChooser:
    def __init__(self, cfg: RouterConfig):
        self.cfg = cfg

    def _proto_idx(self, Tq: int, p: int, device) -> torch.Tensor:
        if p <= 0 or p >= Tq:  # use all
            return torch.arange(Tq, device=device)
        return torch.round(torch.linspace(0, Tq - 1, steps=p, device=device)).long()

    def choose_packed(self, key_states: torch.Tensor, q_summary: torch.Tensor,
                      bsz: int, num_heads: int) -> List[torch.Tensor]:
        """
        key_states: [BH, Tk, d], q_summary: [H, Tq, d] (L2-normalized)
        Returns list of length H with LongTensor[g] per head.
        """
        device = key_states.device
        H = num_heads
        g = int(self.cfg.globals_per_head)
        U_target = max(1, int(self.cfg.top_u))
        p = max(1, int(self.cfg.proto_count))

        # --- mean keys per head over batch, vectorized over heads ---
        BH, Tk, d = key_states.shape
        Kmean = key_states.view(H, bsz, Tk, d).mean(dim=1)              # [H, Tk, d]
        Kbar  = F_normalize_safe(Kmean, dim=-1)                         # [H, Tk, d]

        # --- query prototypes (same indices for all heads; cheap & stable) ---
        Tq = q_summary.size(1)
        idxp = self._proto_idx(Tq, p, device)                           # [p']
        Qp = q_summary.index_select(1, idxp)                            # [H, p', d]

        # --- similarities S = ReLU(Kbar @ Qp^T), batched over heads ---
        # shape: [H, Tk, p']
        S = torch.relu(torch.einsum('hkd,hpd->hkp', Kbar, Qp))

        # --- utility on prototypes (avoid full Tq pass) ---
        mean = S.mean(dim=-1)                                           # [H, Tk]
        mx   = S.max(dim=-1).values                                     # [H, Tk]
        kq   = max(1, int(round(S.size(-1) * self.cfg.topk_frac)))
        topk_mean = torch.topk(S, k=kq, dim=-1, largest=True).values.mean(dim=-1)  # [H, Tk]
        std  = S.std(dim=-1)

        u_full = (self.cfg.w_mean * mean + self.cfg.w_max * mx
                  + self.cfg.w_topk * topk_mean + self.cfg.w_std * std) # [H, Tk]

        if self.cfg.keynorm_exponent != 0.0:
            kn = Kmean.norm(dim=-1).clamp_min(1e-6)                     # [H, Tk]
            u_full = u_full * (kn / kn.mean(dim=1, keepdim=True)).pow(self.cfg.keynorm_exponent)

        # --- Top-U prefilter per head (batched topk) ---
        U = int(min(U_target, Tk))
        if U < g: U = max(g, min(8, Tk))
        U = max(1, min(U, Tk))
        u_vals, top_idx = torch.topk(u_full, k=U, dim=1)                # [H, U]
        # S_sub: [H, U, p']
        S_sub = torch.gather(S, dim=1, index=top_idx.unsqueeze(-1).expand(-1, -1, S.size(-1)))

        # --- Greedy facility-location (vectorized over heads, loop only over g) ---
        H_range = torch.arange(H, device=device)
        p_eff = S_sub.size(-1)
        m = torch.zeros(H, p_eff, device=device, dtype=S_sub.dtype)     # [H, p']
        blocked = torch.zeros(H, U, device=device, dtype=torch.bool)    # [H, U]
        chosen_local = torch.zeros(H, g, device=device, dtype=torch.long)

        steps = min(g, U)
        for r in range(steps):
            gains = torch.relu(S_sub - m.unsqueeze(1)).sum(dim=-1)      # [H, U]
            gains = gains.masked_fill(blocked, -1e9)
            j = gains.argmax(dim=1)                                     # [H]
            chosen_local[:, r] = j
            blocked[H_range, j] = True
            m = torch.maximum(m, S_sub[H_range, j, :])                  # update per head

        # Top-off if needed (rare when U < g)
        if steps < g:
            # pick remaining by utility among available
            remaining_mask = ~blocked
            # set gains to utility among remaining; otherwise -inf
            rem_util = torch.where(remaining_mask, u_vals, torch.full_like(u_vals, -1e9))
            fill = torch.topk(rem_util, k=(g - steps), dim=1).indices    # [H, g-steps]
            chosen_local[:, steps:] = fill

        # Map back to absolute token indices per head: [H, g]
        chosen_abs = torch.gather(top_idx, dim=1, index=chosen_local)

        # Return list of tensors per head
        return [chosen_abs[h].to(device=device, dtype=torch.long) for h in range(H)]




# ===============================
# Window selector (greedy only)
# ===============================

class WindowSelector:
    def __init__(self, cfg: RouterConfig):
        self.cfg = cfg

    def select_greedy(self, scores_win: torch.Tensor, prior_win: torch.Tensor, keys_win=None, k_override=None) -> torch.Tensor:
        BH, Tq, Fw = scores_win.shape
        k = int(self.cfg.k_per_query)
        k = min(int(k_override if k_override is not None else k), Fw)

        if self.cfg.random_selection:
            return torch.randint(low=0, high=Fw, size=(BH, Tq, k),
                                 device=scores_win.device, dtype=torch.long)

        blended = scores_win + self.cfg.alpha_pos_prior * (
            prior_win if prior_win.dim() == 3 else prior_win.unsqueeze(0)
        )

        sel_idx = torch.zeros(BH, Tq, k, dtype=torch.long, device=scores_win.device)
        tmp = blended.clone()

        kv = None
        if keys_win is not None:
            kv = F_normalize_safe(keys_win, dim=-1)  # [BH, Tq, F, d]

        for r in range(k):
            j = tmp.argmax(dim=-1)  # [BH, Tq]
            sel_idx[:, :, r] = j
            if kv is not None and r < k - 1:
                j_exp = j.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, kv.size(-1))
                sel_vec = torch.gather(kv, -2, j_exp).squeeze(-2)  # [BH,Tq,d]
                cos = (kv * sel_vec.unsqueeze(-2)).sum(dim=-1).clamp(min=0)  # [BH,Tq,F]
                tmp = tmp - self.cfg.gamma_diversity * cos
            tmp.scatter_(-1, j.unsqueeze(-1), -1e9)

        return sel_idx


# ===============================
# Attention patch
# ===============================

class BiggerBird(BartAttention):
    def __init__(self, base_attn: BartAttention, cfg: RouterConfig, runtime: RouterRuntime):
        drop_p = base_attn.dropout.p if isinstance(base_attn.dropout, nn.Dropout) else float(base_attn.dropout)
        super().__init__(
            embed_dim=base_attn.embed_dim,
            num_heads=base_attn.num_heads,
            dropout=drop_p,
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

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

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

        # Window indices + prior
        Fw = min(int(self.cfg.fragment_size), src_len)  # clamp
        if is_cross_attention:
            idx_win = build_indices_cross(tgt_len, src_len, Fw, device)
            centers = torch.round(torch.linspace(0, src_len - 1, steps=tgt_len, device=device)).long()
        else:
            idx_win = build_indices_decoder_causal(tgt_len, Fw, device) if self.is_decoder \
                      else build_indices_encoder(tgt_len, Fw, device)
            centers = torch.arange(tgt_len, device=device).long().clamp(0, src_len - 1)

        idx_expanded = idx_win.view(1, tgt_len, Fw, 1).expand(BH, tgt_len, Fw, self.head_dim)
        K_win = torch.gather(K.unsqueeze(1).expand(BH, tgt_len, src_len, self.head_dim), 2, idx_expanded)
        scores_win = torch.einsum('btd,btfd->btf', Q, K_win)  # [BH,Tq,F]
        prior_win = positional_prior(idx_win, centers, tau=max(Fw / 4, 1.0))  # [Tq,F]

        if self.cfg.log_once_pairs and not self._printed_cfg:
            print(f"[QuAttn] heads={self.num_heads} F={Fw} "
                f"k={self.cfg.k_per_query} g={self.cfg.globals_per_head} t={self.cfg.teleports_per_head} "
                f"backends(win=greedy, glob=facility-location) "
                f"is_decoder={self.is_decoder}")
            self._printed_cfg = True


        # ---- window selection (greedy) ----
        Tk = K.shape[1]
        g = int(self.cfg.globals_per_head)
        t = max(0, int(self.cfg.teleports_per_head))

        # Desired M for short sequences only; capped to keep O(n)
        M_desired = max(g + t + 1, int(self.cfg.r_target_softmax * Tk))
        k_here = M_desired - (g + t)
        k_here = max(self.cfg.min_k, min(self.cfg.max_k, k_here))
        k_here = min(k_here, Fw)  # cannot exceed window size

        sel_idx = self.window_selector.select_greedy(scores_win, prior_win, keys_win=K_win, k_override=k_here)
        k = sel_idx.size(-1)

        # Absolute indices for windows
        abs_idx_win = torch.gather(idx_win.unsqueeze(0).expand(BH, -1, -1), -1, sel_idx)  # [BH,Tq,k]

        # ---- globals per head (QUBO; amortized) ----
        H = self.num_heads
        Qh = Q.view(H, bsz, tgt_len, self.head_dim)                # [H,B,Tq,d]
        q_summary = F_normalize_safe(Qh.mean(dim=1), dim=-1)       # [H,Tq,d]

        globals_per_head = self.runtime.maybe_get_shared_globals(getattr(self, "layer_idx", None), src_len)
        if globals_per_head is None:
            globals_per_head = self.global_chooser.choose_packed(
                key_states=K, q_summary=q_summary, bsz=bsz, num_heads=H
            )
            self.runtime.store_globals(getattr(self, "layer_idx", None), src_len, globals_per_head)

        # ---- teleports per head (biased + uniform), force exactly t ----
        t = max(0, int(self.cfg.teleports_per_head))
        teleport_bias = float(self.cfg.teleport_bias_frac)
        tele_list: List[torch.Tensor] = []
        if t > 0:
            tele_biased = int(round(t * teleport_bias))
            tele_uniform = t - tele_biased
            for h in range(H):
                head_slice = K[h::H]                                   # [B,Tk,d]
                Kbar = F.normalize(head_slice.mean(dim=0), dim=-1)      # [Tk,d]
                qh = q_summary[h]                                       # [Tq,d]
                u_long = torch.relu(torch.matmul(Kbar, qh.transpose(0, 1))).mean(dim=1)  # [Tk]
                picks = []
                if tele_biased > 0 and u_long.numel() > 0:
                    k_top = min(tele_biased, u_long.numel())
                    picks.append(torch.topk(u_long, k=k_top).indices)
                if tele_uniform > 0:
                    picks.append(torch.randint(low=0, high=src_len, size=(tele_uniform,), device=device))
                tele = torch.unique(torch.cat(picks)) if picks else torch.empty(0, dtype=torch.long, device=device)
                # top-off to exactly t
                attempts = 0
                while tele.numel() < t and attempts < 3:
                    need = t - tele.numel()
                    extra = torch.randint(low=0, high=src_len, size=(need,), device=device)
                    tele = torch.unique(torch.cat([tele, extra]))
                    attempts += 1
                if tele.numel() > t:
                    tele = tele[:t]
                if tele.numel() < t:
                    pad = (tele[:1].repeat(t - tele.numel())) if tele.numel() > 0 \
                          else torch.zeros(t - tele.numel(), dtype=torch.long, device=device)
                    tele = torch.cat([tele, pad], dim=0)
                tele_list.append(tele.to(device))
        else:
            tele_list = [torch.empty(0, dtype=torch.long, device=device) for _ in range(H)]

        # ---- Build final absolute index tensor per head: [B,Tq,k + g + t] ----
        abs_idx_list = []
        g = int(self.cfg.globals_per_head)
        for h in range(H):
            y = globals_per_head[h].to(device=device, dtype=torch.long)
            # force exactly g
            if y.numel() < g:
                pad = (y[:1].repeat(g - y.numel())) if y.numel() > 0 \
                      else torch.zeros(g - y.numel(), dtype=torch.long, device=device)
                y = torch.cat([y, pad], dim=0)
            elif y.numel() > g:
                y = y[:g]
            y_exp = y.view(1, 1, g).expand(bsz, tgt_len, g)

            tele = tele_list[h]
            if tele.numel() > 0:
                tele_exp = tele.view(1, 1, t).expand(bsz, tgt_len, t)
                parts = [abs_idx_win[h*bsz:(h+1)*bsz], y_exp, tele_exp]
            else:
                parts = [abs_idx_win[h*bsz:(h+1)*bsz], y_exp]
            abs_idx_h = torch.cat(parts, dim=-1)  # [B,Tq,M]
            abs_idx_list.append(abs_idx_h)

        # all heads must now share the same M
        M = abs_idx_list[0].size(-1)
        for i in range(1, len(abs_idx_list)):
            if abs_idx_list[i].size(-1) != M:
                diff = M - abs_idx_list[i].size(-1)
                pad = abs_idx_list[i][..., :1].expand(-1, -1, diff)
                abs_idx_list[i] = torch.cat([abs_idx_list[i], pad], dim=-1)

        abs_idx = torch.cat(abs_idx_list, dim=0)  # [BH,Tq,M]

        # ---- Final attention over selected tokens ----
        idx4 = abs_idx.view(BH, tgt_len, M, 1).expand(BH, tgt_len, M, self.head_dim)
        K_sel = torch.gather(K.unsqueeze(1).expand(BH, tgt_len, src_len, self.head_dim), 2, idx4)
        V_sel = torch.gather(V.unsqueeze(1).expand(BH, tgt_len, src_len, self.head_dim), 2, idx4)
        scores_sel = torch.einsum('btd,btmd->btm', Q, K_sel)

        # log pairs (cost proxy) once
        if self.cfg.log_once_pairs and not self._logged_pairs:
            # --- Softmax-comparison ratio (sparse vs full) ---
            BH = bsz * self.num_heads
            Tk = K.shape[1]
            M_now = abs_idx.size(-1)

            # logits "sent to softmax"
            comps_sparse = BH * tgt_len * M_now                   # sparse
            comps_full_unmasked = BH * tgt_len * Tk               # full (no masking)

            ratio_softmax = comps_sparse / max(1, comps_full_unmasked)

            # Optional: account for padding/causal masks in the "full" baseline
            if attention_mask is not None:
                am_bool = attention_mask if attention_mask.dtype == torch.bool else (attention_mask > -1e8)
                if am_bool.dim() == 3:  # [B, Tq, Tk] -> [B, 1, Tq, Tk]
                    am_bool = am_bool.unsqueeze(1)
                if am_bool.dim() == 2:  # [B, Tk] -> [B, 1, 1, Tk]
                    am_bool = am_bool[:, None, None, :]
                am_bh_full = am_bool.expand(bsz, self.num_heads, tgt_len, Tk).reshape(BH, tgt_len, Tk)
                comps_full_masked = int(am_bh_full.sum().item())  # how many logits are actually "active"
            else:
                comps_full_masked = comps_full_unmasked

            ratio_softmax_eff = comps_sparse / max(1, comps_full_masked)

            print(
                f"[softmax] sparse={comps_sparse:,}  |  full(unmasked)={comps_full_unmasked:,} "
                f"ratio={ratio_softmax:.4f}  |  full(masked)={comps_full_masked:,} eff_ratio={ratio_softmax_eff:.4f}"
)
            self._logged_pairs = True

        # Apply attention mask (memory-safe)
        if attention_mask is not None:
            am_bool = attention_mask if attention_mask.dtype == torch.bool else (attention_mask > -1e8)
            # normalize shapes: [B, 1, 1, Tk] or [B, 1, Tq, Tk]
            if am_bool.dim() == 3:
                am_bool = am_bool.unsqueeze(1)
            if am_bool.dim() == 2:
                am_bool = am_bool[:, None, None, :]  # [B,1,1,Tk]
            # reshape indices by head to avoid BH×Tq×Tk expansion
            abs_idx_hb = abs_idx.view(self.num_heads, bsz, tgt_len, M)  # [H,B,Tq,M]
            allowed_chunks = []
            for h in range(self.num_heads):
                # broadcast over head dimension lazily (no materialization)
                am_small = am_bool.expand(bsz, 1, tgt_len, src_len)     # view
                allowed_h = torch.gather(am_small, -1, abs_idx_hb[h].unsqueeze(1)).squeeze(1)  # [B,Tq,M]
                allowed_chunks.append(allowed_h)
            allowed = torch.cat(allowed_chunks, dim=0)  # [BH,Tq,M]
            scores_sel = scores_sel.masked_fill(~allowed, torch.finfo(scores_sel.dtype).min)


        # Softmax + dropout + output
        attn_probs = F.softmax(scores_sel, dim=-1)
        drop_p = self.dropout.p if isinstance(self.dropout, nn.Dropout) else float(self.dropout)
        attn_probs = F.dropout(attn_probs, p=drop_p, training=self.training)

        attn_output = torch.einsum('btm,btmd->btd', attn_probs, V_sel)
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim) \
                                 .transpose(1, 2).reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        present_key_value = (key_states_full, value_states_full) if use_cache else None
        attn_weights_reshaped = attn_probs.view(bsz, self.num_heads, tgt_len, M) if output_attentions else None

        # Optional debug stats
        if self.cfg.debug_collect:
            mean_dist = 0.0
            if k > 0:
                local_abs = abs_idx[:, :, :k]
                centers = torch.arange(tgt_len, device=device).long().clamp(0, src_len - 1) if not is_cross_attention \
                          else torch.round(torch.linspace(0, src_len - 1, steps=tgt_len, device=device)).long()
                mean_dist = (local_abs - centers.view(1, -1, 1)).abs().float().mean().item()
            jacc = 0.0
            if self.cfg.globals_per_head > 0:
                gl_sets = [set(y.tolist()) for y in globals_per_head]
                if len(gl_sets) >= 2:
                    inter = set.intersection(*gl_sets); union = set.union(*gl_sets)
                    jacc = len(inter) / max(1, len(union))
            self._debug_meta = {"M": M, "mean_local_distance": mean_dist,
                                "global_jaccard": jacc,
                                "globals_h0": (globals_per_head[0].tolist() if self.cfg.globals_per_head > 0 else [])}

        return (attn_output, attn_weights_reshaped, present_key_value) if use_cache \
               else (attn_output, attn_weights_reshaped)


def patch_bart_with_biggerbird(model: nn.Module, cfg: RouterConfig) -> RouterRuntime:
    # Try to detect total heads from encoder self-attn
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


# ===============================
# HF Trainer wrapper
# ===============================

class BiggerBirdBartForSequenceClassification(nn.Module):
    def __init__(self, base_model: nn.Module, cfg: RouterConfig = default_router_config):
        super().__init__()
        self.model = base_model
        self.runtime = patch_bart_with_biggerbird(self.model, cfg)

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
        return_dict=None,
        **kwargs,
    ):
        self.runtime.begin_forward()
        try:
            kwargs.pop("token_type_ids", None)
            kwargs.pop("num_items_in_batch", None)
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
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

# classification.py
# Compare BiggerBird-BART (sparse) vs BigBird on IMDB (long sequences)

from dataclasses import dataclass
import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    TrainerCallback,
    set_seed,
)
from transformers.trainer_utils import EvalPrediction
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


from biggerbird_bart import BiggerBirdBartForSequenceClassification, RouterConfig

# -------------------------------
# Train / Router configs
# -------------------------------

@dataclass
class TrainConfig:
    bart_name: str = "facebook/bart-base"
    bigbird_name: str = "google/bigbird-roberta-base"

    seed: int = 42
    epochs: int = 3

    # For 1k-token sequences, keep per-device batch small; use grad accumulation
    per_device_train_bs: int = 2
    per_device_eval_bs: int = 2
    grad_accum_steps: int = 8          # effective batch ~= 16

    lr: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    max_length: int = 1024             # long enough to trigger sparse benefits
    use_fp16_if_cuda: bool = True
    use_bf16_if_cuda: bool = True      # use BF16 on Ampere+ if available

    # IMDB: 25k/25k; you can subselect for quick runs
    train_samples: int = 2000
    eval_samples: int = 1000

    show_debug_meta: bool = True

train_cfg = TrainConfig()

# BiggerBird router config (long-context defaults)
router_cfg = RouterConfig(
    fragment_size=64,        # F (window)
    k_per_query=8,           # k from F; << BigBird's ~448 local tokens
    globals_per_head=4,      # g
    top_u=16,                # expand to 16 candidates before greedy
    proto_count=32,          # query prototypes for facility-location
    teleports_per_head=2,
    teleport_bias_frac=0.5,
    keynorm_exponent=0.0,
)

# -------------------------------
# Utilities
# -------------------------------

def compute_metrics(eval_pred):
    if isinstance(eval_pred, EvalPrediction):
        logits, labels = eval_pred.predictions, eval_pred.label_ids
    else:
        logits, labels = eval_pred
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds), "f1": f1_score(labels, preds)}

def build_imdb_dataset(tokenizer, max_length: int):
    ds = load_dataset("imdb")
    if train_cfg.train_samples:
        ds["train"] = ds["train"].shuffle(seed=train_cfg.seed).select(range(train_cfg.train_samples))
    if train_cfg.eval_samples:
        ds["test"] = ds["test"].shuffle(seed=train_cfg.seed).select(range(train_cfg.eval_samples))

    def tok_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

    ds = ds.map(tok_fn, batched=True, remove_columns=["text"])
    ds = ds.rename_column("label", "labels")
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return {"train": ds["train"], "validation": ds["test"]}

def device_flags():
    use_cuda = torch.cuda.is_available()
    use_mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    fp16 = bool(train_cfg.use_fp16_if_cuda and use_cuda)
    bf16 = bool(train_cfg.use_bf16_if_cuda and use_cuda and torch.cuda.is_bf16_supported())
    # On MPS, keep fp16/bf16 off; kernels are still finicky with long seqs
    if use_mps:
        fp16 = False
        bf16 = False
    torch_compile = bool(use_cuda)  # generally safe; disable on MPS
    pin_mem = bool(use_cuda)        # keep False on MPS/CPU
    return fp16, bf16, torch_compile, pin_mem

def make_args(out_dir: str) -> TrainingArguments:
    fp16, bf16, torch_compile, pin_mem = device_flags()
    return TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=train_cfg.epochs,
        per_device_train_batch_size=train_cfg.per_device_train_bs,
        per_device_eval_batch_size=train_cfg.per_device_eval_bs,
        gradient_accumulation_steps=train_cfg.grad_accum_steps,
        learning_rate=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
        warmup_ratio=train_cfg.warmup_ratio,
        logging_strategy="steps",
        logging_steps=20,
        eval_strategy="no",                 # evaluate explicitly after training
        save_strategy="no",
        report_to="none",
        fp16=fp16,
        bf16=bf16,
        remove_unused_columns=False,        # we pass all columns already
        dataloader_num_workers=2,           # low for macOS
        dataloader_pin_memory=pin_mem,
        gradient_checkpointing=True,
        torch_compile=torch_compile,
        # padding multiple helps Flash/blocks; harmless otherwise
        optim="adamw_torch",
    )

# Robust BigBird tokenizer (avoid slow->fast conversion path)
def load_bigbird_tok(model_name: str):
    try:
        from transformers import BigBirdTokenizer
        return BigBirdTokenizer.from_pretrained(model_name)  # slow, stable
    except Exception as e_slow:
        print(f"[BigBird] slow tokenizer failed: {e_slow}\nTrying fast tokenizer...", flush=True)
        return AutoTokenizer.from_pretrained(model_name, use_fast=True)

# -------------------------------
# Trainers
# -------------------------------

def train_and_eval_biggerbird(tokenizer, ds):
    model = BiggerBirdBartForSequenceClassification.from_pretrained(train_cfg.bart_name, cfg=router_cfg)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=64)
    args = make_args("./biggerbird-out")

    callback = TrainerCallback()
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[callback],
    )

    print("Training BiggerBird-BART (sparse windows + facility-location globals) ...", flush=True)
    train_res = trainer.train()
    print(train_res.metrics)

    print("Evaluating BiggerBird-BART...", flush=True)
    eval_res = trainer.evaluate()
    print(eval_res)

    # Optional: peek one encoder layer's debug meta
    if train_cfg.show_debug_meta:
        from biggerbird_bart import BiggerBird
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
    )

    print("Training BigBird baseline (sparse) ...", flush=True)
    train_res = base_trainer.train()
    print(train_res.metrics)

    print("Evaluating BigBird...", flush=True)
    eval_res = base_trainer.evaluate()
    print(eval_res)
    return eval_res

# -------------------------------
# Main
# -------------------------------

def main():
    set_seed(train_cfg.seed)

    # BART tokenizer (fast OK)
    bart_tok = AutoTokenizer.from_pretrained(train_cfg.bart_name, use_fast=True)
    ds_bart = build_imdb_dataset(bart_tok, train_cfg.max_length)

    # BigBird tokenizer: prefer slow to avoid Tiktoken conversion glitches
    bigbird_tok = load_bigbird_tok(train_cfg.bigbird_name)
    ds_bigbird = build_imdb_dataset(bigbird_tok, train_cfg.max_length)

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
