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

    @staticmethod
    def _downsample_queries(qh: torch.Tensor, p: int) -> torch.Tensor:
        """
        qh: [Tq, d] (unit-norm recommended)
        Returns [p', d] where p' = min(p, Tq).
        Simple, fast: evenly-spaced subsample along sequence.
        """
        Tq = qh.size(0)
        if p <= 0 or p >= Tq:
            return qh
        # evenly-spaced indices in [0, Tq-1]
        idx = torch.round(torch.linspace(0, Tq - 1, steps=p, device=qh.device)).long()
        return qh.index_select(dim=0, index=idx)

    def choose_packed(self, key_states: torch.Tensor, q_summary: torch.Tensor,
                      bsz: int, num_heads: int) -> List[torch.Tensor]:
        """
        key_states: [BH, Tk, d] (grouped by head)
        q_summary:  [H,  Tq, d] (unit-normalized queries per head, averaged over batch upstream)
        Returns list length H with LongTensor[g] per head (absolute token indices).
        """
        device = key_states.device
        g = int(self.cfg.globals_per_head)
        U_target = max(1, int(self.cfg.top_u))
        p = max(1, int(self.cfg.proto_count))

        out: List[torch.Tensor] = []

        for h in range(num_heads):
            # --- per-head candidate utilities (same as your current blend) ---
            head_slice = key_states[h::num_heads]               # [B, Tk, d]
            Kraw = head_slice.mean(dim=0)                       # [Tk, d]
            Kbar = F_normalize_safe(Kraw, dim=-1)               # [Tk, d]
            Tk = Kbar.size(0)

            qh = q_summary[h]                                   # [Tq, d] (already unit norm)
            S_full = torch.relu(Kbar @ qh.transpose(0, 1))      # [Tk, Tq]

            mean = S_full.mean(dim=1)                           # [Tk]
            mx   = S_full.max(dim=1).values
            kq   = max(1, int(round(S_full.size(1) * self.cfg.topk_frac)))
            topk_mean = torch.topk(S_full, k=kq, dim=1, largest=True).values.mean(dim=1)
            std  = S_full.std(dim=1)

            u_full = (self.cfg.w_mean * mean
                    + self.cfg.w_max  * mx
                    + self.cfg.w_topk * topk_mean
                    + self.cfg.w_std  * std)

            if self.cfg.keynorm_exponent != 0.0:
                kn = Kraw.norm(dim=-1).clamp_min(1e-6)
                u_full = u_full * (kn / kn.mean()).pow(self.cfg.keynorm_exponent)

            # --- Top-U prefilter ---
            U = int(min(U_target, Tk))
            if U < g:
                U = max(g, min(8, Tk))
            U = max(1, min(U, Tk))
            top_idx = torch.topk(u_full, k=U).indices           # [U]
            Kbar_sub = Kbar.index_select(0, top_idx)            # [U, d]

            # --- Query prototypes & similarities for facility-location ---
            q_proto = self._downsample_queries(qh, p)           # [p', d]
            S = torch.relu(Kbar_sub @ q_proto.transpose(0, 1))  # [U, p']

            # --- Greedy facility-location with exact budget g ---
            p_eff = S.size(1)
            m = torch.zeros(p_eff, device=device, dtype=S.dtype)  # current per-prototype max
            chosen_local: List[int] = []
            blocked = torch.zeros(U, device=device, dtype=torch.bool)

            for _ in range(min(g, U)):
                gains = torch.relu(S - m).sum(dim=1)             # [U]
                gains = gains.masked_fill(blocked, -1e9)
                j = int(gains.argmax().item())
                chosen_local.append(j)
                blocked[j] = True
                m = torch.maximum(m, S[j])

            # Top-off if U < g (rare)
            while len(chosen_local) < g:
                # reuse best by utility u_full among remaining candidates
                remaining = (~blocked).nonzero(as_tuple=False).flatten()
                if remaining.numel() == 0:
                    chosen_local.append(0)
                    break
                # map remaining back to absolute indices then to u_full
                rem_abs = top_idx.index_select(0, remaining)
                rem_scores = u_full.index_select(0, rem_abs)
                best_rem = int(remaining[torch.argmax(rem_scores)].item())
                chosen_local.append(best_rem)
                blocked[best_rem] = True

            chosen_abs = top_idx.index_select(0, torch.tensor(chosen_local, device=device))
            out.append(chosen_abs.to(device=device, dtype=torch.long))

        return out



# ===============================
# Window selector (greedy only)
# ===============================

class WindowSelector:
    def __init__(self, cfg: RouterConfig):
        self.cfg = cfg

    def select_greedy(self, scores_win: torch.Tensor, prior_win: torch.Tensor, keys_win=None) -> torch.Tensor:
        BH, Tq, Fw = scores_win.shape
        k = min(int(self.cfg.k_per_query), Fw)

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
        sel_idx = self.window_selector.select_greedy(scores_win, prior_win, keys_win=K_win)  # [BH,Tq,k]
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
            scored_pairs = scores_win.numel()
            dense_pairs = Q.shape[0] * Q.shape[1] * K.shape[1]
            print(f"[pairs] scored={scored_pairs:,} dense={dense_pairs:,} ratio={scored_pairs / max(1, dense_pairs):.6f}")
            self._logged_pairs = True

        # Apply attention mask
        if attention_mask is not None:
            am_bool = attention_mask if attention_mask.dtype == torch.bool else (attention_mask > -1e8)
            if am_bool.dim() == 3:
                am_bool = am_bool.unsqueeze(1)
            if am_bool.dim() == 2:
                am_bool = am_bool[:, None, None, :]
            am_bh = am_bool.expand(bsz, self.num_heads, tgt_len, src_len).reshape(BH, tgt_len, src_len)
            allowed = torch.gather(am_bh, -1, abs_idx)  # [BH,Tq,M]
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