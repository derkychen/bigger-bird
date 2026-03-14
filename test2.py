# Visualize full attention patterns (post-softmax) for an IMDB review
# ---------------------------------------------------------------
# pip install torch transformers matplotlib numpy

import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -----------------------
# Config
# -----------------------
MODEL_ID = "textattack/bert-base-uncased-imdb"  # BERT fine-tuned on IMDB
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 256            # shorten if you want bigger token labels on plots
SHOW_TOKENS = 80         # how many tokens to label on axes (keeps ticks readable)
TEXT = (
    "I went into this film with low expectations, but it completely won me over. "
    "The performances are heartfelt, the pacing is tight, and the ending really lands. "
    "Sure, it's not perfect, yet I left the theater smiling."
)

# -----------------------
# Load model + tokenizer
# -----------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID, output_attentions=True
).to(DEVICE)
model.eval()

# -----------------------
# Encode & run forward
# -----------------------
enc = tokenizer(
    TEXT,
    return_tensors="pt",
    truncation=True,
    max_length=MAX_LEN,
    padding=False
)
enc = {k: v.to(DEVICE) for k, v in enc.items()}
with torch.no_grad():
    out = model(**enc)

# out.attentions is a tuple length = num_layers
# each item: [batch, heads, seq_len, seq_len], already softmaxed
attentions = [a[0].cpu().numpy() for a in out.attentions]  # strip batch -> [heads, S, S]
num_layers = len(attentions)
num_heads = attentions[0].shape[0]
tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0].cpu().tolist())
seq_len = len(tokens)

# -----------------------
# Utilities
# -----------------------
def _crop_for_labels(tokens, show_n=SHOW_TOKENS):
    """Return indices and labels cropped to first show_n tokens (or all if shorter)."""
    n = min(len(tokens), show_n)
    idx = np.arange(n)
    labels = tokens[:n]
    # make WordPiece tokens more legible
    labels = [t.replace("##", "▁") for t in labels]
    return idx, labels

def plot_attention_heatmap(attn_matrix, tokens, title="Attention", vmin=0.0, vmax=1.0):
    """
    attn_matrix: [S, S] post-softmax weights (rows = queries, cols = keys)
    tokens: list of token strings
    """
    S = attn_matrix.shape[0]
    # optionally crop for readability
    n = min(S, SHOW_TOKENS)
    A = attn_matrix[:n, :n]
    x_idx, x_labels = _crop_for_labels(tokens, n)
    y_idx, y_labels = x_idx, x_labels

    plt.figure(figsize=(max(6, n * 0.22), max(5, n * 0.22)))
    im = plt.imshow(A, aspect="auto", vmin=vmin, vmax=vmax)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(title)
    plt.xlabel("Keys (attended to)")
    plt.ylabel("Queries (attending)")
    # tick every k tokens for readability
    k = max(1, n // 20)
    plt.xticks(x_idx[::k], x_labels[::k], rotation=90)
    plt.yticks(y_idx[::k], y_labels[::k])
    plt.tight_layout()
    plt.show()

def head_entropy(attn): 
    """
    attn: [S, S] for a single head (rows sum to 1).
    Returns mean per-query Shannon entropy (base e).
    Lower entropy -> sharper, more 'decisive' attention.
    """
    # numerical stability
    eps = 1e-12
    p = np.clip(attn, eps, 1.0)
    H = -np.sum(p * np.log(p), axis=-1)  # [S]
    return float(np.mean(H))

def rank_heads_by_sharpness(attentions):
    """
    attentions: list over layers, each [H, S, S]
    Returns a list of (layer, head, mean_entropy, mean_max_attn) sorted by entropy asc.
    """
    ranks = []
    for l, layer in enumerate(attentions):
        for h in range(layer.shape[0]):
            attn = layer[h]
            H = head_entropy(attn)
            mean_max = float(np.mean(np.max(attn, axis=-1)))
            ranks.append((l, h, H, mean_max))
    ranks.sort(key=lambda x: (x[2], -x[3]))  # low entropy first; tie-break by higher max
    return ranks

def aggregate_heads(layer_attn, how="mean"):
    """
    layer_attn: [H, S, S]
    how: 'mean' or 'max' or 'weighted' (by per-query max)
    """
    if how == "mean":
        return np.mean(layer_attn, axis=0)
    if how == "max":
        return np.max(layer_attn, axis=0)
    if how == "weighted":
        # weight heads by their average per-query max attention
        weights = []
        for h in range(layer_attn.shape[0]):
            weights.append(np.mean(np.max(layer_attn[h], axis=-1)))
        w = np.array(weights) / (np.sum(weights) + 1e-12)
        return np.tensordot(w, layer_attn, axes=(0, 0))
    raise ValueError("how must be 'mean', 'max', or 'weighted'")

def attention_rollout(attentions, head_aggr="mean"):
    """
    Attention rollout (per Abnar & Zuidema 2020): multiply (I + A) across layers.
    attentions: list length L, each [H, S, S] (post-softmax)
    head_aggr: 'mean'/'max'/'weighted' aggregation across heads per layer.
    Returns: [S, S] rollout matrix highlighting indirect influence through layers.
    """
    S = attentions[0].shape[-1]
    roll = np.eye(S)
    for l in range(len(attentions)):
        A_l = aggregate_heads(attentions[l], how=head_aggr)
        # Normalize rows (should already be softmaxed, but residual will break exact sums)
        A_l = A_l / (A_l.sum(axis=-1, keepdims=True) + 1e-12)
        roll = roll @ (A_l + np.eye(S))
        # Normalize after multiplication to keep scales comparable
        roll = roll / (roll.sum(axis=-1, keepdims=True) + 1e-12)
    return roll

# -----------------------
# Inspect predictions (optional)
# -----------------------
logits = out.logits[0].cpu()
probs = torch.softmax(logits, dim=-1).numpy()
label_id = int(np.argmax(probs))
label = model.config.id2label[label_id] if hasattr(model.config, "id2label") else str(label_id)
print(f"Predicted label: {label}  (probs={probs})")
print(f"Sequence length: {seq_len} tokens")

# -----------------------
# Rank heads to find 'important' ones
# -----------------------
ranks = rank_heads_by_sharpness(attentions)
print("\nTop 5 sharpest heads (low entropy):")
for i in range(min(5, len(ranks))):
    l, h, H, mean_max = ranks[i]
    print(f"  Layer {l:02d}, Head {h:02d} | mean entropy={H:.3f}, mean max attn={mean_max:.3f}")

# -----------------------
# Visualize some patterns
# -----------------------
# 1) Sharpest head found above
best_layer, best_head, _, _ = ranks[0]
title = f"Layer {best_layer} • Head {best_head} (post-softmax)"
plot_attention_heatmap(attentions[best_layer][best_head], tokens, title=title)

# 2) Per-layer aggregated attention (mean across heads)
layer_to_view = best_layer  # you can change this, e.g., last layer: num_layers - 1
A_mean = aggregate_heads(attentions[layer_to_view], how="mean")
plot_attention_heatmap(A_mean, tokens, title=f"Layer {layer_to_view} • mean over {num_heads} heads")

# 3) Across-layer rollout (captures indirect paths)
A_roll = attention_rollout(attentions, head_aggr="mean")
plot_attention_heatmap(A_roll, tokens, title="Attention Rollout across all layers (mean head agg)")

# 4) Optional: visualize how much each token is 'attended to' on average
def token_importance_from_keys(A):
    # column-wise mean attention: on average, how much attention do queries give to each key
    return np.mean(A, axis=0)  # [S]

imp = token_importance_from_keys(A_mean)
topk = 15
idx_sorted = np.argsort(-imp)[:topk]
print("\nTop tokens by mean received attention (layer-mean):")
for i in idx_sorted:
    print(f"{i:>3}  {tokens[i]:<12}  {imp[i]:.4f}")

# If you want to focus attention on a single query token (row), uncomment:
# q = 0  # e.g., [CLS]
# row = attentions[best_layer][best_head][q]  # shape [S]
# plt.figure(figsize=(10, 3))
# plt.bar(np.arange(len(row))[:SHOW_TOKENS], row[:SHOW_TOKENS])
# plt.title(f"Attention distribution for query token #{q} ({tokens[q]})")
# plt.xlabel("Keys")
# plt.ylabel("Weight (softmax)")
# plt.tight_layout()
# plt.show()

