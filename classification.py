# classification.py
# Compare BiggerBird-BART (sparse) vs BigBird on IMDB (long sequences)

from dataclasses import dataclass
import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    TrainerCallback,
    set_seed,
)
from transformers.trainer_utils import EvalPrediction

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
