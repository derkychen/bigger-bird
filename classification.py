# classification.py
# Compare BiggerBird-BART (globals=NEAL QUBO, windows=greedy, teleports)
# vs BigBird on GLUE/SST-2 with small-data regime.

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
    epochs: int = 4
    batch_size: int = 16
    lr: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    max_length: int = 128
    fp16: bool = True

    # data subset for quick runs / fair comparison
    train_samples: int = 2000
    eval_samples: int = 500

    # show one layer's debug meta (BiggerBird)
    show_debug_meta: bool = True


train_cfg = TrainConfig()

# BiggerBird router config (fast & competitive defaults)
router_cfg = RouterConfig(
    fragment_size=16,
    k_per_query=6,
    globals_per_head=4,
    top_u=12,
    proto_count=32,        # try 16–32 if contexts get very long
    teleports_per_head=2,
    teleport_bias_frac=0.5,
    keynorm_exponent=0.0,  # set 0.2–0.4 if key norms correlate with salience
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

def build_dataset_for_tokenizer(tokenizer, max_length: int):
    ds = load_dataset("glue", "sst2")
    ds["train"] = ds["train"].select(range(train_cfg.train_samples))
    ds["validation"] = ds["validation"].select(range(train_cfg.eval_samples))

    def tok_fn(batch):
        return tokenizer(batch["sentence"], truncation=True, max_length=max_length)

    ds = ds.map(tok_fn, batched=True)
    ds = ds.rename_column("label", "labels")
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return ds

def make_args(out_dir: str) -> TrainingArguments:
    return TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=train_cfg.epochs,
        per_device_train_batch_size=train_cfg.batch_size,
        per_device_eval_batch_size=train_cfg.batch_size,
        learning_rate=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
        warmup_ratio=train_cfg.warmup_ratio,
        logging_strategy="steps",
        logging_steps=20,
        eval_strategy="no",                 # evaluate explicitly after training
        save_strategy="no",
        report_to="none",
        fp16=(train_cfg.fp16 and torch.cuda.is_available()),
        gradient_accumulation_steps=1,
        remove_unused_columns=False,
        dataloader_num_workers=0,           # portable (macOS-safe)
        dataloader_pin_memory=torch.cuda.is_available(),
        gradient_checkpointing=True,
        torch_compile=False,
    )


# -------------------------------
# Trainers
# -------------------------------

def train_and_eval_biggerbird(tokenizer, ds):
    model = BiggerBirdBartForSequenceClassification.from_pretrained(train_cfg.bart_name, cfg=router_cfg)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    args = make_args("./qatten-out")

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

    print("Training BiggerBird-BART (globals=NEAL, windows=greedy, teleports) ...", flush=True)
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
    # BigBird fast tokenizer can error on some platforms; use slow tokenizer intentionally.
    # We already passed use_fast=False when creating this tokenizer below.
    model = AutoModelForSequenceClassification.from_pretrained(
        train_cfg.bigbird_name,
        num_labels=2,
    )
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
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

    print("Training BigBird baseline...", flush=True)
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
    ds_bart = build_dataset_for_tokenizer(bart_tok, train_cfg.max_length)

    # BigBird tokenizer: prefer slow to avoid conversion/Tiktoken glitches on macOS
    bigbird_tok = AutoTokenizer.from_pretrained(train_cfg.bigbird_name, use_fast=False)
    ds_bigbird = build_dataset_for_tokenizer(bigbird_tok, train_cfg.max_length)

    # Train + eval
    qa_eval = train_and_eval_biggerbird(bart_tok, ds_bart)
    bb_eval = train_and_eval_bigbird(bigbird_tok, ds_bigbird)

    print("\n==== Summary ====")
    print("[BiggerBird-BART] ", qa_eval)
    print("[BigBird]         ", bb_eval)


if __name__ == "__main__":
    # modest matmul tweak (if CUDA available)
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    main()