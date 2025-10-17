#!/usr/bin/env python3
"""
Self-contained fine-tuning script extracted from 02_classification.ipynb ("Fine-Tuning Transformers").

It:
- Loads the 'emotion' dataset (default) from the Hugging Face Hub
- Tokenizes with AutoTokenizer
- Fine-tunes a sequence classification head with Trainer
- Evaluates with accuracy and weighted F1
- Saves the model, tokenizer, and optionally pushes to the Hub

Example:
  python fine_tune_encoder_for_emotion_classification.py --epochs 2 --batch_size 64 --lr 2e-5 --output_dir runs/distilbert-emotion

With evaluation each epoch and push to Hub (after `huggingface-cli login`):
  python fine_tune_encoder_for_emotion_classification.py --epochs 2 --batch_size 64 --lr 2e-5 --eval_strategy epoch --push_to_hub --hub_model_id YOUR_USERNAME/distilbert-base-uncased-finetuned-emotion
"""

import argparse
import os
import sys
import random
import json
from typing import Dict, Any, Optional

import numpy as np

from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a Transformer for emotion classification.")
    parser.add_argument("--model_ckpt", type=str, default="distilbert-base-uncased",
                        help="Pretrained model checkpoint (HF Hub id or local path).")
    parser.add_argument("--dataset", type=str, default="emotion",
                        help="Dataset repository on the HF Hub (default: emotion).")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to store checkpoints and final model. Defaults to {model_ckpt}-finetuned-emotion.")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Per-device train/eval batch size.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--warmup_ratio", type=float, default=0.0, help="Warmup ratio for the LR scheduler.")
    parser.add_argument("--eval_strategy", type=str, default="epoch", choices=["no", "steps", "epoch"],
                        help="Evaluation strategy.")
    parser.add_argument("--logging_steps", type=int, default=0,
                        help="Log every N steps (0 => auto: len(train)//batch_size).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--fp16", action="store_true", help="Enable mixed precision (CUDA only).")
    parser.add_argument("--bf16", action="store_true", help="Enable bf16 mixed precision on supported hardware.")
    parser.add_argument("--push_to_hub", action="store_true", help="Push the fine-tuned model to the HF Hub.")
    parser.add_argument("--hub_model_id", type=str, default=None, help="Repository name to push to on the Hub.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume training from.")
    parser.add_argument("--train", dest="train", action="store_true", help="Run training (default).")
    parser.add_argument("--no-train", dest="train", action="store_false", help="Skip training.")
    parser.add_argument("--eval", dest="do_eval", action="store_true", help="Run evaluation after training (default).")
    parser.add_argument("--no-eval", dest="do_eval", action="store_false", help="Skip evaluation.")
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="For quick tests, limit number of training samples.")
    parser.add_argument("--max_eval_samples", type=int, default=None,
                        help="For quick tests, limit number of eval samples.")

    parser.set_defaults(train=True, do_eval=True)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"{args.model_ckpt}-finetuned-emotion".replace("/", "-")

    if args.fp16 and not torch.cuda.is_available():
        print("Warning: --fp16 requested but CUDA is not available; disabling fp16.", file=sys.stderr)
        args.fp16 = False

    # Auto-select mixed precision on GPU when user did not request either
    if torch.cuda.is_available():
        if not args.fp16 and not args.bf16:
            try:
                if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                    args.bf16 = True
                else:
                    args.fp16 = True
            except Exception:
                args.fp16 = True
    else:
        args.fp16 = False
        args.bf16 = False

    return args


def compute_metrics_fn(labels: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}


def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return compute_metrics_fn(labels, preds)


def maybe_subset(ds, max_samples: Optional[int]):
    if max_samples is None:
        return ds
    max_n = min(len(ds), int(max_samples))
    return ds.select(range(max_n))


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Reproducibility
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    precision = "bf16" if args.bf16 else ("fp16" if args.fp16 else "fp32")
    print(f"Device: {device} | Precision: {precision}")

    # 1) Load dataset
    print(f"Loading dataset: {args.dataset}")
    emotions = load_dataset(args.dataset)

    # Ensure splits exist
    for split in ["train", "validation"]:
        if split not in emotions:
            raise ValueError(f"Expected split '{split}' not found in dataset '{args.dataset}'.")

    # Label metadata (avoid hard-coding)
    label_feature = emotions["train"].features["label"]
    num_labels = label_feature.num_classes
    label_names = list(label_feature.names)
    print(f"Detected {num_labels} labels: {label_names}")

    # Persist label names for downstream use
    with open(os.path.join(args.output_dir, "label_names.json"), "w", encoding="utf-8") as f:
        json.dump(label_names, f, indent=2)

    # 2) Tokenizer and tokenization
    print(f"Loading tokenizer: {args.model_ckpt}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_ckpt, use_fast=True)

    def tokenize(batch: Dict[str, Any]) -> Dict[str, Any]:
        return tokenizer(batch["text"], padding=False, truncation=True)

    print("Tokenizing dataset ...")
    encoded = emotions.map(tokenize, batched=True, remove_columns=[])

    # Optional subsetting for quick tests
    if args.max_train_samples is not None:
        encoded["train"] = maybe_subset(encoded["train"], args.max_train_samples)
    if args.max_eval_samples is not None:
        encoded["validation"] = maybe_subset(encoded["validation"], args.max_eval_samples)

    # Trainer will handle device placement; use dynamic padding via data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 3) Model
    print(f"Loading classification model: {args.model_ckpt}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_ckpt,
        num_labels=num_labels,
    )

    # 4) Training args and Trainer
    auto_logging_steps = max(1, len(encoded["train"]) // max(1, args.batch_size))
    logging_steps = args.logging_steps if args.logging_steps > 0 else auto_logging_steps

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy=args.eval_strategy,
        logging_steps=logging_steps,
        save_strategy="epoch" if args.eval_strategy == "epoch" else "steps",
        load_best_model_at_end=(args.eval_strategy != "no"),
        metric_for_best_model="f1",
        greater_is_better=True,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        log_level="error",
        seed=args.seed,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to="none",  # avoid requiring external loggers
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=encoded["train"],
        eval_dataset=encoded.get("validation", None),
        compute_metrics=compute_metrics if args.do_eval else None,
    )

    # 5) Train
    if args.train:
        print("Starting training ...")
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # 6) Evaluate
    if args.do_eval and encoded.get("validation", None) is not None:
        print("Evaluating on validation set ...")
        metrics = trainer.evaluate()
        print("Validation metrics:", json.dumps(metrics, indent=2))

    # 7) Save and (optionally) push
    print(f"Saving model to: {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.push_to_hub:
        print("Pushing model to the Hugging Face Hub ...")
        trainer.push_to_hub(commit_message="Training completed!")

    print("Done.")


if __name__ == "__main__":
    main()
