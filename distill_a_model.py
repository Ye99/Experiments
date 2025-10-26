"""
Standalone knowledge distillation script inspired by the code in the
"08_model-compression.ipynb" notebook from the "nlp-with-transformers" repository.

Original reference notebook:
- https://github.com/nlp-with-transformers/notebooks/blob/main/08_model-compression.ipynb

This script distills a teacher Transformer classifier into a smaller student model on GLUE SST-2.
It supports mixed precision and saves the best checkpoint by validation accuracy.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import KLDivLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    get_linear_schedule_with_warmup,
    set_seed as hf_set_seed,
)


DEFAULT_DATASET: str = "glue"
DEFAULT_DATASET_CONFIG: str = "sst2"
DEFAULT_TEACHER: str = "bert-base-uncased"
DEFAULT_STUDENT: str = "distilbert-base-uncased"
DEFAULT_MAX_LENGTH: int = 128
DEFAULT_BATCH_SIZE: int = 32
DEFAULT_LR: float = 3e-5
DEFAULT_WD: float = 0.01
DEFAULT_EPOCHS: int = 3
DEFAULT_WARMUP_RATIO: float = 0.06
DEFAULT_TEMPERATURE: float = 2.0
DEFAULT_ALPHA_CE: float = 0.5
DEFAULT_ALPHA_HARD: float = 0.5
DEFAULT_GRAD_CLIP_NORM: float = 1.0
DEFAULT_EVAL_STEPS: int = 500
DEFAULT_SAVE_TOTAL_LIMIT: int = 1


@dataclass
class TrainConfig:
    dataset_name: str
    dataset_config: str
    teacher_model_name: str
    student_model_name: str
    output_dir: Path
    max_length: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    num_train_epochs: int
    warmup_ratio: float
    temperature: float
    alpha_ce: float
    alpha_hard: float
    grad_clip_norm: float
    eval_steps: int
    save_total_limit: int
    seed: int
    fp16: bool


def parse_args(argv: Optional[List[str]] = None) -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Distill a text classifier (teacher -> student) on GLUE SST-2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_name", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--dataset_config", type=str, default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--teacher_model_name", type=str, default=DEFAULT_TEACHER)
    parser.add_argument("--student_model_name", type=str, default=DEFAULT_STUDENT)
    parser.add_argument("--output_dir", type=str, default="./distilled-sst2")
    parser.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WD)
    parser.add_argument("--num_train_epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--warmup_ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--alpha_ce", type=float, default=DEFAULT_ALPHA_CE)
    parser.add_argument("--alpha_hard", type=float, default=DEFAULT_ALPHA_HARD)
    parser.add_argument("--grad_clip_norm", type=float, default=DEFAULT_GRAD_CLIP_NORM)
    parser.add_argument("--eval_steps", type=int, default=DEFAULT_EVAL_STEPS)
    parser.add_argument("--save_total_limit", type=int, default=DEFAULT_SAVE_TOTAL_LIMIT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")

    args = parser.parse_args(argv)
    return TrainConfig(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        teacher_model_name=args.teacher_model_name,
        student_model_name=args.student_model_name,
        output_dir=Path(args.output_dir),
        max_length=args.max_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        temperature=args.temperature,
        alpha_ce=args.alpha_ce,
        alpha_hard=args.alpha_hard,
        grad_clip_norm=args.grad_clip_norm,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        fp16=args.fp16,
    )


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)


def prepare_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    return tokenizer


def load_and_tokenize(
    dataset_name: str,
    dataset_config: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> Tuple[Dict[str, int], object, object, object]:
    dataset = load_dataset(dataset_name, dataset_config)

    def tokenize_batch(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        return tokenizer(batch["sentence"], truncation=True, max_length=max_length)

    column_names = dataset["train"].column_names
    remove_cols = [c for c in column_names if c != "label"]
    if "sentence" not in column_names:
        # Fallback for datasets with different text column names
        text_col = "sentence" if "sentence" in column_names else (
            "sentence1" if "sentence1" in column_names else (
                "text" if "text" in column_names else column_names[0]
            )
        )

        def tokenize_generic(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
            return tokenizer(batch[text_col], truncation=True, max_length=max_length)

        tokenized = dataset.map(tokenize_generic, batched=True, remove_columns=remove_cols)
    else:
        tokenized = dataset.map(tokenize_batch, batched=True, remove_columns=remove_cols)

    tokenized = tokenized.rename_column("label", "labels")
    format_cols = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in tokenized["train"].column_names:
        format_cols.insert(1, "token_type_ids")
    tokenized.set_format(type="torch", columns=format_cols)

    label2id = {"negative": 0, "positive": 1}
    return label2id, tokenized["train"], tokenized["validation"], tokenized.get("test", None)


def build_dataloaders(
    train_dataset: object,
    eval_dataset: object,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    return train_loader, eval_loader


def load_models(
    teacher_model_name: str,
    student_model_name: str,
    num_labels: int,
    device: torch.device,
) -> Tuple[PreTrainedModel, PreTrainedModel]:
    teacher = AutoModelForSequenceClassification.from_pretrained(
        teacher_model_name, num_labels=num_labels
    )
    student = AutoModelForSequenceClassification.from_pretrained(
        student_model_name, num_labels=num_labels
    )
    teacher.to(device)
    student.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    return teacher, student


def evaluate(
    model: PreTrainedModel,
    data_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    num_correct = 0
    num_total = 0
    total_loss = 0.0
    loss_fn = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in data_loader:
            input_ids: Tensor = batch["input_ids"].to(device)
            attention_mask: Tensor = batch["attention_mask"].to(device)
            labels: Tensor = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss: Tensor = outputs.loss
            logits: Tensor = outputs.logits
            total_loss += loss.item() * input_ids.size(0)
            preds = logits.argmax(dim=-1)
            num_correct += (preds == labels).sum().item()
            num_total += labels.size(0)
    avg_loss = total_loss / max(1, num_total)
    accuracy = num_correct / max(1, num_total)
    return avg_loss, accuracy


def train(
    cfg: TrainConfig,
    teacher: PreTrainedModel,
    student: PreTrainedModel,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    device: torch.device,
) -> None:
    num_training_steps = cfg.num_train_epochs * math.ceil(len(train_loader.dataset) / cfg.batch_size)
    num_warmup_steps = int(cfg.warmup_ratio * num_training_steps)

    optimizer = AdamW(student.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )

    # AMP setup (device-aware)
    device_type = device.type  # 'cuda' | 'mps' | 'cpu'
    use_amp = bool(cfg.fp16)
    if device_type == "cuda" and use_amp:
        scaler = torch.amp.GradScaler("cuda")
    else:
        scaler = None
    autocast_dtype = torch.float16 if device_type in ("cuda", "mps") else torch.bfloat16
    kd_criterion = KLDivLoss(reduction="batchmean")
    ce_criterion = torch.nn.CrossEntropyLoss()

    best_eval_acc = -1.0
    best_dir = cfg.output_dir
    best_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(cfg.num_train_epochs):
        student.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.num_train_epochs}")
        running_loss = 0.0

        for step, batch in enumerate(pbar):
            input_ids: Tensor = batch["input_ids"].to(device)
            attention_mask: Tensor = batch["attention_mask"].to(device)
            labels: Tensor = batch["labels"].to(device)

            with torch.no_grad():
                teacher_logits: Tensor = teacher(
                    input_ids=input_ids, attention_mask=attention_mask
                ).logits

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device_type,
                dtype=autocast_dtype,
                enabled=use_amp,
            ):
                student_outputs = student(input_ids=input_ids, attention_mask=attention_mask)
                student_logits: Tensor = student_outputs.logits

                # KD loss with temperature scaling
                t: float = cfg.temperature
                student_log_probs = F.log_softmax(student_logits / t, dim=-1)
                teacher_probs = F.softmax(teacher_logits / t, dim=-1)
                kd_loss: Tensor = kd_criterion(student_log_probs, teacher_probs) * (t * t)

                # Hard label loss
                hard_loss: Tensor = ce_criterion(student_logits, labels)

                loss: Tensor = cfg.alpha_ce * kd_loss + cfg.alpha_hard * hard_loss

            if scaler is not None:
                scaler.scale(loss).backward()
                if cfg.grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if cfg.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip_norm)
                optimizer.step()

            lr_scheduler.step()
            running_loss += loss.item()
            global_step += 1

            if cfg.eval_steps > 0 and (global_step % cfg.eval_steps == 0):
                eval_loss, eval_acc = evaluate(student, eval_loader, device)
                pbar.set_postfix({
                    "train_loss": f"{running_loss / max(1, step+1):.4f}",
                    "eval_loss": f"{eval_loss:.4f}",
                    "eval_acc": f"{eval_acc:.4f}",
                })

                if eval_acc > best_eval_acc:
                    best_eval_acc = eval_acc
                    save_dir = best_dir
                    # Clear older checkpoints if limit is 1
                    for child in save_dir.iterdir():
                        if child.is_dir():
                            for f in child.iterdir():
                                f.unlink()
                            child.rmdir()
                    student.save_pretrained(save_dir)
                    tokenizer_name = cfg.student_model_name
                    AutoTokenizer.from_pretrained(tokenizer_name).save_pretrained(save_dir)

        # Epoch end evaluation
        eval_loss, eval_acc = evaluate(student, eval_loader, device)
        if eval_acc > best_eval_acc:
            best_eval_acc = eval_acc
            save_dir = best_dir
            for child in save_dir.iterdir():
                if child.is_dir():
                    for f in child.iterdir():
                        f.unlink()
                    child.rmdir()
            student.save_pretrained(save_dir)
            AutoTokenizer.from_pretrained(cfg.student_model_name).save_pretrained(save_dir)

    print(f"Best validation accuracy: {best_eval_acc:.4f}")


def main(argv: Optional[List[str]] = None) -> None:
    cfg = parse_args(argv)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.backends.cudnn.benchmark = True
    set_all_seeds(cfg.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Prepare tokenizer and data
    tokenizer = prepare_tokenizer(cfg.teacher_model_name)
    label2id, train_ds, val_ds, _ = load_and_tokenize(
        cfg.dataset_name, cfg.dataset_config, tokenizer, cfg.max_length
    )
    train_loader, eval_loader = build_dataloaders(train_ds, val_ds, tokenizer, cfg.batch_size)

    # Load models
    teacher, student = load_models(
        cfg.teacher_model_name, cfg.student_model_name, num_labels=len(label2id), device=device
    )

    # Train and save best
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    train(cfg, teacher, student, train_loader, eval_loader, device)
    print(f"Distilled model saved to: {cfg.output_dir}")


if __name__ == "__main__":
    main()


