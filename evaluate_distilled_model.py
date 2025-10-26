"""
Evaluate a distilled text classification model on GLUE SST-2.

Mirrors preprocessing and device handling used in distill_a_model.py.
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Tuple
import os
import time
import json

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a distilled classifier on GLUE SST-2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path", type=str, default=None, help="Path or HF id of a single model to evaluate")
    parser.add_argument("--teacher_path", type=str, default=None, help="Path or HF id of the teacher model (for comparison)")
    parser.add_argument("--student_path", type=str, default=None, help="Path or HF id of the student model (for comparison)")
    parser.add_argument("--dataset_name", type=str, default="glue")
    parser.add_argument("--dataset_config", type=str, default="sst2")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"])
    return parser.parse_args(argv)


def prepare_tokenizer(model_path: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    return tokenizer


def load_and_tokenize(
    dataset_name: str,
    dataset_config: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> Dict[str, object]:
    dataset = load_dataset(dataset_name, dataset_config)

    def tokenize_batch(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        return tokenizer(batch["sentence"], truncation=True, max_length=max_length)

    column_names = dataset["train"].column_names
    remove_cols = [c for c in column_names if c != "label"]

    if "sentence" not in column_names:
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
    return tokenized


def build_dataloader(
    dataset_split: object,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
) -> DataLoader:
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return DataLoader(dataset_split, batch_size=batch_size, shuffle=False, collate_fn=collator)


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
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits: Tensor = outputs.logits
            loss = loss_fn(logits, labels)
            total_loss += loss.item() * input_ids.size(0)
            preds = logits.argmax(dim=-1)
            num_correct += (preds == labels).sum().item()
            num_total += labels.size(0)
    avg_loss = total_loss / max(1, num_total)
    accuracy = num_correct / max(1, num_total)
    return avg_loss, accuracy


def count_parameters(model: PreTrainedModel) -> Tuple[int, int]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_directory_size_bytes(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _weight_sizes_in_dir(path: str) -> Dict[str, int]:
    sizes: Dict[str, int] = {}
    try:
        for root, _, files in os.walk(path):
            for f in files:
                if f.endswith(".safetensors") or f.endswith(".bin"):
                    fp = os.path.join(root, f)
                    try:
                        sizes[f] = os.path.getsize(fp)
                    except OSError:
                        pass
    except Exception:
        pass
    return sizes


def _compute_model_sizes(model_path: str) -> Tuple[int, Dict[str, int]]:
    # Local directory
    if os.path.isdir(model_path):
        return get_directory_size_bytes(model_path), _weight_sizes_in_dir(model_path)

    # Try local cache of a HF repo id (no network)
    try:
        from huggingface_hub import snapshot_download

        cached_dir = snapshot_download(repo_id=model_path, local_files_only=True)
        return get_directory_size_bytes(cached_dir), _weight_sizes_in_dir(cached_dir)
    except Exception:
        pass

    # Fallback: query Hub metadata (requires network) to sum file sizes
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(model_path, files_metadata=True)
        total_size = 0
        weights: Dict[str, int] = {}
        for s in getattr(info, "siblings", []) or []:
            size = getattr(s, "size", None)
            if size is not None:
                total_size += int(size)
                if s.rfilename.endswith(".safetensors") or s.rfilename.endswith(".bin"):
                    weights[os.path.basename(s.rfilename)] = int(size)
        return int(total_size), weights
    except Exception:
        return 0, {}


def _evaluate_one(
    model_path: str,
    tokenizer: PreTrainedTokenizerBase,
    dataset_name: str,
    dataset_config: str,
    split: str,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> Dict[str, object]:
    tokenized = load_and_tokenize(dataset_name, dataset_config, tokenizer, max_length)
    dataset_split = tokenized[split]
    eval_loader = build_dataloader(dataset_split, tokenizer, batch_size)

    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    disk_size_bytes, weights_sizes = _compute_model_sizes(model_path)
    total_params, trainable_params = count_parameters(model)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    eval_loss, eval_acc = evaluate(model, eval_loader, device)
    elapsed_s = time.perf_counter() - start
    num_samples = len(eval_loader.dataset)
    samples_per_s = float(num_samples) / elapsed_s if elapsed_s > 0 else 0.0
    average_latency_ms = (elapsed_s / num_samples) * 1000.0 if num_samples > 0 else None
    peak_mem_bytes = None
    if device.type == "cuda":
        peak_mem_bytes = torch.cuda.max_memory_allocated(device)

    disk_size_mb = round(disk_size_bytes / (1024 * 1024), 2)
    weights_sizes_mb = {k: round(v / (1024 * 1024), 2) for k, v in weights_sizes.items()}

    return {
        "model_path": model_path,
        "device": device.type,
        "loss": round(eval_loss, 4),
        "accuracy": round(eval_acc, 4),
        "num_samples": num_samples,
        "elapsed_s": round(elapsed_s, 4),
        "samples_per_s": round(samples_per_s, 2),
        "average_latency_ms": round(average_latency_ms, 3) if average_latency_ms is not None else None,
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "disk_size_bytes": int(disk_size_bytes),
        "disk_size_mb": disk_size_mb,
        "weights_sizes": weights_sizes,
        "weights_sizes_mb": weights_sizes_mb,
        "peak_cuda_memory_bytes": int(peak_mem_bytes) if peak_mem_bytes is not None else None,
    }


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Comparison mode if both teacher and student are provided
    if args.teacher_path and args.student_path:
        tokenizer = prepare_tokenizer(args.teacher_path)
        teacher_metrics = _evaluate_one(
            model_path=args.teacher_path,
            tokenizer=tokenizer,
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            split=args.split,
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=device,
        )
        student_metrics = _evaluate_one(
            model_path=args.student_path,
            tokenizer=tokenizer,
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            split=args.split,
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=device,
        )

        speedup = None
        if teacher_metrics.get("samples_per_s", 0) and student_metrics.get("samples_per_s", 0):
            denom = float(teacher_metrics["samples_per_s"]) or 1e-9
            speedup = round(float(student_metrics["samples_per_s"]) / denom, 3)

        latency_reduction = None
        if teacher_metrics.get("average_latency_ms") and student_metrics.get("average_latency_ms"):
            denom = float(teacher_metrics["average_latency_ms"]) or 1e-9
            latency_reduction = round(1.0 - (float(student_metrics["average_latency_ms"]) / denom), 3)

        size_reduction = None
        if teacher_metrics.get("disk_size_bytes") and student_metrics.get("disk_size_bytes"):
            denom = float(teacher_metrics["disk_size_bytes"]) or 1e-9
            size_reduction = round(1.0 - (float(student_metrics["disk_size_bytes"]) / denom), 3)

        acc_delta = None
        if (teacher_metrics.get("accuracy") is not None) and (student_metrics.get("accuracy") is not None):
            acc_delta = round(float(student_metrics["accuracy"]) - float(teacher_metrics["accuracy"]), 4)

        comparison = {
            "split": args.split,
            "device": device.type,
            "teacher": teacher_metrics,
            "student": student_metrics,
            "speedup_samples_per_s": speedup,
            "latency_reduction": latency_reduction,
            "size_reduction": size_reduction,
            "accuracy_delta": acc_delta,
        }
        print(json.dumps(comparison, indent=2))
        return

    # Single model evaluation path
    if not args.model_path:
        raise SystemExit("Provide --model_path for single evaluation or both --teacher_path and --student_path for comparison.")

    tokenizer = prepare_tokenizer(args.model_path)
    single = _evaluate_one(
        model_path=args.model_path,
        tokenizer=tokenizer,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=device,
    )
    single["split"] = args.split
    print(json.dumps(single, indent=2))


if __name__ == "__main__":
    main()


