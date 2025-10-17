#!/usr/bin/env python3
"""
Standalone script for training a summarization model with Hugging Face Transformers.
Adapted from the "Training a Summarization Model" section of the
"NLP with Transformers" 06_summarization notebook.

Source: https://github.com/nlp-with-transformers/notebooks/blob/main/06_summarization.ipynb

Dependencies:
- torch
- transformers
- datasets
- evaluate
- numpy

Example:
python fine_tune_encoder_decoder_for_custom_summarization.py \
  --model_name_or_path sshleifer/distilbart-cnn-12-6 \
  --dataset_name cnn_dailymail --dataset_config 3.0.0 \
  --text_column article --summary_column highlights \
  --output_dir ./summarization-bart \
  --per_device_train_batch_size 4 --per_device_eval_batch_size 4 \
  --learning_rate 3e-5 --num_train_epochs 2 \
  --gradient_accumulation_steps 2 --fp16 True --num_beams 4
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import DatasetDict, load_dataset
import evaluate
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    GenerationConfig,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)


@dataclass
class SummarizationConfig:
    model_name_or_path: str
    dataset_name: str
    dataset_config: Optional[str]
    text_column: str
    summary_column: str
    output_dir: str
    max_source_length: int
    max_target_length: int
    val_max_target_length: int
    pad_to_max_length: bool
    num_beams: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    num_train_epochs: float
    lr_scheduler_type: str
    warmup_ratio: float
    warmup_steps: int
    logging_steps: int
    eval_steps: int
    save_steps: int
    save_total_limit: int
    seed: int
    fp16: bool
    predict_with_generate: bool
    max_train_samples: Optional[int]
    max_eval_samples: Optional[int]
    max_predict_samples: Optional[int]
    dataset_cache_dir: Optional[str]


def parse_args() -> SummarizationConfig:
    parser = argparse.ArgumentParser(description="Fine-tune an encoder-decoder model for summarization.")

    # Data and model
    parser.add_argument("--model_name_or_path", type=str, default="sshleifer/distilbart-cnn-12-6")
    parser.add_argument("--dataset_name", type=str, default="cnn_dailymail")
    parser.add_argument("--dataset_config", type=str, default="3.0.0")
    parser.add_argument("--text_column", type=str, default="article")
    parser.add_argument("--summary_column", type=str, default="highlights")
    parser.add_argument("--output_dir", type=str, default="./summarization-model")
    parser.add_argument("--dataset_cache_dir", type=str, default=None)

    # Sequence lengths
    parser.add_argument("--max_source_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=128)
    parser.add_argument("--val_max_target_length", type=int, default=128)
    parser.add_argument("--pad_to_max_length", type=lambda x: x.lower() == "true", default=False)

    # Generation
    parser.add_argument("--num_beams", type=int, default=4)

    # Optimization
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", choices=[
        "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup",
    ])
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=0)

    # Logging / eval / checkpointing
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42)

    # Mixed precision and generation in eval
    parser.add_argument("--fp16", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--predict_with_generate", type=lambda x: x.lower() == "true", default=True)

    # Subsetting
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--max_predict_samples", type=int, default=None)

    args = parser.parse_args()

    return SummarizationConfig(
        model_name_or_path=args.model_name_or_path,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        text_column=args.text_column,
        summary_column=args.summary_column,
        output_dir=args.output_dir,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        val_max_target_length=args.val_max_target_length,
        pad_to_max_length=args.pad_to_max_length,
        num_beams=args.num_beams,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        fp16=args.fp16,
        predict_with_generate=args.predict_with_generate,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        max_predict_samples=args.max_predict_samples,
        dataset_cache_dir=args.dataset_cache_dir,
    )


def build_tokenize_functions(
    tokenizer: AutoTokenizer,
    cfg: SummarizationConfig,
) -> Tuple[Any, Any]:
    padding_strategy = "max_length" if cfg.pad_to_max_length else False

    def tokenize_batch(examples: Dict[str, List[str]]) -> Dict[str, Any]:
        inputs = examples[cfg.text_column]
        targets = examples[cfg.summary_column]

        if tokenizer.__class__.__name__.lower().startswith("t5"):
            inputs = [f"summarize: {text}" for text in inputs]

        model_inputs = tokenizer(
            inputs,
            max_length=cfg.max_source_length,
            padding=padding_strategy,
            truncation=True,
        )

        labels = tokenizer(
            text_target=targets,
            max_length=cfg.max_target_length,
            padding=padding_strategy,
            truncation=True,
        )

        label_ids = labels["input_ids"]
        if cfg.pad_to_max_length:
            label_pad_token_id = -100
            label_ids = [
                [token if token != tokenizer.pad_token_id else label_pad_token_id for token in label_list]
                for label_list in label_ids
            ]
        model_inputs["labels"] = label_ids
        return model_inputs

    def tokenize_batch_for_eval(examples: Dict[str, List[str]]) -> Dict[str, Any]:
        inputs = examples[cfg.text_column]
        targets = examples[cfg.summary_column]

        if tokenizer.__class__.__name__.lower().startswith("t5"):
            inputs = [f"summarize: {text}" for text in inputs]

        model_inputs = tokenizer(
            inputs,
            max_length=cfg.max_source_length,
            padding=padding_strategy,
            truncation=True,
        )

        labels = tokenizer(
            text_target=targets,
            max_length=cfg.val_max_target_length,
            padding=padding_strategy,
            truncation=True,
        )

        label_ids = labels["input_ids"]
        if cfg.pad_to_max_length:
            label_pad_token_id = -100
            label_ids = [
                [token if token != tokenizer.pad_token_id else label_pad_token_id for token in label_list]
                for label_list in label_ids
            ]
        model_inputs["labels"] = label_ids
        return model_inputs

    return tokenize_batch, tokenize_batch_for_eval


def split_into_sentences(text: str) -> List[str]:
    if not text:
        return []
    segments: List[str] = []
    start: int = 0
    for idx, ch in enumerate(text):
        if ch in {".", "?", "!"}:
            segment = text[start : idx + 1].strip()
            if segment:
                segments.append(segment)
            start = idx + 1
    tail = text[start:].strip()
    if tail:
        segments.append(tail)
    return segments


def build_compute_metrics(tokenizer: AutoTokenizer):
    rouge = evaluate.load("rouge")

    def compute_metrics_fn(eval_pred: Any) -> Dict[str, float]:
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = ["\n".join(split_into_sentences(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(split_into_sentences(label.strip())) for label in decoded_labels]

        result = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
        )
        result = {k: round(v * 100, 4) for k, v in result.items()}
        return result

    return compute_metrics_fn


def load_and_prepare_datasets(
    cfg: SummarizationConfig,
    tokenizer: AutoTokenizer,
) -> Tuple[DatasetDict, DatasetDict]:
    raw_datasets: DatasetDict = load_dataset(
        cfg.dataset_name,
        cfg.dataset_config,
        cache_dir=cfg.dataset_cache_dir,
    )

    tokenize_train, tokenize_eval = build_tokenize_functions(tokenizer, cfg)

    processed_train = raw_datasets["train"]
    processed_val = raw_datasets["validation"] if "validation" in raw_datasets else raw_datasets["test"]
    processed_test = raw_datasets.get("test")

    if cfg.max_train_samples:
        processed_train = processed_train.select(range(cfg.max_train_samples))
    if cfg.max_eval_samples:
        processed_val = processed_val.select(range(cfg.max_eval_samples))
    if cfg.max_predict_samples and processed_test is not None:
        processed_test = processed_test.select(range(cfg.max_predict_samples))

    tokenized_train = processed_train.map(
        tokenize_train,
        batched=True,
        remove_columns=processed_train.column_names,
        desc="Tokenizing train split",
    )
    tokenized_val = processed_val.map(
        tokenize_eval,
        batched=True,
        remove_columns=processed_val.column_names,
        desc="Tokenizing validation split",
    )
    tokenized_test = None
    if processed_test is not None:
        tokenized_test = processed_test.map(
            tokenize_eval,
            batched=True,
            remove_columns=processed_test.column_names,
            desc="Tokenizing test split",
        )

    tokenized: DatasetDict = DatasetDict(
        {
            "train": tokenized_train,
            "validation": tokenized_val,
            **({"test": tokenized_test} if tokenized_test is not None else {}),
        }
    )

    return raw_datasets, tokenized


def main() -> None:
    cfg = parse_args()

    os.makedirs(cfg.output_dir, exist_ok=True)
    set_seed(cfg.seed)

    config = AutoConfig.from_pretrained(cfg.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        cfg.model_name_or_path,
        config=config,
        use_safetensors=True,
    )

    try:
        generation_cfg = GenerationConfig.from_pretrained(cfg.model_name_or_path)
    except Exception:
        generation_cfg = GenerationConfig.from_model_config(model.config)
    generation_cfg.max_length = cfg.val_max_target_length
    generation_cfg.num_beams = cfg.num_beams
    model.generation_config = generation_cfg

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    raw_datasets, tokenized_datasets = load_and_prepare_datasets(cfg, tokenizer)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, pad_to_multiple_of=8 if cfg.fp16 else None)

    generation_max_length = cfg.val_max_target_length

    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg.output_dir,
        eval_strategy="steps",
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        num_train_epochs=cfg.num_train_epochs,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        warmup_steps=cfg.warmup_steps,
        logging_steps=cfg.logging_steps,
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        predict_with_generate=cfg.predict_with_generate,
        generation_max_length=generation_max_length,
        generation_num_beams=cfg.num_beams,
        fp16=cfg.fp16 and torch.cuda.is_available(),
        ddp_find_unused_parameters=False,
        load_best_model_at_end=True,
        metric_for_best_model="rougeLsum",
        greater_is_better=True,
        report_to=["tensorboard"],
    )

    compute_metrics_fn = build_compute_metrics(tokenizer)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
    )

    if trainer.is_world_process_zero():
        print("Starting training...")

    train_result = trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    metrics = train_result.metrics
    metrics["train_samples"] = len(tokenized_datasets["train"]) if tokenized_datasets.get("train") is not None else 0
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    if trainer.is_world_process_zero():
        print("Evaluating on validation set...")

    eval_metrics = trainer.evaluate(
        eval_dataset=tokenized_datasets["validation"],
        max_length=generation_max_length,
        num_beams=cfg.num_beams,
    )
    eval_metrics["eval_samples"] = len(tokenized_datasets["validation"]) if tokenized_datasets.get("validation") is not None else 0
    perplexity = math.exp(eval_metrics["eval_loss"]) if eval_metrics.get("eval_loss") is not None else float("nan")
    eval_metrics["perplexity"] = round(perplexity, 4) if math.isfinite(perplexity) else float("nan")
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    if "test" in tokenized_datasets:
        if trainer.is_world_process_zero():
            print("Predicting on test set...")
        test_results = trainer.predict(
            test_dataset=tokenized_datasets["test"],
            max_length=generation_max_length,
            num_beams=cfg.num_beams,
        )
        test_metrics = test_results.metrics
        test_metrics["test_samples"] = len(tokenized_datasets["test"]) if tokenized_datasets.get("test") is not None else 0
        trainer.log_metrics("test", test_metrics)
        trainer.save_metrics("test", test_metrics)

    if trainer.is_world_process_zero():
        print("All done. Artifacts saved to:", cfg.output_dir)


if __name__ == "__main__":
    main()
