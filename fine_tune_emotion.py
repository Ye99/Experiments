import argparse
import os
from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np
from datasets import load_dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)


DEFAULT_MODEL_CKPT: str = "distilbert-base-uncased"
DEFAULT_NUM_LABELS: int = 6
DEFAULT_BATCH_SIZE: int = 64
DEFAULT_LR: float = 2e-5
DEFAULT_EPOCHS: int = 2


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_emotion_dataset() -> DatasetDict:
    return load_dataset("emotion")


def build_tokenizer(model_ckpt: str) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(model_ckpt)


def tokenize_function_builder(tokenizer: AutoTokenizer):
    def tokenize(batch: Dict[str, Any]) -> Dict[str, Any]:
        return tokenizer(batch["text"], padding=True, truncation=True)

    return tokenize


def tokenize_corpus(dataset: DatasetDict, tokenizer: AutoTokenizer) -> DatasetDict:
    tokenize_fn = tokenize_function_builder(tokenizer)
    encoded = dataset.map(tokenize_fn, batched=True, batch_size=None)
    encoded = encoded.remove_columns([c for c in encoded["train"].column_names if c not in {"input_ids", "attention_mask", "label", "text"}])
    encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    return encoded


def build_model(model_ckpt: str, num_labels: int, device: torch.device) -> AutoModelForSequenceClassification:
    model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels)
    return model.to(device)


def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    predictions, labels = eval_pred
    if isinstance(predictions, (list, tuple)):
        predictions = predictions[0]
    preds = predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


@dataclass
class RunConfig:
    model_ckpt: str
    output_dir: str
    batch_size: int
    learning_rate: float
    num_train_epochs: int
    push_to_hub: bool
    log_level: str


def build_training_args(config: RunConfig, train_size: int) -> TrainingArguments:
    logging_steps = max(1, train_size // config.batch_size)
    model_name = f"{config.model_ckpt}-finetuned-emotion"
    output_dir = config.output_dir or model_name
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        disable_tqdm=False,
        logging_steps=logging_steps,
        push_to_hub=config.push_to_hub,
        log_level=config.log_level,
        load_best_model_at_end=False,
        save_strategy="epoch",
    )


def run_training(config: RunConfig) -> None:
    device = get_device()
    dataset = load_emotion_dataset()
    tokenizer = build_tokenizer(config.model_ckpt)
    encoded = tokenize_corpus(dataset, tokenizer)
    model = build_model(config.model_ckpt, DEFAULT_NUM_LABELS, device)

    training_args = build_training_args(config, len(encoded["train"]))
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=encoded["train"],
        eval_dataset=encoded["validation"],
        tokenizer=tokenizer,
    )

    trainer.train()

    metrics = trainer.evaluate()
    for k, v in metrics.items():
        print(f"{k}: {v}")

    if config.push_to_hub:
        trainer.push_to_hub(commit_message="Training completed!")
    else:
        save_dir = training_args.output_dir
        os.makedirs(save_dir, exist_ok=True)
        trainer.save_model(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"Saved model and tokenizer to: {save_dir}")


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT on the emotion dataset")
    parser.add_argument("--model_ckpt", type=str, default=DEFAULT_MODEL_CKPT)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LR)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--log_level", type=str, default="error", choices=["critical", "error", "warning", "info", "debug"])
    args = parser.parse_args()
    return RunConfig(
        model_ckpt=args.model_ckpt,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        push_to_hub=args.push_to_hub,
        log_level=args.log_level,
    )


def main() -> None:
    config = parse_args()
    run_training(config)


if __name__ == "__main__":
    main()


