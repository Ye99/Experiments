# Experiments: Fine-Tuning and Distilling Transformers

Standalone Python scripts that turn notebooks from
[*Natural Language Processing with Transformers*](https://github.com/nlp-with-transformers/notebooks)
into runnable, multi-GPU-friendly experiments with Hugging Face `transformers`.

| # | Script | What it does | Default model / data |
|---|---|---|---|
| 1 | `fine_tune_encoder_for_emotion_classification.py` | Fine-tunes an encoder for 6-class emotion classification (ch. 2) | `distilbert-base-uncased` on `emotion` |
| 2 | `fine_tune_encoder_decoder_for_custom_summarization.py` | Fine-tunes an encoder-decoder for abstractive summarization, scored with ROUGE (ch. 6) | `sshleifer/distilbart-cnn-12-6` on `cnn_dailymail` 3.0.0 |
| 3 | `distill_a_model.py` | Knowledge distillation: trains a small student to copy a larger teacher (ch. 8) | teacher `bert-base-uncased` → student `distilbert-base-uncased` on GLUE SST-2 |
| 3 | `evaluate_distilled_model.py` | Measures accuracy, latency, throughput, parameters and disk size; compares teacher vs student | GLUE SST-2 validation |
| – | `check_torch.py` | Environment sanity check (PyTorch version, CUDA/MPS, a test matmul) | – |

## Set up environment

We use mamba and conda-lock to reproduce the environment exactly (Linux x86_64 and macOS arm64).

1) Ensure mamba and conda-lock

```bash
mamba install -n base -c conda-forge conda-lock
```

2) Create the environment from the lockfile

```bash
cd path/to/repo
conda-lock install --name LLM_experiments conda-lock.yml --mamba
mamba activate LLM_experiments
```

Notes:
- Regenerate the lockfile for both platforms:
  `conda-lock lock -f env.yml --platform osx-arm64 --platform linux-64`
- Fallback (less exact): create the environment from `env.yml`:

  ```bash
  mamba env create -f env.yml -n LLM_experiments
  mamba activate LLM_experiments
  ```

## Quick verification

```bash
python check_torch.py
```

Prints the PyTorch and Python versions, whether CUDA or MPS is available, the selected device, and runs a 1024×1024 matmul on it.

## Experiment 1: fine-tune an encoder for emotion classification

Loads the `emotion` dataset, tokenizes it, fine-tunes a sequence-classification head with `Trainer`, and reports accuracy and weighted F1.

Plain Trainer (uses multiple GPUs via DataParallel if visible):

```bash
python fine_tune_encoder_for_emotion_classification.py --epochs 2 --batch_size 64
```

Accelerate (recommended, DDP):

```bash
accelerate launch --multi_gpu fine_tune_encoder_for_emotion_classification.py --epochs 2 --batch_size 64
```

Tips:
- Force a specific GPU: `CUDA_VISIBLE_DEVICES=0 python fine_tune_encoder_for_emotion_classification.py ...`
- Mixed precision is auto-selected on CUDA (bf16 if supported, else fp16) and disabled on CPU. Override with `--fp16` / `--bf16`.
- Outputs are saved under `<model>-finetuned-emotion/` (ignored by `.gitignore`), including `label_names.json`.
- Push to the Hub with `--push_to_hub --hub_model_id YOUR_USERNAME/<name>` (after `huggingface-cli login`).

### Performance comparison (2× RTX 4060 Ti, CUDA 12.4, batch_size=64, 1 epoch)

| Mode | Command | Train runtime | Steps/s | Samples/s |
|---|---|---:|---:|---:|
| DataParallel (Trainer, plain `python`) | `python fine_tune_encoder_for_emotion_classification.py --epochs 1 --batch_size 64` | ~29.74 s | ~4.20 | ~538 |
| DDP (Accelerate) | `accelerate launch --multi_gpu fine_tune_encoder_for_emotion_classification.py --epochs 1 --batch_size 64` | ~16.26 s | ~7.69 | ~984 |

- Speedup (DDP vs DP): ~1.8× on this setup.

### Reproducibility

- Seed is set via `--seed` (defaults to 42).
- For larger global batch sizes (GPUs × per-device batch × grad_accum), consider LR scaling.

## Experiment 2: fine-tune an encoder-decoder for summarization

Fine-tunes a seq2seq model with `Seq2SeqTrainer`. Evaluation generates summaries with beam search (`--num_beams`) and scores them with ROUGE-1/2/L/Lsum. The defaults use CNN/DailyMail. To use your own data (e.g. customer-support conversations), point `--dataset_name` / `--dataset_config` at any Hub dataset and set `--text_column` / `--summary_column`. Use `--max_train_samples` / `--max_eval_samples` for quick runs.

Plain Trainer (single GPU, or DataParallel if multiple GPUs are visible):

```bash
python fine_tune_encoder_decoder_for_custom_summarization.py \
  --model_name_or_path sshleifer/distilbart-cnn-12-6 \
  --dataset_name cnn_dailymail --dataset_config 3.0.0 \
  --text_column article --summary_column highlights \
  --output_dir ./summarization-model \
  --per_device_train_batch_size 4 --per_device_eval_batch_size 4 \
  --learning_rate 3e-5 --num_train_epochs 3 \
  --gradient_accumulation_steps 1 --fp16 true --num_beams 4
```

Accelerate (recommended, DDP, multi-GPU):

```bash
accelerate launch --multi_gpu fine_tune_encoder_decoder_for_custom_summarization.py \
  --model_name_or_path sshleifer/distilbart-cnn-12-6 \
  --dataset_name cnn_dailymail --dataset_config 3.0.0 \
  --text_column article --summary_column highlights \
  --output_dir ./summarization-model \
  --per_device_train_batch_size 4 --per_device_eval_batch_size 4 \
  --learning_rate 3e-5 --num_train_epochs 3 \
  --gradient_accumulation_steps 1 --fp16 true --num_beams 4
```

Torchrun alternative (multi-GPU):

```bash
torchrun --nproc_per_node=2 fine_tune_encoder_decoder_for_custom_summarization.py ...
```

Result from a run with the settings above (validation set, step 12,000 ≈ 1/3 epoch):

| ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum | eval loss |
|---:|---:|---:|---:|---:|
| 44.51 | 21.31 | 30.54 | 41.78 | 1.678 |

Notes:
- Model weights are loaded with safetensors (`use_safetensors=True`). Prefer models that ship safetensors weights (most BART/T5 repos do). If you only have `.bin` weights, re-download safetensors from the Hub or upgrade PyTorch to >= 2.6.
- Tokenization runs on CPU. For faster preprocessing, add `num_proc=$(nproc)` to `Dataset.map` in the script.
- Outputs and logs are written under `--output_dir` (e.g., `./summarization-model`).

## Experiment 3: distill a model

`distill_a_model.py` is a plain PyTorch training loop. On each batch the frozen teacher and the student both predict, and the student is trained on a mix of two losses:

```
loss = alpha_ce   * KL( softmax(teacher_logits / T) || softmax(student_logits / T) ) * T²   # match the teacher
     + alpha_hard * CrossEntropy(student_logits, labels)                                   # match the true labels
```

- The temperature `T` (`--temperature`, default 2.0) softens both distributions, so the student also learns how confident the teacher is, not just its top answer.
- The `T²` factor keeps the soft loss on the same gradient scale as the hard loss.
- `--alpha_ce` / `--alpha_hard` (default 0.5 / 0.5) weight the two terms.
- It uses AdamW with linear warmup/decay, gradient clipping, and optional mixed precision (`--fp16`). It evaluates every `--eval_steps` and at the end of each epoch, and saves only the best checkpoint (by validation accuracy) to `--output_dir`.

**Use a teacher that is already fine-tuned on the task.** The default `bert-base-uncased` has a randomly initialized classification head, so it can't teach anything. Pass an SST-2 fine-tuned BERT (uncased) instead, for example:

```bash
python distill_a_model.py \
  --dataset_name glue --dataset_config sst2 \
  --teacher_model_name yoshitomo-matsubara/bert-base-uncased-sst2 \
  --student_model_name distilbert-base-uncased \
  --output_dir ./distilled-sst2 --fp16
```

The script runs on a single device (CUDA, MPS, or CPU). It doesn't use Accelerate/DDP, so don't launch it with `--multi_gpu`: that would start independent copies that all write to the same `--output_dir`.

### Evaluate

Single model:

```bash
python evaluate_distilled_model.py \
  --model_path ./distilled-sst2 \
  --dataset_name glue --dataset_config sst2 \
  --split validation --batch_size 64
```

Prints JSON with loss, accuracy, samples/s, average latency, total/trainable parameters, disk and weight-file sizes, and peak CUDA memory.

Compare teacher vs student:

```bash
python evaluate_distilled_model.py \
  --teacher_path yoshitomo-matsubara/bert-base-uncased-sst2 \
  --student_path ./distilled-sst2 \
  --dataset_name glue --dataset_config sst2 \
  --split validation --batch_size 64
```

Prints both models' metrics plus `speedup_samples_per_s`, `latency_reduction`, `size_reduction`, and `accuracy_delta`. Use `--split validation`: GLUE's SST-2 test labels are hidden (all `-1`).

### Limitations

- The data is tokenized with the **teacher's** tokenizer and fed to both models, so teacher and student must share a vocabulary (e.g. BERT-uncased → DistilBERT-uncased).
- Labels are fixed to binary `negative`/`positive`. Other `--dataset_name` values only work for binary single-sentence tasks.
