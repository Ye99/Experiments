# Experiments: DistilBERT Emotion Fine-Tuning

This repo contains a self-contained Python code to fine-tune DistilBERT on the Hugging Face "emotion" dataset:

## Set up environment 

We use mamba and conda-lock to reproduce the environment exactly.

1) Ensure mamba and conda-lock

```bash
mamba install -n base -c conda-forge conda-lock
```

2) Create the environment from the lockfile

```bash
cd path/to/repo
conda-lock install --name finetune_transformer conda-lock.yml --mamba
mamba activate finetune_transformer
```

Notes:
- If you need to regenerate the lock (Linux x86_64): `conda-lock lock -f env.yml -p linux-64`
- As a portable alternative (less exact), you can do:

```bash
mamba env create -f env.yml -n finetune_transformer --channel-priority strict
mamba activate finetune_transformer
```

## Quick verification

```bash
python - <<'PY'
import torch
print('cuda_available=', torch.cuda.is_available())
print('num_devices=', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

## Experiment 1, fine tune encoder for emotion classification

Plain Trainer (will use multiple GPUs via DataParallel if visible):

```bash
python fine_tune_encoder_for_emotion_classification.py --epochs 2 --batch_size 64
```

Accelerate (recommended DDP):

```bash
accelerate launch --multi_gpu fine_tune_encoder_for_emotion_classification.py --epochs 2 --batch_size 64
```

Tips:
- Force a specific GPU: `CUDA_VISIBLE_DEVICES=0 python fine_tune_encoder_for_emotion_classification.py ...`
- Mixed precision is auto-selected: bf16 if supported, else fp16 on CUDA; disabled on CPU.
- Outputs are saved under `<model>-finetuned-emotion/` (ignored by `.gitignore`).


### Performance comparison (2× RTX 4060 Ti, CUDA 12.4, batch_size=64, 1 epoch)

| Mode | Command | Train runtime | Steps/s | Samples/s |
|---|---|---:|---:|---:|
| DataParallel (Trainer, plain `python`) | `python fine_tune_encoder_for_emotion_classification.py --epochs 1 --batch_size 64` | ~29.74 s | ~4.20 | ~538 |
| DDP (Accelerate) | `accelerate launch --multi_gpu fine_tune_encoder_for_emotion_classification.py --epochs 1 --batch_size 64` | ~16.26 s | ~7.69 | ~984 |

- Speedup (DDP vs DP): ~1.8× on this setup.

### Reproducibility

- Seed is set via `--seed` (defaults to 42) in `fine_tune_encoder_for_emotion_classification.py`.
- For larger global batch sizes (GPUs × per-device batch × grad_accum), consider LR scaling.

## Experiement 2, fine tune encoder-decoder for customer support converstation summarization

Plain Trainer (single-GPU or DataParallel if multiple GPUs are visible):

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

Accelerate (recommended DDP, multi-GPU):

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

Notes:
- Tokenization runs on CPU. For faster preprocessing, add `num_proc=$(nproc)` to `Dataset.map` in the script.
- We load model weights with safetensors. Prefer models that provide safetensors weights (e.g., most BART/T5 repos). If you only have `.bin` weights locally, either re-download safetensors from the Hub or upgrade PyTorch to >= 2.6.
- Outputs and logs are written under `--output_dir` (e.g., `./summarization-model`).
