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

## Run: fine-tune (single- or multi-GPU)

Plain Trainer (will use multiple GPUs via DataParallel if visible):

```bash
python fine_tune_transformers.py --epochs 2 --batch_size 64
```

Accelerate (recommended DDP):

```bash
accelerate launch --multi_gpu fine_tune_transformers.py --epochs 2 --batch_size 64
```

Tips:
- Force a specific GPU: `CUDA_VISIBLE_DEVICES=0 python fine_tune_transformers.py ...`
- Mixed precision is auto-selected: bf16 if supported, else fp16 on CUDA; disabled on CPU.
- Outputs are saved under `<model>-finetuned-emotion/` (ignored by `.gitignore`).

## Performance comparison (2× RTX 4060 Ti, CUDA 12.4, batch_size=64, 1 epoch)

| Mode | Command | Train runtime | Steps/s | Samples/s |
|---|---|---:|---:|---:|
| DataParallel (Trainer, plain `python`) | `python fine_tune_transformers.py --epochs 1 --batch_size 64` | ~29.74 s | ~4.20 | ~538 |
| DDP (Accelerate) | `accelerate launch --multi_gpu fine_tune_transformers.py --epochs 1 --batch_size 64` | ~16.26 s | ~7.69 | ~984 |

- Speedup (DDP vs DP): ~1.8× on this setup.

## Reproducibility

- Seed is set via `--seed` (defaults to 42) in `fine_tune_transformers.py`.
- For larger global batch sizes (GPUs × per-device batch × grad_accum), consider LR scaling.
