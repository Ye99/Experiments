# Experiments: NLP with Transformers, plus multi-GPU training

Standalone scripts for three topics from
[*Natural Language Processing with Transformers*](https://github.com/nlp-with-transformers/notebooks) (fine-tuning, summarization, knowledge distillation), plus experiments with multi-GPU training using Hugging Face Accelerate.

| # | Script | What it does | Defaults |
|---|---|---|---|
| 1 | `fine_tune_encoder_for_emotion_classification.py` | Fine-tunes an encoder for 6-class emotion classification (book ch. 2) | `distilbert-base-uncased` on `emotion` |
| 2 | `fine_tune_encoder_decoder_for_custom_summarization.py` | Fine-tunes an encoder-decoder for summarization, scored with ROUGE (ch. 6) | `sshleifer/distilbart-cnn-12-6` on `cnn_dailymail` 3.0.0 |
| 3 | `distill_a_model.py` | Distills a large teacher into a smaller student (ch. 8) | `yoshitomo-matsubara/bert-base-uncased-sst2` → `distilbert-base-uncased` on GLUE SST-2 |
| 3 | `evaluate_distilled_model.py` | Measures accuracy, speed, size and GPU memory; compares teacher vs student | SST-2 validation |

## Setup

The environment is pinned with conda-lock (Linux x86_64 and macOS arm64):

```bash
mamba install -n base -c conda-forge conda-lock
conda-lock install --name LLM_experiments conda-lock.yml --mamba
mamba activate LLM_experiments
python check_torch.py   # prints torch version and device, runs a test matmul
```

- Without conda-lock (less exact): `mamba env create -f env.yml -n LLM_experiments`.
- To regenerate the lockfile: `conda-lock lock -f env.yml --platform osx-arm64 --platform linux-64`.
- The lock pins PyTorch 2.5.1. Recent `transformers` refuses to load `.bin`-only checkpoints on PyTorch < 2.6, so use models that ship `model.safetensors`. All the defaults here do.

## Experiment 1: fine-tune an encoder for emotion classification

Fine-tunes a classification head with `Trainer` and reports accuracy and weighted F1.

```bash
python fine_tune_encoder_for_emotion_classification.py --epochs 2 --batch_size 64                                  # DataParallel if >1 GPU visible
accelerate launch --multi_gpu fine_tune_encoder_for_emotion_classification.py --epochs 2 --batch_size 64           # DDP (recommended)
CUDA_VISIBLE_DEVICES=0 python fine_tune_encoder_for_emotion_classification.py --epochs 2 --batch_size 64           # single GPU
```

- Mixed precision is picked automatically on CUDA (bf16 if supported, else fp16). Override it with `--fp16` / `--bf16`.
- The model is saved to `<model>-finetuned-emotion/`. Add `--push_to_hub --hub_model_id <user>/<name>` to upload it.

### DataParallel vs DDP (2× RTX 4060 Ti, CUDA 12.4, batch_size=64, 1 epoch)

| Mode | Launch | Train runtime | Steps/s | Samples/s |
|---|---|---:|---:|---:|
| DataParallel | `python …` | ~29.7 s | ~4.2 | ~538 |
| DDP | `accelerate launch --multi_gpu …` | ~16.3 s | ~7.7 | ~984 |

DDP is ~1.8× faster. DataParallel runs one process that funnels every step through GPU 0. DDP runs one process per GPU and averages gradients directly between GPUs.

## Experiment 2: fine-tune an encoder-decoder for summarization

Fine-tunes a seq2seq model with `Seq2SeqTrainer`. It evaluates by generating summaries with beam search and scoring them with ROUGE-1/2/L/Lsum. The defaults are listed in the table above (batch 4 per GPU, lr 3e-5, 3 epochs, fp16, 4 beams).

```bash
python fine_tune_encoder_decoder_for_custom_summarization.py                            # DataParallel if >1 GPU visible
accelerate launch --multi_gpu fine_tune_encoder_decoder_for_custom_summarization.py     # DDP (recommended)
torchrun --nproc_per_node=2 fine_tune_encoder_decoder_for_custom_summarization.py       # DDP without Accelerate
```

To use other data, set the dataset and its columns. Pass `--dataset_config ""` if the dataset has no config (the default is CNN/DailyMail's `3.0.0`). For example, dialogue summarization:

```bash
accelerate launch --multi_gpu fine_tune_encoder_decoder_for_custom_summarization.py \
  --dataset_name knkarthick/dialogsum --dataset_config "" --text_column dialogue --summary_column summary
```

For a quick run, add `--max_train_samples 400 --max_eval_samples 40 --max_predict_samples 40`.

Result on the validation set, from DDP on 2 GPUs with `--num_train_epochs 1`, at step 12,000 of 35,890:

| ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum | eval loss |
|---:|---:|---:|---:|---:|
| 44.51 | 21.31 | 30.54 | 41.78 | 1.678 |

`distilbart-cnn-12-6` is already fine-tuned on CNN/DailyMail, and these scores match its [model card](https://huggingface.co/sshleifer/distilbart-cnn-12-6) (ROUGE-2 21.26, ROUGE-L 30.59). So the default run mainly checks that the pipeline works. The real use is fine-tuning on your own data.

## Experiment 3: distill a model

Distillation trains a small **student** to imitate a large, already fine-tuned **teacher**. The goal is a model that is much smaller and faster but nearly as accurate. On each batch, the frozen teacher and the student both predict, and the student learns from a mix of two losses:

```
loss = alpha_ce   * KL( softmax(teacher_logits / T) || softmax(student_logits / T) ) * T²   # imitate the teacher
     + alpha_hard * CrossEntropy(student_logits, labels)                                   # fit the true labels
```

- The temperature `T` (`--temperature`, default 2) softens both distributions, so the student learns how confident the teacher is, not just its top answer. `T²` keeps this loss on the same gradient scale as the hard loss.
- `--alpha_ce` / `--alpha_hard` (default 0.5 / 0.5) weight the two terms.
- The script evaluates every `--eval_steps` and at the end of each epoch, and keeps only the best student (by validation accuracy) in `--output_dir`.

```bash
python distill_a_model.py --fp16                                    # distill into ./distilled-sst2
python evaluate_distilled_model.py --model_path ./distilled-sst2    # one model
python evaluate_distilled_model.py \
  --teacher_path yoshitomo-matsubara/bert-base-uncased-sst2 \
  --student_path ./distilled-sst2                                   # teacher vs student, with speedup and size reduction
```

`distill_a_model.py` runs on a single device (CUDA, MPS or CPU). It doesn't use DDP, so don't launch it with `accelerate launch --multi_gpu`: that would start independent copies that all write to the same `--output_dir`.

### Results (one RTX 4060 Ti, default settings: 3 epochs, batch 32, T=2, α=0.5/0.5)

Accuracy is on the 872-sentence SST-2 validation set. Speed is inference throughput at batch 64 in fp32, after a warm-up pass.

| Model | Layers | Params | Size on disk | Peak GPU memory | Throughput | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Teacher (BERT-base) | 12 | 109.5 M | 418 MiB | 543 MiB | ~1,080 samples/s | 92.55% |
| Distilled student (DistilBERT) | 6 | 67.0 M | 256 MiB | 380 MiB | ~2,050 samples/s | 91.63% |
| **Student vs teacher** | **½** | **−39%** | **−39%** | **−30%** | **1.9× faster** | **−0.9 pt (keeps 99%)** |

The cost is paid once, during training. Every step also runs the teacher, so distillation trains at ~16 steps/s against ~26 for the student alone (3 epochs took ~7 min).

Limitation: the data is tokenized with the teacher's tokenizer and fed to both models, so teacher and student must share a vocabulary (e.g. BERT-uncased → DistilBERT-uncased).
