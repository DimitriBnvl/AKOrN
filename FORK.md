# AKOrN – Fork Notes

This is a personal fork of [autonomousvision/akorn](https://github.com/autonomousvision/akorn).
The upstream repo is the canonical reference for the paper, model architecture, and CLEVRTex experiments.
This fork adds practical changes needed to run **sudoku evaluation across multiple GPUs** (including Kaggle multi-GPU notebooks) and to develop on macOS.

---

## Motivation

The upstream `eval_sudoku.py` is single-process and single-device.
Running `K=4096` energy-based voting on the full OOD test set takes several hours on one GPU.
This fork makes it possible to split that work across N GPUs—each handling a non-overlapping slice of batches—and aggregate the results afterward, without changing the model or the evaluation logic.

Secondary goals: fix a training CSV bug, add `--resume` to `train_sudoku.py`, and make the codebase run on macOS (MPS) or CPU without code changes.

---

## Changes vs upstream

| Area | Change |
|---|---|
| `eval_sudoku.py` | Device-agnostic: `torch.load(..., map_location=device)` and `.to(device)` everywhere instead of hardcoded `.cuda()` |
| `eval_sudoku.py` | `SHARD` / `NSHARD` env vars: each process handles only batch indices where `i % NSHARD == SHARD` |
| `eval_sudoku.py` | `MAXB` env var: stop after accumulating this many boards (0 = no cap; useful for smoke-tests) |
| `eval_sudoku.py` | Final print emits `corrects_vote` and `totals` alongside accuracy, so shard outputs can be aggregated |
| `train_sudoku.py` | `--resume` flag to continue training from a checkpoint |
| `train_sudoku.py` | CSV logging bug fix (missing flush / truncation issue) |
| `requirements.txt` | `tensorflow` instead of `tensorflow-cpu` (no macOS ARM wheel for the cpu variant) |
| `data/download_*.sh` | `curl` instead of `wget` for macOS compatibility |

---

## Multi-GPU evaluation with SHARD / NSHARD / MAXB

### Concept

`NSHARD` is the total number of parallel workers.
`SHARD` (0-indexed) is the index assigned to this worker.
Each worker processes only the batches where `batch_index % NSHARD == SHARD`, so there is no overlap and together they cover every batch exactly once.

`MAXB` caps the number of boards a single shard processes (useful for quick sanity checks).
Set `MAXB=0` (the default) to process the full slice.

### Single-GPU (default, no sharding)

```bash
python eval_sudoku.py \
  --data=ood \
  --model=akorn \
  --model_path=runs/sudoku_akorn/ema_99.pth \
  --T=128 --K=4096 --evote_type=sum
```

Output:
```
shard=0/1 corrects_vote=<N> totals=<T> acc=0.XXXX
```

### Multi-GPU on a single machine (e.g. 4 GPUs)

Run one process per GPU, setting `CUDA_VISIBLE_DEVICES` to pin each process to one device:

```bash
for SHARD in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$SHARD SHARD=$SHARD NSHARD=4 \
    python eval_sudoku.py \
      --data=ood \
      --model=akorn \
      --model_path=runs/sudoku_akorn/ema_99.pth \
      --T=128 --K=4096 --evote_type=sum \
    > shard_${SHARD}.log 2>&1 &
done
wait
```

Collect results:
```bash
grep "shard=" shard_*.log
# shard=0/4 corrects_vote=1821 totals=2250 acc=0.8093
# shard=1/4 corrects_vote=1834 totals=2250 acc=0.8151
# shard=2/4 corrects_vote=1828 totals=2250 acc=0.8124
# shard=3/4 corrects_vote=1815 totals=2248 acc=0.8072
```

Aggregate manually (totals may differ by 1 on the last shard due to dataset size):
```python
import re, glob
corrects, totals = 0, 0
for line in (open(f).read() for f in glob.glob("shard_*.log")):
    m = re.search(r"corrects_vote=(\d+) totals=(\d+)", line)
    if m:
        corrects += int(m.group(1))
        totals   += int(m.group(2))
print(f"Overall accuracy: {corrects/totals:.4f}  ({corrects}/{totals})")
```

### Kaggle notebook (2 GPUs)

Each cell runs on one GPU. Set the env vars at the top of each notebook cell before invoking `eval_sudoku.py`:

**GPU 0 cell:**
```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["SHARD"]  = "0"
os.environ["NSHARD"] = "2"
# os.environ["MAXB"] = "500"  # optional: cap for a quick test

import subprocess
result = subprocess.run([
    "python", "eval_sudoku.py",
    "--data=ood", "--model=akorn",
    "--model_path=runs/sudoku_akorn/ema_99.pth",
    "--T=128", "--K=4096", "--evote_type=sum",
], capture_output=True, text=True)
print(result.stdout)
```

**GPU 1 cell** (change `SHARD` to `"1"` and `CUDA_VISIBLE_DEVICES` to `"1"`).

Then aggregate as shown above.

### MAXB – quick sanity check

```bash
SHARD=0 NSHARD=1 MAXB=200 python eval_sudoku.py \
  --data=ood --model=akorn \
  --model_path=runs/sudoku_akorn/ema_99.pth \
  --T=128 --K=100 --evote_type=sum
# stops after accumulating 200 boards regardless of dataset size
```

---

## Training (with resume)

```bash
# Start
python train_sudoku.py \
  --exp_name=sudoku_akorn \
  --epochs=100 --lr=0.001 --T=16 \
  --use_omega=True --global_omg=True --init_omg=0.5 --learn_omg=True \
  --checkpoint_every=10 --eval_freq=10

# Resume from a checkpoint
python train_sudoku.py \
  --exp_name=sudoku_akorn \
  --resume=runs/sudoku_akorn/checkpoint_epoch_50.pth \
  --epochs=100 --lr=0.001 --T=16 \
  --use_omega=True --global_omg=True --init_omg=0.5 --learn_omg=True \
  --checkpoint_every=10 --eval_freq=10
```

---

## Environment setup

Same as upstream:

```bash
conda create -n akorn python=3.12 -y
conda activate akorn
pip install -r requirements.txt
```

Data download (macOS-compatible scripts using `curl`):
```bash
cd data && bash download_satnet.sh && bash download_rrn.sh && cd ..
```