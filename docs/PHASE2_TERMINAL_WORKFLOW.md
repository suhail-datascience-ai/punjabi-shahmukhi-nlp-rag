# Working the Phase 2 pipeline from a terminal, staged fast → slow

This assumes Ubuntu 26.04, the RTX 3060 (12GB), 16GB system RAM box. Everything below is plain terminal — no Jupyter needed, which also makes it trivial to leave running unattended.

## 0. One-time setup

```bash
sudo apt update && sudo apt install -y tmux htop
git clone <your-repo-url> punjabi-nlp && cd punjabi-nlp
python3 -m venv venv && source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# confirm the GPU is actually visible to PyTorch before doing anything else
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
nvidia-smi   # should list the RTX 3060 and its driver/CUDA version
```

If `torch.cuda.is_available()` prints `False`, stop and fix the driver/CUDA install first — everything below silently falls back to CPU otherwise, which is unusable for Tier 1/2.

## 1. The golden rule: always run long jobs inside `tmux`

SSH/terminal disconnects (laptop sleeps, network blips) will kill a plain foreground process. `tmux` keeps it running regardless.

```bash
tmux new -s pretrain          # start a named session
source venv/bin/activate
python -m src.pretraining.train_mlm --config small 2>&1 | tee logs/train_small_$(date +%F).log
# press Ctrl+B then D to detach — the job keeps running
```

Reattach any time, from any terminal, even after closing your laptop lid and coming back:
```bash
tmux attach -t pretrain
```

List running sessions: `tmux ls`. Kill one when truly done: `tmux kill-session -t pretrain`.

`mkdir -p logs` once at the start — `tee` above writes a permanent log file *and* shows output live, so you don't lose the training curve if the pane scrolls away.

## 2. Watch the GPU while it trains (second terminal / tmux pane)

```bash
watch -n 2 nvidia-smi
```

Split panes instead of juggling windows: inside tmux, `Ctrl+B` then `"` (horizontal split) or `%` (vertical split). What to look for:
- **GPU-Util** should sit high (80-100%) once training is warmed up. If it's low, your dataloader is the bottleneck — increase `dataloader_num_workers` or check disk I/O.
- **Memory-Usage** should have headroom below the 12GB cap. If you see `CUDA out of memory`, halve `--per_device_train_batch_size` and double `--gradient_accumulation_steps` (keeps the effective batch size the same, uses less peak memory).

## 3. The staged plan, as literal commands

Run these in order. Don't skip to Tier 2 — each tier validates the one before it.

```bash
# ---- TIER 0: prove the pipeline works, ~10 minutes, on data you already have ----
python -m src.pretraining.quick_corpus
python -m src.pretraining.train_tokenizer --config tiny --corpus data/pretraining_raw/quick_corpus.txt
python -m src.pretraining.train_mlm --config tiny --corpus data/pretraining_raw/quick_corpus.txt --max_steps 2000
# check: did loss go down? python's stdout / your log file shows train_loss each logging_steps.
# If this fails, fix it here — don't debug a multi-hour Tier 1 run instead.

# ---- TIER 1: your first real, publishable checkpoint ----
# (needs internet on this machine — the corpus download step)
python -m src.pretraining.build_pretraining_corpus --download
python -m src.pretraining.build_pretraining_corpus --extract
python -m src.pretraining.build_pretraining_corpus --clean
python -m src.pretraining.train_tokenizer --config small
tmux new -s tier1
python -m src.pretraining.train_mlm --config small 2>&1 | tee logs/tier1_$(date +%F).log
# Ctrl+B, D to detach. Budget roughly 4-10 hours — check back with `tmux attach -t tier1`.

# ---- TIER 2: Sindhi-scale target, once Tier 1's loss curve has flattened ----
python -m src.pretraining.train_tokenizer --config compact
tmux new -s tier2
python -m src.pretraining.train_mlm --config compact 2>&1 | tee logs/tier2_$(date +%F).log
# multi-day — see section 5 for surviving reboots.
```

## 4. Sanity-check a checkpoint without waiting for training to finish

In a second pane, while Tier 1/2 trains in the first:

```bash
python -c "
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
tok = AutoTokenizer.from_pretrained('models/punjabi_shahmukhi_encoder_small/checkpoint-2000')
model = AutoModelForMaskedLM.from_pretrained('models/punjabi_shahmukhi_encoder_small/checkpoint-2000')
text = 'پنجاب دی راجدھانی [MASK] اے۔'
inputs = tok(text, return_tensors='pt')
mask_idx = (inputs['input_ids'][0] == tok.mask_token_id).nonzero()[0].item()
with torch.no_grad():
    logits = model(**inputs).logits
top5 = torch.topk(logits[0, mask_idx], 5).indices
print([tok.decode([t]) for t in top5])
"
```

If the model is learning, the top-5 predictions for a `[MASK]` slot should start looking like plausible Punjabi words (not random subword fragments) by a few thousand steps in.

## 5. Surviving a reboot mid-training

Checkpoints save every 2000 steps by default (`save_steps=2000` in `train_mlm.py`) to `models/punjabi_shahmukhi_encoder_<tier>/checkpoint-N/`. After any interruption:

```bash
ls models/punjabi_shahmukhi_encoder_small/   # find the latest checkpoint-N
tmux new -s tier1_resume
python -m src.pretraining.train_mlm --config small \
    --resume_from_checkpoint models/punjabi_shahmukhi_encoder_small/checkpoint-8000 \
    2>&1 | tee -a logs/tier1_$(date +%F).log
```

For genuine multi-day unattended runs, also disable suspend so the machine doesn't sleep mid-training:
```bash
systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
# undo later with: systemctl unmask sleep.target suspend.target hibernate.target hybrid-sleep.target
```

## 6. Keep an eye on disk — 1.5TB is plenty, but corpora + checkpoints add up

```bash
df -h /                      # overall free space
du -sh data/ models/ logs/   # what's actually using it
```

`save_total_limit=3` in `train_mlm.py` already caps how many checkpoints pile up per run — old ones auto-delete as new ones save.

## 7. When a tier is done

```bash
python -m src.pretraining.compare_encoders    # head-to-head vs xlm-roberta-base, real numbers
```

Copy the printed table straight into `docs/MODEL_CARD.md`, commit, and only then decide whether the result justifies moving up a tier.
