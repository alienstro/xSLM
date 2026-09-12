# xSLM

A 70M parameter Llama-style language model, trained from scratch on one RTX 3090 in
180 minutes, for about one US dollar.

The model does not hold world knowledge. It learns grammar, style, and document
structure, and it invents facts. That is the expected result at this scale.

- Design: `docs/superpowers/specs/2026-09-12-xslm-design.md`
- Plan: `docs/superpowers/plans/2026-09-12-xslm-implementation.md`

## Setup

```bash
uv sync
cp .env.example .env   # then write the values
uv run pytest -q
```

## The pod run

Deploy an RTX 3090 pod in the RunPod console: Community Cloud, On-Demand, SSH terminal
access on, Jupyter off, container disk 20 GB, volume disk 30 GB. Set `HF_TOKEN`,
`HF_REPO_ID`, `MODEL_LICENSE`, `WANDB_API_KEY`, and `WANDB_PROJECT` in the template.

Write the SSH connection string that the console shows into `.env`:

```bash
POD_SSH="root@<address> -p <port> -i ~/.ssh/id_ed25519"
```

`pod.sh` reads `.env` itself. An exported `POD_SSH` wins over the file, which lets you
point one command at a second pod.

```bash
scripts/pod.sh check                            # nvidia-smi, cmake, tmux, rsync
scripts/pod.sh sync                             # rsync the working tree
scripts/pod.sh setup                            # uv sync, then pytest on CUDA
scripts/pod.sh run scripts/train_tokenizer.py   # 12 min
scripts/pod.sh run scripts/prepare_data.py      # 30 min, resumable
scripts/pod.sh run scripts/train.py --smoke 20  # read the throughput and the VRAM
# write max_steps into configs/base.yaml, then run sync again
scripts/pod.sh run scripts/train.py             # 180 min
scripts/pod.sh watch 20                         # poll about every 10 minutes
scripts/pod.sh run scripts/sample.py
scripts/pod.sh run scripts/push_to_hub.py
scripts/pod.sh run bash scripts/quantize.sh
```

The training run lives in `tmux`, so it survives a closed laptop and a dropped network.

## Layout

| Path | Holds |
|---|---|
| `configs/base.yaml` | Every hyperparameter, and no secret |
| `src/xslm/config.py` | The `LlamaConfig` factory |
| `src/xslm/model.py` | RMSNorm, RoPE, SwiGLU, attention, block, causal model |
| `src/xslm/data.py` | The memory-mapped packed dataset |
| `src/xslm/callbacks.py` | The time limit and the sample callbacks |
| `scripts/` | The six steps, plus the environment guard and the pod helper |
| `tests/` | The pytest suite, CPU only |
