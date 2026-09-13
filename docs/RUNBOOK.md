# xSLM Run Book

The ordered procedure for one training run, and every trap that this project met on
its first run. Follow the steps in order. The traps table gives the symptom, so that
you can find an entry from an error message.

## 1. Before you rent a pod

Run the suite on your own machine. It costs nothing and it catches most defects.

```bash
uv sync
uv run pytest -q          # 94 tests, under 10 seconds
```

Write `.env` from `.env.example`. The file holds every setting, and the repository
never holds it.

| Variable | Where to read it |
|---|---|
| `HF_TOKEN` | huggingface.co → Settings → Access Tokens. Fine-grained, write scope, limited to one repository |
| `HF_REPO_ID` | The repository you want, for example `you/xslm-70m` |
| `MODEL_LICENSE` | `apache-2.0` unless you have a reason |
| `WANDB_API_KEY` | wandb.ai → Settings. Optional; the run continues without it |
| `RUNPOD_API_KEY` | console.runpod.io → Settings → API Keys |
| `POD_SSH` | Step 2 gives it |
| `RUNPOD_POD_ID` | Step 2 gives it |

Add `~/.ssh/id_ed25519.pub` to RunPod → Settings → SSH Public Keys. Do this once.

## 2. Deploy the pod

| Field | Value | Reason |
|---|---|---|
| GPU | One RTX 3090, 24 GB | The compute budget assumes this card |
| Cloud | Community | About half the price of Secure Cloud |
| Pricing | **On-Demand** | Spot is reclaimed after 5 seconds of notice |
| Template | `runpod/pytorch`, CUDA 12.1 or later | Carries the driver and a working `sshd` |
| **Expose TCP ports** | **22** | Without this you get the proxy only. See trap 1 |
| Container disk | 20 GB | RunPod erases this disk on every container restart |
| **Volume disk** | **30 GB or more, at `/workspace`** | Survives a restart. See trap 2 |
| SSH terminal access | On | |
| Start Jupyter notebook | Off | An unused service |

Then read the connection string:

```bash
uv run --env-file .env scripts/runpod.py list   # gives the pod identifier
# write RUNPOD_POD_ID into .env, then:
scripts/pod.sh reconnect                        # writes POD_SSH into .env
```

## 3. Run

```bash
scripts/pod.sh check          # nvidia-smi, cmake, tmux, rsync; installs what is absent
scripts/pod.sh sync           # rsync the working tree to /workspace/xSLM
scripts/pod.sh secrets        # copy .env to the pod with mode 600
scripts/pod.sh setup          # uv sync, then pytest on CUDA

scripts/pod.sh run bash scripts/watchdog.sh 300 15    # arm the safety first
scripts/pod.sh run bash scripts/run_data_pipeline.sh  # steps 1 and 2, about 30 min

scripts/pod.sh watch run_data_pipeline 1              # poll; also refreshes the heartbeat

scripts/pod.sh run scripts/train.py --smoke 20        # read the throughput and the VRAM
# write max_steps into configs/base.yaml, then sync again
scripts/pod.sh run scripts/train.py                   # 180 min
scripts/pod.sh run scripts/sample.py
scripts/pod.sh run scripts/push_to_hub.py
scripts/pod.sh run bash scripts/quantize.sh
scripts/pod.sh verify                                 # eight files must be on the Hub
scripts/pod.sh terminate --yes                        # only after verify exits zero
```

> Warning: poll with `scripts/pod.sh watch` at least every 15 minutes. Every watch
> refreshes `out/HEARTBEAT`. The watchdog stops the pod when that file goes stale,
> because a silent agent means nobody will publish the model.

## 4. The safety net

`scripts/watchdog.sh` runs on the pod, so it works when the laptop sleeps and the
agent session ends. It fires on any of three conditions:

| Condition | Meaning |
|---|---|
| `out/PIPELINE_DONE` exists | The pipeline finished |
| The deadline passed, 300 minutes by default | A hard ceiling on the bill |
| `out/HEARTBEAT` is older than 15 minutes | The agent stopped checking in |

It then asks `verify_upload.py` whether all eight files reached the Hub.

- Every file present: **terminate**. Nothing is left to lose.
- Any file absent: **stop**. The GPU charge ends and the volume keeps the work.

## 5. Traps

| # | Symptom | Cause and fix |
|---|---|---|
| 1 | `Error: Your SSH client doesn't support PTY` | `POD_SSH` points at `ssh.runpod.io`, the proxy. It serves interactive terminals only and carries no `rsync`. Use the "SSH over exposed TCP" string, which needs TCP port 22 exposed at deploy time |
| 2 | `/workspace/xSLM: No such file or directory` after a while | The container restarted and erased the container disk. Deploy with a volume disk, and keep every file under `/workspace` |
| 3 | `ssh: Could not resolve hostname ssh` | `POD_SSH` kept the word `ssh` from the pasted console command. `pod.sh` strips it now |
| 4 | `ssh: connect to host ... port 22: Operation timed out` during `sync` | `rsync` did not get the `-p` and `-i` flags. `pod.sh` passes them inside its `-e` string now |
| 5 | `ssh: connect to host ... Connection refused`, but the pod is RUNNING | The container restarted and RunPod mapped a new public port. Run `scripts/pod.sh reconnect` |
| 6 | `duplicate session: bash` | Two jobs took the same `tmux` name. `pod.sh` names each session after the script, not the interpreter |
| 7 | One log line of several megabytes | `tqdm` writes with a carriage return. `pod.sh watch` splits them now |
| 8 | `RunPod answered 403: error code: 1010` | Cloudflare blocks a request with no `User-Agent`. `runpod.py` sends one now |
| 9 | The pod environment fields are empty | The RunPod template held no variables. Run `scripts/pod.sh secrets` |

## 6. Traps in the model code

These cost nothing on this run because tests caught them, but a rewrite would meet
them again.

| Symptom | Cause |
|---|---|
| The loss falls normally and the samples read oddly | The dataset shifted the labels, and `LlamaForCausalLM` shifts them again inside its loss. Never shift in the dataset |
| An out-of-memory error at micro-batch 32 | The logit tensor sets the peak VRAM, not the optimizer. Three copies of `batch * 1024 * 32768` cost about 8.5 GB |
| The disk fills at hour two | The `Trainer` default keeps every checkpoint. Set `save_total_limit` |
| `AttributeError: 'LlamaConfig' object has no attribute 'rope_theta'` | transformers 5 moved the value into `rope_parameters` |
| The parameter count is `vocab_size * hidden_size` too high | Weight tying needs `_tied_weights_keys` on the class |
| `generate` raises `AttributeError` | transformers 5 removed `GenerationMixin` from `PreTrainedModel`. Inherit it explicitly |
| `convert_hf_to_gguf.py` writes a file that generates nonsense | `rotate_half` used an interleaved layout. The converter assumes the half-split Llama layout |

## 7. Resuming after a stop

A stop keeps the volume, so the tokenizer and the token files survive.

```bash
# Start the pod again in the console, then:
scripts/pod.sh reconnect      # the public port changed
scripts/pod.sh check
scripts/pod.sh sync
scripts/pod.sh setup
scripts/pod.sh run bash scripts/run_data_pipeline.sh   # skips a tokenizer that exists
```

`prepare_data.py` reads the size of `train.bin` and continues from that point.
`train.py` resumes from the newest checkpoint under `out/`.

## 8. Running the GGUF file

Recent llama.cpp builds give `llama-cli` a conversation prompt, and raw completion
moved to `llama-completion`. A base model holds no chat template, so use the
completion tool.

Build with the two flags that the download needs. `LLAMA_CURL` fetches the file,
and `LLAMA_OPENSSL` gives it HTTPS. Without the second one the tool prints
`HTTPS is not supported`, which reads as a network fault but is a build fault.

```bash
apt-get install -y cmake libcurl4-openssl-dev libssl-dev
git clone --depth 1 https://github.com/ggml-org/llama.cpp
cmake -S llama.cpp -B llama.cpp/build \
    -DGGML_CUDA=OFF -DLLAMA_CURL=ON -DLLAMA_OPENSSL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp/build --target llama-completion llama-tokenize -j2
```

`-hf` reads the file from the Hub, so no download step is needed. The quant tag
after the colon picks the file, and it finds the `gguf` folder by itself.

```bash
llama.cpp/build/bin/llama-completion -hf Alienstro/xslm:BF16 \
    -p 'The history of' -n 120 --temp 0.8 --top-p 0.95
```

To read a local file instead, name it with `-m`:

```bash
llama.cpp/build/bin/llama-completion -m xslm-70m-BF16.gguf \
    -p 'The history of' -n 120 --temp 0.8 --top-p 0.95
```

### Proving that llama.cpp splits the text as the training did

`tokenizer.ggml.pre` chooses the regex list that llama.cpp applies. A wrong value
raises no error. It only splits the text differently from the training, which
shows as a quiet loss of quality. Compare the two tokenizers instead of trusting
the value:

```bash
llama.cpp/build/bin/llama-tokenize -m xslm-70m-BF16.gguf --ids \
    -p 'In 1822 the cost was 3.50 dollars, up 12345 percent.'
```

```python
from transformers import AutoTokenizer
print(AutoTokenizer.from_pretrained("Alienstro/xslm").encode(
    "In 1822 the cost was 3.50 dollars, up 12345 percent."))
```

The two lists must match exactly. Each digit must take one token, because the
tokenizer holds `Digits(individual_digits=True)`.

## 9. Running the tuned model

The tuned model answers an instruction. The base model only continues text.

```bash
llama cli --jinja -hf Alienstro/xslm --hf-file gguf/xslm-70m-instruct-Q8_0.gguf
```

> Warning: the `--jinja` flag is not optional. llama.cpp reads a short list of well
> known templates on its old path, and it refuses every other template with
> `this custom template is not supported, try using --jinja`. The flag turns on the
> engine that reads the template of this model.

For the base model, which holds no template, use the completion tool instead:

```bash
llama completion -hf Alienstro/xslm:Q8_0 -p 'The history of' -n 120
```

| Model | File | Tool |
|---|---|---|
| Tuned | `gguf/{name}-instruct-Q8_0.gguf` | `llama cli --jinja` |
| Base | `gguf/{name}-Q8_0.gguf` | `llama completion` |
