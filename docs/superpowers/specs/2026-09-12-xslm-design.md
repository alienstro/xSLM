# xSLM Design Spec

Date: 2026-09-12
Status: Approved for planning

## 1. Purpose

Build a small language model from scratch to learn the full training pipeline. The
person writes the tokenizer training, the data pipeline, the model modules, and the
training configuration. The person trains the model on one RTX 3090 through RunPod.
The total pod time is four hours and 13 minutes.

### 1.1 Explicit non-goal

The model will not hold world knowledge. A model of this size learns grammar, style,
and document structure. It invents facts. This is the expected result at this scale.

If a knowledgeable model is the goal, the correct methods are retrieval-augmented
generation over a document index, or a QLoRA fine-tune of an existing 1B to 8B base
model. Both run on the same 3090. This spec does not cover them.

### 1.2 Success criteria

1. The training run completes inside the time limit and saves a checkpoint.
2. The validation loss decreases smoothly and stays below 3.8 nats per token.
3. `sample.py` produces grammatical English paragraphs of at least 100 tokens.
4. The full test suite passes on CPU and on CUDA.
5. The person can explain each file in the repository.
6. The HuggingFace repository holds the BF16 weights and the four GGUF files.

## 2. Compute budget

The RTX 3090 gives about 35 TFLOP/s in BF16. A realistic model FLOPs utilization on
this card is 30 to 40 percent. The budget therefore uses 25 TFLOP/s.

The training loop gets 180 minutes, which is 10,800 seconds. This gives 2.7e17 FLOPs.

Transformer training costs about `6 * parameters * tokens` FLOPs. The Chinchilla rule
sets `tokens = 20 * parameters`. Solve the two equations:

```
120 * N^2 = 2.7e17
N         = 4.7e7 non-embedding parameters
D         = 9.5e8 tokens
```

The design targets 49.6M non-embedding parameters and 0.6B to 0.9B tokens. The exact
token count depends on the measured throughput.

## 3. Model architecture

The model is a Llama-style causal decoder with pre-normalization.

| Field | Value | Reason |
|---|---|---|
| `d_model` | 640 | 10 heads at 64 dimensions each |
| `n_layers` | 10 | Gives 49.6M non-embedding parameters |
| `n_heads` | 10 | Grouped-query attention adds no value at this size |
| `head_dim` | 64 | The size that FlashAttention kernels prefer |
| `ffn_hidden` | 1728 | 2.7 times `d_model`, rounded to a multiple of 64 |
| `vocab_size` | 32768 | Matches the tokenizer |
| `max_seq_len` | 1024 | A longer context costs tokens the budget does not have |
| Normalization | RMSNorm, epsilon 1e-5 | Cheaper than LayerNorm, no bias term |
| Position encoding | RoPE, theta 10000 | Applied to queries and keys only |
| Activation | SwiGLU | Gate, up, and down projections, all without bias |
| Attention | `F.scaled_dot_product_attention` | Selects the FlashAttention kernel, `is_causal=True` |
| Biases | None anywhere | Standard for modern decoders |
| Embeddings | Tied input and output | Saves 21M parameters |

### 3.1 Parameter count

| Part | Count |
|---|---|
| Attention, each layer | 4 * 640 * 640 = 1,638,400 |
| Feed-forward, each layer | 3 * 640 * 1728 = 3,317,760 |
| Norms, each layer | 1,280 |
| One layer | 4,957,440 |
| Ten layers | 49,574,400 |
| Final norm | 640 |
| Non-embedding total | 49,575,040 |
| Tied embedding | 32768 * 640 = 20,971,520 |
| Grand total | 70,546,560 |

A unit test asserts that the non-embedding count stays between 49M and 51M.

### 3.2 Initialization

All linear weights use a normal distribution with mean 0 and standard deviation 0.02.
The output projection of each attention block and each feed-forward block uses a
smaller standard deviation of `0.02 / sqrt(2 * n_layers)`. This keeps the variance of
the residual stream stable as depth increases.

### 3.3 HuggingFace integration

The `Trainer` class needs a `PreTrainedModel`. The model must also convert to GGUF
without a custom converter. One decision answers both needs: the model uses the Llama
parameter names.

The architecture in section 3 is a Llama architecture. The design therefore keeps the
hand-written modules and changes only the names.

- `XSLMConfig` is a factory. It returns a `LlamaConfig` that holds the fields of the
  table above. The saved `config.json` declares
  `"architectures": ["LlamaForCausalLM"]`.
- `XSLMForCausalLM` subclasses `LlamaPreTrainedModel`. It returns
  `CausalLMOutputWithPast` with a `loss` field when the caller supplies `labels`.

The parameter names are:

```
model.embed_tokens.weight
model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
model.layers.{i}.mlp.{gate,up,down}_proj.weight
model.layers.{i}.input_layernorm.weight
model.layers.{i}.post_attention_layernorm.weight
model.norm.weight
```

`tie_word_embeddings` is true, so the state dictionary holds no `lm_head.weight`.

The person still writes the RMSNorm, the RoPE, the SwiGLU, the attention, and the
block by hand. Only the names come from Llama. A test asserts that the state
dictionary matches `LlamaForCausalLM` key for key and shape for shape. If that test
passes, `convert_hf_to_gguf.py` cannot fail because of the architecture. A reader also
loads the model with `AutoModelForCausalLM` and needs no `trust_remote_code` flag.

## 4. Tokenizer

| Field | Value | Reason |
|---|---|---|
| Library | `tokenizers` | The Rust backend that `transformers` wraps |
| Algorithm | Byte-level BPE | No unknown token is possible, every byte round-trips |
| `vocab_size` | 32768 | A power of two, keeps the tied embedding at 21M parameters |
| Pre-tokenizer | `ByteLevel(add_prefix_space=False)` plus `Digits(individual_digits=True)` | Splitting digits helps the model read numbers |
| Training text | 2 GB sampled from the training corpus | Matches the training distribution |
| Special tokens | `<\|endoftext\|>` at id 0 | One document separator, nothing else |
| Output | `tokenizer.json` | Loads through `PreTrainedTokenizerFast` |

There is no padding token. The data pipeline packs sequences, so no batch ever pads.

The tokenizer trains on the pod in about 12 minutes.

## 5. Data pipeline

The corpus is `HuggingFaceFW/fineweb-edu`, configuration `sample-10BT`. The pipeline
streams it and never downloads the full sample.

Steps:

1. Open the dataset with `datasets.load_dataset(..., streaming=True)`.
2. Tokenize each document and append `<|endoftext|>` after it.
3. Write the token identifiers to one flat `uint16` array on disk. The vocabulary is
   32768, which is below 65536, so `uint16` is safe. One billion tokens occupy 2 GB.
4. Reserve the first 5M tokens as `val.bin`. Write the rest to `train.bin`.
5. Stop when `train.bin` reaches the configured token target.

The script is resumable. It reads the size of `train.bin` at start and continues from
that point. A pod failure at minute 25 does not waste the tokenizer step.

### 5.1 Batching

The training dataset reads `train.bin` through `numpy.memmap`. The operating system
page cache holds the hot pages, so the 2 GB file needs no resident memory.

Each sample picks a random offset and reads 1024 tokens. The sample sets `labels` equal
to `input_ids`. There is no padding, no attention mask, and no wasted computation.

> Warning: do not shift the labels in the dataset. `LlamaForCausalLM` shifts the logits
> and the labels inside its loss function. A second shift in the dataset makes the model
> predict two positions ahead. The loss still falls and the samples still read as
> English, so no test that only watches the loss finds this defect.

A test asserts that the loss of the model equals a hand-written cross entropy between
`logits[:, :-1]` and `input_ids[:, 1:]`. This test finds a double shift.

## 6. Training configuration

| Field | Value |
|---|---|
| Optimizer | AdamW, betas 0.9 and 0.95, epsilon 1e-8 |
| Weight decay | 0.1 on matrices, 0.0 on norms and embeddings |
| Peak learning rate | 1e-3 |
| Schedule | 200 steps of linear warmup, then cosine decay to 1e-4 |
| Gradient clipping | 1.0 on the global norm |
| Precision | BF16 mixed, `bf16=True` |
| Micro-batch size | 16 sequences, raised after the smoke test reports the true VRAM use |
| Gradient accumulation | 16 steps |
| Tokens per optimizer step | 16 * 16 * 1024 = 262,144 |
| `save_total_limit` | 2 |
| Gradient checkpointing | Off. It saves memory the budget does not need and costs 30 percent speed |
| `torch.compile` | Off by default. It is an optional flag, because compilation can fail on a 3090 driver |
| Dataloader workers | 4 |
| `report_to` | `["wandb"]` when `WANDB_API_KEY` exists, else `[]` |

The logit tensor sets the peak VRAM, not the optimizer. The tensor holds
`micro_batch * 1024 * 32768` elements. At micro-batch 32 that is 1.07e9 elements, which
is 2.1 GB in BF16. The cross-entropy loss upcasts the tensor to FP32, which adds 4.3 GB,
and the backward pass adds the gradient. The three copies cost about 8.5 GB, and the
total approaches 18 GB of the 24 GB card.

The optimizer state costs 1.1 GB: 282 MB of FP32 weights, 282 MB of gradients, and
564 MB for the two AdamW moments.

The design therefore starts at micro-batch 16 and keeps the same 262,144 tokens for
each optimizer step. The peak VRAM falls to about 8 GB. Section 6.1 raises the value
after the smoke test measures the true use.

`save_total_limit` is 2. The `Trainer` default keeps every checkpoint. One checkpoint
holds 282 MB of weights and 564 MB of optimizer state. Ten checkpoints would waste
8.5 GB and could fill the disk during the run.

### 6.1 Setting the step count

The cosine schedule must end when training ends, so `max_steps` must be correct before
the full run starts.

The procedure is:

1. Run 20 steps and read `train_tokens_per_second` from the log.
2. Compute `max_steps = (throughput * 10800) / 262144`.
3. Write that number into the configuration and start the full run.

The starting estimate is 2,600 steps. The measured value replaces it.

The smoke test also prints `torch.cuda.max_memory_allocated()`. If the peak stays below
12 GB, raise the micro-batch to 24 and halve the accumulation to keep the tokens for
each optimizer step at 262,144.

### 6.2 Callbacks

- `TimeLimitCallback` sets `control.should_training_stop = True` after
  `time_limit_minutes`. This guarantees a saved checkpoint inside the budget.
- `SampleGenerationCallback` generates 100 tokens every 500 steps and writes them to
  the log. This shows progress that a loss number alone does not show.

### 6.3 Evaluation

The run evaluates on `val.bin` every 250 steps and reports the validation loss and the
perplexity. There is no benchmark suite. Benchmarks such as MMLU and HellaSwag return
chance-level scores at this scale and would waste time.

## 7. Repository layout

```
xSLM/
  pyproject.toml            uv project, Python pinned to 3.12
  configs/base.yaml         every hyperparameter in one place
  src/xslm/
    config.py               XSLMConfig
    model.py                RMSNorm, RoPE, SwiGLU, Attention, Block, XSLMForCausalLM
    data.py                 memmap packed dataset and collator
    callbacks.py            TimeLimitCallback, SampleGenerationCallback
  scripts/
    pod.sh                  the ssh and rsync helper for the pod
    _env.py                 reads one environment variable and checks that it exists
    train_tokenizer.py      step 1
    prepare_data.py         step 2
    train.py                step 3
    sample.py               step 4
    push_to_hub.py          step 5
    quantize.sh             step 6
  tests/                    pytest, runs on CPU
  .env.example              the names of the secrets, with no values
  README.md
```

> Warning: the system Python on the development machine is 3.14.7. Several
> dependencies publish no wheels for 3.14. The project pins Python 3.12 through `uv`,
> so no command touches the system Python.

## 8. Test plan

Every test builds a tiny model with 2 layers and `d_model` 64 and runs on CPU. The full
suite finishes in under 30 seconds.

| Test | Assertion |
|---|---|
| RoPE rotation | The dot product of two vectors does not change when both positions shift by the same amount |
| Causal mask | Changing the token at position `t` does not change the logits at any position below `t` |
| Forward shapes | The logits have shape `(B, T, vocab)` and the loss is a scalar |
| Weight tying | `lm_head.weight` is the same object as the embedding weight |
| Label alignment | The model loss equals a hand-written cross entropy between `logits[:, :-1]` and `input_ids[:, 1:]` |
| Memmap bounds | No sampled offset reads past the end of the file |
| Overfit | 50 steps on 20 tokens drive the loss below 0.1 |
| Parameter count | The production configuration reports 49M to 51M non-embedding parameters |
| Llama parity | The state dictionary keys and shapes equal those of `LlamaForCausalLM` |

The overfit test is the most important one. A model that cannot memorize 20 tokens has
a broken gradient path, and no amount of GPU time will fix it.

## 9. Execution plan

All six steps run on one RunPod 3090 pod. A separate CPU pod saves about 0.15 US
dollars and adds a network volume, a second machine, and a transfer step. The extra
failure surface costs more than the saving.

Optional and free: run `pytest` on the development machine before renting the pod.

| Clock | Step | Duration |
|---|---|---|
| 0:00 | Start the pod, rsync the repository, `uv sync`, run `nvidia-smi` | 5 min |
| 0:05 | Run `pytest` on CUDA | 3 min |
| 0:08 | Run `train_tokenizer.py` | 12 min |
| 0:20 | Run `prepare_data.py` | 30 min |
| 0:50 | Run the 20-step smoke test, then set `max_steps` | 3 min |
| 0:53 | Run the full training | 180 min |
| 3:53 | Run `sample.py`, then push the BF16 weights with `push_to_hub.py` | 7 min |
| 4:00 | Build the `llama-quantize` target of `llama.cpp`. The build needs no CUDA | 5 min |
| 4:05 | Run `quantize.sh`: convert to GGUF, then quantize three times | 4 min |
| 4:09 | Push the `gguf/` folder, download the checkpoint, stop the pod | 4 min |

The total is 4 hours and 13 minutes. The estimated cost is about 1.05 US dollars.
The design does not shorten the training loop to save those 13 minutes, because the
180 minute loop sets the final loss.

### 9.1 Pod requirements

| Field | Value | Reason |
|---|---|---|
| GPU | One RTX 3090, 24 GB | The compute budget of section 2 assumes this card |
| GPU count | 1 | The spec excludes multi-GPU training |
| Cloud | Community | About half the price of Secure Cloud |
| Instance pricing | On-Demand | Never Spot |
| Image | `runpod/pytorch`, CUDA 12.1 or later | Supplies the CUDA driver and a working `sshd` |
| Container disk | 20 GB | RunPod erases this disk when the pod stops |
| Volume disk | 30 GB, mounted at `/workspace` | This disk survives a pod stop |
| SSH terminal access | On | The control channel of section 14 |
| Start Jupyter notebook | Off | The agent works over SSH, so Jupyter adds an unused service |

> Warning: never select Spot or Interruptible pricing. RunPod reclaims a Spot pod after
> about 5 seconds of notice. The 180 minute loop would stop at an arbitrary point.

The project lives in `/workspace/xSLM`. The `.venv`, the `data` folder, and the `out`
folder all live under that path. A pod stop therefore loses no tokens and no
checkpoint, and the resume control of section 10 works.

`pod.sh` runs two checks after the pod boots:

```bash
nvidia-smi          # the driver is 525 or later, and 24576 MiB is free
cmake --version     # the image can build llama-quantize
```

If `cmake` is absent, `pod.sh` installs it with `apt-get`. The check costs three
seconds and prevents a failure at minute 240.

## 10. Risks

| Risk | Effect | Control |
|---|---|---|
| The pod stops during data preparation | The tokens are lost | `prepare_data.py` resumes from the current file size |
| The measured throughput is far below the estimate | The cosine schedule does not finish | The smoke test measures throughput before the full run sets `max_steps` |
| An out-of-memory error occurs | The run stops | Start at micro-batch 32 and raise it only after the smoke test reports the VRAM use |
| The loss diverges | The run wastes time | Gradient clipping at 1.0, plus 200 warmup steps. If the loss still diverges, halve the learning rate |
| `torch.compile` fails on the driver | The run stops at start | The flag is off by default |
| The HuggingFace stream rate-limits | Data preparation slows | Set `HF_TOKEN` in the pod environment |
| The HuggingFace token has no write scope | The weights stay on the pod | `_env.py` checks the token at the start of every script. The BF16 push happens before the quantization step |
| The Q4_K_M file produces broken text | The smallest file is not usable | This result is expected at 70M parameters. The model card states it. Q8_0 is the recommended file |

## 11. Out of scope

Instruction tuning, chat formatting, reinforcement learning from human feedback,
multi-GPU training, model serving, and any retrieval system. Each is a separate project
with its own spec.

Quantization was out of scope in the first version of this spec. Section 13 brings it
in scope.

## 12. Secrets and environment

The project holds three configuration files. No secret ever enters `configs/base.yaml`,
and no secret ever enters the repository.

| File | Committed | Holds |
|---|---|---|
| `configs/base.yaml` | Yes | Every hyperparameter |
| `.env.example` | Yes | The names of the secrets, with no values |
| `.env` | No | The real values on the development machine |

`.env.example` holds these names:

```bash
# A fine-grained token with write scope, limited to HF_REPO_ID only
HF_TOKEN=
HF_REPO_ID=
MODEL_LICENSE=apache-2.0

# Weights and Biases. The training run continues if this value is absent.
WANDB_API_KEY=
WANDB_PROJECT=xslm

# The SSH connection string that the RunPod console shows. Keep the quotes,
# because the value holds spaces. pod.sh reads this file.
POD_SSH=""
```

`POD_SSH` is not a secret, but it changes with every pod, so it belongs with the other
settings and not in the repository. `pod.sh` reads `.env` directly, because
`uv run --env-file` reaches Python scripts only. An exported value wins over the file.

> Warning: a fine-grained token limits the damage if the value leaks. A classic write
> token can rewrite every repository that the account owns. Use a fine-grained token.

### 12.1 How a script reads the environment

The command is `uv run --env-file .env scripts/train.py`. The installed uv is 0.11.17,
which supports `--env-file`. The project therefore needs no `python-dotenv` dependency
and no `load_dotenv()` call.

`scripts/_env.py` supplies one function, `require(name)`. The function reads
`os.environ`, and raises a `RuntimeError` with the fix instruction if the value is
absent. Every script calls it at import time. A missing token then fails in one second,
not after a 30 minute data run.

No script prints a token. `push_to_hub.py` prints the repository identifier only.

### 12.2 The pod

The person does not copy `.env` to the pod. The person types `HF_TOKEN` and
`HF_REPO_ID` into the environment fields of the RunPod template. RunPod injects the
values into the container, and no file on the pod holds them. The scripts then run
without the `--env-file` flag.

This spec configures no RunPod API key, because the person starts the pod through the
RunPod web console. Pod automation is a separate project.

### 12.3 MCP credentials

A credential for an MCP server belongs in `.mcp.json` or in `~/.claude.json`. It does
not belong in `.env`. Write the value as `${VARIABLE}` in `.mcp.json`, so that the file
stays safe to commit.

## 13. Publication and quantization

The target is one HuggingFace repository. The identifier comes from `HF_REPO_ID`.

```
<HF_REPO_ID>
  config.json
  model.safetensors        141 MB, BF16
  tokenizer.json
  README.md
  gguf/
    xslm-70m-BF16.gguf     141 MB
    xslm-70m-Q8_0.gguf      75 MB
    xslm-70m-Q6_K.gguf      58 MB
    xslm-70m-Q4_K_M.gguf    45 MB
```

The four GGUF names are the names that `llama-quantize` uses. `q8`, `q6`, and `q4` are
not valid names.

### 13.1 The two scripts

`scripts/push_to_hub.py` pushes the safetensors file, the `config.json`, the
`tokenizer.json`, and the generated model card. The script is idempotent, so a second
run after a failure is safe.

`scripts/quantize.sh` clones `llama.cpp`, builds the `llama-quantize` target, runs
`convert_hf_to_gguf.py`, runs `llama-quantize` three times, and pushes the `gguf/`
folder.

### 13.2 The model card

`push_to_hub.py` generates the card from `configs/base.yaml`. The numbers in the card
therefore cannot disagree with the numbers of the run.

The card holds:

- The license from `MODEL_LICENSE`.
- The attribution of `HuggingFaceFW/fineweb-edu`, which carries the ODC-By license.
- The full training configuration.
- A limitations section. This section repeats the text of section 1.1: the model
  invents facts, and this result is expected at this scale.
- A note that Q8_0 is the recommended quantized file, and that Q4_K_M at 70M
  parameters is a demonstration only.

## 14. Pod control channel

The person deploys the pod in the RunPod web console. The person then gives the SSH
command to the agent. The agent runs every step over that one SSH connection.

The person keeps the only control that starts a charge. No script deploys a pod.

### 14.0 The terminate command

The first version of this spec configured no RunPod API key at all. The person then
asked for a way to stop the pod from the agent, so that no pod keeps billing after the
run. `RUNPOD_API_KEY` and `RUNPOD_POD_ID` therefore join the `.env` file, and
`scripts/runpod.py` reads the RunPod REST API.

Two rules limit the new risk:

1. `pod.sh terminate` needs the `--yes` flag. Without the flag the command stops and
   prints what the termination destroys.
2. No step of the pipeline calls terminate. The agent runs it only after the
   HuggingFace push is verified, because a terminate before that push destroys the
   only copy of the model.

This spec still configures no RunPod MCP server, and no script deploys a pod.

### 14.1 Setup

The person adds `~/.ssh/id_ed25519.pub` to the RunPod account, under Settings, SSH
Public Keys. The account then accepts that key for every later pod.

### 14.2 Code transfer

The project has no git remote. `scripts/pod.sh` therefore copies the working tree with
`rsync` over the same SSH connection. `rsync` excludes `.git`, `.venv`, `data`, and
`out`. The transfer moves about 100 KB and takes one second.

### 14.3 The long run

> Warning: a plain `ssh pod "python train.py"` command loses the training run when the
> connection drops. Always start the run inside `tmux`.

The agent starts the training detached:

```bash
ssh pod 'tmux new-session -d -s train "uv run scripts/train.py 2>&1 | tee out/train.log"'
```

The agent then reads progress in short commands:

```bash
ssh pod 'tail -5 out/train.log'
```

The agent polls about every 10 minutes and reports the step, the loss, and the
throughput. The run survives a closed laptop, a network failure, and an ended agent
session, because `tmux` owns the process.

Every SSH command uses `-o ServerAliveInterval=30`, so an idle connection does not
drop.

### 14.4 Weights and Biases

The `Trainer` reports to Weights and Biases, so the person watches the loss curve in a
browser during the run. The `tmux` log stays the primary record.

`train.py` reads `WANDB_API_KEY`. If the value is absent, the script sets `report_to`
to an empty list and writes one warning line. The 180 minute run never stops because
of a telemetry failure.
