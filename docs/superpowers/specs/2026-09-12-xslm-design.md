# xSLM Design Spec

Date: 2026-09-12
Status: Approved for planning

## 1. Purpose

Build a small language model from scratch to learn the full training pipeline. The
person writes the tokenizer training, the data pipeline, the model modules, and the
training configuration. The person trains the model on one RTX 3090 through RunPod.
The total pod time is four hours.

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

The `Trainer` class needs a `PreTrainedModel`. The design supplies two small wrappers:

- `XSLMConfig(PretrainedConfig)` holds the fields in the table above.
- `XSLMForCausalLM(PreTrainedModel)` holds the modules and returns
  `CausalLMOutputWithPast` with a `loss` field when the caller supplies `labels`.

The wrappers add about 30 lines. The attention, the RoPE, the RMSNorm, the SwiGLU, and
the block all remain hand-written `nn.Module` classes.

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

Each sample picks a random offset and reads 1025 tokens. The first 1024 tokens are the
input. The last 1024 tokens are the labels. There is no padding, no attention mask, and
no wasted computation.

## 6. Training configuration

| Field | Value |
|---|---|
| Optimizer | AdamW, betas 0.9 and 0.95, epsilon 1e-8 |
| Weight decay | 0.1 on matrices, 0.0 on norms and embeddings |
| Peak learning rate | 1e-3 |
| Schedule | 200 steps of linear warmup, then cosine decay to 1e-4 |
| Gradient clipping | 1.0 on the global norm |
| Precision | BF16 mixed, `bf16=True` |
| Micro-batch size | 32 sequences, raised until VRAM reaches about 21 GB |
| Gradient accumulation | 8 steps |
| Tokens per optimizer step | 32 * 8 * 1024 = 262,144 |
| Gradient checkpointing | Off. It saves memory the budget does not need and costs 30 percent speed |
| `torch.compile` | Off by default. It is an optional flag, because compilation can fail on a 3090 driver |
| Dataloader workers | 4 |

### 6.1 Setting the step count

The cosine schedule must end when training ends, so `max_steps` must be correct before
the full run starts.

The procedure is:

1. Run 20 steps and read `train_tokens_per_second` from the log.
2. Compute `max_steps = (throughput * 10800) / 262144`.
3. Write that number into the configuration and start the full run.

The starting estimate is 2,600 steps. The measured value replaces it.

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
    train_tokenizer.py      step 1
    prepare_data.py         step 2
    train.py                step 3
    sample.py               step 4
  tests/                    pytest, runs on CPU
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
| Label shift | A packed sample's labels equal its inputs shifted by one position |
| Memmap bounds | No sampled offset reads past the end of the file |
| Overfit | 50 steps on 20 tokens drive the loss below 0.1 |
| Parameter count | The production configuration reports 49M to 51M non-embedding parameters |

The overfit test is the most important one. A model that cannot memorize 20 tokens has
a broken gradient path, and no amount of GPU time will fix it.

## 9. Execution plan

All four steps run on one RunPod 3090 pod. A separate CPU pod saves about 0.15 US
dollars and adds a network volume, a second machine, and a transfer step. The extra
failure surface costs more than the saving.

Optional and free: run `pytest` on the development machine before renting the pod.

| Clock | Step | Duration |
|---|---|---|
| 0:00 | Start the pod, clone the repository, `uv sync`, run `nvidia-smi` | 5 min |
| 0:05 | Run `pytest` on CUDA | 3 min |
| 0:08 | Run `train_tokenizer.py` | 12 min |
| 0:20 | Run `prepare_data.py` | 30 min |
| 0:50 | Run the 20-step smoke test, then set `max_steps` | 3 min |
| 0:53 | Run the full training | 180 min |
| 3:53 | Run `sample.py`, download the checkpoint, stop the pod | 7 min |

The total is 4 hours. The estimated cost is about 1.00 US dollar.

### 9.1 Pod requirements

- One RTX 3090 with 24 GB of VRAM.
- A container disk of at least 30 GB, because `train.bin` needs 2 GB and the
  HuggingFace cache needs room.
- A CUDA 12.1 or later PyTorch image.

## 10. Risks

| Risk | Effect | Control |
|---|---|---|
| The pod stops during data preparation | The tokens are lost | `prepare_data.py` resumes from the current file size |
| The measured throughput is far below the estimate | The cosine schedule does not finish | The smoke test measures throughput before the full run sets `max_steps` |
| An out-of-memory error occurs | The run stops | Start at micro-batch 32 and raise it only after the smoke test reports the VRAM use |
| The loss diverges | The run wastes time | Gradient clipping at 1.0, plus 200 warmup steps. If the loss still diverges, halve the learning rate |
| `torch.compile` fails on the driver | The run stops at start | The flag is off by default |
| The HuggingFace stream rate-limits | Data preparation slows | Set `HF_TOKEN` in the pod environment |

## 11. Out of scope

Instruction tuning, chat formatting, reinforcement learning from human feedback,
quantization, multi-GPU training, model serving, and any retrieval system. Each is a
separate project with its own spec.
