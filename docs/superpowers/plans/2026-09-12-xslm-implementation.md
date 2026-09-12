# xSLM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> This repository sets a resource limit that forbids subagents. Use
> superpowers:executing-plans in the current session.

**Goal:** Build, train, and publish a 70M parameter Llama-style language model, with a
BF16 checkpoint and four GGUF quantizations on the HuggingFace Hub.

**Architecture:** The person writes RMSNorm, RoPE, SwiGLU, attention, and the decoder
block by hand. The parameter names copy the Llama names, so the state dictionary matches
`LlamaForCausalLM` key for key. The HuggingFace `Trainer` drives the loop. The data
pipeline packs tokens into one flat `uint16` file and reads it through `numpy.memmap`.
The agent controls the RunPod pod over SSH, and the 180 minute run lives in `tmux`.

**Tech Stack:** Python 3.12 through uv, PyTorch, transformers, tokenizers, datasets,
numpy, Weights and Biases, llama.cpp for the GGUF conversion.

**Spec:** `docs/superpowers/specs/2026-09-12-xslm-design.md`

## Global Constraints

- Python is pinned to 3.12. The system Python is 3.14.7 and has no wheels for several
  dependencies. No command touches the system Python.
- Every command runs through `uv run`. Never call `python` or `pip` directly.
- No module has a bias term.
- The model uses the Llama parameter names. Task 5 asserts this.
- The dataset never shifts the labels. `LlamaForCausalLM` shifts them internally.
- `vocab_size` is 32768, which is below 65536, so the token file uses `uint16`.
- No secret enters `configs/base.yaml` or any committed file.
- No script prints a token value.
- Every test runs on CPU. The full suite finishes in under 30 seconds.
- The non-embedding parameter count stays between 49M and 51M.
- Tokens for each optimizer step stay at 262,144, whatever the micro-batch is.
- `uv` resolved transformers 5.17.0 and torch 2.14.0. Three differences from
  transformers 4 govern this plan:
  1. `PreTrainedModel` does not inherit `GenerationMixin`. Every causal model must
     inherit it explicitly.
  2. `state_dict()` holds `lm_head.weight` even when the weights are tied, because
     PyTorch writes a tied parameter under every registered name.
  3. `TrainingArguments` no longer accepts `include_tokens_per_second`. The smoke test
     measures the throughput with its own clock.

---

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | The uv project, the pinned Python, the dependencies |
| `configs/base.yaml` | Every hyperparameter, and no secret |
| `.env.example` | The names of the secrets, with no values |
| `src/xslm/config.py` | One factory that returns a `LlamaConfig` |
| `src/xslm/model.py` | RMSNorm, RoPE, SwiGLU, attention, block, and the causal model |
| `src/xslm/data.py` | The memmap packed dataset |
| `src/xslm/callbacks.py` | The time limit callback and the sample callback |
| `scripts/_env.py` | Reads one environment variable and checks that it exists |
| `scripts/train_tokenizer.py` | Step 1, trains the byte-level BPE tokenizer |
| `scripts/prepare_data.py` | Step 2, writes `train.bin` and `val.bin`, and resumes |
| `scripts/train.py` | Step 3, the smoke test and the full run |
| `scripts/sample.py` | Step 4, generates text from a checkpoint |
| `scripts/push_to_hub.py` | Step 5, pushes the BF16 weights and the model card |
| `scripts/quantize.sh` | Step 6, builds llama.cpp and writes the four GGUF files |
| `scripts/pod.sh` | The SSH and rsync helper for the pod |
| `tests/` | The pytest suite, CPU only |

---

### Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Create: `configs/base.yaml`
- Create: `.env.example`
- Create: `src/xslm/__init__.py`
- Test: `tests/test_config_file.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `configs/base.yaml`, which every later task reads. The file holds the keys
  `model`, `tokenizer`, `data`, `training`, and `publish`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config_file.py
from pathlib import Path

import yaml

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"


def test_config_holds_the_model_shape():
    cfg = yaml.safe_load(CONFIG.read_text())
    model = cfg["model"]
    assert model["hidden_size"] == 640
    assert model["num_hidden_layers"] == 10
    assert model["num_attention_heads"] == 10
    assert model["intermediate_size"] == 1728
    assert model["vocab_size"] == 32768
    assert model["max_position_embeddings"] == 1024


def test_tokens_for_each_optimizer_step_stay_at_262144():
    cfg = yaml.safe_load(CONFIG.read_text())["training"]
    tokens = (
        cfg["per_device_train_batch_size"]
        * cfg["gradient_accumulation_steps"]
        * yaml.safe_load(CONFIG.read_text())["model"]["max_position_embeddings"]
    )
    assert tokens == 262144


def test_config_holds_no_secret():
    text = CONFIG.read_text().lower()
    for name in ("hf_token", "wandb_api_key", "api_key", "password"):
        assert name not in text
```

- [ ] **Step 2: Run the test to verify that it fails**

Run: `uv run pytest tests/test_config_file.py -v`
Expected: FAIL, because `configs/base.yaml` does not exist.

- [ ] **Step 3: Write the project files**

```toml
# pyproject.toml
[project]
name = "xslm"
version = "0.1.0"
requires-python = "==3.12.*"
dependencies = [
    "torch>=2.5",
    "transformers>=4.45",
    "tokenizers>=0.20",
    "datasets>=3.0",
    "accelerate>=1.0",
    "numpy>=2.0",
    "pyyaml>=6.0",
    "huggingface-hub>=0.26",
    "wandb>=0.18",
    "tqdm>=4.66",
]

[dependency-groups]
dev = ["pytest>=8.3"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/xslm"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

```
3.12
```
Write that single line to `.python-version`.

```yaml
# configs/base.yaml
model:
  hidden_size: 640
  num_hidden_layers: 10
  num_attention_heads: 10
  num_key_value_heads: 10
  intermediate_size: 1728
  vocab_size: 32768
  max_position_embeddings: 1024
  rms_norm_eps: 1.0e-5
  rope_theta: 10000.0
  initializer_range: 0.02
  tie_word_embeddings: true

tokenizer:
  vocab_size: 32768
  sample_bytes: 2000000000
  output_path: data/tokenizer.json

data:
  dataset: HuggingFaceFW/fineweb-edu
  dataset_config: sample-10BT
  train_path: data/train.bin
  val_path: data/val.bin
  val_tokens: 5000000
  target_tokens: 900000000

training:
  output_dir: out
  per_device_train_batch_size: 16
  gradient_accumulation_steps: 16
  learning_rate: 1.0e-3
  min_learning_rate: 1.0e-4
  warmup_steps: 200
  max_steps: 2600
  weight_decay: 0.1
  adam_beta1: 0.9
  adam_beta2: 0.95
  adam_epsilon: 1.0e-8
  max_grad_norm: 1.0
  bf16: true
  gradient_checkpointing: false
  torch_compile: false
  dataloader_num_workers: 4
  logging_steps: 10
  eval_steps: 250
  save_steps: 250
  save_total_limit: 2
  sample_every_steps: 500
  time_limit_minutes: 180
  seed: 1337

publish:
  model_name: xslm-70m
  quant_types:
    - Q8_0
    - Q6_K
    - Q4_K_M
```

```bash
# .env.example
# A fine-grained token with write scope, limited to HF_REPO_ID only.
HF_TOKEN=
HF_REPO_ID=
MODEL_LICENSE=apache-2.0

# Weights and Biases. The training run continues if this value is absent.
WANDB_API_KEY=
WANDB_PROJECT=xslm
```

Write an empty `src/xslm/__init__.py`.

- [ ] **Step 4: Create the environment and run the test**

Run:
```bash
uv sync
uv run pytest tests/test_config_file.py -v
```
Expected: PASS, three tests.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .python-version configs/base.yaml .env.example src/xslm/__init__.py tests/test_config_file.py uv.lock
git commit -m "feat: add the uv project, the base configuration, and the secret names"
```

---

### Task 2: The configuration factory

**Files:**
- Create: `src/xslm/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: `configs/base.yaml` from Task 1.
- Produces:
  - `load_yaml(path: str | Path) -> dict`
  - `build_config(**overrides) -> LlamaConfig`
  - `count_non_embedding_parameters(config: LlamaConfig) -> int`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from transformers import LlamaConfig

from xslm.config import build_config, count_non_embedding_parameters


def test_build_config_returns_a_llama_config():
    cfg = build_config()
    assert isinstance(cfg, LlamaConfig)
    assert cfg.model_type == "llama"
    assert cfg.hidden_size == 640
    assert cfg.num_attention_heads == 10
    assert cfg.hidden_size // cfg.num_attention_heads == 64
    assert cfg.tie_word_embeddings is True
    assert cfg.attention_bias is False
    assert cfg.mlp_bias is False
    assert cfg.hidden_act == "silu"


def test_overrides_reach_the_config():
    cfg = build_config(num_hidden_layers=2, hidden_size=64, num_attention_heads=2)
    assert cfg.num_hidden_layers == 2
    assert cfg.hidden_size == 64


def test_the_non_embedding_count_stays_between_49m_and_51m():
    count = count_non_embedding_parameters(build_config())
    assert 49_000_000 <= count <= 51_000_000
```

- [ ] **Step 2: Run the test to verify that it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'xslm.config'`.

- [ ] **Step 3: Write the implementation**

```python
# src/xslm/config.py
"""One factory that turns configs/base.yaml into a LlamaConfig.

The model uses the Llama parameter names, so llama.cpp converts it to GGUF
without a custom converter, and transformers loads it without trust_remote_code.
"""

from pathlib import Path

import yaml
from transformers import LlamaConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"

END_OF_TEXT_ID = 0


def load_yaml(path=DEFAULT_CONFIG_PATH):
    """Read one YAML file and return a dictionary."""
    return yaml.safe_load(Path(path).read_text())


def build_config(config_path=DEFAULT_CONFIG_PATH, **overrides):
    """Return a LlamaConfig that holds the model fields of the YAML file."""
    fields = dict(load_yaml(config_path)["model"])
    fields.update(overrides)
    return LlamaConfig(
        hidden_size=fields["hidden_size"],
        num_hidden_layers=fields["num_hidden_layers"],
        num_attention_heads=fields["num_attention_heads"],
        num_key_value_heads=fields["num_key_value_heads"],
        intermediate_size=fields["intermediate_size"],
        vocab_size=fields["vocab_size"],
        max_position_embeddings=fields["max_position_embeddings"],
        rms_norm_eps=fields["rms_norm_eps"],
        rope_theta=fields["rope_theta"],
        initializer_range=fields["initializer_range"],
        tie_word_embeddings=fields["tie_word_embeddings"],
        hidden_act="silu",
        attention_bias=False,
        mlp_bias=False,
        bos_token_id=END_OF_TEXT_ID,
        eos_token_id=END_OF_TEXT_ID,
        pad_token_id=None,
        use_cache=True,
    )


def count_non_embedding_parameters(config):
    """Return the parameter count of the blocks and the final norm.

    The count excludes the tied embedding. The formula follows section 3.1 of
    the design spec, so the test needs no model instance.
    """
    d = config.hidden_size
    attention = 4 * d * d
    feed_forward = 3 * d * config.intermediate_size
    norms = 2 * d
    one_layer = attention + feed_forward + norms
    return one_layer * config.num_hidden_layers + d
```

> Note: `num_key_value_heads` equals `num_attention_heads`, so the attention holds four
> square projections, and the formula `4 * d * d` is correct.

- [ ] **Step 4: Run the test to verify that it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS, three tests. The count is 49,575,040.

- [ ] **Step 5: Commit**

```bash
git add src/xslm/config.py tests/test_config.py
git commit -m "feat: add the LlamaConfig factory and the parameter count"
```

---

### Task 3: RMSNorm and RoPE

**Files:**
- Create: `src/xslm/model.py`
- Test: `tests/test_rope.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `class RMSNorm(nn.Module)` with one parameter named `weight`
  - `build_rope_cache(head_dim: int, max_seq_len: int, theta: float, device, dtype) -> tuple[Tensor, Tensor]`
  - `rotate_half(x: Tensor) -> Tensor`
  - `apply_rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]`

> Warning: `rotate_half` splits the last dimension into two halves. This is the
> HuggingFace layout. `convert_hf_to_gguf.py` permutes the query and key weights on the
> assumption of this layout. An interleaved layout would produce a GGUF file that
> generates nonsense.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rope.py
import torch

from xslm.model import RMSNorm, apply_rope, build_rope_cache


def test_rmsnorm_gives_unit_root_mean_square():
    norm = RMSNorm(16, eps=1e-5)
    x = torch.randn(2, 4, 16) * 7.0
    out = norm(x)
    rms = out.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_rmsnorm_holds_one_parameter_named_weight():
    names = [name for name, _ in RMSNorm(16, eps=1e-5).named_parameters()]
    assert names == ["weight"]


HEAD_DIM, MAX_SEQ = 64, 128


def rope_tables():
    return build_rope_cache(HEAD_DIM, MAX_SEQ, 10000.0, torch.device("cpu"), torch.float32)


def rotate(vector, position):
    """Rotate one vector to one position. apply_rope rotates a pair, so pass it twice."""
    cos, sin = rope_tables()
    rotated, _ = apply_rope(vector, vector, cos[position : position + 1], sin[position : position + 1])
    return rotated


def test_rope_keeps_the_dot_product_when_both_positions_shift():
    torch.manual_seed(0)
    q = torch.randn(1, 1, 1, HEAD_DIM)
    k = torch.randn(1, 1, 1, HEAD_DIM)
    near = (rotate(q, 3) * rotate(k, 8)).sum()
    far = (rotate(q, 13) * rotate(k, 18)).sum()
    assert torch.allclose(near, far, atol=1e-4)


def test_rope_changes_the_dot_product_when_the_distance_changes():
    torch.manual_seed(0)
    q = torch.randn(1, 1, 1, HEAD_DIM)
    k = torch.randn(1, 1, 1, HEAD_DIM)
    close = (rotate(q, 0) * rotate(k, 1)).sum()
    distant = (rotate(q, 0) * rotate(k, 9)).sum()
    assert not torch.allclose(close, distant, atol=1e-3)
```

- [ ] **Step 2: Run the test to verify that it fails**

Run: `uv run pytest tests/test_rope.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'xslm.model'`.

- [ ] **Step 3: Write the implementation**

```python
# src/xslm/model.py
"""The hand-written modules of the xSLM model.

Every parameter carries the name that Llama gives it, so the state dictionary
matches LlamaForCausalLM key for key. Task 5 asserts this.
"""

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    """Root mean square normalization, with a scale and no bias."""

    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def build_rope_cache(head_dim, max_seq_len, theta, device, dtype):
    """Return the cosine table and the sine table for rotary position encoding."""
    exponent = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim
    inverse_frequency = 1.0 / (theta**exponent)
    positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(positions, inverse_frequency)
    table = torch.cat((angles, angles), dim=-1)
    return table.cos().to(dtype), table.sin().to(dtype)


def rotate_half(x):
    """Rotate the two halves of the last dimension. This is the Llama layout."""
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rope(q, k, cos, sin):
    """Rotate the queries and the keys. The values stay unchanged."""
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
```

- [ ] **Step 4: Run the test to verify that it passes**

Run: `uv run pytest tests/test_rope.py -v`
Expected: PASS, four tests.

- [ ] **Step 5: Commit**

```bash
git add src/xslm/model.py tests/test_rope.py
git commit -m "feat: add RMSNorm and the rotary position encoding"
```

---

### Task 4: Attention, SwiGLU, and the decoder block

**Files:**
- Modify: `src/xslm/model.py`
- Test: `tests/test_block.py`

**Interfaces:**
- Consumes: `RMSNorm`, `build_rope_cache`, `apply_rope` from Task 3.
- Produces:
  - `class Attention(nn.Module)` with `q_proj`, `k_proj`, `v_proj`, `o_proj`
  - `class MLP(nn.Module)` with `gate_proj`, `up_proj`, `down_proj`
  - `class DecoderLayer(nn.Module)` with `self_attn`, `mlp`, `input_layernorm`,
    `post_attention_layernorm`
  - Each `forward(hidden_states, cos, sin)` returns one tensor of the input shape.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_block.py
import torch

from xslm.model import MLP, Attention, DecoderLayer, build_rope_cache


def small():
    return dict(hidden_size=64, num_heads=4, intermediate_size=128, eps=1e-5)


def test_no_module_holds_a_bias():
    layer = DecoderLayer(**small())
    names = [name for name, _ in layer.named_parameters()]
    assert not any(name.endswith("bias") for name in names)


def test_the_parameter_names_copy_the_llama_names():
    names = {name for name, _ in DecoderLayer(**small()).named_parameters()}
    assert names == {
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
    }


def test_the_block_keeps_the_shape():
    layer = DecoderLayer(**small())
    cos, sin = build_rope_cache(16, 8, 10000.0, torch.device("cpu"), torch.float32)
    x = torch.randn(2, 8, 64)
    assert layer(x, cos, sin).shape == (2, 8, 64)


def test_the_mlp_follows_the_swiglu_formula():
    mlp = MLP(hidden_size=8, intermediate_size=16)
    x = torch.randn(1, 3, 8)
    expected = mlp.down_proj(torch.nn.functional.silu(mlp.gate_proj(x)) * mlp.up_proj(x))
    assert torch.allclose(mlp(x), expected, atol=1e-6)


def test_the_attention_hides_the_future():
    torch.manual_seed(0)
    attention = Attention(hidden_size=64, num_heads=4)
    cos, sin = build_rope_cache(16, 8, 10000.0, torch.device("cpu"), torch.float32)
    x = torch.randn(1, 8, 64)
    changed = x.clone()
    changed[:, 5:, :] = torch.randn(1, 3, 64)
    before = attention(x, cos, sin)
    after = attention(changed, cos, sin)
    assert torch.allclose(before[:, :5], after[:, :5], atol=1e-5)
    assert not torch.allclose(before[:, 5:], after[:, 5:], atol=1e-5)
```

- [ ] **Step 2: Run the test to verify that it fails**

Run: `uv run pytest tests/test_block.py -v`
Expected: FAIL with `ImportError: cannot import name 'Attention'`.

- [ ] **Step 3: Write the implementation**

Append to `src/xslm/model.py`:

```python
class Attention(nn.Module):
    """Multi-head causal attention. The kernel choice belongs to PyTorch."""

    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states, cos, sin):
        batch, seq_len, hidden_size = hidden_states.shape
        shape = (batch, seq_len, self.num_heads, self.head_dim)
        q = self.q_proj(hidden_states).view(shape).transpose(1, 2)
        k = self.k_proj(hidden_states).view(shape).transpose(1, 2)
        v = self.v_proj(hidden_states).view(shape).transpose(1, 2)
        q, k = apply_rope(q, k, cos[:seq_len], sin[:seq_len])
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attended = attended.transpose(1, 2).reshape(batch, seq_len, hidden_size)
        return self.o_proj(attended)


class MLP(nn.Module):
    """The SwiGLU feed-forward network, with no bias."""

    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states):
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class DecoderLayer(nn.Module):
    """One pre-normalization decoder block."""

    def __init__(self, hidden_size, num_heads, intermediate_size, eps=1e-5):
        super().__init__()
        self.self_attn = Attention(hidden_size, num_heads)
        self.mlp = MLP(hidden_size, intermediate_size)
        self.input_layernorm = RMSNorm(hidden_size, eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps)

    def forward(self, hidden_states, cos, sin):
        hidden_states = hidden_states + self.self_attn(self.input_layernorm(hidden_states), cos, sin)
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
```

- [ ] **Step 4: Run the test to verify that it passes**

Run: `uv run pytest tests/test_block.py -v`
Expected: PASS, five tests.

- [ ] **Step 5: Commit**

```bash
git add src/xslm/model.py tests/test_block.py
git commit -m "feat: add the attention, the SwiGLU network, and the decoder block"
```

---

### Task 5: The causal model

**Files:**
- Modify: `src/xslm/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `build_config` from Task 2, and every module from Task 3 and Task 4.
- Produces:
  - `class XSLMModel(LlamaPreTrainedModel)` with `embed_tokens`, `layers`, `norm`
  - `class XSLMForCausalLM(LlamaPreTrainedModel, GenerationMixin)` with `model` and
    `lm_head`. `forward(input_ids, labels=None, **kwargs) -> CausalLMOutputWithPast`.
  - `build_model(config) -> XSLMForCausalLM`

> Warning: since transformers 4.50, `PreTrainedModel` no longer inherits
> `GenerationMixin`. The class must inherit it explicitly, or `generate` raises an
> `AttributeError` at minute 240 inside `sample.py`.

> Warning: this model computes no key and value cache. Every call to `generate` must
> pass `use_cache=False`. The published `LlamaForCausalLM` of Task 13 does hold a cache,
> so the published model needs no such flag.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model.py
import torch
import torch.nn.functional as F
from transformers import LlamaForCausalLM

from xslm.config import build_config, count_non_embedding_parameters
from xslm.model import build_model


def tiny():
    return build_config(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=128,
        vocab_size=128,
        max_position_embeddings=32,
    )


def test_the_forward_pass_gives_the_right_shapes():
    config = tiny()
    model = build_model(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    out = model(input_ids=input_ids, labels=input_ids)
    assert out.logits.shape == (2, 16, config.vocab_size)
    assert out.loss.ndim == 0


def test_the_state_dictionary_matches_llama_key_for_key():
    config = tiny()
    mine = {k: tuple(v.shape) for k, v in build_model(config).state_dict().items()}
    theirs = {k: tuple(v.shape) for k, v in LlamaForCausalLM(config).state_dict().items()}
    assert mine == theirs


def test_the_embedding_is_tied_to_the_output():
    model = build_model(tiny())
    assert model.lm_head.weight is model.model.embed_tokens.weight
    # PyTorch writes a tied parameter under every registered name, so the key stays.
    state = model.state_dict()
    assert torch.equal(state["lm_head.weight"], state["model.embed_tokens.weight"])


def test_the_loss_equals_a_hand_written_next_token_cross_entropy():
    config = tiny()
    model = build_model(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    with torch.no_grad():
        out = model(input_ids=input_ids, labels=input_ids)
        expected = F.cross_entropy(
            out.logits[:, :-1].reshape(-1, config.vocab_size),
            input_ids[:, 1:].reshape(-1),
        )
    assert torch.allclose(out.loss, expected, atol=1e-5)


def test_the_residual_projections_start_smaller_than_the_input_projections():
    model = build_model(tiny())
    layer = model.model.layers[0]
    assert layer.self_attn.o_proj.weight.std() < layer.self_attn.q_proj.weight.std()
    assert layer.mlp.down_proj.weight.std() < layer.mlp.gate_proj.weight.std()


def test_the_production_model_holds_between_49m_and_51m_non_embedding_parameters():
    config = build_config()
    counted = count_non_embedding_parameters(config)
    model = build_model(build_config(vocab_size=64))
    measured = sum(p.numel() for n, p in model.named_parameters() if "embed_tokens" not in n)
    assert 49_000_000 <= counted <= 51_000_000
    assert measured == counted


def test_fifty_steps_on_twenty_tokens_drive_the_loss_below_zero_point_one():
    torch.manual_seed(0)
    config = tiny()
    model = build_model(config).train()
    input_ids = torch.randint(0, config.vocab_size, (1, 20))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    loss = None
    for _ in range(50):
        optimizer.zero_grad()
        loss = model(input_ids=input_ids, labels=input_ids).loss
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.1
```

- [ ] **Step 2: Run the test to verify that it fails**

Run: `uv run pytest tests/test_model.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_model'`.

- [ ] **Step 3: Write the implementation**

Append to `src/xslm/model.py`:

```python
import math

from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LlamaPreTrainedModel


class XSLMModel(LlamaPreTrainedModel):
    """The embedding, the stack of decoder layers, and the final norm."""

    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            DecoderLayer(
                config.hidden_size,
                config.num_attention_heads,
                config.intermediate_size,
                config.rms_norm_eps,
            )
            for _ in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        head_dim = config.hidden_size // config.num_attention_heads
        cos, sin = build_rope_cache(
            head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            torch.device("cpu"),
            torch.float32,
        )
        # persistent=False keeps the tables out of the state dictionary, so the
        # parity test against LlamaForCausalLM passes.
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.post_init()

    def forward(self, input_ids):
        hidden_states = self.embed_tokens(input_ids)
        cos = self.rope_cos.to(hidden_states.dtype)
        sin = self.rope_sin.to(hidden_states.dtype)
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin)
        return self.norm(hidden_states)


class XSLMForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    """The decoder with a tied output projection."""

    def __init__(self, config):
        super().__init__(config)
        self.model = XSLMModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        self._scale_the_residual_projections()

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def _scale_the_residual_projections(self):
        """Shrink the projections that write into the residual stream.

        The smaller standard deviation keeps the variance of the residual stream
        stable as the depth increases. See section 3.2 of the design spec.
        """
        std = self.config.initializer_range / math.sqrt(2 * self.config.num_hidden_layers)
        for name, parameter in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(parameter, mean=0.0, std=std)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids=None, labels=None, **kwargs):
        hidden_states = self.model(input_ids)
        logits = self.lm_head(hidden_states).float()
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits)


def build_model(config):
    """Return one XSLMForCausalLM for the given configuration."""
    return XSLMForCausalLM(config)
```

> Note: `logits.float()` upcasts before the loss. This matches the HuggingFace Llama
> behaviour and keeps the loss stable in BF16. Section 6 of the spec counts the memory
> that this copy costs.

- [ ] **Step 4: Run the test to verify that it passes**

Run: `uv run pytest tests/test_model.py -v`
Expected: PASS, seven tests. The overfit test is the important one. If it fails, the
gradient path is broken, and no GPU time repairs that.

- [ ] **Step 5: Commit**

```bash
git add src/xslm/model.py tests/test_model.py
git commit -m "feat: add the causal model with the Llama parameter names"
```

---

### Task 6: The packed dataset

**Files:**
- Create: `src/xslm/data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `class PackedDataset(torch.utils.data.Dataset)`, built as
    `PackedDataset(path: str | Path, seq_len: int, seed: int = 1337)`
  - `__getitem__(index) -> dict[str, torch.Tensor]` with the keys `input_ids` and
    `labels`, each of shape `(seq_len,)` and dtype `torch.long`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_data.py
import numpy as np
import pytest
import torch

from xslm.data import PackedDataset


@pytest.fixture
def token_file(tmp_path):
    path = tmp_path / "tokens.bin"
    np.arange(5000, dtype=np.uint16).tofile(path)
    return path


def test_a_sample_holds_the_requested_length(token_file):
    sample = PackedDataset(token_file, seq_len=128)[0]
    assert sample["input_ids"].shape == (128,)
    assert sample["input_ids"].dtype == torch.long


def test_the_labels_equal_the_inputs(token_file):
    sample = PackedDataset(token_file, seq_len=64)[3]
    assert torch.equal(sample["input_ids"], sample["labels"])


def test_no_sample_reads_past_the_end_of_the_file(token_file):
    dataset = PackedDataset(token_file, seq_len=64, seed=7)
    for index in range(len(dataset)):
        values = dataset[index]["input_ids"]
        assert values.max().item() < 5000
        assert values.shape == (64,)


def test_the_same_index_gives_the_same_sample(token_file):
    dataset = PackedDataset(token_file, seq_len=32, seed=11)
    assert torch.equal(dataset[5]["input_ids"], dataset[5]["input_ids"])


def test_a_file_shorter_than_one_sequence_raises(tmp_path):
    path = tmp_path / "short.bin"
    np.arange(10, dtype=np.uint16).tofile(path)
    with pytest.raises(ValueError, match="shorter than"):
        PackedDataset(path, seq_len=64)
```

- [ ] **Step 2: Run the test to verify that it fails**

Run: `uv run pytest tests/test_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'xslm.data'`.

- [ ] **Step 3: Write the implementation**

```python
# src/xslm/data.py
"""The packed dataset that reads one flat uint16 token file.

The dataset never shifts the labels. The model shifts the logits and the labels
inside its loss function. A second shift here would teach the model to predict two
positions ahead.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

TOKEN_DTYPE = np.uint16


class PackedDataset(Dataset):
    """Read a random window of tokens from a memory-mapped token file."""

    def __init__(self, path, seq_len, seed=1337):
        self.path = Path(path)
        self.seq_len = seq_len
        self.seed = seed
        self.tokens = np.memmap(self.path, dtype=TOKEN_DTYPE, mode="r")
        self.last_start = len(self.tokens) - seq_len
        if self.last_start < 1:
            raise ValueError(
                f"{self.path} holds {len(self.tokens)} tokens, which is shorter than "
                f"the sequence length {seq_len}."
            )

    def __len__(self):
        return self.last_start // self.seq_len

    def __getitem__(self, index):
        generator = np.random.default_rng(self.seed + index)
        start = int(generator.integers(0, self.last_start))
        window = self.tokens[start : start + self.seq_len]
        input_ids = torch.from_numpy(window.astype(np.int64))
        return {"input_ids": input_ids, "labels": input_ids.clone()}
```

- [ ] **Step 4: Run the test to verify that it passes**

Run: `uv run pytest tests/test_data.py -v`
Expected: PASS, five tests.

- [ ] **Step 5: Commit**

```bash
git add src/xslm/data.py tests/test_data.py
git commit -m "feat: add the memory-mapped packed dataset"
```

---

### Task 7: The training callbacks

**Files:**
- Create: `src/xslm/callbacks.py`
- Test: `tests/test_callbacks.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `class TimeLimitCallback(TrainerCallback)`, built as
    `TimeLimitCallback(time_limit_minutes: float)`
  - `class SampleGenerationCallback(TrainerCallback)`, built as
    `SampleGenerationCallback(tokenizer, prompt: str, every_steps: int, max_new_tokens: int = 100)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_callbacks.py
from transformers import TrainerControl, TrainerState

from xslm.callbacks import TimeLimitCallback


def test_the_run_continues_before_the_limit():
    callback = TimeLimitCallback(time_limit_minutes=60)
    control = TrainerControl()
    callback.on_step_end(args=None, state=TrainerState(), control=control)
    assert control.should_training_stop is False


def test_the_run_stops_after_the_limit():
    callback = TimeLimitCallback(time_limit_minutes=0)
    control = TrainerControl()
    callback.on_step_end(args=None, state=TrainerState(), control=control)
    assert control.should_training_stop is True
    assert control.should_save is True


def test_the_clock_starts_at_the_beginning_of_training():
    callback = TimeLimitCallback(time_limit_minutes=60)
    callback.on_train_begin(args=None, state=TrainerState(), control=TrainerControl())
    assert callback.started_at is not None
```

- [ ] **Step 2: Run the test to verify that it fails**

Run: `uv run pytest tests/test_callbacks.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'xslm.callbacks'`.

- [ ] **Step 3: Write the implementation**

```python
# src/xslm/callbacks.py
"""Two callbacks: one stops the run on time, one shows what the model writes."""

import time

import torch
from transformers import TrainerCallback


class TimeLimitCallback(TrainerCallback):
    """Stop the run after a fixed number of minutes and save a checkpoint.

    The pod rental is fixed, so the run must end with a saved checkpoint inside the
    budget. A cosine schedule that does not finish is better than no checkpoint.
    """

    def __init__(self, time_limit_minutes):
        self.limit_seconds = time_limit_minutes * 60.0
        self.started_at = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.started_at = time.monotonic()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self.started_at is None:
            self.started_at = time.monotonic()
        if time.monotonic() - self.started_at >= self.limit_seconds:
            control.should_save = True
            control.should_training_stop = True
        return control


class SampleGenerationCallback(TrainerCallback):
    """Write one short sample to the log at a fixed interval.

    A loss number does not show whether the text reads as English. A sample does.
    """

    def __init__(self, tokenizer, prompt, every_steps, max_new_tokens=100):
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.every_steps = every_steps
        self.max_new_tokens = max_new_tokens

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None or state.global_step == 0:
            return control
        if state.global_step % self.every_steps != 0:
            return control
        was_training = model.training
        model.eval()
        input_ids = self.tokenizer(self.prompt, return_tensors="pt").input_ids
        with torch.no_grad():
            # This model holds no key and value cache, so use_cache must be False.
            generated = model.generate(
                input_ids.to(model.device),
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                use_cache=False,
            )
        text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"\n[step {state.global_step}] {text}\n", flush=True)
        if was_training:
            model.train()
        return control
```

- [ ] **Step 4: Run the test to verify that it passes**

Run: `uv run pytest tests/test_callbacks.py -v`
Expected: PASS, three tests.

- [ ] **Step 5: Commit**

```bash
git add src/xslm/callbacks.py tests/test_callbacks.py
git commit -m "feat: add the time limit callback and the sample callback"
```

---

### Task 8: The environment guard

**Files:**
- Create: `scripts/_env.py`
- Test: `tests/test_env.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `require(name: str) -> str`, which raises `RuntimeError` when the value is absent
  - `optional(name: str, default: str | None = None) -> str | None`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_env.py
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from _env import optional, require  # noqa: E402


def test_require_returns_the_value(monkeypatch):
    monkeypatch.setenv("XSLM_TEST_KEY", "value")
    assert require("XSLM_TEST_KEY") == "value"


def test_require_raises_when_the_value_is_absent(monkeypatch):
    monkeypatch.delenv("XSLM_TEST_KEY", raising=False)
    with pytest.raises(RuntimeError, match="XSLM_TEST_KEY"):
        require("XSLM_TEST_KEY")


def test_the_message_names_the_fix(monkeypatch):
    monkeypatch.delenv("XSLM_TEST_KEY", raising=False)
    with pytest.raises(RuntimeError, match="--env-file"):
        require("XSLM_TEST_KEY")


def test_the_message_never_holds_a_value(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "secret-value-do-not-print")
    monkeypatch.delenv("XSLM_TEST_KEY", raising=False)
    with pytest.raises(RuntimeError) as error:
        require("XSLM_TEST_KEY")
    assert "secret-value-do-not-print" not in str(error.value)


def test_optional_returns_the_default(monkeypatch):
    monkeypatch.delenv("XSLM_TEST_KEY", raising=False)
    assert optional("XSLM_TEST_KEY", "fallback") == "fallback"
```

- [ ] **Step 2: Run the test to verify that it fails**

Run: `uv run pytest tests/test_env.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named '_env'`.

- [ ] **Step 3: Write the implementation**

```python
# scripts/_env.py
"""Read the environment, and fail in one second instead of after 30 minutes."""

import os

FIX = (
    "Copy .env.example to .env, write the value, and run the script as\n"
    "    uv run --env-file .env scripts/<name>.py\n"
    "On the pod, set the value in the RunPod template environment fields instead."
)


def require(name):
    """Return the value of one environment variable, or raise."""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"The environment variable {name} is empty or absent.\n{FIX}")
    return value


def optional(name, default=None):
    """Return the value of one environment variable, or the default."""
    return os.environ.get(name) or default
```

- [ ] **Step 4: Run the test to verify that it passes**

Run: `uv run pytest tests/test_env.py -v`
Expected: PASS, five tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/_env.py tests/test_env.py
git commit -m "feat: add the environment guard that fails early"
```

---

### Task 9: The tokenizer trainer

**Files:**
- Create: `scripts/train_tokenizer.py`
- Test: `tests/test_tokenizer.py`

**Interfaces:**
- Consumes: `load_yaml` from Task 2.
- Produces:
  - `build_tokenizer(texts: Iterable[str], vocab_size: int) -> tokenizers.Tokenizer`
  - `save_tokenizer(tokenizer, path) -> PreTrainedTokenizerFast`
  - A command line entry point that writes `data/tokenizer.json`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tokenizer.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_tokenizer import build_tokenizer, save_tokenizer  # noqa: E402

CORPUS = [
    "The cat sat on the mat in 2024.",
    "A second sentence gives the trainer more merges to find.",
    "Numbers such as 42 and 1993 appear in ordinary text.",
    "Education research covers reading, writing, and arithmetic.",
] * 40


def test_the_end_of_text_token_takes_identifier_zero():
    tokenizer = build_tokenizer(CORPUS, vocab_size=300)
    assert tokenizer.token_to_id("<|endoftext|>") == 0


def test_every_byte_survives_a_round_trip():
    tokenizer = build_tokenizer(CORPUS, vocab_size=300)
    text = "Unseen bytes: é 中 \U0001f600 and a tab\there."
    assert tokenizer.decode(tokenizer.encode(text).ids) == text


def test_the_tokenizer_splits_the_digits():
    tokenizer = build_tokenizer(CORPUS, vocab_size=300)
    pieces = tokenizer.encode("1993").tokens
    assert len(pieces) == 4


def test_the_saved_file_loads_through_transformers(tmp_path):
    path = tmp_path / "tokenizer.json"
    fast = save_tokenizer(build_tokenizer(CORPUS, vocab_size=300), path)
    assert path.exists()
    assert fast.decode(fast("hello world").input_ids) == "hello world"
```

- [ ] **Step 2: Run the test to verify that it fails**

Run: `uv run pytest tests/test_tokenizer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'train_tokenizer'`.

- [ ] **Step 3: Write the implementation**

```python
# scripts/train_tokenizer.py
"""Step 1. Train a byte-level BPE tokenizer on a sample of the training corpus.

Byte-level BPE makes an unknown token impossible, because every byte round-trips.
The Digits pre-tokenizer splits every number into single digits, which helps the
model read numbers.
"""

import argparse
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from xslm.config import load_yaml

END_OF_TEXT = "<|endoftext|>"


def build_tokenizer(texts, vocab_size):
    """Train one byte-level BPE tokenizer on an iterable of strings."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Digits(individual_digits=True),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[END_OF_TEXT],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    return tokenizer


def save_tokenizer(tokenizer, path):
    """Write tokenizer.json and return the transformers wrapper."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(path))
    return PreTrainedTokenizerFast(
        tokenizer_file=str(path),
        bos_token=END_OF_TEXT,
        eos_token=END_OF_TEXT,
        unk_token=None,
        pad_token=None,
    )


def stream_texts(dataset_name, dataset_config, sample_bytes):
    """Yield documents from the streamed corpus until the byte budget runs out."""
    from datasets import load_dataset

    stream = load_dataset(dataset_name, name=dataset_config, split="train", streaming=True)
    seen = 0
    for record in stream:
        text = record["text"]
        seen += len(text.encode("utf-8"))
        yield text
        if seen >= sample_bytes:
            return


def main():
    parser = argparse.ArgumentParser(description="Train the xSLM tokenizer.")
    parser.add_argument("--config", default=None)
    arguments = parser.parse_args()

    config = load_yaml(arguments.config) if arguments.config else load_yaml()
    settings = config["tokenizer"]
    texts = stream_texts(
        config["data"]["dataset"],
        config["data"]["dataset_config"],
        settings["sample_bytes"],
    )
    tokenizer = build_tokenizer(texts, settings["vocab_size"])
    save_tokenizer(tokenizer, settings["output_path"])
    print(f"Wrote {settings['output_path']} with {tokenizer.get_vocab_size()} tokens.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify that it passes**

Run: `uv run pytest tests/test_tokenizer.py -v`
Expected: PASS, four tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/train_tokenizer.py tests/test_tokenizer.py
git commit -m "feat: add the byte-level BPE tokenizer trainer"
```

---

### Task 10: The data preparation script

**Files:**
- Create: `scripts/prepare_data.py`
- Test: `tests/test_prepare_data.py`

**Interfaces:**
- Consumes: `load_yaml` from Task 2, `data/tokenizer.json` from Task 9.
- Produces:
  - `tokens_on_disk(path) -> int`
  - `append_tokens(path, ids: list[int]) -> int`
  - A command line entry point that writes `data/val.bin` and `data/train.bin`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prepare_data.py
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from prepare_data import append_tokens, tokens_on_disk  # noqa: E402


def test_an_absent_file_holds_no_tokens(tmp_path):
    assert tokens_on_disk(tmp_path / "missing.bin") == 0


def test_append_returns_the_running_total(tmp_path):
    path = tmp_path / "train.bin"
    assert append_tokens(path, [1, 2, 3]) == 3
    assert append_tokens(path, [4, 5]) == 5
    assert tokens_on_disk(path) == 5


def test_the_file_holds_uint16_values_in_order(tmp_path):
    path = tmp_path / "train.bin"
    append_tokens(path, [7, 8, 9])
    assert np.fromfile(path, dtype=np.uint16).tolist() == [7, 8, 9]


def test_a_token_above_the_uint16_range_raises(tmp_path):
    with pytest.raises(ValueError, match="65535"):
        append_tokens(tmp_path / "train.bin", [70000])
```

- [ ] **Step 2: Run the test to verify that it fails**

Run: `uv run pytest tests/test_prepare_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'prepare_data'`.

- [ ] **Step 3: Write the implementation**

```python
# scripts/prepare_data.py
"""Step 2. Stream the corpus, tokenize it, and write one flat uint16 token file.

The script is resumable. It reads the size of train.bin at the start and continues
from that point, so a pod failure wastes no tokenizer work.
"""

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

from xslm.config import load_yaml

TOKEN_DTYPE = np.uint16
MAX_TOKEN_ID = 65535


def tokens_on_disk(path):
    """Return the number of tokens already written to the file."""
    path = Path(path)
    if not path.exists():
        return 0
    return path.stat().st_size // np.dtype(TOKEN_DTYPE).itemsize


def append_tokens(path, ids):
    """Append token identifiers to the file and return the running total."""
    array = np.asarray(ids, dtype=np.int64)
    if array.size and array.max() > MAX_TOKEN_ID:
        raise ValueError(
            f"A token identifier is above {MAX_TOKEN_ID}, so uint16 cannot hold it."
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        array.astype(TOKEN_DTYPE).tofile(handle)
    return tokens_on_disk(path)


def main():
    parser = argparse.ArgumentParser(description="Write train.bin and val.bin.")
    parser.add_argument("--config", default=None)
    arguments = parser.parse_args()

    from datasets import load_dataset

    config = load_yaml(arguments.config) if arguments.config else load_yaml()
    settings = config["data"]
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=config["tokenizer"]["output_path"])
    end_of_text = tokenizer.convert_tokens_to_ids("<|endoftext|>")

    val_path = Path(settings["val_path"])
    train_path = Path(settings["train_path"])
    val_target = settings["val_tokens"]
    train_target = settings["target_tokens"]

    written_val = tokens_on_disk(val_path)
    written_train = tokens_on_disk(train_path)
    print(f"Resuming with {written_val} validation tokens and {written_train} train tokens.")

    stream = load_dataset(
        settings["dataset"], name=settings["dataset_config"], split="train", streaming=True
    )
    progress = tqdm(total=train_target, initial=written_train, unit="tok", unit_scale=True)
    for record in stream:
        if written_train >= train_target:
            break
        ids = tokenizer(record["text"], add_special_tokens=False).input_ids
        ids.append(end_of_text)
        if written_val < val_target:
            written_val = append_tokens(val_path, ids)
            continue
        before = written_train
        written_train = append_tokens(train_path, ids)
        progress.update(written_train - before)
    progress.close()
    print(f"Wrote {written_val} validation tokens and {written_train} train tokens.")


if __name__ == "__main__":
    main()
```

> Note: the script never rewinds the stream on a resume. It therefore writes different
> documents after a restart. This is correct, because the corpus is far larger than the
> token target and every document is independent.

- [ ] **Step 4: Run the test to verify that it passes**

Run: `uv run pytest tests/test_prepare_data.py -v`
Expected: PASS, four tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/prepare_data.py tests/test_prepare_data.py
git commit -m "feat: add the resumable data preparation script"
```

---

### Task 11: The training script

**Files:**
- Create: `scripts/train.py`
- Test: `tests/test_train_helpers.py`

**Interfaces:**
- Consumes: `build_config`, `build_model`, `PackedDataset`, both callbacks, `optional`.
- Produces:
  - `compute_max_steps(tokens_per_second: float, seconds: float, tokens_per_step: int) -> int`
  - `resolve_report_to() -> list[str]`
  - A command line entry point with a `--smoke N` flag.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_train_helpers.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train import compute_max_steps, resolve_report_to  # noqa: E402


def test_the_step_count_follows_the_throughput():
    assert compute_max_steps(63_000.0, 10_800.0, 262_144) == 2595


def test_the_step_count_is_never_zero():
    assert compute_max_steps(1.0, 1.0, 262_144) == 1


def test_weights_and_biases_is_on_when_the_key_exists(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "a-value")
    assert resolve_report_to() == ["wandb"]


def test_weights_and_biases_is_off_when_the_key_is_absent(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    assert resolve_report_to() == []
```

- [ ] **Step 2: Run the test to verify that it fails**

Run: `uv run pytest tests/test_train_helpers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'train'`.

- [ ] **Step 3: Write the implementation**

```python
# scripts/train.py
"""Step 3. Run the smoke test or the full training run."""

import argparse
import math
import sys
from pathlib import Path

import torch
from transformers import PreTrainedTokenizerFast, Trainer, TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _env import optional  # noqa: E402

from xslm.callbacks import SampleGenerationCallback, TimeLimitCallback  # noqa: E402
from xslm.config import build_config, load_yaml  # noqa: E402
from xslm.data import PackedDataset  # noqa: E402
from xslm.model import build_model  # noqa: E402


def compute_max_steps(tokens_per_second, seconds, tokens_per_step):
    """Return the step count that fills the time budget. Never return zero."""
    return max(1, int(tokens_per_second * seconds / tokens_per_step))


def resolve_report_to():
    """Return the reporting targets.

    The 180 minute run must never stop because of a telemetry failure, so an absent
    key turns the reporting off instead of raising.
    """
    if optional("WANDB_API_KEY"):
        return ["wandb"]
    print("WANDB_API_KEY is absent, so the run reports to no dashboard.", flush=True)
    return []


def main():
    parser = argparse.ArgumentParser(description="Train the xSLM model.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--smoke", type=int, default=0, help="Run this many steps and stop.")
    arguments = parser.parse_args()

    config = load_yaml(arguments.config) if arguments.config else load_yaml()
    settings = config["training"]
    seq_len = config["model"]["max_position_embeddings"]

    torch.manual_seed(settings["seed"])
    model = build_model(build_config())
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=config["tokenizer"]["output_path"])

    train_dataset = PackedDataset(config["data"]["train_path"], seq_len, settings["seed"])
    eval_dataset = PackedDataset(config["data"]["val_path"], seq_len, settings["seed"])

    max_steps = arguments.smoke if arguments.smoke else settings["max_steps"]
    training_arguments = TrainingArguments(
        output_dir=settings["output_dir"],
        max_steps=max_steps,
        per_device_train_batch_size=settings["per_device_train_batch_size"],
        gradient_accumulation_steps=settings["gradient_accumulation_steps"],
        learning_rate=settings["learning_rate"],
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": settings["min_learning_rate"]},
        warmup_steps=settings["warmup_steps"],
        weight_decay=settings["weight_decay"],
        adam_beta1=settings["adam_beta1"],
        adam_beta2=settings["adam_beta2"],
        adam_epsilon=settings["adam_epsilon"],
        max_grad_norm=settings["max_grad_norm"],
        bf16=settings["bf16"] and torch.cuda.is_available(),
        gradient_checkpointing=settings["gradient_checkpointing"],
        torch_compile=settings["torch_compile"],
        dataloader_num_workers=settings["dataloader_num_workers"],
        logging_steps=settings["logging_steps"],
        eval_strategy="no" if arguments.smoke else "steps",
        eval_steps=settings["eval_steps"],
        save_strategy="no" if arguments.smoke else "steps",
        save_steps=settings["save_steps"],
        save_total_limit=settings["save_total_limit"],
        report_to=resolve_report_to(),
        seed=settings["seed"],
    )

    callbacks = []
    if not arguments.smoke:
        callbacks.append(TimeLimitCallback(settings["time_limit_minutes"]))
        callbacks.append(
            SampleGenerationCallback(
                tokenizer,
                prompt="The history of",
                every_steps=settings["sample_every_steps"],
            )
        )

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks,
    )
    result = trainer.train()

    if arguments.smoke:
        # transformers 5 removed include_tokens_per_second, so measure it here.
        tokens_per_step = (
            settings["per_device_train_batch_size"]
            * settings["gradient_accumulation_steps"]
            * seq_len
        )
        throughput = arguments.smoke * tokens_per_step / result.metrics["train_runtime"]
        seconds = settings["time_limit_minutes"] * 60
        print(f"train_tokens_per_second = {throughput:.0f}")
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / 2**30
            print(f"peak VRAM = {peak:.1f} GiB")
        print(f"Set max_steps to {compute_max_steps(throughput, seconds, tokens_per_step)}")
        return

    trainer.save_model(settings["output_dir"])
    tokenizer.save_pretrained(settings["output_dir"])
    metrics = trainer.evaluate()
    print(f"eval_loss = {metrics['eval_loss']:.4f}")
    print(f"perplexity = {math.exp(metrics['eval_loss']):.2f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify that it passes**

Run: `uv run pytest tests/test_train_helpers.py -v`
Expected: PASS, four tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/train.py tests/test_train_helpers.py
git commit -m "feat: add the training script with a smoke mode"
```

---

### Task 12: The sampling script

**Files:**
- Create: `scripts/sample.py`
- Test: `tests/test_sample.py`

**Interfaces:**
- Consumes: `build_model` from Task 5.
- Produces:
  - `generate_text(model, tokenizer, prompt, max_new_tokens, temperature, top_p) -> str`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sample.py
import sys
from pathlib import Path

import torch
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from sample import generate_text  # noqa: E402

from xslm.config import build_config
from xslm.model import build_model


def tiny_tokenizer(tmp_path):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=300,
        special_tokens=["<|endoftext|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(["hello world of small models"] * 50, trainer=trainer)
    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))
    return PreTrainedTokenizerFast(tokenizer_file=str(path), eos_token="<|endoftext|>")


def test_generate_returns_a_string_that_starts_with_the_prompt(tmp_path):
    torch.manual_seed(0)
    tokenizer = tiny_tokenizer(tmp_path)
    model = build_model(build_config(
        hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=4, intermediate_size=128,
        vocab_size=len(tokenizer), max_position_embeddings=64,
    )).eval()
    text = generate_text(model, tokenizer, "hello", max_new_tokens=8)
    assert isinstance(text, str)
    assert text.startswith("hello")
```

- [ ] **Step 2: Run the test to verify that it fails**

Run: `uv run pytest tests/test_sample.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sample'`.

- [ ] **Step 3: Write the implementation**

```python
# scripts/sample.py
"""Step 4. Generate text from a trained checkpoint."""

import argparse

import torch
from transformers import AutoTokenizer

from xslm.config import load_yaml
from xslm.model import XSLMForCausalLM


def generate_text(model, tokenizer, prompt, max_new_tokens=200, temperature=0.8, top_p=0.95):
    """Return the prompt plus the generated continuation."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        # This model holds no key and value cache, so use_cache must be False.
        generated = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            use_cache=False,
        )
    return tokenizer.decode(generated[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Generate text from a checkpoint.")
    parser.add_argument("--prompt", default="The history of")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--checkpoint", default=None)
    arguments = parser.parse_args()

    config = load_yaml()
    checkpoint = arguments.checkpoint or config["training"]["output_dir"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = XSLMForCausalLM.from_pretrained(checkpoint).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    print(generate_text(model, tokenizer, arguments.prompt, arguments.max_new_tokens))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify that it passes**

Run: `uv run pytest tests/test_sample.py -v`
Expected: PASS, one test.

- [ ] **Step 5: Commit**

```bash
git add scripts/sample.py tests/test_sample.py
git commit -m "feat: add the sampling script"
```

---

### Task 13: The HuggingFace push

**Files:**
- Create: `scripts/push_to_hub.py`
- Test: `tests/test_push_to_hub.py`

**Interfaces:**
- Consumes: `require`, `optional` from Task 8, `build_config` and `build_model`.
- Produces:
  - `to_llama(model: XSLMForCausalLM) -> LlamaForCausalLM`
  - `build_model_card(config: dict, repo_id: str, license_name: str) -> str`
  - A command line entry point that uploads the weights, the tokenizer, and the card.

> Warning: the published repository must declare `LlamaForCausalLM`, not
> `XSLMForCausalLM`. `to_llama` therefore loads the state dictionary into a real
> `LlamaForCausalLM` and saves that. The parity test of Task 5 makes this load safe.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_push_to_hub.py
import sys
from pathlib import Path

from transformers import LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from push_to_hub import build_model_card, to_llama  # noqa: E402

from xslm.config import build_config, load_yaml
from xslm.model import build_model


def tiny():
    return build_config(
        hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=4, intermediate_size=128, vocab_size=128,
        max_position_embeddings=32,
    )


def test_the_state_dictionary_loads_into_llama_without_a_missing_key():
    config = tiny()
    published = to_llama(build_model(config))
    assert isinstance(published, LlamaForCausalLM)
    assert published.config.architectures == ["LlamaForCausalLM"]


def test_the_published_model_gives_the_same_logits():
    import torch

    torch.manual_seed(0)
    config = tiny()
    mine = build_model(config).eval()
    theirs = to_llama(mine).eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    with torch.no_grad():
        assert torch.allclose(
            mine(input_ids=input_ids).logits,
            theirs(input_ids=input_ids).logits,
            atol=1e-4,
        )


def test_the_card_names_the_license_the_corpus_and_the_limitation():
    card = build_model_card(load_yaml(), "someone/xslm-70m", "apache-2.0")
    assert "license: apache-2.0" in card
    assert "HuggingFaceFW/fineweb-edu" in card
    assert "ODC-By" in card
    assert "invents facts" in card
    assert "Q8_0" in card


def test_the_card_never_holds_a_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "secret-value-do-not-print")
    card = build_model_card(load_yaml(), "someone/xslm-70m", "apache-2.0")
    assert "secret-value-do-not-print" not in card
```

- [ ] **Step 2: Run the test to verify that it fails**

Run: `uv run pytest tests/test_push_to_hub.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'push_to_hub'`.

- [ ] **Step 3: Write the implementation**

```python
# scripts/push_to_hub.py
"""Step 5. Publish the BF16 weights, the tokenizer, and the model card.

The script is idempotent, so a second run after a failure is safe.
"""

import argparse
import sys
from pathlib import Path

import torch
from huggingface_hub import HfApi
from transformers import AutoTokenizer, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _env import optional, require  # noqa: E402

from xslm.config import load_yaml  # noqa: E402
from xslm.model import XSLMForCausalLM  # noqa: E402

CARD = """---
license: {license_name}
language:
- en
library_name: transformers
pipeline_tag: text-generation
datasets:
- HuggingFaceFW/fineweb-edu
tags:
- llama
- gguf
---

# {model_name}

A {parameters} parameter Llama-style decoder, trained from scratch on one RTX 3090 in
{minutes} minutes. The project exists to teach the full training pipeline: the
tokenizer, the data pipeline, the modules, and the training configuration.

## Limitations

This model does not hold world knowledge. A model of this size learns grammar, style,
and document structure. It invents facts. This is the expected result at this scale,
and it is not a defect to report.

For a model that answers questions correctly, use retrieval-augmented generation over a
document index, or a QLoRA fine-tune of an existing 1B to 8B base model.

## Architecture

| Field | Value |
|---|---|
| Non-embedding parameters | {non_embedding} |
| Total parameters | {parameters} |
| `hidden_size` | {hidden_size} |
| `num_hidden_layers` | {num_hidden_layers} |
| `num_attention_heads` | {num_attention_heads} |
| `intermediate_size` | {intermediate_size} |
| `vocab_size` | {vocab_size} |
| `max_position_embeddings` | {max_position_embeddings} |
| Normalization | RMSNorm |
| Position encoding | RoPE, theta {rope_theta} |
| Activation | SwiGLU |
| Embeddings | Tied input and output |

## Training

| Field | Value |
|---|---|
| Corpus | `HuggingFaceFW/fineweb-edu`, `{dataset_config}`, ODC-By license |
| Tokenizer | Byte-level BPE, {vocab_size} tokens |
| Optimizer | AdamW, betas {adam_beta1} and {adam_beta2} |
| Peak learning rate | {learning_rate} |
| Schedule | {warmup_steps} warmup steps, then cosine decay to {min_learning_rate} |
| Tokens for each optimizer step | {tokens_per_step} |
| Precision | BF16 mixed |
| Hardware | One RTX 3090 |

## Files

| File | Use |
|---|---|
| `model.safetensors` | The BF16 weights, for `transformers` |
| `gguf/{model_name}-BF16.gguf` | The unquantized GGUF file |
| `gguf/{model_name}-Q8_0.gguf` | **The recommended quantized file** |
| `gguf/{model_name}-Q6_K.gguf` | Smaller, with a visible quality cost |
| `gguf/{model_name}-Q4_K_M.gguf` | A demonstration only |

> At 70M parameters there is no redundancy to absorb quantization error. Q4_K_M often
> produces broken text. Use Q8_0.

## Use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
model = AutoModelForCausalLM.from_pretrained("{repo_id}")
```
"""


def to_llama(model):
    """Return a LlamaForCausalLM that holds the same weights.

    The published repository must declare LlamaForCausalLM, so that a reader loads it
    with AutoModelForCausalLM and no trust_remote_code flag.
    """
    published = LlamaForCausalLM(model.config)
    published.load_state_dict(model.state_dict(), strict=True)
    published.config.architectures = ["LlamaForCausalLM"]
    return published


def build_model_card(config, repo_id, license_name):
    """Return the model card text, built from the configuration file."""
    model = config["model"]
    training = config["training"]
    hidden = model["hidden_size"]
    layers = model["num_hidden_layers"]
    non_embedding = layers * (
        4 * hidden * hidden + 3 * hidden * model["intermediate_size"] + 2 * hidden
    ) + hidden
    total = non_embedding + model["vocab_size"] * hidden
    tokens_per_step = (
        training["per_device_train_batch_size"]
        * training["gradient_accumulation_steps"]
        * model["max_position_embeddings"]
    )
    return CARD.format(
        license_name=license_name,
        repo_id=repo_id,
        model_name=config["publish"]["model_name"],
        dataset_config=config["data"]["dataset_config"],
        non_embedding=f"{non_embedding:,}",
        parameters=f"{total:,}",
        minutes=training["time_limit_minutes"],
        tokens_per_step=f"{tokens_per_step:,}",
        **{key: model[key] for key in (
            "hidden_size", "num_hidden_layers", "num_attention_heads",
            "intermediate_size", "vocab_size", "max_position_embeddings", "rope_theta",
        )},
        **{key: training[key] for key in (
            "adam_beta1", "adam_beta2", "learning_rate", "min_learning_rate",
            "warmup_steps",
        )},
    )


def main():
    parser = argparse.ArgumentParser(description="Push the model to the HuggingFace Hub.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--staging", default="out/publish")
    arguments = parser.parse_args()

    token = require("HF_TOKEN")
    repo_id = require("HF_REPO_ID")
    license_name = optional("MODEL_LICENSE", "apache-2.0")

    config = load_yaml()
    checkpoint = arguments.checkpoint or config["training"]["output_dir"]
    staging = Path(arguments.staging)
    staging.mkdir(parents=True, exist_ok=True)

    model = XSLMForCausalLM.from_pretrained(checkpoint).to(torch.bfloat16)
    to_llama(model).save_pretrained(staging, safe_serialization=True)
    AutoTokenizer.from_pretrained(checkpoint).save_pretrained(staging)
    (staging / "README.md").write_text(build_model_card(config, repo_id, license_name))

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=str(staging), repo_id=repo_id, repo_type="model")
    print(f"Pushed the BF16 weights and the model card to {repo_id}.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify that it passes**

Run: `uv run pytest tests/test_push_to_hub.py -v`
Expected: PASS, four tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/push_to_hub.py tests/test_push_to_hub.py
git commit -m "feat: add the HuggingFace push and the generated model card"
```

---

### Task 14: The quantization script

**Files:**
- Create: `scripts/quantize.sh`
- Test: `tests/test_shell_scripts.py`

**Interfaces:**
- Consumes: the staged folder `out/publish` from Task 13.
- Produces: `out/gguf/xslm-70m-{BF16,Q8_0,Q6_K,Q4_K_M}.gguf`, uploaded under `gguf/`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_shell_scripts.py
import subprocess
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def test_quantize_is_valid_bash():
    subprocess.run(["bash", "-n", str(SCRIPTS / "quantize.sh")], check=True)


def test_quantize_stops_on_an_error():
    text = (SCRIPTS / "quantize.sh").read_text()
    assert "set -euo pipefail" in text


def test_quantize_builds_every_requested_type():
    text = (SCRIPTS / "quantize.sh").read_text()
    for quant in ("Q8_0", "Q6_K", "Q4_K_M"):
        assert quant in text
```

- [ ] **Step 2: Run the test to verify that it fails**

Run: `uv run pytest tests/test_shell_scripts.py -v`
Expected: FAIL, because `scripts/quantize.sh` does not exist.

- [ ] **Step 3: Write the implementation**

```bash
#!/usr/bin/env bash
# Step 6. Build llama.cpp, convert the model to GGUF, and quantize it three times.
#
# The build needs no CUDA. llama-quantize is CPU work, and a 70M model quantizes
# in seconds.
set -euo pipefail

STAGING="${1:-out/publish}"
OUTPUT="${2:-out/gguf}"
NAME="xslm-70m"
LLAMA_DIR="${LLAMA_CPP_DIR:-llama.cpp}"

mkdir -p "$OUTPUT"

if [ ! -d "$LLAMA_DIR" ]; then
    git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
fi

if [ ! -x "$LLAMA_DIR/build/bin/llama-quantize" ]; then
    cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DGGML_CUDA=OFF -DLLAMA_CURL=OFF
    cmake --build "$LLAMA_DIR/build" --target llama-quantize -j "$(nproc)"
fi

uv run python "$LLAMA_DIR/convert_hf_to_gguf.py" "$STAGING" \
    --outfile "$OUTPUT/$NAME-BF16.gguf" \
    --outtype bf16

for QUANT in Q8_0 Q6_K Q4_K_M; do
    "$LLAMA_DIR/build/bin/llama-quantize" \
        "$OUTPUT/$NAME-BF16.gguf" \
        "$OUTPUT/$NAME-$QUANT.gguf" \
        "$QUANT"
done

ls -lh "$OUTPUT"

uv run python - <<'PYTHON'
import os
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
api.upload_folder(
    folder_path="out/gguf",
    path_in_repo="gguf",
    repo_id=os.environ["HF_REPO_ID"],
    repo_type="model",
)
print("Pushed the gguf folder.")
PYTHON
```

Make it executable: `chmod +x scripts/quantize.sh`

> Note: `convert_hf_to_gguf.py` needs the `gguf` package. `llama.cpp` ships it under
> `gguf-py`. If the import fails, run `uv pip install gguf` inside the project
> environment.

- [ ] **Step 4: Run the test to verify that it passes**

Run: `uv run pytest tests/test_shell_scripts.py -v`
Expected: PASS, three tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/quantize.sh tests/test_shell_scripts.py
git commit -m "feat: add the GGUF conversion and the three quantizations"
```

---

### Task 15: The pod helper

**Files:**
- Create: `scripts/pod.sh`
- Modify: `tests/test_shell_scripts.py`

**Interfaces:**
- Consumes: nothing in the repository.
- Produces: the commands `pod.sh check`, `pod.sh sync`, `pod.sh setup`, `pod.sh run`,
  `pod.sh watch`. Each reads `POD_SSH` from the environment.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_shell_scripts.py`:

```python
def test_pod_is_valid_bash():
    subprocess.run(["bash", "-n", str(SCRIPTS / "pod.sh")], check=True)


def test_pod_keeps_the_connection_alive():
    assert "ServerAliveInterval=30" in (SCRIPTS / "pod.sh").read_text()


def test_pod_starts_the_long_run_inside_tmux():
    text = (SCRIPTS / "pod.sh").read_text()
    assert "tmux new-session -d" in text


def test_pod_checks_for_cmake_before_the_run():
    assert "cmake --version" in (SCRIPTS / "pod.sh").read_text()
```

- [ ] **Step 2: Run the test to verify that it fails**

Run: `uv run pytest tests/test_shell_scripts.py -v`
Expected: FAIL, because `scripts/pod.sh` does not exist.

- [ ] **Step 3: Write the implementation**

```bash
#!/usr/bin/env bash
# The SSH and rsync helper for the RunPod pod.
#
# Set POD_SSH to the connection string that the RunPod console shows, for example:
#     export POD_SSH="root@213.173.98.10 -p 40123 -i ~/.ssh/id_ed25519"
set -euo pipefail

REMOTE_DIR="/workspace/xSLM"
SSH_FLAGS="-o ServerAliveInterval=30 -o ServerAliveCountMax=6 -o StrictHostKeyChecking=accept-new"

if [ -z "${POD_SSH:-}" ]; then
    echo "POD_SSH is empty. Export the connection string from the RunPod console." >&2
    exit 1
fi

remote() {
    # shellcheck disable=SC2086
    ssh $SSH_FLAGS $POD_SSH "$@"
}

case "${1:-}" in
check)
    remote 'nvidia-smi'
    remote 'cmake --version || (apt-get update -qq && apt-get install -y -qq cmake)'
    remote 'tmux -V || (apt-get update -qq && apt-get install -y -qq tmux)'
    ;;
sync)
    # shellcheck disable=SC2086
    rsync -az --delete \
        --exclude '.git' --exclude '.venv' --exclude 'data' --exclude 'out' \
        --exclude 'llama.cpp' --exclude '__pycache__' \
        -e "ssh $SSH_FLAGS" \
        ./ "${POD_SSH%% *}:$REMOTE_DIR/"
    ;;
setup)
    remote "cd $REMOTE_DIR && curl -LsSf https://astral.sh/uv/install.sh | sh && ~/.local/bin/uv sync"
    remote "cd $REMOTE_DIR && ~/.local/bin/uv run pytest -q"
    ;;
run)
    shift
    remote "cd $REMOTE_DIR && mkdir -p out && tmux new-session -d -s train \"~/.local/bin/uv run $* 2>&1 | tee out/train.log\""
    echo "Started in tmux. Read it with: scripts/pod.sh watch"
    ;;
watch)
    remote "tail -n ${2:-20} $REMOTE_DIR/out/train.log"
    ;;
*)
    echo "Usage: pod.sh {check|sync|setup|run <command>|watch [lines]}" >&2
    exit 1
    ;;
esac
```

Make it executable: `chmod +x scripts/pod.sh`

> Warning: `rsync` uses `--delete`, so it removes files on the pod that the local tree
> does not hold. The excluded folders `data`, `out`, and `llama.cpp` therefore survive
> every sync. Never remove those exclusions.

> Note: `rsync` needs the plain `host:path` form, so the script strips the flags from
> `POD_SSH` with `${POD_SSH%% *}` and passes them through the `-e` flag instead.

- [ ] **Step 4: Run the test to verify that it passes**

Run: `uv run pytest tests/test_shell_scripts.py -v`
Expected: PASS, seven tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/pod.sh tests/test_shell_scripts.py
git commit -m "feat: add the pod helper for ssh, rsync, and tmux"
```

---

### Task 16: The README and the full suite

**Files:**
- Create: `README.md`
- Test: the whole suite

**Interfaces:**
- Consumes: every earlier task.
- Produces: nothing that code imports.

- [ ] **Step 1: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS, 59 tests, in under 30 seconds.

- [ ] **Step 2: Write the README**

````markdown
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

```bash
export POD_SSH="root@<address> -p <port> -i ~/.ssh/id_ed25519"

scripts/pod.sh check                       # nvidia-smi, cmake, tmux
scripts/pod.sh sync                        # rsync the working tree
scripts/pod.sh setup                       # uv sync, then pytest on CUDA
scripts/pod.sh run scripts/train_tokenizer.py
scripts/pod.sh run scripts/prepare_data.py
scripts/pod.sh run scripts/train.py --smoke 20    # read the throughput
# write max_steps into configs/base.yaml, then sync again
scripts/pod.sh run scripts/train.py
scripts/pod.sh watch 20                    # poll about every 10 minutes
scripts/pod.sh run scripts/sample.py
scripts/pod.sh run scripts/push_to_hub.py
scripts/pod.sh run bash scripts/quantize.sh
```

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
````

- [ ] **Step 3: Run the full suite once more**

Run: `uv run pytest -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: add the README with the setup and the pod run"
```

---

## Definition of done

1. `uv run pytest -q` passes on the development machine and on the pod.
2. The training run saves a checkpoint inside the time limit.
3. The validation loss stays below 3.8 nats for each token.
4. `sample.py` writes grammatical English of at least 100 tokens.
5. The HuggingFace repository holds `model.safetensors` and four GGUF files.
6. `AutoModelForCausalLM.from_pretrained(HF_REPO_ID)` works without `trust_remote_code`.
