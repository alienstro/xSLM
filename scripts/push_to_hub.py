"""Step 5. Publish the BF16 weights, the tokenizer, and the model card.

The script is idempotent, so a second run after a failure is safe.
"""

import argparse
import sys
from pathlib import Path

import torch
from huggingface_hub import HfApi
from transformers import LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _env import optional, require  # noqa: E402

from xslm.config import load_tokenizer, load_yaml  # noqa: E402
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
| `gguf/{model_name}-BF16.gguf` | The unquantized GGUF file, in the training dtype |
| `gguf/{model_name}-F16.gguf` | The unquantized GGUF file, for the tools that read no BF16 |
| `gguf/{model_name}-Q8_0.gguf` | **The recommended quantized file** |
| `gguf/{model_name}-Q6_K.gguf` | Smaller, with a visible quality cost |
| `gguf/{model_name}-Q4_K_M.gguf` | A demonstration only |

At 70M parameters there is no redundancy to absorb quantization error. Q4_K_M often
produces broken text. Use Q8_0.

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
    non_embedding = (
        layers * (4 * hidden * hidden + 3 * hidden * model["intermediate_size"] + 2 * hidden)
        + hidden
    )
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
        **{
            key: model[key]
            for key in (
                "hidden_size",
                "num_hidden_layers",
                "num_attention_heads",
                "intermediate_size",
                "vocab_size",
                "max_position_embeddings",
                "rope_theta",
            )
        },
        **{
            key: training[key]
            for key in (
                "adam_beta1",
                "adam_beta2",
                "learning_rate",
                "min_learning_rate",
                "warmup_steps",
            )
        },
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
    # load_tokenizer names the end of text token, which the saved config lacks.
    load_tokenizer(checkpoint).save_pretrained(staging)
    (staging / "README.md").write_text(build_model_card(config, repo_id, license_name))

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=str(staging), repo_id=repo_id, repo_type="model")
    print(f"Pushed the BF16 weights and the model card to {repo_id}.")


if __name__ == "__main__":
    main()
