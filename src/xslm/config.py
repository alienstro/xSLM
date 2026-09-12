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
        # transformers 5 moved rope_theta into the rope_parameters dictionary.
        rope_parameters={"rope_type": "default", "rope_theta": fields["rope_theta"]},
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


def rope_theta_of(config):
    """Return the RoPE theta of a config, whatever the transformers version."""
    parameters = getattr(config, "rope_parameters", None)
    if parameters:
        return parameters["rope_theta"]
    return config.rope_theta


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
