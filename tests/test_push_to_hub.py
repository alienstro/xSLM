import sys
from pathlib import Path

import torch
from transformers import LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from push_to_hub import build_model_card, to_llama  # noqa: E402

from xslm.config import build_config, load_yaml  # noqa: E402
from xslm.model import build_model  # noqa: E402


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


def test_the_state_dictionary_loads_into_llama_without_a_missing_key():
    published = to_llama(build_model(tiny()))
    assert isinstance(published, LlamaForCausalLM)
    assert published.config.architectures == ["LlamaForCausalLM"]


def test_the_published_model_gives_the_same_logits():
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
