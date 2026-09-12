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
