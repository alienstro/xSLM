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
