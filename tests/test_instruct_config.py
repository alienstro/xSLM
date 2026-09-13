"""The instruction stage reads its settings from the same YAML file."""

from xslm.config import load_yaml


def settings():
    return load_yaml()["instruct"]


def test_the_config_names_the_dataset_and_the_base_model():
    assert settings()["dataset"] == "databricks/databricks-dolly-15k"
    assert settings()["base_model"]


def test_the_learning_rate_stays_far_under_the_pretraining_rate():
    """A large rate makes the model forget the language that it learned."""
    config = load_yaml()
    assert settings()["learning_rate"] <= config["training"]["learning_rate"] / 20


def test_the_run_holds_a_time_limit():
    assert 0 < settings()["time_limit_minutes"] <= 60
