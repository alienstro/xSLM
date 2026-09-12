import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from sample import generate_text  # noqa: E402

from xslm.config import build_config  # noqa: E402
from xslm.model import build_model  # noqa: E402


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
    model = build_model(
        build_config(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=128,
            vocab_size=len(tokenizer),
            max_position_embeddings=64,
        )
    ).eval()
    text = generate_text(model, tokenizer, "hello", max_new_tokens=8)
    assert isinstance(text, str)
    assert text.startswith("hello")
