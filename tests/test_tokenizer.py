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
