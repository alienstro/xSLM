"""Tests for the instruction data: the format, the mask, and the template."""

import torch

from xslm.instruct import (
    IGNORE_INDEX,
    InstructDataset,
    build_example,
    format_prompt,
)

PAIR = {
    "instruction": "Name the capital of France.",
    "context": "",
    "response": "The capital of France is Paris.",
}


class FakeTokenizer:
    """One token for each word, so a test can count the tokens by hand."""

    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [len(word) for word in text.split()]


def test_the_prompt_names_the_instruction_and_asks_for_the_response():
    prompt = format_prompt(PAIR["instruction"], PAIR["context"])
    assert "### Instruction:" in prompt
    assert PAIR["instruction"] in prompt
    assert prompt.rstrip().endswith("### Response:")


def test_the_prompt_holds_the_context_when_the_pair_gives_one():
    prompt = format_prompt("Summarize this.", "The sky is blue.")
    assert "### Input:" in prompt
    assert "The sky is blue." in prompt
    assert prompt.index("### Input:") < prompt.index("### Response:")


def test_the_prompt_omits_the_input_heading_when_the_context_is_empty():
    assert "### Input:" not in format_prompt("Say hello.", "")


def test_the_example_masks_every_prompt_token():
    """The model must learn to write the response, never to write the question.

    A label that the loss reads on a prompt token teaches the model to invent
    instructions, which shows as a model that asks instead of answers.
    """
    tokenizer = FakeTokenizer()
    input_ids, labels = build_example(tokenizer, PAIR, max_length=128)
    prompt_length = len(tokenizer.encode(format_prompt(PAIR["instruction"], PAIR["context"])))
    assert labels[:prompt_length] == [IGNORE_INDEX] * prompt_length
    assert all(label != IGNORE_INDEX for label in labels[prompt_length:])


def test_the_example_keeps_the_input_ids_unmasked():
    """Only the labels carry the mask. The model still reads the whole prompt."""
    tokenizer = FakeTokenizer()
    input_ids, labels = build_example(tokenizer, PAIR, max_length=128)
    assert IGNORE_INDEX not in input_ids
    assert len(input_ids) == len(labels)


def test_the_example_ends_with_the_end_of_text_token():
    """Without a stop token the model writes until it runs out of room."""
    tokenizer = FakeTokenizer()
    input_ids, labels = build_example(tokenizer, PAIR, max_length=128)
    assert input_ids[-1] == tokenizer.eos_token_id
    assert labels[-1] == tokenizer.eos_token_id


def test_a_long_example_stops_at_the_maximum_length():
    tokenizer = FakeTokenizer()
    pair = {"instruction": "word " * 200, "context": "", "response": "word " * 200}
    input_ids, labels = build_example(tokenizer, pair, max_length=64)
    assert len(input_ids) == 64
    assert len(labels) == 64


def test_the_dataset_pads_to_one_length_and_masks_the_padding():
    """A padded label must never reach the loss, or the model learns to pad."""
    tokenizer = FakeTokenizer()
    dataset = InstructDataset([PAIR, PAIR], tokenizer, max_length=32)
    item = dataset[0]
    assert item["input_ids"].shape == (32,)
    assert item["labels"].shape == (32,)
    padding = item["attention_mask"] == 0
    assert torch.all(item["labels"][padding] == IGNORE_INDEX)


def test_the_dataset_reports_its_length():
    dataset = InstructDataset([PAIR, PAIR, PAIR], FakeTokenizer(), max_length=32)
    assert len(dataset) == 3


def test_the_padding_changes_no_loss():
    """The model reads no attention mask, so the padding must not matter.

    Causal attention lets a position read only the positions before it, and the
    padding sits at the end, so no real token reads a padded token. The padded
    labels hold IGNORE_INDEX, so the loss reads no padded position either.
    """
    from xslm.config import build_config
    from xslm.model import build_model

    config = build_config(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=128,
        vocab_size=128,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    model = build_model(config).eval()

    torch.manual_seed(1)
    short = torch.randint(1, 128, (1, 20))
    labels = short.clone()
    padding = torch.zeros((1, 12), dtype=torch.long)
    padded = torch.cat([short, padding], dim=1)
    padded_labels = torch.cat([labels, torch.full((1, 12), IGNORE_INDEX)], dim=1)

    with torch.no_grad():
        plain = model(input_ids=short, labels=labels).loss
        with_padding = model(input_ids=padded, labels=padded_labels).loss
    assert torch.allclose(plain, with_padding, atol=1e-5)
