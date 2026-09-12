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
