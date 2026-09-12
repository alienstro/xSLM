"""Step 2. Stream the corpus, tokenize it, and write one flat uint16 token file.

The script is resumable. It reads the size of train.bin at the start and continues
from that point, so a pod failure wastes no tokenizer work.
"""

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

from xslm.config import load_yaml

TOKEN_DTYPE = np.uint16
MAX_TOKEN_ID = 65535


def tokens_on_disk(path):
    """Return the number of tokens already written to the file."""
    path = Path(path)
    if not path.exists():
        return 0
    return path.stat().st_size // np.dtype(TOKEN_DTYPE).itemsize


def append_tokens(path, ids):
    """Append token identifiers to the file and return the running total."""
    array = np.asarray(ids, dtype=np.int64)
    if array.size and array.max() > MAX_TOKEN_ID:
        raise ValueError(
            f"A token identifier is above {MAX_TOKEN_ID}, so uint16 cannot hold it."
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        array.astype(TOKEN_DTYPE).tofile(handle)
    return tokens_on_disk(path)


def main():
    parser = argparse.ArgumentParser(description="Write train.bin and val.bin.")
    parser.add_argument("--config", default=None)
    arguments = parser.parse_args()

    from datasets import load_dataset

    config = load_yaml(arguments.config) if arguments.config else load_yaml()
    settings = config["data"]
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=config["tokenizer"]["output_path"])
    end_of_text = tokenizer.convert_tokens_to_ids("<|endoftext|>")

    val_path = Path(settings["val_path"])
    train_path = Path(settings["train_path"])
    val_target = settings["val_tokens"]
    train_target = settings["target_tokens"]

    written_val = tokens_on_disk(val_path)
    written_train = tokens_on_disk(train_path)
    print(f"Resuming with {written_val} validation tokens and {written_train} train tokens.")

    stream = load_dataset(
        settings["dataset"], name=settings["dataset_config"], split="train", streaming=True
    )
    progress = tqdm(total=train_target, initial=written_train, unit="tok", unit_scale=True)
    for record in stream:
        if written_train >= train_target:
            break
        ids = tokenizer(record["text"], add_special_tokens=False).input_ids
        ids.append(end_of_text)
        if written_val < val_target:
            written_val = append_tokens(val_path, ids)
            continue
        before = written_train
        written_train = append_tokens(train_path, ids)
        progress.update(written_train - before)
    progress.close()
    print(f"Wrote {written_val} validation tokens and {written_train} train tokens.")


if __name__ == "__main__":
    main()
