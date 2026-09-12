"""The packed dataset that reads one flat uint16 token file.

The dataset never shifts the labels. The model shifts the logits and the labels
inside its loss function. A second shift here would teach the model to predict two
positions ahead.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

TOKEN_DTYPE = np.uint16


class PackedDataset(Dataset):
    """Read a random window of tokens from a memory-mapped token file."""

    def __init__(self, path, seq_len, seed=1337):
        self.path = Path(path)
        self.seq_len = seq_len
        self.seed = seed
        self.tokens = np.memmap(self.path, dtype=TOKEN_DTYPE, mode="r")
        self.last_start = len(self.tokens) - seq_len
        if self.last_start < 1:
            raise ValueError(
                f"{self.path} holds {len(self.tokens)} tokens, which is shorter than "
                f"the sequence length {seq_len}."
            )

    def __len__(self):
        return self.last_start // self.seq_len

    def __getitem__(self, index):
        generator = np.random.default_rng(self.seed + index)
        start = int(generator.integers(0, self.last_start))
        window = self.tokens[start : start + self.seq_len]
        input_ids = torch.from_numpy(window.astype(np.int64))
        return {"input_ids": input_ids, "labels": input_ids.clone()}
