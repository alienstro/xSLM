"""The instruction data: the prompt format, the label mask, and the dataset.

Pretraining taught the model to continue any text. Instruction tuning teaches it
one behaviour: read an instruction, then write a response, then stop.

The markers are plain text, not new tokens. A new token would force a resize of the
tied embedding, and every resize risks the parity with LlamaForCausalLM that lets
llama.cpp convert the model without a custom converter. The end of text token
already exists, so it ends each example.
"""

import torch
from torch.utils.data import Dataset

# The loss ignores this label. PyTorch cross_entropy defines the value.
IGNORE_INDEX = -100

WITH_CONTEXT = (
    "### Instruction:\n{instruction}\n\n### Input:\n{context}\n\n### Response:"
)
WITHOUT_CONTEXT = "### Instruction:\n{instruction}\n\n### Response:"

# llama.cpp reads this template from the GGUF metadata, so llama cli and the web UI
# format the input the same way that this file formats the training data.
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "### Instruction:\n{{ message['content'] }}\n\n### Response:\n"
    "{% elif message['role'] == 'assistant' %}"
    "{{ message['content'] }}{{ eos_token }}\n\n"
    "{% endif %}"
    "{% endfor %}"
)


def format_prompt(instruction, context=""):
    """Return the prompt text that ends where the response begins."""
    if context and context.strip():
        return WITH_CONTEXT.format(instruction=instruction.strip(), context=context.strip())
    return WITHOUT_CONTEXT.format(instruction=instruction.strip())


def build_example(tokenizer, pair, max_length):
    """Return the token ids and the labels for one pair.

    The labels hold IGNORE_INDEX over the prompt, so the loss reads the response
    alone. A loss that reads the prompt teaches the model to invent instructions,
    which shows as a model that asks questions instead of answering them.
    """
    prompt = format_prompt(pair["instruction"], pair.get("context", ""))
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    response_ids = tokenizer.encode("\n" + pair["response"].strip(), add_special_tokens=False)
    response_ids = response_ids + [tokenizer.eos_token_id]

    input_ids = (prompt_ids + response_ids)[:max_length]
    labels = ([IGNORE_INDEX] * len(prompt_ids) + response_ids)[:max_length]
    return input_ids, labels


class InstructDataset(Dataset):
    """Hold the pairs, and pad each one to the same length."""

    def __init__(self, pairs, tokenizer, max_length=512):
        self.pairs = list(pairs)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        input_ids, labels = build_example(self.tokenizer, self.pairs[index], self.max_length)
        padding = self.max_length - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * padding
        # The padded label carries IGNORE_INDEX, or the model learns to write padding.
        input_ids = input_ids + [self.tokenizer.eos_token_id] * padding
        labels = labels + [IGNORE_INDEX] * padding
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
