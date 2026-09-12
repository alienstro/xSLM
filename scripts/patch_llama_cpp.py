"""Teach the GGUF converter which pre-tokenizer this tokenizer uses.

convert_hf_to_gguf.py matches a checksum of the tokenizer against a table of known
models, and it raises NotImplementedError for a tokenizer that it has never seen.
Every newly trained tokenizer is unknown, so the table needs one more entry.

The value chooses the regex list that llama.cpp applies at inference. The xSLM
tokenizer splits each digit, then applies the GPT-2 pattern:

    Digits(individual_digits=True), ByteLevel(add_prefix_space=False, use_regex=True)

llama.cpp calls that exact pair "refact", in llama-vocab.cpp:

    "\\p{N}",
    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",

A wrong value here changes no weight and raises no error. It only makes llama.cpp
split the text differently from the training, which shows as a quiet loss of
quality. Never guess this value.
"""

import sys
from pathlib import Path

# The converter prints this value when it fails. Never compute it by hand: the
# probe string belongs to the converter, and a copy of it drifts.
CHKHSH = "84585b9518029852d17987d7b642aba9cc67ce328a318273ea9a60a3b78b85cb"
PRE_TYPE = "refact"
FUNCTION = "def get_vocab_base_pre"
ANCHOR = '        if chkhsh == "'
ADDITION = f'''        if chkhsh == "{CHKHSH}":
            # The xSLM tokenizer splits each digit, then applies the GPT-2 pattern.
            res = "{PRE_TYPE}"
'''


def patch(path):
    """Insert the xSLM entry into the checksum table. Return True when it writes."""
    text = Path(path).read_text()
    if CHKHSH in text:
        print(f"{path} already holds the xSLM checksum.")
        return False
    # The file holds the string chkhsh in more than one method, so start the search
    # inside the method that reads the table.
    start = text.find(FUNCTION)
    if start < 0:
        raise SystemExit(f"{path} holds no {FUNCTION}. The converter changed.")
    index = text.find(ANCHOR, start)
    if index < 0:
        raise SystemExit(f"{path} holds no checksum table. The converter changed.")
    Path(path).write_text(text[:index] + ADDITION + text[index:])
    print(f"Wrote the xSLM checksum into {path}, as pre type {PRE_TYPE}.")
    return True


if __name__ == "__main__":
    patch(sys.argv[1] if len(sys.argv) > 1 else "llama.cpp/conversion/base.py")
