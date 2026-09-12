#!/usr/bin/env bash
# Step 6. Build llama.cpp, convert the model to GGUF, and quantize it three times.
#
# The script writes two sixteen bit files. BF16 keeps the dtype that trained the
# model, and F16 serves the older tools that read no BF16 tensor. Each one is
# converted from the safetensors file, because a conversion from BF16 to F16
# would flush the smallest weights to zero.
#
# The build needs no CUDA. llama-quantize is CPU work, and a 70M model quantizes
# in seconds.
set -euo pipefail

STAGING="${1:-out/publish}"
OUTPUT="${2:-out/gguf}"
NAME="xslm-70m"
LLAMA_DIR="${LLAMA_CPP_DIR:-llama.cpp}"

mkdir -p "$OUTPUT"

if [ ! -d "$LLAMA_DIR" ]; then
    git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
fi

if [ ! -x "$LLAMA_DIR/build/bin/llama-quantize" ]; then
    cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DGGML_CUDA=OFF -DLLAMA_CURL=OFF
    cmake --build "$LLAMA_DIR/build" --target llama-quantize -j "$(nproc)"
fi

# The converter knows no checksum for a tokenizer that nobody has published.
uv run python scripts/patch_llama_cpp.py "$LLAMA_DIR/conversion/base.py"

uv run --with gguf --with sentencepiece --with protobuf python "$LLAMA_DIR/convert_hf_to_gguf.py" "$STAGING" \
    --outfile "$OUTPUT/$NAME-BF16.gguf" \
    --outtype bf16

uv run --with gguf --with sentencepiece --with protobuf python "$LLAMA_DIR/convert_hf_to_gguf.py" "$STAGING" \
    --outfile "$OUTPUT/$NAME-F16.gguf" \
    --outtype f16

for QUANT in Q8_0 Q6_K Q4_K_M; do
    "$LLAMA_DIR/build/bin/llama-quantize" \
        "$OUTPUT/$NAME-BF16.gguf" \
        "$OUTPUT/$NAME-$QUANT.gguf" \
        "$QUANT"
done

ls -lh "$OUTPUT"

uv run python -c "
import os
from huggingface_hub import HfApi

api = HfApi(token=os.environ['HF_TOKEN'])
api.upload_folder(
    folder_path='$OUTPUT',
    path_in_repo='gguf',
    repo_id=os.environ['HF_REPO_ID'],
    repo_type='model',
)
print('Pushed the gguf folder.')
"
