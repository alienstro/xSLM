"""Step 4. Generate text from a trained checkpoint."""

import argparse

import torch
from transformers import AutoTokenizer

from xslm.config import load_yaml
from xslm.model import XSLMForCausalLM


def generate_text(model, tokenizer, prompt, max_new_tokens=200, temperature=0.8, top_p=0.95):
    """Return the prompt plus the generated continuation."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        # This model holds no key and value cache, so use_cache must be False.
        generated = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            use_cache=False,
        )
    return tokenizer.decode(generated[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Generate text from a checkpoint.")
    parser.add_argument("--prompt", default="The history of")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--checkpoint", default=None)
    arguments = parser.parse_args()

    config = load_yaml()
    checkpoint = arguments.checkpoint or config["training"]["output_dir"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = XSLMForCausalLM.from_pretrained(checkpoint).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    print(generate_text(model, tokenizer, arguments.prompt, arguments.max_new_tokens))


if __name__ == "__main__":
    main()
