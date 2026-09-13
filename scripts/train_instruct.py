"""Step 3b. Teach the pretrained model to answer an instruction.

Pretraining taught the model to continue any text. This stage teaches one
behaviour: read an instruction, write a response, then stop. The algorithm does
not change. Only the data changes, and the loss reads the response alone.
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import Trainer, TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _env import optional  # noqa: E402

from xslm.callbacks import TimeLimitCallback  # noqa: E402
from xslm.config import load_tokenizer, load_yaml  # noqa: E402
from xslm.instruct import CHAT_TEMPLATE, InstructDataset, format_prompt  # noqa: E402
from xslm.model import XSLMForCausalLM  # noqa: E402


def load_pairs(dataset_name):
    """Return the instruction pairs as a list of dictionaries."""
    from datasets import load_dataset

    rows = load_dataset(dataset_name, split="train")
    return [
        {
            "instruction": row["instruction"],
            "context": row.get("context", ""),
            "response": row["response"],
        }
        for row in rows
        if row["response"].strip()
    ]


def resolve_report_to():
    """Return the reporting targets. An absent key must never stop the run."""
    if optional("WANDB_API_KEY"):
        return ["wandb"]
    print("WANDB_API_KEY is absent, so the run reports to no dashboard.", flush=True)
    return []


def show_one_answer(model, tokenizer, instruction):
    """Print one answer, so the log shows whether the model answers at all."""
    prompt = format_prompt(instruction) + "\n"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        # The model holds no key and value cache, so use_cache must be False.
        generated = model.generate(
            input_ids,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            use_cache=False,
            eos_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"\n--- sample ---\n{text}\n--- end ---\n", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Instruction tune the xSLM model.")
    parser.add_argument("--smoke", type=int, default=0, help="Run this many steps and stop.")
    arguments = parser.parse_args()

    config = load_yaml()
    settings = config["instruct"]
    torch.manual_seed(settings["seed"])

    model = XSLMForCausalLM.from_pretrained(settings["base_model"])
    tokenizer = load_tokenizer(settings["base_model"])
    tokenizer.chat_template = CHAT_TEMPLATE

    pairs = load_pairs(settings["dataset"])
    print(f"Read {len(pairs)} instruction pairs.", flush=True)
    split = settings["val_pairs"]
    train_dataset = InstructDataset(pairs[split:], tokenizer, settings["max_length"])
    eval_dataset = InstructDataset(pairs[:split], tokenizer, settings["max_length"])

    training_arguments = TrainingArguments(
        output_dir=settings["output_dir"],
        num_train_epochs=settings["num_train_epochs"],
        max_steps=arguments.smoke if arguments.smoke else -1,
        per_device_train_batch_size=settings["per_device_train_batch_size"],
        gradient_accumulation_steps=settings["gradient_accumulation_steps"],
        learning_rate=settings["learning_rate"],
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": settings["min_learning_rate"]},
        warmup_steps=settings["warmup_steps"],
        weight_decay=settings["weight_decay"],
        bf16=torch.cuda.is_available(),
        logging_steps=settings["logging_steps"],
        eval_strategy="no" if arguments.smoke else "steps",
        eval_steps=settings["eval_steps"],
        save_strategy="no" if arguments.smoke else "steps",
        save_steps=settings["save_steps"],
        save_total_limit=settings["save_total_limit"],
        report_to=resolve_report_to(),
        seed=settings["seed"],
    )

    callbacks = [] if arguments.smoke else [TimeLimitCallback(settings["time_limit_minutes"])]
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks,
    )
    trainer.train()

    if arguments.smoke:
        show_one_answer(model.eval(), tokenizer, "Name three colours.")
        return

    trainer.save_model(settings["output_dir"])
    # The template travels with the tokenizer, so llama.cpp writes it into the GGUF
    # metadata and llama cli formats the input the way this run formatted the data.
    tokenizer.save_pretrained(settings["output_dir"])
    metrics = trainer.evaluate()
    print(f"eval_loss = {metrics['eval_loss']:.4f}")
    model.eval()
    for instruction in ("Name three colours.", "What is the capital of France?"):
        show_one_answer(model, tokenizer, instruction)


if __name__ == "__main__":
    main()
