"""Step 3. Run the smoke test or the full training run."""

import argparse
import math
import sys
from pathlib import Path

import torch
from transformers import PreTrainedTokenizerFast, Trainer, TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _env import optional  # noqa: E402

from xslm.callbacks import SampleGenerationCallback, TimeLimitCallback  # noqa: E402
from xslm.config import END_OF_TEXT, build_config, load_yaml  # noqa: E402
from xslm.data import PackedDataset  # noqa: E402
from xslm.model import build_model  # noqa: E402


def compute_max_steps(tokens_per_second, seconds, tokens_per_step):
    """Return the step count that fills the time budget. Never return zero."""
    return max(1, int(tokens_per_second * seconds / tokens_per_step))


def resolve_report_to():
    """Return the reporting targets.

    The 180 minute run must never stop because of a telemetry failure, so an absent
    key turns the reporting off instead of raising.
    """
    if optional("WANDB_API_KEY"):
        return ["wandb"]
    print("WANDB_API_KEY is absent, so the run reports to no dashboard.", flush=True)
    return []


def main():
    parser = argparse.ArgumentParser(description="Train the xSLM model.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--smoke", type=int, default=0, help="Run this many steps and stop.")
    arguments = parser.parse_args()

    config = load_yaml(arguments.config) if arguments.config else load_yaml()
    settings = config["training"]
    seq_len = config["model"]["max_position_embeddings"]

    torch.manual_seed(settings["seed"])
    model = build_model(build_config())
    # Name the special tokens here. PreTrainedTokenizerFast reads none of them from
    # tokenizer.json, and a saved config without eos_token makes a model that never
    # stops generating.
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=config["tokenizer"]["output_path"],
        bos_token=END_OF_TEXT,
        eos_token=END_OF_TEXT,
    )

    train_dataset = PackedDataset(config["data"]["train_path"], seq_len, settings["seed"])
    eval_dataset = PackedDataset(config["data"]["val_path"], seq_len, settings["seed"])

    max_steps = arguments.smoke if arguments.smoke else settings["max_steps"]
    training_arguments = TrainingArguments(
        output_dir=settings["output_dir"],
        max_steps=max_steps,
        per_device_train_batch_size=settings["per_device_train_batch_size"],
        gradient_accumulation_steps=settings["gradient_accumulation_steps"],
        learning_rate=settings["learning_rate"],
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": settings["min_learning_rate"]},
        warmup_steps=settings["warmup_steps"],
        weight_decay=settings["weight_decay"],
        adam_beta1=settings["adam_beta1"],
        adam_beta2=settings["adam_beta2"],
        adam_epsilon=settings["adam_epsilon"],
        max_grad_norm=settings["max_grad_norm"],
        bf16=settings["bf16"] and torch.cuda.is_available(),
        gradient_checkpointing=settings["gradient_checkpointing"],
        torch_compile=settings["torch_compile"],
        dataloader_num_workers=settings["dataloader_num_workers"],
        logging_steps=settings["logging_steps"],
        eval_strategy="no" if arguments.smoke else "steps",
        eval_steps=settings["eval_steps"],
        save_strategy="no" if arguments.smoke else "steps",
        save_steps=settings["save_steps"],
        save_total_limit=settings["save_total_limit"],
        report_to=resolve_report_to(),
        seed=settings["seed"],
    )

    callbacks = []
    if not arguments.smoke:
        callbacks.append(TimeLimitCallback(settings["time_limit_minutes"]))
        callbacks.append(
            SampleGenerationCallback(
                tokenizer,
                prompt="The history of",
                every_steps=settings["sample_every_steps"],
            )
        )

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks,
    )
    result = trainer.train()

    if arguments.smoke:
        # transformers 5 removed include_tokens_per_second, so measure it here.
        tokens_per_step = (
            settings["per_device_train_batch_size"]
            * settings["gradient_accumulation_steps"]
            * seq_len
        )
        throughput = arguments.smoke * tokens_per_step / result.metrics["train_runtime"]
        seconds = settings["time_limit_minutes"] * 60
        print(f"train_tokens_per_second = {throughput:.0f}")
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / 2**30
            print(f"peak VRAM = {peak:.1f} GiB")
        print(f"Set max_steps to {compute_max_steps(throughput, seconds, tokens_per_step)}")
        return

    trainer.save_model(settings["output_dir"])
    tokenizer.save_pretrained(settings["output_dir"])
    metrics = trainer.evaluate()
    print(f"eval_loss = {metrics['eval_loss']:.4f}")
    print(f"perplexity = {math.exp(metrics['eval_loss']):.2f}")


if __name__ == "__main__":
    main()
