"""Two callbacks: one stops the run on time, one shows what the model writes."""

import time

import torch
from transformers import TrainerCallback


class TimeLimitCallback(TrainerCallback):
    """Stop the run after a fixed number of minutes and save a checkpoint.

    The pod rental is fixed, so the run must end with a saved checkpoint inside the
    budget. A cosine schedule that does not finish is better than no checkpoint.
    """

    def __init__(self, time_limit_minutes):
        self.limit_seconds = time_limit_minutes * 60.0
        self.started_at = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.started_at = time.monotonic()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self.started_at is None:
            self.started_at = time.monotonic()
        if time.monotonic() - self.started_at >= self.limit_seconds:
            control.should_save = True
            control.should_training_stop = True
        return control


class SampleGenerationCallback(TrainerCallback):
    """Write one short sample to the log at a fixed interval.

    A loss number does not show whether the text reads as English. A sample does.
    """

    def __init__(self, tokenizer, prompt, every_steps, max_new_tokens=100):
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.every_steps = every_steps
        self.max_new_tokens = max_new_tokens

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None or state.global_step == 0:
            return control
        if state.global_step % self.every_steps != 0:
            return control
        was_training = model.training
        model.eval()
        input_ids = self.tokenizer(self.prompt, return_tensors="pt").input_ids
        with torch.no_grad():
            # This model holds no key and value cache, so use_cache must be False.
            generated = model.generate(
                input_ids.to(model.device),
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                use_cache=False,
            )
        text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"\n[step {state.global_step}] {text}\n", flush=True)
        if was_training:
            model.train()
        return control
