# Rules for an agent that works on xSLM

Read this file before you rent a pod, start a run, or publish a model. Every rule
here comes from a failure that happened once. Do not repeat the failure.

## 1. Arm the watchdog before any other pod work

Arm the watchdog first, before the sync, before the setup, and before any run.

```bash
scripts/pod.sh run bash scripts/watchdog.sh <deadline-minutes> 15
```

The pod bills by the hour. A laptop that sleeps, a session that ends, or a run that
hangs leaves the pod running until the money runs out. The watchdog is the only
guard that works when nobody watches.

A run that starts before the watchdog is a run with no guard. Never do it.

## 2. Point the safety check at the current stage first

`scripts/watchdog.sh` asks `scripts/verify_upload.py` one question: does the Hub
hold every file? The answer decides what happens to the pod.

| Answer | Action | Result |
|---|---|---|
| Every file is present | terminate | The disk dies, and nothing is lost |
| Any file is absent | stop | The billing ends, and the volume keeps the work |

So `REQUIRED` in `verify_upload.py` must name the files of the stage that you are
about to run, **before** you start that stage.

> Warning: a `REQUIRED` list that names only the finished work passes at once. The
> watchdog then terminates the pod and destroys the run that you just made. This
> happened during the instruction tuning run. Add the new files to `REQUIRED`
> first, then sync, then start the run.

Prove the check fails before you trust it:

```bash
scripts/pod.sh verify    # must exit non-zero while the new work is unpublished
```

## 3. Refresh the heartbeat every 15 minutes

Every `scripts/pod.sh watch` touches `out/HEARTBEAT`. The watchdog stops the pod
when that file is older than 15 minutes, because a silent agent means that nobody
will publish the model.

Read the log at least every 15 minutes while a run is alive. A long wait needs a
loop that polls, not a single long sleep.

## 4. Never terminate before the check passes

```bash
scripts/pod.sh verify          # must exit zero
scripts/pod.sh terminate --yes # only then
```

> Warning: `terminate` destroys the pod disk and cannot be undone. The pod holds
> the only copy of the weights until the Hub holds them.

## 5. Leave no process behind

Kill every process that you started on a shared machine when you finish. Stray
inference processes compete for the cores and make the next measurement wrong. A
model that read 243 tokens a second measured 3.6 tokens a second because three
stray processes shared four cores.

```bash
pkill -f 'llama serve'; pkill -f 'llama cli'
```

## 6. Read one number before you trust a run

A loss that starts far from `ln(vocab_size)` means a broken loss, not a hard task.
For this model that number is `ln(32768) = 10.4`.

Check the first logged loss of every training run. A loss of 160 means that the
loss is 16 times too large, which was the gradient accumulation bug.

## 7. Test the reload, not only the training

A model that trains well can still load broken. `from_pretrained` builds the model
on the meta device and fills it from the checkpoint, so a buffer that no checkpoint
holds keeps empty memory.

Run `scripts/sample.py` after every training run, and read the text. A validation
loss proves nothing about the file that other people download.

## 8. Never guess a tokenizer value

`tokenizer.ggml.pre` chooses the regex list that llama.cpp applies. A wrong value
raises no error. It only splits the text differently from the training, which shows
as a quiet loss of quality.

Prove the two tokenizers agree:

```bash
llama.cpp/build/bin/llama-tokenize -m <file>.gguf --ids -p '<text with digits>'
```

Compare against `AutoTokenizer.encode` of the same text. The lists must match.

## 9. Take the checksum from the tool that reports it

`convert_hf_to_gguf.py` computes a checksum over a probe string that it owns. A
hand written copy of that string drifts, and the checksum then never matches. Read
the value that the converter prints when it fails.

## 10. Write the test before the fix

Every fix in this repository has a test that fails without it. A silent bug, such
as a label mask or a loss scale, needs a test that reads the number, not a test
that reads the shape.

## 11. Prove the template before you destroy the pod

A chat template that reaches the GGUF can still fail to render. Run the tuned file
once, and read the answer, before you terminate the pod:

```bash
llama cli --jinja -hf <repo> --hf-file gguf/<name>-instruct-Q8_0.gguf
```

llama.cpp refuses a custom template on its old path, and it names the remedy in the
error: `this custom template is not supported, try using --jinja`.
