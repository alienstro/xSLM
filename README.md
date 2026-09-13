# xSLM

A 70M parameter Llama-style language model, written from scratch, pretrained on one
GPU in 180 minutes, then instruction tuned in 8 minutes. The whole run costs about
one US dollar.

The model is on the Hub: **[Alienstro/xslm](https://huggingface.co/Alienstro/xslm)**

> The model writes fluent English and invents facts. At 70M parameters there is no
> room to store world knowledge, and 629M training tokens is about one five
> thousandth of what a small production model reads. Read
> [What the model cannot do](#what-the-model-cannot-do) before you judge an answer.

## Run it

The tuned model answers an instruction. Install llama.cpp from
[llama.app](https://llama.app), then:

```bash
llama cli --jinja -hf Alienstro/xslm \
    --hf-file gguf/xslm-70m-instruct-Q8_0.gguf \
    --temp 0.3 --repeat-penalty 1.2
```

```
> Who is George Washington?
George Washington was born in 1867 and died on April 25, 1943. He has been the most
famous American president since he was elected to the United States Senate in 1907.
```

Three flags matter, and each one fixes a real fault:

| Flag | Without it |
|---|---|
| `--jinja` | llama.cpp refuses the template: `this custom template is not supported` |
| `--temp 0.3` | The default 0.8 picks unlikely tokens, and a small model drifts |
| `--repeat-penalty 1.2` | The default 1.0 lets the model loop one sentence forever |

Type `/clear` between two questions on different subjects. The chat keeps every
earlier turn, and a poor answer pollutes the next one.

The base model holds no template, and it continues text instead of answering it:

```bash
llama completion -hf Alienstro/xslm:Q8_0 -p 'The history of' -n 120
```

## Results

| Stage | Data | Tokens or pairs | Time | Loss |
|---|---|---|---|---|
| Pretraining | `HuggingFaceFW/fineweb-edu`, `sample-10BT` | 629M tokens | 2 h 58 m | 3.4195 |
| Instruction tuning | `databricks/databricks-dolly-15k` | 14,700 pairs, 3 epochs | 8 m | 2.7801 |

A perplexity of 30.55 follows from the pretraining loss. The two losses compare to
nothing: the first reads every token of web text, and the second reads the response
tokens of a conversation alone.

## The model

| Field | Value |
|---|---|
| Parameters | 70,546,560, of which 49,575,040 sit outside the embedding |
| Layers | 10 |
| Hidden size | 640 |
| Heads | 10 |
| Feed forward | 1728, SwiGLU |
| Context | 1024 tokens |
| Vocabulary | 32,768 byte level BPE, trained on the same corpus |
| Normalization | RMSNorm, pre-normalization |
| Positions | RoPE, theta 10000 |
| Embedding | Tied to the output projection |

Every parameter carries the name that Llama gives it, so `transformers` loads the
model without `trust_remote_code`, and llama.cpp converts it without a custom
converter. A test asserts that the state dictionary matches `LlamaForCausalLM` key
for key.

## What the model cannot do

It does not know facts. It learns grammar, style, and the shape of a document, and
it fills every slot with a word that fits the shape.

| Model | Parameters | Training tokens |
|---|---|---|
| **xSLM** | 70M | 0.6B |
| Qwen3-0.6B | 600M | about 30T |
| Llama 3 8B | 8B | 15T |

Factual recall needs parameters to hold the fact and tokens to teach it. This model
has neither. That is the expected result at this size, not a defect.

## Build it yourself

```bash
uv sync
cp .env.example .env   # then write the values
uv run pytest -q       # 123 tests, CPU only
```

Read **[AGENTS.md](AGENTS.md)** before you rent a pod. It holds eleven rules, and
each one comes from a failure that happened once in this repository. The first rule
is to arm the watchdog before any other pod work.

Read **[docs/RUNBOOK.md](docs/RUNBOOK.md)** for the ordered procedure. It indexes
every trap by its symptom, so a future run finds the entry from the error text.

```bash
scripts/pod.sh check                                  # nvidia-smi, cmake, tmux, rsync
scripts/pod.sh sync                                   # rsync the working tree
scripts/pod.sh secrets                                # copy .env with mode 600
scripts/pod.sh setup                                  # uv sync, then pytest on CUDA

scripts/pod.sh run bash scripts/watchdog.sh 300 15    # arm the safety net first
scripts/pod.sh run bash scripts/run_data_pipeline.sh  # tokenizer and token files
scripts/pod.sh watch run_data_pipeline 1              # poll, and refresh the heartbeat

scripts/pod.sh run scripts/train.py --smoke 20        # read the throughput and the VRAM
# write max_steps into configs/base.yaml, then sync again
scripts/pod.sh run scripts/train.py                   # 180 min
scripts/pod.sh run scripts/sample.py                  # read the text before you publish
scripts/pod.sh run scripts/push_to_hub.py
scripts/pod.sh run bash scripts/quantize.sh

scripts/pod.sh run scripts/train_instruct.py          # 8 min
scripts/pod.sh run scripts/push_to_hub.py --stage instruct
scripts/pod.sh run bash scripts/quantize.sh out/publish-instruct out/gguf-instruct xslm-70m-instruct

scripts/pod.sh verify                                 # every file must be on the Hub
scripts/pod.sh terminate --yes                        # only after verify exits zero
```

Every run lives in `tmux`, so it survives a closed laptop and a dropped network.

### The safety net

`scripts/watchdog.sh` runs on the pod, so it works when the laptop sleeps and the
agent session ends. It fires on a deadline, on a finished pipeline, or when
`out/HEARTBEAT` grows older than 15 minutes. It then asks `verify_upload.py` whether
the Hub holds every file.

| Answer | Action |
|---|---|
| Every file is present | Terminate. Nothing is left to lose. |
| Any file is absent | Stop. The billing ends, and the volume keeps the work. |

> Warning: `REQUIRED` in `verify_upload.py` must name the files of the stage that
> you are about to run, before you start it. A list that names only finished work
> passes at once, and the watchdog then destroys the run that you just made.

## Layout

| Path | Holds |
|---|---|
| `configs/base.yaml` | Every hyperparameter, and no secret |
| `src/xslm/config.py` | The `LlamaConfig` factory, and the tokenizer loader |
| `src/xslm/model.py` | RMSNorm, RoPE, SwiGLU, attention, block, causal model |
| `src/xslm/data.py` | The memory-mapped packed dataset |
| `src/xslm/instruct.py` | The prompt format, the label mask, and the chat template |
| `src/xslm/callbacks.py` | The time limit and the sample callbacks |
| `scripts/` | The pipeline steps, the pod helper, and the watchdog |
| `tests/` | 123 tests, CPU only |
| `AGENTS.md` | The standing rules for an agent |
| `docs/RUNBOOK.md` | The procedure, and every trap indexed by symptom |

## Licence

Apache 2.0. The instruction data carries CC BY-SA 3.0 from Databricks.
