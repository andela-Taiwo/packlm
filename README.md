# packlm

Small decoder-only language models for tokenizer and data-mix experiments,
trained on CPU in float64 so every run is exactly repeatable.

Documents are packed several to a row before training, which keeps rows full
and steps cheap. The packed layout is only a layout: the same corpus and
config should train the same model whichever way the packer arranged it.

The models stack pre-norm blocks, each a sequence mixer (causal attention, a
short gated convolution, or a gated linear recurrence) followed by a dense MLP
or a mixture of experts with optional expert capacity.

## Setup

Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` installs the CPU build of PyTorch.

## Quick start

From the repository root:

```bash
python -m lm.pack --docs data/smoke/corpus.jsonl --ctx 32 --micro-batches 2 --out data/smoke/shards
python -m lm.init --config configs/smoke.toml --seed 0
python -m lm.train --config configs/smoke.toml
```

## Documentation

- [`docs/README.md`](docs/README.md): commands, config reference, output format, memory budget
- [`docs/data.md`](docs/data.md): corpus format, packing and the step shard format
- [`docs/objective.md`](docs/objective.md): exactly what one optimizer step minimises

## Tests

```bash
pip install pytest
pytest
```

## Licence

MIT. See [LICENSE](LICENSE).
