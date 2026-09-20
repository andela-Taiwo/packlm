# packlm

[![CI](https://github.com/andela-Taiwo/packlm/actions/workflows/ci.yml/badge.svg)](https://github.com/andela-Taiwo/packlm/actions/workflows/ci.yml)

Small decoder-only language models for tokenizer and data-mix experiments,
trained on CPU in float64 so every run is exactly repeatable.

Documents are packed several to a row before training, which keeps rows full
and steps cheap. The packed layout is only a layout: the same corpus and
config should train the same model whichever way the packer arranged it.

The models stack pre-norm blocks, each a sequence mixer (causal attention, a
short gated convolution, or a gated linear recurrence) followed by a dense MLP
or a mixture of experts with optional expert capacity.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11 to 3.13.

```bash
uv sync
```

This installs the exact versions in `uv.lock`, with the CPU build of PyTorch.

## Quick start

From the repository root:

```bash
uv run python -m lm.pack --docs data/smoke/corpus.jsonl --ctx 32 --micro-batches 2 --out data/smoke/shards
uv run python -m lm.init --config configs/smoke.toml --seed 0
uv run python -m lm.train --config configs/smoke.toml
```

## Docker

The image is a multi-stage build: dependencies are installed from the lockfile
in a builder stage, and the runtime stage holds only the virtual environment
and the code, running as a non-root user.

```bash
docker build -t packlm .
docker run --rm packlm lm.train --help
docker run --rm -v "$PWD/runs:/app/runs" --entrypoint sh packlm -c '
  python -m lm.pack --docs data/smoke/corpus.jsonl --ctx 32 --micro-batches 2 --out data/smoke/shards &&
  python -m lm.init --config configs/smoke.toml --seed 0 &&
  python -m lm.train --config configs/smoke.toml'
```

## Logging

stdout carries only the documented output. Diagnostics are structured logs on
stderr, one JSON object per line:

```bash
LM_LOG_LEVEL=DEBUG uv run python -m lm.train --config configs/smoke.toml 2> train.log.jsonl
```

| Variable | Values | Default |
|---|---|---|
| `LM_LOG_LEVEL` | `DEBUG`, `INFO`, `WARNING`, `ERROR` | `WARNING` |
| `LM_LOG_FORMAT` | `json`, `text` | `json` |

`INFO` logs the start and end of each run; `DEBUG` adds one event per step
with its timing, loss and gradient norm.

## Documentation

- [`docs/README.md`](docs/README.md): commands, config reference, output format, memory budget
- [`docs/data.md`](docs/data.md): corpus format, packing and the step shard format
- [`docs/objective.md`](docs/objective.md): exactly what one optimizer step minimises

## Development

```bash
uv run ruff check .
uv run pytest
```

CI runs lint, the tests on Python 3.11 and 3.12, and a Docker build with a
smoke run on every push and pull request to `main`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, the pull request checklist
and how to raise an issue.

## Licence

MIT. See [LICENSE](LICENSE).
