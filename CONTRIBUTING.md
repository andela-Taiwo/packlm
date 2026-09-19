# Contributing to packlm

Thanks for taking the time to help. This guide covers setting up, making a
change and raising an issue.

## Setting up

You need [uv](https://docs.astral.sh/uv/) and Python 3.11, 3.12 or 3.13.

```bash
git clone https://github.com/andela-Taiwo/packlm.git
cd packlm
uv sync            # installs the locked dependencies, dev tools included
uv run pytest      # should pass before you change anything
```

Every dependency is pinned. `pyproject.toml` pins the direct ones and `uv.lock`
pins everything else with hashes. To change a version, edit `pyproject.toml`,
run `uv lock`, and commit both files together.

## Making a change

1. **Open or find an issue first** for anything bigger than a typo, so we can
   agree on the approach before you spend time on it.
2. **Branch from `main`**, named for the change: `fix/positions-reset`,
   `feat/json-export`, `docs/objective-terms`.
3. **Keep the documented contract.** `docs/` is the specification:
   - stdout carries only the step lines and `wrote <out>`; diagnostics go
     through `lm/log.py` to stderr;
   - a run must give identical weights when repeated;
   - memory per step must stay within the budget in `docs/README.md`;
   - if a change alters documented behaviour, update `docs/` in the same PR.
4. **Add or update tests** for any behaviour you change.
5. **Check locally:**
   ```bash
   uv run ruff check .
   uv run pytest
   ```
6. **Write clear commits**: a short imperative subject ("Reset scan state at
   segment starts"), and a body explaining why when it isn't obvious.
7. **Open a pull request** against `main` and fill in the template. CI runs
   lint, tests on Python 3.11 and 3.12, and a Docker build with a smoke run;
   all three must pass before merge.

No new runtime dependencies without discussing it in an issue first. The
project deliberately depends on NumPy and PyTorch only.

## Raising an issue

Please [search existing issues](https://github.com/andela-Taiwo/packlm/issues)
first, then open a new one with the template that fits:

- **Bug report**: something doesn't behave as `docs/` says.
- **Feature request**: something new you'd like packlm to do.

A good bug report includes:

- what you ran (the exact command and config);
- what you expected, with the part of `docs/` that says so if you can;
- what happened instead (paste the step lines, error or traceback);
- your environment: `uv run python --version`, OS and CPU architecture, and
  `uv run python -c "import torch, numpy; print(torch.__version__, numpy.__version__)"`;
- the smallest corpus that shows the problem, if the bug depends on the data;
- structured logs if useful: rerun with `LM_LOG_LEVEL=DEBUG` and attach stderr.

For a security problem, don't open a public issue; contact the maintainer
through their GitHub profile instead.
