import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def run_module(module, *args, cwd):
    env = dict(os.environ, PYTHONPATH=str(ROOT), OMP_NUM_THREADS="1")
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        cwd=cwd, env=env, capture_output=True, text=True, check=False,
    )


@pytest.fixture
def smoke_run(tmp_path):
    """A copy of the smoke config and corpus in a scratch directory."""
    (tmp_path / "configs").mkdir()
    (tmp_path / "data" / "smoke").mkdir(parents=True)
    shutil.copy(ROOT / "configs" / "smoke.toml", tmp_path / "configs" / "smoke.toml")
    shutil.copy(ROOT / "data" / "smoke" / "corpus.jsonl", tmp_path / "data" / "smoke" / "corpus.jsonl")
    return tmp_path
