from pathlib import Path

import pytest
from lm.config import load_config

ROOT = Path(__file__).resolve().parent.parent

BASE = """
[model]
vocab = 16
d_model = 8
n_heads = 2
ctx = 8
mixers = ["attn"]
{model_extra}

[train]
lr = 0.01
{train_extra}

[run]
shards = "shards"
init = "init.npz"
out = "out/final.npz"
"""


def write(tmp_path, model_extra="", train_extra=""):
    path = tmp_path / "run.toml"
    path.write_text(BASE.format(model_extra=model_extra, train_extra=train_extra))
    return path


def test_smoke_config_loads():
    cfg = load_config(ROOT / "configs" / "smoke.toml")
    assert cfg.model.mlp == "moe"
    assert cfg.model.experts == 4 and cfg.model.top_k == 2
    assert cfg.train.beta1 == 0.9 and cfg.train.beta2 == 0.95
    assert Path(cfg.out).is_absolute()


def test_defaults_and_paths(tmp_path):
    cfg = load_config(write(tmp_path))
    assert cfg.model.positions == "rope"
    assert cfg.model.d_ff == 32
    assert cfg.train.eps == 1e-8
    assert cfg.out == str(tmp_path / "out" / "final.npz")
    assert cfg.steps is None


@pytest.mark.parametrize(
    "model_extra, train_extra",
    [
        ("capacity_factor = -1", ""),
        ("mlp = \"moe\"\nexperts = 2\ntop_k = 3", ""),
        ("positions = \"spiral\"", ""),
        ("", "warmup = 1.5"),
        ("", "betas = [0.9, 1.0]"),
        ("", "eps = 0"),
        ("", "clip = nan"),
    ],
)
def test_invalid_values_are_refused(tmp_path, model_extra, train_extra):
    with pytest.raises((KeyError, ValueError)):
        load_config(write(tmp_path, model_extra, train_extra))


def test_rope_needs_even_head_width(tmp_path):
    path = write(tmp_path)
    path.write_text(path.read_text().replace("n_heads = 2", "n_heads = 8"))
    with pytest.raises(ValueError):
        load_config(path)


def test_missing_table(tmp_path):
    path = tmp_path / "run.toml"
    path.write_text("[model]\nvocab = 16\n")
    with pytest.raises(KeyError):
        load_config(path)
