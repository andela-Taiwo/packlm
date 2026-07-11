import re

import numpy as np

from conftest import run_module

STEP_LINE = re.compile(
    r"^step \d+ \| loss \S+ \| ce \S+ \| balance \S+ \| z \S+ \| targets \d+ \| gnorm \S+$"
)


def _pack_init(cwd):
    r = run_module("lm.pack", "--docs", "data/smoke/corpus.jsonl", "--ctx", "32",
                   "--micro-batches", "2", "--out", "data/smoke/shards", cwd=cwd)
    assert r.returncode == 0, r.stderr
    r = run_module("lm.init", "--config", "configs/smoke.toml", "--seed", "0", cwd=cwd)
    assert r.returncode == 0, r.stderr


def test_readme_commands_train_the_smoke_model(smoke_run):
    _pack_init(smoke_run)
    r = run_module("lm.train", "--config", "configs/smoke.toml", cwd=smoke_run)
    assert r.returncode == 0, r.stderr

    lines = r.stdout.splitlines()
    assert len(lines) == 9
    assert all(STEP_LINE.match(line) for line in lines[:8])
    assert lines[-1] == f"wrote {(smoke_run / 'runs' / 'smoke' / 'final.npz').resolve()}"

    with np.load(smoke_run / "runs/smoke/init.npz") as a, np.load(smoke_run / "runs/smoke/final.npz") as b:
        assert set(a.files) == set(b.files)
        assert any(not np.array_equal(a[k], b[k]) for k in a.files)


def _arrays(path):
    with np.load(path) as z:
        return {k: z[k].copy() for k in z.files}


def _assert_same(a, b):
    assert set(a) == set(b)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])


def test_training_is_repeatable(smoke_run):
    _pack_init(smoke_run)
    first = run_module("lm.train", "--config", "configs/smoke.toml", cwd=smoke_run)
    weights = _arrays(smoke_run / "runs/smoke/final.npz")
    second = run_module("lm.train", "--config", "configs/smoke.toml", cwd=smoke_run)
    assert first.stdout == second.stdout
    _assert_same(weights, _arrays(smoke_run / "runs/smoke/final.npz"))


def test_init_is_deterministic_per_seed(smoke_run):
    _pack_init(smoke_run)
    first = _arrays(smoke_run / "runs/smoke/init.npz")
    run_module("lm.init", "--config", "configs/smoke.toml", "--seed", "0", cwd=smoke_run)
    _assert_same(first, _arrays(smoke_run / "runs/smoke/init.npz"))


def test_refused_config_writes_nothing(smoke_run):
    _pack_init(smoke_run)
    cfg = smoke_run / "configs" / "bad.toml"
    cfg.write_text((smoke_run / "configs" / "smoke.toml").read_text().replace("lr = 0.01", "lr = -1"))
    r = run_module("lm.train", "--config", "configs/bad.toml", cwd=smoke_run)
    assert r.returncode != 0
    assert not (smoke_run / "runs" / "smoke" / "final.npz").exists()
