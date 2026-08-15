import json

from conftest import run_module

PACK = ("lm.pack", "--docs", "data/smoke/corpus.jsonl", "--ctx", "32", "--micro-batches", "2",
        "--out", "data/smoke/shards")
INIT = ("lm.init", "--config", "configs/smoke.toml", "--seed", "0")
TRAIN = ("lm.train", "--config", "configs/smoke.toml")
DEBUG_JSON = {"LM_LOG_LEVEL": "DEBUG", "LM_LOG_FORMAT": "json"}


def _records(stderr):
    """The JSON log records in stderr, skipping anything else (a traceback, a library warning)."""
    records = []
    for line in stderr.splitlines():
        if not line.startswith("{"):
            continue
        record = json.loads(line)
        assert {"ts", "level", "logger", "event"} <= record.keys()
        records.append(record)
    return records


def test_logging_is_silent_by_default(smoke_run):
    r = run_module(*PACK, cwd=smoke_run, env={"LM_LOG_LEVEL": "WARNING"})
    assert r.returncode == 0, r.stderr
    assert r.stderr == ""


def test_pack_logs_json_to_stderr_and_keeps_stdout(smoke_run):
    r = run_module(*PACK, cwd=smoke_run, env=DEBUG_JSON)
    assert r.returncode == 0, r.stderr
    assert r.stdout == "packed 8 steps into data/smoke/shards\n"
    events = [rec["event"] for rec in _records(r.stderr)]
    assert events.count("step_packed") == 8
    assert events[-1] == "pack_done"


def test_train_logging_never_touches_stdout(smoke_run):
    assert run_module(*PACK, cwd=smoke_run).returncode == 0
    assert run_module(*INIT, cwd=smoke_run).returncode == 0
    quiet = run_module(*TRAIN, cwd=smoke_run, env={"LM_LOG_LEVEL": "WARNING"})
    loud = run_module(*TRAIN, cwd=smoke_run, env=DEBUG_JSON)
    assert quiet.returncode == loud.returncode == 0
    assert loud.stdout == quiet.stdout

    records = _records(loud.stderr)
    events = [r["event"] for r in records]
    assert events[0] == "run_start" and events[-1] == "run_done"
    steps = [r for r in records if r["event"] == "step_done"]
    assert [r["step"] for r in steps] == list(range(1, 9))


def test_refused_config_is_logged(smoke_run):
    cfg = smoke_run / "configs" / "bad.toml"
    cfg.write_text((smoke_run / "configs" / "smoke.toml").read_text().replace("lr = 0.01", "lr = -1"))
    r = run_module("lm.train", "--config", "configs/bad.toml", cwd=smoke_run, env=DEBUG_JSON)
    assert r.returncode != 0
    refused = [rec for rec in _records(r.stderr) if rec["event"] == "config_refused"]
    assert refused and "lr" in refused[0]["reason"]
