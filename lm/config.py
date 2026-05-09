"""Run configs: one TOML file per run, see docs/README.md."""

import math
import os
import tomllib
from dataclasses import dataclass

MIXERS = ("attn", "conv", "scan")


@dataclass(frozen=True)
class ModelConfig:
    vocab: int
    d_model: int
    n_heads: int
    ctx: int
    positions: str
    mixers: tuple
    conv_kernel: int
    mlp: str
    d_ff: int
    experts: int
    top_k: int
    capacity_factor: float


@dataclass(frozen=True)
class TrainConfig:
    lr: float
    warmup: int
    weight_decay: float
    beta1: float
    beta2: float
    eps: float
    clip: float
    balance_coef: float
    z_coef: float


@dataclass(frozen=True)
class RunConfig:
    model: ModelConfig
    train: TrainConfig
    shards: str
    init: str
    out: str
    steps: int | None


def _path(base, p):
    if not isinstance(p, str) or not p:
        raise ValueError(f"a path must be a non-empty string, not {p!r}")
    return os.path.normpath(p if os.path.isabs(p) else os.path.join(base, p))


def _int(table, key, default=None, low=None):
    """A TOML integer (not a float or a boolean), at least ``low``."""
    if key not in table:
        if default is None:
            raise KeyError(f"missing required key {key!r}")
        return default
    v = table[key]
    if isinstance(v, bool) or not isinstance(v, int):
        raise ValueError(f"{key} must be an integer, not {v!r}")
    if low is not None and v < low:
        raise ValueError(f"{key} must be at least {low}, not {v}")
    return v


def _float(table, key, default=None, low=None, positive=False, below=None):
    """A finite TOML number (integer or float, not a boolean) in range."""
    if key not in table:
        if default is None:
            raise KeyError(f"missing required key {key!r}")
        v = default
    else:
        v = table[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"{key} must be a number, not {v!r}")
        v = float(v)
    if not math.isfinite(v):
        raise ValueError(f"{key} must be finite, not {v}")
    if positive and not v > 0:
        raise ValueError(f"{key} must be positive, not {v}")
    if low is not None and v < low:
        raise ValueError(f"{key} must not be below {low}, not {v}")
    if below is not None and not v < below:
        raise ValueError(f"{key} must be below {below}, not {v}")
    return v


def _choice(table, key, default, options):
    v = table.get(key, default)
    if v not in options:
        raise ValueError(f"{key} must be one of {options}, not {v!r}")
    return v


def load_config(path):
    """Read and check a run config; raise (KeyError or ValueError) for anything README does not allow."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    for name in ("model", "train", "run"):
        if not isinstance(raw.get(name), dict):
            raise KeyError(f"missing table [{name}]")
    m, t, r = raw["model"], raw["train"], raw["run"]
    mixers = m.get("mixers")
    if mixers is None:
        raise KeyError("missing required key 'mixers'")
    if not isinstance(mixers, list) or not mixers or any(x not in MIXERS for x in mixers):
        raise ValueError(f"mixers must be a non-empty list of {MIXERS}, not {mixers!r}")
    d_model = _int(m, "d_model", low=1)
    model = ModelConfig(
        vocab=_int(m, "vocab", low=4),
        d_model=d_model,
        n_heads=_int(m, "n_heads", 4, low=1),
        ctx=_int(m, "ctx", low=2),
        positions=_choice(m, "positions", "rope", ("rope", "learned")),
        mixers=tuple(mixers),
        conv_kernel=_int(m, "conv_kernel", 4, low=1),
        mlp=_choice(m, "mlp", "dense", ("dense", "moe")),
        d_ff=_int(m, "d_ff", 4 * d_model, low=1),
        experts=_int(m, "experts", 0, low=0),
        top_k=_int(m, "top_k", 1, low=1),
        capacity_factor=_float(m, "capacity_factor", 0.0, low=0.0),
    )
    _check_model(model)
    betas = t.get("betas", [0.9, 0.95])
    if not isinstance(betas, list) or len(betas) != 2:
        raise ValueError(f"betas must be a list of two numbers, not {betas!r}")
    b = {"beta1": betas[0], "beta2": betas[1]}
    train = TrainConfig(
        lr=_float(t, "lr", positive=True),
        warmup=_int(t, "warmup", 0, low=0),
        weight_decay=_float(t, "weight_decay", 0.0, low=0.0),
        beta1=_float(b, "beta1", low=0.0, below=1.0),
        beta2=_float(b, "beta2", low=0.0, below=1.0),
        eps=_float(t, "eps", 1e-8, positive=True),
        clip=_float(t, "clip", 0.0, low=0.0),
        balance_coef=_float(t, "balance_coef", 0.0, low=0.0),
        z_coef=_float(t, "z_coef", 0.0, low=0.0),
    )
    steps = _int(r, "steps", low=0) if "steps" in r else None
    base = os.path.dirname(os.path.abspath(path))
    return RunConfig(
        model=model,
        train=train,
        shards=_path(base, r.get("shards")),
        init=_path(base, r.get("init")),
        out=_path(base, r.get("out")),
        steps=steps,
    )


def _check_model(m):
    """What the keys must satisfy together."""
    if m.d_model % m.n_heads:
        raise ValueError("d_model must be a multiple of n_heads")
    if m.positions == "rope" and (m.d_model // m.n_heads) % 2:
        raise ValueError("rope needs an even head width (d_model / n_heads)")
    if m.mlp == "moe" and m.top_k > m.experts:
        raise ValueError("moe needs top_k <= experts")
