"""Train an LM on a directory of packed step shards.

    python -m lm.train --config configs/smoke.toml
"""

import argparse
import logging
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from lm.config import load_config
from lm.data import micro_batches, step_shards
from lm.log import event, get_logger
from lm.model import DTYPE, LM
from lm.weights import load_weights, save_weights

log = get_logger("lm.train")


def lr_at(cfg, step):
    if cfg.warmup and step <= cfg.warmup:
        return cfg.lr * step / cfg.warmup
    return cfg.lr


def micro_loss(model, tokens, segments, tcfg):
    """Loss terms for one micro-batch of packed rows."""
    logits, routes = model(tokens)
    # next-token targets; padding is never a target
    targets = tokens[:, 1:]
    keep = segments[:, 1:] != 0
    lg = logits[:, :-1]
    ce = F.cross_entropy(lg[keep], targets[keep], reduction="mean")
    z = torch.logsumexp(logits, -1).pow(2).mean()
    bal = torch.zeros((), dtype=DTYPE)
    for probs, top_i in routes:
        n_exp = probs.shape[-1]
        p = probs.reshape(-1, n_exp)
        chosen = F.one_hot(top_i.reshape(-1, top_i.shape[-1]), n_exp).sum(1).to(DTYPE)
        f = chosen.mean(0) / top_i.shape[-1]
        bal = bal + n_exp * (f * p.mean(0)).sum()
    loss = ce + tcfg.balance_coef * bal + tcfg.z_coef * z
    return loss, ce, bal, z, int(keep.sum())


def train(cfg):
    torch.set_num_threads(1)
    model = LM(cfg.model)
    load_weights(model, cfg.init)
    t = cfg.train
    opt = torch.optim.AdamW(model.parameters(), lr=t.lr, betas=(t.beta1, t.beta2), eps=t.eps,
                            weight_decay=t.weight_decay)
    shards = step_shards(cfg.shards)
    if cfg.steps is not None:
        shards = shards[: cfg.steps]
    event(log, logging.INFO, "run_start", shards=len(shards), mixers=list(cfg.model.mixers), mlp=cfg.model.mlp,
          experts=cfg.model.experts, top_k=cfg.model.top_k, capacity_factor=cfg.model.capacity_factor, out=cfg.out)
    started = time.perf_counter()
    for step, path in enumerate(shards, 1):
        step_started = time.perf_counter()
        mbs = list(micro_batches(path))
        n_micro = len(mbs)
        opt.zero_grad(set_to_none=True)
        tot = ce_t = bal_t = z_t = 0.0
        n_tgt = 0
        for tokens, segments, _ in mbs:
            loss, ce, bal, z, n = micro_loss(model, tokens, segments, t)
            (loss / n_micro).backward()
            tot += loss.item() / n_micro
            ce_t += ce.item() / n_micro
            bal_t += bal.item() / n_micro
            z_t += z.item() / n_micro
            n_tgt += n
        for p in model.parameters():  # every parameter is stepped every step
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        gnorm = math.sqrt(sum(float(p.grad.pow(2).sum()) for p in model.parameters() if p.grad is not None))
        if t.clip > 0 and gnorm > t.clip:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(t.clip / gnorm)
        for g in opt.param_groups:
            g["lr"] = lr_at(t, step)
        opt.step()
        print(f"step {step} | loss {tot:.12e} | ce {ce_t:.12e} | balance {bal_t:.12e} | z {z_t:.12e} "
              f"| targets {n_tgt} | gnorm {gnorm:.12e}", flush=True)
        event(log, logging.DEBUG, "step_done", step=step, shard=os.path.basename(path), micro_batches=n_micro,
              targets=n_tgt, loss=tot, gnorm=gnorm, seconds=round(time.perf_counter() - step_started, 4))
    os.makedirs(os.path.dirname(cfg.out) or ".", exist_ok=True)
    save_weights(model, cfg.out)
    event(log, logging.INFO, "run_done", steps=len(shards), out=cfg.out,
          seconds=round(time.perf_counter() - started, 3))
    print(f"wrote {cfg.out}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    try:
        cfg = load_config(args.config)
    except (KeyError, ValueError) as exc:
        event(log, logging.ERROR, "config_refused", config=args.config, reason=str(exc))
        raise
    train(cfg)


if __name__ == "__main__":
    np.seterr(all="raise")
    sys.exit(main())
