"""Pack a corpus into step shards, see docs/data.md.

    python -m lm.pack --docs corpus.jsonl --ctx 48 --micro-batches 2 --out data/run

Each corpus line is {"step": N, "tokens": [...]}: one document of the N-th
optimizer step, without BOS or EOS. A document becomes BOS + tokens + EOS and
is cut into windows of at most ctx tokens; the step's windows are placed into
rows of ctx tokens (first fit, longest first unless --layout says otherwise),
rows are added empty until their number divides evenly into micro-batches, and
the step is written as step_NNNNNN.npz, one set of arrays per micro-batch.
"""

import argparse
import json
import logging
import os
import random

import numpy as np

from lm.log import event, get_logger

log = get_logger("lm.pack")

PAD, BOS, EOS = 0, 1, 2


def windows(doc, ctx):
    seq = [BOS, *doc, EOS]
    return [seq[i:i + ctx] for i in range(0, len(seq), ctx)]


def save_step(path, tokens, segments, order):
    """Write a step: arrays [micro_batches, rows, ctx] stored one micro-batch per key."""
    arrays = {}
    for m in range(tokens.shape[0]):
        arrays[f"tokens_{m}"] = tokens[m]
        arrays[f"segments_{m}"] = segments[m]
        arrays[f"order_{m}"] = order[m]
    np.savez(path, **arrays)


def place(idx, lens, ctx, layout, rng):
    order = list(idx)
    if layout == "ffd":
        order.sort(key=lambda i: (-lens[i], i))
    elif layout == "shuffled":
        rng.shuffle(order)
    rows = []
    for i in order:
        if layout != "single":
            for row in rows:
                if sum(lens[x] for x in row) + lens[i] <= ctx:
                    row.append(i)
                    break
            else:
                rows.append([i])
        else:
            rows.append([i])
    return rows


def pack_step(docs, ctx, micro, layout="ffd", seed=0, extra_rows=0):
    rng = random.Random(seed)
    if ctx < 2:
        raise ValueError("ctx must be at least 2")
    wins = [w for d in docs for w in windows(d, ctx)]
    starts, at = [], 0
    for w in wins:  # each window's first index in the step's token order
        starts.append(at)
        at += len(w)
    idx = list(range(len(wins)))
    rows = place(idx, [len(w) for w in wins], ctx, layout, rng)
    rows += [[] for _ in range(extra_rows)]
    while len(rows) % micro:
        rows.append([])
    tokens = np.zeros((len(rows), ctx), np.int32)
    segments = np.zeros((len(rows), ctx), np.int32)
    order = np.full((len(rows), ctx), -1, np.int32)
    for r, row in enumerate(rows):
        at = 0
        for s, i in enumerate(row, 1):
            w = wins[i]
            tokens[r, at:at + len(w)] = w
            segments[r, at:at + len(w)] = s
            order[r, at:at + len(w)] = np.arange(starts[i], starts[i] + len(w))
            at += len(w)
    b = len(rows) // micro
    return tokens.reshape(micro, b, ctx), segments.reshape(micro, b, ctx), order.reshape(micro, b, ctx)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--docs", required=True)
    ap.add_argument("--ctx", type=int, required=True)
    ap.add_argument("--micro-batches", type=int, default=1)
    ap.add_argument("--layout", choices=("ffd", "ordered", "shuffled", "single"), default="ffd")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    steps = {}
    with open(args.docs) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                n = rec["step"]
                if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 999999:
                    raise ValueError(f"step must be an integer from 1 to 999999, not {n!r}")
                steps.setdefault(n, []).append([int(t) for t in rec["tokens"]])
    os.makedirs(args.out, exist_ok=True)
    for n in sorted(steps):
        packed = pack_step(steps[n], args.ctx, args.micro_batches, args.layout, args.seed + n)
        save_step(os.path.join(args.out, f"step_{n:06d}.npz"), *packed)
        event(log, logging.DEBUG, "step_packed", step=n, documents=len(steps[n]),
              rows=int(packed[0].shape[0] * packed[0].shape[1]))
    event(log, logging.INFO, "pack_done", steps=len(steps), out=args.out, layout=args.layout, ctx=args.ctx,
          micro_batches=args.micro_batches)
    print(f"packed {len(steps)} steps into {args.out}")


if __name__ == "__main__":
    main()
