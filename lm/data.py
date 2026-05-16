"""Packed step shards, see docs/data.md."""

import os
import re

import numpy as np
import torch

_NAME = re.compile(r"^step_(\d{6})\.npz$")


def step_shards(root):
    """The step shard paths under root, in step order."""
    found = sorted((int(m.group(1)), n) for n in os.listdir(root) if (m := _NAME.match(n)))
    return [os.path.join(root, n) for _, n in found]


def micro_batches(path):
    """The step's micro-batches in order, each (tokens, segments, order) as int64 [rows, ctx].

    Only the micro-batch being returned is read from the shard."""
    with np.load(path) as z:
        n = sum(1 for k in z.files if k.startswith("tokens_"))
        for m in range(n):
            tokens, segments, order = (torch.from_numpy(z[f"{k}_{m}"].astype(np.int64))
                                       for k in ("tokens", "segments", "order"))
            if not tokens.shape == segments.shape == order.shape or tokens.dim() != 2:
                raise ValueError(f"{path}: micro-batch {m} arrays differ in shape")
            yield tokens, segments, order
