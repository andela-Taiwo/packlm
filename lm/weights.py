"""Weights files: an .npz with one float64 array per parameter, keyed by its name."""

import os
import tempfile

import numpy as np
import torch


def save_weights(model, path):
    arrays = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    d = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".npz")
    with os.fdopen(fd, "wb") as f:
        np.savez(f, **arrays)
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)


def load_weights(model, path):
    with np.load(path) as z:
        state = {k: torch.from_numpy(z[k]) for k in z.files}
    model.load_state_dict(state, strict=True)
