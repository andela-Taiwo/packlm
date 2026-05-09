"""Write a run's initial weights.

    python -m lm.init --config configs/smoke.toml --seed 0
"""

import argparse
import os

from lm.config import load_config
from lm.model import LM, init_weights
from lm.weights import save_weights


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    model = LM(cfg.model)
    init_weights(model, args.seed)
    os.makedirs(os.path.dirname(cfg.init) or ".", exist_ok=True)
    save_weights(model, cfg.init)
    print(f"wrote {cfg.init}")


if __name__ == "__main__":
    main()
