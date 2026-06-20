# lm

Small decoder-only LMs for the tokenizer and data-mix experiments. Everything
runs on CPU in float64, and a run gives the same weights every time it is
repeated.

## Commands

Run from the repository root (`lm` must be importable, which it is there):

    python -m lm.pack --docs data/smoke/corpus.jsonl --ctx 32 --micro-batches 2 --out data/smoke/shards
    python -m lm.init --config configs/smoke.toml --seed 0
    python -m lm.train --config configs/smoke.toml

`lm.pack` turns a corpus into step shards (format in `data.md`). `lm.init`
writes a run's initial weights to the config's `init` path. `lm.train` trains
from the weights in `init` over the step shards in `shards`, in the order of
their step numbers (which need not be contiguous), and writes the final
weights to `out`.

## Config

TOML, three tables. Paths in `[run]` are relative to the config file.

`[model]`

| key | | |
|---|---|---|
| `vocab` | required | token ids are `0 .. vocab-1`; 0, 1 and 2 are PAD, BOS and EOS; at least 4 |
| `d_model` | required | a multiple of `n_heads` |
| `ctx` | required | row length of the shards, at least 2; also the size of the learned position table |
| `n_heads` | 4 | attention heads; with RoPE the head width `d_model / n_heads` must be even |
| `positions` | `"rope"` | `"rope"` (rotary, applied in attention) or `"learned"` (a table added to the token embeddings) |
| `mixers` | required | one entry per block: `"attn"`, `"conv"` or `"scan"` |
| `conv_kernel` | 4 | how many positions back (including the current one) a `conv` mixer reads |
| `mlp` | `"dense"` | `"dense"` or `"moe"` |
| `d_ff` | 4 × `d_model` | hidden width of every MLP and expert |
| `experts` | 0 | number of experts, `moe` only (so an `moe` model has to set it) |
| `top_k` | 1 | experts each token is sent to, `moe` only; at most `experts` |
| `capacity_factor` | 0 | `moe` only: how many of the step's assignments each expert takes (see `objective.md`); 0 is no limit |

`[train]`

| key | | |
|---|---|---|
| `lr` | required | peak learning rate, above 0 |
| `warmup` | 0 | steps of linear warmup from 0 to `lr`; constant after |
| `weight_decay` | 0 | AdamW decoupled weight decay |
| `betas` | `[0.9, 0.95]` | two numbers, each at least 0 and below 1 |
| `eps` | 1e-8 | above 0 |
| `clip` | 0 | clip the step's gradient to this global norm; 0 is off |
| `balance_coef` | 0 | weight of the MoE load-balancing loss |
| `z_coef` | 0 | weight of the z-loss |

`[run]`: `shards`, `init`, `out` (paths) and optionally `steps`, which stops
after that many step shards (0 takes no step and writes the init weights as
`out`).

Integer keys (`vocab`, `d_model`, `ctx`, `n_heads`, `conv_kernel`, `d_ff`,
`experts`, `top_k`, `warmup`, `steps`) take TOML integers, at least 1 (at
least 0 for `experts`, `warmup` and `steps`). The other numbers take any
finite number, at least 0 unless the table says otherwise. Every key is
checked, including one the model does not use (a `capacity_factor` on a dense
model is accepted and has no effect). `mixers` is a non-empty list. Keys not
listed here are ignored.

`lm.train` refuses a config it cannot train (a table or a required key
missing, or a value outside what is described above): it exits non-zero and
leaves nothing behind. No `out` is written, no directory is created and no
file anywhere is created, changed or removed; a scratch file it removes again
before it exits leaves nothing.

`shards`, `init` and `out` are made absolute and normalised; directories
missing above `init` (for `lm.init`) and `out` are created.

## Steps and micro-batches

One shard is one optimizer step. A shard's rows come in micro-batches because
a whole step's activations don't fit on the nodes we train on, so how much
memory a run needs is set by the size of a micro-batch, not by how many of them
a step has. Measured as the peak resident memory of the trainer or of any
process it starts, whether or not it waits for that process, a step of any
number of micro-batches may need more than a step of just one micro-batch of
that size by at most the larger of 48 MiB and half of that micro-batch's
activations (its peak against the peak for a micro-batch of one row), plus two
bits for every assignment in the step (see `objective.md`) in each MoE block.
The shard's list of its arrays, which `np.load` reads when it opens one, is not
counted.

What a step optimizes is in `objective.md`. After the gradient of the step's
loss is taken it is clipped (if `clip` is set) and AdamW takes one step, as
`torch.optim.AdamW` computes it in float64, overflow included. Every
parameter takes that step, weight decay included, whether or not this step's
loss depends on it.

## Output

The weights file is an `.npz` with one float64 array per parameter, named as in
the model's `state_dict()`. `init` and `out` have the same format.

`lm.train` prints one line per step:

    step 3 | loss 3.71... | ce 3.65... | balance 1.02... | z 9.87... | targets 412 | gnorm 4.20...

The number after `step` counts the steps the run has taken, from 1. `loss`
is the step's objective, `ce`, `balance` and `z` its three terms before their
coefficients (`balance` summed over the MoE blocks, 0 for a dense
model), `targets` the number of tokens the step predicts, and `gnorm` the
global norm of the step's gradient before clipping. The numbers are printed
with `%.12e`. The last line is `wrote <out>`, with `out` as an absolute,
normalised path. Nothing else is printed on stdout.
