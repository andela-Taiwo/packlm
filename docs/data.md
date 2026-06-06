# Corpus and step shards

## Corpus

JSON lines, one document per line:

    {"step": 4, "tokens": [17, 9, 33, 5]}

`step` is the optimizer step the document belongs to (an integer from 1 to
999999; a corpus need not use every number) and `tokens` its token ids, without BOS or EOS (any id below `vocab` can turn up in a document, 0, 1
and 2 included). A step's documents are trained on together, and which step a
document lands in is decided upstream.

The step's token order is its documents in the order the corpus lists them,
each as `BOS tokens... EOS`, one token after another.

## Packing

Each document becomes `BOS tokens... EOS`. If that is longer than `ctx` (at
least 2) it is cut into consecutive windows of `ctx` tokens, the last one
shorter. A window is
a sequence in its own right: it is not continued from the window before it.

A step's windows are placed into rows of `ctx` tokens, several to a row where
they fit, with padding after the last window of a row. How windows are
arranged into rows is a layout choice and carries no meaning for training.
`lm.pack --layout` picks one and does what it names: `ffd` (the default) fills
rows first fit, longest window first; `ordered` first fit in the step's token
order; `shuffled` first fit in an order shuffled by `--seed` (the same seed
always shuffles the same way); `single` puts one window to a row. First fit
puts each window in the first row opened that still has room for it, after
the windows already there, and opens a row when none has. Rows are then added,
empty, until their number divides evenly by the number of micro-batches.

## Shards

`step_NNNNNN.npz`, one per step, named for the step's number in the corpus
(stored compressed or not; `np.load` reads either). For each micro-batch `m`
(0, 1, ...) it holds three int32 arrays of shape `[rows, ctx]`, so a
micro-batch can be read on its own:

- `tokens_m`: token ids, PAD (0) on padding.
- `segments_m`: which window of its row each position holds, numbered 1, 2,
  ... from the left of each row; 0 on padding. Positions of one window are
  contiguous and share a number; neighbouring windows never do. This, not the
  token id, is what says a position is padding.
- `order_m`: each position's index in the step's token order (0, 1, ... over
  the step's real tokens); -1 on padding.

Every step has at least one target (see `objective.md`).
