# Training objective

What one optimizer step minimises. This is the contract experiments are
compared under: two runs over the same corpus and config should train the same
model whatever the packer did with the layout.

## Sequences

A step is the set of sequences (windows, see `data.md`) in its shard. Rows,
micro-batches and padding are only how those sequences are laid out for the
compute; they are not part of the objective.

The model's output at a position of a sequence is what the model computes for
that sequence on its own: the sequence alone in a row, starting at position 0,
with nothing before it and nothing after it. The one exception is MoE
capacity, below, which is decided over the whole step.

## MoE routing and capacity

In an MoE block each token's router picks its `top_k` experts (the largest
softmax probabilities), and the token's output is the sum of those experts'
outputs weighted by their probabilities renormalised over the `top_k` picked.
A token's pick of one expert is an assignment.

With `capacity_factor` `c > 0`, each expert of a block takes at most
`C = ceil(c * top_k * R / E)` of the step's assignments to it (`R` and `E` as
below): the first `C` in the step's token order (`data.md`). An assignment
past that is dropped: that expert contributes nothing to that token, and the
weights of the token's other experts stay as they were.

## Terms

In a sequence `x_0 ... x_{n-1}`, every position `i < n-1` has a target,
`x_{i+1}`. A sequence of one token has none. `T` is the number of targets in
the step and `R` the number of tokens in the step's sequences (every position
of every sequence, padding excluded).

- `ce`: the summed cross-entropy of every target in the step, divided by `T`.
- `z`: over the same positions, the summed square of `logsumexp` of the
  output logits, divided by `T`.
- `balance`: for each MoE block, take the router's softmax probabilities `p_t`
  and the `top_k` experts it chooses for each of the step's `R` tokens
  (whether capacity drops them or not). With `E` experts, `f_e` is the number
  of tokens that chose expert `e` divided by `top_k * R`, and `P_e` is the mean
  of `p_t[e]` over the `R` tokens. The block's term is `E * sum_e f_e * P_e`, with `f_e` treated as a constant (it
  carries no gradient). `balance` is the sum of the blocks' terms, 0 for a dense
  model.

The step's loss is

    loss = ce + balance_coef * balance + z_coef * z

where a term whose coefficient is 0 is left out (so an infinite `z` with
`z_coef = 0` leaves the loss at `ce`), and the step's gradient is the gradient
of that loss with respect to the weights.
