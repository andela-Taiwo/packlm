import numpy as np
import pytest

from lm.pack import BOS, EOS, pack_step, windows


def test_windows_add_bos_eos_and_split():
    assert windows([5, 6, 7], ctx=8) == [[BOS, 5, 6, 7, EOS]]
    assert windows([5, 6, 7], ctx=2) == [[BOS, 5], [6, 7], [EOS]]


@pytest.mark.parametrize("layout", ["ffd", "ordered", "shuffled", "single"])
def test_pack_step_covers_the_step_token_order_once(layout):
    docs = [[3, 4, 5, 6, 7, 8], [9], [10, 11, 12], [], [13, 14, 15, 16, 17, 18, 19, 20]]
    tokens, segments, order = pack_step(docs, ctx=6, micro=2, layout=layout, seed=1)
    assert tokens.shape == segments.shape == order.shape
    assert tokens.shape[0] == 2 and tokens.shape[2] == 6

    real = order[order >= 0]
    n_tokens = sum(len(d) + 2 for d in docs)
    assert sorted(real.tolist()) == list(range(n_tokens))
    assert np.all((segments == 0) == (order == -1))
    assert np.all(tokens[segments == 0] == 0)


def test_single_layout_puts_one_window_per_row():
    docs = [[3], [4], [5, 6]]
    _, segments, _ = pack_step(docs, ctx=8, micro=1, layout="single")
    assert segments.max() == 1


def test_rows_divide_into_micro_batches():
    tokens, _, _ = pack_step([[3, 4]] * 5, ctx=8, micro=3)
    assert tokens.shape[0] == 3


def test_shuffled_is_repeatable_for_a_seed():
    docs = [[i] * (i % 4 + 1) for i in range(3, 20)]
    a = pack_step(docs, ctx=8, micro=1, layout="shuffled", seed=7)
    b = pack_step(docs, ctx=8, micro=1, layout="shuffled", seed=7)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)
