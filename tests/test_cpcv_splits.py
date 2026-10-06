"""Combinatorial purged cross-validation must not leak and must cover every bar.

The splitter is only worth having if its guarantees hold for every split, so
each property is checked over a spread of group counts, purges, embargoes and
window positions rather than one hand-picked case.
"""

from __future__ import annotations

from math import comb

import pytest

from honest_backtest.walkforward import CombinatorialPurgedSplitter

CASES = [
    # (n_groups, n_test_groups, purge, embargo, start, end)
    (6, 2, 0, 0, 0, 600),
    (6, 2, 10, 5, 0, 600),
    (6, 2, 25, 0, 100, 703),
    (5, 1, 7, 3, 0, 251),
    (8, 3, 12, 4, 50, 1050),
    (4, 2, 3, 2, 0, 41),
    (10, 4, 5, 5, 0, 1000),
]


def _test_bars(split) -> set[int]:
    return set(split.test_indices())


@pytest.mark.parametrize("n, k, purge, embargo, start, end", CASES)
def test_no_training_bar_is_inside_a_test_block(n, k, purge, embargo, start, end):
    """No split trains on a bar it tests."""
    splitter = CombinatorialPurgedSplitter(n, k, purge=purge, embargo=embargo)
    for split in splitter.split(end, start):
        assert not set(split.train_indices()) & _test_bars(split)


@pytest.mark.parametrize("n, k, purge, embargo, start, end", CASES)
def test_purge_and_embargo_are_honoured(n, k, purge, embargo, start, end):
    """No training bar within ``purge`` before, or ``purge + embargo`` after, a block."""
    splitter = CombinatorialPurgedSplitter(n, k, purge=purge, embargo=embargo)
    for split in splitter.split(end, start):
        train = split.train_indices()
        for b_start, b_end in split.test_blocks:
            for i in train:
                assert not b_start - purge <= i < b_start, (
                    f"bar {i} trains within the purge before block {b_start}"
                )
                assert not b_end <= i < b_end + purge + embargo, (
                    f"bar {i} trains within the purge or embargo after block {b_end}"
                )


@pytest.mark.parametrize("n, k, purge, embargo, start, end", CASES)
def test_only_the_purge_and_embargo_are_dropped(n, k, purge, embargo, start, end):
    """Every window bar is test, train, or within a purge or embargo zone."""
    splitter = CombinatorialPurgedSplitter(n, k, purge=purge, embargo=embargo)
    for split in splitter.split(end, start):
        train, test = set(split.train_indices()), _test_bars(split)
        assert len(train) == split.n_train and len(test) == split.n_test
        for i in set(range(start, end)) - train - test:
            assert any(
                b_start - purge <= i < b_end + purge + embargo
                for b_start, b_end in split.test_blocks
            ), f"bar {i} was dropped from training for no reason"
        assert all(start <= i < end for i in train | test)


@pytest.mark.parametrize("n, k, purge, embargo, start, end", CASES)
def test_every_bar_is_tested_exactly_c_n_minus_1_k_minus_1_times(
    n, k, purge, embargo, start, end
):
    """Each bar is tested once per path, never more, never less."""
    splitter = CombinatorialPurgedSplitter(n, k, purge=purge, embargo=embargo)
    splits = splitter.split(end, start)
    assert len(splits) == comb(n, k) == splitter.n_splits
    counts = dict.fromkeys(range(start, end), 0)
    for split in splits:
        for i in split.test_indices():
            counts[i] += 1
    assert set(counts.values()) == {comb(n - 1, k - 1)} == {splitter.n_paths}


@pytest.mark.parametrize("n, k, purge, embargo, start, end", CASES)
def test_paths_reconstruct_the_full_series(n, k, purge, embargo, start, end):
    """Each path walks the whole window once, using every test exactly once."""
    splitter = CombinatorialPurgedSplitter(n, k, purge=purge, embargo=embargo)
    splits = splitter.split(end, start)
    groups = splitter.groups(end, start)
    paths = splitter.paths(splits)
    assert len(paths) == splitter.n_paths
    used = set()
    for path in paths:
        assert [g for g, _ in path] == list(range(n))
        bars = [i for g, _ in path for i in range(*groups[g])]
        assert bars == list(range(start, end))
        for g, s in path:
            assert g in splits[s].test_groups, (
                "a path took a group its split did not test"
            )
            used.add((g, s))
    every_test = {(g, s.index) for s in splits for g in s.test_groups}
    assert used == every_test, "each (group, split) test belongs to exactly one path"


def test_groups_are_contiguous_and_near_equal():
    """Groups tile the window with sizes differing by at most one bar."""
    spans = CombinatorialPurgedSplitter(6, 2).groups(703, 100)
    assert spans[0][0] == 100 and spans[-1][1] == 703
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:], strict=False))
    sizes = [e - s for s, e in spans]
    assert max(sizes) - min(sizes) <= 1


def test_adjacent_test_groups_merge_into_one_block():
    """Nothing is purged between two bars that are both tested."""
    splitter = CombinatorialPurgedSplitter(6, 2, purge=5)
    split = next(s for s in splitter.split(600) if s.test_groups == (2, 3))
    assert split.test_blocks == ((200, 400),)
    assert split.train_segments == ((0, 195), (405, 600))


def test_a_hand_checked_split_with_embargo():
    """Purge on both sides, embargo only after, checked by hand."""
    splitter = CombinatorialPurgedSplitter(5, 2, purge=3, embargo=4)
    split = next(s for s in splitter.split(100) if s.test_groups == (0, 2))
    assert split.test_blocks == ((0, 20), (40, 60))
    # After block 1: purge 3 + embargo 4. Before block 2: purge 3.
    assert split.train_segments == ((27, 37), (67, 100))


def test_bad_arguments_are_refused():
    """Impossible settings fail with a reason instead of a silent split."""
    with pytest.raises(ValueError, match="n_groups"):
        CombinatorialPurgedSplitter(1, 1)
    with pytest.raises(ValueError, match="n_test_groups"):
        CombinatorialPurgedSplitter(4, 4)
    with pytest.raises(ValueError, match="non-negative"):
        CombinatorialPurgedSplitter(4, 2, purge=-1)
    with pytest.raises(ValueError, match="fewer bars"):
        CombinatorialPurgedSplitter(6, 2).split(5)
    with pytest.raises(ValueError, match="no training bars"):
        CombinatorialPurgedSplitter(3, 2, purge=50).split(90)
