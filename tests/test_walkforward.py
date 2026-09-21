"""Walk-forward splits must be strictly ordered and must not reuse data.

Two failure modes are being guarded against. The obvious one is training on
the period you then report. The quieter one is overlapping test windows,
which lets a single lucky stretch be counted several times and inflates the
apparent sample size.
"""

from __future__ import annotations

import pytest

from honest_backtest.walkforward import Fold, WalkForwardSplitter


def test_test_windows_are_non_overlapping():
    """The headline property: no bar is scored twice."""
    folds = WalkForwardSplitter(train_size=100, test_size=50).split(500)
    assert len(folds) > 1
    seen: set[int] = set()
    for fold in folds:
        window = set(range(fold.test_start, fold.test_end))
        assert not (window & seen), f"fold {fold.index} reuses bars already scored"
        seen |= window


def test_test_windows_are_contiguous_with_the_default_step():
    folds = WalkForwardSplitter(train_size=100, test_size=50).split(500)
    for a, b in zip(folds, folds[1:]):
        assert a.test_end == b.test_start


def test_training_always_ends_before_testing_begins():
    """No parameter may be chosen using data from the period it is scored on."""
    for anchored in (False, True):
        folds = WalkForwardSplitter(100, 50, anchored=anchored).split(500)
        assert folds
        for fold in folds:
            assert fold.train_end == fold.test_start
            assert fold.train_start < fold.train_end <= fold.test_start < fold.test_end


def test_every_index_stays_inside_the_series():
    n = 437
    folds = WalkForwardSplitter(120, 60).split(n)
    for fold in folds:
        assert 0 <= fold.train_start
        assert fold.test_end <= n
        assert fold.eval_start >= 0


def test_rolling_windows_have_constant_training_length():
    folds = WalkForwardSplitter(100, 50, anchored=False).split(500)
    assert {f.n_train for f in folds} == {100}


def test_anchored_windows_expand_from_zero():
    folds = WalkForwardSplitter(100, 50, anchored=True).split(500)
    assert all(f.train_start == 0 for f in folds)
    lengths = [f.n_train for f in folds]
    assert lengths == sorted(lengths)
    assert lengths[0] < lengths[-1]


def test_all_test_windows_have_equal_length():
    """A trailing partial window is dropped, not shortened."""
    folds = WalkForwardSplitter(100, 50).split(437)
    assert {f.n_test for f in folds} == {50}
    assert folds[-1].test_end <= 437


def test_overlapping_steps_are_refused_by_default():
    with pytest.raises(ValueError, match="overlap"):
        WalkForwardSplitter(train_size=100, test_size=50, step=25)


def test_overlap_is_possible_but_must_be_asked_for():
    splitter = WalkForwardSplitter(100, 50, step=25, allow_overlap=True)
    folds = splitter.split(500)
    windows = [set(range(f.test_start, f.test_end)) for f in folds]
    assert windows[0] & windows[1], "opting in should actually produce overlap"


def test_a_series_too_short_for_one_fold_yields_nothing():
    assert WalkForwardSplitter(400, 100).split(450) == []
    assert WalkForwardSplitter(400, 100).split(0) == []


def test_exactly_one_fold_fits():
    folds = WalkForwardSplitter(400, 100).split(500)
    assert len(folds) == 1
    assert folds[0].train_start == 0
    assert folds[0].test_end == 500


def test_warmup_bars_come_from_the_training_region():
    """Warm-up must never reach back past the start of training."""
    folds = WalkForwardSplitter(100, 50, warmup=30).split(500)
    for fold in folds:
        assert fold.eval_start == fold.test_start - 30
        assert fold.eval_start >= fold.train_start
    with pytest.raises(ValueError, match="warmup cannot exceed"):
        WalkForwardSplitter(20, 50, warmup=30)


def test_slices_describe_the_windows_they_claim_to():
    fold = WalkForwardSplitter(100, 50, warmup=10).split(300)[0]
    assert fold.train_slice() == slice(0, 100)
    assert fold.test_slice() == slice(100, 150)
    assert fold.eval_slice() == slice(90, 150)


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError):
        WalkForwardSplitter(0, 50)
    with pytest.raises(ValueError):
        WalkForwardSplitter(100, 0)
    with pytest.raises(ValueError):
        WalkForwardSplitter(100, 50, step=0)
    with pytest.raises(ValueError):
        WalkForwardSplitter(100, 50, warmup=-1)
    with pytest.raises(ValueError):
        WalkForwardSplitter(100, 50).split(-1)


def test_fold_rejects_an_inconsistent_construction():
    """A fold whose train window does not abut its test window is a bug."""
    with pytest.raises(ValueError, match="must end exactly"):
        Fold(index=0, train_start=0, train_end=90, test_start=100, test_end=150)
    with pytest.raises(ValueError, match="empty training window"):
        Fold(index=0, train_start=100, train_end=100, test_start=100, test_end=150)
    with pytest.raises(ValueError, match="empty test window"):
        Fold(index=0, train_start=0, train_end=100, test_start=100, test_end=100)


def test_fold_count_is_what_arithmetic_says():
    n, train, test = 1000, 200, 100
    folds = WalkForwardSplitter(train, test).split(n)
    assert len(folds) == (n - train) // test
