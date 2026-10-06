"""Walk-forward validation: rolling, strictly ordered train/test splits.

A single in-sample fit tells you which parameter would have worked, which is
not the same as which parameter will work. Walk-forward answers the second
question by re-fitting on a training window and scoring only on the window
that comes *after* it, then rolling forward.

Two properties are enforced here and asserted in the test suite:

* every training window ends strictly before its test window begins, so no
  parameter is ever chosen using data from the period it is scored on;
* with the default step, test windows are contiguous and non-overlapping, so
  stitching them together produces an out-of-sample track record in which
  every bar is counted exactly once.

The second point matters more than it sounds. Overlapping test windows are a
quiet way to reuse the same lucky period several times.

Walk-forward produces exactly one out-of-sample path, so its Sharpe is one
draw from a distribution nobody gets to see. :class:`CombinatorialPurgedSplitter`
implements combinatorial purged cross-validation (Lopez de Prado, *Advances
in Financial Machine Learning*, ch. 12), which tests every bar several times
under different training sets and stitches the results into several complete
backtest paths, so the out-of-sample Sharpe comes with a spread.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from math import comb


@dataclass(frozen=True)
class Fold:
    """One train/test split.

    All bounds are half-open ``[start, end)`` indices into the bar series.

    Attributes:
        index: Zero-based fold number.
        train_start: First training bar.
        train_end: One past the last training bar; equals ``test_start``.
        test_start: First scored out-of-sample bar.
        test_end: One past the last scored bar.
        warmup: Bars replayed before ``test_start`` so that the strategy's
            trailing statistics are warm when scoring begins. These bars are
            simulated but never scored, and they are always drawn from the
            training region, so they leak nothing.
    """

    index: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    warmup: int = 0

    def __post_init__(self) -> None:
        if self.train_end != self.test_start:
            raise ValueError("train window must end exactly where the test window starts")
        if self.train_start >= self.train_end:
            raise ValueError("empty training window")
        if self.test_start >= self.test_end:
            raise ValueError("empty test window")
        if self.warmup < 0:
            raise ValueError("warmup must be non-negative")

    @property
    def n_train(self) -> int:
        """Number of training bars."""
        return self.train_end - self.train_start

    @property
    def n_test(self) -> int:
        """Number of scored test bars."""
        return self.test_end - self.test_start

    @property
    def eval_start(self) -> int:
        """First bar actually replayed for the test run, including warm-up."""
        return self.test_start - self.warmup

    def train_slice(self) -> slice:
        """Slice covering the training bars."""
        return slice(self.train_start, self.train_end)

    def eval_slice(self) -> slice:
        """Slice covering warm-up plus test bars, i.e. what the engine replays."""
        return slice(self.eval_start, self.test_end)

    def test_slice(self) -> slice:
        """Slice covering the scored test bars only."""
        return slice(self.test_start, self.test_end)


class WalkForwardSplitter:
    """Generates rolling or anchored walk-forward folds.

    Args:
        train_size: Bars in each training window. With ``anchored=True`` this
            is the size of the *first* window; later windows expand.
        test_size: Bars in each test window.
        step: Bars to advance between folds. Defaults to ``test_size``, which
            is the setting that makes test windows contiguous and
            non-overlapping. Any smaller value reuses data and is rejected
            unless ``allow_overlap`` is set.
        anchored: If ``True``, every training window starts at bar 0
            (expanding window). If ``False``, the training window rolls.
        warmup: Bars of pre-test history to replay unscored.
        allow_overlap: Permit ``step < test_size``. Off by default, because
            overlapping test windows double-count returns.
    """

    def __init__(
        self,
        train_size: int,
        test_size: int,
        step: int | None = None,
        anchored: bool = False,
        warmup: int = 0,
        allow_overlap: bool = False,
    ) -> None:
        if train_size < 1 or test_size < 1:
            raise ValueError("train_size and test_size must be >= 1")
        if warmup < 0:
            raise ValueError("warmup must be non-negative")
        if warmup > train_size:
            raise ValueError(
                "warmup cannot exceed train_size; warm-up bars must come from "
                "the training region or they would leak"
            )
        self.train_size = int(train_size)
        self.test_size = int(test_size)
        self.step = int(step) if step is not None else int(test_size)
        if self.step < 1:
            raise ValueError("step must be >= 1")
        if self.step < self.test_size and not allow_overlap:
            raise ValueError(
                f"step={self.step} < test_size={self.test_size} would make test "
                "windows overlap and double-count returns; pass "
                "allow_overlap=True only if you know why you want that"
            )
        self.anchored = bool(anchored)
        self.warmup = int(warmup)

    def split(self, n_samples: int) -> list[Fold]:
        """Produce the folds for a series of ``n_samples`` bars.

        A trailing partial test window is dropped rather than shortened, so
        every fold is scored over an equal-length window.

        Args:
            n_samples: Length of the bar series.

        Returns:
            Folds in chronological order; empty if the series is too short.

        Raises:
            ValueError: If ``n_samples`` is negative.
        """
        if n_samples < 0:
            raise ValueError("n_samples must be non-negative")

        folds: list[Fold] = []
        test_start = self.train_size
        index = 0
        while test_start + self.test_size <= n_samples:
            train_start = 0 if self.anchored else test_start - self.train_size
            folds.append(
                Fold(
                    index=index,
                    train_start=train_start,
                    train_end=test_start,
                    test_start=test_start,
                    test_end=test_start + self.test_size,
                    warmup=self.warmup,
                )
            )
            index += 1
            test_start += self.step
        return folds

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        kind = "anchored" if self.anchored else "rolling"
        return (
            f"WalkForwardSplitter({kind}, train={self.train_size}, "
            f"test={self.test_size}, step={self.step})"
        )


#: A half-open ``[start, end)`` range of bar indices.
Span = tuple[int, int]


@dataclass(frozen=True)
class CombinatorialSplit:
    """One train/test split of combinatorial purged cross-validation.

    All bounds are half-open ``[start, end)`` indices into the bar series.

    Attributes:
        index: Zero-based split number, in the order the combinations are
            generated.
        test_groups: Indices of the groups held out for testing, ascending.
        test_blocks: The test bars as contiguous blocks. Adjacent test
            groups merge into one block, since there is nothing to purge
            between two bars that are both tested.
        train_segments: The training bars as contiguous segments, after the
            test blocks, the purge and the embargo have been removed. Each
            segment is replayed on its own, so a training run never sees a
            bar outside its segment.
    """

    index: int
    test_groups: tuple[int, ...]
    test_blocks: tuple[Span, ...]
    train_segments: tuple[Span, ...]

    @property
    def n_train(self) -> int:
        """Number of training bars."""
        return sum(end - start for start, end in self.train_segments)

    @property
    def n_test(self) -> int:
        """Number of test bars."""
        return sum(end - start for start, end in self.test_blocks)

    def train_indices(self) -> list[int]:
        """Every training bar index, ascending."""
        return [i for start, end in self.train_segments for i in range(start, end)]

    def test_indices(self) -> list[int]:
        """Every test bar index, ascending."""
        return [i for start, end in self.test_blocks for i in range(start, end)]


class CombinatorialPurgedSplitter:
    """Combinatorial purged and embargoed cross-validation splits.

    The window ``[start, end)`` is cut into ``n_groups`` contiguous groups of
    near-equal size. Every combination of ``n_test_groups`` groups is held
    out once as the test set, giving ``C(n_groups, n_test_groups)`` splits.
    The training set of a split is every other bar of the window, less:

    * ``purge`` bars on **both** sides of each test block. A strategy with a
      lookback carries information across the boundary in both directions:
      training bars just after a block compute their signals from test-block
      prices, and positions taken just before a block are paid out inside it.
    * a further ``embargo`` bars **after** each test block, for serial
      correlation that outlasts the lookback.

    Every group is tested ``C(n_groups - 1, n_test_groups - 1)`` times, and
    :meth:`paths` assigns those tests to that many complete backtest paths,
    each of which covers the whole window exactly once.

    Args:
        n_groups: Number of contiguous groups, ``N``.
        n_test_groups: Groups held out per split, ``k``; ``1 <= k < N``.
        purge: Training bars removed on each side of every test block.
        embargo: Extra training bars removed after every test block.
    """

    def __init__(
        self, n_groups: int, n_test_groups: int, purge: int = 0, embargo: int = 0
    ) -> None:
        """Validate and store the split settings."""
        if n_groups < 2:
            raise ValueError("n_groups must be >= 2")
        if not 1 <= n_test_groups < n_groups:
            raise ValueError("n_test_groups must be between 1 and n_groups - 1")
        if purge < 0 or embargo < 0:
            raise ValueError("purge and embargo must be non-negative")
        self.n_groups = int(n_groups)
        self.n_test_groups = int(n_test_groups)
        self.purge = int(purge)
        self.embargo = int(embargo)

    @property
    def n_splits(self) -> int:
        """Number of train/test splits, ``C(N, k)``."""
        return comb(self.n_groups, self.n_test_groups)

    @property
    def n_paths(self) -> int:
        """Number of complete backtest paths, ``C(N - 1, k - 1)``.

        This is also the number of times every bar is tested.
        """
        return comb(self.n_groups - 1, self.n_test_groups - 1)

    def groups(self, end: int, start: int = 0) -> list[Span]:
        """Cut ``[start, end)`` into the contiguous groups.

        Sizes differ by at most one bar; the first groups take the remainder.

        Args:
            end: One past the last bar of the window.
            start: First bar of the window.

        Returns:
            ``n_groups`` spans in chronological order, covering the window.

        Raises:
            ValueError: If the window has fewer bars than groups.
        """
        n = end - start
        if start < 0 or n < self.n_groups:
            raise ValueError(
                f"window [{start}, {end}) has fewer bars than the {self.n_groups} groups"
            )
        size, extra = divmod(n, self.n_groups)
        spans: list[Span] = []
        cursor = start
        for g in range(self.n_groups):
            length = size + (1 if g < extra else 0)
            spans.append((cursor, cursor + length))
            cursor += length
        return spans

    def split(self, end: int, start: int = 0) -> list[CombinatorialSplit]:
        """Produce every split of the window ``[start, end)``.

        Args:
            end: One past the last bar of the window.
            start: First bar of the window. Bars before it are never trained
                on or tested, but remain available as warm-up history.

        Returns:
            ``C(N, k)`` splits, in lexicographic order of their test groups.

        Raises:
            ValueError: If the window is too short, or the purge and embargo
                leave some split with no training bars.
        """
        spans = self.groups(end, start)
        splits: list[CombinatorialSplit] = []
        for index, test_groups in enumerate(
            itertools.combinations(range(self.n_groups), self.n_test_groups)
        ):
            blocks = _merge([spans[g] for g in test_groups])
            removed = [
                (
                    max(start, b_start - self.purge),
                    min(end, b_end + self.purge + self.embargo),
                )
                for b_start, b_end in blocks
            ]
            train = _subtract((start, end), _merge(removed))
            if not train:
                raise ValueError(
                    f"split {test_groups} has no training bars left after a purge of "
                    f"{self.purge} and an embargo of {self.embargo}"
                )
            splits.append(
                CombinatorialSplit(
                    index=index,
                    test_groups=tuple(test_groups),
                    test_blocks=tuple(blocks),
                    train_segments=tuple(train),
                )
            )
        return splits

    def paths(self, splits: list[CombinatorialSplit]) -> list[list[tuple[int, int]]]:
        """Assign each split's test groups to complete backtest paths.

        Path ``p`` takes, for every group, the ``p``-th split (in split order)
        that tested it. Each path is then a full walk through the window in
        which every bar is out of sample, and every ``(group, split)`` test
        is used by exactly one path.

        Args:
            splits: The output of :meth:`split` for the same splitter.

        Returns:
            ``n_paths`` lists of ``(group, split_index)``, one entry per group
            in chronological order.
        """
        tested_by: list[list[int]] = [[] for _ in range(self.n_groups)]
        for s in splits:
            for g in s.test_groups:
                tested_by[g].append(s.index)
        if any(len(t) != self.n_paths for t in tested_by):
            raise ValueError("splits do not come from this splitter")
        return [
            [(g, tested_by[g][p]) for g in range(self.n_groups)]
            for p in range(self.n_paths)
        ]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        """Summarise the settings."""
        return (
            f"CombinatorialPurgedSplitter(N={self.n_groups}, k={self.n_test_groups}, "
            f"purge={self.purge}, embargo={self.embargo})"
        )


def _merge(spans: list[Span]) -> list[Span]:
    """Merge overlapping or touching spans; the result is sorted."""
    merged: list[Span] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _subtract(window: Span, holes: list[Span]) -> list[Span]:
    """The parts of ``window`` not covered by the sorted, disjoint ``holes``."""
    out: list[Span] = []
    cursor = window[0]
    for start, end in holes:
        if start > cursor:
            out.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < window[1]:
        out.append((cursor, window[1]))
    return out
