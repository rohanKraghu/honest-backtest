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
"""

from __future__ import annotations

from dataclasses import dataclass


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
