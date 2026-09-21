"""The headline claim, asserted as a test.

The whole point of the project is that Sharpe falls as assumptions are
removed. That claim should not live only in a README table that could drift
away from the code; it is pinned here.

A deliberately small configuration is used so the suite stays fast. The
full-size run lives in ``run_experiment.py``.
"""

from __future__ import annotations

import pytest

from honest_backtest.experiments import StudyConfig, run_study
from honest_backtest.synthetic import SyntheticConfig

SMALL = StudyConfig(
    synthetic=SyntheticConfig(n_bars=1260),
    seed=20260921,
    lookback_grid=(5, 20, 60),
    train_size=252,
    test_size=252,
)


@pytest.fixture(scope="module")
def study():
    """One small study, shared across the assertions below."""
    return run_study(SMALL)


def test_the_study_has_five_stages(study):
    assert len(study.stages) == 5
    assert [s.index for s in study.stages] == [1, 2, 3, 4, 5]


def test_only_the_last_stage_claims_to_be_honest(study):
    assert [s.honest for s in study.stages] == [False, False, False, False, True]


def test_only_the_first_stage_uses_look_ahead(study):
    assert study.stages[0].name.startswith("Naive")
    assert study.stages[0].assumption_removed.startswith("nothing")


def test_look_ahead_inflates_sharpe_enormously(study):
    """The single biggest lie a backtest tells."""
    naive, pit = study.stages[0].metrics.sharpe, study.stages[1].metrics.sharpe
    assert naive > 3.0, "the leak should look spectacular"
    assert pit < naive / 3.0, "removing it should destroy most of the apparent edge"


def test_adding_slippage_hurts(study):
    assert study.stages[2].metrics.sharpe < study.stages[1].metrics.sharpe


def test_adding_commissions_hurts(study):
    assert study.stages[3].metrics.sharpe <= study.stages[2].metrics.sharpe


def test_the_honest_stage_is_far_below_the_naive_one(study):
    assert study.honest_sharpe < study.stages[0].metrics.sharpe


def test_the_naive_result_is_implausible_and_that_is_the_tell(study):
    """Near-zero drawdown with a huge Sharpe is the signature of a leak."""
    naive = study.stages[0].metrics
    assert naive.sharpe > 3.0
    assert naive.max_drawdown > -0.15


def test_all_stages_are_scored_on_the_same_window(study):
    """Stages 1-4 must not be measured over a different period than stage 5.

    Otherwise the comparison confounds "how the parameter was chosen" with
    "which years were measured".
    """
    expected = study.scored_end - study.scored_start
    for stage in study.stages:
        assert stage.metrics.n_periods == expected, (
            f"stage {stage.index} is scored over {stage.metrics.n_periods} bars, "
            f"not {expected}"
        )


def test_walk_forward_refits_per_fold(study):
    assert len(study.stages[4].chosen_lookback) == len(study.folds)
    assert len(study.folds) >= 2


def test_the_single_fit_stages_use_one_parameter(study):
    for stage in study.stages[:4]:
        assert len(stage.chosen_lookback) == 1
        assert stage.chosen_lookback[0] in SMALL.lookback_grid


def test_the_event_loop_actually_ran(study):
    assert study.n_events["MARKET"] > 0
    assert study.n_events["FILL"] == study.n_events["ORDER"]


def test_the_oracle_ceiling_is_reported_and_modest(study):
    """Context for the honest number: the edge that was actually injected."""
    assert 0.3 < study.oracle_sharpe < 3.0
    assert study.stages[0].metrics.sharpe > study.oracle_sharpe, (
        "a leaky backtest should report more than the theoretical maximum -- "
        "that impossibility is the clearest evidence the number is fake"
    )


def test_the_study_is_reproducible():
    a = run_study(SMALL)
    b = run_study(SMALL)
    assert [s.metrics.sharpe for s in a.stages] == [s.metrics.sharpe for s in b.stages]


def test_a_series_too_short_for_a_fold_is_rejected():
    tiny = StudyConfig(synthetic=SyntheticConfig(n_bars=300), train_size=504, test_size=252)
    with pytest.raises(ValueError, match="too short"):
        run_study(tiny)
