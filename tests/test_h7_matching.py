import numpy as np
import pytest

from backend.research.h7_matching import (
    EVENT_EXCLUSION_RADIUS_BARS,
    MATCH_WINDOW_BARS,
    MAX_CONTROLS,
    MIN_CONTROLS,
    VOLATILITY_LOOKBACK,
    select_h7_controls,
    volatility_buckets,
)


def test_volatility_uses_previous_1000_values_and_excludes_current_bar():
    atr = np.ones(VOLATILITY_LOOKBACK + 2, dtype=float)
    atr[-1] = 2.0

    ratio, bucket = volatility_buckets(atr)

    assert np.isnan(ratio[VOLATILITY_LOOKBACK - 1])
    assert bucket[VOLATILITY_LOOKBACK - 1] == -1
    assert ratio[VOLATILITY_LOOKBACK] == pytest.approx(1.0)
    assert bucket[VOLATILITY_LOOKBACK] == 2
    assert ratio[-1] == pytest.approx(2.0)
    assert bucket[-1] == 4


def test_volatility_bucket_boundaries_are_frozen():
    # Each current ATR is divided by a previous median of 1.0.
    previous = np.ones(VOLATILITY_LOOKBACK)
    current = np.array([0.79, 0.8, 0.999, 1.0, 1.249, 1.25, 1.599, 1.6])
    atr = np.concatenate([previous, current])
    _, bucket = volatility_buckets(atr)

    assert bucket[-len(current):].tolist() == [0, 1, 1, 2, 2, 3, 3, 4]


def test_invalid_atr_and_zero_median_remain_warmup():
    atr = np.ones(VOLATILITY_LOOKBACK + 2)
    atr[VOLATILITY_LOOKBACK] = np.nan
    atr[VOLATILITY_LOOKBACK + 1] = 0.0
    ratio, bucket = volatility_buckets(atr)

    assert np.isnan(ratio[-2:]).all()
    assert bucket[-2:].tolist() == [-1, -1]


def test_volatility_rejects_invalid_lookback_and_dimensions():
    with pytest.raises(ValueError, match="positive integer"):
        volatility_buckets([1.0, 1.0], lookback=0)
    with pytest.raises(ValueError, match="one-dimensional"):
        volatility_buckets([[1.0, 1.0]])


def make_event(**overrides):
    row = {
        "symbol": "EURUSD",
        "period": "development",
        "hour": 10,
        "volatility_bucket": 2,
        "direction": -1,
        "idx": 10_000,
    }
    row.update(overrides)
    return row


def candidate(idx, **overrides):
    row = make_event(idx=idx)
    row.update(overrides)
    return row


def test_selects_up_to_five_nearest_controls_with_fixed_window():
    event = make_event()
    candidates = [candidate(i) for i in (9_000, 9_700, 9_900, 10_100, 10_200, 11_440, 11_441)]
    used = set()

    chosen = select_h7_controls(event, candidates, [10_000], used)

    assert chosen == (9_900, 10_100, 10_200, 9_700, 9_000)
    assert len(chosen) == MAX_CONTROLS
    assert (event["symbol"], 10_200) in used
    assert 11_441 not in chosen  # no widening beyond +/-1,440


def test_fixed_window_edges_are_inclusive():
    event = make_event()
    candidates = [candidate(10_000 - MATCH_WINDOW_BARS),
                  candidate(10_000 + MATCH_WINDOW_BARS)]
    chosen = select_h7_controls(event, candidates, [10_000], set())
    assert chosen == (10_000 - MATCH_WINDOW_BARS, 10_000 + MATCH_WINDOW_BARS)


def test_excludes_controls_within_96_bars_of_any_qualifying_sweep():
    event = make_event()
    candidates = [candidate(9_800), candidate(9_900), candidate(10_104),
                  candidate(10_103), candidate(10_200), candidate(10_296),
                  candidate(10_297)]
    # The second sweep is not the event being matched, but its exclusion zone
    # must still be applied. Exactly 96 bars is excluded; 97 bars is eligible.
    chosen = select_h7_controls(
        event, candidates, [10_000, 10_200], set()
    )
    assert 10_104 not in chosen
    assert 10_200 not in chosen
    assert 10_296 not in chosen
    assert 10_103 in chosen
    assert 10_297 in chosen


def test_requires_exact_matching_keys():
    event = make_event()
    candidates = [
        candidate(9_900, period="validation"),
        candidate(9_901, hour=11),
        candidate(9_902, volatility_bucket=1),
        candidate(9_903, direction=1),
        candidate(9_904, symbol="GBPUSD"),
    ]
    assert select_h7_controls(event, candidates, [10_000], set()) == ()


def test_three_controls_is_minimum_and_four_or_five_are_allowed():
    event = make_event()
    used = set()
    three = [candidate(i) for i in (9_900, 10_100, 10_200)]
    chosen = select_h7_controls(event, three, [10_000], used)
    assert len(chosen) == MIN_CONTROLS
    assert len(used) == 3

    four = [candidate(i) for i in (9_800, 9_900, 10_100, 10_200)]
    assert len(select_h7_controls(event, four, [10_000], set())) == 4


def test_fewer_than_three_are_reportable_but_not_reserved():
    event = make_event()
    used = set()
    chosen = select_h7_controls(
        event, [candidate(9_900), candidate(10_100)], [10_000], used
    )
    assert chosen == (9_900, 10_100)
    assert used == set()


def test_control_is_never_reused_when_shared_used_set_is_passed():
    event = make_event()
    used = set()
    first = select_h7_controls(
        event, [candidate(9_900), candidate(10_100), candidate(10_200)],
        [10_000], used
    )
    second = select_h7_controls(
        event, [candidate(9_900), candidate(10_100), candidate(10_200)],
        [10_000], used
    )
    assert len(first) == 3
    assert second == ()


def test_exclusion_radius_boundary_is_inclusive():
    event = make_event()
    candidates = [candidate(10_000 + EVENT_EXCLUSION_RADIUS_BARS),
                  candidate(10_000 + EVENT_EXCLUSION_RADIUS_BARS + 1)]
    chosen = select_h7_controls(event, candidates, [10_000], set())
    assert chosen == (10_000 + EVENT_EXCLUSION_RADIUS_BARS + 1,)


def test_rejects_duplicate_candidate_indices_and_invalid_event_direction():
    event = make_event()
    with pytest.raises(ValueError, match="duplicate candidate"):
        select_h7_controls(
            event, [candidate(9_900), candidate(9_900)], [10_000], set()
        )
    with pytest.raises(ValueError, match="direction"):
        select_h7_controls(
            make_event(direction=0), [], [10_000], set()
        )



def test_accepts_numpy_integer_indices_from_dataframe_rows():
    event = make_event(idx=np.int64(10_000))
    candidates = [
        candidate(np.int64(9_900)),
        candidate(np.int64(10_100)),
        candidate(np.int64(10_200)),
    ]
    chosen = select_h7_controls(
        event, candidates, np.array([10_000], dtype=np.int64), set()
    )
    assert chosen == (9_900, 10_100, 10_200)



def test_requires_current_event_in_complete_qualifying_event_set():
    with pytest.raises(ValueError, match="must include the event"):
        select_h7_controls(
            make_event(), [candidate(9_900)], [9_800], set()
        )
