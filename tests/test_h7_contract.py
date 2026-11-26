from datetime import datetime, timedelta, timezone

import pytest

from backend.research.h7_contract import H7Event, LAYER_NAMES, LayerResult

UTC = timezone.utc
BASE = datetime(2023, 1, 2, 8, 0, tzinfo=UTC)


def make_event(flags=(False, False, False, False), observed_offsets=(1, 2, 3, 4)):
    layers = tuple(
        LayerResult(name, flag, BASE + timedelta(minutes=15 * offset))
        for name, flag, offset in zip(LAYER_NAMES, flags, observed_offsets)
    )
    return H7Event(
        symbol="EURUSD",
        parent_time=BASE,
        sweep_time=BASE + timedelta(minutes=15),
        decision_time=BASE + timedelta(minutes=15 * 17),
        entry_time=BASE + timedelta(minutes=15 * 18),
        direction=1,
        layers=layers,
    )


def test_dose_is_sum_of_four_independent_binary_layers():
    assert make_event((False, False, False, False)).dose == 0
    assert make_event((True, False, False, False)).dose == 1
    assert make_event((True, True, False, False)).dose == 2
    assert make_event((True, True, True, False)).dose == 3
    assert make_event((True, True, True, True)).dose == 4


def test_layer_observed_after_decision_cutoff_is_rejected():
    with pytest.raises(ValueError, match="after the common decision cutoff"):
        make_event((True, False, False, False), observed_offsets=(1, 2, 3, 19))


def test_missing_layer_is_not_silently_treated_as_false():
    with pytest.raises(ValueError, match="exactly 4 layer results"):
        H7Event(
            symbol="EURUSD",
            parent_time=BASE,
            sweep_time=BASE + timedelta(minutes=15),
            decision_time=BASE + timedelta(minutes=255),
            entry_time=BASE + timedelta(minutes=270),
            direction=1,
            layers=(),
        )


def test_duplicate_or_unknown_layer_names_are_rejected():
    layers = list(make_event().layers)
    layers[3] = LayerResult(LAYER_NAMES[2], False, BASE + timedelta(minutes=60))
    with pytest.raises(ValueError, match="each of"):
        H7Event("EURUSD", BASE, BASE + timedelta(minutes=15),
                BASE + timedelta(minutes=255), BASE + timedelta(minutes=270), 1, tuple(layers))


def test_event_requires_utc_aware_timestamps():
    with pytest.raises(ValueError, match="timezone-aware"):
        LayerResult(LAYER_NAMES[0], True, datetime(2023, 1, 2, 8, 0))


def test_event_timeline_is_strictly_ordered():
    event = make_event()
    with pytest.raises(ValueError, match="entry_time must follow decision_time"):
        H7Event(event.symbol, event.parent_time, event.sweep_time,
                event.decision_time, event.decision_time, event.direction, event.layers)


def test_event_deduplication_key_is_symbol_parent_and_direction():
    event = make_event()
    assert event.event_key == ("EURUSD", BASE, 1)


def test_layer_map_uses_stable_layer_names():
    event = make_event((True, False, True, False))
    assert event.layer_map() == {
        "L1_liquidity": True,
        "L2_support_resistance": False,
        "L3_structure": True,
        "L4_fvg": False,
    }


def test_boolean_direction_is_rejected_even_though_bool_is_an_int_subclass():
    event = make_event()
    with pytest.raises(ValueError, match="direction must be integer"):
        H7Event(event.symbol, event.parent_time, event.sweep_time,
                event.decision_time, event.entry_time, True, event.layers)


def test_mutable_layer_container_is_rejected():
    event = make_event()
    with pytest.raises(TypeError, match="immutable tuple"):
        H7Event(event.symbol, event.parent_time, event.sweep_time,
                event.decision_time, event.entry_time, event.direction, list(event.layers))
