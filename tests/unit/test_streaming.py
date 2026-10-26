import json

import pytest

from recsys.features.realtime import UserActivityAggregator
from recsys.features.store import InMemoryFeatureStore, UserRealtimeFeatures
from recsys.streaming.bus import InMemoryBroker
from recsys.streaming.events import EventDecodeError, RatingEvent, decode_event
from recsys.streaming.processor import StreamProcessor, run_consumer
from recsys.streaming.producer import iter_events, replay
from recsys.streaming.sinks import ParquetEventSink, read_events


def ev(user: int, item: int, ts: int, rating: float = 4.0) -> RatingEvent:
    return RatingEvent.create(user, item, rating, ts, produced_at=0.0)


# ------------------------------------------------------------------------ schema
def test_event_roundtrip_and_deterministic_id() -> None:
    event = ev(1, 2, 100)
    assert decode_event(event.to_bytes()) == event
    assert ev(1, 2, 100).event_id == event.event_id
    assert ev(1, 2, 101).event_id != event.event_id
    assert event.key == b"1"


def test_unknown_fields_are_ignored_for_forward_compatibility() -> None:
    payload = json.loads(ev(1, 2, 100).to_bytes()) | {"device": "tv", "new_optional": 1}
    assert decode_event(json.dumps(payload)).user_id == 1


@pytest.mark.parametrize(
    "raw",
    [
        b"not json",
        json.dumps({"event_id": "x", "user_id": 1, "item_id": 2, "rating": 9, "event_ts": 1}),
        json.dumps(
            {
                "schema_version": 2,
                "event_id": "x",
                "user_id": 1,
                "item_id": 2,
                "rating": 4,
                "event_ts": 1,
            }
        ),
        b"\xff\xfe",
    ],
)
def test_invalid_payloads_raise_decode_error(raw) -> None:
    with pytest.raises(EventDecodeError):
        decode_event(raw)


# ------------------------------------------------------------------------ state
def test_aggregator_keeps_last_n_in_event_time_order() -> None:
    store = InMemoryFeatureStore()
    agg = UserActivityAggregator(store, max_items=3)
    for item, ts in [(10, 100), (11, 300), (12, 200), (13, 400), (14, 50)]:  # out of order
        agg.apply(ev(1, item, ts))
    assert agg.flush() == [1]
    state = store.get_user_realtime(1)
    assert state.recent_items == [12, 11, 13]  # newest 3 by event time
    assert state.recent_ts == [200, 300, 400]
    assert state.event_count == 5 and state.last_event_ts == 400
    assert agg.flush() == []  # nothing dirty


def test_aggregator_rerating_replaces_item_and_bootstraps_from_store() -> None:
    store = InMemoryFeatureStore()
    store.write_user_realtime({7: UserRealtimeFeatures([1, 2], [3.0, 4.0], [10, 20], 2, 20)})
    agg = UserActivityAggregator(store, max_items=5)
    agg.apply(ev(7, 1, 30, rating=5.0))
    agg.flush()
    state = store.get_user_realtime(7)
    assert state.recent_items == [2, 1] and state.recent_ratings == [4.0, 5.0]
    assert state.event_count == 3


def test_realtime_features_after_cutoff() -> None:
    feats = UserRealtimeFeatures([1, 2, 3], [3.0, 4.0, 5.0], [10, 20, 30], 3, 30)
    assert feats.after(15) == ([2, 3], [4.0, 5.0])
    assert feats.after(100) == ([], [])


# ------------------------------------------------------------------------ processing
def test_processor_dedupes_routes_invalid_to_dlq_and_invalidates_cache(tmp_path) -> None:
    store = InMemoryFeatureStore()
    dlq: list[tuple[bytes, str]] = []
    invalidated: list[int] = []
    processor = StreamProcessor(
        UserActivityAggregator(store),
        sink=ParquetEventSink(tmp_path),
        dead_letter=lambda raw, err: dlq.append((raw, err)),
        on_users_updated=invalidated.extend,
    )
    batch = [ev(1, 5, 100).to_bytes(), ev(1, 5, 100).to_bytes(), b"{bad", ev(2, 6, 101).to_bytes()]
    result = processor.process_batch(batch)
    assert (result.received, result.processed, result.duplicates, result.invalid) == (4, 2, 1, 1)
    assert len(dlq) == 1 and sorted(invalidated) == [1, 2]
    assert store.get_user_realtime(1).recent_items == [5]
    processor.checkpoint()
    events = read_events(tmp_path)
    assert len(events) == 2 and set(events.user_id) == {1, 2}


def test_end_to_end_replay_through_in_memory_broker(tmp_path, prepared) -> None:
    broker = InMemoryBroker(partitions=3)
    test_window = prepared.split.test.head(300)
    sent = replay(broker, "ratings.v1", iter_events(test_window))
    assert sent == broker.size("ratings.v1") == 300
    store = InMemoryFeatureStore()
    processor = StreamProcessor(UserActivityAggregator(store, 20), ParquetEventSink(tmp_path))
    total = run_consumer(
        broker.consumer("ratings.v1", "g1"), processor, poll_timeout=0, max_idle_polls=1
    )
    assert total.processed == 300
    assert len(store) == test_window["user_id"].nunique()
    assert len(read_events(tmp_path)) == 300
    assert sum(broker.committed("ratings.v1", "g1")) == 300


def test_at_least_once_redelivery_is_idempotent(tmp_path) -> None:
    broker = InMemoryBroker(partitions=1)
    for i in range(10):
        broker.publish("t", b"1", ev(1, 100 + i, 1000 + i).to_bytes())
    store = InMemoryFeatureStore()
    processor = StreamProcessor(UserActivityAggregator(store, 50), ParquetEventSink(tmp_path))
    # consumer "crashes" after processing without committing offsets ...
    crashed = broker.consumer("t", "g")
    processor.process_batch(crashed.poll_batch(max_messages=10))
    # ... a new consumer in the same group re-reads everything from the last commit
    result = processor.process_batch(broker.consumer("t", "g").poll_batch(max_messages=10))
    assert result.duplicates == 10 and result.processed == 0
    assert store.get_user_realtime(1).recent_items == list(range(100, 110))


def test_sink_partitions_by_event_date(tmp_path) -> None:
    sink = ParquetEventSink(tmp_path, flush_events=2)
    sink.append(ev(1, 1, 1_500_000_000))
    assert not sink.should_flush()
    sink.append(ev(1, 2, 1_600_000_000))
    assert sink.should_flush()
    written = sink.flush()
    assert {p.parent.name for p in written} == {"date=2017-07-14", "date=2020-09-13"}
    assert read_events(tmp_path, since_ts=1_550_000_000)["item_id"].tolist() == [2]
    assert read_events(tmp_path / "missing").empty
