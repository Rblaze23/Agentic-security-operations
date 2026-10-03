from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest
from pydantic import ValidationError
from sqlalchemy import Engine

from secops.db.events_loader import event_id_for
from secops.schemas.tools import EventSearchInput, EventSummary, RelatedEventsInput
from secops.tools.events import MAX_LIMIT, MAX_WINDOW_HOURS, EventStoreTools


@pytest.fixture(scope="module")
def tools(event_engine: Engine) -> EventStoreTools:
    return EventStoreTools(event_engine)


def _window(flows: pd.DataFrame, day: str) -> tuple[datetime, datetime]:
    d = flows[flows["day"].astype(str) == day]
    return d["Timestamp"].min().to_pydatetime(), d["Timestamp"].max().to_pydatetime() + timedelta(
        seconds=1
    )


def test_search_events_input_bounds() -> None:
    t0 = datetime(2017, 7, 4, 12, 0, tzinfo=UTC)
    with pytest.raises(ValidationError, match="tz"):
        EventSearchInput(start=datetime(2017, 7, 4, 12, 0), end=t0 + timedelta(hours=1))
    with pytest.raises(ValidationError, match="after"):
        EventSearchInput(start=t0, end=t0)
    with pytest.raises(ValidationError, match="window"):
        EventSearchInput(start=t0, end=t0 + timedelta(hours=MAX_WINDOW_HOURS, seconds=1))
    with pytest.raises(ValidationError):
        EventSearchInput(start=t0, end=t0 + timedelta(hours=1), limit=MAX_LIMIT + 1)
    with pytest.raises(ValidationError):
        EventSearchInput(start=t0, end=t0 + timedelta(hours=1), source_ip="not-an-ip")
    with pytest.raises(ValidationError):
        EventSearchInput(start=t0, end=t0 + timedelta(hours=1), surprise=1)  # type: ignore[call-arg]


def test_search_by_window_and_filters_matches_pandas(
    tools: EventStoreTools, flows: pd.DataFrame
) -> None:
    start, end = _window(flows, "tuesday")
    res = tools.search_events(EventSearchInput(start=start, end=end, limit=MAX_LIMIT))
    expected = flows[(flows["day"].astype(str) == "tuesday")]
    assert res.total_matched == len(expected)
    assert res.truncated == (len(expected) > MAX_LIMIT)
    assert len(res.events) == min(len(expected), MAX_LIMIT)
    assert all(isinstance(e, EventSummary) for e in res.events)
    assert res.events == sorted(res.events, key=lambda e: (e.timestamp, e.event_id))

    res2 = tools.search_events(
        EventSearchInput(start=start, end=end, source_ip="172.16.0.1", destination_port=21)
    )
    exp2 = expected[(expected["Src IP"].astype(str) == "172.16.0.1") & (expected["Dst Port"] == 21)]
    assert res2.total_matched == len(exp2) > 0
    expected_ids = {
        event_id_for(str(d), int(i)) for d, i in zip(exp2["day"], exp2["id"], strict=True)
    }
    assert {e.event_id for e in res2.events} <= expected_ids
    assert all(e.destination_port == 21 and str(e.source_ip) == "172.16.0.1" for e in res2.events)
    assert res2.source == "event_store"


def test_search_returns_utc_timestamps_and_no_ground_truth(
    tools: EventStoreTools, flows: pd.DataFrame
) -> None:
    start, end = _window(flows, "monday")
    e = tools.search_events(EventSearchInput(start=start, end=end, limit=5)).events[0]
    assert e.timestamp.tzinfo is not None and e.timestamp.utcoffset() == timedelta(0)
    dumped = e.model_dump()
    for forbidden in ("label", "label_raw", "family", "is_attack", "features"):
        assert forbidden not in dumped


def test_related_events_unknown_and_self_exclusion(
    tools: EventStoreTools, flows: pd.DataFrame
) -> None:
    missing = tools.get_related_events(RelatedEventsInput(event_id=10**12))
    assert missing.status == "not_found" and missing.anchor is None

    attacker = flows[flows["Src IP"].astype(str) == "172.16.0.1"].sort_values("Timestamp")
    anchor_row = attacker.iloc[len(attacker) // 2]
    anchor_id = event_id_for(str(anchor_row["day"]), int(anchor_row["id"]))
    res = tools.get_related_events(RelatedEventsInput(event_id=anchor_id, window_minutes=30))
    assert res.status == "found" and res.anchor is not None and res.anchor.event_id == anchor_id
    lo = anchor_row["Timestamp"] - pd.Timedelta(minutes=30)
    hi = anchor_row["Timestamp"] + pd.Timedelta(minutes=30)
    same_src = flows[
        (flows["Src IP"].astype(str) == "172.16.0.1")
        & (flows["Timestamp"] >= lo)
        & (flows["Timestamp"] <= hi)
        & (flows["id"] != anchor_row["id"])
    ]
    assert res.same_source.count == len(same_src)
    assert res.same_source.distinct_destination_ports == same_src["Dst Port"].nunique()
    assert res.same_source.distinct_destination_ips == same_src["Dst IP"].astype(str).nunique()
    assert anchor_id not in res.same_source.sample_event_ids
    assert len(res.same_source.sample_event_ids) <= 20
    assert res.same_source.total_fwd_bytes == int(same_src["Total Length of Fwd Packet"].sum())
    top = res.same_source.top_destination_ports
    assert len(top) <= 10 and top == sorted(top, key=lambda p: (-p.count, p.port))
    same_dst = flows[
        (flows["Dst IP"].astype(str) == str(anchor_row["Dst IP"]))
        & (flows["Timestamp"] >= lo)
        & (flows["Timestamp"] <= hi)
        & (flows["id"] != anchor_row["id"])
    ]
    assert res.same_destination.count == len(same_dst)
    pair = (
        same_src[same_dst.index.intersection(same_src.index)]
        if False
        else same_src[same_src["Dst IP"].astype(str) == str(anchor_row["Dst IP"])]
    )
    assert res.same_pair.count == len(pair)


def test_related_events_window_bounds() -> None:
    with pytest.raises(ValidationError):
        RelatedEventsInput(event_id=1, window_minutes=31)
    with pytest.raises(ValidationError):
        RelatedEventsInput(event_id=1, window_minutes=0)
