"""Event-store tools: search flows in a bounded window; aggregate a flow's neighbourhood."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session

from secops.db.models import Event
from secops.schemas.tools import (
    MAX_LIMIT,
    MAX_WINDOW_HOURS,
    SAMPLE_IDS,
    TOP_PORTS,
    Aggregate,
    EventSearchInput,
    EventSearchResult,
    EventSummary,
    PortCount,
    RelatedEvents,
    RelatedEventsInput,
)
from secops.tools.base import ToolSpec

__all__ = ["MAX_LIMIT", "MAX_WINDOW_HOURS", "EventStoreTools"]


def _to_us(dt: datetime) -> int:
    return int(dt.astimezone(UTC).timestamp() * 1_000_000)


def _from_us(us: int) -> datetime:
    return datetime.fromtimestamp(us / 1_000_000, tz=UTC)


def _summary(e: Event) -> EventSummary:
    return EventSummary(
        event_id=e.event_id,
        timestamp=_from_us(e.ts_us),
        source_ip=e.source_ip,  # type: ignore[arg-type]
        source_port=e.source_port,
        destination_ip=e.destination_ip,  # type: ignore[arg-type]
        destination_port=e.destination_port,
        protocol=e.protocol,
        duration_ms=e.flow_duration_us / 1000.0,
        fwd_packets=e.fwd_packets,
        bwd_packets=e.bwd_packets,
        fwd_bytes=e.fwd_bytes,
        bwd_bytes=e.bwd_bytes,
        syn_count=e.syn_count,
        fin_count=e.fin_count,
        rst_count=e.rst_count,
    )


def _empty_aggregate() -> Aggregate:
    return Aggregate(
        count=0,
        distinct_destination_ips=0,
        distinct_destination_ports=0,
        top_destination_ports=[],
        total_fwd_bytes=0,
        total_bwd_bytes=0,
        first_seen=None,
        last_seen=None,
        sample_event_ids=[],
    )


class EventStoreTools:
    """Both tools share one read-only engine."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    # ---- search_events -------------------------------------------------------------------
    def search_events(self, inp: EventSearchInput) -> EventSearchResult:
        conds: list[Any] = [Event.ts_us >= _to_us(inp.start), Event.ts_us < _to_us(inp.end)]
        if inp.source_ip is not None:
            conds.append(Event.source_ip == str(inp.source_ip))
        if inp.destination_ip is not None:
            conds.append(Event.destination_ip == str(inp.destination_ip))
        if inp.destination_port is not None:
            conds.append(Event.destination_port == inp.destination_port)
        if inp.protocol is not None:
            conds.append(Event.protocol == inp.protocol)
        with Session(self.engine) as conn:
            total = conn.scalar(select(func.count()).select_from(Event).where(*conds)) or 0
            rows = conn.execute(
                select(Event).where(*conds).order_by(Event.ts_us, Event.event_id).limit(inp.limit)
            ).scalars()
            events = [_summary(e) for e in rows]
        return EventSearchResult(
            events=events, total_matched=int(total), truncated=total > len(events)
        )

    # ---- get_related_events --------------------------------------------------------------
    def get_related_events(self, inp: RelatedEventsInput) -> RelatedEvents:
        with Session(self.engine) as conn:
            anchor = conn.execute(
                select(Event).where(Event.event_id == inp.event_id)
            ).scalar_one_or_none()
            if anchor is None:
                empty = _empty_aggregate()
                return RelatedEvents(
                    status="not_found",
                    anchor=None,
                    window_minutes=inp.window_minutes,
                    same_source=empty,
                    same_destination=empty.model_copy(),
                    same_pair=empty.model_copy(),
                )
            half = inp.window_minutes * 60 * 1_000_000
            lo, hi = anchor.ts_us - half, anchor.ts_us + half
            base = [Event.ts_us >= lo, Event.ts_us <= hi, Event.event_id != anchor.event_id]
            same_source = self._aggregate(conn, [*base, Event.source_ip == anchor.source_ip])
            same_destination = self._aggregate(
                conn, [*base, Event.destination_ip == anchor.destination_ip]
            )
            same_pair = self._aggregate(
                conn,
                [
                    *base,
                    Event.source_ip == anchor.source_ip,
                    Event.destination_ip == anchor.destination_ip,
                ],
            )
            return RelatedEvents(
                status="found",
                anchor=_summary(anchor),
                window_minutes=inp.window_minutes,
                same_source=same_source,
                same_destination=same_destination,
                same_pair=same_pair,
            )

    @staticmethod
    def _aggregate(conn: Any, conds: list[Any]) -> Aggregate:
        totals = select(
            func.count(),
            func.count(func.distinct(Event.destination_ip)),
            func.count(func.distinct(Event.destination_port)),
            func.coalesce(func.sum(Event.fwd_bytes), 0),
            func.coalesce(func.sum(Event.bwd_bytes), 0),
            func.min(Event.ts_us),
            func.max(Event.ts_us),
        ).where(*conds)
        count, d_ips, d_ports, fwd, bwd, first, last = conn.execute(totals).one()
        if not count:
            return _empty_aggregate()
        ports = conn.execute(
            select(Event.destination_port, func.count())
            .where(*conds)
            .group_by(Event.destination_port)
            .order_by(func.count().desc(), Event.destination_port)
            .limit(TOP_PORTS)
        ).all()
        sample = (
            conn.execute(
                select(Event.event_id)
                .where(*conds)
                .order_by(Event.ts_us, Event.event_id)
                .limit(SAMPLE_IDS)
            )
            .scalars()
            .all()
        )
        return Aggregate(
            count=int(count),
            distinct_destination_ips=int(d_ips),
            distinct_destination_ports=int(d_ports),
            top_destination_ports=[PortCount(port=int(p), count=int(c)) for p, c in ports],
            total_fwd_bytes=int(fwd),
            total_bwd_bytes=int(bwd),
            first_seen=_from_us(int(first)),
            last_seen=_from_us(int(last)),
            sample_event_ids=[int(i) for i in sample],
        )

    # ---- registry ------------------------------------------------------------------------
    def specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="search_events",
                description=(
                    "Search network flow records in the event store within a time window of at "
                    f"most {MAX_WINDOW_HOURS} hours, optionally filtered by source IP, "
                    "destination IP, destination port and protocol. Returns at most "
                    f"{MAX_LIMIT} flows ordered by time plus the total number matched. It cannot "
                    "see labels, payloads or hosts outside the dataset."
                ),
                input_model=EventSearchInput,
                output_model=EventSearchResult,
                run=self.search_events,
            ),
            ToolSpec(
                name="get_related_events",
                description=(
                    "Aggregate the flows around one event (by event_id) within ±window_minutes "
                    "(max 30): counts, distinct destinations and ports, top ports, byte totals and "
                    "sample event ids for flows from the same source, to the same destination, and "
                    "for the same source-destination pair. The anchor event is excluded from the "
                    "aggregates. Returns status=not_found for an unknown event_id. It cannot see "
                    "labels or payloads."
                ),
                input_model=RelatedEventsInput,
                output_model=RelatedEvents,
                run=self.get_related_events,
            ),
        ]
