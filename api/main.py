"""
FastAPI backend -- read-only views into the pipeline's Postgres tables, plus
one write path: reviewing flagged events.

Deliberately does NOT talk to Kafka or trigger any pipeline component --
producer/consumer scripts run independently, this only reads (and for
review actions, writes) Postgres. See config/config.py for the shared
postgres_url / engine setup (src/db_session.py).

Endpoints:
    GET  /occupancy                 -> current counts (whitelisted / non-whitelisted / total inside)
    GET  /events                    -> recent event log, filterable
    GET  /events/{event_id}         -> single event detail
    GET  /review-queue              -> events with requires_review=True and review_status='pending'
    POST /events/{event_id}/review  -> approve/reject a flagged event (never touches vehicle_state)
    GET  /vehicles                  -> current vehicle_state table (who's inside right now, by plate)

Run:
    uvicorn api.main:app --reload
"""
from datetime import datetime, timezone
from uuid import UUID

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func, select

from config.config import config
from src.db_models import VehicleEvent, VehicleState
from src.db_session import get_session
from src.whitelist_matcher import load_whitelist

from api.schemas import OccupancyCount, ReviewAction, VehicleEventOut, VehicleStateOut

app = FastAPI(title="Vehicle Monitoring System API", version="0.1.0")

# Wide open for local/dev use with a Vite frontend on a different port.
# Tighten this to the actual deployed frontend origin before shipping to
# production -- "*" is fine for a portfolio demo, not for anything handling
# real gate data.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _whitelist_set() -> set[str]:
    # Loaded fresh per call rather than cached at startup -- this is a small
    # JSON file read, not a model load, so the cost of staying in sync with
    # config/whitelist.json edits outweighs the (negligible) cost of
    # re-reading it. If whitelist.json grows large or edits become frequent
    # enough for this to matter, revisit with an actual cache + invalidation.
    return set(load_whitelist(config.project_root / "config" / "whitelist.json"))


@app.get("/occupancy", response_model=OccupancyCount)
def get_occupancy():
    """
    Current count of vehicles inside, split by whitelist status.

    Reads vehicle_state directly (current_state='inside'), NOT vehicle_events
    -- vehicle_state is the table specifically maintained to answer "what's
    the state right now" without scanning full history. See db_models.py's
    own docstring reasoning for why the two tables are split this way.

    Whitelist status is determined by checking plate_text against the
    whitelist, not the event's match_type -- vehicle_state doesn't store
    match_type, and since it's already keyed off the whitelist plate for
    matched vehicles (see kafka_consumer_buisness_logic.py's state_key
    logic), a plate_text in vehicle_state is whitelisted if and only if it's
    literally in the whitelist file.

    NOTE (known, deliberate limitation -- see project decisions): counts for
    non-whitelisted vehicles can be inflated by OCR read variance, since
    each distinct OCR string for an unrecognized plate gets its own
    vehicle_state row. This was a conscious scope decision, not an oversight
    -- flagged here so the number isn't read as more precise than it is.
    """
    whitelist = _whitelist_set()
    with get_session() as session:
        inside_plates = session.scalars(
            select(VehicleState.plate_text).where(VehicleState.current_state == "inside")
        ).all()

    whitelisted_inside = sum(1 for p in inside_plates if p in whitelist)
    non_whitelisted_inside = len(inside_plates) - whitelisted_inside

    return OccupancyCount(
        whitelisted_inside=whitelisted_inside,
        non_whitelisted_inside=non_whitelisted_inside,
        total_inside=len(inside_plates),
    )


@app.get("/vehicles", response_model=list[VehicleStateOut])
def list_vehicles(state: str | None = Query(None, description="Filter by 'inside' or 'outside'")):
    """Full current vehicle_state table, optionally filtered by state."""
    if state is not None and state not in ("inside", "outside"):
        raise HTTPException(400, "state must be 'inside' or 'outside'")

    whitelist = _whitelist_set()
    with get_session() as session:
        stmt = select(VehicleState)
        if state is not None:
            stmt = stmt.where(VehicleState.current_state == state)
        rows = session.scalars(stmt.order_by(VehicleState.updated_at.desc())).all()

    return [
        VehicleStateOut(
            plate_text=r.plate_text,
            current_state=r.current_state,
            is_whitelisted=r.plate_text in whitelist,
            last_event_id=r.last_event_id,
            updated_at=r.updated_at,
        )
        for r in rows
    ]


@app.get("/events", response_model=list[VehicleEventOut])
def list_events(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    plate_text: str | None = None,
    requires_review: bool | None = None,
):
    """Recent event log, most recent first. Filterable by plate or review flag."""
    with get_session() as session:
        stmt = select(VehicleEvent)
        if plate_text is not None:
            stmt = stmt.where(VehicleEvent.plate_text == plate_text)
        if requires_review is not None:
            stmt = stmt.where(VehicleEvent.requires_review == requires_review)
        stmt = stmt.order_by(VehicleEvent.event_timestamp.desc()).limit(limit).offset(offset)
        events = session.scalars(stmt).all()
        return events  # from_attributes handles ORM -> Pydantic conversion


@app.get("/events/{event_id}", response_model=VehicleEventOut)
def get_event(event_id: UUID):
    with get_session() as session:
        event = session.get(VehicleEvent, event_id)
        if event is None:
            raise HTTPException(404, f"Event {event_id} not found")
        return event


@app.get("/review-queue", response_model=list[VehicleEventOut])
def get_review_queue(limit: int = Query(50, ge=1, le=500)):
    """
    Events still needing human attention: requires_review=True AND
    review_status='pending'. Once approved/rejected, an event drops out of
    this queue automatically (review_status changes), even though
    requires_review stays True forever as the pipeline's original,
    unmodified flag -- see review_status's own reasoning in db_models.py for
    why these are kept as two separate, non-overwriting facts.
    """
    with get_session() as session:
        stmt = (
            select(VehicleEvent)
            .where(VehicleEvent.requires_review == True)  # noqa: E712
            .where(VehicleEvent.review_status == "pending")
            .order_by(VehicleEvent.event_timestamp.asc())  # oldest first -- FIFO review queue
            .limit(limit)
        )
        return session.scalars(stmt).all()


@app.post("/events/{event_id}/review", response_model=VehicleEventOut)
def review_event(event_id: UUID, action: ReviewAction):
    """
    Approve or reject a flagged event. This is a record-keeping action ONLY
    -- it never touches vehicle_state, regardless of the decision. A human
    confirming "yes, this transition was legitimate" does not retroactively
    fix vehicle_state if the pipeline had already left it unchanged (invalid
    transitions never advance state -- see src/state_machine.py). This was a
    deliberate choice: current occupancy state should always reflect what
    the automated pipeline decided at the time, not a later manual
    correction, so review stays purely advisory/audit-trail rather than a
    second write path into live state.
    """
    if action.status not in ("approved", "rejected"):
        raise HTTPException(400, "status must be 'approved' or 'rejected'")

    with get_session() as session:
        event = session.get(VehicleEvent, event_id)
        if event is None:
            raise HTTPException(404, f"Event {event_id} not found")

        event.review_status = action.status
        event.reviewed_at = datetime.now(timezone.utc)
        event.reviewer_note = action.reviewer_note
        session.flush()
        session.refresh(event)
        return event