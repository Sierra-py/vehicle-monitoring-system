"""
Stage 2 consumer: vehicle.ocr_results -> whitelist match + state machine -> Postgres

Reads the enriched detection+OCR event, matches the plate against the
whitelist (src/whitelist_matcher.py), looks up current state for that plate
and applies the entry/exit state machine (src/state_machine.py), then writes
one full row to vehicle_events and upserts vehicle_state. This is the piece
that closes the loop -- nothing before this stage touches the whitelist or
vehicle_state table.

requires_review is set True if EITHER the whitelist match was ambiguous OR
the state transition was invalid -- both flow into the same review queue,
per the schema design (single requires_review/review_reason pair on
vehicle_events).

Loads the whitelist once at startup, not per-message, same reasoning as
model loading elsewhere in this codebase.

Usage:
    python scripts/kafka_consumer_business_logic.py
"""
import json
import time
import uuid
from datetime import datetime
from pathlib import Path

from kafka import KafkaConsumer
from kafka.errors import KafkaError, NoBrokersAvailable
from sqlalchemy import select

from config.config import config
from src.db_models import VehicleEvent, VehicleState
from src.db_session import get_session
from src.state_machine import apply_transition
from src.whitelist_matcher import load_whitelist, match_plate


def build_consumer(retries: int = 5, delay_seconds: float = 3.0) -> KafkaConsumer:
    # Same cold-start retry as scripts/kafka_consumer_ocr.py -- see that
    # module's docstring for the full explanation of the underlying
    # kafka-python/Windows issue. Kept consistent across both consumers
    # rather than only fixing the one that was written first.
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            return KafkaConsumer(
                config.kafka_topic_ocr_results,
                bootstrap_servers=config.kafka_bootstrap_servers,
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                auto_offset_reset="earliest",
                group_id="business-logic-consumer-group",
            )
        except (NoBrokersAvailable, KafkaError, ValueError) as e:
            last_error = e
            print(f"[startup] consumer connection attempt {attempt}/{retries} failed: {e!r}")
            time.sleep(delay_seconds)
    raise RuntimeError(
        f"Could not connect consumer to Kafka after {retries} attempts. "
        f"Last error: {last_error!r}."
    ) from last_error


def get_current_state(session, plate_text: str | None) -> str | None:
    """None plate_text or no prior row -> None (state_machine.py treats
    None as 'outside')."""
    if not plate_text:
        return None
    row = session.get(VehicleState, plate_text)
    return row.current_state if row else None


def upsert_vehicle_state(session, plate_text: str, new_state: str, event_id: uuid.UUID):
    row = session.get(VehicleState, plate_text)
    if row is None:
        row = VehicleState(
            plate_text=plate_text,
            current_state=new_state,
            last_event_id=event_id,
        )
        session.add(row)
    else:
        row.current_state = new_state
        row.last_event_id = event_id
        row.updated_at = datetime.now()


def process_message(detection: dict, whitelist: list[str]):
    plate_text = detection.get("plate_text") or None
    ocr_is_valid = detection.get("ocr_is_valid", True)  # default True for older/malformed messages, don't silently block on a missing key

    # A low-confidence or too-short OCR read (ocr_is_valid=False, set by
    # validate_ocr_result() in src/ocr.py) is NOT trustworthy enough to act
    # on as an identity. Still log it to vehicle_events for the record, but
    # do not run it through whitelist matching or the state machine, and
    # never let it touch vehicle_state -- a garbage read should not flip
    # someone's inside/outside status or spawn a bogus tracked "vehicle".
    #
    # This does NOT catch a confidently WRONG read (OCR sure of itself but
    # misreading a character) -- confidence measures the model's certainty,
    # not correctness, and there's no cheap fix for that half of the
    # problem. This only removes the cheap, detectable failure mode.
    if not ocr_is_valid:
        match_result = None
        state_key = None
        transition = apply_transition(None, detection["claimed_direction"])
        # Force state_after back to state_before -- we are NOT committing
        # this transition (state_key is None so upsert_vehicle_state is
        # skipped below regardless), this just keeps the logged row
        # internally consistent rather than claiming a transition happened.
        transition.state_after = transition.state_before
        transition.transition_valid = False
        requires_review = True
        review_reason = (f"OCR read rejected before whitelist/state processing: "
                          f"{detection.get('ocr_reason', 'invalid OCR result')}.")
    else:
        match_result = match_plate(plate_text, whitelist) if plate_text else None

    with get_session() as session:
        # Kafka is at-least-once delivery -- if this consumer crashes or
        # restarts before its offset commits for a message, that message
        # gets redelivered on restart. event_id is the natural idempotency
        # key (generated once per detection by the producer), so check for
        # an existing row FIRST, before any whitelist/state-machine work
        # runs, not just before the final insert -- a replayed message
        # should be a clean no-op, not a wasted (or accidentally
        # state-mutating) reprocessing pass.
        existing = session.execute(
            select(VehicleEvent).where(VehicleEvent.event_id == uuid.UUID(detection["event_id"]))
        ).scalar_one_or_none()
        if existing is not None:
            print(f"[dedup] {detection['event_id'][:8]}: already processed, skipping replay")
            return existing, existing.requires_review

        if ocr_is_valid:
            # State machine keys off the WHITELIST plate (canonical identity),
            # not the raw OCR text -- a fuzzy-matched plate should accumulate
            # state under its real identity, not under every OCR variant that
            # happens to fuzzy-match to it. If there's no whitelist match at
            # all, fall back to tracking state under the raw OCR text so
            # unrecognized vehicles still get logged consistently.
            state_key = (match_result.whitelist_match if match_result and match_result.whitelist_match
                         else plate_text)
            current_state = get_current_state(session, state_key)

            transition = apply_transition(current_state, detection["claimed_direction"])

            requires_review = bool(
                (match_result.requires_review if match_result else False)
                or transition.requires_review
            )
            review_reasons = [
                r for r in [
                    match_result.review_reason if match_result else None,
                    transition.review_reason,
                ] if r
            ]
            review_reason = " | ".join(review_reasons) if review_reasons else None

        event = VehicleEvent(
            event_id=uuid.UUID(detection["event_id"]),
            camera_id=detection["camera_id"],
            event_timestamp=datetime.fromisoformat(detection["event_timestamp"]),
            image_path=detection["image_path"],
            yolo_confidence=detection.get("yolo_confidence"),
            claimed_direction=detection.get("claimed_direction"),
            plate_text=plate_text,
            ocr_confidence=detection.get("ocr_confidence"),
            ocr_is_valid=detection.get("ocr_is_valid"),
            ocr_reason=detection.get("ocr_reason"),
            whitelist_match=match_result.whitelist_match if match_result else None,
            match_type=match_result.match_type if match_result else "no_match",
            match_edit_distance=match_result.match_edit_distance if match_result else None,
            vehicle_state_before=transition.state_before,
            vehicle_state_after=transition.state_after,
            transition_valid=transition.transition_valid,
            requires_review=requires_review,
            review_reason=review_reason,
        )
        session.add(event)
        session.flush()  # so event.event_id is available for the FK below

        # Only track state under a real plate identity (whitelist match or
        # raw OCR text) -- if plate_text is empty/None entirely, there's no
        # key to upsert against, so vehicle_state is left untouched.
        if state_key:
            upsert_vehicle_state(session, state_key, transition.state_after, event.event_id)

        return event, requires_review


def run_consumer():
    print("Loading whitelist...")
    whitelist = load_whitelist(config.project_root / "config" / "whitelist.json")
    print(f"Loaded {len(whitelist)} whitelist entries.")

    print("Connecting consumer...")
    consumer = build_consumer()

    print(f"Listening on '{config.kafka_topic_ocr_results}', writing to Postgres.\n")

    # Same cold-start poll-cycle crash as scripts/kafka_consumer_ocr.py --
    # ValueError: Invalid file descriptor: -1, raised inside the FIRST poll
    # cycle for a brand-new consumer group, not during connection (which is
    # why build_consumer()'s own retry doesn't catch it). Wrapping the
    # message loop itself the same way Stage 1 does, so a fresh consumer
    # group on this topic doesn't hit an unhandled crash on first run.
    max_loop_restarts = 2
    for loop_attempt in range(1, max_loop_restarts + 1):
        try:
            _consume_loop(consumer, whitelist)
            break
        except ValueError as e:
            if "Invalid file descriptor" not in str(e) or loop_attempt == max_loop_restarts:
                raise
            print(f"[startup] known first-run coordinator race hit ({e!r}), "
                  f"reconnecting (attempt {loop_attempt}/{max_loop_restarts})...")
            consumer = build_consumer()


def _consume_loop(consumer, whitelist):
    for message in consumer:
        detection = message.value
        event, requires_review = process_message(detection, whitelist)

        flag = " [REVIEW]" if requires_review else ""
        print(f"[db] {detection['event_id'][:8]}: plate='{event.plate_text}' "
              f"match={event.match_type} "
              f"{event.vehicle_state_before}->{event.vehicle_state_after} "
              f"valid={event.transition_valid}{flag}")


if __name__ == "__main__":
    run_consumer()