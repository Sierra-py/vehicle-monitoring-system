"""
Per-plate inside/outside state machine.

Rules (agreed earlier): a plate can be 'inside' or 'outside'. An entry is only
valid from 'outside'. An exit is only valid from 'inside'. Any vehicle can
enter or exit independently at any time -- this is NOT a strict alternating
toggle, multiple different plates can transition concurrently with no
interaction between them.

Invalid transitions (e.g. an exit claimed for a plate that was never seen
entering, or two entries in a row with no exit between them) are NOT
auto-resolved either way -- per the earlier decision, these get flagged for
human review rather than the system guessing "we probably missed an event" or
silently rejecting the read. A missed prior detection and a genuine security
anomaly (tailgating, duplicate/cloned plate) look identical from inside a
single event, and only a human reviewing the surrounding context can tell
them apart.

A plate with no prior record defaults to 'outside' -- consistent with "you
have to enter before you can exit."
"""
from dataclasses import dataclass
from datetime import datetime


@dataclass
class StateTransitionResult:
    state_before: str          # 'inside' | 'outside'
    state_after: str           # 'inside' | 'outside' -- may equal state_before if invalid
    transition_valid: bool
    requires_review: bool
    review_reason: str | None


def apply_transition(current_state: str | None, claimed_direction: str) -> StateTransitionResult:
    """
    current_state: 'inside' | 'outside' | None (None = no prior record for this plate)
    claimed_direction: 'entry' | 'exit'

    Does NOT touch the database -- pure function, same reasoning as
    whitelist_matcher.match_plate: easy to test in isolation, the Stage 2
    consumer is responsible for reading current_state from vehicle_state and
    writing the result back.
    """
    state_before = current_state if current_state is not None else "outside"

    if claimed_direction == "entry":
        if state_before == "outside":
            return StateTransitionResult(state_before, "inside", True, False, None)
        # state_before == "inside": entry claimed for a plate already inside
        return StateTransitionResult(
            state_before, state_before, False, True,
            f"Entry claimed but plate is already 'inside' (no exit recorded since last entry). "
            f"Possible missed exit event, duplicate/cloned plate, or sensor error."
        )

    elif claimed_direction == "exit":
        if state_before == "inside":
            return StateTransitionResult(state_before, "outside", True, False, None)
        # state_before == "outside": exit claimed for a plate never seen entering
        return StateTransitionResult(
            state_before, state_before, False, True,
            f"Exit claimed but plate is 'outside' (no entry on record). "
            f"Possible missed entry event, tailgating, or misread plate matching an unrelated vehicle."
        )

    else:
        raise ValueError(f"claimed_direction must be 'entry' or 'exit', got {claimed_direction!r}")