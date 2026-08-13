"""
Pydantic schemas for the FastAPI layer -- separate from src/db_models.py
(the SQLAlchemy ORM models) on purpose. The API's response shape shouldn't
be forced to mirror the DB schema exactly (e.g. occupancy is a computed
aggregate with no table of its own), and keeping them separate means a
future DB column rename doesn't automatically become an API contract
change.
"""
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class OccupancyCount(BaseModel):
    whitelisted_inside: int
    non_whitelisted_inside: int
    total_inside: int


class VehicleEventOut(BaseModel):
    event_id: UUID
    camera_id: str
    event_timestamp: datetime
    image_path: str

    yolo_confidence: float | None
    claimed_direction: str | None

    plate_text: str | None
    ocr_confidence: float | None
    ocr_is_valid: bool | None
    ocr_reason: str | None

    whitelist_match: str | None
    match_type: str | None
    match_edit_distance: int | None

    vehicle_state_before: str | None
    vehicle_state_after: str | None
    transition_valid: bool | None

    requires_review: bool
    review_reason: str | None
    review_status: str
    reviewed_at: datetime | None
    reviewer_note: str | None

    created_at: datetime | None

    class Config:
        from_attributes = True


class VehicleStateOut(BaseModel):
    plate_text: str
    current_state: str
    is_whitelisted: bool
    last_event_id: UUID | None
    updated_at: datetime

    class Config:
        from_attributes = True


class ReviewAction(BaseModel):
    status: str  # "approved" | "rejected"
    reviewer_note: str | None = None