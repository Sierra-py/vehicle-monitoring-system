"""

"""

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class VehicleEvent(Base):
    __tablename__ = "vehicle_events"

    event_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    camera_id: Mapped[str] = mapped_column(String, nullable=False)
    event_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    image_path: Mapped[str] = mapped_column(String, nullable=False)

    # detection stage
    yolo_confidence: Mapped[float | None] = mapped_column(Float)
    claimed_direction: Mapped[str | None] = mapped_column(String)

    # OCR stage
    plate_text: Mapped[str | None] = mapped_column(String)
    ocr_confidence: Mapped[float | None] = mapped_column(Float)
    ocr_is_valid: Mapped[bool | None] = mapped_column(Boolean)
    ocr_reason : Mapped[str | None] = mapped_column(String)

    # whitelist matching stage
    whitelist_match: Mapped[str | None] = mapped_column(String)
    match_type: Mapped[str | None] = mapped_column(String)
    match_edit_distance: Mapped[int | None] = mapped_column(Integer)

    # vehicle state machine stage
    vehicle_state_before: Mapped[str | None] = mapped_column(String)
    vehicle_state_after: Mapped[str | None] = mapped_column(String)
    transition_valid: Mapped[bool | None] = mapped_column(Boolean)

    requires_review: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    review_reason: Mapped[str | None] = mapped_column(String)

    created_at: Mapped[str | None] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        CheckConstraint("claimed_direction IN ('entry', 'exit')", name="ck_claimed_direction"),
        CheckConstraint("match_type IN ('exact', 'fuzzy', 'ambiguous', 'no_match')", name="ck_match_type"),
        CheckConstraint("vehicle_state_before IN ('inside', 'outside')", name="ck_state_before"),
        CheckConstraint("vehicle_state_after IN ('inside', 'outside')", name="ck_state_after"),
        Index("idx_vehicle_events_plate", "plate_text"),
        Index("idx_vehicle_events_timestamp", "event_timestamp"),
    )


class VehicleState(Base):
    __tablename__ = "vehicle_state"

    plate_text: Mapped[str] = mapped_column(String, primary_key=True)
    current_state: Mapped[str] = mapped_column(String, nullable=False)
    last_event_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("vehicle_events.event_id")
    )
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        CheckConstraint("current_state IN ('inside', 'outside')", name="ck_current_state"),
    )