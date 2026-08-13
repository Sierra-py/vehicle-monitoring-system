-- Adds human-review resolution tracking to vehicle_events.
-- requires_review (existing) is set automatically by the pipeline.
-- review_status/reviewed_at/reviewer_note are set by a human via the API,
-- and are a separate, independent fact from requires_review -- approving or
-- rejecting an event never changes requires_review or vehicle_state
-- retroactively. This keeps "the pipeline flagged this" and "a human looked
-- at it" as two distinct, non-overwriting pieces of information.
--Usage(Windows/Powershell) : Get-Content .\migrations\002_add_review_status.sql | docker exec -i vms_postgres psql -U postgres -d vehicle_monitoring                                              

ALTER TABLE vehicle_events
    ADD COLUMN IF NOT EXISTS review_status TEXT NOT NULL DEFAULT 'pending',
    ADD COLUMN IF NOT EXISTS reviewed_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS reviewer_note TEXT;

ALTER TABLE vehicle_events
    ADD CONSTRAINT ck_review_status CHECK (review_status IN ('pending', 'approved', 'rejected'));

CREATE INDEX IF NOT EXISTS idx_vehicle_events_requires_review ON vehicle_events (requires_review) WHERE requires_review = TRUE;