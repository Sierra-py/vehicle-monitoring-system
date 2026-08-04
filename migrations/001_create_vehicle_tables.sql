-- 

CREATE TABLE IF NOT EXISTS vehicle_events (
    event_id UUID PRIMARY KEY,
    camera_id TEXT NOT NULL,
    event_timestamp TIMESTAMPTZ NOT NULL,
    image_path TEXT NOT NULL,

    -- detection stage (YOLO)
    yolo_confidence FLOAT,
    claimed_direction TEXT CHECK (claimed_direction IN ('entry', 'exit')),

    -- OCR stage
    plate_text TEXT,
    ocr_confidence FLOAT,
    ocr_is_vaild BOOLEAN,
    ocr_reason TEXT,

    -- whitelist matching stage
    whitelist_match TEXT,       -- the matched whitelist plate, or NULL if no match
    match_type TEXT CHECK (match_type IN ('exact', 'fuzzy', 'ambiguous', 'no_match')),
    match_edit_distance INT,

    -- vehicle state machine stage
    vehicle_state_before TEXT CHECK (vehicle_state_before IN ('inside', 'outside')),
    vehicle_state_after TEXT CHECK (vehicle_state_after IN ('inside', 'outside')),
    transition_valid BOOLEAN,

    -- review queue: covers BOTH ambiguous fuzzy matches and invalid state
    -- transitions with one flag + a human-readable reason, rather than two
    -- separate ad-hoc mechanisms for what's the same underlying concepts
    -- ("a human needs to look at this event before it's trusted").
    requires_review BOOLEAN NOT NULL DEFAULT FALSE,
    review_reason TEXT,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_vehicle_events_plate ON vehicle_events (plate_text);
CREATE INDEX IF NOT EXISTS idx_vehicle_events_timestamp ON vehicle_events (event_timestamp);
CREATE INDEX IF NOT EXISTS idx_vehicle_events_requires_review ON vehicle_events (requires_review) WHERE requires_review = TRUE;

-- Current state per plate. Kept separate from vehicle_events (which is an apeend-only log)
-- because "what's the current state of X" needs a fast, directly queryable answer, not a 
-- derived scan over full event history each time.
CREATE TABLE IF NOT EXISTS vehicle_state (
    plate_text TEXT PRIMARY KEY,
    current_state TEXT NOT NULL CHECK (current_state IN ('inside', 'outside')),
    last_event_id UUID REFERENCES vehicle_events(event_id),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);