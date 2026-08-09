"""
Simulated camera producer.

Mimics what a real camera+detector process would do: watch for a vehicle,
and the MOMENT one is detected, publish a lightweight event to Kafka
immediately -- not wait for OCR or any downstream processing (that decoupling
is the whole point of using Kafka here, per the rush-hour/throughput
discussion: detection and full processing shouldn't be coupled).

What this does NOT do, deliberately, matching earlier design decisions:
  - Does NOT send the raw image bytes through Kafka. The image is written to
    disk once; only the file path goes in the Kafka message. Kafka is built
    for high-throughput small messages, not bulk binary transfer -- sending
    full frames through it is the wrong tool for the job (see prior
    discussion on Kafka message size limits and broker design intent).
  - Does NOT run OCR here. This producer's only job is "a vehicle was
    detected, here's where the image is" -- OCR is Stage 1 consumer's job,
    reading from vehicle.detections and publishing to vehicle.ocr_results.
  - Does NOT decide the whitelist match or state transition. That's Stage 2.

Direction (entry/exit) is randomly assigned per event for the simulation,
since there's no real second camera/gate context to infer it from -- this is
a documented simplification, not a claim that direction can be reliably
inferred from a single static image. A real deployment would get this from
which physical camera/gate triggered the event, not from the image content.

Usage:
    python scripts/producer_simulator.py [--interval-seconds 2] [--loop]
"""
import argparse
import json
import random
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from kafka.errors import NoBrokersAvailable, KafkaError
from kafka import KafkaProducer
from ultralytics import YOLO

from config.config import config
from src.detection import crop_detections

# Where simulated "captured frames" get copied to, so the Kafka message can
# reference a stable path independent of where the source image originally
# lived. Mirrors what a real camera pipeline would do: save the frame, then
# reference it by path.
CAPTURES_DIR = config.data_dir / "captures"


def build_producer(retries: int = 5, delay_seconds: float = 3.0) -> KafkaProducer:
    last_error = None
    for attempt in range(1, retries+1):
        try:
            return KafkaProducer(
                bootstrap_servers=config.kafka_bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
        except (NoBrokersAvailable, KafkaError, ValueError) as e:
            last_error = e
            print(f"[startup] Producer connection attempt {attempt}/{retries} failed: {e!r}")
            time.sleep(delay_seconds)
    raise RuntimeError(
        f"Could not connect Producer to Kafka after {retries} attempts. "
        f"Last error: {last_error!r}."
    ) from last_error



def iter_source_images():
    """
    Yields image paths from the simulation pool in random order, so repeated
    runs don't always process images in the same sequence -- closer to how
    vehicles would actually arrive in an unpredictable order.
    """
    img_dir = config.simulation_data_dir / "images"
    images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    random.shuffle(images)
    return images


def run_producer(interval_seconds: float, loop: bool):
    model = YOLO(str(config.yolo100ep_best_weights))
    producer = build_producer()
    CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Publishing to topic '{config.kafka_topic_detections}' "
          f"on {config.kafka_bootstrap_servers}")
    print(f"Reading source images from {config.simulation_data_dir / "images"}\n")

    events_published = 0
    events_skipped_no_detection = 0

    while True:
        for img_path in iter_source_images():
            # Run detection now (not in advance) -- mirrors a real camera
            # process, where the frame is only "captured" and evaluated at
            # the moment it's needed, not pre-processed in bulk beforehand.
            crops, _ = crop_detections(model, img_path)

            if not crops:
                # No plate-shaped region found above confidence/size
                # thresholds -- a real camera would simply not trigger an
                # event here either. Logged locally for visibility into how
                # often this happens during a demo run, not published to
                # Kafka (an event with no detection isn't a vehicle event).
                events_skipped_no_detection += 1
                print(f"[skip] {img_path.name}: no plate detected, no event published")
                time.sleep(interval_seconds)
                continue

            # Highest-confidence crop only, consistent with how the rest of
            # the pipeline already picks one candidate per image.
            best_crop, best_conf = max(crops, key=lambda c: c[1])

            event_id = str(uuid.uuid4())
            captured_path = CAPTURES_DIR / f"{event_id}.jpg"
            best_crop.save(captured_path)

            message = {
                "event_id": event_id,
                "camera_id": "sim_camera_1",
                "event_timestamp": datetime.now(timezone.utc).isoformat(),
                "image_path": str(captured_path),
                "yolo_confidence": best_conf,
                # Simulated -- a real deployment infers this from which
                # physical camera/gate triggered the event, not from image
                # content. See module docstring.
                "claimed_direction": random.choice(["entry", "exit"]),
            }

            producer.send(config.kafka_topic_detections, value=message)
            producer.flush()
            events_published += 1

            print(f"[event] {img_path.name} -> event_id={event_id[:8]} "
                  f"conf={best_conf:.2f} direction={message['claimed_direction']}")

            time.sleep(interval_seconds)

        print(f"\nPass complete. Published: {events_published}, "
              f"skipped (no detection): {events_skipped_no_detection}")

        if not loop:
            break
        print("Looping over simulation pool again...\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulated camera producer")
    parser.add_argument("--interval-seconds", type=float, default=2.0,
                         help="Delay between simulated frame arrivals")
    parser.add_argument("--loop", action="store_true",
                         help="Keep cycling through the simulation pool instead of stopping after one pass")
    args = parser.parse_args()

    run_producer(interval_seconds=args.interval_seconds, loop=args.loop)