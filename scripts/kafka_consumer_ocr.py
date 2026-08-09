"""
Stage 1 consumer: vehicle.detections -> OCR -> vehicle.ocr_results

Reads a lightweight detection event (image path + metadata), runs the actual
OCR pipeline on that image (same recognize_plate_with_validation used
elsewhere, no separate/simplified logic here), and publishes the result
downstream. Does NOT touch the whitelist or vehicle state -- that's Stage 2's
job, kept separate so each stage can be restarted, scaled, or debugged
independently (see the two-topic design discussion).

Loads both OCR models once at startup, not per-message -- same reasoning as
everywhere else models get loaded in this codebase (src/ocr.py, src/detection.py
docstrings): model loading is the expensive part, reuse the instance.

CONSUMER STARTUP RETRY: kafka-python has a known issue on Windows where
starting a consumer against a topic with no prior broker/group-coordinator
state yet (a true cold start, before any producer has ever talked to that
topic) crashes with "ValueError: Invalid file descriptor: -1" deep inside
the library's own socket cleanup path, rather than failing gracefully or
retrying itself. Confirmed via testing: running the producer first (which
warms up that topic/broker metadata) avoids it entirely; a genuinely cold
start reproduces it reliably. build_consumer() below retries the connection
a few times with a delay rather than requiring "always run the producer
first" as an unstated operational rule -- a consumer should be able to start
before a producer exists and just wait, so this papers over a real library
rough edge rather than working around a design flaw of ours.

Usage:
    python scripts/consumer_ocr.py
"""
import json
import time
from pathlib import Path

from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError, NoBrokersAvailable
from PIL import Image

from config.config import config
from src.ocr import load_ocr_model, load_finetuned_ocr_model, recognize_plate_with_validation


def build_consumer(retries: int = 5, delay_seconds: float = 3.0) -> KafkaConsumer:
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            return KafkaConsumer(
                config.kafka_topic_detections,
                bootstrap_servers=config.kafka_bootstrap_servers,
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                # Start from the beginning on first run (no committed offset
                # yet) -- lets you replay events already sitting on the
                # topic rather than only seeing new ones from this point
                # forward. Useful for a demo/dev loop.
                auto_offset_reset="earliest",
                group_id="ocr-consumer-group",
            )
        except (NoBrokersAvailable, KafkaError, ValueError) as e:
            last_error = e
            print(f"[startup] consumer connection attempt {attempt}/{retries} failed: {e!r}")
            time.sleep(delay_seconds)
    raise RuntimeError(
        f"Could not connect consumer to Kafka after {retries} attempts. "
        f"Last error: {last_error!r}. If this is a true cold start (topic has "
        f"never had a producer publish to it), try running "
        f"scripts/producer_simulator.py first."
    ) from last_error


def build_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=config.kafka_bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )


def run_consumer():
    print("Loading OCR models...")
    single_line_model = load_finetuned_ocr_model()
    two_line_model = load_ocr_model()

    print("Connecting consumer...")
    consumer = build_consumer()
    producer = build_producer()

    print(f"Listening on '{config.kafka_topic_detections}', "
          f"publishing results to '{config.kafka_topic_ocr_results}'\n")

    # The crash (ValueError: Invalid file descriptor: -1) happens inside the
    # FIRST poll cycle for a brand-new consumer group -- a known kafka-python
    # race on Windows between socket setup and the coordinator's initial
    # group-join, not during connection itself (build_consumer() already
    # succeeded by this point, both in the crashing and non-crashing runs).
    # Once the group exists on the broker (even from a crashed first attempt),
    # this doesn't recur -- confirmed: run 1 crashes, run 2 onward is clean,
    # regardless of whether the producer ran in between.
    #
    # Rather than requiring "just run it twice," retry the consumption loop
    # itself once on this specific error, so a fresh clone doesn't hit a
    # confusing crash on first-ever run.
    max_loop_restarts = 2
    for loop_attempt in range(1, max_loop_restarts + 1):
        try:
            _consume_loop(consumer, producer, single_line_model, two_line_model)
            break
        except ValueError as e:
            if "Invalid file descriptor" not in str(e) or loop_attempt == max_loop_restarts:
                raise
            print(f"[startup] known first-run coordinator race hit ({e!r}), "
                  f"reconnecting (attempt {loop_attempt}/{max_loop_restarts})...")
            consumer = build_consumer()


def _consume_loop(consumer, producer, single_line_model, two_line_model):
    for message in consumer:
        detection = message.value
        image_path = Path(detection["image_path"])

        if not image_path.exists():
            # Should not normally happen -- the producer just wrote this
            # file. Logged rather than silently skipped, since a missing
            # image means this event can never be OCR'd, which is worth
            # knowing about rather than losing quietly.
            print(f"[error] {detection['event_id'][:8]}: image not found at {image_path}, skipping")
            continue

        cropped_plate = Image.open(image_path)
        ocr_result = recognize_plate_with_validation(single_line_model, two_line_model, cropped_plate)

        enriched = {
            **detection,
            "plate_text": ocr_result["text"],
            "ocr_confidence": ocr_result["avg_char_confidence"],
            "ocr_is_valid": ocr_result["is_valid"],
            "ocr_reason": ocr_result["reason"],
        }

        producer.send(config.kafka_topic_ocr_results, value=enriched)
        producer.flush()

        status = ocr_result["text"] or "(no text)"
        print(f"[ocr] {detection['event_id'][:8]}: {status} "
              f"(valid={ocr_result['is_valid']}, conf={ocr_result['avg_char_confidence']:.2f})")


if __name__ == "__main__":
    run_consumer()