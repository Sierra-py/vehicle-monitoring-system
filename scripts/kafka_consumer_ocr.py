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

Usage:
    python scripts/consumer_ocr.py
"""
import json
from pathlib import Path

from kafka import KafkaConsumer, KafkaProducer
from PIL import Image

from config.config import config
from src.ocr import load_ocr_model, load_finetuned_ocr_model, recognize_plate_with_validation


def build_consumer() -> KafkaConsumer:
    return KafkaConsumer(
        config.kafka_topic_detections,
        bootstrap_servers=config.kafka_bootstrap_servers,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        # Start from the beginning of the topic on first run (no committed
        # offset yet) -- lets you replay events already sitting on the topic
        # rather than only seeing new ones from this point forward. Useful
        # for a demo/dev loop where you want to reprocess what the producer
        # already published.
        auto_offset_reset="earliest",
        group_id="ocr-consumer-group",
    )


def build_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=config.kafka_bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )


def run_consumer():
    print("Loading OCR models...")
    single_line_model = load_finetuned_ocr_model()
    two_line_model = load_ocr_model()

    consumer = build_consumer()
    producer = build_producer()

    print(f"Listening on '{config.kafka_topic_detections}', "
          f"publishing results to '{config.kafka_topic_ocr_results}'\n")

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