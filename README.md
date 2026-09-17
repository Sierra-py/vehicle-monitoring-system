# Vehicle Monitoring System

An end-to-end machine learning system that monitors traffic entering and exiting a gate in real-time by identifying and extracting license plate numbers of vehicles. 

### Demo

[See on Youtube](https://youtu.be/v2cA06Qhw2Q)

### What it does

The system monitors traffic moving in and out from a gated facility and matches the vehicle from a whitelisted vehicles list and in realtime keep tracks of the total number of vehicle inside the facility.

It keeps track of the events and stores it in the database for auditing and reviewing if needed.

The system also tracks each vehicle's current state. If a vehicle attempts an entry while already marked inside, or an exit while already marked outside, the event is flagged for human review rather than silently accepted.

The system is not made for gate control because the OCR accuracy is not good enough to be trusted with gate control. It is made to be used along side the conventional gate security as a monitoring and audit layer.

### How it works

Detection : YOLO model takes an image and finds the lisence plate of the vehicle. Crops it and saves it to disk.
OCR : OCR model takes this cropped image of lisence plate and extracts the text of license plate.

Whitelist matching allows for OCR misreads within known confusion patterns (0/O, 1/I, 5/S, 8/B) — but only those specific substitutions, not arbitrary fuzzy matching, to avoid two different real plates cross-matching on a misread.

Invalid state transitions (e.g. an exit with no prior entry) are never auto-resolved — they're flagged for human review, since a missed detection and a genuine security anomaly look identical from a single event.

* Data Pipeline:
    1. Simulation script / edge device runs YOLO detection, then publishes the cropped plate + metadata to Kafka
    2. OCR consumer reads from Kafka, runs OCR, publishes enriched result
    3. business-logic matches the plate text with whitelisted vehicles and looks up the vehicle's current state from the database. It checks if the transition is valid or not and then publishes the event and make changes in db.

### Tech Stack

Detection/OCR : YOLO, fast-plate-ocr

Infrastructure : Kafka, Postgres, Docker

Backend : FastAPI, SQLAlchemy 

Frontend : React, Vite

### How to run locally (Windows/PowerShell):

- Clone this repo 
- `docker compose up -d` 
- run migrations scripts:
  - `psql -U postgres -d vehicle_monitoring -f migrations/001_create_vehicle_tables.sql`
  - `psql -U postgres -d vehicle_monitoring -f migrations/002_add_review_status.sql`
- `pip install -r requirements.txt` (install -dev too if you want to run training pipeline not required for inference)
- `npm install` for frontend
- run the scripts simultaneously:
  1. `python -m scripts.kafka_producer_simulator`
  2. `python -m scripts.kafka_consumer_ocr`
  3. `python -m scripts.kafka_consumer_buisness_logic`
  4. run api: `uvicorn api.main:app --reload`
  5. run frontend: `npm --prefix frontend run dev`
  6. Visit `http://localhost:5173/` for frontend and `http://localhost:8000/docs#/` for fastapi docs

* Note: Make sure you have the simulation dataset and model weights saved in correct location.

### Limitations

- Due to unavailability of proper data the OCR accuracy isn't very good. It caps at ~88% that too tested on small test set.
- The system isn't deployed yet, due to the requirement of paid deployment services. The system includes kafka, postgres db and ML inference, deploying all of that in one project is heavy and free platforms like Streamlit can't support it. 
- Current simulation doesn't truly mimic the actual real life situation. Currently the simulation feeds one image in regular time interval to kafka. But in real life a continuous live video will be fed and it will extract frames from it once a vehicle is detected. This also means it'll need a object tracker or a debounce logic to know when a vehicle has completed it's transition. This will be added in future with a more suitable video simulator.
- OCR confidence measures certainty, not correctness — a confidently wrong read isn't currently caught anywhere in the pipeline.
- Vehicle counts for non-whitelisted vehicles can be inflated — each distinct OCR misread of an unrecognized plate is currently tracked as a separate vehicle, since there's no whitelist identity to collapse it onto.

### What's next

The project isn't finished yet. But I need more and suitable data to fix current limitations.

1. More labeled OCR data — the current fine-tune is capped by dataset size, not model capacity.
2. Real or synthetic video data — needed to properly build and test debounce/frame-burst handling, which the current static-image simulator can't exercise meaningfully.
3. An ingestion script for real camera/edge-device input, replacing the simulator.

### Contact

Feel free to collaborate to give feedback/suggestions.

- Email : sys_admin.me@proton.me
- Twitter : [Sierra_ix](https://x.com/Sierra_ix)
- LinkedIn : [Shaswat Shukla](https://www.linkedin.com/in/shaswat-shukla-48636a417/)
