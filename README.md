* ### To run frontend dashboard:
  *  npm --prefix frontend run dev
* ### To run the fastAPI backend: 
  * uvicorn api.main:app --reload
* ### To run kafka services: 
  * python -m scripts.kafka_producer_simulator --loop
  * python -m scripts.kafka_consumer_ocr
  * python -m scripts.kafka_consumer_buisness_logic
* ### Make sure the docker container is running

### The api for ultralytics plate dataset ndjson file:
    
