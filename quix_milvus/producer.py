import quixstreams as qx
import json
import time
from datetime import datetime

def run_producer(topic: str = "simpsons", broker: str = "localhost:29092"):
    """Simple producer that sends Simpsons questions"""
    client = qx.KafkaStreamingClient(broker)
    producer = client.get_topic_producer(topic)
    
    questions = [
        "How has the animation style changed?",
        "What happened to key characters?",
        "How has the humor evolved?",
        "What are the biggest changes in recent seasons?",
    ]
    
    try:
        while True:
            for q in questions:
                data = {
                    "timestamp": datetime.now().isoformat(),
                    "question": q
                }
                producer.get_or_create_stream().parameters \
                    .buffer.add_timestamp(datetime.now()) \
                    .add_value("data", json.dumps(data)) \
                    .write()
                print(f"Sent: {q}")
                time.sleep(2)
    except KeyboardInterrupt:
        print("Stopping...")

if __name__ == "__main__":
    run_producer() 