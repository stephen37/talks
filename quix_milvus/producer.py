from quixstreams.kafka import Producer, Consumer
import json
import time

# Context messages to be stored in Milvus
messages = [
    {"chat_id": "id1", "text": "The latest developments in artificial intelligence have revolutionized how we approach problem solving"},
    {"chat_id": "id2", "text": "Climate change poses significant challenges to global ecosystems and human societies"},
    {"chat_id": "id3", "text": "Quantum computing promises to transform cryptography and drug discovery"},
    {"chat_id": "id4", "text": "Sustainable energy solutions are crucial for addressing environmental concerns"},
]

def cleanup_topic():
    """Delete and recreate the topic to ensure clean state"""
    print("\nCleaning up Kafka topic...")
    
    consumer = Consumer(
        broker_address="localhost:29092",
        consumer_group="cleanup-consumer",
        auto_offset_reset="earliest"
    )
    
    try:
        # Try to subscribe - this will fail if topic doesn't exist
        consumer.subscribe(["messages"])
        msg = consumer.poll(timeout=1.0)
        if msg:
            print("Found existing messages, recreating topic...")
            consumer.close()
            
            # Create producer with admin rights to delete topic
            with Producer(
                broker_address="localhost:29092",
                extra_config={
                    "allow.auto.create.topics": "true",
                },
            ) as producer:
                producer.delete_topics(["messages"])
                time.sleep(2)  # Wait for deletion
                
    except Exception as e:
        print(f"Topic doesn't exist yet: {e}")
    finally:
        consumer.close()

def main():
    # Clean up topic first
    cleanup_topic()
    
    with Producer(
        broker_address="localhost:29092", 
        extra_config={
            "allow.auto.create.topics": "true",
        },
    ) as producer:
        print("\nSending messages to be stored in Milvus...")
        for message in messages:
            print(f'Sending: "{message["text"]}"')
            producer.produce(
                topic="messages",
                key=message["chat_id"].encode(),
                value=json.dumps(message).encode(),
            )
            time.sleep(1)  # Wait for processing
        print("\nAll messages sent!")

if __name__ == "__main__":
    main()