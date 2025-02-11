from quixstreams.kafka import Producer
import json
import time

# First, send some context messages
messages = [
    {"chat_id": "id1", "text": "The latest developments in artificial intelligence have revolutionized how we approach problem solving", "is_question": False},
    {"chat_id": "id2", "text": "Climate change poses significant challenges to global ecosystems and human societies", "is_question": False},
    {"chat_id": "id3", "text": "Quantum computing promises to transform cryptography and drug discovery", "is_question": False},
    {"chat_id": "id4", "text": "Sustainable energy solutions are crucial for addressing environmental concerns", "is_question": False},
]

# Then, send some questions
questions = [
    {"chat_id": "q1", "text": "What are the main impacts of climate change?", "is_question": True},
    {"chat_id": "q2", "text": "How is AI changing problem solving?", "is_question": True},
    {"chat_id": "q3", "text": "What are the applications of quantum computing?", "is_question": True},
]

def main():
    with Producer(
        broker_address="localhost:29092", 
        extra_config={
            "allow.auto.create.topics": "true",
        },
    ) as producer:
        # First send context messages
        print("\nSending context messages...")
        for message in messages:
            print(f'Sending: "{message["text"]}"')
            producer.produce(
                topic="messages",
                key=message["chat_id"].encode(),
                value=json.dumps(message).encode(),
            )
            time.sleep(1)  # Wait for processing
            
        # Then send questions
        print("\nSending questions...")
        for question in questions:
            print(f'Sending question: "{question["text"]}"')
            producer.produce(
                topic="messages",
                key=question["chat_id"].encode(),
                value=json.dumps(question).encode(),
            )
            time.sleep(2)  # Give more time for RAG processing

if __name__ == "__main__":
    main()