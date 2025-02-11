from quixstreams import Application
from sentence_transformers import SentenceTransformer
import numpy as np
from pymilvus import MilvusClient, DataType
from langchain_community.llms import Ollama
from quixstreams.kafka import Consumer
import json
from typing import List, Dict, Any
from datetime import datetime
import uuid

def setup_milvus():
    try:
        client = MilvusClient()
        print("Successfully connected to Milvus")
        
        collection_name = "chat_messages"
        
        if client.has_collection(collection_name):
            client.drop_collection(collection_name)
            
        schema = MilvusClient.create_schema(
            auto_id=False,
        )
        
        schema.add_field(field_name="chat_id", datatype=DataType.VARCHAR, max_length=100, is_primary=True)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=768)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="timestamp", datatype=DataType.VARCHAR, max_length=100)
        
        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            description="Messages with embeddings"
        )
        
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="IVF_FLAT",
            metric_type="L2",
            params={"nlist": 128}
        )
        
        client.create_index(
            collection_name=collection_name,
            index_params=index_params
        )

        client.load_collection(collection_name)        
        
        print(f"Collection '{collection_name}' created and loaded successfully")
        return client, collection_name
        
    except Exception as e:
        print(f"Failed to setup Milvus: {e}")
        raise

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
llm = Ollama(model="mistral-small")

def check_new_messages(consumer: Consumer) -> List[Dict[str, Any]]:
    """Check for any new messages in Kafka"""
    messages = []
    msg = consumer.poll(timeout=1.0)
    
    if msg and msg.value():
        try:
            data = json.loads(msg.value().decode())
            if not data.get("is_question", False):
                messages.append(data)
        except json.JSONDecodeError:
            pass
            
    return messages

def generate_rag_response(question: str, context: list[str]) -> str:
    prompt = f"""Use the following pieces of context to answer the question. If you don't know the answer, just say that you don't know.

Context:
{"\n".join(context)}

Question: {question}

Answer:"""
    
    return llm.invoke(prompt)

def process_message(message):
    # Check if it's a question
    is_question = message.get("is_question", False)
    text = message["text"]
    embedding = model.encode(text)
    
    # Add timestamp if not present
    if "timestamp" not in message:
        message["timestamp"] = datetime.now().isoformat()
        
    # Insert the new message if it's not a question
    if not is_question:
        client.insert(
            collection_name=collection_name,
            data={
                "chat_id": message["chat_id"],
                "text": text,
                "embedding": embedding.tolist(),
                "timestamp": message["timestamp"]
            }
        )            

    # Search for similar messages in Milvus
    results = client.search(
        collection_name=collection_name,
        data=[embedding.tolist()],
        limit=3,
        output_fields=["text"],
        metric_type="COSINE"
    )

    similar_texts = []
    similarities = []

    # Process Milvus results
    if results[0]:
        similar_texts = [hit["entity"]["text"] for hit in results[0]]
        similarities = [hit["distance"] for hit in results[0]]

    message["similar_texts"] = similar_texts
    message["similarities"] = similarities
    
    if is_question:
        message["rag_response"] = generate_rag_response(text, similar_texts) if similar_texts else "No relevant context found to answer the question."

    return message

if __name__ == "__main__":
    print("Starting the RAG demo...")
    client, collection_name = setup_milvus()
    
    try:
        # Setup Kafka consumer for checking new messages
        consumer = Consumer(
            broker_address="localhost:29092",
            consumer_group="rag-demo-consumer",
            auto_offset_reset="earliest"
        )
        consumer.subscribe(["messages"])
        
        print("\nReady to answer questions!")
        print("New messages will be processed before each question.")
        print("Type 'quit' to exit.")
        
        while True:
            try:
                new_messages = check_new_messages(consumer)
                if new_messages:
                    print(f"\nProcessing {len(new_messages)} new message(s)...")
                    for msg in new_messages:
                        process_message(msg)
                    print("Messages processed and stored in Milvus")
                
                question = input("\nEnter your question (or 'quit' to exit): ")
                if question.lower() == 'quit':
                    break
                    
                # Process the question
                result = process_message({
                    "chat_id": f"q_{uuid.uuid4()}",
                    "text": question,
                    "is_question": True,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Print the result
                print(f'\nQuestion: {result["text"]}')
                if result["similar_texts"]:
                    print("\nRelevant context:")
                    for text, sim in zip(result["similar_texts"], result["similarities"]):
                        print(f"- {text} (similarity: {sim:.3f})")
                print(f'\nAnswer: {result["rag_response"]}')
                
            except Exception as e:
                print(f"\nError: {e}")
                print("Continuing...")
                continue
            
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        consumer.close()
        client.close()