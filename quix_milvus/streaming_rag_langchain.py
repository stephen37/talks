from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from quixstreams.kafka import Consumer
import json
import logging
import sys
import time

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

def setup_rag_components():
    """Initialize RAG components"""
    # Initialize components
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    llm = Ollama(model="mistral-small")
    
    # Set up empty Milvus vector store
    vector_store = Milvus.from_texts(
        texts=["Initial empty document"],  # Need at least one document to create collection
        embedding=embeddings,
        connection_args={"host": "localhost", "port": "19530"},
        collection_name="streaming_rag_demo",
        drop_old=True
    )
    
    # Create RAG prompt
    template = """Answer the question based only on the following context:

{context}

Question: {question}
Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Create RAG chain
    rag_chain = (
        {"context": vector_store.as_retriever(), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return vector_store, rag_chain

def process_stream(vector_store):
    """Process all messages from Kafka stream and add to RAG system"""
    consumer = Consumer(
        broker_address="localhost:29092",
        consumer_group="rag-consumer",
        auto_offset_reset="earliest"
    )
    consumer.subscribe(["messages"])
    
    print("\nProcessing messages from Kafka...")
    messages = []
    empty_polls = 0
    max_empty_polls = 5  # Wait for a few empty polls before giving up
    
    while empty_polls < max_empty_polls:
        msg = consumer.poll(timeout=1.0)  # Increased timeout
        if msg is None:
            empty_polls += 1
            continue
            
        if msg.error():
            print(f"Consumer error: {msg.error()}")
            continue
            
        try:
            value = json.loads(msg.value().decode())
            text = value["text"]
            print(f"\nReceived message: {text}")
            
            # Add text directly to vector store instead of collecting messages
            vector_store.add_texts([text])
            empty_polls = 0  # Reset counter when we get a message
            
        except Exception as e:
            print(f"Error processing message: {e}")
    
    consumer.close()
    print("\nFinished processing messages from Kafka")

def main():
    # Initialize RAG system
    vector_store, rag_chain = setup_rag_components()
    
    # Try initial query before streaming data
    print("\nInitial Query (before streaming):")
    question = "What do you know about artificial intelligence developments?"
    print(f"Question: {question}")
    print(f"Answer: {rag_chain.invoke(question)}\n")
    
    # Process streaming data
    process_stream(vector_store)
    
    # Query after receiving streaming data
    print("\nQuery after streaming:")
    question = "What do you know about artificial intelligence developments?"
    print(f"Question: {question}")
    print(f"Answer: {rag_chain.invoke(question)}\n")
    
    # Try another query about climate change
    print("\nQuery about climate change:")
    question = "What information do you have about climate change?"
    print(f"Question: {question}")
    print(f"Answer: {rag_chain.invoke(question)}\n")

if __name__ == "__main__":
    main() 