import quixstreams as qx
import json
from datetime import datetime

# Import RAG components
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_milvus import Milvus
from transformers import TextStreamer
from unsloth import FastLanguageModel

def setup_rag():
    """Setup RAG pipeline"""
    # Load model
    model_name = "unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=4048,
        load_in_4bit=True
    )
    FastLanguageModel.for_inference(model)
    
    # Setup embeddings and vector store
    embeddings = SentenceTransformerEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    vectorstore = Milvus.from_documents(
        documents=[],  # We'll add documents later
        embedding=embeddings,
        connection_args={"uri": "./milvus_demo.db"},
    )
    
    # Setup RAG chain
    template = """Use the following pieces of context to answer the question.
    If you don't know the answer, just say that you don't know.
    
    Context: {context}
    Question: {question}
    
    Answer:"""
    
    prompt = PromptTemplate.from_template(template)
    retriever = vectorstore.as_retriever()
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    
    return rag_chain, retriever

def run_consumer(topic: str = "simpsons", broker: str = "localhost:29092"):
    """Simple consumer that processes questions with RAG"""
    # Setup RAG pipeline
    rag_chain, retriever = setup_rag()
    
    # Setup Quix consumer
    client = qx.KafkaStreamingClient(broker)
    consumer = client.get_topic_consumer(topic)
    
    def on_data(stream_consumer: qx.StreamConsumer, data: dict):
        try:
            data = json.loads(data["data"])  # Unpack the data
            question = data["question"]
            print(f"\nReceived question: {question}")
            
            # Process with RAG
            result = rag_chain.invoke(question)
            print(f"\nAnswer: {result}\n")
            
        except Exception as e:
            print(f"Error processing: {e}")
    
    consumer.on_data_received = on_data
    print("Listening for questions... Press CTRL-C to exit")
    qx.App.run()

if __name__ == "__main__":
    run_consumer() 