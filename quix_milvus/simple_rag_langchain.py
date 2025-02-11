from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import logging
import sys

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Create sample documents
documents = [
    "Milvus is a vector database built for scalable similarity search.",
    "LangChain is a framework for developing applications powered by language models.",
    "Ollama runs large language models locally."
]

# Initialize components
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
llm = Ollama(model="mistral-small")

# Set up Milvus vector store
vector_store = Milvus.from_texts(
    texts=documents,
    embedding=embeddings,
    connection_args={"host": "localhost", "port": "19530"},
    collection_name="rag_demo",
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

# Query the system
print("\nInitial RAG Query:")
question = "What is Milvus and what is it used for?"
print(f"Question: {question}")
print(f"Answer: {rag_chain.invoke(question)}\n")

# Show that new information is not available
print("\nQuery for missing information:")
question = "What is the capital of France?"
print(f"Question: {question}")
print(f"Answer: {rag_chain.invoke(question)}\n") 