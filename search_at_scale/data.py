import os
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()

ZILLIZ_URI = os.getenv("URI")
ZILLIZ_TOKEN = os.getenv("TOKEN")


def create_index_coll_milvus():
    # Connect to Milvus
    connections.connect(db_name="cohere_wiki", uri=ZILLIZ_URI, port="19530", token=ZILLIZ_TOKEN)

    # Define the schema for the collection
    embedding_dim = 1024
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text_vector", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=5000),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=5000)
    ]

    schema = CollectionSchema(fields, "Schema for text and paragraph embeddings")

    # Create the collection if it doesn't exist
    collection_name = "cohere_embeddings"
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name=collection_name)
        
    collection = Collection(name=collection_name, schema=schema)

    # Define index parameters
    index_params = [
        {
            "field_name": "text_vector",
            "index_type": "HNSW",
            "metric_type": "COSINE",
        }
    ]

    # Create indexes
    for index_param in index_params:
        collection.create_index(index_param["field_name"], index_param)

    # Load the collection
    collection.load()

    print("Collection created and indexes set up successfully!")
    return collection


def insert_batch(collection, batch_data):
    collection.insert(batch_data)
    batch_data.clear()

def insert_data(collection):
    batch_size = 1000  # Adjust the batch size as needed
    batch_data = []
    
    lang = "simple"  # Use the Simple English Wikipedia subset
    docs = load_dataset("Cohere/wikipedia-2023-11-embed-multilingual-v3", lang, split="train", streaming=True)

    for doc in tqdm(docs, desc="Streaming and preparing data for Milvus"):
        title = doc['title'][:4500]
        text = doc['text'][:4500]
        emb = doc['emb']  # Assuming emb is the embedding vector
    
        batch_data.append({
            "title": title,
            "text_vector": emb,
            "text": text
        })
        if len(batch_data) >= batch_size:
            insert_batch(collection, batch_data)
    if batch_data:
        insert_batch(collection,batch_data)



if __name__ == "__main__":
    collection = create_index_coll_milvus()    
    insert_data(collection=collection)
    

