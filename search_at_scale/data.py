import os
from pymilvus import MilvusClient
from datasets import load_dataset
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

ZILLIZ_URI = os.getenv("URI")
ZILLIZ_TOKEN = os.getenv("TOKEN")


def create_index_coll_milvus():
    # Create MilvusClient instance
    # client = MilvusClient(uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)
    client = MilvusClient(uri="http://localhost:19530")
    # Define the schema for the collection
    embedding_dim = 1024
    collection_name = "test_collection"

    # Create the collection if it doesn't exist
    if not client.has_collection(collection_name):
        from pymilvus import CollectionSchema, FieldSchema, DataType

        schema = CollectionSchema([
            FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema("text_vector", DataType.FLOAT_VECTOR, dim=embedding_dim),
            FieldSchema("title", DataType.VARCHAR, max_length=5000),
            FieldSchema("text", DataType.VARCHAR, max_length=5000)
        ])
        client.create_collection(collection_name=collection_name, schema=schema)

    # # Define and create index
    # index_params = {
    #     "metric_type": "COSINE",
    #     "index_type": "HNSW",
    #     "params": {"M": 8, "efConstruction": 64},
    # }
    # client.create_index(
    #     collection_name=collection_name,
    #     field_name="text_vector",
    #     index_params=index_params,
    # )

    # Load the collection
    # client.load_collection(collection_name=collection_name)

    print("Collection created and indexes set up successfully!")
    return client, collection_name


def insert_batch(client, collection_name, batch_data):
    client.insert(collection_name=collection_name, data=batch_data)
    batch_data.clear()


def insert_data(client, collection_name):
    batch_size = 1000  # Adjust the batch size as needed
    batch_data = []

    docs = load_dataset(
        "Cohere/wikipedia-2023-11-embed-multilingual-v3",
        "en",
        split="train",
        streaming=True,
    )

    for doc in tqdm(docs, desc="Streaming and preparing data for Milvus"):
        title = doc["title"][:4500]
        text = doc["text"][:4500]
        emb = doc["emb"]  # The embedding vector

        batch_data.append({"title": title, "text_vector": emb, "text": text})
        if len(batch_data) >= batch_size:
            insert_batch(client, collection_name, batch_data)
    if batch_data:
        insert_batch(client, collection_name, batch_data)


if __name__ == "__main__":
    client, collection_name = create_index_coll_milvus()
    # insert_data(client=client, collection_name=collection_name)
