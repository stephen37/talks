from datasets import load_dataset
import s3fs
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from dotenv import load_dotenv
import os 

load_dotenv()

# Configure S3 credentials
s3 = s3fs.S3FileSystem(
    key=os.getenv("AWS_ACCESS_KEY_ID"), secret=os.getenv("AWS_SECRET_ACCESS_KEY")
)

# Prepare S3 output directory
output_dir = "s3://cohere-embeddings/wikipedia-2023-11-embed-multilingual-v3"

# Load the dataset in streaming mode
dataset = load_dataset(
    "Cohere/wikipedia-2023-11-embed-multilingual-v3",
    "en",
    split="train",
    streaming=True,
)


# Function to write batches to S3
def write_batch_to_s3(batch, file_number):
    table = pa.Table.from_pydict(batch)
    s3_path = f"{output_dir}/part-{file_number:05d}.parquet"
    with s3.open(s3_path, "wb") as f:
        pq.write_table(table, f)


# Process and upload the data in batches
batch_size = 100000  # Adjust based on your needs and memory constraints
current_batch = {}
file_number = 0

for i, example in tqdm(enumerate(dataset)):
    for key, value in example.items():
        if key not in current_batch:
            current_batch[key] = []
        current_batch[key].append(value)

    if (i + 1) % batch_size == 0:
        write_batch_to_s3(current_batch, file_number)
        file_number += 1
        current_batch = {}

# Write any remaining data
if current_batch:
    write_batch_to_s3(current_batch, file_number)

print(f"Dataset streaming and upload complete. Data saved to {output_dir}")
