# Streaming RAG with Milvus, Quix, and Mistral

![Streaming RAG Demo](Streaming_RAG_Demo_LangChain.png)

This repository demonstrates a real-time Retrieval-Augmented Generation (RAG) system that combines:
- Vector search with Milvus
- Streaming data with Quix (Kafka)
- Local LLM inference with Mistral (via Ollama)

## Overview

The demo shows how to:
1. Set up a basic RAG system with LangChain and Milvus
2. Stream new data through Kafka using Quix
3. Update the knowledge base in real-time
4. Query the system before and after receiving new information

## Requirements

- Python 3.11+
- Docker (for Milvus and Kafka)
- Ollama
- uv (install with `pip install uv`)

## Quick Start

1. Create a virtual environment and install dependencies:
```bash
uv sync
source .venv/bin/activate
```

2. Start Milvus and Kafka:
```bash
docker-compose up -d
```

3. Make sure you have downloaded the `mistral-small` model, you can do this by running:
```bash
ollama pull mistral-small
```

4. Open and run the Jupyter notebook:
```bash
jupyter notebook RAG_Streaming_Demo.ipynb
```

## How it Works

1. **Initial Setup**: 
   - Initializes an empty Milvus collection
   - Sets up LangChain with Mistral LLM
   - Configures Kafka consumer/producer

2. **Demo Flow**:
   - Shows initial RAG query (empty knowledge base)
   - Streams sample messages through Kafka
   - Updates vector store with new information
   - Demonstrates improved responses with new knowledge

## Architecture

```
[Kafka/Quix] -> [Consumer] -> [Milvus Vector DB] -> [RAG Chain] -> [Mistral LLM]
```

## Files

- `RAG_Streaming_Demo.ipynb`: Main demo notebook
- `docker-compose.yml`: Milvus and Kafka setup
- `pyproject.toml`: Project dependencies

## Notes

- This is a demo setup focused on clarity over production readiness
- Uses local LLM inference for simplicity
- Kafka setup is minimal for demonstration purposes
