# Search at Scale using Milvus 

[Milvus](https://github.com/milvus-io/milvus), the open-source vector database is capable of handling Billions+ vectors. In this example, I am showing the capabilities of Milvus at Millions+ scale. 

I am using [Cohere's embeddings of Wikipedia ](https://huggingface.co/datasets/Cohere/wikipedia-2023-11-embed-multilingual-v3). The data is pushed to a Zilliz cluster, where Milvus is hosted. 

In the Jupyter Notebook, you'll learn how to:

* Filter data using Metadata in Milvus
* Perform a Vector Search
* Integrate Milvus with a simple RAG application

--- 
If you enjoyed this blog post, consider giving us a star on [Github](https://github.com/milvus-io/milvus) and joining our [Discord](https://discord.gg/FG6hMJStWu) to share your experiences with the community.