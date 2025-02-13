from browser_use import Agent, Browser, BrowserConfig, Controller
from langchain_google_genai import ChatGoogleGenerativeAI
from pymilvus import MilvusClient, DataType, Function, FunctionType
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from typing import List
import asyncio
import os
from dotenv import load_dotenv
import numpy as np

load_dotenv()

# Initialize sentence transformer for embeddings
encoder = SentenceTransformer('all-MiniLM-L6-v2')

# Define output format
class NewsItem(BaseModel):
    date: str
    title: str
    link: str

class NewsOutput(BaseModel):
    latest_news: List[NewsItem]

def init_milvus():
    """Initialize Milvus connection and create collection if not exists."""
    client = MilvusClient(uri="http://localhost:19530")
    
    if client.has_collection('web_data'):
        client.load_collection('web_data')
        return client
    
    # Create schema
    schema = client.create_schema()
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="url", datatype=DataType.VARCHAR, max_length=500)
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535, enable_analyzer=True)
    schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=384)  # MiniLM dimension
    
    # Add BM25 function for full-text search
    bm25_function = Function(
        name="content_bm25_emb",
        input_field_names=["content"],
        output_field_names=["sparse"],
        function_type=FunctionType.BM25,
    )
    schema.add_function(bm25_function)
    
    # Prepare index parameters
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="AUTOINDEX",
        metric_type="COSINE"
    )
    index_params.add_index(
        field_name="sparse",
        index_type="AUTOINDEX",
        metric_type="BM25"
    )
    
    # Create collection with schema and indexes
    client.create_collection(
        collection_name='web_data',
        schema=schema,
        index_params=index_params
    )
    
    # Load collection for searching
    client.load_collection('web_data')
    return client

class WebScraper:
    def __init__(self, task: str):
        self.browser = Browser(
            config=BrowserConfig(
                chrome_instance_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
            )
        )
        
        # Create controller with output model
        self.controller = Controller(output_model=NewsOutput)
        
        self.agent = Agent(
            task=task,
            llm=ChatGoogleGenerativeAI(
                model='gemini-2.0-flash-exp',
                google_api_key=os.getenv("GEMINI_API_KEY")
            ),
            browser=self.browser,
            controller=self.controller  # Use controller instead of direct output_model
        )
        self.client = init_milvus()

    async def scrape_and_store(self):
        """Scrape website based on task and store in Milvus."""
        history = await self.agent.run()
        
        # Process and store the result
        if history:
            # Get the final parsed result
            result = history.final_result()
            if result:
                # Parse as NewsOutput
                news_data = NewsOutput.model_validate_json(result)
                
                # Convert to string for storage
                content = result
                embedding = encoder.encode(content).tolist()
                
                # Use python.org as URL since that's what we're scraping
                current_url = "https://python.org"
                
                # Insert data
                self.client.insert(
                    collection_name='web_data',
                    data=[{
                        'url': current_url,
                        'content': content,
                        'embedding': embedding
                    }]
                )
                
                # Return the structured data
                return {"status": "success", "url": current_url, "data": news_data}
        
        return {"status": "error", "message": "No content scraped"}

    async def close(self):
        """Close browser connection."""
        await self.browser.close()

    def search(self, query: str, limit: int = 5, use_vector: bool = True):
        """Search stored data using vector similarity or full-text search."""
        try:
            self.client.load_collection('web_data')
        except Exception as e:
            print(f"Warning: Error loading collection: {e}")
            
        try:
            if use_vector:
                query_embedding = encoder.encode(query).tolist()
                results = self.client.search(
                    collection_name='web_data',
                    data=[query_embedding],
                    anns_field="embedding",
                    params={
                        "metric_type": "COSINE",
                        "params": {"nprobe": 10}
                    },
                    limit=limit,
                    output_fields=["url", "content"]
                )
            else:
                results = self.client.search(
                    collection_name='web_data',
                    data=[query],
                    anns_field="sparse",
                    params={
                        "drop_ratio_search": 0.2
                    },
                    limit=limit,
                    output_fields=["url", "content"]
                )

            # Process results
            processed_results = []
            for hits in results:
                for hit in hits:
                    entity = hit.get('entity', {})
                    score = 1 - hit.get('distance', 0) if use_vector else hit.get('score', 0)
                    
                    # Try to parse content as NewsOutput
                    content = entity.get('content', '')
                    try:
                        news_data = NewsOutput.model_validate_json(content)
                        # Format news items nicely
                        formatted_content = []
                        for item in news_data.latest_news:
                            formatted_content.append(
                                f"📅 {item.date}\n"
                                f"📰 {item.title}\n"
                                f"🔗 {item.link}"
                            )
                        content = "\n\n".join(formatted_content)
                    except:
                        # Fallback to raw content if parsing fails
                        pass
                    
                    processed_results.append({
                        "url": entity.get('url', 'unknown'),
                        "content": content,
                        "score": round(score, 4)
                    })
            return processed_results

        except Exception as e:
            print(f"Error during search: {e}")
            return []

async def main():
    scraper = WebScraper("Go to python.org and extract the latest news")
    try:
        result = await scraper.scrape_and_store()
        print("\n" + "="*50)
        print("🤖 Scraping Results")
        print("="*50)
        
        if result.get("status") == "success" and result.get("data"):
            news_data = result["data"]
            for item in news_data.latest_news:
                print(f"\n📅 {item.date}")
                print(f"📰 {item.title}")
                print(f"🔗 {item.link}")
        
        # Test vector search
        vector_results = scraper.search("python news", use_vector=True)
        print("\n" + "="*50)
        print("🔍 Vector Search Results")
        print("="*50)
        for r in vector_results:
            print(f"\n📊 Relevance Score: {r['score']}")
            print(f"🌐 URL: {r['url']}")
            print(f"\n{r['content']}")
            print("-"*50)
        
        # Test full-text search
        text_results = scraper.search("python", use_vector=False)
        print("\n" + "="*50)
        print("🔍 Full-text Search Results")
        print("="*50)
        for r in text_results:
            print(f"\n📊 Relevance Score: {r['score']}")
            print(f"🌐 URL: {r['url']}")
            print(f"\n{r['content']}")
            print("-"*50)
            
    finally:
        await scraper.close()

if __name__ == '__main__':
    asyncio.run(main()) 