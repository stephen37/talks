from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import uvicorn
from scraper import WebScraper
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the templates directory
app.mount("/static", StaticFiles(directory="templates"), name="static")

scraper = WebScraper()
llm = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash-exp',
    google_api_key=os.getenv("GEMINI_API_KEY")
)

class Query(BaseModel):
    text: str

class ScrapingTask(BaseModel):
    task: str

@app.get("/")
async def read_root():
    return FileResponse("templates/index.html")

@app.post("/scrape")
async def scrape_website(task: ScrapingTask):
    try:
        result = await scraper.scrape_and_store(task.task)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_endpoint(query: Query):
    # Get relevant context from Milvus
    vector_results = scraper.search(query.text, limit=3)
    context = "\n\n".join([r["content"] for r in vector_results])
    
    # Construct prompt with context
    prompt = f"""Based on the following context, please answer the question. If the context doesn't contain relevant information, say so.

Context:
{context}

Question: {query.text}

Answer:"""
    
    response = llm.invoke(prompt)
    return {"response": response.content}

@app.get("/search")
async def search(q: str, use_vector: bool = True):
    results = scraper.search(q, use_vector=use_vector)
    return {"results": results}

if __name__ == "__main__":
    uvicorn.run("chat_app:app", host="0.0.0.0", port=8000, reload=True) 