# main.py
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from app.chatbot import Chatbot
import time

# Initialize FastAPI
app = FastAPI(title="RAG Chatbot")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class QueryRequest(BaseModel):
    query: str
    temperature: float = 0.5
    include_thinking: bool = False

class QueryResponse(BaseModel):
    answer: str
    thinking: str = None
    processing_time: float

# Initialize chatbot
script_dir = Path(__file__).resolve().parent
document_path = script_dir / "data" / "manual.txt"
chatbot = Chatbot(document_path)

# Start monitoring server
#start_monitoring_server()

@app.on_event("startup")
async def startup_event():
    # Initialize the chatbot on startup
    chatbot.initialize()

@app.post("/query")
async def process_query(request: QueryRequest):
    start_time = time.time()
    
    try:
        result = chatbot.chat(
            request.query, 
            temperature=request.temperature,
            include_thinking=request.include_thinkingç
        )
        
        processing_time = time.time() - start_time
        
        if request.include_thinking:
            return {
                "answer": result["answer"],
                "thinking": result["thinking"],
                "processing_time": processing_time
            }
        else:
            return {
                "answer": result,
                "processing_time": processing_time
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
