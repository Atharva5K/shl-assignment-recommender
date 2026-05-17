"""
FastAPI application: main entry point for the SHL Assessment Recommender.

Per Strategy doc Sections 4-5:
- Stateless API with /health and /chat endpoints
- Service preloads FAISS index, metadata, and embedding model at startup
- Schema compliance is strict
"""
import logging
import time
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.retrieval import retriever
from app.llm import llm_client
from app.recommender import process_chat

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ─── Pydantic Models (strict schema compliance) ───

class Message(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., description="Full conversation history")


class Recommendation(BaseModel):
    name: str = Field(..., description="Assessment name from catalog")
    url: str = Field(..., description="Catalog URL")
    test_type: str = Field(..., description="Assessment type abbreviation")


class ChatResponse(BaseModel):
    reply: str = Field(..., description="Agent's response text")
    recommendations: List[Recommendation] = Field(
        default_factory=list, 
        description="Empty when clarifying/refusing; 1-10 items when recommending"
    )
    end_of_conversation: bool = Field(
        default=False, 
        description="True only when the agent considers the task complete"
    )


class HealthResponse(BaseModel):
    status: str = "ok"


# ─── Application Lifecycle ───

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: preload FAISS index, embedding model, and LLM client.
    Per Strategy Section 20: Preload at startup to minimize per-request latency.
    """
    logger.info("Starting SHL Assessment Recommender...")
    start = time.time()
    
    # Load retrieval system (embedding model + FAISS index)
    retriever.load()
    
    # Initialize LLM client
    llm_client.initialize()
    
    elapsed = time.time() - start
    logger.info(f"Startup complete in {elapsed:.1f}s")
    
    yield
    
    logger.info("Shutting down SHL Assessment Recommender...")


# ─── FastAPI App ───

app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational agent for recommending SHL assessments",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Endpoints ───

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    Per Strategy Section 5: Returns {"status": "ok"} with HTTP 200.
    No retrieval or LLM inference occurs inside this endpoint.
    """
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint. Stateless - every call carries full conversation history.
    Per Strategy Sections 4-5.
    
    The system:
    1. Reconstructs intent from conversation history
    2. Classifies query type
    3. Retrieves relevant assessments (if needed)
    4. Generates grounded response
    """
    start = time.time()
    
    try:
        # Convert Pydantic models to dicts for processing
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Run the recommendation pipeline
        result = process_chat(messages)
        
        elapsed = time.time() - start
        logger.info(
            f"Chat processed in {elapsed:.2f}s | "
            f"recommendations: {len(result.get('recommendations', []))} | "
            f"end: {result.get('end_of_conversation', False)}"
        )
        
        # Build validated response
        recommendations = []
        for rec in result.get("recommendations", []):
            recommendations.append(Recommendation(
                name=rec.get("name", ""),
                url=rec.get("url", ""),
                test_type=rec.get("test_type", "K"),
            ))
        
        return ChatResponse(
            reply=result.get("reply", ""),
            recommendations=recommendations,
            end_of_conversation=result.get("end_of_conversation", False),
        )
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        # Return a safe response rather than crashing
        return ChatResponse(
            reply="I apologize, but I encountered an error processing your request. Could you please try again?",
            recommendations=[],
            end_of_conversation=False,
        )
