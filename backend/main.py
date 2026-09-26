"""
backend/main.py

FastAPI application entry point for the Internship FAQ & AI Assistant.

Endpoints:
    GET  /        -> health/status check
    POST /chat    -> main chatbot endpoint

Run with (from project root):
    uvicorn backend.main:app --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.chatbot import get_engine
from backend.config import settings

app = FastAPI(
    title="Internship FAQ & General AI Assistant",
    description=(
        "FastAPI backend that classifies internship-related questions with a "
        "TF-IDF + Logistic Regression model trained on a combined "
        "customer-service + internship-FAQ dataset, then generates the final "
        "natural-language reply with the Gemini API - answering internship "
        "questions from the FAQ knowledge base, and general questions from "
        "Gemini's own knowledge."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="The user's chat message")


class ChatResponse(BaseModel):
    response: str
    intent: str
    confidence: float
    domain: str


@app.on_event("startup")
def startup_event():
    get_engine()
    print("[main] FastAPI app started. Chatbot engine ready.")


@app.get("/")
def read_root():
    return {
        "status": "ok",
        "message": "Internship FAQ & General AI Assistant API is running.",
        "docs": "/docs",
        "chat_endpoint": "POST /chat",
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        engine = get_engine()
        result = engine.handle_message(request.message)
        return ChatResponse(
            response=result.response,
            intent=result.intent,
            confidence=result.confidence,
            domain=result.domain,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
