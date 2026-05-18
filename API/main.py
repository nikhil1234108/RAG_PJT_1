"""
api/main.py
FastAPI backend for Blackcoffer RAG + NLP Agent.
Run with: uvicorn api.main:app --reload --port 8000
"""

import os
import sys
import io
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

app = FastAPI(
    title="Blackcoffer RAG + NLP Agent",
    description=(
        "LangGraph agent with KMeans cluster-guided retrieval, "
        "reranking, chatbot, and voice bot."
    ),
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup ────────────────────────────────────────────────────────────────────

@app.on_event("startup")
def startup_event():
    """Initialize PostgreSQL tables on startup."""
    from chatbot.session import init_db
    init_db()


# ── Lazy load ──────────────────────────────────────────────────────────────────

_vectorstore  = None
_agent_graph  = None


def get_components():
    global _vectorstore, _agent_graph
    if _agent_graph is None:
        from vectorstore.vector_store import load_vectorstore
        from graph.graph import build_agent_graph

        _vectorstore = load_vectorstore()
        _agent_graph = build_agent_graph(_vectorstore)
    return _vectorstore, _agent_graph


# ── Request / Response models ──────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question:  str = Field(description="User question for LangGraph agent.")
    thread_id: str = Field(
        default="default",
        description="Conversation session ID for PostgreSQL checkpointing."
    )

class QueryResponse(BaseModel):
    answer:    str
    sources:   Optional[list] = None
    thread_id: Optional[str]  = None

class MetricsRequest(BaseModel):
    text:   str            = Field(description="Raw article text to analyse.")
    url_id: Optional[str]  = Field(default=None)

class MetricsResponse(BaseModel):
    url_id:  Optional[str]
    metrics: Dict[str, Any]

class ChatRequest(BaseModel):
    session_id: str = Field(description="Session ID from /chat/session.")
    message:    str = Field(description="User message.")

class ChatResponse(BaseModel):
    session_id: str
    response:   str
    sources:    list = []

class SessionResponse(BaseModel):
    session_id: str
    mode:       str


# ══════════════════════════════════════════════════════════════════════════════
# HEALTH
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "status":  "ok",
        "message": "Blackcoffer RAG Agent is running.",
        "version": "3.0.0",
    }


# ══════════════════════════════════════════════════════════════════════════════
# LANGGRAPH AGENT
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/query", response_model=QueryResponse)
def query_agent(req: QueryRequest):
    """
    Runs full LangGraph agent pipeline.
    cluster_router → cluster-only RAG + rerank → LLM.
    State checkpointed to PostgreSQL per thread_id.
    """
    try:
        vs, agent = get_components()
        from graph.graph import run_agent
        answer = run_agent(agent, req.question, thread_id=req.thread_id)
        return QueryResponse(
            answer    = answer,
            thread_id = req.thread_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# NLP METRICS
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/metrics", response_model=MetricsResponse)
def compute_metrics(req: MetricsRequest):
    """Runs all NLP tools on raw text."""
    try:
        from tools.nlp_tools import analyse_article
        metrics = analyse_article(req.text)
        return MetricsResponse(url_id=req.url_id, metrics=metrics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/articles/{url_id}/metrics")
def article_metrics(url_id: str):
    """Fetches scraped article by URL_ID and runs all NLP tools."""
    articles_dir = os.path.join(os.path.dirname(__file__), "..", "data", "articles")
    fpath        = os.path.join(articles_dir, f"{url_id}.txt")

    if not os.path.exists(fpath):
        raise HTTPException(
            status_code=404,
            detail=f"Article {url_id} not found. Run scraper first."
        )

    with open(fpath, encoding="utf-8") as f:
        text = f.read()

    from tools.nlp_tools import analyse_article
    return {"url_id": url_id, "metrics": analyse_article(text)}


# ══════════════════════════════════════════════════════════════════════════════
# CLUSTERS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/clusters")
def get_clusters():
    """Returns KMeans cluster assignments."""
    from vectorstore.clustering import load_cluster_results
    data = load_cluster_results()
    if not data:
        raise HTTPException(
            status_code=404,
            detail="Clusters not built yet. Run build_clusters() first."
        )
    return {
        "n_clusters": data["kmeans"]["n_clusters"],
        "clusters":   data["kmeans"]["clusters"],
        "inertia":    data["kmeans"]["inertia"],
    }


@app.get("/articles/{url_id}/cluster")
def article_cluster(url_id: str):
    """Returns which cluster an article belongs to."""
    try:
        from tools.nlp_tools import get_article_cluster_tool
        return get_article_cluster_tool.run(url_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# CHATBOT
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/chat/session", response_model=SessionResponse)
def new_chat_session(mode: str = "text"):
    """
    Creates a new chat session.
    Returns session_id to use in all /chat/message calls.
    """
    try:
        vs, _ = get_components()
        from chatbot.chat_agent import chatengine
        engine     = chatengine(vs, voice_mode=(mode == "voice"))
        session_id = engine.new_session(mode=mode)
        return SessionResponse(session_id=session_id, mode=mode)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/message", response_model=ChatResponse)
def chat_message(req: ChatRequest):
    """
    Sends message to chatbot.
    cluster_router → cluster-only RAG + rerank → memory injection → LLM.
    Persists to PostgreSQL per session_id.
    """
    try:
        vs, _ = get_components()
        from chatbot.chat_agent import chatengine
        engine = chatengine(vs)
        result = engine.chat(req.session_id, req.message)
        return ChatResponse(
            session_id = result["session_id"],
            response   = result["response"],
            sources    = result["sources"],
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/history/{session_id}")
def chat_history(session_id: str, last_n: int = 20):
    """Returns conversation history for a session."""
    try:
        from chatbot.session import get_history
        history = get_history(session_id, last_n=last_n)
        return {"session_id": session_id, "history": list(history)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/chat/session/{session_id}")
def delete_chat_session(session_id: str):
    """Deletes a session and all its messages."""
    try:
        from chatbot.session import delete_session
        delete_session(session_id)
        return {"deleted": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# VOICE BOT
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/voice/session", response_model=SessionResponse)
def new_voice_session():
    """Creates a new voice session."""
    try:
        vs, _ = get_components()
        from voice.voice_bot import VoiceBot
        bot        = VoiceBot(vs)
        session_id = bot.new_session()
        return SessionResponse(session_id=session_id, mode="voice")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/voice/message")
async def voice_message(
    session_id: str,
    audio:      UploadFile = File(...),
    language:   str        = "en",
):
    """
    Accepts audio → STT → cluster RAG + rerank → LLM → TTS → returns audio.
    Response: audio/mpeg stream.
    Headers: X-Transcript, X-Response, X-Latency-Ms.
    """
    try:
        vs, _ = get_components()
        from voice.voice_bot import VoiceBot

        audio_bytes = await audio.read()
        bot         = VoiceBot(vs)
        result      = bot.process_audio(session_id, audio_bytes, language)

        if result.get("error"):
            raise HTTPException(status_code=400, detail=result["error"])

        return StreamingResponse(
            io.BytesIO(result["audio"]),
            media_type = "audio/mpeg",
            headers    = {
                "X-Transcript": result["transcript"],
                "X-Response":   result["response"][:200],
                "X-Latency-Ms": str(result["latency"]["total_ms"]),
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))