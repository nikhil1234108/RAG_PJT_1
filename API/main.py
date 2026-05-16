import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

app = FastAPI(
    title="API for AI Agent",
    description="This API allows you to interact with an AI agent that can perform various tasks based on your input.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

_vectorstore = None
_rag_chain = None
_agent_graph = None

def get_components():
    global _vectorstore, _agent_graph
    if _agent_graph is None:
        from vectorstore.vector_store import build_vector_store
        from graph.graph import build_agent_graph

        _vectorstore = build_vector_store()
        _agent_graph = build_agent_graph(_vectorstore)
    return _vectorstore, None, _agent_graph

class query_request(BaseModel):
     question: str = Field(
        description="User question to run through the LangGraph agent.")
     
     thread_id: str = Field(
        default="thread_1",
        description="Conversation session ID for PostgreSQL checkpointing. "
                    "Same thread_id = same conversation history. "
                    "Different thread_id = fresh conversation."
    )
     
class MetricRequest(BaseModel):
    text: str = Field(description = "raw article text to analyse")
    url_id: Optional[str] = Field(
        default = None,
        description = "optional url_id to link the metrics to a specific article in the database"
    )

class QueryResponse(BaseModel):
    answer: str = Field(description="raw article text analysis")
    thread_id: Optional[str] = Field(
        default = None,
        description = "optional thread_id to link the response to a specific conversation"
    )
    sources: Optional[list] = Field(
        default = None, description = "list of url_ids that were the source of the information in the answer"
    )

class MetricResponse(BaseModel):
    url_id: Optional[str] = Field(
        default = None, description = "optional url_id to link the metrics to a specific article in the database"
    )
    metrics: Dict[str, Any] = Field(description = "dictionary of metrics calculated from the input text")


@app.get("/")
def root():
    return {
        "status":  "ok",
        "message": "Blackcoffer RAG Agent is running.",
        "version": "2.0.0",
    }


@app.get("/favicon.ico", status_code=204)
def favicon():
    return None


@app.post("/query", response_model=QueryResponse)
def query_agent(req:query_request):
    try:
        vs, chain, agent = get_components()
        from graph.graph import run_agent_graph
        answer = run_agent_graph(agent, req.question, req.thread_id)
        return QueryResponse(answer=answer, thread_id=req.thread_id )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/metrics", response_model=MetricResponse)
def calculate_metrics(req: MetricRequest):
    try:
        from tools.nlp_tools import analyse_article
        metrics = analyse_article(req.text)
        return MetricResponse(url_id=req.url_id, metrics=metrics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/rag", response_model=QueryResponse)
def rag_only(req: query_request):
    try:
        vs, chain, _ = get_components()
        from chains.rag_chain import query_rag
        result = query_rag(chain, req.question)
        return QueryResponse(answer=result["answer"], thread_id=req.thread_id, sources=result["sources"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/articles/{url_id}/metrics", response_model=MetricResponse)
def article_metrics(url_id:str):
    articles_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","data","articles")
    file_path = os.path.join(articles_dir, f"{url_id}.txt")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Article with url_id {url_id} not found.")
    
    text = None
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    from tools.nlp_tools import analyse_article
    metrics = analyse_article(text)
    return MetricResponse(url_id=url_id, metrics=metrics)

@app.get("/clusters")
def get_clusters():
    from vectorstore.clustering import load_clusters_results
    clusters = load_clusters_results()
    if not clusters:
        raise HTTPException(status_code=404, detail="No cluster data found.")
    
    return {
        "n_clusters":clusters["k_means"]["n_clusters"],
        "inertia":clusters["k_means"]["inertia"],
        "silhouette_score":clusters["k_means"]["silhouette_score"],
        "clusters":clusters["k_means"]["clusters"]
    }

@app.get("/articles/{url_id}/clusters")
def get_article_cluster(url_id:str):
    from tools.nlp_tools import get_article_cluster_tool
    cluster_info = get_article_cluster_tool(url_id)
    if not cluster_info:
        raise HTTPException(status_code=404, detail=f"No cluster info found for article with url_id {url_id}.")
    return cluster_info
    
    


    
        