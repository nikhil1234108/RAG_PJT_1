import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from langgraph.graph import END, StateGraph


class AgentState(BaseModel):
    user_query: str = Field(description="Original user question.")
    route: Optional[str] = Field(default=None, description="rag | analyse | compare | both")
    target_cluster_ids: Optional[List[str]] = Field(default=None, description="Cluster-filtered article URL IDs.")
    rag_answer: Optional[str] = Field(default=None, description="Answer returned by the RAG chain.")
    rag_sources: Optional[List[str]] = Field(default=None, description="URL IDs used by the RAG chain.")
    analysis_result: Optional[Dict[str, Any]] = Field(default=None, description="Results from NLP analysis tools.")
    article_text: Optional[str] = Field(default=None, description="Cached full article text for analysis.")
    final_answer: Optional[str] = Field(default=None, description="Merged final response.")

    model_config = {"arbitrary_types_allowed": True}


ANALYSE_KEYWORDS = [
    "tech stack",
    "technology",
    "framework",
    "library",
    "tools used",
    "complexity",
    "difficult",
    "advanced",
    "topic",
    "category",
    "domain",
    "readability",
    "fog",
    "summarise",
    "summary",
    "what was built",
    "what problem",
    "analyse",
    "analyze",
    "breakdown",
]

COMPARE_KEYWORDS = [
    "compare",
    "difference",
    "versus",
    "vs",
    "which is more",
    "rank",
    "most complex",
    "highest",
    "lowest",
    "most advanced",
    "list all",
]

DATA_ARTICLES_DIR = os.path.join(PROJECT_ROOT, "data", "articles")


import contextlib

@contextlib.contextmanager
def get_checkpointer():
    db_url = os.getenv("DATABASE_URL")
    use_postgres = False
    if db_url:
        if db_url.startswith("postgresql+psycopg://"):
            db_url = db_url.replace("postgresql+psycopg://", "postgresql://")
        try:
            import psycopg
            with psycopg.connect(db_url):
                pass
            use_postgres = True
        except Exception as e:
            print(f"Postgres connection failed: {e}. Falling back to MemorySaver.")
    
    if use_postgres:
        from langgraph.checkpoint.postgres import PostgresSaver
        with PostgresSaver.from_conn_string(db_url) as checkpointer:
            yield checkpointer
    else:
        raise ConnectionError("PostgreSQL connection required but DATABASE_URL not configured or connection failed.")


def _build_cluster_rag_chain(vectorstore, cluster_ids: List[str]):
    from chains.rag_chain import SYSTEM_PROMPT, get_llm
    from langchain_classic.chains import RetrievalQA
    from vectorstore.vector_store import get_cluster_filtered_retriever

    retriever = get_cluster_filtered_retriever(vectorstore, cluster_ids, top_k=5, faiss_store_k=20)
    return RetrievalQA.from_chain_type(
        llm=get_llm(),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": SYSTEM_PROMPT},
    )


def _retrieve_cluster_docs(vectorstore, query: str, cluster_ids: Optional[List[str]], k: int = 5):
    if cluster_ids:
        from vectorstore.vector_store import get_cluster_filtered_retriever

        retriever = get_cluster_filtered_retriever(vectorstore, cluster_ids, top_k=k, faiss_store_k=20)
        docs = retriever.invoke(query)
        return docs if isinstance(docs, list) else [docs]
    return vectorstore.similarity_search(query, k=k)


def _load_article_text(url_id: str) -> str:
    article_path = os.path.join(DATA_ARTICLES_DIR, f"{url_id}.txt")
    if not os.path.exists(article_path):
        return ""
    with open(article_path, encoding="utf-8", errors="replace") as file:
        return file.read().strip()


def _select_primary_article_text(vectorstore, state: AgentState) -> str:
    docs = _retrieve_cluster_docs(vectorstore, state.user_query, state.target_cluster_ids, k=1)
    if not docs:
        return ""

    url_id = docs[0].metadata.get("url_id")
    if url_id:
        article_text = _load_article_text(url_id)
        if article_text:
            return article_text
    return docs[0].page_content


def _select_compare_articles(vectorstore, state: AgentState, k: int = 5) -> List[Dict[str, str]]:
    docs = _retrieve_cluster_docs(vectorstore, state.user_query, state.target_cluster_ids, k=max(k, 8))
    selected: List[Dict[str, str]] = []
    seen = set()

    for doc in docs:
        url_id = doc.metadata.get("url_id")
        if not url_id or url_id in seen:
            continue
        text = _load_article_text(url_id) or doc.page_content
        selected.append({"url_id": url_id, "text": text})
        seen.add(url_id)
        if len(selected) >= k:
            break

    return selected


def router_node(state: AgentState) -> AgentState:
    query = state.user_query.lower()
    is_compare = any(keyword in query for keyword in COMPARE_KEYWORDS)
    is_analyse = any(keyword in query for keyword in ANALYSE_KEYWORDS)

    if is_compare:
        route = "compare"
    elif is_analyse:
        route = "analyse"
    else:
        route = "rag"

    print(f"[Router] '{state.user_query}' -> {route}")
    return state.model_copy(update={"route": route})

def cluster_router_node(state: AgentState) -> AgentState:
    from vectorstore.clustering import load_cluster_results
    from vectorstore.vector_store import get_embeddings

    data = load_cluster_results()
    if not data or "kmeans" not in data:
        print("[ClusterRouter] No clusters found - full search fallback")
        return state.model_copy(update={"target_cluster_ids": None})

    centroids = np.array(data["kmeans"]["centroids"], dtype=np.float32)
    embeddings = get_embeddings()
    query_vector = np.array(embeddings.embed_query(state.user_query), dtype=np.float32)
    distances = np.linalg.norm(centroids - query_vector, axis=1)
    nearest_cluster = int(np.argmin(distances))
    cluster_key = f"cluster_{nearest_cluster}"
    cluster_url_ids = data["kmeans"]["clusters"].get(cluster_key, [])

    print(
        f"[ClusterRouter] -> {cluster_key} "
        f"(dist={distances[nearest_cluster]:.4f}, {len(cluster_url_ids)} articles)"
    )
    return state.model_copy(update={"target_cluster_ids": cluster_url_ids})


def build_rag_node(vectorstore):
    def rag_node(state: AgentState) -> AgentState:
        """
        cluster_ids MUST be set by cluster_router_node.
        No fallback — if cluster_ids empty something went wrong upstream.
        """
        from chains.rag_chain import query_rag

        cluster_ids = state.target_cluster_ids

        if not cluster_ids:
            print("[RAG] ERROR: cluster_ids empty — cluster_router did not run")
            return state.model_copy(update={
                "rag_answer":  "Could not determine relevant cluster. "
                               "Please rephrase your question.",
                "rag_sources": [],
            })

        print(f"[RAG] Searching cluster: {cluster_ids}")

        result = query_rag(
            vectorstore = vectorstore,
            question    = state.user_query,
            cluster_ids = cluster_ids,
        )

        return state.model_copy(update={
            "rag_answer":  result["answer"],
            "rag_sources": result["sources"],
        })
    return rag_node


def build_analyse_node(vectorstore):
    def analyse_node(state: AgentState) -> AgentState:
        print(f"[Analyse] {state.user_query}")
        article_text = state.article_text or _select_primary_article_text(vectorstore, state)

        if not article_text:
            return state.model_copy(update={"analysis_result": {"error": "No article found"}})

        from tools.nlp_tools import (
            complexity_classifier_tool,
            project_summary_tool,
            readability_tool,
            tech_stack_extractor_tool,
            topic_classifier_tool,
        )

        query = state.user_query.lower()
        result: Dict[str, Any] = {}

        if any(keyword in query for keyword in ["tech", "stack", "framework", "library", "tool", "language"]):
            result["tech_stack"] = tech_stack_extractor_tool.run(article_text)
        if any(keyword in query for keyword in ["complex", "difficult", "advanced", "basic", "level"]):
            result["complexity"] = complexity_classifier_tool.run(article_text)
        if any(keyword in query for keyword in ["topic", "category", "domain", "type", "field"]):
            result["topic"] = topic_classifier_tool.run(article_text)
        if any(keyword in query for keyword in ["read", "fog", "sentence", "word count"]):
            result["readability"] = readability_tool.run(article_text)
        if any(keyword in query for keyword in ["summar", "what was built", "what problem", "explain", "describe"]):
            result["summary"] = project_summary_tool.run(article_text)
        if not result:
            result["summary"] = project_summary_tool.run(article_text)
            result["tech_stack"] = tech_stack_extractor_tool.run(article_text)

        return state.model_copy(update={"analysis_result": result, "article_text": article_text})

    return analyse_node


def build_compare_node(vectorstore):
    def compare_node(state: AgentState) -> AgentState:
        print(f"[Compare] {state.user_query}")

        from tools.nlp_tools import (
            complexity_classifier_tool,
            tech_stack_extractor_tool,
            topic_classifier_tool,
        )

        comparison = []
        for article in _select_compare_articles(vectorstore, state, k=5):
            complexity = complexity_classifier_tool.run(article["text"])
            tech_stack = tech_stack_extractor_tool.run(article["text"])
            topic = topic_classifier_tool.run(article["text"])
            comparison.append(
                {
                    "url_id": article["url_id"],
                    "complexity": complexity.get("complexity"),
                    "fog_index": complexity.get("fog_index"),
                    "tech_count": tech_stack.get("total_count", 0),
                    "topic": topic.get("primary_topic"),
                    "techs": tech_stack.get("technologies_found", []),
                }
            )

        query = state.user_query.lower()
        if "complex" in query or "advanced" in query:
            comparison.sort(key=lambda row: row["fog_index"] or 0, reverse=True)
        elif "tech" in query:
            comparison.sort(key=lambda row: row["tech_count"], reverse=True)

        return state.model_copy(update={"analysis_result": {"comparison": comparison}})

    return compare_node


def synthesis_node(state: AgentState) -> AgentState:
    parts: List[str] = []

    if state.rag_answer:
        parts.append(f"**Answer:**\n{state.rag_answer}")
        if state.rag_sources:
            parts.append(f"**Sources:** {', '.join(state.rag_sources)}")

    result = state.analysis_result or {}

    if result.get("error"):
        parts.append(f"Warning: {result['error']}")

    if "summary" in result:
        lines = [
            f"- **{key}:** {value}"
            for key, value in result["summary"].items()
            if key in ["PROJECT", "PROBLEM", "TECH", "OUTCOME", "DOMAIN"]
        ]
        if lines:
            parts.append("**Project Summary:**\n" + "\n".join(lines))

    if "tech_stack" in result:
        tech_stack = result["tech_stack"]
        categories = "\n".join(
            f"  - {category}: {', '.join(techs)}"
            for category, techs in tech_stack.get("by_category", {}).items()
        )
        parts.append(f"**Tech Stack ({tech_stack.get('total_count', 0)} found):**\n{categories}")

    if "complexity" in result:
        complexity = result["complexity"]
        parts.append(
            f"**Complexity:** {complexity.get('complexity')} - {complexity.get('reason')}\n"
            f"  FOG: {complexity.get('fog_index')} | Words: {complexity.get('word_count')}"
        )

    if "topic" in result:
        topic = result["topic"]
        secondary = f" (also: {topic.get('secondary_topic')})" if topic.get("secondary_topic") else ""
        parts.append(f"**Domain:** {topic.get('primary_topic')}{secondary}")

    if "readability" in result:
        readability = result["readability"]
        parts.append(
            f"**Readability:** {readability.get('readability_label')} "
            f"(FOG: {readability.get('fog_index')}, "
            f"Avg sentence: {readability.get('avg_sentence_length')} words)"
        )

    if "comparison" in result:
        rows = result["comparison"]
        table = (
            "**Comparison:**\n"
            "| Article | Topic | Complexity | FOG | Tech Count |\n"
            "|---|---|---|---|---|\n"
        )
        for row in rows:
            table += (
                f"| {row['url_id']} | {row['topic']} | {row['complexity']} | "
                f"{row['fog_index']} | {row['tech_count']} |\n"
            )
        parts.append(table)

    final_answer = "\n\n".join(parts) if parts else "I couldn't find a relevant answer."
    return state.model_copy(update={"final_answer": final_answer})


def route_after_cluster(state: AgentState) -> str:
    return state.route or "rag"


def after_rag(state: AgentState) -> str:
    return "analyse" if state.route == "both" else "synthesis"


def build_agent_graph(vectorstore, checkpointer=None):
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("cluster_router", cluster_router_node)
    graph.add_node("rag", build_rag_node(vectorstore))
    graph.add_node("analyse", build_analyse_node(vectorstore))
    graph.add_node("compare", build_compare_node(vectorstore))
    graph.add_node("synthesis", synthesis_node)

    graph.set_entry_point("router")
    graph.add_edge("router", "cluster_router")

    graph.add_conditional_edges(
        "cluster_router",
        route_after_cluster,
        {
            "rag": "rag",
            "both": "rag",
            "analyse": "analyse",
            "compare": "compare",
        },
    )

    graph.add_conditional_edges(
        "rag",
        after_rag,
        {
            "analyse": "analyse",
            "synthesis": "synthesis",
        },
    )

    graph.add_edge("analyse", "synthesis")
    graph.add_edge("compare", "synthesis")
    graph.add_edge("synthesis", END)

    return graph.compile(checkpointer=checkpointer)


def run_agent(graph, query: str, thread_id: str = "default") -> str:
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = AgentState(user_query=query)
    result = graph.invoke(initial_state, config=config)

    if isinstance(result, AgentState):
        return result.final_answer or "No answer generated."
    if isinstance(result, dict):
        return result.get("final_answer", "No answer generated.")
    return "No answer generated."


if __name__ == "__main__":
    from ingest.chunker import chunk_docs, doc_loader
    from vectorstore.vector_store import load_vectorstore

    vectorstore = load_vectorstore(backend="faiss")
    with get_checkpointer() as checkpointer:
        if hasattr(checkpointer, "setup"):
            checkpointer.setup()
        graph = build_agent_graph(vectorstore, checkpointer)
        print(run_agent(graph, "i want to do cloud based projects , tell me how many projects are done and give count with names and id"))
