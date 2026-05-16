import os
from typing import List, Tuple
from functools import lru_cache
from langchain_core.documents import Document

@lru_cache(maxsize=1)
def _load_cross_encoder():
    """
    Loads cross-encoder model once and caches.
    ms-marco-MiniLM-L-6-v2:
      - Trained on MS MARCO passage ranking dataset
      - Takes (query, passage) pair as input
      - Outputs single relevance score
      - Much more precise than bi-encoder cosine similarity
    """
    from sentence_transformers import CrossEncoder
    print("[Reranker] Loading cross-encoder...")
    return CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_length = 512
    )

def rerank_with_cross_encoder(query:str, docs:List[Document], top_k:int=5) -> List[Document]:
    """
    Re-scores docs using cross-encoder.
    Returns top_k docs sorted by relevance score descending.

    Cross-encoder reads query + doc together:
      Input:  "[CLS] query [SEP] document passage [SEP]"
      Output: relevance score (higher = more relevant)

    Unlike bi-encoder (FAISS), cross-encoder has full attention
    between query and document — no information loss from compression.
    """

    if not docs:
        return docs

    model = _load_cross_encoder()
    pairs = [(query, doc.page_content) for doc in docs]
    scores = model.predict(pairs)

    scored = sorted(
        zip(scores, docs),
        key=lambda x: x[0],
        reverse=True
    )
    top_docs = [doc for _, doc in scored[:top_k]]
    print("[Reranker] Reranked " + str(len(docs)) + " docs to " + str(top_k))
    return top_docs

def rerank_with_cohere(query:str, docs:List[Document], top_k:int=5):
    """
    Cohere Rerank API — highest accuracy reranker available.
    Requires COHERE_API_KEY environment variable.

    Uses Cohere's rerank-english-v3.0 model.
    Returns docs sorted by Cohere relevance score.
    """
    import cohere
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise ValueError("COHERE_API_KEY not set in environment")

    co = cohere.Client(api_key)
    passages = [doc.page_content for doc in docs]

    response = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=passages,
        top_n=top_k
    )

    top_docs = [docs[result.index] for result in response.results]
    print(f"[Reranker] Cohere: {len(docs)} → {len(top_docs)} docs")
    print(f"  Top scores: {[round(r.relevance_score, 4) for r in response.results]}")
    return top_docs

def rerank(query:str, docs:List[Document], top_k:int=5, provider:str="auto"):
    """
    Reranks retrieved documents against query.

    provider: 'cohere' | 'cross_encoder' | 'auto'
      auto → cohere if COHERE_API_KEY set, else cross_encoder

    Args:
      query:    user query string
      docs:     FAISS retrieved documents (typically k=20)
      top_k:    number to return after reranking (typically 5)

    Returns:
      top_k documents sorted by reranker relevance score
    """
    if not docs:
        return docs

    if provider == "auto":
        provider = "cohere" if os.getenv("COHERE_API_KEY") else "cross_encoder"

    print(f"[Reranker] Provider: {provider}, "
          f"input: {len(docs)} docs, output: {top_k}")

    if provider == "cohere":
        return rerank_with_cohere(query, docs, top_k)
    elif provider == "cross_encoder":
        return rerank_with_cross_encoder(query, docs, top_k)
    else:
        raise ValueError(f"Unsupported reranker provider: {provider}")

if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from chains.rag_chain import query_rag, build_rag_chain
    from vectorstore.vector_store import build_vector_store
    from ingest.chunker import doc_loader, chunk_docs
    
    docs = doc_loader()
    chunks = chunk_docs(docs)
    vectorstore = build_vector_store(chunks)
    chain = build_rag_chain(vectorstore, chunks)
    
    query = "Tell me about any of the Blackcoffers computer cloud based projects"
    query_result = query_rag(chain, query)
    docs = query_result["source_documents"]
    print(type(docs[0]))
    reranked = rerank(query=query, docs=docs, top_k=5)
    print(reranked)