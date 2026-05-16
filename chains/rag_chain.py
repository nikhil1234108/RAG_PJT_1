"""
rag_chain.py
LangChain RetrievalQA with reranking.
Standard retriever uses FAISS k=20 → rerank → top 5 → LLM.
"""
import os
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from typing import List
from langchain_core.documents import Document
SYSTEM_PROMPT = PromptTemplate(
    template = """you are an expert analyst for Blackcoffer's consulting project corpus.
Use ONLY the context below to answer. Be specific and cite article IDs.
Context:{context}

Question: {question}

Answer (mention URL_ID if relevant):
""".strip(),

    input_variables = ["context", "question"]

)


def get_llm():
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
        llm = HuggingFaceEndpoint(
            repo_id="meta-llama/Llama-3.1-8B-Instruct",
            task="conversation",
            temperature = 0
        )
        return ChatHuggingFace(llm=llm)
    from langchain_community.llms import Ollama
    return Ollama(model="mistral", temperature=0)

def build_rag_chain(vectorstore, cluster_ids: list, chunks:List[Document]=None, llm=None):
    if not cluster_ids:
        raise ValueError(
            "build_rag_chain requires cluster_ids. "
            "cluster_router_node must run before rag_node."
        )
    if llm is None:
        llm = get_llm()

    from vectorstore.vector_store import get_cluster_filtered_retriever
    retriever = get_cluster_filtered_retriever(
        vectorstore = vectorstore,
        cluster_url_ids = cluster_ids,
        )

    final_retriever = retriever

    try:
        if chunks:
            bm25retriever = BM25Retriever.from_documents(chunks)
            bm25retriever.k = 15
            final_retriever = EnsembleRetriever(
                retrievers=[retriever, bm25retriever],
                weights=[0.5, 0.5],
            )
        else:
            final_retriever = retriever
            print("[RAG] No chunks provided, skipping BM25 fallback.")
    except ImportError:
        print("[RAG] rank_bm25 is not installed. Falling back to similarity retriever only.")

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type = "stuff",
        retriever = final_retriever,
        return_source_documents = True,
        chain_type_kwargs={"prompt":SYSTEM_PROMPT},
    )

def query_rag(vectorstore, question: str, cluster_ids: list, llm=None) -> dict:
    """
    Builds cluster-only chain and queries.
    cluster_ids is REQUIRED — comes from AgentState.target_cluster_ids.
    """
    chain  = build_rag_chain(vectorstore, cluster_ids=cluster_ids, llm=llm)
    response = chain.invoke({"query": question})

    sources = list({
        doc.metadata.get("url_id","unknown")
        for doc in response.get("source_documents", [])
    })

    return {
        "answer": response["result"],
        "sources": sources,
        "source_documents": response.get("source_documents", [])
    }

if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    
    from vectorstore.vector_store import load_vectorstore
    from ingest.chunker import doc_loader, chunk_docs
    from vectorstore.clustering import load_cluster_results
    
    vectorstore = load_vectorstore()
    clusters = load_cluster_results()
    cluster_ids = clusters["kmeans"]["clusters"]["cluster_0"]
    docs = doc_loader()
    chunks = chunk_docs(docs)

    
    rag_chain = build_rag_chain(vectorstore, cluster_ids, chunks)
    
    question = "Tell me about any of the Blackcoffer's projects?"
    response = query_rag(rag_chain, question)
    print(response)

