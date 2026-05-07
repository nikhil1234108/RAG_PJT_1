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

def build_rag_chain(vectorstore,chunks:List[Document], llm=None):
    if llm is None:
        llm = get_llm()

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":5})
    final_retriever = retriever

    try:
        bm25retriever = BM25Retriever.from_documents(chunks)
        bm25retriever.k = 5
        final_retriever = EnsembleRetriever(
            retrievers=[retriever, bm25retriever],
            weights=[0.5, 0.5],
        )
    except ImportError:
        print("[RAG] rank_bm25 is not installed. Falling back to similarity retriever only.")

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type = "stuff",
        retriever = final_retriever,
        return_source_documents = True,
        chain_type_kwargs={"prompt":SYSTEM_PROMPT},
    )

def query_rag(chain, question:str):
    response = chain.invoke({"query":question})
    sources = list({
        doc.metadata.get("url_id","unknown")
        for doc in response.get("source_documents", [])
    })

    return {"result":response["result"], "sources":sources}

