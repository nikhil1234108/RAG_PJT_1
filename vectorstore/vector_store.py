import numpy as np
import uuid
import time
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import os
import sys
from typing import List, Optional
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import faiss

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
load_dotenv(os.path.join(PROJECT_ROOT, ".env.txt"))

PRIMARY_STORE_PATH = os.path.join(PROJECT_ROOT, "data", "faiss_index")
LEGACY_STORE_PATH = os.path.join(PROJECT_ROOT, "data", "faiss_index", "data")


def get_store_path() -> str:
    if os.path.exists(os.path.join(PRIMARY_STORE_PATH, "index.faiss")):
        return PRIMARY_STORE_PATH
    if os.path.exists(os.path.join(LEGACY_STORE_PATH, "index.faiss")):
        return LEGACY_STORE_PATH
    return PRIMARY_STORE_PATH

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_VECTOR_BACKEND = "faiss"
DEFAULT_COLLECTION_NAME = "rag_articles"

DIM=384

def get_embeddings() -> HuggingFaceEmbeddings:
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    model_kwargs = {"device": "cpu", "local_files_only": True}
    if hf_token:
        model_kwargs["token"] = hf_token

    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs=model_kwargs,
        encode_kwargs={"normalize_embeddings": True},
    )

def get_vector_backend(backend: Optional[str] = None) -> str:
    return (backend or os.getenv("VECTOR_STORE", DEFAULT_VECTOR_BACKEND)).strip().lower()

def get_postgres_connection_string() -> str:
    for env_name in ("PGVECTOR_CONNECTION_STRING", "POSTGRES_CONNECTION_STRING", "DATABASE_URL"):
        value = os.getenv(env_name)
        if value:
            return value
    raise ValueError(
        "Postgres vector store needs a connection string. Set DATABASE_URL, "
        "POSTGRES_CONNECTION_STRING, or PGVECTOR_CONNECTION_STRING, for example: "
        "postgresql+psycopg://user:password@localhost:5432/dbname"
    )

def get_collection_name() -> str:
    return os.getenv("PGVECTOR_COLLECTION_NAME", DEFAULT_COLLECTION_NAME)

def build_vector_store(chunks:List[Document], force_build:bool=False, backend: Optional[str]=None):
    selected_backend = get_vector_backend(backend)
    if selected_backend in {"postgres", "pgvector"}:
        return build_postgres_vector_store(chunks, force_build=force_build)
    if selected_backend != "faiss":
        raise ValueError(f"Unsupported VECTOR_STORE backend: {selected_backend}")
    return build_faiss_vector_store(chunks, force_build=force_build)

def build_faiss_vector_store(chunks:List[Document], force_build:bool=False) -> FAISS:
    store_path = get_store_path()
    if os.path.exists(store_path) and not force_build:
        print(f"Loading Existing Vectors from {store_path}")
        return load_vectorstore()

    print(f"Embedding {len(chunks)} chunks...")
    embeddings = get_embeddings()
    texts = [chunk.page_content for chunk in chunks]
    vectors = np.array(embeddings.embed_documents(texts), dtype=np.float32)
    print(f"vectors shape: {vectors.shape}")

    index = faiss.IndexFlatIP(DIM)
    index.add(vectors)
    print(f"IndexFlatIP built. {index.ntotal} vectors indexed.")


    index_to_docstore_id, docstore_dict = {}, {}
    for i, chunk in enumerate(chunks):
        doc_id = str(uuid.uuid4())
        index_to_docstore_id[i] = doc_id
        docstore_dict[doc_id] = chunk

    vectorstore = FAISS(
        embedding_function=embeddings.embed_query,
        index = index,
        docstore = InMemoryDocstore(docstore_dict),
        index_to_docstore_id = index_to_docstore_id,
    )

    os.makedirs(store_path, exist_ok=True)
    print(f"Saving vectors to {store_path}")
    vectorstore.save_local(store_path)
    print(f"FAISS index saved to {store_path}")
    return vectorstore

def build_postgres_vector_store(chunks:List[Document], force_build:bool=False):
    from langchain_community.vectorstores import PGVector

    connection_string = get_postgres_connection_string()
    embeddings = get_embeddings()
    collection_name = get_collection_name()

    print(f"Embedding and saving {len(chunks)} chunks to Postgres collection '{collection_name}'...")
    start_time = time.time()
    print("Starting PGVector.from_documents...")
    vectorstore = PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        connection_string=connection_string,
        pre_delete_collection=force_build,
        use_jsonb=True,
        embedding_length=DIM,
    )
    print(f"Finished PGVector.from_documents in {time.time() - start_time:.2f} seconds.")
    print("Postgres vector store ready.")
    return vectorstore

def load_vectorstore(backend: Optional[str]=None):
    selected_backend = get_vector_backend(backend)
    if selected_backend in {"postgres", "pgvector"}:
        from langchain_community.vectorstores import PGVector

        return PGVector(
            connection_string=get_postgres_connection_string(),
            embedding_function=get_embeddings(),
            collection_name=get_collection_name(),
            use_jsonb=True,
            embedding_length=DIM,
        )
    if selected_backend != "faiss":
        raise ValueError(f"Unsupported VECTOR_STORE backend: {selected_backend}")

    embeddings = get_embeddings()
    store_path = get_store_path()
    return FAISS.load_local(
        store_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )

def get_cluster_filtered_retriever(vectorstore, cluster_url_ids: list, k: int = 5):

    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever

    from typing import List
    from pydantic import Field

    class ClusterFilteredRetriever(BaseRetriever):
        cluster_ids: list = Field(default_factory=list)
        top_k: int        = Field(default=5)

        def _get_relevant_documents(self, query: str) -> List[Document]:
            cluster_chunks = []
            cluster_positions = []
            for pos, doc_id in vectorstore.index_to_docstore_id.items():
                doc = vectorstore.docstore.search(doc_id)
                if doc.metadata.get("url_id") in self.cluster_ids:
                    cluster_chunks.append(doc)
                    cluster_positions.append(pos)

            if not cluster_chunks:
                print("[ClusterRetriever] No chunks found — full search fallback")
                return vectorstore.similarity_search(query, k=self.top_k)

            dim = vectorstore.index.d
            vectors = np.zeros((len(cluster_positions), dim), dtype = np.float32)

            for i, pos in enumerate(cluster_positions):
                vectorstore.index.reconstruct(pos, vectors[i])

            embeddings = get_embeddings()
            query_vector = np.array(embeddings.embed_query(query),dtype=np.float32).reshape(1, -1)

            temp_index = faiss.IndexFlatIP(dim)
            temp_index.add(vectors)

            distances, indices = temp_index.search(query_vector, k=self.top_k)
            results = []
            for idx in indices[0]:
                if idx != -1:
                    results.append(cluster_chunks[idx])

            return results

    return ClusterFilteredRetriever(cluster_ids=cluster_url_ids, top_k=k)


if __name__ == "__main__":
    import sys
    print("Step 1: Building article-level clusters...")
    from vectorstore.clustering import build_clusters
    build_clusters()

    print("Step 1: Building article-level clusters...")

    from ingest.chunker import doc_loader, chunk_docs
    docs = doc_loader()
    chunks = chunk_docs(docs)
    vs = build_vector_store(chunks, force_build=True, backend="faiss")
    results = vs.similarity_search("LangChain RAG pipeline", k=3)
    for r in results:
        print(f"\n{r.metadata['url_id']}: {r.page_content[:150]}")
