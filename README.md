# Advanced Agentic RAG Pipeline

## Overview
This project implements an intelligent, agent-based Retrieval-Augmented Generation (RAG) pipeline built using **LangGraph** and **LangChain**. It analyzes a corpus of consulting projects, retrieves highly relevant context using a hybrid retrieval strategy (Similarity + BM25 Ensemble), and synthesizes analytical answers using Large Language Models (such as Llama-3.1-8B-Instruct or Mistral).

*Note: Due to time constraints, the user interface currently operates as a local script/CLI, and some originally planned features (such as a full chatbot UI and cloud/virtual database integrations) have been deferred to future iterations.*

## Architecture & Core Features
* **Graph-Based Agent Routing:** Uses LangGraph to intelligently route user queries into different execution paths (`rag`, `analyse`, `compare`) based on the semantic intent of the query.
* **Hybrid Retrieval:** Employs an `EnsembleRetriever` combining dense vector search (FAISS/PGVector) with sparse keyword search (`rank_bm25`) for robust and highly accurate context retrieval.
* **K-Means Clustering & Filtering:** Documents are pre-clustered. The query is dynamically routed to the nearest document cluster to narrow down the search space and improve accuracy.
* **Advanced NLP Analysis Tools:** Custom tools designed to analyze documents for:
  * Tech Stack Extraction
  * Project Summary & Problem Breakdown
  * Readability & FOG Index Analysis
  * Complexity Classification
  * Topic/Domain Classification
* **State Checkpointing:** Persistent graph state is managed via PostgreSQL (`PostgresSaver`), with an automatic fallback mechanism to an in-memory checkpointer (`MemorySaver`) if the database is unavailable.

## System Architecture

```mermaid
graph TD
    User([User Query]) --> Router
    Router[Intent Router] --> ClusterRouter[Cluster Router]
    
    ClusterRouter -->|Route: rag / both| RAG[RAG Node]
    ClusterRouter -->|Route: analyse| Analyse[Analyse Node]
    ClusterRouter -->|Route: compare| Compare[Compare Node]
    
    RAG -->|Route: both| Analyse
    RAG -->|Route: rag| Synthesis[Synthesis Node]
    
    Analyse --> Synthesis
    Compare --> Synthesis
    
    Synthesis --> Output([Final Answer])
    
    subgraph Data Layer
        FAISS[(FAISS / PGVector)]
        BM25[(rank_bm25)]
    end
    
    subgraph Retrieval Pipeline
        Ensemble[Ensemble Retriever]
        FAISS -.-> Ensemble
        BM25 -.-> Ensemble
    end
    
    Ensemble -.-> RAG
    
    subgraph State Management
        DB[(PostgreSQL / Memory)]
    end
    Router -.-> DB
    Synthesis -.-> DB
```

## Deep Dive into Project Structure & Files

The project is modularized into several key components to cleanly separate concerns.

### 1. `graph/` (State Graph & Routing)
- **`graph.py`**: The central orchestrator. It defines the LangGraph `AgentState` and creates the main workflow `StateGraph`. It contains multiple nodes (`router`, `cluster_router`, `rag`, `analyse`, `compare`, `synthesis`) that guide the flow of execution based on user intent. This is also where the `PostgresSaver` (or `MemorySaver` fallback) checkpointer is configured to persist conversation state.

### 2. `chains/` (Retrieval & LLM Chains)
- **`rag_chain.py`**: Configures the foundational LangChain `RetrievalQA` pipeline. It sets up the system prompts, instantiates the Large Language Model (Llama-3.1 via HuggingFace or Mistral via Ollama), and creates the `EnsembleRetriever` to blend dense vector retrieval (FAISS/PGVector) and sparse keyword retrieval (`rank_bm25`).

### 3. `vectorstore/` (Storage & Clustering)
- **`vector_store.py`**: Handles building, saving, and loading the vector databases. It supports both local FAISS indexes and PGVector. It also contains the `ClusterFilteredRetriever`, which dynamically filters semantic searches to only look within specific document clusters.
- **`clustering.py`**: Contains the logic to apply K-Means clustering algorithms over the embedded article vectors. Clustering helps drastically reduce the search space and improve retrieval precision.
- **`visualize_umap.py`**: A utility script used to visualize the high-dimensional document clusters in a 2D space using UMAP (Uniform Manifold Approximation and Projection).

### 4. `tools/` (Document Analysis)
- **`nlp_tools.py`**: Contains custom natural language processing tools built to analyze documents post-retrieval. It provides functionality to calculate FOG indices (readability), classify project complexity, extract technology stacks, determine domains/topics, and summarize project problems and outcomes.

### 5. `ingest/` (Data Loading)
- **`scraper.py`**: Houses the logic used to scrape or gather the original consulting project documents and articles.
- **`chunker.py`**: Responsible for loading the raw documents and intelligently chunking the text into smaller, overlapping segments suitable for vector embedding and retrieval.

### 6. Root Configuration
- **`main.py`**: An alternative entry point.
- **`.env` & `.env.example`**: Store critical environment variables such as `DATABASE_URL` (for Postgres checkpointer) and API keys (like `HF_TOKEN`).
- **`requirements.txt`**: Lists all necessary Python dependencies (`langgraph`, `langchain`, `psycopg`, `faiss-cpu`, `rank_bm25`, etc.).

## Setup & Installation

1. **Install dependencies:**
   Ensure you have a Python virtual environment activated, then install the required packages:
   ```bash
   pip install -r requirements.txt
   pip install rank_bm25
   ```

2. **Environment Variables:**
   Create a `.env` file in the root directory and configure the following variables:
   ```env
   HF_TOKEN=your_huggingface_token
   VECTOR_STORE=faiss # or postgres
   DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/rag_db
   ```
   *(Ensure you replace `YOUR_PASSWORD` with your actual local PostgreSQL password. If the connection fails, the script will gracefully fall back to an in-memory saver).*

3. **Run the Agent:**
   Execute the main graph script to run a query through the pipeline:
   ```bash
   python graph/graph.py
   ```

## Limitations & Deferred Plans (Future Scope)
Due to strict time constraints during development, the following planned features were scoped out and deferred for future development:
1. **Chatbot Interface:** The system currently runs as a programmatic script without a conversational web user interface (like Streamlit, Gradio, or a React frontend).
2. **Virtual / Cloud Databases:** The project relies entirely on a local PostgreSQL instance and local FAISS indexes rather than managed cloud vector databases (e.g., Pinecone, Weaviate, or managed cloud SQL).
3. **API Deployment:** Future plans include wrapping the LangGraph agent in a FastAPI server to serve it as an independent backend microservice.
