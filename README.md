# Advanced Agentic RAG Pipeline with Voice & NLP Analytics

## Overview
This project implements an intelligent, agent-based Retrieval-Augmented Generation (RAG) pipeline backed by a **FastAPI** server. It analyzes a corpus of consulting projects, retrieves highly relevant context using a hybrid retrieval strategy, synthesizes analytical answers using Large Language Models, and maintains multi-turn conversation memory backed by PostgreSQL. The system also features a complete voice integration (STT & TTS) for conversational interactions and dedicated NLP endpoints for deep document analysis.

## Architecture & Core Features

### 1. API & Backend (FastAPI)
- Exposes robust REST endpoints for creating sessions (`/chat/session`), interacting with the AI (`/chat/message`), and running NLP metrics on specific articles (`/articles/{url_id}/metrics`).
- Fully supports cross-origin resource sharing (CORS) for front-end integration.

### 2. Conversational Memory & State
- **Two-Layer Memory System**:
  - **Long-term (PostgreSQL)**: Persists full conversation history across multiple turns and page reloads, keyed by unique `session_id`.
  - **Short-term (LangChain)**: Buffers recent messages and injects a dynamic summary of older context into the system prompt to maintain relevance without exceeding token limits.

### 3. Voice Integration (STT & TTS)
- Built-in Speech-to-Text and Text-to-Speech handlers in the `voice/` directory.
- A dedicated "Voice Mode" flag alters the LLM's system prompt to produce concise, conversational responses devoid of markdown formatting for seamless audio playback.

### 4. Hybrid Retrieval & Clustering
- **Ensemble Retrieval**: Blends dense vector search (FAISS) with sparse keyword search (BM25) for high accuracy.
- **Dynamic Clustering**: Pre-clusters documents (K-Means) and routes queries to the nearest document cluster to drastically reduce the search space and improve precision.

### 5. Advanced NLP Analysis Tools
Custom tools designed to analyze documents post-retrieval via the `/metrics` API:
- Tech Stack Extraction (spaCy NER + rules)
- Project Summary & Problem Breakdown
- Readability & FOG Index Analysis
- Complexity & Topic Classification
- Named Entity Recognition

---

## System Architecture

```mermaid
graph TD
    Client([Client App / Postman])
    
    subgraph FastAPI Backend
        SessionAPI[/chat/session/]
        ChatAPI[/chat/message/]
        MetricsAPI[/articles/url_id/metrics/]
    end
    
    Client --> SessionAPI
    Client --> ChatAPI
    Client --> MetricsAPI
    
    subgraph Chatbot Engine
        Agent[Chat Agent]
        PromptBuilder[Dynamic Prompt Builder]
        MemManager[Memory Manager]
    end
    
    ChatAPI --> Agent
    Agent --> MemManager
    Agent --> PromptBuilder
    
    subgraph Memory Layer
        Postgres[(PostgreSQL)]
        LCBuffer[LangChain Summary Buffer]
    end
    
    MemManager <--> Postgres
    MemManager <--> LCBuffer
    
    subgraph Retrieval Pipeline
        ClusterFilter[Cluster Filter]
        Ensemble[Ensemble Retriever]
        FAISS[(FAISS)]
        BM25[(BM25)]
    end
    
    Agent --> ClusterFilter
    ClusterFilter --> Ensemble
    FAISS -.-> Ensemble
    BM25 -.-> Ensemble
    
    subgraph Voice & NLP
        STT[Speech-to-Text]
        TTS[Text-to-Speech]
        NLPTools[NLP Analytics Suite]
    end
    
    Client -.-> STT
    Agent -.-> TTS
    MetricsAPI --> NLPTools
    
    PromptBuilder --> LLM((Large Language Model))
    LLM --> Agent
```

## Deep Dive into Project Structure

### 1. `API/` (Backend Server)
- **`main.py`**: The FastAPI entry point. Defines all REST endpoints, initializes the PostgreSQL database tables, and orchestrates requests between the Chat Engine and the NLP tools.

### 2. `chatbot/` (Agent & Orchestration)
- **`chat_agent.py`**: The core stateful engine. It validates sessions, retrieves context, injects memory, and streams outputs from the LLM.
- **`session.py`**: Manages PostgreSQL database connections to persist conversation histories.
- **`memory.py`**: Bridges the gap between raw PostgreSQL logs and LangChain's conversational buffer memory.
- **`prompt.py`**: Defines the production-grade system prompt architecture, applying dynamic constraints (like Voice Mode) based on the session state.

### 3. `voice/` (Audio Processing)
- **`stt.py` & `tts.py`**: Handle speech-to-text transcription and text-to-speech synthesis to allow fully conversational voice interfaces.
- **`voice_bot.py`**: A specialized bot handler for voice-first interactions.

### 4. `tools/` (NLP & Analytics)
- **`nlp_tools.py`**: Houses all document analysis algorithms (FOG index, syllable counters, entity extraction, etc.). 

### 5. `vectorstore/` & `chains/` (Retrieval)
- **`vector_store.py`**: Builds and manages the FAISS vector database.
- **`clustering.py`**: K-Means logic for document grouping.
- **`rag_chain.py`**: Binds the LLM (HuggingFace/Ollama) to the Ensemble Retriever.

---

## Setup & Installation

1. **Install dependencies:**
   Ensure you have a Python virtual environment activated, then install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables:**
   Create a `.env` file in the root directory and configure the following variables:
   ```env
   HF_TOKEN=your_huggingface_token
   DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/rag_db
   ```

3. **Run the FastAPI Server:**
   Start the application using Uvicorn:
   ```bash
   uvicorn API.main:app --reload --port 8000
   ```

4. **Interact via API:**
   - Create a session: `POST http://localhost:8000/chat/session?mode=text`
   - Send a message: `POST http://localhost:8000/chat/message`
     ```json
     {
       "session_id": "<your-session-id>",
       "message": "What technologies were used in the latest cloud project?"
     }
     ```
