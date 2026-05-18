"""
chat_engine.py
Core chat engine — orchestrates:
  - PostgreSQL session + message persistence
  - Memory injection into system prompt
  - RAG retrieval for grounding
  - LLM generation with full prompt architecture
  - Response post-processing
"""

import os
import sys

# Add the project root to sys.path so that 'chatbot' can be imported when running this file directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional
from chatbot.session import(
    create_session, get_session,get_history,
    save_message, format_memory_context, init_db
)
import numpy as np
from chatbot.memory import history_to_langchain_messages
from chatbot.prompt import build_chat_prompt
from chains.rag_chain import get_llm

class chatengine:
    """
    Stateful chat engine with session_id-based memory.

    Usage:
      engine = ChatEngine(vectorstore)
      session_id = engine.new_session()
      response = engine.chat(session_id, "Tell me about LangChain projects")
      response = engine.chat(session_id, "Which one is most complex?")
    """

    def __init__(self, vectorstore, voice_mode:bool=False):
        self.vectorstore = vectorstore
        self.voice_mode = voice_mode
        self.llm = get_llm()
        init_db()

    def new_session(self, mode:str = "text") -> str:
        """Creates new PostgreSQL session. Returns session_id."""

        session_id = create_session(mode=mode)
        print(f"[ChatEngine] New session: {session_id}")
        return session_id

    
    def _retrieve_context(self,query:str) ->str:
        
        """Cluster-based retrieval only.
        Finds nearest cluster first, then searches cluster chunks.
        No full FAISS search.
        """
        try:
            from vectorstore.clustering import load_cluster_results
            from vectorstore.vector_store import get_embeddings
            
            data = load_cluster_results()
            
            if not data or "kmeans" not in data:
                return ""
            
            centroids    = np.array(data["kmeans"]["centroids"], dtype=np.float32)
            embeddings   = get_embeddings()
            query_vector = np.array(embeddings.embed_query(query),
            dtype=np.float32
            )
            distances       = np.linalg.norm(centroids - query_vector, axis=1)
            nearest_cluster = int(np.argmin(distances))
            cluster_key     = f"cluster_{nearest_cluster}"
            cluster_ids     = data["kmeans"]["clusters"].get(cluster_key, [])

            if not cluster_ids:
                return ""

            print(f"[ChatEngine] Cluster: {cluster_key} "f"({len(cluster_ids)} articles)")

            from chains.rag_chain import build_rag_chain
            chain  = build_rag_chain(self.vectorstore, cluster_ids=cluster_ids)
            result = chain.invoke({"query": query})
            docs   = result.get("source_documents", [])

            parts = []
            for doc in docs:
                url_id = doc.metadata.get("url_id", "unknown")
                parts.append(f"[{url_id}]: {doc.page_content[:400]}")

            return "\n\n".join(parts)

        except Exception as e:
            print(f"[ChatEngine] Context retrieval error: {e}")
            return ""

    def build_response(self,session_id:str,user_input:str, retrieved_context:str) -> str:
        """
        Builds prompt with memory + context → calls LLM → returns response.
        """

        history = get_history(session_id, last_n=10)
        lc_mesages = history_to_langchain_messages(history)
        memory_ctx = format_memory_context(history)
        prompt = build_chat_prompt(
            memory_context=memory_ctx,
            retrieved_context=retrieved_context,
            voice_mode=self.voice_mode,
        ) 
        messages = prompt.format_messages(
            history = lc_mesages,
            input   = user_input,
        )
        response = self.llm.invoke(messages)
        return response.content if hasattr(response, "content") else str(response)   
    
    def chat(self, session_id:str, user_input:str, source: bool = True) -> dict:
        """
        Main chat method. Full pipeline:
          1. Validate session
          2. Retrieve RAG context
          3. Inject memory + context into prompt
          4. Generate LLM response
          5. Save user + assistant messages to PostgreSQL
          6. Return response + metadata
        """
        session = get_session(session_id)

        if not session:
            raise ValueError(f"No session found for id: {session_id}")

        # Retrieve context
        retrieved_context = self._retrieve_context(user_input)

        # Generate response
        response_text = self.build_response(
            session_id, user_input, retrieved_context
        )

        save_message(session_id, "user", user_input)
        save_message(session_id,"assistant", response_text, metadata = {
            "retrieved_context":retrieved_context[:200],
            "voice_mode":self.voice_mode,
        })

        return {
            "session_id": session_id,
            "response": response_text,
            "sources": self._extract_source_ids(retrieved_context) if retrieved_context else [],
        }
    def _extract_source_ids(self, context: str) -> list:
        """Extracts article IDs from retrieved context string."""
        import re
        return list(set(re.findall(r'\[([a-zA-Z0-9_]+)\]:', context)))

    def get_history(self, session_id: str, last_n: int = 20) -> list:
        """Returns conversation history for a session."""
        return get_history(session_id, last_n=last_n)


if __name__ == "__main__":
    from vectorstore.vector_store import load_vectorstore
    vectorstore = load_vectorstore()
    engine = chatengine(vectorstore)
    session_id = engine.new_session()
    print(f"[ChatEngine] New session: {session_id}")
    
