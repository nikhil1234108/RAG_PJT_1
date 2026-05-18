"""
memory.py
Conversation memory management.
Combines PostgreSQL long-term memory with
LangChain in-memory short-term buffer for current turn.

Two-layer memory:
  Layer 1 — PostgreSQL (persistent):
    Full conversation history across sessions
    Survives restarts, page reloads, crashes

  Layer 2 — LangChain buffer (in-memory):
    Current session messages as LangChain message objects
    Used directly by the ChatPromptTemplate history slot
"""

from typing import List, Dict
from langchain_classic.memory import ConversationSummaryBufferMemory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

def build_langchain_memory(llm, max_token_limit:int=2000):
    """
    ConversationSummaryBufferMemory:
      - Keeps recent messages as-is (buffer)
      - Summarizes older messages when buffer exceeds max_token_limit
      - Best of both worlds: detail for recent, summary for old
    """

    return ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=max_token_limit,
        return_messages=True, # Critical for ChatPromptTemplate
        memory_key="history"
    )

def load_history_into_memory(memory:ConversationSummaryBufferMemory, history:List[Dict[str, str]]):
    """
    Loads PostgreSQL history into LangChain memory object.
    Called at session start to restore previous conversation.
    """
    if not history:
        return memory
    
    # Manually append messages to the internal buffer
    for msg in history:
        if msg["role"] == "user":
            memory.chat_memory.add_user_message(msg["content"])
        elif msg["role"] == "assistant":
            memory.chat_memory.add_ai_message(msg["content"])
        elif msg["role"] == "system":
            # Some memories support system messages, otherwise skip
            pass
    return memory

def history_to_langchain_messages(history:List[Dict[str,str]]) -> List[BaseMessage]:
    """
    Lightweight converter if not using the full buffer memory.
    """
    messages = []
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    return messages
