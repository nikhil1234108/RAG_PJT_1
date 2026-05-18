from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def build_chat_prompt(memory_context: str = "", retrieved_context: str = "", voice_mode: bool = False):
    system_msg = (
        "You are an expert AI assistant for Blackcoffer's consulting projects.\n"
        "Your goal is to answer the user's questions based on the provided context.\n\n"
    )

    if voice_mode:
        system_msg += "VOICE MODE ACTIVE: Keep your response concise, conversational, and completely free of markdown formatting (like asterisks, hash symbols, or bullet points) since it will be spoken out loud.\n\n"
    else:
        system_msg += "Provide detailed, structured answers.\n\n"

    system_msg += "--- Retrieved Article Context ---\n"
    system_msg += f"{retrieved_context if retrieved_context else 'No specific articles retrieved.'}\n\n"

    system_msg += "--- Older Conversation Summary ---\n"
    system_msg += f"{memory_context if memory_context else 'No prior memory.'}\n"

    return ChatPromptTemplate.from_messages([
        ("system", system_msg),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
