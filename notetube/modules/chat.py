"""
modules/chat.py
RAG-based conversational chat for NoteTube.
Implements the same chain pattern as rag_project.ipynb:
  RunnableParallel → prompt | llm | StrOutputParser
"""

from typing import List, Dict, Tuple
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from modules.vectorstore import get_retriever
from modules.prompts import CHAT_PROMPT


def _format_docs(docs) -> str:
    """Join retrieved document contents with double newlines — matches notebook pattern."""
    return "\n\n".join(doc.page_content for doc in docs)


def format_chat_history(history: List[Dict[str, str]]) -> str:
    """
    Format a list of chat turns into a plain-text string for the prompt.

    Args:
        history: List of {'role': 'user'|'assistant', 'content': str} dicts.

    Returns:
        Multi-line string of formatted chat history (last 6 turns).
    """
    if not history:
        return "No previous conversation."

    lines = []
    for turn in history[-6:]:  # Keep last 6 turns for context window efficiency
        role = "You" if turn["role"] == "user" else "Assistant"
        lines.append(f"{role}: {turn['content']}")
    return "\n".join(lines)


def answer_question(
    question: str,
    vectorstore: FAISS,
    llm: ChatGoogleGenerativeAI,
    chat_history: List[Dict[str, str]] = None,
    k: int = 4,
) -> Tuple[str, List]:
    """
    Answer a question about the video using RAG.

    Implements the notebook's chain pattern:
        RunnableParallel(context: retriever|format_docs, question: passthrough)
        | prompt | llm | StrOutputParser

    Args:
        question: User's question string.
        vectorstore: Populated FAISS index with transcript chunks.
        llm: ChatGoogleGenerativeAI instance.
        chat_history: Previous conversation turns (optional).
        k: Number of context chunks to retrieve.

    Returns:
        Tuple of (answer_string, retrieved_docs).
    """
    if chat_history is None:
        chat_history = []

    retriever = get_retriever(vectorstore, k=k)

    # Retrieve docs for returning alongside the answer
    retrieved_docs = retriever.invoke(question)

    # Build context string (notebook pattern: "\n\n".join of page_content)
    context_text = _format_docs(retrieved_docs)
    history_str = format_chat_history(chat_history)

    # Use the notebook-style chain: parallel_chain | prompt | llm | parser
    parser = StrOutputParser()

    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(_format_docs),
        "question": RunnablePassthrough(),
        "chat_history": RunnableLambda(lambda _: history_str),
    })

    main_chain = parallel_chain | CHAT_PROMPT | llm | parser

    answer = main_chain.invoke(question)
    return answer.strip(), retrieved_docs
