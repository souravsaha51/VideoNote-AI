"""
modules/prompts.py
All PromptTemplate definitions for NoteTube.
"""

from langchain_core.prompts import PromptTemplate


# ── Summary Prompts ────────────────────────────────────────────────────────────

SUMMARY_SHORT_PROMPT = PromptTemplate(
    input_variables=["transcript"],
    template="""You are an expert content summarizer. Read the following YouTube video transcript and write a concise summary in 3–5 sentences. Capture the main topic, key message, and the most important takeaway.

TRANSCRIPT:
{transcript}

CONCISE SUMMARY:""",
)

SUMMARY_DETAILED_PROMPT = PromptTemplate(
    input_variables=["transcript"],
    template="""You are an expert content analyst. Read the following YouTube video transcript and produce a detailed, structured summary.

Format your response with these sections:
## Overview
(2–3 sentence description of the video)

## Main Topics Covered
(Bullet list of the key topics discussed)

## Key Points
(Detailed bullet points of the most important information)

## Conclusion
(What the video concludes or recommends)

TRANSCRIPT:
{transcript}

DETAILED SUMMARY:""",
)

# ── Insights Prompt ────────────────────────────────────────────────────────────

INSIGHTS_PROMPT = PromptTemplate(
    input_variables=["transcript"],
    template="""You are a learning specialist. Analyze the following YouTube video transcript and extract the 5–8 most valuable insights, lessons, or actionable takeaways.

Format each insight as a short, punchy bullet point starting with an emoji. Be specific and practical.

TRANSCRIPT:
{transcript}

KEY INSIGHTS:""",
)

# ── Timestamp Highlights Prompt ────────────────────────────────────────────────

TIMESTAMPS_PROMPT = PromptTemplate(
    input_variables=["transcript"],
    template="""You are a video editor assistant. Analyze the following YouTube video transcript and identify the 5–8 most important moments worth bookmarking.

For each moment, provide:
- A short descriptive label (5–8 words)
- The approximate position in the video (early, middle, or late)
- Why this moment is important

Transcript:
{transcript}

IMPORTANT MOMENTS:""",
)

# ── RAG Chat Prompt ────────────────────────────────────────────────────────────

CHAT_PROMPT = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template="""You are an intelligent video assistant for NoteTube. Answer the user's question based ONLY on the provided video transcript context.

Rules:
- Answer only from the context below; do not hallucinate
- If the answer is not in the context, say: "I couldn't find information about that in this video."
- Be concise but thorough
- Reference specific points from the video when possible
- If relevant, mention where in the video (early, middle, late) the information appears

PREVIOUS CONVERSATION:
{chat_history}

VIDEO CONTEXT:
{context}

USER QUESTION: {question}

ANSWER:""",
)

# ── Learning Notes Prompt ──────────────────────────────────────────────────────

NOTES_PROMPT = PromptTemplate(
    input_variables=["transcript"],
    template="""You are a study notes creator. Transform the following YouTube video transcript into well-structured learning notes.

Format:
## 📚 Topic
(Main topic of the video)

## 🎯 Learning Objectives
(What you will learn)

## 📝 Notes
(Organized notes with sub-headings)

## 💡 Key Terms
(Important terms and their definitions)

## ✅ Summary
(3-sentence takeaway)

TRANSCRIPT:
{transcript}

LEARNING NOTES:""",
)
