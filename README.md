# VideoNote AI – AI Powered YouTube Assistant

This is an AI-powered YouTube video summarization and conversational assistant platform built using Streamlit, LangChain, LangGraph, Gemini 2.5 Flash Lite, FAISS, and advanced Retrieval-Augmented Generation (RAG) architecture.



# Description

The VideoNote AI system enables users to summarize long YouTube videos instantly, interact with video content using conversational AI, and retrieve context-aware insights using semantic search and advanced retrieval pipelines.

The application extracts YouTube transcripts, performs semantic chunking, generates embeddings using Google Generative AI Embeddings, stores vectors in FAISS, and retrieves relevant contextual information using hybrid retrieval and RAG workflows.

The platform is designed for:
- Podcasts and interviews
- AI and technology videos
- Business and finance discussions
- Educational lectures
- News and research content



# Features

## AI Video Summarization
- Generate concise and detailed summaries from YouTube videos.
- Extract key insights and learning points.

## Conversational AI Chat
- Ask contextual questions directly from video transcripts.
- Generate semantic answers using Retrieval-Augmented Generation (RAG).

## Advanced Retrieval Pipeline
- Hybrid retrieval using vector + keyword search.
- Query rewriting and multi-query retrieval.
- MMR retrieval and contextual compression.

## Timestamp-Based Navigation
- Navigate directly to important video moments.
- Retrieve timestamp-aware contextual responses.

## LangGraph Orchestration
- Multi-step AI workflow orchestration.
- Dynamic routing between summarization and retrieval agents.

## LangSmith Observability
- Monitor token usage, latency, retrieval quality, and chain execution.
- Debug and trace multi-stage LLM workflows.

## Evaluation Pipeline
- Evaluate retrieval and generation quality using RAGAS metrics:
  - Faithfulness
  - Context Precision
  - Context Recall
  - Answer Relevancy

## Modular Scalable Architecture
- Separate ingestion, indexing, retrieval, and generation layers.
- Production-ready modular backend design.



# Technologies Used

## Frontend
- Streamlit

## Backend
- Python
- FastAPI

## AI Frameworks
- LangChain
- LangGraph
- LangSmith

## LLM & Embeddings
- Gemini 2.5 Flash Lite
- Google Generative AI Embeddings

## Vector Database
- FAISS
- ChromaDB

## Retrieval & Evaluation
- RAGAS
- BM25 Retrieval
- Cross-Encoder Re-ranking

## APIs & Utilities
- YouTube Transcript API
- python-dotenv
- sentence-transformers



# System Architecture

```text
User Input
   ↓
Streamlit Frontend
   ↓
LangGraph Orchestrator
   ↓
Transcript Extraction
   ↓
Document Cleaning
   ↓
Semantic Chunking
   ↓
Embedding Generation
   ↓
FAISS Vector Storage
   ↓
Hybrid Retrieval
   ↓
Re-ranking + Compression
   ↓
Prompt Engineering
   ↓
Gemini Generation
   ↓
Guardrails + Memory
   ↓
Final AI Response
```



# Workflow

## Step 1 — Transcript Extraction
- Extract transcripts using YouTube Transcript API.
- Handle multilingual transcript support.

## Step 2 — Document Ingestion
- Clean transcript text.
- Normalize contextual information.

## Step 3 — Semantic Chunking
- Split transcripts using RecursiveCharacterTextSplitter.
- Preserve semantic continuity and timestamps.

## Step 4 — Embedding Generation
- Generate embeddings using Google Generative AI Embeddings.

## Step 5 — Vector Indexing
- Store embeddings in FAISS vector database.
- Enable semantic similarity retrieval.

## Step 6 — Advanced Retrieval
- Query rewriting
- Multi-query retrieval
- Hybrid retrieval
- MMR retrieval
- Re-ranking
- Contextual compression

## Step 7 — AI Generation
- Generate summaries, insights, timestamp answers, and conversational responses using Gemini 2.5 Flash Lite.

## Step 8 — Evaluation & Observability
- Monitor workflows using LangSmith.
- Evaluate response quality using RAGAS.



# Installation

## 1. Clone Repository

```bash
git clone <repository_url>
cd VideoNote AI
```

## 2. Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```



# Required Dependencies

```bash
pip install streamlit
pip install langchain
pip install langchain-community
pip install langchain-core
pip install langchain-google-genai
pip install langgraph
pip install langsmith
pip install python-dotenv
pip install youtube-transcript-api
pip install faiss-cpu
pip install chromadb
pip install sentence-transformers
pip install rank-bm25
pip install ragas
pip install fastapi
pip install uvicorn
```



# Environment Variables

Create a `.env` file:

```env
GEMINI_API_KEY=your_gemini_api_key

LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=VideoNote AI
```



# Run the Application

## Streamlit Frontend

```bash
streamlit run app.py
```



# Project Structure

```text
VideoNote AI/
│
├── app.py
├── requirements.txt
├── .env
│
├── frontend/
├── orchestrator/
├── ingestion/
├── indexing/
├── retrieval/
├── generation/
├── agents/
├── evaluation/
├── memory/
└── utils/
```



# Advanced Features

- Conversational AI over video transcripts
- Timestamp-aware retrieval
- Semantic video exploration
- Multi-agent orchestration
- Retrieval evaluation pipeline
- Context-aware summarization
- Hybrid semantic retrieval
- Hallucination reduction guardrails
- Memory-enabled conversational workflows



# Future Improvements

- Multi-video comparison
- AI-generated quizzes
- Flashcard generation
- Voice-based summarization
- PDF export support
- Multi-language translation
- User authentication
- Cloud deployment
- Real-time streaming responses



# Performance Optimization

- FAISS vector indexing for fast retrieval
- Contextual compression to reduce token usage
- Re-ranking for improved retrieval accuracy
- Modular microservice-ready architecture
- LangSmith monitoring for debugging and observability



# Use Cases

- Educational content summarization
- AI research videos
- Podcast learning
- Business and finance insights
- Technical interview preparation
- News and trend analysis



# Conclusion

The VideoNote AI project demonstrates the implementation of a production-grade AI-powered YouTube knowledge assistant using advanced RAG architecture, LangGraph orchestration, semantic retrieval, and conversational AI workflows.

The platform combines modern LLM engineering techniques including:
- hybrid retrieval
- vector databases
- contextual compression
- agentic workflows
- observability pipelines
- evaluation frameworks

to deliver scalable, intelligent, and context-aware video understanding systems for educational, research, and productivity use cases.
