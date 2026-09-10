# CSTutorAI

> An Interactive, Personalized, Source-Grounded AI Tutor for Computer Science Education and Career Preparation.

## Overview

CSTutorAI is an AI-powered educational platform designed to act as an interactive Computer Science tutor, coding mentor, DSA coach, and technical interview assistant.

The system combines a Large Language Model (LLM) with Retrieval-Augmented Generation (RAG) to provide grounded responses using curated Computer Science educational resources.

## Core Objectives

- Teach Computer Science concepts interactively
- Provide source-grounded explanations
- Reduce unsupported LLM-generated information
- Adapt explanations to student understanding
- Generate and evaluate practice questions
- Assist with programming and debugging
- Teach Data Structures and Algorithms problem-solving patterns
- Support technical interview preparation
- Track student strengths and weaknesses
- Generate personalized learning paths

## Planned Technologies

- Python
- FastAPI
- Retrieval-Augmented Generation (RAG)
- Vector Search
- Embedding Models
- Instruction-Tuned LLM
- React
- PostgreSQL

## Project Status

✅ **RAG Retrieval Foundation — Complete**

The retrieval subsystem is now reliable, tested, and end-to-end verified:

- Ingestion pipeline: documents → chunks → embeddings → persistent FAISS index
- Metadata-preserving retrieval with source attribution
- FastAPI `/query` endpoint returning source-grounded knowledge chunks
- 25 passing tests (pytest)
- Configuration via environment variables

See [PROJECT_STATUS.md](PROJECT_STATUS.md) for full details.

### Next: LLM Integration

The next phase will add LLM-powered grounded response generation, tutoring behavior, and personalized learning.

## Research Focus

The project will investigate whether retrieval-grounded and personalized LLM-based tutoring can improve the accuracy, reliability, and educational usefulness of general-purpose LLM responses for Computer Science learning.