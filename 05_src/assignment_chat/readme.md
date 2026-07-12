# Assignment Chat (Simplified)

This folder contains a simplified implementation for Assignment 2 in `./05_src/assignment_chat`.

## Chat Personality

The assistant is **ByteBuddy**, a concise and upbeat study coach.

## Services Implemented

### 1) API Service (`fetch_space_news`)

- Backend: Spaceflight News API (public endpoint).
- Behavior: Fetches recent articles and returns a rewritten digest in natural language.
- Requirement match: Uses an API and transforms output instead of returning raw payload text.

### 2) Semantic Query Service (`semantic_course_lookup`)

- Backend: ChromaDB with **file persistence** at `assignment_chat/chroma_store`.
- Dataset: Small local JSON knowledge base in `assignment_chat/data/knowledge_base.json`.
- Embeddings:
  - SentenceTransformer embedding function (`all-MiniLM-L6-v2`) when available.
  - Fallback: Chroma default embedding function.
- Requirement match: Semantic retrieval over local corpus with persistent Chroma collection.

### 3) Function Calling Service (`build_study_plan`)

- Backend: Deterministic local function (tool call).
- Behavior: Generates a compact multi-day study plan from topic + time constraints.
- Requirement match: Uses Function Calling as required for the third service option.

## Guardrails

Implemented in `guardrails.py`:

- Blocks prompt-injection style attempts to reveal/modify hidden instructions.
- Rejects restricted topics:
  - Cats or dogs
  - Horoscopes or zodiac signs
  - Taylor Swift

## Conversation Memory

The Gradio chat uses message history each turn and sends recent context back to the model.

- Current strategy: keep the latest 12 turns (`MAX_CONTEXT_TURNS`) to control context length.

## File Overview

- `app.py`: Gradio interface and message/history handling.
- `main.py`: LangGraph workflow with tool-calling loop.
- `services.py`: Three required services.
- `guardrails.py`: Input guardrails and refusal logic.
- `prompts.py`: System prompt and assistant style.
- `data/knowledge_base.json`: Local semantic corpus.

## Run

From `05_src`:

```bash
python -m assignment_chat.app
```

Quick validation: run `python -m assignment_chat.demo_checks` from `05_src` to verify all required service and guardrail paths.

Ensure your environment has course dependencies installed.
