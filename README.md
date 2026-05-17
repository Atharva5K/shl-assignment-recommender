# SHL Conversational Assessment Recommender

A conversational AI agent that recommends SHL assessments through natural language dialogue.

## Architecture

**Retrieval-first, grounded RAG application** — not a generic chatbot.

```
Conversation History → Intent Reconstruction → Query Type Detection
    → Semantic Retrieval + Metadata Filtering → Reranking
    → Grounded LLM Generation → Strict JSON Response
```

### Tech Stack
| Component | Choice | Reason |
|-----------|--------|--------|
| Backend | FastAPI | Lightweight, async, evaluator-friendly |
| Embeddings | all-MiniLM-L6-v2 | Free, fast CPU inference, good semantic quality |
| Vector DB | FAISS (local) | No external service, no cost, fast similarity search |
| LLM | Llama 3.1 8B via Groq | Free tier, low latency, sufficient for grounded generation |
| Hosting | Render free tier | Easy deploy, public endpoint |

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set API key
```bash
cp .env.example .env
# Edit .env and add your Groq API key
```

### 3. Build FAISS index
```bash
python -m scripts.build_index
```

### 4. Run locally
```bash
uvicorn app.main:app --reload --port 8000
```

### 5. Test
```bash
# Health check
curl http://localhost:8000/health

# Chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "I am hiring a Java developer"}]}'
```

## API Endpoints

### GET /health
Returns `{"status": "ok"}` with HTTP 200.

### POST /chat
Stateless endpoint. Every call carries full conversation history.

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "Hiring a Java developer who works with stakeholders"},
    {"role": "assistant", "content": "Sure. What is seniority level?"},
    {"role": "user", "content": "Mid-level, around 4 years"}
  ]
}
```

**Response:**
```json
{
  "reply": "Here are 5 assessments that fit a mid-level Java dev...",
  "recommendations": [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/...", "test_type": "K"}
  ],
  "end_of_conversation": false
}
```

## Conversational Behaviors
- **Clarify**: Asks focused questions for vague queries
- **Recommend**: Returns 1-10 grounded assessments
- **Refine**: Updates shortlist without restarting
- **Compare**: Grounded comparisons from catalog data
- **Refuse**: Stays in scope, rejects off-topic/injection

## Design Decisions
1. **Retrieval-first**: Recommendations come from FAISS semantic search, never LLM memory
2. **Hybrid filtering**: Semantic similarity + metadata constraints (job level, duration, etc.)
3. **Reranking**: Multi-signal scoring (semantic + skill overlap + metadata match)
4. **Grounded generation**: LLM explains retrieved candidates, doesn't invent them
5. **Strict validation**: All URLs/names validated against catalog before returning

## Deployment (Render)
```bash
git push  # render.yaml auto-deploys
```

Or manual: Connect GitHub repo on render.com, set GROQ_API_KEY env var.
