<![CDATA[<div align="center">

# 🧠 LocalMind AI

### A Fully Offline Agentic AI Assistant for Windows

**Zero Cloud Dependency · DirectML GPU Acceleration · Self-Healing Retrieval**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-DirectML-7B2D8B?style=for-the-badge&logo=onnx&logoColor=white)](https://onnxruntime.ai)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-000000?style=for-the-badge&logo=ollama&logoColor=white)](https://ollama.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)

---

*LocalMind is a lightweight agentic AI assistant that runs **entirely on your machine**. It accepts natural language queries, classifies intent using a locally hosted LLM, retrieves context through an ONNX-powered semantic search pipeline, and returns grounded answers — all without sending a single byte to the cloud.*

</div>

---

## 🏗️ System Architecture

```mermaid
graph TB
    subgraph USER["👤 User Interface"]
        A["Natural Language Query"]
    end

    subgraph API["⚡ FastAPI Layer"]
        B["POST /ask"]
        C["POST /ingest"]
        D["GET /health"]
    end

    subgraph AGENT["🤖 LangGraph Agent"]
        E["Intent Classifier"]
        F["Context Retriever"]
        G["Answer Generator"]
        H["Self-Critique Engine"]
        I["Query Refiner"]
        J["Direct Responder"]
    end

    subgraph ML["🧮 ML Services"]
        K["ONNX Runtime\n+ DirectML GPU"]
        L["FAISS Vector Index"]
        M["Ollama LLM\n(Mistral 7B)"]
    end

    subgraph DATA["💾 Local Storage"]
        N["ONNX Model\n(all-MiniLM-L6-v2)"]
        O["Document Store\n(Markdown/Text)"]
        P["FAISS Index\n+ Metadata"]
    end

    A --> B
    B --> E
    E -->|"qa / summarize"| F
    E -->|"chitchat"| J
    F --> G
    G --> H
    H -->|"confidence < 0.6"| I
    H -->|"confidence ≥ 0.6"| B
    I --> F
    J --> B

    F <--> K
    F <--> L
    E <--> M
    G <--> M
    H <--> M
    I <--> M
    J <--> M

    K <--> N
    L <--> P
    C --> O

    style USER fill:#1a1a2e,stroke:#e94560,color:#fff
    style API fill:#16213e,stroke:#0f3460,color:#fff
    style AGENT fill:#0f3460,stroke:#533483,color:#fff
    style ML fill:#533483,stroke:#e94560,color:#fff
    style DATA fill:#1a1a2e,stroke:#533483,color:#fff
```

---

## 🔄 Agent Workflow — The Self-Critique Loop

This is the core intelligence of LocalMind. The agent doesn't just generate answers — it **judges its own output** and iteratively improves until confidence is high enough.

```mermaid
flowchart TD
    START(["🚀 User Query Received"]) --> CLASSIFY

    CLASSIFY{"🏷️ Intent Classification\n(Ollama)"}
    CLASSIFY -->|"💬 chitchat"| DIRECT["💬 Direct LLM Response"]
    CLASSIFY -->|"❓ qa"| EMBED
    CLASSIFY -->|"📝 summarize"| EMBED

    EMBED["🔢 Embed Query\n(ONNX + DirectML)"]
    EMBED --> SEARCH["🔍 FAISS Semantic Search\n(Top-K Retrieval)"]
    SEARCH --> GENERATE["✍️ Generate Answer\n(Ollama + Contexts)"]

    GENERATE --> CRITIQUE{"🧐 Self-Critique\nScore Answer"}
    CRITIQUE --> SCORE{"📊 Confidence ≥ 0.6?"}

    SCORE -->|"✅ Yes"| ACCEPT["✅ Return Grounded Answer"]
    SCORE -->|"❌ No"| RETRY{"🔄 Retries < 2?"}

    RETRY -->|"Yes"| REFINE["🔧 Refine Search Query\n(Based on Critique Feedback)"]
    RETRY -->|"No"| LOWCONF["⚠️ Return Low-Confidence Answer"]

    REFINE --> EMBED

    DIRECT --> DONE(["📤 Response Sent"])
    ACCEPT --> DONE
    LOWCONF --> DONE

    style START fill:#00b894,stroke:#00b894,color:#fff
    style DONE fill:#00b894,stroke:#00b894,color:#fff
    style CLASSIFY fill:#6c5ce7,stroke:#6c5ce7,color:#fff
    style EMBED fill:#0984e3,stroke:#0984e3,color:#fff
    style SEARCH fill:#0984e3,stroke:#0984e3,color:#fff
    style GENERATE fill:#e17055,stroke:#e17055,color:#fff
    style CRITIQUE fill:#fdcb6e,stroke:#fdcb6e,color:#000
    style SCORE fill:#fdcb6e,stroke:#fdcb6e,color:#000
    style ACCEPT fill:#00b894,stroke:#00b894,color:#fff
    style REFINE fill:#e84393,stroke:#e84393,color:#fff
    style RETRY fill:#d63031,stroke:#d63031,color:#fff
    style LOWCONF fill:#d63031,stroke:#d63031,color:#fff
    style DIRECT fill:#74b9ff,stroke:#74b9ff,color:#000
```

---

## 🧮 Embedding & Retrieval Pipeline

```mermaid
flowchart LR
    subgraph EXPORT["📦 One-Time Setup"]
        A["PyTorch\nSentence-Transformer"] -->|"Optimum Export"| B["ONNX Model\n(all-MiniLM-L6-v2)"]
        B -->|"Optional"| C["INT8 Quantized\nONNX Model"]
    end

    subgraph INGEST["📥 Document Ingestion"]
        D["Markdown / Text\nDocuments"] -->|"Recursive Chunking\n(512 chars, 64 overlap)"| E["Text Chunks"]
        E -->|"ONNX Embedding"| F["384-dim Vectors"]
        F -->|"Index"| G["FAISS IndexFlatIP"]
    end

    subgraph QUERY["🔍 Query Time"]
        H["User Query"] -->|"ONNX Embedding"| I["384-dim Query Vector"]
        I -->|"Inner Product Search"| G
        G -->|"Top-K Results"| J["Ranked Chunks\n+ Similarity Scores"]
    end

    style EXPORT fill:#2d3436,stroke:#636e72,color:#fff
    style INGEST fill:#2d3436,stroke:#636e72,color:#fff
    style QUERY fill:#2d3436,stroke:#636e72,color:#fff
```

---

## 📊 Self-Critique Scoring Rubric

The agent evaluates every generated answer before returning it:

```mermaid
pie title Confidence Score Composition
    "Relevance (40%)" : 40
    "Groundedness (40%)" : 40
    "Completeness (20%)" : 20
```

| Dimension | Weight | Score Range | Description |
|:---|:---:|:---:|:---|
| **Relevance** | 40% | 0.0 – 1.0 | Does the answer directly address the question? |
| **Groundedness** | 40% | 0.0 – 1.0 | Is every claim supported by retrieved context? |
| **Completeness** | 20% | 0.0 – 1.0 | Does it cover all answerable aspects? |

> **Threshold:** If `weighted_score < 0.6`, the agent refines its search query using the critique feedback and re-retrieves (up to 2 retries).

---

## 🚀 Quick Start

### Option A: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/Nagendramanthena/localmind-ai.git
cd localmind-ai

# Start everything (Ollama + Agent API)
docker-compose up --build

# Test it!
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is deep learning?"}'
```

### Option B: Native Windows Setup

```mermaid
flowchart LR
    A["1️⃣ Install\nPython 3.11+\n& Ollama"] --> B["2️⃣ Export\nONNX Model"]
    B --> C["3️⃣ Build\nFAISS Index"]
    C --> D["4️⃣ Start\nAPI Server"]
    D --> E["5️⃣ Query\nvia curl/browser"]

    style A fill:#00b894,stroke:#00b894,color:#fff
    style B fill:#0984e3,stroke:#0984e3,color:#fff
    style C fill:#6c5ce7,stroke:#6c5ce7,color:#fff
    style D fill:#e17055,stroke:#e17055,color:#fff
    style E fill:#fdcb6e,stroke:#fdcb6e,color:#000
```

#### Step 1: Install Dependencies

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install runtime dependencies
pip install -r requirements.txt

# For DirectML GPU acceleration (Windows only):
pip uninstall onnxruntime
pip install onnxruntime-directml

# Install and start Ollama, then pull the model
ollama pull mistral:7b
```

#### Step 2: Export Embedding Model to ONNX

```bash
pip install -r requirements-export.txt
python scripts/export_model.py            # Standard export
python scripts/export_model.py --quantize  # With INT8 quantization
```

#### Step 3: Build the Search Index

```bash
python scripts/build_index.py                       # Index sample docs
python scripts/build_index.py --docs /path/to/docs   # Index your own docs
```

#### Step 4: Start the Server

```bash
copy .env.example .env    # Edit configuration as needed
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## 📡 API Reference

### `POST /ask` — Query the Agent

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?", "top_k": 5}'
```

<details>
<summary>📋 Example Response</summary>

```json
{
  "answer": "Machine learning (ML) is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. According to the knowledge base, it focuses on developing programs that can access data and learn for themselves. There are three main types: Supervised Learning (trained on labeled data), Unsupervised Learning (finds patterns in unlabeled data), and Reinforcement Learning (learns through rewards).",
  "intent": "qa",
  "confidence": 0.87,
  "sources": ["machine_learning.md"],
  "retries_used": 0,
  "critique": {
    "relevance": 0.92,
    "groundedness": 0.88,
    "completeness": 0.78,
    "feedback": "Answer comprehensively covers the definition and types. Could mention deep learning as a subset."
  }
}
```

</details>

### `POST /ingest` — Add Documents at Runtime

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Your new document content..."], "metadata": [{"source": "new_doc.md"}]}'
```

### `GET /health` — Health Check

```bash
curl http://localhost:8000/health
# {"status": "healthy", "ollama": true, "onnx": true, "index_size": 42}
```

### `GET /stats` — Index Statistics

```bash
curl http://localhost:8000/stats
# {"index_size": 42, "embedding_dimension": 384, "ollama_model": "mistral:7b", ...}
```

---

## 📁 Project Structure

```
localmind-ai/
├── 📂 app/
│   ├── main.py                    # FastAPI entry point & endpoints
│   ├── config.py                  # Centralized settings (from .env)
│   ├── 📂 agent/
│   │   ├── graph.py               # LangGraph StateGraph wiring
│   │   ├── nodes.py               # All graph node functions
│   │   └── prompts.py             # Prompt templates (easily tunable)
│   ├── 📂 models/
│   │   ├── schemas.py             # Pydantic API request/response models
│   │   └── state.py               # LangGraph shared state definition
│   ├── 📂 services/
│   │   ├── embedding_service.py   # ONNX Runtime inference (DirectML)
│   │   ├── llm_service.py         # Ollama LLM wrapper
│   │   └── retrieval_service.py   # FAISS vector search + persistence
│   └── 📂 utils/
│       └── logging.py             # Structured logging (structlog)
├── 📂 scripts/
│   ├── export_model.py            # PyTorch → ONNX export
│   └── build_index.py             # Document → FAISS ingestion
├── 📂 data/
│   ├── 📂 documents/              # Knowledge base (markdown/text files)
│   └── 📂 index/                  # Persisted FAISS index + metadata
├── 📂 models/
│   └── 📂 onnx/                   # Exported ONNX model + tokenizer
├── Dockerfile                     # Multi-stage build (no PyTorch in runtime!)
├── docker-compose.yml             # Ollama sidecar + Agent API
├── requirements.txt               # Runtime Python dependencies
├── requirements-export.txt        # Export-only deps (PyTorch, Optimum)
├── .env.example                   # Configuration template
└── README.md
```

---

## ⚙️ Configuration

All settings are controlled via environment variables or a `.env` file:

| Variable | Default | Description |
|:---|:---|:---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `mistral:7b` | LLM model for intent/generation/critique |
| `EMBEDDING_MODEL_PATH` | `./models/onnx` | Path to exported ONNX model |
| `ONNX_PROVIDERS` | `auto` | `auto` / `DmlExecutionProvider` / `CPUExecutionProvider` |
| `FAISS_INDEX_PATH` | `./data/index` | FAISS index persistence directory |
| `TOP_K` | `5` | Context chunks per retrieval |
| `CONFIDENCE_THRESHOLD` | `0.6` | Self-critique pass/fail threshold |
| `MAX_RETRIES` | `2` | Max re-retrieval attempts on low confidence |
| `CHUNK_SIZE` | `512` | Document chunk size (characters) |
| `CHUNK_OVERLAP` | `64` | Overlap between consecutive chunks |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

---

## 🖥️ DirectML GPU Acceleration (Windows)

LocalMind uses ONNX Runtime with **DirectML** for hardware-agnostic GPU acceleration on Windows:

```mermaid
flowchart LR
    A["ONNX Model"] --> B["ONNX Runtime"]
    B --> C{"DirectML\nExecution Provider"}
    C --> D["🟢 NVIDIA GPU"]
    C --> E["🔴 AMD GPU"]
    C --> F["🔵 Intel GPU"]
    C --> G["🟡 Qualcomm GPU"]

    style C fill:#7B2D8B,stroke:#7B2D8B,color:#fff
    style D fill:#76b900,stroke:#76b900,color:#fff
    style E fill:#ed1c24,stroke:#ed1c24,color:#fff
    style F fill:#0071c5,stroke:#0071c5,color:#fff
    style G fill:#3253dc,stroke:#3253dc,color:#fff
```

**Setup:**
```bash
pip uninstall onnxruntime
pip install onnxruntime-directml
```

> Works with **any DirectX 12 capable GPU**. Set `ONNX_PROVIDERS=auto` and LocalMind auto-detects DirectML on Windows, falling back to CPU elsewhere.

---

## 🐳 Docker Architecture

```mermaid
flowchart TB
    subgraph COMPOSE["docker-compose.yml"]
        subgraph OLLAMA["🦙 Ollama Container"]
            O1["Ollama Server\n:11434"]
            O2["Model Storage\n(Volume)"]
        end

        subgraph INIT["🔄 Init Container"]
            I1["ollama pull mistral:7b"]
        end

        subgraph AGENT["🤖 Agent Container"]
            A1["FastAPI + Uvicorn\n:8000"]
            A2["ONNX Model\n(baked in)"]
            A3["FAISS Index\n(mounted)"]
        end
    end

    INIT -->|"depends_on"| OLLAMA
    AGENT -->|"OLLAMA_BASE_URL"| OLLAMA

    style COMPOSE fill:#1a1a2e,stroke:#2496ED,color:#fff
    style OLLAMA fill:#16213e,stroke:#0f3460,color:#fff
    style INIT fill:#16213e,stroke:#0f3460,color:#fff
    style AGENT fill:#16213e,stroke:#0f3460,color:#fff
```

The Dockerfile uses a **multi-stage build**:
- **Stage 1 (builder):** Includes PyTorch + Optimum → exports ONNX model
- **Stage 2 (runtime):** Lean image (~1.2 GB) with only ONNX Runtime, no PyTorch

---

## 🔧 Troubleshooting

| Issue | Solution |
|:---|:---|
| `No .onnx file found` | Run `python scripts/export_model.py` first |
| `Connection refused` to Ollama | Start Ollama: `ollama serve` |
| Empty search results | Build the index: `python scripts/build_index.py` |
| DirectML not detected | Falls back to CPU automatically; check DirectX 12 support |
| Out of GPU memory | Use a smaller LLM: `ollama pull phi3:mini` |
| Slow first request | Normal — ONNX graph compilation on first inference |

---

## 📄 License

MIT — use it however you like.

---

<div align="center">

**Built with ❤️ for fully offline AI**

*Your data never leaves your machine.*

</div>
]]>
