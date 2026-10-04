# Streaming Live RAG \u2014 Samsung PRISM Hackathon (Theme 04)

This project implements the Samsung PRISM Streaming Live RAG concept, a real-time multi-agent conversational RAG engine with Server-Sent Events (SSE) streaming and real/mock Speech-to-Text (STT) capabilities. It strictly adheres to SRD-SLRAG-001 and PRD-SLRAG-001.

## Project Overview

The Streaming Live RAG Engine bridges the gap between raw real-time speech input and factually grounded, synthesized answers. It processes incoming text/transcripts continuously, evaluates them for completeness, and intelligently retrieves and fuses relevant internal corporate knowledge before synthesizing a grounded response.

### Current Architecture Flow

```text
Microphone / Browser Text Input
         \u2193
   STT Adapter Layer (Real WebSocket / Mock Replay)
         \u2193
Normalized Transcript Events
         \u2193
Retrieval Controller (WAIT / RETRIEVE / SUPPRESS)
         \u2193
Multi-Intent Decomposition
         \u2193
Hybrid Retrieval (BM25 + Vector via ChromaDB)
         \u2193
RRF Fusion / Deduplication
         \u2193
Session-Aware Synthesis + Delta Refinement
         \u2193
Grounded Answer + Citations + Telemetry (SSE)
```

## Current Status

**IMPLEMENTED & VERIFIED**
- **Streaming Transcript Replay / Mock STT:** Deterministic fallback handling.
- **Real STT Integration:** AssemblyAI live streaming via WebSocket proxy.
- **Retrieval Controller:** Working WAIT, RETRIEVE, and SUPPRESS mechanisms based on partial text stability.
- **Multi-Intent Decomposition:** Reliably separates compound queries.
- **Hybrid Retrieval:** Parallel BM25 + Vector search execution.
- **Reciprocal Rank Fusion (RRF) & Deduplication:** Accurately merges and ranks chunk evidence.
- **Session Refinement (Delta Engine):** Accurately updates prior session bounds when users refine queries (e.g., "Wait, it was international").
- **Grounded Synthesis & Citation Logic:** System correctly outputs `[DOC_ID \u00a7Section]` citations inline and avoids hallucinating facts without evidence.
- **Telemetry:** Comprehensive SSE trace events logging internal metrics (latencies, tokens, decisions).
- **G1-G6 Evaluation Harness:** Built-in programmatic benchmark runner demonstrating PRISM READY targets.

**NOT YET VERIFIED / PLANNED**
- Remote production deployment.
- Additional STT providers beyond AssemblyAI.

## Quick Start

### 1. Clone the repository

```bash
git clone <repository-url>
cd streaming-live-rag
```

### 2. Install dependencies

Ensure you are using Python 3.9+ (tested on Python 3.10+).

```bash
pip install -r requirements.txt
```

### 3. Configure the environment

Copy the example environment configuration:
```bash
cp .env.example .env
```
(Optional) Edit `.env` to configure your AssemblyAI API key if you plan to use real STT.

### 4. Run the application

The repository uses a single FastAPI backend that serves both the API and the front-end dashboard.

**Option A: Mock Mode (No API keys required, fully reproducible)**
```bash
SLRAG_INPUT_MODE=mock uvicorn app.main:app --host 0.0.0.0 --port 8000
```
*(On Windows PowerShell, prefix with `$env:SLRAG_INPUT_MODE="mock";`)*

**Option B: Real STT Mode (Requires AssemblyAI key)**
```bash
SLRAG_INPUT_MODE=stt SLRAG_STT_PROVIDER=assemblyai SLRAG_STT_API_KEY=your_key uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 5. Access the application

Open your browser and navigate to the integrated dashboard:
**http://localhost:8000/**

## Running Tests & Benchmarks

To execute the unit tests (pytest must be installed or run as a module):

```bash
python -m pytest tests/unit/ -v
```

To run the deterministic G1-G6 PRISM Benchmark Harness (which currently passes 100% of the gates):

```bash
python benchmarks/run.py
```

## Docker Support

The project includes a `Dockerfile` and `docker-compose.yml` for isolated deployment.

```bash
docker-compose up --build
```
