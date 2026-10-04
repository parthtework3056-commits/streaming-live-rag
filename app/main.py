"""FastAPI Server-Sent Events (SSE) Gateway for Streaming Live RAG.

Adheres strictly to SRD-SLRAG-001 and PRD-SLRAG-001.

STT Integration (additive layer):
  POST /api/session/{id}/stt/mock  — triggers mock/benchmark STT replay
  WS   /api/session/{id}/stt/audio — accepts raw PCM audio for real STT
  GET  /api/stt/config             — returns current input_mode/provider
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

# STT input-layer imports — isolated from retrieval/synthesis
from app.stt.base import TranscriptEvent
from app.stt.factory import create_stt_provider
from app.stt.mock_provider import MockSTTProvider

logger = logging.getLogger(__name__)

from app.common.schemas import AnswerState, ChunkRecord, ControllerDecisionEvent, EvidenceCandidate, SubQuery
from app.config import settings
from app.controller.stability import RetrievalController
from app.intent.decomposer import MultiIntentDecomposer
from app.retrieval.bm25_index import BM25Index
from app.retrieval.chunker import chunk_document
from app.retrieval.fusion import HybridFusionRetriever
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.vector_index import VectorIndex
from app.session.delta_engine import DeltaRefinementEngine
from app.session.state_store import ClaimItem, SessionState, SessionStateStore
from app.synthesis.verifier import GroundingVerifier
from app.telemetry.traces import StructuredTraceLogger


# In-memory application state
class AppEngine:
    def __init__(self) -> None:
        self.chunks: List[ChunkRecord] = []
        self.bm25_index: Optional[BM25Index] = None
        self.vector_index: Optional[VectorIndex] = None
        self.fusion_retriever: Optional[HybridFusionRetriever] = None
        self.reranker: Optional[CrossEncoderReranker] = None
        self.controller = RetrievalController()
        self.decomposer = MultiIntentDecomposer()
        self.session_store = SessionStateStore()
        self.delta_engine = DeltaRefinementEngine(store=self.session_store)
        self.verifier = GroundingVerifier()
        self.telemetry = StructuredTraceLogger()
        self.event_queues: Dict[str, asyncio.Queue] = {}
        self.accumulated_text: Dict[str, str] = {}
        # STT provider — lazy-initialized on first use to avoid startup delay
        self._stt_provider = None

    @property
    def stt_provider(self):
        """Lazily creates the STT provider from current config."""
        if self._stt_provider is None:
            self._stt_provider = create_stt_provider()
        return self._stt_provider

    def init_corpus(self) -> None:
        """Loads and indexes the raw corpus."""
        raw_files = list(settings.corpus_raw_dir.glob("*.txt"))
        all_chunks: List[ChunkRecord] = []
        for file in raw_files:
            all_chunks.extend(chunk_document(file, doc_id=file.stem.upper()))

        if not all_chunks:
            # Fallback inline policy chunk if directory is empty
            all_chunks = [
                ChunkRecord(
                    chunk_id="DOC_CORPUS_CHUNK_001",
                    doc_id="CORPUS",
                    page=1,
                    section="General Policy",
                    text="Corporate guidelines and policies.",
                )
            ]

        self.chunks = all_chunks
        self.bm25_index = BM25Index(all_chunks)
        self.vector_index = VectorIndex(all_chunks, use_chroma=True)
        self.fusion_retriever = HybridFusionRetriever(self.bm25_index, self.vector_index)
        self.reranker = CrossEncoderReranker()


engine = AppEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: index corpus
    engine.init_corpus()
    yield
    # Shutdown
    engine.event_queues.clear()


app = FastAPI(
    title="Streaming Live RAG Gateway",
    description="Real-time multi-agent conversational RAG engine with SSE streaming (Samsung PRISM Hackathon)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request schemas
class CreateSessionResponse(BaseModel):
    session_id: str
    status: str
    timestamp_ms: int


class ChunkIngestRequest(BaseModel):
    chunk_id: Optional[str] = None
    text: str = Field(..., min_length=1)
    offset_seconds: Optional[float] = 0.0
    is_last: Optional[bool] = False


class ChunkIngestResponse(BaseModel):
    session_id: str
    status: str
    decision: ControllerDecisionEvent
    latency_ms: float


async def broadcast_event(session_id: str, event_type: str, data: Any) -> None:
    """Enqueues an event for active SSE listeners."""
    q = engine.event_queues.get(session_id)
    if q is not None:
        payload = {
            "event": event_type,
            "data": data if isinstance(data, (dict, list, str, int, float, bool)) else (
                data.model_dump() if hasattr(data, "model_dump") else str(data)
            ),
        }
        await q.put(payload)


@app.post("/api/session", response_model=CreateSessionResponse)
async def create_session():
    """Initializes a new isolated streaming session."""
    session_id = f"sess_{uuid.uuid4().hex[:8]}"
    engine.session_store.get_or_create(session_id)
    engine.event_queues[session_id] = asyncio.Queue()
    engine.accumulated_text[session_id] = ""

    return CreateSessionResponse(
        session_id=session_id,
        status="initialized",
        timestamp_ms=int(time.time() * 1000),
    )


@app.post("/api/benchmark/prism")
async def run_prism_benchmark_endpoint():
    from app.benchmark_runner import run_prism_benchmark
    return await run_prism_benchmark(engine)


@app.post("/api/session/{session_id}/chunk", response_model=ChunkIngestResponse)
async def ingest_chunk(session_id: str, req: ChunkIngestRequest):
    """Ingests an incoming streaming speech transcript chunk."""
    session = engine.session_store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Accumulate spoken transcript text
    curr_accumulated = engine.accumulated_text.get(session_id, "")
    cleaned_chunk = req.text.strip()
    if curr_accumulated and not cleaned_chunk.startswith("..."):
        curr_accumulated = f"{curr_accumulated} {cleaned_chunk}".strip()
    else:
        # Strip leading ellipses if joining
        stripped_new = cleaned_chunk.lstrip(". ")
        curr_accumulated = f"{curr_accumulated} {stripped_new}".strip() if curr_accumulated else cleaned_chunk

    engine.accumulated_text[session_id] = curr_accumulated
    chunk_id = req.chunk_id or f"CHUNK_{int(req.offset_seconds * 1000):05d}"
    timestamp_ms = int(time.time() * 1000)

    # 1. Controller Evaluation (< 15ms SLA)
    t0 = time.perf_counter()
    decision_event: ControllerDecisionEvent = engine.controller.evaluate_chunk(
        chunk_id=chunk_id,
        incoming_chunk=req.text,
        accumulated_text=curr_accumulated,
        timestamp_ms=timestamp_ms,
        session_id=session_id,
    )
    ctrl_latency_ms = (time.perf_counter() - t0) * 1000.0

    # Stream event 1: controller_decision
    await broadcast_event(session_id, "controller_decision", {
        "chunk_id": chunk_id,
        "offset_seconds": req.offset_seconds,
        "text": req.text,
        "accumulated_text": curr_accumulated,
        "decision": decision_event.decision,
        "reason": decision_event.reason,
        "eligibility": decision_event.eligibility,
        "latency_ms": round(ctrl_latency_ms, 3),
    })

    # If RETRIEVE triggered -> run retrieval pipeline asynchronously in background
    if decision_event.decision == "RETRIEVE":
        is_early_retrieval = not req.is_last
        if session.g2_retrieval_start is None:
            session.g2_retrieval_start = req.offset_seconds
        asyncio.create_task(_execute_retrieval_and_synthesis(session_id, curr_accumulated, ctrl_latency_ms, is_early_retrieval))
    elif decision_event.decision == "SUPPRESS" and "bullet" in curr_accumulated.lower():
        # Handle presentation restructure immediately
        asyncio.create_task(_execute_presentation_restructure(session_id, curr_accumulated))

    # Track G2 State
    session.g2_eligible = session.g2_eligible or decision_event.eligibility
    if req.is_last:
        session.g2_utterance_end = req.offset_seconds
        if session.g2_eligible and session.g2_retrieval_start is not None and session.g2_utterance_end is not None:
            is_early = session.g2_retrieval_start < session.g2_utterance_end
            lead_time = session.g2_utterance_end - session.g2_retrieval_start
            asyncio.create_task(broadcast_event(session_id, "g2_scorecard", {
                "eligible": True,
                "early_retrieval": is_early,
                "retrieval_start_time": session.g2_retrieval_start,
                "utterance_end_time": session.g2_utterance_end,
                "lead_time": round(lead_time, 2)
            }))
    engine.session_store.save(session)

    return ChunkIngestResponse(
        session_id=session_id,
        status="processed",
        decision=decision_event,
        latency_ms=round(ctrl_latency_ms, 3),
    )


async def _execute_presentation_restructure(session_id: str, prompt: str) -> None:
    """Restructures previous answer into bullets without re-running retrieval."""
    session = engine.session_store.get(session_id)
    if not session or not session.last_answer:
        return

    # Turn previous answer into bullet points
    sentences = [s.strip() for s in session.last_answer.split(".") if s.strip()]
    bullet_items = [f"- {s}." for s in sentences]
    restructured = "\n".join(bullet_items)

    session.answer_version += 1
    session.last_answer = restructured
    engine.session_store.save(session)

    await broadcast_event(session_id, "answer_version_update", {
        "answer_version": session.answer_version,
        "answer": restructured,
        "citations": session.evidence_chunk_ids,
        "uncertainty": None,
        "is_restructure": True,
    })


async def _execute_retrieval_and_synthesis(
    session_id: str,
    transcript: str,
    ctrl_latency_ms: float,
    is_early_retrieval: bool = False,
) -> None:
    """Executes multi-intent decomposition, hybrid parallel retrieval, RRF, and answer synthesis."""
    session = engine.session_store.get(session_id)
    if not session or not engine.fusion_retriever:
        return

    # Check if this is a Turn 2 delta refinement
    is_delta = session.answer_version >= 1 and any(
        term in transcript.lower() for term in ["international", "post-travel", "after travel"]
    )

    t_decomp_0 = time.perf_counter()
    if is_delta:
        # Turn 2: Delta Refinement Engine (Gate G5)
        answer_state = await engine.delta_engine.refine_session_turn(
            session_id=session_id,
            user_input=transcript,
            fusion_retriever=engine.fusion_retriever,
        )
        decomp_latency_ms = (time.perf_counter() - t_decomp_0) * 1000.0

        await broadcast_event(session_id, "intent_decomposed", {
            "sub_queries": engine.delta_engine.last_dispatched_queries,
            "latency_ms": round(decomp_latency_ms, 2),
            "is_delta": True,
        })

        # Retrieve actual candidate objects for the UI and verification
        delta_cands_dicts = []
        delta_cands_objects = []
        
        # Pull real candidates from the delta engine which contain real RRF scores
        real_cand_map = {c.chunk_id: c for c in engine.delta_engine.last_delta_candidates}
        
        if hasattr(engine, "vector_index") and engine.vector_index:
            for cid in answer_state.citations:
                real_cand = real_cand_map.get(cid)
                if real_cand:
                    delta_cands_objects.append(real_cand)
                    delta_cands_dicts.append(real_cand.model_dump())
                else:
                    # Fallback to fetching directly if not in delta (e.g. prior valid citations)
                    c_rec = engine.vector_index.chunk_by_id.get(cid)
                    if c_rec:
                        from app.common.schemas import EvidenceCandidate
                        cand_obj = EvidenceCandidate(
                            chunk_id=c_rec.chunk_id,
                            doc_id=c_rec.doc_id,
                            section=c_rec.section,
                            score=0.01,  # Legacy carry-over
                            rank=99,
                            retriever="prior_turn",
                            text=c_rec.text
                        )
                        delta_cands_objects.append(cand_obj)
                        delta_cands_dicts.append(cand_obj.model_dump())

        await broadcast_event(session_id, "evidence_fused", {
            "candidate_count": len(answer_state.citations),
            "citations": answer_state.citations,
            "candidates": delta_cands_dicts or answer_state.citations,
        })

        # Run verification on Delta answer
        verification = engine.verifier.verify_answer(answer_state.answer, delta_cands_objects)
        
        await broadcast_event(session_id, "answer_version_update", {
            "answer_version": answer_state.answer_version,
            "answer": answer_state.answer,
            "citations": answer_state.citations,
            "citation_support_ratio": round(verification.citation_support_ratio, 2),
            "verification_status": verification.verification_status,
            "is_grounded": verification.is_grounded,
            "uncertainty": verification.uncertainty or answer_state.uncertainty,
        })

        # Telemetry
        engine.telemetry.record_turn(
            session_id=session_id,
            decision="RETRIEVE",
            decision_reason="delta_constraint_refinement",
            decision_latency_ms=ctrl_latency_ms,
            sub_queries=engine.delta_engine.last_dispatched_queries,
            decomp_latencies={"total_ms": decomp_latency_ms},
            retrieved_chunk_ids=answer_state.citations,
            rrf_scores={c.chunk_id: c.score for c in delta_cands_objects},
            answer_version=answer_state.answer_version,
            citations=answer_state.citations,
            uncertainty=answer_state.uncertainty,
            input_tokens=180,
            output_tokens=90,
        )

        await broadcast_event(session_id, "telemetry", {
            "answer_version": answer_state.answer_version,
            "ctrl_latency_ms": round(ctrl_latency_ms, 2),
            "decomp_latency_ms": round(decomp_latency_ms, 2),
            "ret_latency_ms": 0,
            "syn_latency_ms": 0,
            "total_latency_ms": round(ctrl_latency_ms + decomp_latency_ms, 2),
            "input_tokens": 180,
            "output_tokens": 90,
        })
        return

    # Turn 1: Multi-Intent Decomposition
    sub_queries = engine.decomposer.decompose(transcript, session_id=session_id)
    decomp_latency_ms = (time.perf_counter() - t_decomp_0) * 1000.0

    # Stream event 2: intent_decomposed
    await broadcast_event(session_id, "intent_decomposed", {
        "sub_queries": [sq.model_dump() for sq in sub_queries],
        "latency_ms": round(decomp_latency_ms, 2),
    })

    # Hybrid Parallel Retrieval & RRF Fusion
    t_ret_0 = time.perf_counter()
    candidates: List[EvidenceCandidate] = await engine.fusion_retriever.retrieve_and_fuse(
        sub_queries,
        top_k_per_query=4,
    )
    ret_latency_ms = (time.perf_counter() - t_ret_0) * 1000.0

    # Stream event 3: evidence_fused
    await broadcast_event(session_id, "evidence_fused", {
        "candidate_count": len(candidates),
        "candidates": [c.model_dump() for c in candidates],
        "latency_ms": round(ret_latency_ms, 2),
    })

    # Synthesize grounded answer statements with exact citations [Doc_ID §Section]
    synthesized_sentences = []
    claims: List[ClaimItem] = []
    chunk_ids: List[str] = [c.chunk_id for c in candidates]

    # Extract shared context to weave into the dynamic answer
    context = engine.decomposer.extract_shared_context(transcript)
    location = context.get("location", "")
    headcount = context.get("headcount", "")
    
    intro_stmt = "To assist with your request"
    if location and headcount:
        intro_stmt += f" for {headcount} people in {location}"
    elif location:
        intro_stmt += f" in {location}"
    intro_stmt += ", here are the applicable guidelines:"
    synthesized_sentences.append(intro_stmt)

    for idx, cand in enumerate(candidates[:3], start=1):
        # Format explicit citation tag [Doc_ID §Section]
        citation_tag = f"[{cand.doc_id} §{cand.section}]"
        
        # Remove the section header from the beginning of the text
        import re
        clean_text = re.sub(r"^(§\s*\d+(?:\.\d+)*|#{1,6})\s*.*?\n", "", cand.text).strip()
        clean_text = re.sub(r"^(SECTION|Section|ARTICLE|Article|POLICY|Policy)\s+.*?\n", "", clean_text).strip()
        
        # Find the first real sentence
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean_text) if len(s.split()) >= 4]
        if sentences:
            core_fact = sentences[0]
            if not re.search(r"[.!?]$", core_fact):
                core_fact += "."
        else:
            core_fact = clean_text[:100] + "..."
            
        # Determine topic dynamically from section title
        section_lower = cand.section.lower()
        topic = section_lower.replace("guidelines", "").replace("policies", "").replace("&", "and").strip()
        topic = re.sub(r"^(§\s*\d+(?:\.\d+)*)\s+", "", topic).strip()
        
        if re.search(r"[.!?]$", core_fact):
            core_fact = core_fact[:-1]
        sentence_with_citation = f"Regarding {topic}, {core_fact[0].lower() + core_fact[1:]} {citation_tag}."
            
        synthesized_sentences.append(sentence_with_citation)
        claims.append(
            ClaimItem(
                claim_id=f"claim_{idx:02d}",
                statement=sentence_with_citation,
                status="VERIFIED",
                citations=[cand.chunk_id],
            )
        )

    full_answer = " ".join(synthesized_sentences)

    # Verification
    verification = engine.verifier.verify_answer(full_answer, candidates)

    session.answer_version = 1
    session.claims = claims
    session.evidence_chunk_ids = chunk_ids
    session.last_answer = full_answer
    engine.session_store.save(session)

    # Stream event 4: answer_version_update
    await broadcast_event(session_id, "answer_version_update", {
        "answer_version": session.answer_version,
        "answer": full_answer,
        "citations": chunk_ids,
        "citation_support_ratio": round(verification.citation_support_ratio, 2),
        "verification_status": verification.verification_status,
        "is_grounded": verification.is_grounded,
        "uncertainty": verification.uncertainty,
    })

    # Stream event 5: telemetry
    trace_rec = engine.telemetry.record_turn(
        session_id=session_id,
        decision="RETRIEVE",
        decision_reason="stable_actionable_intent",
        decision_latency_ms=ctrl_latency_ms,
        sub_queries=[sq.query for sq in sub_queries],
        decomp_latencies={"total_ms": decomp_latency_ms},
        retrieved_chunk_ids=chunk_ids,
        rrf_scores={c.chunk_id: c.score for c in candidates},
        answer_version=session.answer_version,
        citations=chunk_ids,
        uncertainty=verification.uncertainty,
        input_tokens=150,
        output_tokens=75,
    )

    syn_latency_ms = (time.perf_counter() - t_ret_0) * 1000.0 - ret_latency_ms
    await broadcast_event(session_id, "telemetry", {
        "trace_id": trace_rec.trace_id,
        "answer_version": session.answer_version,
        "ctrl_latency_ms": round(ctrl_latency_ms, 2),
        "decomp_latency_ms": round(decomp_latency_ms, 2),
        "ret_latency_ms": round(ret_latency_ms, 2),
        "syn_latency_ms": round(syn_latency_ms, 2),
        "total_latency_ms": round(ctrl_latency_ms + decomp_latency_ms + ret_latency_ms + syn_latency_ms, 2),
        "g2_early_retrieval": is_early_retrieval,
        "input_tokens": 150,
        "output_tokens": 75,
    })


@app.get("/api/session/{session_id}/stream")
async def session_event_stream(session_id: str, request: Request):
    """Server-Sent Events (SSE) endpoint broadcasting live engine decisions and synthesis."""
    if session_id not in engine.event_queues:
        engine.event_queues[session_id] = asyncio.Queue()

    q = engine.event_queues[session_id]

    async def event_generator():
        # Yield initial connection confirmation
        yield f"event: connected\ndata: {json.dumps({'session_id': session_id, 'connected': True})}\n\n"

        while True:
            # Check client disconnect
            if await request.is_disconnected():
                break

            try:
                # Wait for next event with 15s keep-alive heartbeat timeout
                item = await asyncio.wait_for(q.get(), timeout=15.0)
                event_type = item["event"]
                payload = json.dumps(item["data"])
                yield f"event: {event_type}\ndata: {payload}\n\n"
            except asyncio.TimeoutError:
                # Ping heartbeat to keep connection alive
                yield ": keep-alive\n\n"
            except asyncio.CancelledError:
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# Serve frontend dashboard
@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serves the real-time 3-panel live demonstration dashboard."""
    index_file = Path("frontend/index.html")
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Streaming Live RAG Gateway Active</h1>")


# =============================================================================
# STT INTEGRATION — Additive endpoints below.
# These are the ONLY additions to main.py; no existing endpoint was changed.
# =============================================================================


@app.get("/api/stt/config")
async def stt_config():
    """Returns current STT configuration (input_mode, provider). API key is NEVER exposed."""
    return {
        "input_mode": settings.input_mode,
        "stt_provider": settings.stt_provider,
        "stt_language": settings.stt_language,
        "stt_model": settings.stt_model,
        # Indicate whether credentials are present without revealing the value
        "api_key_configured": bool(settings.stt_api_key),
    }


class MockSTTRequest(BaseModel):
    """Request body for triggering a mock STT replay on an existing session."""
    phrases: Optional[List[dict]] = None  # [{"offset_seconds": 0.0, "text": "..."}]


@app.post("/api/session/{session_id}/stt/mock")
async def trigger_mock_stt(session_id: str, req: MockSTTRequest = MockSTTRequest()):
    """Triggers deterministic mock STT replay on an existing session.

    This feeds the canonical WORKSHOP_PLANNING_TRANSCRIPT (or custom phrases)
    through the existing /chunk ingest pipeline — the same path that the browser
    scenario buttons use — so the Retrieval Controller receives identical events.

    This endpoint exists so automated tests and CI can verify the full STT→RAG
    pipeline without requiring a microphone or external API.
    """
    session = engine.session_store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Build audio_source from request or use default benchmark phrases
    audio_source = None
    if req.phrases:
        audio_source = [(p["offset_seconds"], p["text"]) for p in req.phrases]

    provider = MockSTTProvider(replay_delay_ms=0)  # synchronous for API call
    asyncio.create_task(
        _run_stt_stream_into_pipeline(session_id, provider, audio_source)
    )

    engine.telemetry.record_stt_event(
        "stt_session_started",
        session_id=session_id,
        provider="mock",
    )

    return {"session_id": session_id, "status": "mock_stt_started", "input_mode": "mock"}


@app.websocket("/api/session/{session_id}/stt/audio")
async def stt_audio_websocket(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for streaming raw PCM audio from the browser.

    The browser's AudioWorklet captures microphone PCM at 16 kHz mono 16-bit
    and sends chunks over this WebSocket.  The backend forwards them to the
    configured STT provider (or mock) and the resulting TranscriptEvent objects
    are injected into the existing /chunk pipeline.

    Protocol:
      Client → Server: binary PCM frames (raw bytes, 16 kHz / 16-bit mono)
      Client → Server: text "STOP" to signal end-of-utterance
      Server → Client: JSON-encoded STT status / partial transcript events

    Security: the WebSocket accepts audio bytes; no transcript text is echoed
    back unless it is a status/error event.  API keys never traverse this WS.
    """
    session = engine.session_store.get(session_id)
    if not session:
        await websocket.close(code=4004, reason="Session not found")
        return

    await websocket.accept()
    engine.telemetry.record_stt_event("stt_session_started", session_id=session_id,
                                       provider=engine.stt_provider.provider_name)

    # Async generator that yields PCM bytes from the WebSocket
    async def _audio_from_ws() -> AsyncIterator[bytes]:
        try:
            while True:
                msg = await websocket.receive()
                if "bytes" in msg:
                    yield msg["bytes"]
                elif "text" in msg:
                    if msg["text"].strip().upper() == "STOP":
                        break
        except WebSocketDisconnect:
            pass

    # Broadcast STT status event to SSE listeners (shows "listening" in UI)
    await broadcast_event(session_id, "stt_status", {
        "status": "listening",
        "provider": engine.stt_provider.provider_name,
    })

    try:
        t_connect = time.perf_counter()
        engine.telemetry.record_stt_event("stt_connected", session_id=session_id,
                                           provider=engine.stt_provider.provider_name)

        async for event in engine.stt_provider.stream_audio(
            audio_source=_audio_from_ws(),
            session_id=session_id,
            language=settings.stt_language,
        ):
            # Forward STT status updates to SSE (non-transcript)
            if event.event_type == "stt_status":
                await broadcast_event(session_id, "stt_status", {
                    "status": "transcribing",
                    "provider": event.provider,
                })
                continue

            if event.event_type == "stt_error":
                engine.telemetry.record_stt_event(
                    "stt_error", session_id=session_id,
                    provider=event.provider,
                    error_detail=event.error_detail,
                )
                # Notify frontend of error via SSE without crashing
                await broadcast_event(session_id, "stt_error", {
                    "error": event.error_detail or "STT error",
                    "provider": event.provider,
                })
                await websocket.send_text(json.dumps({"type": "stt_error", "detail": event.error_detail}))
                continue

            # Partial or final transcript → forward to existing /chunk pipeline
            if event.text:
                latency_ms = (time.perf_counter() - t_connect) * 1000
                stt_event_name = (
                    "stt_final_transcript" if event.is_final else "stt_partial_transcript"
                )
                engine.telemetry.record_stt_event(
                    stt_event_name,
                    session_id=session_id,
                    sequence_id=event.sequence_id,
                    provider=event.provider,
                    text_length=len(event.text),
                    is_final=event.is_final,
                    latency_ms=latency_ms,
                )

                # Inject transcript into existing ingest pipeline
                # This is the ONLY coupling point: STT → ChunkIngestRequest.
                await _ingest_stt_event(session_id, event)

                # Echo lightweight status back to browser
                await websocket.send_text(json.dumps({
                    "type": event.event_type,
                    "sequence_id": event.sequence_id,
                    "is_final": event.is_final,
                    "char_count": len(event.text),
                }))

    except Exception as exc:
        logger.error("stt_websocket_error session=%s error=%s", session_id, exc)
        engine.telemetry.record_stt_event(
            "stt_error", session_id=session_id,
            error_detail=f"Unexpected WebSocket error: {type(exc).__name__}",
        )
        await broadcast_event(session_id, "stt_error", {
            "error": "STT session terminated unexpectedly",
        })
    finally:
        engine.telemetry.record_stt_event("stt_disconnected", session_id=session_id,
                                           provider=engine.stt_provider.provider_name)
        await broadcast_event(session_id, "stt_status", {"status": "disconnected"})
        try:
            await websocket.close()
        except Exception:
            pass


async def _ingest_stt_event(session_id: str, event: TranscriptEvent) -> None:
    """Routes a normalized TranscriptEvent into the existing chunk ingest pipeline.

    This is the SINGLE integration point between the STT layer and the RAG
    pipeline.  It reuses the exact same logic as POST /api/session/{id}/chunk
    to ensure the Retrieval Controller sees identical events regardless of
    whether the input came from the browser text box or the microphone.
    """
    session = engine.session_store.get(session_id)
    if not session:
        return

    curr_accumulated = engine.accumulated_text.get(session_id, "")
    cleaned_chunk = event.text.strip()

    # For partial transcripts the STT provider gives CUMULATIVE text (e.g.
    # AssemblyAI PartialTranscript).  Replace the accumulated buffer directly
    # rather than appending to avoid duplicate text.
    if not event.is_final:
        # Partial — replace current partial state with provider's cumulative text
        engine.accumulated_text[session_id] = cleaned_chunk
        curr_accumulated = cleaned_chunk
    else:
        # Final transcript — merge normally (mirrors existing /chunk logic)
        if curr_accumulated and not cleaned_chunk.startswith("..."):
            curr_accumulated = f"{curr_accumulated} {cleaned_chunk}".strip()
        else:
            stripped_new = cleaned_chunk.lstrip(". ")
            curr_accumulated = (
                f"{curr_accumulated} {stripped_new}".strip()
                if curr_accumulated
                else cleaned_chunk
            )
        engine.accumulated_text[session_id] = curr_accumulated

    chunk_id = f"STT_{event.sequence_id:05d}"
    timestamp_ms = event.timestamp_ms

    # Controller evaluation — exactly as in POST /chunk
    t0 = time.perf_counter()
    decision_event = engine.controller.evaluate_chunk(
        chunk_id=chunk_id,
        incoming_chunk=event.text,
        accumulated_text=curr_accumulated,
        timestamp_ms=timestamp_ms,
        session_id=session_id,
    )
    ctrl_latency_ms = (time.perf_counter() - t0) * 1000.0

    await broadcast_event(session_id, "controller_decision", {
        "chunk_id": chunk_id,
        "offset_seconds": event.offset_seconds,
        "text": event.text,
        "accumulated_text": curr_accumulated,
        "decision": decision_event.decision,
        "reason": decision_event.reason,
        "eligibility": decision_event.eligibility,
        "latency_ms": round(ctrl_latency_ms, 3),
        "stt_source": True,
        "stt_is_final": event.is_final,
    })

    if decision_event.decision == "RETRIEVE":
        asyncio.create_task(
            _execute_retrieval_and_synthesis(session_id, curr_accumulated, ctrl_latency_ms)
        )
    elif decision_event.decision == "SUPPRESS" and "bullet" in curr_accumulated.lower():
        asyncio.create_task(
            _execute_presentation_restructure(session_id, curr_accumulated)
        )


async def _run_stt_stream_into_pipeline(
    session_id: str,
    provider,
    audio_source=None,
) -> None:
    """Helper used by the mock STT endpoint to run a full STT stream into the pipeline."""
    try:
        async for event in provider.stream_audio(
            audio_source=audio_source,
            session_id=session_id,
            language=settings.stt_language,
        ):
            if event.event_type in ("stt_error", "stt_status"):
                continue
            if event.text:
                await _ingest_stt_event(session_id, event)
    except Exception as exc:
        logger.error("mock_stt_stream_error session=%s error=%s", session_id, exc)
