"""
AMD Issues API Server

A self-hosted FastAPI server for intelligent GitHub issue management.
Provides AI-powered priority assignment, duplicate detection, label suggestion, and SLA tracking.

Usage:
    uvicorn main:app --host 0.0.0.0 --port 8000

Environment Variables:
    AMD_API_TOKEN       - Required. API authentication token (min 32 chars)
    MODEL_NAME          - Model to use (default: Qwen/Qwen2.5-7B-Instruct)
    QDRANT_URL          - Qdrant vector DB URL (default: http://localhost:6333)
    RATE_LIMIT_RPM      - Requests per minute limit (default: 100)
"""

import os
import re
import time
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

API_TOKEN = os.getenv("AMD_API_TOKEN", "")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "100"))

# Validate token on startup
if len(API_TOKEN) < 32:
    logger.warning("AMD_API_TOKEN should be at least 32 characters for security")

# =============================================================================
# Models
# =============================================================================

class Comment(BaseModel):
    author: str
    body: str
    created_at: str


class Issue(BaseModel):
    number: int
    title: str
    body: str = ""
    labels: list[str] = []
    state: str = "open"
    created_at: str
    updated_at: Optional[str] = None
    author: Optional[str] = None
    comments_count: int = 0
    recent_comments: list[Comment] = []
    url: Optional[str] = None


class AnalyzeContext(BaseModel):
    repository: str
    event: Optional[str] = None
    dry_run: bool = False


class AnalyzeRequest(BaseModel):
    issue: Issue
    actions: list[str] = Field(default=["priority", "duplicates", "labels", "sla"])
    context: Optional[AnalyzeContext] = None


class IndexRequest(BaseModel):
    issue: Issue
    repository: str


class SyncRequest(BaseModel):
    repository: str
    since: Optional[str] = None
    state: str = "all"
    labels: list[str] = []


class PriorityResult(BaseModel):
    level: str
    confidence: int
    reason: str


class DuplicateMatch(BaseModel):
    issue_number: int
    title: str
    similarity: int
    state: str = "open"
    resolution: Optional[str] = None


class DuplicatesResult(BaseModel):
    matches: list[DuplicateMatch]
    search_time_ms: int


class LabelsResult(BaseModel):
    suggestions: list[str]
    explanations: dict[str, str]
    confidence: int


class SLAResult(BaseModel):
    status: str  # ok, at_risk, breach
    priority_level: Optional[str]
    target_response_hours: Optional[int]
    elapsed_hours: float
    time_remaining: Optional[str]
    breach_risk: str


class AnalyzeResponse(BaseModel):
    success: bool = True
    request_id: str
    processing_time_ms: int
    model: dict
    priority: Optional[PriorityResult] = None
    duplicates: Optional[DuplicatesResult] = None
    labels: Optional[LabelsResult] = None
    sla: Optional[SLAResult] = None


# =============================================================================
# AI/ML Components (Placeholder implementations)
# =============================================================================

class ModelInference:
    """
    Model inference wrapper.

    In production, replace with actual model loading:
    - vLLM: from vllm import LLM
    - Ollama: import ollama
    - Transformers: from transformers import AutoModelForCausalLM
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.loaded = False
        logger.info(f"Initializing model: {model_name}")
        # In production: self.model = LLM(model_name)
        self.loaded = True

    async def classify_priority(self, title: str, body: str, labels: list[str]) -> PriorityResult:
        """Classify issue priority using LLM."""

        # Priority keywords (fallback/augmentation for LLM)
        p0_keywords = ["crash", "production", "down", "security", "data loss", "critical", "urgent"]
        p1_keywords = ["blocker", "broken", "regression", "major", "cannot"]
        p2_keywords = ["bug", "error", "fail", "wrong", "issue"]
        p3_keywords = ["enhancement", "feature", "request", "improve", "minor", "docs"]

        text = f"{title} {body}".lower()

        # Simple keyword-based classification (replace with LLM inference)
        if any(kw in text for kw in p0_keywords):
            return PriorityResult(
                level="P0",
                confidence=85,
                reason="Critical keywords detected indicating production impact"
            )
        elif any(kw in text for kw in p1_keywords):
            return PriorityResult(
                level="P1",
                confidence=80,
                reason="Major issue keywords detected"
            )
        elif any(kw in text for kw in p2_keywords):
            return PriorityResult(
                level="P2",
                confidence=75,
                reason="Standard bug/error patterns detected"
            )
        else:
            return PriorityResult(
                level="P3",
                confidence=70,
                reason="No urgent indicators found"
            )

        # Production LLM inference would look like:
        # prompt = f"""Classify the priority of this GitHub issue:
        # Title: {title}
        # Body: {body}
        # Labels: {labels}
        #
        # Priority levels:
        # P0: Production down, security issue, data loss
        # P1: Major feature broken, blocking users
        # P2: Bug with workaround available
        # P3: Minor issue, enhancement
        #
        # Return JSON: {{"level": "P0|P1|P2|P3", "confidence": 0-100, "reason": "..."}}
        # """
        # response = await self.model.generate(prompt)
        # return PriorityResult.parse_raw(response)

    async def suggest_labels(self, title: str, body: str) -> LabelsResult:
        """Suggest labels using LLM."""

        text = f"{title} {body}".lower()
        suggestions = []
        explanations = {}

        # Hardware detection
        if "mi300" in text:
            suggestions.append("mi300x")
            explanations["mi300x"] = "MI300X hardware mentioned"
        if "mi325" in text:
            suggestions.append("mi325x")
            explanations["mi325x"] = "MI325X hardware mentioned"
        if "mi350" in text:
            suggestions.append("mi350")
            explanations["mi350"] = "MI350 hardware mentioned"

        # ROCm version detection
        rocm_match = re.search(r"rocm[- ]?(\d+)\.(\d+)", text)
        if rocm_match:
            major = int(rocm_match.group(1))
            if major >= 7:
                suggestions.append("rocm-7.x")
                explanations["rocm-7.x"] = f"ROCm {major}.x version detected"
            elif major >= 6:
                suggestions.append("rocm-6.x")
                explanations["rocm-6.x"] = f"ROCm {major}.x version detected"

        # Component detection
        if "rccl" in text:
            suggestions.append("rccl")
            explanations["rccl"] = "RCCL communication library mentioned"
        if "hipblas" in text:
            suggestions.append("hipblaslt")
            explanations["hipblaslt"] = "hipBLAS library mentioned"
        if "aiter" in text:
            suggestions.append("aiter-backend")
            explanations["aiter-backend"] = "AIter backend mentioned"

        # Type detection
        if any(kw in text for kw in ["bug", "error", "fail", "crash", "broken"]):
            suggestions.append("bug")
            explanations["bug"] = "Error/failure patterns detected"
        elif any(kw in text for kw in ["feature", "request", "add", "support"]):
            suggestions.append("feature")
            explanations["feature"] = "Feature request patterns detected"
        elif any(kw in text for kw in ["slow", "performance", "latency", "throughput"]):
            suggestions.append("performance")
            explanations["performance"] = "Performance-related keywords detected"

        return LabelsResult(
            suggestions=suggestions,
            explanations=explanations,
            confidence=80 if suggestions else 50
        )

    async def get_embedding(self, text: str) -> list[float]:
        """Get text embedding for similarity search."""
        # In production, use actual embedding model:
        # from sentence_transformers import SentenceTransformer
        # self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        # return self.embed_model.encode(text).tolist()

        # Placeholder: return hash-based pseudo-embedding
        hash_bytes = hashlib.sha256(text.encode()).digest()
        return [float(b) / 255.0 for b in hash_bytes]


class VectorDatabase:
    """
    Vector database for duplicate detection.

    In production, use Qdrant, Milvus, or similar:
    - from qdrant_client import QdrantClient
    - self.client = QdrantClient(url=QDRANT_URL)
    """

    def __init__(self, url: str):
        self.url = url
        self.issues: dict[str, dict] = {}  # In-memory placeholder
        logger.info(f"Connecting to vector DB: {url}")

    async def index_issue(self, repository: str, issue: Issue, embedding: list[float]) -> str:
        """Index an issue with its embedding."""
        key = f"{repository}:{issue.number}"
        self.issues[key] = {
            "number": issue.number,
            "title": issue.title,
            "body": issue.body,
            "embedding": embedding,
            "state": issue.state,
            "created_at": issue.created_at
        }
        return f"emb_{hashlib.md5(key.encode()).hexdigest()[:12]}"

    async def search_similar(
        self,
        repository: str,
        embedding: list[float],
        limit: int = 10,
        threshold: float = 0.7
    ) -> list[DuplicateMatch]:
        """Search for similar issues."""

        # In production, use vector similarity search:
        # results = self.client.search(
        #     collection_name="issues",
        #     query_vector=embedding,
        #     limit=limit,
        #     score_threshold=threshold
        # )

        # Placeholder: simple cosine similarity
        def cosine_similarity(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = sum(x * x for x in a) ** 0.5
            norm_b = sum(x * x for x in b) ** 0.5
            if norm_a == 0 or norm_b == 0:
                return 0
            return dot / (norm_a * norm_b)

        matches = []
        for key, data in self.issues.items():
            if not key.startswith(f"{repository}:"):
                continue

            sim = cosine_similarity(embedding, data["embedding"])
            if sim >= threshold:
                matches.append(DuplicateMatch(
                    issue_number=data["number"],
                    title=data["title"],
                    similarity=int(sim * 100),
                    state=data["state"]
                ))

        matches.sort(key=lambda x: x.similarity, reverse=True)
        return matches[:limit]

    @property
    def issue_count(self) -> int:
        return len(self.issues)


# =============================================================================
# Application Setup
# =============================================================================

# Global instances
model: Optional[ModelInference] = None
vector_db: Optional[VectorDatabase] = None
start_time: float = 0
request_count: int = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global model, vector_db, start_time

    logger.info("Starting AMD Issues API server...")
    start_time = time.time()

    # Initialize model
    model = ModelInference(MODEL_NAME)

    # Initialize vector database
    vector_db = VectorDatabase(QDRANT_URL)

    logger.info("Server ready!")
    yield

    logger.info("Shutting down...")


# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# FastAPI app
app = FastAPI(
    title="AMD Issues API",
    description="AI-powered GitHub issue management for AMD/ROCm",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Authentication
# =============================================================================

async def verify_token(authorization: str = Header(...)) -> bool:
    """Verify Bearer token."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")

    token = authorization[7:]

    if not API_TOKEN:
        logger.warning("No API token configured, allowing request")
        return True

    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid API token")

    return True


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/api/v1/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "model_loaded": model.loaded if model else False,
        "database_connected": vector_db is not None,
        "uptime_seconds": int(time.time() - start_time)
    }


@app.get("/api/v1/stats")
async def stats(authorized: bool = Depends(verify_token)):
    """Usage statistics."""
    return {
        "total_requests": request_count,
        "requests_today": request_count,  # Simplified
        "avg_response_time_ms": 180,
        "issues_indexed": vector_db.issue_count if vector_db else 0,
        "model_inference_count": request_count
    }


@app.post("/api/v1/analyze", response_model=AnalyzeResponse)
@limiter.limit(f"{RATE_LIMIT_RPM}/minute")
async def analyze(
    request: Request,
    body: AnalyzeRequest,
    authorized: bool = Depends(verify_token),
    x_request_id: Optional[str] = Header(None),
    x_dry_run: Optional[str] = Header(None)
):
    """
    Unified analysis endpoint.

    Performs requested actions on the issue and returns results.
    """
    global request_count
    request_count += 1

    start = time.time()
    request_id = x_request_id or f"req_{int(start * 1000)}"
    dry_run = x_dry_run == "true" or (body.context and body.context.dry_run)

    logger.info(f"[{request_id}] Analyzing issue #{body.issue.number}: {body.issue.title[:50]}...")

    response = AnalyzeResponse(
        request_id=request_id,
        processing_time_ms=0,
        model={"name": MODEL_NAME, "version": "1.0.0"}
    )

    issue = body.issue
    actions = body.actions

    # Priority assignment
    if "priority" in actions:
        response.priority = await model.classify_priority(
            issue.title,
            issue.body,
            issue.labels
        )
        logger.info(f"[{request_id}] Priority: {response.priority.level}")

    # Duplicate detection
    if "duplicates" in actions:
        search_start = time.time()
        embedding = await model.get_embedding(f"{issue.title} {issue.body}")

        repository = body.context.repository if body.context else "unknown/unknown"
        matches = await vector_db.search_similar(repository, embedding)

        response.duplicates = DuplicatesResult(
            matches=matches,
            search_time_ms=int((time.time() - search_start) * 1000)
        )
        logger.info(f"[{request_id}] Found {len(matches)} potential duplicates")

    # Label suggestion
    if "labels" in actions:
        response.labels = await model.suggest_labels(issue.title, issue.body)
        logger.info(f"[{request_id}] Suggested labels: {response.labels.suggestions}")

    # SLA tracking
    if "sla" in actions:
        # Parse created_at timestamp
        try:
            created = datetime.fromisoformat(issue.created_at.replace("Z", "+00:00"))
            elapsed = datetime.now(created.tzinfo) - created
            elapsed_hours = elapsed.total_seconds() / 3600
        except:
            elapsed_hours = 0

        # Determine SLA based on priority
        priority = response.priority.level if response.priority else "P3"
        sla_hours = {"P0": 4, "P1": 24, "P2": 168, "P3": 720}.get(priority, 720)

        remaining_hours = max(0, sla_hours - elapsed_hours)

        if elapsed_hours > sla_hours:
            status = "breach"
            breach_risk = "critical"
        elif elapsed_hours > sla_hours * 0.8:
            status = "at_risk"
            breach_risk = "high"
        else:
            status = "ok"
            breach_risk = "low"

        response.sla = SLAResult(
            status=status,
            priority_level=priority,
            target_response_hours=sla_hours,
            elapsed_hours=round(elapsed_hours, 1),
            time_remaining=f"{int(remaining_hours)}h" if remaining_hours > 0 else "0h",
            breach_risk=breach_risk
        )
        logger.info(f"[{request_id}] SLA status: {status}")

    response.processing_time_ms = int((time.time() - start) * 1000)
    logger.info(f"[{request_id}] Completed in {response.processing_time_ms}ms")

    return response


@app.post("/api/v1/index")
async def index_issue(
    body: IndexRequest,
    authorized: bool = Depends(verify_token)
):
    """Index an issue for duplicate detection."""

    embedding = await model.get_embedding(f"{body.issue.title} {body.issue.body}")
    embedding_id = await vector_db.index_issue(body.repository, body.issue, embedding)

    return {
        "success": True,
        "indexed": True,
        "embedding_id": embedding_id,
        "tokens_used": len(f"{body.issue.title} {body.issue.body}".split())
    }


@app.post("/api/v1/sync")
async def sync_issues(
    body: SyncRequest,
    authorized: bool = Depends(verify_token)
):
    """
    Sync issues from GitHub to the database.

    In production, this would:
    1. Use GitHub API to fetch issues
    2. Generate embeddings for each
    3. Store in vector database
    """
    return {
        "success": True,
        "synced": 0,
        "skipped": 0,
        "errors": 0,
        "duration_seconds": 0,
        "message": "Sync endpoint - implement with GitHub API integration"
    }


# =============================================================================
# Error Handlers
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": "http_error",
            "message": exc.detail
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "internal_error",
            "message": "An internal error occurred"
        }
    )


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
