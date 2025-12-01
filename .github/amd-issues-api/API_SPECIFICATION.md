# AMD Issues API Specification

Self-hosted API server for AMD issue management with AI-powered analysis.

## Overview

This API provides a centralized endpoint for:
- **Priority Assignment** - AI-based priority classification (P0-P3)
- **Duplicate Detection** - Vector similarity search against issue database
- **Label Suggestion** - Intelligent label recommendations
- **SLA Tracking** - Response time monitoring and breach detection

## Authentication

All requests require Bearer token authentication:

```
Authorization: Bearer <AMD_ISSUES_API_TOKEN>
```

### Token Requirements
- Minimum 32 characters
- Should be rotated periodically
- Store in GitHub Secrets as `AMD_ISSUES_API_TOKEN`

### Rate Limiting
- Default: 100 requests/minute per token
- Burst: 10 requests/second
- Returns `429 Too Many Requests` with `Retry-After` header when exceeded

---

## Endpoints

### POST /api/v1/analyze

Unified endpoint for all analysis operations.

#### Request Headers

| Header | Required | Description |
|--------|----------|-------------|
| `Authorization` | Yes | Bearer token |
| `Content-Type` | Yes | `application/json` |
| `X-GitHub-Repository` | Yes | `owner/repo` format |
| `X-Request-ID` | No | Unique request identifier for tracing |
| `X-Dry-Run` | No | If `true`, returns analysis without side effects |

#### Request Body

```json
{
  "issue": {
    "number": 12345,
    "title": "ROCm 6.3 fails to compile with MI300X",
    "body": "Getting compilation errors when building vLLM...",
    "labels": ["bug", "rocm"],
    "state": "open",
    "created_at": "2025-01-15T10:30:00Z",
    "updated_at": "2025-01-15T10:30:00Z",
    "author": "username",
    "comments_count": 2,
    "recent_comments": [
      {
        "author": "maintainer",
        "body": "Can you provide more details?",
        "created_at": "2025-01-15T11:00:00Z"
      }
    ],
    "url": "https://github.com/owner/repo/issues/12345"
  },
  "actions": ["priority", "duplicates", "labels", "sla"],
  "context": {
    "repository": "owner/repo",
    "event": "issues.opened",
    "dry_run": false
  }
}
```

#### Response

```json
{
  "success": true,
  "request_id": "abc123",
  "processing_time_ms": 245,
  "model": {
    "name": "qwen2.5-8b-instruct",
    "version": "1.0.0"
  },
  "priority": {
    "level": "P1",
    "confidence": 85,
    "reason": "Production blocker affecting MI300X compilation with ROCm 6.3"
  },
  "duplicates": {
    "matches": [
      {
        "issue_number": 11234,
        "title": "MI300X build failure with ROCm 6.2",
        "similarity": 78,
        "state": "closed",
        "resolution": "fixed"
      }
    ],
    "search_time_ms": 45
  },
  "labels": {
    "suggestions": ["mi300x", "rocm-6.x", "build-failure", "needs-triage"],
    "explanations": {
      "mi300x": "Issue mentions MI300X hardware",
      "rocm-6.x": "ROCm 6.3 version detected",
      "build-failure": "Compilation/build error described",
      "needs-triage": "New issue requiring team review"
    },
    "confidence": 92
  },
  "sla": {
    "status": "ok",
    "priority_level": "P1",
    "target_response_hours": 24,
    "elapsed_hours": 0.5,
    "time_remaining": "23h 30m",
    "breach_risk": "low"
  }
}
```

---

### POST /api/v1/index

Index a new issue into the duplicate detection database.

#### Request Body

```json
{
  "issue": {
    "number": 12345,
    "title": "Issue title",
    "body": "Issue body",
    "labels": ["bug"],
    "state": "open",
    "created_at": "2025-01-15T10:30:00Z"
  },
  "repository": "owner/repo"
}
```

#### Response

```json
{
  "success": true,
  "indexed": true,
  "embedding_id": "emb_abc123",
  "tokens_used": 156
}
```

---

### POST /api/v1/sync

Batch sync issues from GitHub to the database.

#### Request Body

```json
{
  "repository": "owner/repo",
  "since": "2025-01-01T00:00:00Z",
  "state": "all",
  "labels": ["amd", "rocm"]
}
```

#### Response

```json
{
  "success": true,
  "synced": 150,
  "skipped": 23,
  "errors": 0,
  "duration_seconds": 45
}
```

---

### GET /api/v1/health

Health check endpoint.

#### Response

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "model_loaded": true,
  "database_connected": true,
  "uptime_seconds": 86400
}
```

---

### GET /api/v1/stats

Usage statistics.

#### Response

```json
{
  "total_requests": 15234,
  "requests_today": 342,
  "avg_response_time_ms": 180,
  "issues_indexed": 5678,
  "model_inference_count": 12000
}
```

---

## Error Responses

### 400 Bad Request

```json
{
  "success": false,
  "error": "validation_error",
  "message": "Issue number is required",
  "details": {
    "field": "issue.number",
    "constraint": "required"
  }
}
```

### 401 Unauthorized

```json
{
  "success": false,
  "error": "unauthorized",
  "message": "Invalid or missing API token"
}
```

### 429 Rate Limited

```json
{
  "success": false,
  "error": "rate_limited",
  "message": "Rate limit exceeded",
  "retry_after": 30
}
```

### 500 Internal Server Error

```json
{
  "success": false,
  "error": "internal_error",
  "message": "Model inference failed",
  "request_id": "abc123"
}
```

---

## Priority Classification Rules

The model uses these guidelines for priority assignment:

| Priority | Criteria | Target Response |
|----------|----------|-----------------|
| **P0** | Production down, security vulnerability, data loss | 4 hours |
| **P1** | Major feature broken, blocking multiple users | 24 hours |
| **P2** | Feature degraded, workaround available | 1 week |
| **P3** | Minor issue, enhancement request, nice-to-have | 1 month |

### Keywords that influence priority:

**P0 indicators:**
- "production", "crash", "data loss", "security", "urgent", "critical"
- "cannot use", "completely broken", "blocks all"

**P1 indicators:**
- "blocker", "regression", "major", "broken"
- "no workaround", "affects many"

**P2 indicators:**
- "workaround", "intermittent", "sometimes"
- "degraded", "slow"

**P3 indicators:**
- "enhancement", "feature request", "nice to have"
- "minor", "cosmetic", "documentation"

---

## Duplicate Detection

Uses vector embeddings for semantic similarity:

1. **Indexing**: New issues are embedded using the model and stored in vector DB
2. **Search**: Query embedding compared against database using cosine similarity
3. **Threshold**: Matches above 70% similarity are returned

### Similarity Interpretation

| Similarity | Interpretation |
|------------|----------------|
| 90-100% | Almost certainly duplicate |
| 80-89% | Likely duplicate, manual review recommended |
| 70-79% | Possibly related, worth checking |
| <70% | Probably different issues |

---

## Label Categories

### Hardware Labels
- `mi300x`, `mi325x`, `mi350`

### Software Labels
- `rocm-6.x`, `rocm-7.x`
- `aiter-backend`, `rccl`, `hipblaslt`

### Type Labels
- `bug`, `feature`, `docs`, `performance`, `infrastructure`

### Status Labels
- `needs-triage`, `needs-info`, `confirmed`, `in-progress`

### Priority Labels
- `P0`, `P1`, `P2`, `P3`

---

## Deployment Notes

### Recommended Infrastructure

- **Compute**: 1x GPU (A10G/L4) for model inference, or CPU with quantized model
- **Memory**: 16GB+ RAM
- **Storage**: 50GB+ for vector database
- **Model**: Qwen2.5-8B-Instruct (or 3B for lower resource usage)

### Vector Database Options

- **Qdrant** - Recommended, easy to deploy
- **Milvus** - Scalable, more complex
- **Chroma** - Simple, good for small scale
- **pgvector** - If already using PostgreSQL

### Model Serving Options

- **vLLM** - High throughput, production ready
- **Ollama** - Easy local deployment
- **llama.cpp** - CPU-friendly, quantized models
- **TGI** - Hugging Face's solution
