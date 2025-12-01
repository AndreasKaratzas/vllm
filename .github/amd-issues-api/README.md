# AMD Issues API

Self-hosted AI-powered GitHub issue management server for AMD/ROCm issues.

## Features

- **Priority Assignment** - AI classifies issues as P0-P3 based on content
- **Duplicate Detection** - Vector similarity search finds related issues
- **Label Suggestion** - Intelligent label recommendations
- **SLA Tracking** - Monitor response times and flag breaches

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  GitHub Action  │────▶│  AMD Issues API  │────▶│  Qdrant Vector  │
│  (Workflow)     │     │  (FastAPI)       │     │  Database       │
└─────────────────┘     └────────┬─────────┘     └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  LLM Backend    │
                        │  (Qwen/Ollama/  │
                        │   vLLM)         │
                        └─────────────────┘
```

## Quick Start

### 1. Generate API Token

```bash
# Generate a secure token (min 32 characters)
openssl rand -hex 32
```

### 2. Create Environment File

```bash
cat > .env << EOF
AMD_API_TOKEN=your_generated_token_here
MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
RATE_LIMIT_RPM=100
EOF
```

### 3. Deploy with Docker Compose

```bash
# Basic deployment (CPU inference)
docker compose up -d

# With Ollama for local LLM
docker compose --profile ollama up -d

# With vLLM for GPU inference
docker compose --profile gpu up -d
```

### 4. Configure GitHub Secrets

In your repository settings, add:

| Secret | Value |
|--------|-------|
| `AMD_ISSUES_API_TOKEN` | Your generated token |

### 5. Update Workflow

Edit `.github/workflows/prototype_amd_issues_api.yml`:

```yaml
env:
  AMD_ISSUES_API_URL: 'https://your-server.example.com/api/v1'
```

## Deployment Options

### Option A: Local/VM Deployment

Best for: Testing, small teams

```bash
# Install dependencies
cd server
pip install -r requirements.txt

# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Start API server
AMD_API_TOKEN=your_token uvicorn main:app --host 0.0.0.0 --port 8000
```

### Option B: Cloud Deployment (Recommended)

Best for: Production, public repositories

**AWS/GCP/Azure:**
1. Deploy on a VM with GPU (for vLLM) or CPU (for Ollama/llama.cpp)
2. Set up HTTPS with reverse proxy (nginx/Caddy)
3. Configure firewall to allow GitHub Actions IPs

**Railway/Render/Fly.io:**
1. Deploy using `docker-compose.yml`
2. Configure environment variables
3. Get public HTTPS URL

### Option C: Kubernetes

Best for: Large scale, high availability

```yaml
# Example Helm values
amd-issues-api:
  replicas: 2
  resources:
    requests:
      memory: "4Gi"
      cpu: "2"
    limits:
      memory: "8Gi"
      cpu: "4"
  env:
    AMD_API_TOKEN:
      secretKeyRef:
        name: amd-issues-api
        key: token
```

## LLM Backend Options

### Ollama (Easiest)

```bash
# Pull model
docker exec ollama ollama pull qwen2.5:7b

# Or for smaller footprint
docker exec ollama ollama pull qwen2.5:3b
```

### vLLM (Fastest, requires GPU)

```bash
# Included in docker-compose with --profile gpu
docker compose --profile gpu up -d
```

### llama.cpp (CPU-friendly)

```bash
# Download quantized model
wget https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf

# Run with llama.cpp
./llama-server -m qwen2.5-7b-instruct-q4_k_m.gguf -c 4096
```

## Security

### Token Validation

All requests require a Bearer token:

```
Authorization: Bearer <AMD_ISSUES_API_TOKEN>
```

### Rate Limiting

Default: 100 requests/minute per token

Configure with `RATE_LIMIT_RPM` environment variable.

### Network Security

1. **Firewall**: Only allow GitHub Actions IPs
   - See: https://api.github.com/meta (actions key)

2. **HTTPS**: Always use TLS in production

3. **Token Rotation**: Rotate tokens periodically

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | Health check |
| `/api/v1/stats` | GET | Usage statistics |
| `/api/v1/analyze` | POST | Unified analysis |
| `/api/v1/index` | POST | Index issue |
| `/api/v1/sync` | POST | Batch sync |

See [API_SPECIFICATION.md](./API_SPECIFICATION.md) for full documentation.

## Customization

### Priority Rules

Edit `ModelInference.classify_priority()` in `server/main.py`:

```python
# Add your own keywords
p0_keywords = ["production", "crash", "your_keyword"]
```

### Label Detection

Edit `ModelInference.suggest_labels()`:

```python
# Add hardware detection
if "your_hardware" in text:
    suggestions.append("your-label")
```

### Fine-Tuning

For best results, fine-tune the model on your issue history:

1. Export issues: `gh issue list --json number,title,body,labels`
2. Format as training data
3. Fine-tune with PEFT/LoRA
4. Deploy custom model

## Monitoring

### Logs

```bash
docker compose logs -f api
```

### Metrics

The `/api/v1/stats` endpoint provides:
- Total requests
- Issues indexed
- Model inference count

### Health Checks

```bash
curl http://localhost:8000/api/v1/health
```

## Troubleshooting

### Common Issues

**API returns 401:**
- Check `AMD_API_TOKEN` matches in server and GitHub secret

**Slow responses:**
- Use GPU backend (vLLM)
- Use smaller model (3B instead of 7B)
- Use quantized model

**Out of memory:**
- Reduce `max-model-len`
- Use quantized model
- Increase server RAM

**No duplicates found:**
- Run `/api/v1/sync` to index existing issues
- Check vector database is running

## Contributing

1. Fork the repository
2. Create feature branch
3. Submit pull request

## License

MIT
