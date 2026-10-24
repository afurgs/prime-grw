# prime-grw

**GRW operations evaluation suite** — match scoring, schema validation, diff classification, transcript extraction, ops brief generation, and lifecycle decisions. Standalone package, installable from GitHub, no platform dependency.

## Install

```bash
# From GitHub
git clone https://github.com/afurgs/prime-grw.git
cd prime-grw
uv sync --extra full

# Or as a dependency
uv add "prime-grw @ git+https://github.com/afurgs/prime-grw"
```

## Run

```bash
# Run an evaluation
uv run prime-grw eval --model z-ai/glm-5.3 --n 20

# Validate tasks (model-free)
uv run prime-grw validate

# Run probes
uv run prime-grw probes

# List task types
uv run prime-grw tasks

# Start HTTP API server (for remote agents)
uv run prime-grw serve --port 8300
```

## Task Types

| Type | Weight | What It Tests |
|---|---|---|
| brief | 10% | Ops brief generation from growth data |
| extraction | 20% | Meeting-transcript deal intelligence |
| validation | 20% | Record schema validation (slug-gated) |
| match | 15% | Supply-demand match scoring (banded) |
| diff | 15% | CRM sync diff classification |
| lifecycle | 20% | Deal lifecycle disposition decisions |

## HTTP API

When serving (`prime-grw serve`), remote agents can interact via REST:

```bash
# Health check
curl http://localhost:8300/health

# List tasks
curl http://localhost:8300/tasks

# Run probes
curl http://localhost:8300/probes

# Trigger evaluation
curl -X POST http://localhost:8300/eval \
  -H "Content-Type: application/json" \
  -d '{"model": "z-ai/glm-5.3", "n": 20}'
```

## Architecture

```
Agent → CLI / HTTP → prime-grw package → verifiers v1 → evaluation
```

No Prime Agent. No Hub. No MCP. Just Python + HTTP.

## License

MIT
