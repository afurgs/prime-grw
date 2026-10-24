"""HTTP API server for prime-grw — remote agents interact via REST."""
import json
import subprocess
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Optional

app = FastAPI(title="prime-grw", version="0.6.0")
START = time.time()


class EvalRequest(BaseModel):
    model: str = "z-ai/glm-5.3"
    n: int = Field(default=20, ge=1, le=200)
    r: int = Field(default=1, ge=1, le=3)
    task_type: Optional[str] = None


@app.get("/health")
async def health():
    return {"status": "ok", "service": "prime-grw", "uptime": time.time() - START}


@app.get("/tasks")
async def tasks():
    return {"task_types": {"brief": 0.10, "extraction": 0.20, "validation": 0.20,
                            "match": 0.15, "diff": 0.15, "lifecycle": 0.20}}


@app.get("/probes")
async def probes():
    try:
        from .prime_grw import run_probes
        return {"probes": run_probes(), "ceiling": 0.25}
    except Exception as e:
        return {"error": str(e)}


@app.post("/eval")
async def eval(req: EvalRequest):
    """Trigger an evaluation (runs synchronously for small n)."""
    import asyncio
    cmd = f"uv run vf-eval prime-grw -m {req.model} -n {req.n} -r {req.r} -c 8 --no-push"
    if req.task_type:
        cmd += f" --select.include.names [prime-grw/{req.task_type}]"
    proc = await asyncio.create_subprocess_shell(cmd,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    stdout, stderr = await proc.communicate()
    return {"exit_code": proc.returncode, "output": stdout.decode()[-2000:], "error": stderr.decode()[:500] if proc.returncode else None}


@app.get("/", response_class=HTMLResponse)
async def index():
    index = Path(__file__).parent.parent / "static" / "index.html"
    if index.exists():
        return index.read_text()
    return "<h1>prime-grw server</h1><p>POST /eval to run an evaluation.</p>"
