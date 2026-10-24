"""
prime-grw HTTP server — remote agents interact via REST.

Calls the evaluation directly (in-process), not via subprocess.
"""
import json
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

START = time.time()


def create_app(evaluate_fn=None):
    """Create the FastAPI app. The evaluate_fn is the in-process evaluation function."""
    app = FastAPI(title="prime-grw", version="0.6.0")

    class EvalRequest(BaseModel):
        model: str = "z-ai/glm-5.3"
        n: int = Field(default=20, ge=1, le=200)
        r: int = Field(default=1, ge=1, le=3)
        task_type: Optional[str] = None
        temperature: float = 0.7

    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "prime-grw", "uptime": time.time() - START}

    @app.get("/tasks")
    async def tasks():
        from prime_grw.prime_grw import PrimeGRWConfig, PrimeGRWTaskset
        from collections import defaultdict
        cfg = PrimeGRWConfig(id="prime-grw")
        taskset = PrimeGRWTaskset(cfg)
        all_tasks = taskset.load()
        counts = defaultdict(int)
        for t in all_tasks:
            counts[t.data.name.split("/")[-1]] += 1
        return {"task_types": dict(counts), "total": len(all_tasks)}

    @app.get("/probes")
    async def probes():
        from prime_grw.prime_grw import run_probes
        results = run_probes()
        return {"probes": results, "ceiling": 0.25,
                "verdict": "PASS" if max(results.values()) < 0.25 else "FAIL"}

    @app.post("/eval")
    async def eval(req: EvalRequest):
        """Run an evaluation synchronously (in-process, no subprocess)."""
        if evaluate_fn is None:
            return {"error": "evaluate_fn not configured"}
        import asyncio
        result = await evaluate_fn(
            model=req.model, n=req.n, r=req.r,
            task_type=req.task_type, temperature=req.temperature
        )
        return result

    @app.get("/", response_class=HTMLResponse)
    async def index():
        static = Path(__file__).parent.parent / "static" / "index.html"
        if static.exists():
            return static.read_text()
        return "<h1>prime-grw</h1><p>POST /eval to run an evaluation.</p>"

    return app
