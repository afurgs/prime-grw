"""
prime-grw — standalone evaluation suite for GRW operations tasks.

The package IS the app. No subprocess, no CLI wrapping, no Prime Agent.
Agents interact via Python import, CLI command, or HTTP API — all in-process.

Usage:
    prime-grw eval --model z-ai/glm-5.3 --n 20
    prime-grw serve --port 8300
    prime-grw probes
    prime-grw tasks
"""
import argparse
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

# Import the taskset directly from the package
from prime_grw.prime_grw import PrimeGRWConfig, PrimeGRWTaskset, run_probes


def load_tasks(n=None, task_type=None):
    """Load tasks from the taskset."""
    cfg = PrimeGRWConfig(id="prime-grw")
    taskset = PrimeGRWTaskset(cfg)
    tasks = taskset.load()
    if task_type:
        tasks = [t for t in tasks if t.data.name.endswith(f"/{task_type}")]
    if n:
        tasks = tasks[:n]
    return tasks


async def evaluate(model: str, n: int = 20, r: int = 1,
                   task_type: str = None, temperature: float = 0.7) -> dict:
    """Run evaluation in-process via verifiers v1."""
    import verifiers.v1 as vf

    tasks = load_tasks(n=n, task_type=task_type)
    if not tasks:
        return {"error": "no tasks found"}

    agent_cfg = vf.AgentConfig(model=model, temperature=temperature)
    results = []

    async with vf.make_agent(agent_cfg) as agent:
        for task in tasks:
            for rollout in range(r):
                trace = await agent.run(task)
                reward = await task.reward(trace)
                results.append({
                    "task": task.data.name,
                    "reward": reward,
                    "stop": trace.stop_condition,
                    "tokens": trace.num_total_tokens,
                })

    # Aggregate
    by_task = defaultdict(list)
    for r_ in results:
        task_name = r_["task"].split("/")[-1]
        by_task[task_name].append(r_["reward"])

    per_task = {k: round(sum(v) / len(v), 4) for k, v in by_task.items()}
    headline = round(sum(r["reward"] for r in results) / len(results), 4)

    return {
        "model": model,
        "n": len(tasks),
        "r": r,
        "headline": headline,
        "per_task": per_task,
        "total_tokens": sum(r["tokens"] for r in results),
        "n_episodes": len(results),
    }


def cli_main():
    parser = argparse.ArgumentParser(
        prog="prime-grw",
        description="GRW operations evaluation suite — six task types, standalone."
    )
    sub = parser.add_subparsers(dest="command")

    # eval
    p = sub.add_parser("eval", help="Run an evaluation")
    p.add_argument("--model", "-m", required=True)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--r", type=int, default=1)
    p.add_argument("--task-type", "-t")
    p.add_argument("--temperature", type=float, default=0.7)

    # probes
    sub.add_parser("probes", help="Run degenerate-policy probes")

    # tasks
    sub.add_parser("tasks", help="List task types")

    # serve
    p_s = sub.add_parser("serve", help="Start HTTP API server")
    p_s.add_argument("--host", default="0.0.0.0")
    p_s.add_argument("--port", type=int, default=8300)

    args = parser.parse_args()

    if args.command == "eval":
        result = asyncio.run(evaluate(
            model=args.model, n=args.n, r=args.r,
            task_type=args.task_type, temperature=args.temperature
        ))
        print(json.dumps(result, indent=2))

    elif args.command == "probes":
        results = run_probes()
        print(json.dumps(results, indent=1))
        verdict = "PASS" if max(results.values()) < 0.25 else "FAIL"
        print(f"\nCeiling: 0.25 | Verdict: {verdict}")

    elif args.command == "tasks":
        tasks = load_tasks()
        counts = defaultdict(int)
        for t in tasks:
            counts[t.data.name.split("/")[-1]] += 1
        total = len(tasks)
        print(f"Task types ({total} tasks):")
        for name, count in sorted(counts.items()):
            print(f"  {name:<15} {count:>3} tasks ({count/total:.0%})")

    elif args.command == "serve":
        from prime_grw.server import create_app
        import uvicorn
        app = create_app(evaluate_fn=evaluate)
        print(f"prime-grw serving on {args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)

    else:
        parser.print_help()


if __name__ == "__main__":
    cli_main()
