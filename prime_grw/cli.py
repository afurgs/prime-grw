"""
prime-grw CLI — standalone evaluation and serving for the GRW (Growth) operations suite.

Commands:
    prime-grw eval       Run an evaluation
    prime-grw validate   Run model-free task validation (gold + noop)
    prime-grw probes     Run the degenerate-policy probe suite
    prime-grw serve      Start an HTTP API server for remote agents
    prime-grw tasks      List task types and mixture weights

Usage:
    uv run prime-grw eval --model z-ai/glm-5.3 --n 20
    uv run prime-grw serve --port 8300
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        prog="prime-grw",
        description="GRW operations evaluation suite — match, validation, diff, extraction, brief, lifecycle"
    )
    sub = parser.add_subparsers(dest="command")

    # eval
    p_eval = sub.add_parser("eval", help="Run an evaluation")
    p_eval.add_argument("--model", "-m", required=True, help="Model ID (e.g. z-ai/glm-5.3)")
    p_eval.add_argument("--n", type=int, default=20, help="Number of tasks (default 20)")
    p_eval.add_argument("--r", "--rollouts", type=int, default=1, help="Rollouts per task")
    p_eval.add_argument("--task-type", "-t", help="Filter to one task type")
    p_eval.add_argument("--temperature", type=float, default=0.7)
    p_eval.add_argument("--max-tokens", type=int, default=3072)
    p_eval.add_argument("--concurrent", "-c", type=int, default=8)
    p_eval.add_argument("--output", "-o", help="Output directory")

    # validate
    p_val = sub.add_parser("validate", help="Model-free validation (gold + noop checks)")

    # probes
    p_probes = sub.add_parser("probes", help="Run degenerate-policy probe suite")

    # serve
    p_serve = sub.add_parser("serve", help="Start HTTP API server")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8300)

    # tasks
    p_tasks = sub.add_parser("tasks", help="List task types")

    args = parser.parse_args()

    if args.command == "eval":
        cmd = ["uv", "run", "vf-eval", "prime-grw",
               "--model", args.model,
               "--num-examples", str(args.n),
               "--rollouts-per-example", str(args.r),
               "--max-concurrent", str(args.concurrent)]
        if args.task_type:
            cmd += ["--select.include.names", f"[prime-grw/{args.task_type}]"]
        if args.output:
            cmd += ["--output-dir", args.output]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd)

    elif args.command == "validate":
        subprocess.run(["uv", "run", "vf-validate", "prime-grw",
                        "--runtime.type", "subprocess"])

    elif args.command == "probes":
        try:
            import prime_grw.prime_grw as pe
            results = pe.run_probes() if hasattr(pe, 'run_probes') else {}
            if not results:
                # Try the probes registry
                sys.path.insert(0, str(Path(__file__).parent))
                from prime_grw.probes import PROBES
                print(f"Probe registry: {list(PROBES.keys())}")
                print("Run tier-1 via: uv run pytest tests/")
            else:
                print(json.dumps(results, indent=1))
        except ImportError as e:
            print(f"Import error: {e}")
            print("Run probes via: uv run python -c 'from prime_grw.prime_grw import run_probes; print(run_probes())'")

    elif args.command == "serve":
        from prime_grw.server import app
        import uvicorn
        print(f"Starting prime-grw server on {args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)

    elif args.command == "tasks":
        TASKS = {
            "brief": 0.10, "extraction": 0.20, "validation": 0.20,
            "match": 0.15, "diff": 0.15, "lifecycle": 0.20,
        }
        print("Task types (GRW mixture):")
        for name, weight in TASKS.items():
            print(f"  {name:<15} {weight:>5.0%}")
        print(f"  {'Total':<15} {sum(TASKS.values()):>5.0%}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
