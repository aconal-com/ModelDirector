"""Real-world benchmark: run a battery of diverse prompts through ModelDirector
and report the cost savings vs. a naive 'always use the premium model' baseline.

Output is printed to stdout (table + JSON) and written to
``benchmarks/output/results.json`` so the README can be generated from it.

Usage:
    OPENROUTER_API_KEY=... .venv/bin/python -m benchmarks.run_benchmark
    OPENROUTER_API_KEY=... .venv/bin/python -m benchmarks.run_benchmark --config examples/config.yaml
    OPENROUTER_API_KEY=... .venv/bin/python -m benchmarks.run_benchmark --out benchmarks/output/results.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# Allow running as `python -m benchmarks.run_benchmark` from the project root.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from modeldirector.loader import load_config  # noqa: E402
from modeldirector.models import SelectionResult  # noqa: E402
from modeldirector.selector import ModelDirector, SelectorError  # noqa: E402


# --- The benchmark prompt battery ---------------------------------------------

# 30 tasks, intentionally spanning a wide difficulty range.  Each entry is
# (category, prompt) so we can report per-category breakdowns.
TASKS: list[tuple[str, str]] = [
    # Trivial - cheap model should be enough
    ("trivial", "Translate the word 'hello' to French."),
    ("trivial", "What is 2 + 2?"),
    ("trivial", "Sort this list: [3, 1, 4, 1, 5, 9, 2, 6]"),
    ("trivial", "Convert 'cat' to uppercase."),
    ("trivial", "What's the capital of Japan?"),

    # Simple Q&A / lookup
    ("simple_qa", "List three primary colours."),
    ("simple_qa", "Name a programming language created by Guido van Rossum."),
    ("simple_qa", "Who wrote 'Pride and Prejudice'?"),
    ("simple_qa", "What HTTP status code means 'Not Found'?"),
    ("simple_qa", "Define the term 'API' in one sentence."),

    # Short summarisation
    ("summarisation", "Summarise this in one sentence: The quick brown fox jumps over the lazy dog. The dog barks. The fox runs away."),
    ("summarisation", "TLDR this commit message: 'Fix off-by-one error in pagination cursor when offset equals total count.'"),
    ("summarisation", "Summarise: 'Paris is the capital of France. It has the Eiffel Tower. It is in Europe.'"),

    # Light coding
    ("light_coding", "Write a Python function that returns the length of a string."),
    ("light_coding", "Write a regex that matches valid email addresses."),
    ("light_coding", "Fix the bug: `def add(a, b): return a - b`"),
    ("light_coding", "Write a one-liner to flatten a list of lists in Python."),

    # Medium coding
    ("medium_coding", "Write a Python class for a Stack with push, pop, and is_empty methods."),
    ("medium_coding", "Implement binary search in Python with O(log n) time complexity."),
    ("medium_coding", "Write a SQL query to find the second-highest salary from an Employee table."),
    ("medium_coding", "Write a function to debounce API calls in JavaScript."),

    # Complex reasoning
    ("reasoning", "If a train leaves at 9am going 60 mph and another at 10am going 80 mph in the same direction, when does the second catch up?"),
    ("reasoning", "Explain the difference between TCP and UDP in two paragraphs."),
    ("reasoning", "A farmer has 17 sheep. All but 9 die. How many are left?"),

    # Complex coding / architecture
    ("hard_coding", "Design a thread-safe LRU cache in Python. Include the eviction policy and concurrency strategy."),
    ("hard_coding", "Write a function to merge k sorted linked lists. Explain the time complexity."),
    ("hard_coding", "Design a URL shortener like bit.ly. Cover hashing, collisions, and database schema."),
    ("hard_coding", "Implement a simple Raft consensus algorithm in pseudocode, covering leader election and log replication."),

    # Long-form writing
    ("writing", "Write a polite out-of-office email for next week."),
    ("writing", "Write a haiku about Python."),
]


# --- Result types ------------------------------------------------------------


@dataclass
class TaskResult:
    category: str
    prompt: str
    selected_model: str
    selected_cost: float
    premium_cost: float
    savings: float               # 0.0 - 1.0
    overall_scores: dict[str, int]
    selector_latency_s: float
    reason: str
    policy: str


@dataclass
class BenchmarkReport:
    timestamp: str
    config_path: str
    selector_model: str
    baseline_model: str
    baseline_cost: float
    tasks: list[TaskResult] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)


# --- Core run ----------------------------------------------------------------


def run_benchmark(config_path: Path, baseline_id: str) -> BenchmarkReport:
    cfg = load_config(config_path)
    director = ModelDirector(cfg)
    profiles_by_id = {m.id: m for m in cfg.models}
    if baseline_id not in profiles_by_id:
        raise ValueError(
            f"baseline model '{baseline_id}' not in config. "
            f"Available: {list(profiles_by_id)}"
        )
    baseline_cost = float(profiles_by_id[baseline_id].cost or 0.0)

    report = BenchmarkReport(
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        config_path=str(config_path),
        selector_model=f"{cfg.selector.provider}/{cfg.selector.model}",
        baseline_model=baseline_id,
        baseline_cost=baseline_cost,
    )

    for i, (category, prompt) in enumerate(TASKS, start=1):
        sys.stdout.write(f"\r  [{i:>2}/{len(TASKS)}] {category:<15} ", )
        sys.stdout.flush()
        t0 = time.perf_counter()
        try:
            result: SelectionResult = director.select(prompt)
        except SelectorError as e:
            report.errors.append({"category": category, "prompt": prompt, "error": str(e)})
            print(f"  [ERROR] {e}")
            continue
        elapsed = time.perf_counter() - t0

        chosen_cost = float(profiles_by_id[result.selected_model].cost or 0.0)
        savings = (baseline_cost - chosen_cost) / baseline_cost if baseline_cost else 0.0

        report.tasks.append(
            TaskResult(
                category=category,
                prompt=prompt,
                selected_model=result.selected_model,
                selected_cost=chosen_cost,
                premium_cost=baseline_cost,
                savings=savings,
                overall_scores={k: v.overall for k, v in result.scores.items()},
                selector_latency_s=elapsed,
                reason=result.reason,
                policy=result.policy,
            )
        )

    print()  # newline after the progress line
    return report


# --- Reporting ---------------------------------------------------------------


def _pct(n: float) -> str:
    return f"{n * 100:.1f}%"


def render_report(report: BenchmarkReport) -> str:
    if not report.tasks:
        return "No tasks completed."

    # Per-model counts
    picks: dict[str, int] = {}
    for t in report.tasks:
        picks[t.selected_model] = picks.get(t.selected_model, 0) + 1

    # Per-category savings
    by_cat: dict[str, list[float]] = {}
    for t in report.tasks:
        by_cat.setdefault(t.category, []).append(t.savings)

    total_savings = sum(t.savings for t in report.tasks) / len(report.tasks)
    latency = [t.selector_latency_s for t in report.tasks]
    p50 = statistics.median(latency)
    p95 = sorted(latency)[int(0.95 * len(latency)) - 1]

    lines = []
    lines.append("")
    lines.append("=" * 70)
    lines.append("  ModelDirector benchmark")
    lines.append("=" * 70)
    lines.append(f"  Timestamp         : {report.timestamp}")
    lines.append(f"  Config            : {report.config_path}")
    lines.append(f"  Selector LLM      : {report.selector_model}")
    lines.append(f"  Baseline (always) : {report.baseline_model} (cost={report.baseline_cost})")
    lines.append(f"  Tasks run         : {len(report.tasks)} / {len(TASKS)}")
    if report.errors:
        lines.append(f"  Errors            : {len(report.errors)}")
    lines.append("")

    lines.append("  Model picks")
    lines.append("  " + "-" * 40)
    for model_id, count in sorted(picks.items(), key=lambda x: -x[1]):
        bar = "#" * count
        lines.append(f"    {model_id:<10} {count:>3}  {bar}")
    lines.append("")

    lines.append("  Savings vs always-premium baseline")
    lines.append("  " + "-" * 40)
    lines.append(f"    Mean savings    : {_pct(total_savings)}")
    lines.append(f"    Per-category    :")
    for cat, savings in sorted(by_cat.items()):
        lines.append(f"      {cat:<15}  mean savings {_pct(sum(savings) / len(savings))}  (n={len(savings)})")
    lines.append("")

    lines.append("  Selector latency")
    lines.append("  " + "-" * 40)
    lines.append(f"    p50  : {p50:.2f}s")
    lines.append(f"    p95  : {p95:.2f}s")
    lines.append(f"    max  : {max(latency):.2f}s")
    lines.append("")

    # Detailed per-task output
    lines.append("  Per-task detail")
    lines.append("  " + "-" * 70)
    lines.append(f"    {'category':<15} {'picked':<10} {'savings':<8} {'latency':<8} prompt")
    for t in report.tasks:
        prompt_short = t.prompt[:50] + ("..." if len(t.prompt) > 50 else "")
        lines.append(
            f"    {t.category:<15} {t.selected_model:<10} {_pct(t.savings):<8} "
            f"{t.selector_latency_s:.2f}s     {prompt_short}"
        )
    lines.append("")

    return "\n".join(lines)


# --- CLI ---------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the ModelDirector benchmark.")
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=_ROOT / "examples" / "config.yaml",
        help="Path to a ModelDirector YAML config.",
    )
    parser.add_argument(
        "--baseline",
        "-b",
        default="premium",
        help="Model id to use as the cost baseline (default: 'premium').",
    )
    parser.add_argument(
        "--out",
        "-o",
        type=Path,
        default=_ROOT / "benchmarks" / "output" / "results.json",
        help="Where to write the raw JSON report.",
    )
    args = parser.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY not set.", file=sys.stderr)
        return 1
    if not args.config.exists():
        print(f"ERROR: config not found: {args.config}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = run_benchmark(args.config, args.baseline)
    print(render_report(report))
    args.out.write_text(json.dumps(asdict(report), indent=2))
    print(f"  Raw JSON written to: {args.out}")
    return 0 if not report.errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
