"""Automated Benchmark Runner for Streaming Live RAG (Theme 04, Samsung PRISM Hackathon).

Evaluates Gates G1 through G6 and emits a formatted summary scorecard.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from benchmarks.evaluator import BenchmarkEvaluator, BenchmarkSummary


def run_benchmark_suite() -> int:
    """Executes the full benchmark evaluation and prints scorecard."""
    print("=" * 86)
    print("      STREAMING LIVE RAG ENGINE -- BENCHMARK VALIDATION HARNESS (THEME 04)")
    print("      Strict SRD-SLRAG-001 & PRD-SLRAG-001 Governance Verification")
    print("=" * 86)

    evaluator = BenchmarkEvaluator()
    summary: BenchmarkSummary = evaluator.evaluate_all()

    print(f"\nTotal Fixture Scenarios Evaluated: {summary.total_test_cases}\n")
    print("-" * 86)
    print(f"{'GATE':<36} | {'TARGET':<16} | {'ACTUAL':<16} | {'STATUS'}")
    print("-" * 86)

    for g in summary.gate_results:
        status_clean = "[PASS]" if g.passed else "[FAIL]"
        print(f"{g.name:<36} | {g.target:<16} | {g.actual:<16} | {status_clean}")
        print(f"  +-- Details: {g.details}")

    print("-" * 86)

    if summary.all_passed:
        print("\n>>> ALL 6 GATES SUCCESSFULLY PASSED (100% COMPLIANT) <<<")
        print(">>> Ready for Production Deployment and Docker Containerization.\n")
        return 0
    else:
        print("\n>>> BENCHMARK FAILED: One or more gates did not meet target <<<\n")
        return 1


if __name__ == "__main__":
    exit_code = run_benchmark_suite()
    sys.exit(exit_code)
