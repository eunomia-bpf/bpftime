#!/usr/bin/env python3
"""Assemble the GPU verifier paper summary from generated benchmark artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent

CORRECTNESS_MD = SCRIPT_DIR / "correctness_results.md"
CORRECTNESS_CSV = SCRIPT_DIR / "correctness_results.csv"
FALSE_POSITIVE_MD = SCRIPT_DIR / "false_positive_results.md"
FALSE_POSITIVE_CSV = SCRIPT_DIR / "false_positive_results.csv"
PERF_MD = SCRIPT_DIR / "perf_breakdown_results.md"
PERF_JSON = SCRIPT_DIR / "perf_breakdown_results.json"
COMPARISON_MD = SCRIPT_DIR / "comparison_table.md"
SUMMARY_MD = SCRIPT_DIR / "EVAL_SUMMARY.md"


def require_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    return path.read_text(encoding="utf-8").strip()


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    correctness_rows = load_csv_rows(CORRECTNESS_CSV)
    false_positive_rows = load_csv_rows(FALSE_POSITIVE_CSV)
    perf_rows = load_json(PERF_JSON)

    correctness_matches = sum(
        1 for row in correctness_rows if row["Expected"] == row["Result"]
    )
    correctness_total = len(correctness_rows)

    example_total = len(false_positive_rows)
    pass_count = sum(
        1 for row in false_positive_rows if row["Classification"] == "pass"
    )
    true_positive_count = sum(
        1 for row in false_positive_rows if row["Classification"] == "true positive"
    )
    false_positive_count = sum(
        1 for row in false_positive_rows if row["Classification"] == "false positive"
    )
    true_positive_examples = [
        row["Example"]
        for row in false_positive_rows
        if row["Classification"] == "true positive"
    ]
    false_positive_examples = [
        row["Example"]
        for row in false_positive_rows
        if row["Classification"] == "false positive"
    ]

    smallest_perf = perf_rows[0]
    largest_perf = perf_rows[-1]
    max_prevail = max(float(row["prevail_time_us"]) for row in perf_rows)

    summary = (
        "# GPU Verifier Evaluation Summary\n\n"
        "This file consolidates the regenerated Release-build evaluation artifacts in "
        "Markdown tables suitable for downstream LaTeX conversion.\n\n"
        "## Key Findings\n\n"
        f"- RQ1 correctness: {correctness_matches}/{correctness_total} rows matched the expected verdict.\n"
        f"- RQ2 example sweep: {pass_count} pass, {true_positive_count} true positive, and "
        f"{false_positive_count} false positive results across {example_total} `example/gpu/*/*.bpf.c` files.\n"
        f"- RQ3 performance: median total verifier time rises from {smallest_perf['total_time_us']:.3f} μs at "
        f"size {int(smallest_perf['size'])} to {largest_perf['total_time_us']:.3f} μs at size "
        f"{int(largest_perf['size'])}; median PREVAIL time stays at {max_prevail:.3f} μs or below because "
        "`skip_prevail=true` is the default path.\n"
        "- RQ6 comparison: the SIMT-aware verifier catches GPU-specific divergence, prohibited-helper, "
        "varying-atomic, varying-key, and helper-budget patterns that `no verification` misses, while "
        "standard PREVAIL design coverage remains focused on classic eBPF safety classes.\n\n"
        "## Numbers To Cite\n\n"
        "| Metric | Value |\n"
        "| --- | --- |\n"
        f"| Correctness rows matching expectation | {correctness_matches} / {correctness_total} |\n"
        f"| Example corpus size | {example_total} |\n"
        f"| Example corpus true positives | {true_positive_count} |\n"
        f"| Example corpus false positives | {false_positive_count} |\n"
        f"| Perf median total @ {int(smallest_perf['size'])} instructions | {smallest_perf['total_time_us']:.3f} μs |\n"
        f"| Perf median total @ {int(largest_perf['size'])} instructions | {largest_perf['total_time_us']:.3f} μs |\n"
        f"| Perf median SIMT pass @ {int(largest_perf['size'])} instructions | {largest_perf['simt_time_us']:.3f} μs |\n"
        f"| Largest median PREVAIL time in perf JSON | {max_prevail:.3f} μs |\n\n"
        "## RQ1 - Correctness Table\n\n"
        f"{require_text(CORRECTNESS_MD)}\n\n"
        "## RQ2 - False Positive Analysis\n\n"
        f"{require_text(FALSE_POSITIVE_MD)}\n\n"
        "## RQ3 - Performance Breakdown\n\n"
        f"{require_text(PERF_MD)}\n\n"
        "## RQ6 - Comparison Table\n\n"
        f"{require_text(COMPARISON_MD)}\n"
    )

    if true_positive_examples:
        summary += (
            "\nExamples classified as `true positive`: "
            + ", ".join(f"`{name}`" for name in true_positive_examples)
            + ".\n"
        )
    if false_positive_examples:
        summary += (
            "Examples classified as `false positive`: "
            + ", ".join(f"`{name}`" for name in false_positive_examples)
            + ".\n"
        )

    SUMMARY_MD.write_text(summary, encoding="utf-8")
    print(f"Wrote {SUMMARY_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
