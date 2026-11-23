#!/usr/bin/env python3
"""Summarize the CI micro-benchmark results as Markdown and flat JSON.

Reads the result files written by benchmark/uprobe/benchmark.py,
benchmark/syscall/benchmark.py and benchmark/mpk/benchmark.py, and emits:

* a flat JSON object mapping "suite/environment/operation" to the average
  time in nanoseconds (used as the baseline for later runs), and
* a Markdown table, optionally compared against such a baseline file,
  that the Benchmarks workflow posts as a sticky pull-request comment.

Missing or malformed result files are reported in the output instead of
failing, so a partially failed benchmark run still produces a summary.
"""
import argparse
import glob
import json
import os
import sys

# Hidden marker used by the workflow to find and update its own comment.
MARKER = "<!-- bpftime-benchmark-summary -->"


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        print(f"warning: cannot read {path}: {e}", file=sys.stderr)
        return None


def collect_avgs(metrics, prefix, out):
    """Add "<prefix>/<op>" -> avg for every op dict that carries an avg."""
    if not isinstance(metrics, dict):
        return
    for op, value in metrics.items():
        if isinstance(value, dict) and isinstance(value.get("avg"), (int, float)):
            out[f"{prefix}/{op}"] = float(value["avg"])


def collect_results(root):
    results = {}
    missing = []
    skipped = []

    uprobe = load_json(os.path.join(root, "uprobe", "benchmark-output.json"))
    if uprobe is None:
        missing.append("uprobe")
    else:
        for env in ("kernel_uprobe", "userspace_uprobe", "embed"):
            metrics = uprobe.get(env)
            # benchmark.py writes {"skipped": "<reason>"} for an
            # environment it could not run (e.g. kernel uprobe without sudo).
            if isinstance(metrics, dict) and "skipped" in metrics:
                skipped.append((f"uprobe/{env}", str(metrics["skipped"])))
                continue
            collect_avgs(metrics, f"uprobe/{env}", results)

    syscall_files = sorted(
        glob.glob(os.path.join(root, "syscall", "syscall_benchmark_results_*.json"))
    )
    syscall = load_json(syscall_files[-1]) if syscall_files else None
    if syscall is None:
        missing.append("syscall")
    else:
        for env, value in (syscall.get("results") or {}).items():
            if isinstance(value, dict) and isinstance(value.get("avg"), (int, float)):
                results[f"syscall/{env}"] = float(value["avg"])

    mpk = load_json(os.path.join(root, "mpk", "benchmark-output.json"))
    if mpk is None:
        missing.append("mpk")
    else:
        for env in ("mpk", "standard"):
            collect_avgs(mpk.get(env), f"mpk/{env}", results)

    return results, missing, skipped


def format_delta(current, base):
    if not base:
        return "n/a"
    return f"{(current - base) / base * 100:+.2f}%"


def short_sha(sha):
    return f"`{sha[:7]}`" if sha else "unknown commit"


def describe_baseline(info):
    """Name the baseline run so a stale artifact is visible in the table."""
    text = "*Baseline: `Benchmarks` run"
    if info.get("run"):
        text += f" {info['run']}"
    text += f" on `master` at {short_sha(info.get('sha'))}"
    if info.get("created"):
        text += f", created {info['created']}"
    return text + ".*"


def render_markdown(results, missing, baseline, title, baseline_info=None,
                    commit=None, skipped=()):
    baseline_info = baseline_info or {}
    lines = [MARKER, f"## {title}", ""]
    if not results:
        lines.append("No benchmark results were produced; see the job log.")
    else:
        if baseline:
            lines += [
                "| Benchmark | Base avg (ns) | This run avg (ns) | Change |",
                "|-----------|---------------|-------------------|--------|",
            ]
        else:
            lines += [
                "| Benchmark | Avg (ns) |",
                "|-----------|----------|",
            ]
        for key in sorted(results):
            current = results[key]
            if baseline:
                base = baseline.get(key)
                base_str = f"{base:.2f}" if isinstance(base, (int, float)) else "n/a"
                base_val = base if isinstance(base, (int, float)) else None
                lines.append(
                    f"| `{key}` | {base_str} | {current:.2f} | "
                    f"{format_delta(current, base_val)} |"
                )
            else:
                lines.append(f"| `{key}` | {current:.2f} |")
        lines.append("")
        lines.append("*Times in nanoseconds, lower is better. "
                     "Shared CI runners are noisy; treat small changes with care.*")
        if commit:
            lines.append(f"*This run: {short_sha(commit)}.*")
        if baseline:
            lines.append(describe_baseline(baseline_info))
        else:
            lines.append("*No baseline from `master` was available for comparison.*")
    if skipped:
        lines.append("")
        lines.append("Skipped: " + ", ".join(
            f"`{name}` ({reason})" for name, reason in skipped))
    if missing:
        lines.append("")
        lines.append("Missing results: " + ", ".join(f"`{m}`" for m in missing))
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="benchmark directory containing the result files",
    )
    parser.add_argument("--baseline", help="flat JSON summary to compare against")
    parser.add_argument("--output-json", help="write the flat JSON summary here")
    parser.add_argument("--output-md", help="write the Markdown summary here")
    parser.add_argument("--baseline-run", help="id of the run the baseline came from")
    parser.add_argument("--baseline-sha", help="commit the baseline run tested")
    parser.add_argument("--baseline-created", help="creation time of the baseline run")
    parser.add_argument("--commit", help="commit this run tested")
    parser.add_argument("--title", default="bpftime micro-benchmark results")
    args = parser.parse_args()

    results, missing, skipped = collect_results(args.root)

    baseline = None
    if args.baseline and os.path.exists(args.baseline):
        loaded = load_json(args.baseline)
        if isinstance(loaded, dict):
            baseline = loaded

    baseline_info = {
        "run": args.baseline_run,
        "sha": args.baseline_sha,
        "created": args.baseline_created,
    }
    markdown = render_markdown(results, missing, baseline, args.title,
                               baseline_info, args.commit, skipped)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2, sort_keys=True)
    if args.output_md:
        with open(args.output_md, "w") as f:
            f.write(markdown)
    else:
        print(markdown)


if __name__ == "__main__":
    main()
