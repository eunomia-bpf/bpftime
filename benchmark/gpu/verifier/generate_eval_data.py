#!/usr/bin/env python3
"""Generate paper-ready GPU verifier evaluation tables."""

from __future__ import annotations

import csv
import platform
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
BUILD_DIR = REPO_ROOT / "build"
EVAL_BINARY = BUILD_DIR / "bpftime-verifier" / "bpftime_gpu_verifier_eval"
OBJECT_ROOT = SCRIPT_DIR / "objects"

UNSAFE_DIR = REPO_ROOT / "bpftime-verifier" / "test" / "gpu_unsafe_programs"
EXAMPLE_DIR = REPO_ROOT / "example" / "gpu"

CORRECTNESS_MD = SCRIPT_DIR / "correctness_results.md"
CORRECTNESS_CSV = SCRIPT_DIR / "correctness_results.csv"
FALSE_POSITIVE_MD = SCRIPT_DIR / "false_positive_results.md"
FALSE_POSITIVE_CSV = SCRIPT_DIR / "false_positive_results.csv"
COMPARISON_MD = SCRIPT_DIR / "comparison_table.md"

REAL_CORRECTNESS_EXAMPLES = [
    {
        "label": "cuda-counter",
        "source": EXAMPLE_DIR / "cuda-counter" / "cuda_probe.bpf.c",
    },
    {
        "label": "directly_run_on_gpu",
        "source": EXAMPLE_DIR / "directly_run_on_gpu" / "directly_run.bpf.c",
    },
]

CORRECTNESS_ORDER = [
    "varying_branch",
    "prohibited_helper",
    "varying_atomic",
    "varying_map_key",
    "resource_exceeded",
    "safe_counter",
    "safe_block_idx_branch",
]

CORRECTNESS_PROPERTY_OVERRIDES = {
    "varying_branch": "Varying branch condition",
    "prohibited_helper": "Prohibited helper (membar)",
    "varying_atomic": "Varying atomic address",
    "varying_map_key": "Varying map key",
    "resource_exceeded": "Helper/resource budget exceeded",
    "safe_counter": "None (safe)",
    "safe_block_idx_branch": "None (safe block-uniform branch)",
}

TRUE_POSITIVE_EXAMPLES = {
    "cudagraph/cuda_probe.bpf.c",
    "gpu_shared_map/gpu_shared_map.bpf.c",
    "host_map_test/host_map_test.bpf.c",
    "kernel_trace/kernel_trace.bpf.c",
    "threadscheduling/threadscheduling.bpf.c",
}

COMPARISON_PATTERNS = [
    {
        "label": "Varying branch condition",
        "kind": "object",
        "source": UNSAFE_DIR / "varying_branch.bpf.c",
        "representative": "gpu_unsafe_programs/varying_branch.bpf.c",
        "prevail_design": "MISS",
    },
    {
        "label": "Prohibited helper (membar)",
        "kind": "object",
        "source": UNSAFE_DIR / "prohibited_helper.bpf.c",
        "representative": "gpu_unsafe_programs/prohibited_helper.bpf.c",
        "prevail_design": "MISS",
    },
    {
        "label": "Varying atomic address",
        "kind": "object",
        "source": UNSAFE_DIR / "varying_atomic.bpf.c",
        "representative": "gpu_unsafe_programs/varying_atomic.bpf.c",
        "prevail_design": "MISS",
    },
    {
        "label": "Varying map key",
        "kind": "object",
        "source": UNSAFE_DIR / "varying_map_key.bpf.c",
        "representative": "gpu_unsafe_programs/varying_map_key.bpf.c",
        "prevail_design": "MISS",
    },
    {
        "label": "Helper-call budget exceeded",
        "kind": "object",
        "source": UNSAFE_DIR / "resource_exceeded.bpf.c",
        "representative": "gpu_unsafe_programs/resource_exceeded.bpf.c",
        "prevail_design": "MISS",
    },
    {
        "label": "Memory safety (null deref)",
        "kind": "builtin",
        "name": "null_deref",
        "representative": "builtin:null_deref",
        "prevail_design": "CATCH",
    },
    {
        "label": "Division by zero",
        "kind": "builtin",
        "name": "division_by_zero",
        "representative": "builtin:division_by_zero",
        "prevail_design": "CATCH",
    },
    {
        "label": "Unbounded loop (self-loop)",
        "kind": "builtin",
        "name": "resource_exceeded",
        "representative": "builtin:resource_exceeded",
        "prevail_design": "CATCH",
    },
]


@dataclass
class CompileResult:
    source: Path
    object_path: Path
    success: bool
    stderr: str


def ensure_eval_binary() -> None:
    if EVAL_BINARY.exists():
        return

    if not (BUILD_DIR / "CMakeCache.txt").exists():
        raise FileNotFoundError(
            f"Build directory is not configured: {BUILD_DIR}. Run CMake first."
        )

    subprocess.run(
        [
            "cmake",
            "--build",
            str(BUILD_DIR),
            "--target",
            "bpftime_gpu_verifier_eval",
            "-j",
            str(max(1, os_cpu_count())),
        ],
        cwd=REPO_ROOT,
        check=True,
    )


def os_cpu_count() -> int:
    try:
        import os

        return os.cpu_count() or 1
    except Exception:
        return 1


def detect_bpf_arch() -> str:
    machine = platform.machine().lower()
    if machine in {"x86_64", "amd64"}:
        return "x86"
    if machine in {"aarch64", "arm64"}:
        return "arm64"
    if machine.startswith("arm"):
        return "arm"
    if machine.startswith("riscv64"):
        return "riscv"
    if machine.startswith("ppc64"):
        return "powerpc"
    if machine in {"loongarch64", "loong64"}:
        return "loongarch"
    raise RuntimeError(f"Unsupported BPF target architecture mapping for {machine}")


def bpf_compile_command(source: Path, object_path: Path) -> list[str]:
    arch = detect_bpf_arch()
    return [
        "clang",
        "-target",
        "bpf",
        "-O2",
        "-g",
        f"-D__TARGET_ARCH_{arch}",
        f"-I{REPO_ROOT / 'third_party' / 'vmlinux'}",
        f"-I{REPO_ROOT / 'third_party' / 'bpftool' / 'libbpf' / 'src'}",
        "-c",
        str(source),
        "-o",
        str(object_path),
    ]


def object_output_path(source: Path, category: str) -> Path:
    rel = source.relative_to(REPO_ROOT)
    name = "__".join(rel.parts[-2:]).replace(".c", ".o")
    return OBJECT_ROOT / category / name


def compile_source(source: Path, category: str) -> CompileResult:
    object_path = object_output_path(source, category)
    object_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        bpf_compile_command(source, object_path),
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return CompileResult(
        source=source,
        object_path=object_path,
        success=completed.returncode == 0,
        stderr=(completed.stdout + completed.stderr).strip(),
    )


def run_eval_object(object_path: Path, mode: str) -> dict:
    completed = subprocess.run(
        [str(EVAL_BINARY), "verify-object", str(object_path), "--mode", mode],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    import json

    return json.loads(completed.stdout)


def run_eval_builtin(name: str, mode: str) -> dict:
    completed = subprocess.run(
        [str(EVAL_BINARY), "verify-builtin", name, "--mode", mode],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    import json

    return json.loads(completed.stdout)


def section_suffix(section_name: str) -> str:
    if section_name.startswith("kprobe/"):
        return "entry"
    if section_name.startswith("kretprobe/"):
        return "return"
    if section_name.startswith("uprobe"):
        return "uprobe"
    if section_name.startswith("uretprobe/"):
        return "uretprobe"
    return section_name


def collapse_ws(value: str) -> str:
    return " ".join(value.split())


def summarize_error(error: str) -> str:
    lines = [line.strip() for line in error.splitlines() if line.strip()]
    if not lines:
        return ""

    diagnostics = [line for line in lines if re.match(r"^\d+:", line)]
    diagnostics = [
        line
        for line in diagnostics
        if re.search(r"[A-Za-z]", line.split(":", 1)[1])
    ]
    if diagnostics:
        return "; ".join(diagnostics[-3:])

    for line in reversed(lines):
        if " at instruction " in line or "resource budget exceeded" in line:
            return collapse_ws(line)

    return collapse_ws(lines[-1])


def count_loc(source: Path) -> int:
    text = source.read_text(encoding="utf-8")
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    lines = []
    for line in text.splitlines():
        line = re.sub(r"//.*", "", line).strip()
        if line:
            lines.append(line)
    return len(lines)


def parse_comment_metadata(source: Path) -> tuple[str, str]:
    text = source.read_text(encoding="utf-8")
    property_match = re.search(r"Safety property:\s*(.+)", text)
    expected_match = re.search(r"Expected verifier result:\s*(PASS|REJECT)", text)
    if not property_match or not expected_match:
        raise ValueError(f"Missing correctness metadata in {source}")
    return property_match.group(1).strip(), expected_match.group(1).strip()


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    def escape_cell(cell: str) -> str:
        return cell.replace("|", r"\|")

    header_line = "| " + " | ".join(headers) + " |"
    divider_line = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(escape_cell(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header_line, divider_line, *body]) + "\n"


def write_csv(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def verify_program_rows(source: Path, category: str, mode: str) -> dict:
    compile_result = compile_source(source, category)
    if not compile_result.success:
        raise RuntimeError(f"Failed to compile {source}: {compile_result.stderr}")
    return run_eval_object(compile_result.object_path, mode)


def make_correctness_rows() -> tuple[list[list[str]], list[list[str]]]:
    md_rows: list[list[str]] = []
    csv_rows: list[list[str]] = []

    for base_name in CORRECTNESS_ORDER:
        source = UNSAFE_DIR / f"{base_name}.bpf.c"
        property_text, expected = parse_comment_metadata(source)
        property_text = CORRECTNESS_PROPERTY_OVERRIDES.get(base_name, property_text)
        verify_payload = verify_program_rows(source, "correctness", "gpu-simt")
        for program in verify_payload["programs"]:
            row = [
                base_name,
                str(count_loc(source)),
                property_text,
                expected,
                "PASS" if program["passed"] else "REJECT",
                str(round(program["total_time_us"])),
            ]
            md_rows.append(row)
            csv_rows.append(row)

    for example in REAL_CORRECTNESS_EXAMPLES:
        source = example["source"]
        verify_payload = verify_program_rows(source, "correctness", "gpu-simt")
        for program in verify_payload["programs"]:
            row = [
                f"{example['label']} ({section_suffix(program['section_name'])})",
                str(count_loc(source)),
                "None (real example)",
                "PASS",
                "PASS" if program["passed"] else "REJECT",
                str(round(program["total_time_us"])),
            ]
            md_rows.append(row)
            csv_rows.append(row)

    return md_rows, csv_rows


def classify_example(example_name: str, failures: list[str]) -> str:
    if not failures:
        return "pass"
    if example_name in TRUE_POSITIVE_EXAMPLES:
        return "true positive"
    return "false positive"


def make_false_positive_rows() -> tuple[list[list[str]], list[list[str]]]:
    md_rows: list[list[str]] = []
    csv_rows: list[list[str]] = []

    for source in sorted(EXAMPLE_DIR.glob("*/*.bpf.c")):
        example_name = str(source.relative_to(EXAMPLE_DIR))
        verify_payload = verify_program_rows(source, "examples", "gpu-simt")
        sections = [program["section_name"] for program in verify_payload["programs"]]
        failures = [
            f"{program['section_name']}: {summarize_error(program['error_message'])}"
            for program in verify_payload["programs"]
            if not program["passed"]
        ]

        row = [
            example_name,
            ", ".join(sections),
            "PASS" if not failures else "REJECT",
            " // ".join(failures),
            classify_example(example_name, failures),
        ]
        md_rows.append(row)
        csv_rows.append(row)

    return md_rows, csv_rows


def evaluate_comparison_pattern(pattern: dict) -> str:
    if pattern["kind"] == "builtin":
        payload = run_eval_builtin(pattern["name"], "gpu-simt")
        return "MISS" if payload["programs"][0]["passed"] else "CATCH"

    payload = verify_program_rows(pattern["source"], "comparison", "gpu-simt")
    return "MISS" if all(program["passed"] for program in payload["programs"]) else "CATCH"


def make_comparison_rows() -> list[list[str]]:
    rows: list[list[str]] = []
    for pattern in COMPARISON_PATTERNS:
        rows.append(
            [
                pattern["label"],
                pattern["representative"],
                "MISS",
                pattern["prevail_design"],
                evaluate_comparison_pattern(pattern),
            ]
        )
    return rows


def write_correctness_outputs() -> None:
    headers = ["Program", "LOC", "Safety Property", "Expected", "Result", "Time (μs)"]
    md_rows, csv_rows = make_correctness_rows()
    CORRECTNESS_MD.write_text(markdown_table(headers, md_rows), encoding="utf-8")
    write_csv(CORRECTNESS_CSV, headers, csv_rows)


def write_false_positive_outputs() -> None:
    headers = ["Example", "Sections", "Result", "Errors", "Classification"]
    md_rows, csv_rows = make_false_positive_rows()
    FALSE_POSITIVE_MD.write_text(markdown_table(headers, md_rows), encoding="utf-8")
    write_csv(FALSE_POSITIVE_CSV, headers, csv_rows)


def write_comparison_output() -> None:
    headers = [
        "Pattern",
        "Representative Input",
        "No Verification",
        "Standard PREVAIL (design)",
        "SIMT-aware (measured)",
    ]
    rows = make_comparison_rows()
    note = (
        "\nNotes:\n"
        "- `No Verification` is the baseline with the verifier disabled, so unsafe programs are not intercepted.\n"
        "- `Standard PREVAIL (design)` is design-based coverage, not a measurement on GPU object files.\n"
        "- `Unbounded loop (self-loop)` uses builtin `resource_exceeded`; it is distinct from "
        "`gpu_unsafe_programs/resource_exceeded.bpf.c`, which models helper-call budget exhaustion.\n"
    )
    COMPARISON_MD.write_text(markdown_table(headers, rows) + note, encoding="utf-8")


def main() -> int:
    ensure_eval_binary()
    OBJECT_ROOT.mkdir(parents=True, exist_ok=True)

    write_correctness_outputs()
    write_false_positive_outputs()
    write_comparison_output()

    print(f"Wrote {CORRECTNESS_MD}")
    print(f"Wrote {CORRECTNESS_CSV}")
    print(f"Wrote {FALSE_POSITIVE_MD}")
    print(f"Wrote {FALSE_POSITIVE_CSV}")
    print(f"Wrote {COMPARISON_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
