"""Generate §9 Markdown and CSV summary tables from evaluation JSONL logs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics
from typing import Any


MAIN_METHOD_ORDER = (
    "rebranch",
    "uniform_dr",
    "single_twin",
    "agentic_genplan",
    "physcap",
    "twin_mpc",
    "twin_mpc_receding",
    "oneshot_calib",
    "oracle_twin",
)

ABLATION_ORDER = (
    "full",
    "random_ce",
    "worst_ce",
    "no_retune_or_branch",
    "no_lifting",
    "no_branch_verify",
    "physcap_primitives",
)

METRIC_COLS = (
    ("zero_shot_success_rate", "Zero-Shot SR", True),
    ("sim_false_positive_rate", "Sim FP", True),
    ("safety_violation_rate", "Safety Viol.", True),
    ("redundant_probe_rate", "Redund. Probe", True),
    ("missed_probe_rate", "Missed Probe", True),
    ("branch_discrimination_accuracy", "Branch Acc.", True),
    ("dev_llm_calls", "Dev LLM", False),
    ("dev_tokens", "Dev Tok", False),
    ("runtime_llm_calls", "RT LLM", False),
    ("runtime_tokens", "RT Tok", False),
    ("mean_exec_time_s", "Exec (s)", False),
)


def _load_records(input_dirs: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for d in input_dirs:
        jsonl_path = d / "runs.jsonl"
        if jsonl_path.exists():
            for line in jsonl_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def _fmt_mean_std(vals: list[float], is_rate: bool) -> str:
    if not vals:
        return "—"
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    if is_rate:
        return f"{100.0 * m:.1f}±{100.0 * s:.1f}%"
    if abs(m) >= 100.0:
        return f"{m:.0f}±{s:.0f}"
    return f"{m:.2f}±{s:.2f}"


def _aggregate_group(rows: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    agg: dict[str, tuple[float, float]] = {}
    for key, _, _ in METRIC_COLS:
        vals = [float(r["metrics"][key]) for r in rows if r.get("metrics", {}).get(key) is not None]
        if vals:
            m = statistics.mean(vals)
            s = statistics.stdev(vals) if len(vals) > 1 else 0.0
            agg[key] = (m, s)
    return agg


def _common_cases(rows: list[dict[str, Any]], group_key: str) -> set[tuple[str, int]]:
    """Use identical task/seed support for every aggregate comparator."""
    groups: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for row in rows:
        groups[str(row[group_key])].add((str(row["task"]), int(row["seed"])))
    return set.intersection(*groups.values())


def _coverage_note(cases: set[tuple[str, int]]) -> str:
    tasks = ", ".join(sorted({task for task, _ in cases}))
    return (f"Aggregate uses {len(cases)} common task/seed pairs: {tasks}. "
            "± is the SD across these task/seed results, not a confidence interval.\n")


def _write_csv(path: Path, table_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not table_rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(table_rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table_rows)


def generate_tables(input_dirs: list[Path], output_dir: Path) -> dict[str, Path]:
    """Read `runs.jsonl` files and emit `main_results.{md,csv}` and `ablation_results.{md,csv}`."""
    records = _load_records(input_dirs)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Separate main baselines (variant == "full") from ablations
    main_records = [r for r in records if r.get("variant", "full") == "full"]
    ablation_records = [
        r for r in records if r.get("method") == "rebranch"
    ]

    # 1. Main Results Table (overall per method & split, plus per-task hidden breakdown)
    main_csv_rows: list[dict[str, Any]] = []
    md_lines: list[str] = [
        "# Main Benchmark Results (§9)\n",
        "`physcap` is the legacy identifier of the local online-probe proxy; it is not a PhysCaP reproduction. "
        "The legacy `ood` split samples in-range boundary cases.\n",
    ]

    splits = sorted({str(r.get("split", "hidden")) for r in main_records}) or ["hidden"]
    for split in splits:
        split_rows = [r for r in main_records if r.get("split", "hidden") == split]
        if not split_rows:
            continue
        md_lines.append(f"## Aggregate ({split} split)\n")
        common = _common_cases(split_rows, "method")
        md_lines.append(_coverage_note(common))
        header = "| Method | " + " | ".join(lbl for _, lbl, _ in METRIC_COLS) + " |"
        sep = "|" + "---|" * (len(METRIC_COLS) + 1)
        md_lines.extend([header, sep])

        methods = sorted(
            {str(r["method"]) for r in split_rows},
            key=lambda m: (MAIN_METHOD_ORDER.index(m) if m in MAIN_METHOD_ORDER else 999, m),
        )
        for method in methods:
            m_rows = [r for r in split_rows if r["method"] == method and (str(r["task"]), int(r["seed"])) in common]
            agg = _aggregate_group(m_rows)
            cells = [method]
            csv_row: dict[str, Any] = {"split": split, "method": method, "n_runs": len(m_rows)}
            for key, _, is_rate in METRIC_COLS:
                vals = [float(r["metrics"][key]) for r in m_rows if r.get("metrics", {}).get(key) is not None]
                cells.append(_fmt_mean_std(vals, is_rate))
                if key in agg:
                    csv_row[f"{key}_mean"] = round(agg[key][0], 6)
                    csv_row[f"{key}_std"] = round(agg[key][1], 6)
                else:
                    csv_row[f"{key}_mean"] = ""
                    csv_row[f"{key}_std"] = ""
            md_lines.append("| " + " | ".join(cells) + " |")
            main_csv_rows.append(csv_row)
        md_lines.append("")

        # Per-task Zero-Shot SR breakdown
        tasks = sorted({str(r["task"]) for r in split_rows})
        if tasks:
            md_lines.append(f"### Per-Task Zero-Shot Success Rate ({split} split)\n")
            t_header = "| Task | " + " | ".join(methods) + " |"
            t_sep = "|" + "---|" * (len(methods) + 1)
            md_lines.extend([t_header, t_sep])
            for task in tasks:
                row_cells = [task]
                for method in methods:
                    tm_rows = [
                        r for r in split_rows if r["task"] == task and r["method"] == method
                    ]
                    vals = [float(r["metrics"]["zero_shot_success_rate"]) for r in tm_rows]
                    row_cells.append(_fmt_mean_std(vals, is_rate=True))
                md_lines.append("| " + " | ".join(row_cells) + " |")
            md_lines.append("")

    main_md_path = output_dir / "main_results.md"
    main_csv_path = output_dir / "main_results.csv"
    main_md_path.write_text("\n".join(md_lines), encoding="utf-8")
    _write_csv(main_csv_path, main_csv_rows)

    # 2. Ablation Results Table
    abl_csv_rows: list[dict[str, Any]] = []
    abl_md_lines: list[str] = ["# ReBranch Ablation Results (§7 & §9)\n"]
    abl_splits = sorted({str(r.get("split", "hidden")) for r in ablation_records}) or ["hidden"]
    for split in abl_splits:
        split_rows = [r for r in ablation_records if r.get("split", "hidden") == split]
        if not split_rows:
            continue
        abl_md_lines.append(f"## Ablations ({split} split)\n")
        common = _common_cases(split_rows, "variant")
        abl_md_lines.append(_coverage_note(common))
        header = "| Variant | " + " | ".join(lbl for _, lbl, _ in METRIC_COLS) + " |"
        sep = "|" + "---|" * (len(METRIC_COLS) + 1)
        abl_md_lines.extend([header, sep])

        variants = sorted(
            {str(r.get("variant", "full")) for r in split_rows},
            key=lambda v: (ABLATION_ORDER.index(v) if v in ABLATION_ORDER else 999, v),
        )
        for variant in variants:
            v_rows = [r for r in split_rows if r.get("variant", "full") == variant and (str(r["task"]), int(r["seed"])) in common]
            agg = _aggregate_group(v_rows)
            cells = [variant]
            csv_row = {"split": split, "variant": variant, "n_runs": len(v_rows)}
            for key, _, is_rate in METRIC_COLS:
                vals = [float(r["metrics"][key]) for r in v_rows if r.get("metrics", {}).get(key) is not None]
                cells.append(_fmt_mean_std(vals, is_rate))
                if key in agg:
                    csv_row[f"{key}_mean"] = round(agg[key][0], 6)
                    csv_row[f"{key}_std"] = round(agg[key][1], 6)
                else:
                    csv_row[f"{key}_mean"] = ""
                    csv_row[f"{key}_std"] = ""
            abl_md_lines.append("| " + " | ".join(cells) + " |")
            abl_csv_rows.append(csv_row)
        abl_md_lines.append("")

        tasks = sorted({str(r["task"]) for r in split_rows})
        if tasks:
            abl_md_lines.append(f"### Per-Task Ablation Zero-Shot SR ({split} split)\n")
            t_header = "| Task | " + " | ".join(variants) + " |"
            t_sep = "|" + "---|" * (len(variants) + 1)
            abl_md_lines.extend([t_header, t_sep])
            for task in tasks:
                row_cells = [task]
                for variant in variants:
                    tv_rows = [
                        r
                        for r in split_rows
                        if r["task"] == task and r.get("variant", "full") == variant
                    ]
                    vals = [float(r["metrics"]["zero_shot_success_rate"]) for r in tv_rows]
                    row_cells.append(_fmt_mean_std(vals, is_rate=True))
                abl_md_lines.append("| " + " | ".join(row_cells) + " |")
            abl_md_lines.append("")

    abl_md_path = output_dir / "ablation_results.md"
    abl_csv_path = output_dir / "ablation_results.csv"
    abl_md_path.write_text("\n".join(abl_md_lines), encoding="utf-8")
    _write_csv(abl_csv_path, abl_csv_rows)

    return {
        "main_md": main_md_path,
        "main_csv": main_csv_path,
        "ablation_md": abl_md_path,
        "ablation_csv": abl_csv_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ReBranch §9 result tables.")
    parser.add_argument(
        "--input-dir",
        nargs="+",
        default=["outputs/m4"],
        help="Directories containing runs.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/tables",
        help="Output directory for Markdown and CSV tables",
    )
    args = parser.parse_args()
    paths = generate_tables(
        [Path(p) for p in args.input_dir],
        Path(args.output_dir),
    )
    for k, p in paths.items():
        print(f"{k}: {p}")


if __name__ == "__main__":
    main()
