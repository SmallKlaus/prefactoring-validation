#!/usr/bin/env python3
"""
Prefactoring Results Analyzer
==============================
Reads the JSONL output from prefactoring_pipeline.py and produces:
  - Aggregate statistics per project
  - Breakdown by issue type, confidence level, refactoring type
  - A filtered list of high-confidence prefactoring cases for manual review
  - A CSV-friendly summary

Usage:
  python analyze_results.py --results flink_prefactoring_results.jsonl [--output-csv flink_summary.csv]
  python analyze_results.py --results results/*.jsonl --combine   # merge multiple projects
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  Warning: bad JSON on line {i} of {path}: {e}", file=sys.stderr)
    return records


def analyze(records: list[dict], label: str = ""):
    print(f"\n{'='*60}")
    print(f"  PREFACTORING RESULTS  {f'— {label}' if label else ''}")
    print(f"{'='*60}")

    total = len(records)
    if total == 0:
        print("  No records.")
        return {}

    true_cases   = [r for r in records if r.get("prefactoring_detected") is True]
    false_cases  = [r for r in records if r.get("prefactoring_detected") is False]
    error_cases  = [r for r in records if r.get("prefactoring_detected") is None]

    print(f"\n  Total records            : {total}")
    print(f"  prefactoring = True      : {len(true_cases)}  ({100*len(true_cases)/total:.1f}%)")
    print(f"  prefactoring = False     : {len(false_cases)}  ({100*len(false_cases)/total:.1f}%)")
    print(f"  Error / undetermined     : {len(error_cases)}")

    # ---- Confidence breakdown ----
    conf_counts = Counter(r.get("confidence", "?") for r in true_cases)
    print(f"\n  Confidence (true cases):")
    for level in ("high", "medium", "low", "error", "?"):
        n = conf_counts.get(level, 0)
        if n:
            print(f"    {level:<8} : {n}")

    # ---- By issue type ----
    by_type = defaultdict(lambda: {"true": 0, "false": 0})
    for r in records:
        itype = r.get("issue_type", "Unknown")
        key   = "true" if r.get("prefactoring_detected") else "false"
        by_type[itype][key] += 1

    print(f"\n  By JIRA issue type:")
    for itype, counts in sorted(by_type.items()):
        total_t = counts["true"] + counts["false"]
        pct = 100 * counts["true"] / total_t if total_t else 0
        print(f"    {itype:<20}: {counts['true']:>3} true / {total_t:>3} total  ({pct:.0f}%)")

    # ---- Structural signal breakdown ----
    struct_ref_true  = sum(1 for r in true_cases  if r.get("structural_refactoring"))
    struct_ref_false = sum(1 for r in false_cases if r.get("structural_refactoring"))
    struct_td_true   = sum(1 for r in true_cases  if r.get("structural_td_fixed"))
    struct_td_false  = sum(1 for r in false_cases if r.get("structural_td_fixed"))

    print(f"\n  Structural refactoring present:")
    print(f"    Among prefactoring=True  cases: {struct_ref_true}/{len(true_cases)}")
    print(f"    Among prefactoring=False cases: {struct_ref_false}/{len(false_cases)}")
    print(f"\n  Structural TD fixed:")
    print(f"    Among prefactoring=True  cases: {struct_td_true}/{len(true_cases)}")
    print(f"    Among prefactoring=False cases: {struct_td_false}/{len(false_cases)}")

    # ---- Top refactoring types in true cases ----
    ref_counter: Counter = Counter()
    for r in true_cases:
        for rtype, count in r.get("refactorings", {}).items():
            ref_counter[rtype] += count
    if ref_counter:
        print(f"\n  Top refactoring types in prefactoring=True cases:")
        for rtype, cnt in ref_counter.most_common(10):
            print(f"    {rtype:<35}: {cnt}")

    # ---- High-confidence true cases (manual review list) ----
    high_conf_true = [
        r for r in true_cases
        if r.get("confidence") in ("high", "medium")
    ]
    print(f"\n  High/medium-confidence prefactoring cases ({len(high_conf_true)}):")
    for r in sorted(high_conf_true, key=lambda x: x.get("confidence", "z")):
        print(f"    [{r.get('confidence','?').upper()}] {r['jira_id']:20s}  "
              f"type={r.get('issue_type','?'):<15}  "
              f"refactorings={r.get('refactoring_count',0)}")
        if r.get("reasoning"):
            # Print first sentence of reasoning
            sentence = r["reasoning"].split(".")[0] + "."
            print(f"           → {sentence}")

    return {
        "total": total,
        "prefactoring_true": len(true_cases),
        "prefactoring_false": len(false_cases),
        "errors": len(error_cases),
        "true_pct": round(100 * len(true_cases) / total, 1) if total else 0,
        "high_conf_true": len(high_conf_true),
    }


def write_csv(records: list[dict], path: Path):
    if not records:
        return
    fields = [
        "jira_id", "issue_type", "prefactoring_detected", "confidence",
        "structural_refactoring", "structural_td_fixed",
        "refactoring_count", "fixed_count", "new_count",
        "diff_available", "reasoning"
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k, "") for k in fields})
    print(f"\n  CSV written → {path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze prefactoring pipeline results.")
    parser.add_argument("--results",    required=True, nargs="+", type=Path,
                        help="One or more JSONL result files")
    parser.add_argument("--output-csv", type=Path, default=None,
                        help="Write summary CSV to this path")
    parser.add_argument("--combine",    action="store_true",
                        help="Merge all files and report combined stats")
    args = parser.parse_args()

    all_records = []
    for path in args.results:
        records = load_jsonl(path)
        print(f"Loaded {len(records)} records from {path}")
        all_records.extend(records)
        if not args.combine:
            analyze(records, label=path.stem)

    if args.combine and len(args.results) > 1:
        analyze(all_records, label="COMBINED")

    if args.output_csv:
        write_csv(all_records, args.output_csv)


if __name__ == "__main__":
    main()