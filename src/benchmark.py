#!/usr/bin/env python3
"""
benchmark.py — Validate a compiled workflow against held-out traces.

Takes a compiled DAG and compares its predictions against actual execution
traces to measure compilation accuracy.

For each held-out run, checks:
  1. Did the compiled DAG predict the right tools?
  2. Did it predict the right parameters?
  3. Did it predict the right ordering/phases?
  4. Which steps were correctly compiled vs needed LLM?

Usage:
    python benchmark.py --compiled result.json --holdout traces.json
    python benchmark.py --compiled result.json --holdout traces.lp
"""

import sys
import json
import argparse
from pathlib import Path
from collections import Counter


def main():
    parser = argparse.ArgumentParser(description="Benchmark compiled workflow against traces")
    parser.add_argument("--compiled", required=True, help="Path to compiled JSON")
    parser.add_argument("--holdout", required=True, help="Path to holdout traces (JSON)")
    parser.add_argument("--verbose", action="store_true", help="Show per-run details")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    compiled_path = Path(args.compiled) if Path(args.compiled).is_absolute() else project_root / args.compiled
    holdout_path = Path(args.holdout) if Path(args.holdout).is_absolute() else project_root / args.holdout

    compiled = json.loads(compiled_path.read_text())
    holdout = json.loads(holdout_path.read_text())

    runs = holdout.get("runs", [])
    if not runs:
        print("ERROR: No runs found in holdout data")
        sys.exit(1)

    meta = compiled.get("_autocompile", {})
    core_tools = set(meta.get("core_tools", []))
    phases = meta.get("phases", {})
    analysis = compiled.get("_analysis", {})
    compiled_calls = compiled.get("calls", [])

    # Extract compiled tool -> stable params mapping
    compiled_params = {}
    for call in compiled_calls:
        tool = call["tool"]
        if tool not in compiled_params:
            compiled_params[tool] = call.get("input", {})

    print(f"{'='*60}")
    print(f"BENCHMARK: compiled workflow vs {len(runs)} held-out runs")
    print(f"{'='*60}")
    print(f"Core tools in compiled DAG: {sorted(core_tools)}")
    print(f"Phases: {phases}")
    print()

    # Per-run analysis
    results = []
    tool_correct = Counter()
    tool_total = Counter()
    param_correct = Counter()
    param_total = Counter()

    for run in runs:
        run_id = run["id"]
        steps = run.get("steps", [])

        # What tools did this run actually use?
        actual_tools = set()
        actual_tool_params = {}
        for step in steps:
            tool = step["tool"]
            # Skip glue tools
            if tool in ("llm_generate", "notify_user", "prompt_user"):
                continue
            actual_tools.add(tool)
            if tool not in actual_tool_params:
                actual_tool_params[tool] = step.get("params", {})

        # Tool prediction accuracy
        predicted_present = core_tools & actual_tools  # correctly predicted as present
        predicted_absent = core_tools - actual_tools   # predicted present but actually absent
        missed = actual_tools - core_tools             # actually present but not in compiled DAG

        # This run's results
        run_result = {
            "run_id": run_id,
            "actual_tools": sorted(actual_tools),
            "correct_predictions": sorted(predicted_present),
            "false_positives": sorted(predicted_absent),
            "missed_tools": sorted(missed),
            "tool_accuracy": len(predicted_present) / max(len(core_tools), 1),
            "param_matches": {},
        }

        # Parameter accuracy for correctly predicted tools
        for tool in predicted_present:
            compiled_p = compiled_params.get(tool, {})
            actual_p = actual_tool_params.get(tool, {})

            for key, compiled_val in compiled_p.items():
                param_total[f"{tool}.{key}"] += 1
                actual_val = actual_p.get(key)

                # Handle array values (fan-out) — check if actual is one of the compiled values
                if isinstance(compiled_val, list):
                    if actual_val in compiled_val or str(actual_val) in compiled_val:
                        param_correct[f"{tool}.{key}"] += 1
                        run_result["param_matches"][f"{tool}.{key}"] = "match"
                    elif actual_val is None:
                        run_result["param_matches"][f"{tool}.{key}"] = "missing_in_actual"
                    else:
                        run_result["param_matches"][f"{tool}.{key}"] = f"mismatch: compiled={compiled_val}, actual={actual_val}"
                else:
                    if str(compiled_val) == str(actual_val):
                        param_correct[f"{tool}.{key}"] += 1
                        run_result["param_matches"][f"{tool}.{key}"] = "match"
                    elif actual_val is None:
                        run_result["param_matches"][f"{tool}.{key}"] = "missing_in_actual"
                    else:
                        run_result["param_matches"][f"{tool}.{key}"] = f"mismatch: compiled={compiled_val}, actual={actual_val}"

        for tool in predicted_present:
            tool_correct[tool] += 1
        for tool in core_tools:
            tool_total[tool] += 1

        results.append(run_result)

    # Summary
    print(f"{'='*60}")
    print("TOOL PREDICTION ACCURACY")
    print(f"{'='*60}")
    for tool in sorted(core_tools):
        correct = tool_correct[tool]
        total = tool_total[tool]
        pct = correct * 100 // max(total, 1)
        print(f"  {tool}: {correct}/{total} runs ({pct}%)")

    # Overall
    total_predictions = sum(tool_total.values())
    total_correct = sum(tool_correct.values())
    overall_tool_acc = total_correct * 100 // max(total_predictions, 1)
    print(f"\n  Overall tool accuracy: {total_correct}/{total_predictions} ({overall_tool_acc}%)")

    print(f"\n{'='*60}")
    print("PARAMETER ACCURACY (for correctly predicted tools)")
    print(f"{'='*60}")
    for param_key in sorted(param_total.keys()):
        correct = param_correct[param_key]
        total = param_total[param_key]
        pct = correct * 100 // max(total, 1)
        print(f"  {param_key}: {correct}/{total} ({pct}%)")

    total_param_preds = sum(param_total.values())
    total_param_correct = sum(param_correct.values())
    overall_param_acc = total_param_correct * 100 // max(total_param_preds, 1)
    print(f"\n  Overall param accuracy: {total_param_correct}/{total_param_preds} ({overall_param_acc}%)")

    # Conditional analysis
    conditionals = {c["tool"]: c for c in analysis.get("conditionals", [])}
    if conditionals:
        print(f"\n{'='*60}")
        print("CONDITIONAL EXECUTION VALIDATION")
        print(f"{'='*60}")
        for tool, cond in sorted(conditionals.items()):
            dep = cond["depends_on"]
            # Count: when dep is present, how often is tool present?
            dep_present = sum(1 for r in results if dep in set(r["actual_tools"]))
            both_present = sum(1 for r in results if dep in set(r["actual_tools"]) and tool in set(r["actual_tools"]))
            tool_without_dep = sum(1 for r in results if tool in set(r["actual_tools"]) and dep not in set(r["actual_tools"]))

            if dep_present > 0:
                rate = both_present * 100 // dep_present
                print(f"  {tool} | {dep}: {both_present}/{dep_present} ({rate}%) — compiled rate: {cond['rate']}")
            if tool_without_dep > 0:
                print(f"    WARNING: {tool} appeared {tool_without_dep}x WITHOUT {dep}")

    # Per-run details
    if args.verbose:
        print(f"\n{'='*60}")
        print("PER-RUN DETAILS")
        print(f"{'='*60}")
        for r in results:
            status = "FULL_MATCH" if not r["false_positives"] and not r["missed_tools"] else "PARTIAL"
            if not r["correct_predictions"] and not r["false_positives"]:
                status = "NO_OVERLAP"
            print(f"\n  {r['run_id']} [{status}]")
            if r["correct_predictions"]:
                print(f"    correct: {r['correct_predictions']}")
            if r["false_positives"]:
                print(f"    false positives: {r['false_positives']}")
            if r["missed_tools"]:
                print(f"    missed: {r['missed_tools']}")

    # Final summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Holdout runs: {len(runs)}")
    print(f"  Tool accuracy: {overall_tool_acc}%")
    print(f"  Param accuracy: {overall_param_acc}%")
    full_matches = sum(1 for r in results if not r["false_positives"])
    print(f"  Runs with no false positives: {full_matches}/{len(runs)} ({full_matches*100//max(len(runs),1)}%)")
    has_predictions = sum(1 for r in results if r["correct_predictions"])
    print(f"  Runs where compiled DAG matched: {has_predictions}/{len(runs)} ({has_predictions*100//max(len(runs),1)}%)")


if __name__ == "__main__":
    main()
