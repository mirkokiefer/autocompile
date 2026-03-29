#!/usr/bin/env python3
"""
compile.py — Observe repeated workflow traces, compile them into
the most efficient equivalent program.

Usage:
    python compile.py --traces examples/travel-updates/traces.lp --strategy asp
    python compile.py --traces traces.lp --rules rules/mine_patterns_relaxed.lp
"""

import sys
import json
import argparse
from pathlib import Path

try:
    import clingo
except ImportError:
    print("ERROR: clingo not installed. Run: pip install clingo")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Compile observed workflows into deterministic programs")
    parser.add_argument("--traces", default="examples/travel-updates/traces.lp", help="Path to trace facts (.lp)")
    parser.add_argument("--rules", default="rules/mine_patterns.lp", help="Path to ASP rules file")
    parser.add_argument("--strategy", default="asp", choices=["asp"], help="Compilation strategy")
    parser.add_argument("--verbose", action="store_true", help="Show all Clingo output")
    parser.add_argument("--output", default="result.json", help="Output compiled workflow")
    args = parser.parse_args()

    # Resolve paths relative to project root (parent of src/)
    project_root = Path(__file__).parent.parent
    traces_path = Path(args.traces) if Path(args.traces).is_absolute() else project_root / args.traces
    rules_path = Path(args.rules) if Path(args.rules).is_absolute() else project_root / args.rules
    output_path = Path(args.output) if Path(args.output).is_absolute() else project_root / args.output

    if not traces_path.exists():
        print(f"ERROR: Traces file not found: {traces_path}")
        sys.exit(1)

    if not rules_path.exists():
        print(f"ERROR: Rules file not found: {rules_path}")
        sys.exit(1)

    if args.strategy == "asp":
        results = run_asp_strategy(traces_path, rules_path, args.verbose)
    else:
        print(f"ERROR: Unknown strategy: {args.strategy}")
        sys.exit(1)

    # Display results
    print(f"{'='*60}")
    print(f"PATTERN MINING RESULTS ({results['job_count']} runs analyzed)")
    print(f"{'='*60}")
    print()

    print(f"Core tools (appear in >=50% of runs):")
    for tool in sorted(results["core_tools"]):
        count = results["tool_counts"].get(tool, "?")
        print(f"  {tool} — in {count}/{results['job_count']} runs")
    print()

    if results["parallel_pairs"]:
        print("Parallel pairs (same orchestration round):")
        for t1, t2 in sorted(results["parallel_pairs"]):
            print(f"  {t1} || {t2}")
        print()

    if results["consistent_order"]:
        print("Consistent ordering:")
        for t1, t2 in sorted(results["consistent_order"]):
            print(f"  {t1} -> {t2}")
        print()

    if results["conflicting_order"]:
        print("Conflicting orderings (resolved by optimization):")
        for t1, t2 in sorted(results["conflicting_order"]):
            print(f"  {t1} <-> {t2}")
        for t1, t2 in sorted(results["chosen_order"]):
            print(f"  resolved: {t1} -> {t2}")
        print()

    if results["conditionals"]:
        print("Conditional execution (branching):")
        for b, a in sorted(results["conditionals"]):
            rate = results["conditional_rates"].get((b, a))
            if rate:
                print(f"  {b} runs conditionally on {a} ({rate[0]}/{rate[1]} runs = {rate[0]*100//rate[1]}%)")
            else:
                print(f"  {b} -> conditional on {a}")
        print()

    if results["fusion_candidates"]:
        print("Fusion candidates (can be merged into single operation):")
        for a, b in sorted(results["fusion_candidates"]):
            print(f"  {a} + {b}")
        print()

    if results["mutually_exclusive"]:
        print("Mutually exclusive (branch alternatives):")
        for a, b in sorted(results["mutually_exclusive"]):
            print(f"  {a} XOR {b}")
        print()

    if results["stable_params"]:
        print("Stable parameters:")
        for tool in sorted(results["stable_params"]):
            print(f"  {tool}:")
            for key, vals in sorted(results["stable_params"][tool].items()):
                for val in vals:
                    display_val = val if len(val) <= 50 else val[:47] + "..."
                    print(f"    {key}: {display_val}")
        print()

    if results["variable_params"]:
        print("Variable parameters (runtime-dependent):")
        for tool in sorted(results["variable_params"]):
            print(f"  {tool}: {', '.join(sorted(results['variable_params'][tool]))}")
        print()

    # Synthesize compiled workflow
    compiled = synthesize(results)

    print(f"{'='*60}")
    print("COMPILED WORKFLOW")
    print(f"{'='*60}")
    print(json.dumps(compiled, indent=2))
    print()

    # Write output
    with open(output_path, "w") as f:
        json.dump(compiled, f, indent=2)
    print(f"Written to {output_path}")

    # Workflow summary
    print()
    print(f"{'='*60}")
    print("COMPILATION SUMMARY")
    print(f"{'='*60}")
    describe_workflow(compiled, results)


def run_asp_strategy(traces_path, rules_path, verbose=False):
    """Run the ASP/Clingo pattern mining strategy."""
    print(f"Loading traces from {traces_path}")
    print(f"Loading rules from {rules_path}")
    print()

    # Use optimization mode to find optimal answer set
    ctl = clingo.Control(["--opt-mode=optN", "1"])
    ctl.load(str(traces_path))
    ctl.load(str(rules_path))
    ctl.ground([("base", [])])

    results = {
        "core_tools": [],
        "tool_counts": {},
        "consistent_order": [],
        "conflicting_order": [],
        "chosen_order": [],
        "parallel_pairs": [],
        "stable_params": {},
        "variable_params": {},
        "phase_zero": [],
        "job_count": 0,
        "conditionals": [],
        "conditional_rates": {},
        "co_occurs": [],
        "fusion_candidates": [],
        "mutually_exclusive": [],
    }

    def on_model(model):
        # Clear lists on each model (we want the optimal one)
        for key in results:
            if isinstance(results[key], list):
                results[key].clear()
            elif isinstance(results[key], dict) and key != "tool_counts":
                results[key].clear()

        for atom in model.symbols(shown=True):
            name = atom.name
            a = atom.arguments

            if name == "core_tool":
                results["core_tools"].append(a[0].string)
            elif name == "tool_job_count":
                results["tool_counts"][a[0].string] = a[1].number
            elif name == "consistent_order":
                results["consistent_order"].append((a[0].string, a[1].string))
            elif name == "conflicting_order":
                results["conflicting_order"].append((a[0].string, a[1].string))
            elif name == "chosen_order":
                results["chosen_order"].append((a[0].string, a[1].string))
            elif name == "parallel_pair":
                results["parallel_pairs"].append((a[0].string, a[1].string))
            elif name == "stable_param":
                results["stable_params"].setdefault(a[0].string, {}).setdefault(a[1].string, []).append(a[2].string)
            elif name == "variable_param":
                results["variable_params"].setdefault(a[0].string, []).append(a[1].string)
            elif name == "phase_zero":
                results["phase_zero"].append(a[0].string)
            elif name == "job_count":
                results["job_count"] = a[0].number
            elif name == "conditional_tool":
                results["conditionals"].append((a[0].string, a[1].string))
            elif name == "conditional_rate":
                results["conditional_rates"][(a[0].string, a[1].string)] = (a[2].number, a[3].number)
            elif name == "co_occurs":
                results["co_occurs"].append((a[0].string, a[1].string))
            elif name == "fusion_candidate":
                results["fusion_candidates"].append((a[0].string, a[1].string))
            elif name == "mutually_exclusive":
                results["mutually_exclusive"].append((a[0].string, a[1].string))

    handle = ctl.solve(on_model=on_model)

    if not handle.satisfiable:
        print("UNSATISFIABLE — no patterns found. Check your traces.")
        sys.exit(1)

    return results


def synthesize(results):
    """Turn mined patterns into a compiled workflow DAG."""
    core_tools = sorted(results["core_tools"])
    orderings = results["consistent_order"] + results.get("chosen_order", [])
    stable_params = results["stable_params"]

    # Build phases from ordering
    predecessors = {}
    for t1, t2 in orderings:
        if t1 in core_tools and t2 in core_tools:
            predecessors.setdefault(t2, set()).add(t1)

    phase_assignment = {}
    remaining = set(core_tools)
    phase = 0
    while remaining:
        ready = [t for t in remaining
                 if all(p in phase_assignment for p in predecessors.get(t, set()))]
        if not ready:
            ready = list(remaining)
        for t in ready:
            phase_assignment[t] = phase
            remaining.discard(t)
        phase += 1

    phases = {}
    for tool, ph in sorted(phase_assignment.items(), key=lambda x: (x[1], x[0])):
        phases.setdefault(ph, []).append(tool)

    # Generate calls
    calls = []
    step_ids = {}
    step_counter = 0

    for phase_num in sorted(phases.keys()):
        for tool in phases[phase_num]:
            step_counter += 1
            step_id = f"step_{step_counter}_{tool}"
            step_ids[tool] = step_id

            tool_input = {}
            if tool in stable_params:
                for key, vals in stable_params[tool].items():
                    tool_input[key] = vals[0] if len(vals) == 1 else vals

            waits_for = []
            for pred in predecessors.get(tool, set()):
                if pred in step_ids:
                    ref = step_ids[pred]
                    if isinstance(ref, list):
                        waits_for.extend(ref)
                    else:
                        waits_for.append(ref)

            call_spec = {
                "id": step_id,
                "tool": tool,
                "input": tool_input,
                "compilation": "compiled",
            }
            if waits_for:
                call_spec["waits_for"] = sorted(waits_for)

            # Fan-out: if account has multiple stable values, expand into parallel calls
            if "account" in tool_input and isinstance(tool_input["account"], list):
                accounts = tool_input.pop("account")
                expanded_ids = []
                for i, account in enumerate(accounts):
                    expanded_id = f"{step_id}_{i}"
                    expanded_call = {
                        "id": expanded_id,
                        "tool": tool,
                        "input": {**tool_input, "account": account},
                        "compilation": "compiled",
                    }
                    if waits_for:
                        expanded_call["waits_for"] = sorted(waits_for)
                    calls.append(expanded_call)
                    expanded_ids.append(expanded_id)
                step_ids[tool] = expanded_ids
                continue

            calls.append(call_spec)

    # Resolve fan-out references in waits_for
    for call_spec in calls:
        if "waits_for" in call_spec:
            expanded = []
            for dep in call_spec["waits_for"]:
                for tool_name, ref in step_ids.items():
                    if isinstance(ref, list) and dep == f"step_{list(step_ids.keys()).index(tool_name)+1}_{tool_name}":
                        expanded.extend(ref)
                        break
                else:
                    expanded.append(dep)
            call_spec["waits_for"] = sorted(set(expanded))

    # Count total observed steps across all runs (approximate from tool counts)
    total_observed = sum(results["tool_counts"].get(t, 0) for t in results["tool_counts"])
    compiled_steps = len(calls)

    return {
        "_autocompile": {
            "strategy": "asp",
            "source_runs": results["job_count"],
            "core_tools": core_tools,
            "phases": {str(k): v for k, v in phases.items()},
        },
        "_boundary": {
            "compiled_steps": compiled_steps,
            "total_observed_tool_types": len(results["tool_counts"]),
            "core_tool_types": len(core_tools),
            "compilation_ratio": round(len(core_tools) / max(len(results["tool_counts"]), 1), 2),
        },
        "_analysis": {
            "conditionals": [{"tool": b, "depends_on": a,
                              "rate": f"{results['conditional_rates'].get((b,a),(0,0))[0]}/{results['conditional_rates'].get((b,a),(0,0))[1]}"}
                             for b, a in results.get("conditionals", [])],
            "fusion_candidates": [list(pair) for pair in results.get("fusion_candidates", [])],
            "mutually_exclusive": [list(pair) for pair in results.get("mutually_exclusive", [])],
            "conflicting_orders_resolved": [{"chose": f"{t1} -> {t2}"} for t1, t2 in results.get("chosen_order", [])],
            "variable_params": results.get("variable_params", {}),
        },
        "calls": calls,
    }


def describe_workflow(compiled, results):
    """Human-readable workflow description."""
    phases = compiled.get("_autocompile", {}).get("phases", {})
    boundary = compiled.get("_boundary", {})

    for phase_num in sorted(phases.keys()):
        tools = phases[phase_num]
        print(f"\nPhase {phase_num}:")
        for tool in tools:
            params = results["stable_params"].get(tool, {})
            param_str = ", ".join(
                f"{k}={v[0] if len(v)==1 else v}"
                for k, v in sorted(params.items())
                if k != "account"
            )
            accounts = params.get("account", [])
            if len(accounts) > 1:
                print(f"  {tool} x {len(accounts)} accounts (parallel fan-out)")
            else:
                print(f"  {tool}")
            if param_str:
                print(f"    params: {param_str}")

    print(f"\nBoundary:")
    print(f"  Compiled step types: {boundary.get('core_tool_types', '?')}/{boundary.get('total_observed_tool_types', '?')}")
    print(f"  Compilation ratio: {boundary.get('compilation_ratio', '?')}")
    print(f"\nSteps that remain as llm_invoke:")
    print(f"  (Tools not in core set need LLM orchestration or appear inconsistently)")
    non_core = set(results["tool_counts"].keys()) - set(results["core_tools"])
    for tool in sorted(non_core):
        count = results["tool_counts"].get(tool, 0)
        print(f"  {tool} — in {count}/{results['job_count']} runs")


if __name__ == "__main__":
    main()
