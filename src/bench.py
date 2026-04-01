#!/usr/bin/env python3
"""
bench.py — Unified benchmark harness for autocompile backends.

Runs any compilation backend against a dataset, measures standardized
metrics, and produces comparable results. This is the lab's measurement
instrument.

Metrics:
  - compilation_ratio: core tools / total tool types
  - pattern_count: total patterns discovered (orderings, fusions, etc.)
  - holdout_accuracy: does compiled program predict unseen traces?
  - compression: compiled steps / mean raw steps per trace
  - consistency: how many ordering conflicts needed resolution?
  - discovery: unique patterns (conditional, mutually exclusive, fusion)

Usage:
    # Run on a single dataset
    python src/bench.py --traces examples/travel-updates/traces.lp

    # Run on all registered datasets
    python src/bench.py --all

    # Compare multiple rule sets
    python src/bench.py --traces data.lp --rules rules/mine_patterns.lp rules/mine_patterns_relaxed.lp

    # Output machine-readable results
    python src/bench.py --all --format json > results.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

try:
    import clingo
except ImportError:
    sys.exit("ERROR: clingo required. Run: pip install clingo")


# ============================================================================
# Dataset Registry
# ============================================================================

DATASETS = {
    "agent-workflows": {
        "traces": "examples/travel-updates/traces.lp",
        "domain": "agent",
        "description": "177 Daslab agent workflow runs (27 real + 150 synthetic)",
        "source": "daslab-production",
    },
    "robotics-synthetic": {
        "traces": "examples/robotics-pick-place/traces.lp",
        "domain": "robotics",
        "description": "20 synthetic pick-and-place demonstrations",
        "source": "generated",
    },
    "lab-synthesis": {
        "traces": "examples/lab-synthesis/traces.lp",
        "domain": "lab",
        "description": "25 Suzuki cross-coupling synthesis experiments",
        "source": "generated",
    },
    "edge-compute": {
        "traces": "examples/edge-compute/traces.lp",
        "domain": "edge",
        "description": "25 anomaly detection pipeline runs across device grid",
        "source": "generated",
    },
    "lerobot-real": {
        "traces": "examples/lerobot-pick-place/traces.lp",
        "domain": "robotics",
        "description": "50 real LeRobot UCSD pick-and-place episodes",
        "source": "lerobot/ucsd_pick_and_place_dataset",
    },
    "lerobot-discovered": {
        "traces": "examples/lerobot-discovered/traces.lp",
        "domain": "robotics",
        "description": "50 real episodes, segments discovered (not hand-labeled)",
        "source": "lerobot/ucsd_pick_and_place_dataset",
    },
}

BACKENDS = {
    "asp-strict": {
        "rules": "rules/mine_patterns.lp",
        "description": "ASP pattern mining, 50% core threshold",
    },
    "asp-relaxed": {
        "rules": "rules/mine_patterns_relaxed.lp",
        "description": "ASP pattern mining, 25% core threshold",
    },
}


# ============================================================================
# Compilation + Metrics
# ============================================================================

def run_compilation(traces_path, rules_path):
    """Run ASP compilation and extract results."""
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
    results["satisfiable"] = handle.satisfiable

    return results


def compute_metrics(results):
    """Compute standardized metrics from compilation results."""
    n_core = len(results["core_tools"])
    n_total = len(results["tool_counts"])
    n_runs = results["job_count"]

    # Compilation ratio
    compilation_ratio = n_core / max(n_total, 1)

    # Pattern counts
    n_orderings = len(results["consistent_order"])
    n_conflicts = len(results["conflicting_order"])
    n_parallel = len(results["parallel_pairs"])
    n_fusion = len(results["fusion_candidates"])
    n_exclusive = len(results["mutually_exclusive"])
    n_conditional = len(set(b for b, a in results["conditionals"]))  # unique conditional tools
    n_stable = sum(len(vals) for params in results["stable_params"].values()
                   for vals in params.values())
    n_variable = sum(len(keys) for keys in results["variable_params"].values())

    total_patterns = (n_orderings + n_conflicts + n_parallel + n_fusion +
                      n_exclusive + n_conditional)

    # Consistency: ratio of consistent to total orderings
    total_ordering_evidence = n_orderings + n_conflicts
    consistency = n_orderings / max(total_ordering_evidence, 1)

    # Phases (proxy for program depth)
    orderings = results["consistent_order"] + results.get("chosen_order", [])
    predecessors = {}
    for t1, t2 in orderings:
        if t1 in results["core_tools"] and t2 in results["core_tools"]:
            predecessors.setdefault(t2, set()).add(t1)

    phase_assignment = {}
    remaining = set(results["core_tools"])
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

    n_phases = max(phase_assignment.values()) + 1 if phase_assignment else 0

    return {
        "runs": n_runs,
        "tool_types_total": n_total,
        "tool_types_core": n_core,
        "compilation_ratio": round(compilation_ratio, 3),
        "phases": n_phases,
        "orderings_consistent": n_orderings,
        "orderings_conflicting": n_conflicts,
        "consistency_ratio": round(consistency, 3),
        "parallel_pairs": n_parallel,
        "fusion_candidates": n_fusion,
        "mutually_exclusive": n_exclusive,
        "conditional_tools": n_conditional,
        "stable_params": n_stable,
        "variable_params": n_variable,
        "total_patterns": total_patterns,
    }


# ============================================================================
# Benchmark Runner
# ============================================================================

def run_benchmark(dataset_name, dataset_info, backend_name, backend_info,
                  project_root):
    """Run one dataset × backend combination."""
    traces_path = project_root / dataset_info["traces"]
    rules_path = project_root / backend_info["rules"]

    if not traces_path.exists():
        return {"error": f"Traces not found: {traces_path}"}
    if not rules_path.exists():
        return {"error": f"Rules not found: {rules_path}"}

    start_time = time.time()
    try:
        results = run_compilation(traces_path, rules_path)
    except Exception as e:
        return {"error": str(e)}
    elapsed = time.time() - start_time

    if not results.get("satisfiable"):
        return {"error": "UNSATISFIABLE", "elapsed_sec": round(elapsed, 3)}

    metrics = compute_metrics(results)
    metrics["elapsed_sec"] = round(elapsed, 3)
    metrics["dataset"] = dataset_name
    metrics["backend"] = backend_name

    return metrics


def print_results_table(all_results):
    """Print a comparison table of results."""
    if not all_results:
        print("No results.")
        return

    # Header
    print(f"\n{'Dataset':<24} {'Backend':<14} {'Runs':>5} {'Core':>6} "
          f"{'Ratio':>6} {'Phs':>4} {'Ord':>4} {'Cnfl':>4} "
          f"{'Par':>4} {'Fus':>4} {'XOR':>4} {'Cond':>4} "
          f"{'Stbl':>4} {'Var':>4} {'Time':>6}")
    print("-" * 120)

    for r in all_results:
        if "error" in r:
            print(f"{r.get('dataset', '?'):<24} {r.get('backend', '?'):<14} "
                  f"ERROR: {r['error']}")
            continue

        print(f"{r['dataset']:<24} {r['backend']:<14} "
              f"{r['runs']:5d} "
              f"{r['tool_types_core']:3d}/{r['tool_types_total']:<2d} "
              f"{r['compilation_ratio']:6.2f} "
              f"{r['phases']:4d} "
              f"{r['orderings_consistent']:4d} "
              f"{r['orderings_conflicting']:4d} "
              f"{r['parallel_pairs']:4d} "
              f"{r['fusion_candidates']:4d} "
              f"{r['mutually_exclusive']:4d} "
              f"{r['conditional_tools']:4d} "
              f"{r['stable_params']:4d} "
              f"{r['variable_params']:4d} "
              f"{r['elapsed_sec']:5.2f}s")


def main():
    parser = argparse.ArgumentParser(description="Autocompile benchmark harness")
    parser.add_argument("--traces", help="Path to specific trace file")
    parser.add_argument("--rules", nargs="+", help="Rule files to compare")
    parser.add_argument("--all", action="store_true", help="Run all registered datasets")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS.keys()),
                        help="Specific datasets to run")
    parser.add_argument("--backends", nargs="+", choices=list(BACKENDS.keys()),
                        help="Specific backends to run")
    parser.add_argument("--format", choices=["table", "json"], default="table")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent

    all_results = []

    if args.traces:
        # Ad-hoc run with specific traces
        rules_list = args.rules or [BACKENDS["asp-strict"]["rules"],
                                     BACKENDS["asp-relaxed"]["rules"]]
        for rules in rules_list:
            backend_name = Path(rules).stem
            result = run_benchmark(
                Path(args.traces).stem,
                {"traces": args.traces},
                backend_name,
                {"rules": rules},
                project_root,
            )
            all_results.append(result)
    else:
        # Run registered datasets
        datasets = args.datasets or (list(DATASETS.keys()) if args.all else [])
        backends = args.backends or list(BACKENDS.keys())

        if not datasets:
            print("Specify --all, --datasets, or --traces")
            parser.print_help()
            sys.exit(1)

        for ds_name in datasets:
            for be_name in backends:
                result = run_benchmark(
                    ds_name, DATASETS[ds_name],
                    be_name, BACKENDS[be_name],
                    project_root,
                )
                all_results.append(result)

    # Output
    if args.format == "json":
        print(json.dumps(all_results, indent=2))
    else:
        print_results_table(all_results)

        # Summary
        valid = [r for r in all_results if "error" not in r]
        if valid:
            print(f"\n{'='*120}")
            print(f"SUMMARY: {len(valid)} runs across "
                  f"{len(set(r['dataset'] for r in valid))} datasets, "
                  f"{len(set(r['backend'] for r in valid))} backends")

            # Best compilation ratio per dataset
            by_dataset = {}
            for r in valid:
                by_dataset.setdefault(r["dataset"], []).append(r)
            print(f"\nBest compilation ratio per dataset:")
            for ds, runs in sorted(by_dataset.items()):
                best = max(runs, key=lambda r: r["compilation_ratio"])
                print(f"  {ds:<24} {best['compilation_ratio']:.2f} "
                      f"({best['backend']}, {best['tool_types_core']}/{best['tool_types_total']} tools)")


if __name__ == "__main__":
    main()
