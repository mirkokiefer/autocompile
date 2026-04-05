#!/usr/bin/env python3
"""
compile.py — Datalog-based pattern mining for autocompile.

Port of src/compile.py's ASP strategy to pure-Python Datalog.
Computes the full monotonic fragment; prints results side-by-side
with Clingo for validation.

Usage:
    python -m datalog.compile --traces examples/travel-updates/traces.lp
    python -m datalog.compile --traces examples/travel-updates/traces.lp --compare
"""

import argparse
import re
import time
from pathlib import Path

from .engine import DatalogEngine
from .mine_patterns import GLUE_TOOLS, build_rules, compute_aggregates


def parse_lp_facts(path: Path) -> list[tuple[str, list[str]]]:
    """Parse .lp fact files into (relation, [args]) tuples."""
    facts = []
    text = path.read_text()

    # Match: relation("arg1", "arg2", ...).  or  relation("arg").
    pattern = re.compile(r'^(\w+)\(([^)]*)\)\s*\.', re.MULTILINE)

    for m in pattern.finditer(text):
        relation = m.group(1)
        args_str = m.group(2)
        args = []
        for arg in re.findall(r'"([^"]*)"', args_str):
            args.append(arg)
        if args:
            facts.append((relation, args))

    return facts


def load_traces(engine: DatalogEngine, path: Path):
    """Load .lp trace facts into the Datalog engine."""
    facts = parse_lp_facts(path)
    count = 0
    for relation, args in facts:
        engine.add_fact(relation, *args)
        count += 1

    # Add glue tool facts
    for tool in GLUE_TOOLS:
        engine.add_fact("glue_tool", tool)

    return count


def run_datalog(traces_path: Path, verbose: bool = False) -> dict:
    """Run the Datalog pattern mining pipeline."""
    engine = DatalogEngine()

    # Load trace facts
    t0 = time.perf_counter()
    n_facts = load_traces(engine, traces_path)
    t_load = time.perf_counter() - t0

    # Register rules and solve monotonic fragment
    build_rules(engine)
    t1 = time.perf_counter()
    engine.solve(verbose=verbose)
    t_solve = time.perf_counter() - t1

    # Compute aggregates (counts, thresholds, derived relations)
    t2 = time.perf_counter()
    compute_aggregates(engine)
    t_agg = time.perf_counter() - t2

    # Now re-solve with core_tool facts available (for precedes_in_job, parallel_pair, etc.)
    t3 = time.perf_counter()
    engine.solve(verbose=verbose)
    t_resolve = time.perf_counter() - t3

    # Recompute aggregates that depend on the second solve
    t4 = time.perf_counter()
    # Clear derived aggregates so they're recomputed fresh
    for rel in ["consistent_order", "conflicting_order", "order_evidence",
                "conditional_tool", "conditional_rate", "fusion_candidate",
                "mutually_exclusive", "phase_zero"]:
        engine.facts.pop(rel, None)
    compute_aggregates(engine)
    t_agg2 = time.perf_counter() - t4

    total = t_load + t_solve + t_agg + t_resolve + t_agg2

    return {
        "engine": engine,
        "timing": {
            "load": t_load,
            "solve_monotonic": t_solve,
            "aggregates": t_agg,
            "solve_with_core": t_resolve,
            "aggregates_2": t_agg2,
            "total": total,
        },
        "n_facts": n_facts,
    }


def print_results(engine: DatalogEngine, timing: dict, n_facts: int):
    """Print pattern mining results."""
    facts = engine.facts

    job_count = next(iter(facts.get("job_count", {(0,)})))[0]
    core_tools = sorted(t for (t,) in facts.get("core_tool", set()))
    tool_counts = {t: n for (t, n) in facts.get("tool_job_count", set())}

    print(f"{'='*60}")
    print(f"DATALOG PATTERN MINING ({job_count} runs, {n_facts} input facts)")
    print(f"{'='*60}")
    print(f"Timing: {timing['total']*1000:.1f}ms total")
    print(f"  load: {timing['load']*1000:.1f}ms")
    print(f"  solve (monotonic): {timing['solve_monotonic']*1000:.1f}ms")
    print(f"  aggregates: {timing['aggregates']*1000:.1f}ms")
    print(f"  solve (with core_tool): {timing['solve_with_core']*1000:.1f}ms")
    print(f"  aggregates (final): {timing['aggregates_2']*1000:.1f}ms")
    print()

    print(f"Core tools (>= 50% of runs):")
    for t in core_tools:
        print(f"  {t}: {tool_counts.get(t, '?')}/{job_count}")
    print()

    parallel = sorted(facts.get("parallel_pair", set()))
    if parallel:
        print("Parallel pairs:")
        for t1, t2 in parallel:
            print(f"  {t1} || {t2}")
        print()

    consistent = sorted(facts.get("consistent_order", set()))
    if consistent:
        print("Consistent ordering:")
        for t1, t2 in consistent:
            print(f"  {t1} -> {t2}")
        print()

    conflicting = sorted(facts.get("conflicting_order", set()))
    if conflicting:
        print("Conflicting orderings (need Clingo Layer 2 to resolve):")
        for t1, t2 in conflicting:
            ev1 = ev2 = "?"
            for (a, b, n) in facts.get("order_evidence", set()):
                if a == t1 and b == t2:
                    ev1 = n
                if a == t2 and b == t1:
                    ev2 = n
            print(f"  {t1} <-> {t2}  (evidence: {t1}->{t2}={ev1}, {t2}->{t1}={ev2})")
        print()

    conditionals = sorted(facts.get("conditional_tool", set()))
    if conditionals:
        print("Conditional execution:")
        rates = {(b, a): (nb, na) for (b, a, nb, na) in facts.get("conditional_rate", set())}
        for b, a in conditionals:
            nb, na = rates.get((b, a), ("?", "?"))
            print(f"  {b} if {a} ({nb}/{na})")
        print()

    fusion = sorted(facts.get("fusion_candidate", set()))
    if fusion:
        print("Fusion candidates:")
        for a, b in fusion:
            print(f"  {a} + {b}")
        print()

    exclusive = sorted(facts.get("mutually_exclusive", set()))
    if exclusive:
        print("Mutually exclusive:")
        for a, b in exclusive:
            print(f"  {a} XOR {b}")
        print()

    # Summary: what Layer 2 (Clingo) still needs to decide
    if conflicting:
        print(f"{'='*60}")
        print("LAYER 2 NEEDED (Clingo choice/optimization):")
        print(f"  {len(conflicting)} conflicting orderings to resolve")
        print(f"  Phase assignment depends on resolved orderings")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Datalog pattern mining for autocompile")
    parser.add_argument("--traces", default="examples/travel-updates/traces.lp")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--compare", action="store_true", help="Compare with Clingo output")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    traces_path = Path(args.traces) if Path(args.traces).is_absolute() else project_root / args.traces

    if not traces_path.exists():
        print(f"ERROR: {traces_path} not found")
        return

    result = run_datalog(traces_path, verbose=args.verbose)
    print_results(result["engine"], result["timing"], result["n_facts"])

    if args.compare:
        print()
        print(f"\n{'='*60}")
        print("CLINGO COMPARISON")
        print(f"{'='*60}")
        try:
            import sys
            sys.path.insert(0, str(project_root / "src"))
            from compile import run_asp_strategy
            rules_path = project_root / "rules" / "mine_patterns.lp"

            t0 = time.perf_counter()
            clingo_results = run_asp_strategy(traces_path, rules_path, False)
            t_clingo = time.perf_counter() - t0

            print(f"Clingo time: {t_clingo*1000:.1f}ms")
            print()

            # Compare core tools
            dl_core = sorted(t for (t,) in result["engine"].facts.get("core_tool", set()))
            cl_core = sorted(clingo_results["core_tools"])
            _compare("Core tools", dl_core, cl_core)

            # Compare parallel pairs
            dl_par = sorted(result["engine"].facts.get("parallel_pair", set()))
            cl_par = sorted(clingo_results["parallel_pairs"])
            _compare("Parallel pairs", dl_par, cl_par)

            # Compare conflicting orderings
            dl_conf = sorted(result["engine"].facts.get("conflicting_order", set()))
            cl_conf = sorted(clingo_results["conflicting_order"])
            _compare("Conflicting orderings", dl_conf, cl_conf)

            # Compare conditionals
            dl_cond = sorted(result["engine"].facts.get("conditional_tool", set()))
            cl_cond = sorted(clingo_results["conditionals"])
            _compare("Conditionals", dl_cond, cl_cond)

            # Compare mutually exclusive
            dl_excl = sorted(result["engine"].facts.get("mutually_exclusive", set()))
            cl_excl = sorted(clingo_results["mutually_exclusive"])
            _compare("Mutually exclusive", dl_excl, cl_excl)

        except ImportError as e:
            print(f"Cannot compare: {e}")


def _compare(label: str, datalog: list, clingo: list):
    dl_set = set(map(tuple, datalog))
    cl_set = set(map(tuple, clingo))
    if dl_set == cl_set:
        print(f"  {label}: MATCH ({len(dl_set)} items)")
    else:
        print(f"  {label}: MISMATCH")
        only_dl = dl_set - cl_set
        only_cl = cl_set - dl_set
        if only_dl:
            print(f"    Only in Datalog: {only_dl}")
        if only_cl:
            print(f"    Only in Clingo:  {only_cl}")


if __name__ == "__main__":
    main()
