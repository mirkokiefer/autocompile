#!/usr/bin/env python3
"""
souffle_compile.py — Run pattern mining via Soufflé (compiled Datalog).

Converts .lp trace facts to Soufflé's tab-separated format, runs Soufflé,
and compares results with Clingo.

Usage:
    python -m datalog.souffle_compile --traces examples/travel-updates/traces.lp
    python -m datalog.souffle_compile --traces examples/travel-updates/traces.lp --compare
"""

import argparse
import csv
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path


def parse_lp_facts(path: Path) -> dict[str, list[list[str]]]:
    """Parse .lp fact files into {relation: [[args]]}."""
    facts: dict[str, list[list[str]]] = {}
    text = path.read_text()
    pattern = re.compile(r'^(\w+)\(([^)]*)\)\s*\.', re.MULTILINE)

    for m in pattern.finditer(text):
        relation = m.group(1)
        args = re.findall(r'"([^"]*)"', m.group(2))
        if args:
            facts.setdefault(relation, []).append(args)

    return facts


def write_souffle_facts(facts: dict[str, list[list[str]]], facts_dir: Path):
    """Write facts as tab-separated files for Soufflé."""
    for relation, rows in facts.items():
        with open(facts_dir / f"{relation}.facts", "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            for row in rows:
                writer.writerow(row)


def read_souffle_output(output_dir: Path) -> dict[str, list[tuple]]:
    """Read Soufflé's tab-separated output files."""
    results = {}
    for f in output_dir.iterdir():
        if f.suffix == ".csv":
            relation = f.stem
            rows = []
            with open(f) as fh:
                reader = csv.reader(fh, delimiter="\t")
                for row in reader:
                    # Try to parse numbers
                    parsed = []
                    for v in row:
                        try:
                            parsed.append(int(v))
                        except ValueError:
                            parsed.append(v)
                    rows.append(tuple(parsed))
            results[relation] = rows
    return results


def run_souffle(traces_path: Path, rules_path: Path, verbose: bool = False) -> dict:
    """Run Soufflé pattern mining pipeline."""
    with tempfile.TemporaryDirectory() as tmpdir:
        facts_dir = Path(tmpdir) / "facts"
        output_dir = Path(tmpdir) / "output"
        facts_dir.mkdir()
        output_dir.mkdir()

        # Parse and write facts
        t0 = time.perf_counter()
        facts = parse_lp_facts(traces_path)
        write_souffle_facts(facts, facts_dir)
        t_load = time.perf_counter() - t0

        if verbose:
            for rel, rows in facts.items():
                print(f"  {rel}: {len(rows)} facts")

        # Run Soufflé
        t1 = time.perf_counter()
        result = subprocess.run(
            ["souffle", "-F", str(facts_dir), "-D", str(output_dir), str(rules_path)],
            capture_output=True,
            text=True,
        )
        t_solve = time.perf_counter() - t1

        if result.returncode != 0:
            print(f"Soufflé error:\n{result.stderr}")
            return None

        if verbose and result.stderr:
            print(f"Soufflé stderr: {result.stderr}")

        # Read outputs
        t2 = time.perf_counter()
        outputs = read_souffle_output(output_dir)
        t_read = time.perf_counter() - t2

        return {
            "outputs": outputs,
            "timing": {
                "load": t_load,
                "solve": t_solve,
                "read": t_read,
                "total": t_load + t_solve + t_read,
            },
        }


def print_results(outputs: dict[str, list[tuple]], timing: dict):
    """Print Soufflé results."""
    job_count = outputs.get("job_count", [(0,)])[0][0]
    core_tools = sorted(t for (t,) in outputs.get("core_tool", []))
    tool_counts = {t: n for (t, n) in outputs.get("tool_job_count", [])}

    print(f"{'='*60}")
    print(f"SOUFFLÉ PATTERN MINING ({job_count} runs)")
    print(f"{'='*60}")
    print(f"Timing: {timing['total']*1000:.1f}ms total")
    print(f"  load/convert: {timing['load']*1000:.1f}ms")
    print(f"  soufflé solve: {timing['solve']*1000:.1f}ms")
    print(f"  read output: {timing['read']*1000:.1f}ms")
    print()

    print("Core tools (>= 50% of runs):")
    for t in core_tools:
        print(f"  {t}: {tool_counts.get(t, '?')}/{job_count}")
    print()

    parallel = sorted(outputs.get("parallel_pair", []))
    if parallel:
        print("Parallel pairs:")
        for t1, t2 in parallel:
            print(f"  {t1} || {t2}")
        print()

    consistent = sorted(outputs.get("consistent_order", []))
    if consistent:
        print("Consistent ordering:")
        for t1, t2 in consistent:
            print(f"  {t1} -> {t2}")
        print()

    conflicting = sorted(outputs.get("conflicting_order", []))
    if conflicting:
        print("Conflicting orderings (need Clingo to resolve):")
        evidence = {(t1, t2): n for t1, t2, n in outputs.get("order_evidence", [])}
        for t1, t2 in conflicting:
            ev1 = evidence.get((t1, t2), "?")
            ev2 = evidence.get((t2, t1), "?")
            print(f"  {t1} <-> {t2}  ({t1}->{t2}={ev1}, {t2}->{t1}={ev2})")
        print()

    conditionals = sorted(outputs.get("conditional_tool", []))
    if conditionals:
        print("Conditional execution:")
        rates = {(b, a): (nb, na) for b, a, nb, na in outputs.get("conditional_rate", [])}
        for b, a in conditionals:
            nb, na = rates.get((b, a), ("?", "?"))
            print(f"  {b} if {a} ({nb}/{na})")
        print()

    fusion = sorted(outputs.get("fusion_candidate", []))
    if fusion:
        print("Fusion candidates:")
        for a, b in fusion:
            print(f"  {a} + {b}")
        print()

    exclusive = sorted(outputs.get("mutually_exclusive", []))
    if exclusive:
        print("Mutually exclusive:")
        for a, b in exclusive:
            print(f"  {a} XOR {b}")
        print()

    if conflicting:
        print(f"{'='*60}")
        print(f"LAYER 2 NEEDED: {len(conflicting)} conflicting orderings for Clingo")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Soufflé pattern mining for autocompile")
    parser.add_argument("--traces", default="examples/travel-updates/traces.lp")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    traces_path = Path(args.traces) if Path(args.traces).is_absolute() else project_root / args.traces
    rules_path = Path(__file__).parent / "mine_patterns.dl"

    if not traces_path.exists():
        print(f"ERROR: {traces_path} not found")
        return

    # Check souffle is installed
    if subprocess.run(["which", "souffle"], capture_output=True).returncode != 0:
        print("ERROR: souffle not installed. Run: brew install souffle")
        return

    result = run_souffle(traces_path, rules_path, verbose=args.verbose)
    if result is None:
        return

    print_results(result["outputs"], result["timing"])

    if args.compare:
        print(f"\n{'='*60}")
        print("COMPARISON: Soufflé vs Clingo")
        print(f"{'='*60}")
        try:
            import sys
            sys.path.insert(0, str(project_root / "src"))
            from compile import run_asp_strategy

            clingo_rules = project_root / "rules" / "mine_patterns.lp"
            t0 = time.perf_counter()
            clingo_results = run_asp_strategy(traces_path, clingo_rules, False)
            t_clingo = time.perf_counter() - t0

            print(f"Clingo time: {t_clingo*1000:.1f}ms")
            print()

            so = result["outputs"]
            _compare("Core tools",
                     sorted(t for (t,) in so.get("core_tool", [])),
                     sorted(clingo_results["core_tools"]))
            _compare("Parallel pairs",
                     sorted(so.get("parallel_pair", [])),
                     sorted(clingo_results["parallel_pairs"]))
            _compare("Conflicting orderings",
                     sorted(so.get("conflicting_order", [])),
                     sorted(clingo_results["conflicting_order"]))
            _compare("Conditionals",
                     sorted(so.get("conditional_tool", [])),
                     sorted(clingo_results["conditionals"]))
            _compare("Mutually exclusive",
                     sorted(so.get("mutually_exclusive", [])),
                     sorted(clingo_results["mutually_exclusive"]))

        except ImportError as e:
            print(f"Cannot compare: {e}")


def _compare(label: str, souffle: list, clingo: list):
    s_set = set(map(tuple, souffle))
    c_set = set(map(tuple, clingo))
    if s_set == c_set:
        print(f"  {label}: MATCH ({len(s_set)} items)")
    else:
        print(f"  {label}: MISMATCH")
        only_s = s_set - c_set
        only_c = c_set - s_set
        if only_s:
            print(f"    Only in Soufflé: {only_s}")
        if only_c:
            print(f"    Only in Clingo:  {only_c}")


if __name__ == "__main__":
    main()
