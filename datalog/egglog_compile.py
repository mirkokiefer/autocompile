#!/usr/bin/env python3
"""
egglog_compile.py — Pattern mining via egglog (Datalog + equality saturation).

egglog = Datalog + e-graphs. It handles the monotonic rules natively
and can express rewrite-based optimizations that neither Soufflé nor Clingo can.

Negation and aggregates are computed in Python post-processing (egglog
doesn't support stratified negation — it's not traditional Datalog,
it's equality saturation with Datalog-style rules).

Usage:
    python -m datalog.egglog_compile --traces examples/travel-updates/traces.lp
    python -m datalog.egglog_compile --traces examples/travel-updates/traces.lp --compare
"""

import argparse
import re
import time
from pathlib import Path

from egglog import EGraph, relation, rule, vars_, String, ne, get_callable_args, get_literal_value


GLUE_TOOLS = {"llm_generate", "notify_user", "prompt_user"}


def parse_lp_facts(path: Path) -> dict[str, list[list[str]]]:
    facts: dict[str, list[list[str]]] = {}
    text = path.read_text()
    pattern = re.compile(r'^(\w+)\(([^)]*)\)\s*\.', re.MULTILINE)
    for m in pattern.finditer(text):
        relation_name = m.group(1)
        args = re.findall(r'"([^"]*)"', m.group(2))
        if args:
            facts.setdefault(relation_name, []).append(args)
    return facts


def run_egglog(traces_path: Path, verbose: bool = False) -> dict:
    """Run egglog pattern mining pipeline."""

    t0 = time.perf_counter()
    raw_facts = parse_lp_facts(traces_path)
    t_load = time.perf_counter() - t0

    # --- Declare relations ---
    job = relation("job", String)
    call = relation("call", String, String, String, String)
    depends = relation("depends", String, String, String)
    spawned_by = relation("spawned_by", String, String, String)
    param = relation("param", String, String, String, String)
    glue_tool = relation("glue_tool", String)

    actionable_call = relation("actionable_call", String, String, String)
    tool_in_job = relation("tool_in_job", String, String)
    known_tool = relation("known_tool", String)
    core_tool = relation("core_tool", String)
    depends_transitive = relation("depends_transitive", String, String, String)
    precedes_in_job = relation("precedes_in_job", String, String, String)
    parallel_pair = relation("parallel_pair", String, String)
    both_in_any_job = relation("both_in_any_job", String, String)
    known_param = relation("known_param", String, String, String)

    # Variables
    j, c, t, s = vars_("j c t s", String)
    c1, c2, t1, t2 = vars_("c1 c2 t1 t2", String)
    p, p1, p2 = vars_("p p1 p2", String)
    a, b, x = vars_("a b x", String)
    k, v = vars_("k v", String)

    # --- Build e-graph ---
    t1_time = time.perf_counter()
    egraph = EGraph()

    # Register facts
    fact_commands = []
    for tool_name in GLUE_TOOLS:
        fact_commands.append(glue_tool(tool_name))

    for rel_name, rows in raw_facts.items():
        rel_fn = {"job": job, "call": call, "depends": depends,
                  "spawned_by": spawned_by, "param": param}.get(rel_name)
        if rel_fn is not None:
            for row in rows:
                fact_commands.append(rel_fn(*row))

    egraph.register(*fact_commands)

    # Pre-compute core tools (egglog has no aggregates)
    job_count = len(raw_facts.get("job", []))
    tool_jobs: dict[str, set[str]] = {}
    for row in raw_facts.get("call", []):
        j_id, c_id, t_name, status = row
        if t_name not in GLUE_TOOLS and status in ("completed", "failed"):
            tool_jobs.setdefault(t_name, set()).add(j_id)

    core_facts = []
    for tool_name, jobs_set in tool_jobs.items():
        if len(jobs_set) * 2 >= job_count:
            core_facts.append(core_tool(tool_name))
    if core_facts:
        egraph.register(*core_facts)

    t_setup = time.perf_counter() - t1_time

    # --- Register rules ---
    # Use fresh variables for each rule to avoid conflicts
    j1, c1a, t1a = vars_("j1 c1a t1a", String)
    j2, c2a, t2a = vars_("j2 c2a t2a", String)
    j3, a3, b3 = vars_("j3 a3 b3", String)
    j4, a4, b4, c4 = vars_("j4 a4 b4 c4", String)
    j5, c5, t5, s5 = vars_("j5 c5 t5 s5", String)
    j6, c6, t6 = vars_("j6 c6 t6", String)
    j7, c7a, c7b, t7a, t7b, p7 = vars_("j7 c7a c7b t7a t7b p7", String)
    j8, c8a, c8b, t8a, t8b, p8a, p8b = vars_("j8 c8a c8b t8a t8b p8a p8b", String)
    j9, c9a, c9b, t9a, t9b, p9, p9b = vars_("j9 c9a c9b t9a t9b p9 p9b", String)
    j10, a10, b10 = vars_("j10 a10 b10", String)
    j11, c11, t11, k11, v11 = vars_("j11 c11 t11 k11 v11", String)

    # Pre-compute actionable_call facts (egglog can't negate glue_tool)
    ac_facts = []
    for row in raw_facts.get("call", []):
        j_id, c_id, t_name, status = row
        if t_name not in GLUE_TOOLS and status in ("completed", "failed"):
            ac_facts.append(actionable_call(j_id, c_id, t_name))
    egraph.register(*ac_facts)

    egraph.register(
        # tool_in_job
        rule(actionable_call(j1, c1a, t1a)).then(tool_in_job(j1, t1a)),
        # known_tool
        rule(tool_in_job(j2, t2a)).then(known_tool(t2a)),

        # depends_transitive (recursive) — this is THE Datalog win
        rule(depends(j3, a3, b3)).then(depends_transitive(j3, a3, b3)),
        rule(depends(j4, a4, b4), depends_transitive(j4, b4, c4)).then(depends_transitive(j4, a4, c4)),

        # precedes_in_job: via transitive dependency on parents
        rule(
            actionable_call(j8, c8a, t8a), actionable_call(j8, c8b, t8b),
            spawned_by(j8, c8a, p8a), spawned_by(j8, c8b, p8b),
            depends_transitive(j8, p8b, p8a),
            core_tool(t8a), core_tool(t8b),
            ne(p8a).to(p8b), ne(t8a).to(t8b),
        ).then(precedes_in_job(j8, t8a, t8b)),

        # precedes_in_job: direct dependency
        rule(
            actionable_call(j9, c9a, t9a), actionable_call(j9, c9b, t9b),
            spawned_by(j9, c9b, p9b),
            depends(j9, p9b, c9a),
            core_tool(t9a), core_tool(t9b),
            ne(t9a).to(t9b),
        ).then(precedes_in_job(j9, t9a, t9b)),

        # known_param
        rule(
            actionable_call(j11, c11, t11), param(j11, c11, k11, v11),
            core_tool(t11),
        ).then(known_param(t11, k11, v11)),

        # Note: parallel_pair and both_in_any_job need string < comparison
        # which egglog doesn't support. Computed in post-processing.
    )

    # --- Solve ---
    t2_time = time.perf_counter()
    egraph.run(100)
    t_solve = time.perf_counter() - t2_time

    # --- Extract results ---
    t3_time = time.perf_counter()
    outputs = {}

    # Extract using function_values
    for name, rel_fn in [
        ("core_tool", core_tool),
        ("precedes_in_job", precedes_in_job),
        ("known_tool", known_tool),
        ("tool_in_job", tool_in_job),
        ("known_param", known_param),
    ]:
        try:
            size = egraph.function_size(rel_fn)
            if size > 0:
                rows = []
                for expr in egraph.function_values(rel_fn):
                    args = get_callable_args(expr)
                    row = tuple(get_literal_value(a) for a in args)
                    rows.append(row)
                outputs[name] = rows
        except Exception as e:
            if verbose:
                print(f"  Could not extract {name}: {e}")

    # Post-process: compute negation-dependent relations in Python
    _compute_post_processing(outputs, raw_facts, tool_jobs, job_count)

    t_extract = time.perf_counter() - t3_time

    return {
        "outputs": outputs,
        "timing": {
            "load": t_load,
            "setup": t_setup,
            "solve": t_solve,
            "extract": t_extract,
            "total": t_load + t_setup + t_solve + t_extract,
        },
    }


def _compute_post_processing(outputs: dict, raw_facts: dict, tool_jobs: dict, job_count: int):
    """Compute negation/aggregate dependent relations in Python."""

    tool_counts = {t: len(js) for t, js in tool_jobs.items()}
    core_set = {t for (t,) in outputs.get("core_tool", [])}

    # tool_in_job as set for lookups
    tij = set()
    for row in outputs.get("tool_in_job", []):
        tij.add(tuple(row))

    all_tools = {t for (t,) in outputs.get("known_tool", [])}

    # parallel_pair (needs string < which egglog doesn't support)
    # Compute from actionable_call + spawned_by
    ac_by_job: dict[str, list[tuple]] = {}
    sb_lookup: dict[tuple, str] = {}  # (job, step) -> parent
    for row in raw_facts.get("call", []):
        j_id, c_id, t_name, status = row
        if t_name not in GLUE_TOOLS and status in ("completed", "failed"):
            ac_by_job.setdefault(j_id, []).append((c_id, t_name))
    for row in raw_facts.get("spawned_by", []):
        sb_lookup[(row[0], row[1])] = row[2]

    parallel_pairs = set()
    for j_id, calls in ac_by_job.items():
        # Group by parent
        by_parent: dict[str, list[str]] = {}
        for c_id, t_name in calls:
            parent = sb_lookup.get((j_id, c_id))
            if parent and t_name in core_set:
                by_parent.setdefault(parent, []).append(t_name)
        for parent, tools in by_parent.items():
            unique = set(tools)
            for t1 in unique:
                for t2 in unique:
                    if t1 < t2:
                        parallel_pairs.add((t1, t2))
    outputs["parallel_pair"] = sorted(parallel_pairs)

    # both_in_any_job
    both = set()
    for j_id, t_name in tij:
        for j_id2, t_name2 in tij:
            if j_id == j_id2 and t_name < t_name2:
                both.add((t_name, t_name2))
    outputs["both_in_any_job"] = sorted(both)

    # b_without_a
    b_without_a = set()
    for b_tool in all_tools:
        for a_tool in all_tools:
            if a_tool == b_tool:
                continue
            for j_id, t_name in tij:
                if t_name == b_tool and (j_id, a_tool) not in tij:
                    b_without_a.add((b_tool, a_tool))
                    break

    # conditional_tool
    conditionals = []
    for a_tool in all_tools:
        na = tool_counts.get(a_tool, 0)
        for b_tool in all_tools:
            if a_tool == b_tool:
                continue
            nb = tool_counts.get(b_tool, 0)
            if nb >= na or nb == 0:
                continue
            if nb * 10 < na:
                continue
            if (b_tool, a_tool) not in b_without_a:
                conditionals.append((b_tool, a_tool))
                outputs.setdefault("conditional_rate", []).append((b_tool, a_tool, nb, na))
    outputs["conditional_tool"] = conditionals

    # split_occurrence
    split = set()
    for a_tool in all_tools:
        for b_tool in all_tools:
            if a_tool >= b_tool:
                continue
            for j_id, t_name in tij:
                if t_name == a_tool and (j_id, b_tool) not in tij:
                    split.add((a_tool, b_tool))
                    break
            for j_id, t_name in tij:
                if t_name == b_tool and (j_id, a_tool) not in tij:
                    split.add((a_tool, b_tool))
                    break

    # mutually_exclusive
    exclusive = []
    for a_tool in all_tools:
        for b_tool in all_tools:
            if a_tool >= b_tool:
                continue
            if tool_jobs.get(a_tool) and tool_jobs.get(b_tool):
                if (a_tool, b_tool) not in both:
                    exclusive.append((a_tool, b_tool))
    outputs["mutually_exclusive"] = exclusive

    # consistent_order / conflicting_order
    precedes = outputs.get("precedes_in_job", [])
    pair_evidence: dict[tuple, set] = {}
    for row in precedes:
        j_id, t1, t2 = row
        pair_evidence.setdefault((t1, t2), set()).add(j_id)

    consistent = []
    conflicting = []
    order_evidence = []
    for (t1, t2), evidence in pair_evidence.items():
        reverse = pair_evidence.get((t2, t1), set())
        if not reverse:
            consistent.append((t1, t2))
        elif t1 < t2:
            conflicting.append((t1, t2))
            order_evidence.append((t1, t2, len(evidence)))
            order_evidence.append((t2, t1, len(reverse)))
    outputs["consistent_order"] = consistent
    outputs["conflicting_order"] = conflicting
    outputs["order_evidence"] = order_evidence

    # fusion_candidate
    consistent_set = set(map(tuple, consistent))
    has_intermediate = set()
    for (a_tool, x_tool) in consistent_set:
        if x_tool not in core_set:
            continue
        for (x2, b_tool) in consistent_set:
            if x2 == x_tool and a_tool != x_tool and x_tool != b_tool:
                has_intermediate.add((a_tool, b_tool))
    fusion = []
    for (a_tool, b_tool) in consistent_set:
        if a_tool in core_set and b_tool in core_set:
            if (a_tool, b_tool) not in split and (a_tool, b_tool) not in has_intermediate:
                fusion.append((a_tool, b_tool))
    outputs["fusion_candidate"] = fusion

    # tool_job_count
    outputs["tool_job_count"] = [(t, n) for t, n in tool_counts.items()]
    outputs["job_count"] = [(job_count,)]


def print_results(outputs: dict, timing: dict):
    job_count = outputs.get("job_count", [(0,)])[0][0]
    core_tools = sorted(t for (t,) in outputs.get("core_tool", []))
    tool_counts = {t: n for (t, n) in outputs.get("tool_job_count", [])}

    print(f"{'='*60}")
    print(f"EGGLOG PATTERN MINING ({job_count} runs)")
    print(f"{'='*60}")
    print(f"Timing: {timing['total']*1000:.1f}ms total")
    print(f"  load: {timing['load']*1000:.1f}ms")
    print(f"  setup (facts+rules): {timing['setup']*1000:.1f}ms")
    print(f"  egglog solve: {timing['solve']*1000:.1f}ms")
    print(f"  extract+post: {timing['extract']*1000:.1f}ms")
    print()

    print("Core tools (>= 50% of runs):")
    for t in core_tools:
        print(f"  {t}: {tool_counts.get(t, '?')}/{job_count}")
    print()

    parallel = sorted(outputs.get("parallel_pair", []))
    if parallel:
        print("Parallel pairs:")
        for row in parallel:
            print(f"  {row[0]} || {row[1]}")
        print()

    conflicting = sorted(outputs.get("conflicting_order", []))
    if conflicting:
        print("Conflicting orderings:")
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

    exclusive = sorted(outputs.get("mutually_exclusive", []))
    if exclusive:
        print("Mutually exclusive:")
        for a, b in exclusive:
            print(f"  {a} XOR {b}")
        print()

    if conflicting:
        print(f"{'='*60}")
        print(f"LAYER 2 NEEDED: {len(conflicting)} conflicting orderings")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="egglog pattern mining")
    parser.add_argument("--traces", default="examples/travel-updates/traces.lp")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    traces_path = Path(args.traces) if Path(args.traces).is_absolute() else project_root / args.traces

    if not traces_path.exists():
        print(f"ERROR: {traces_path} not found")
        return

    result = run_egglog(traces_path, verbose=args.verbose)
    if result is None:
        return

    print_results(result["outputs"], result["timing"])

    if args.compare:
        print(f"\n{'='*60}")
        print("COMPARISON: egglog vs Clingo")
        print(f"{'='*60}")
        try:
            import sys
            sys.path.insert(0, str(project_root / "src"))
            from compile import run_asp_strategy

            clingo_rules = project_root / "rules" / "mine_patterns.lp"
            t0 = time.perf_counter()
            clingo_results = run_asp_strategy(traces_path, clingo_rules, False)
            t_clingo = time.perf_counter() - t0
            print(f"Clingo time: {t_clingo*1000:.1f}ms\n")

            so = result["outputs"]
            _compare("Core tools",
                     sorted(t for (t,) in so.get("core_tool", [])),
                     sorted(clingo_results["core_tools"]))
            _compare("Parallel pairs",
                     sorted(tuple(r) for r in so.get("parallel_pair", [])),
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


def _compare(label, engine_list, clingo_list):
    s_set = set(map(tuple, engine_list))
    c_set = set(map(tuple, clingo_list))
    if s_set == c_set:
        print(f"  {label}: MATCH ({len(s_set)} items)")
    else:
        print(f"  {label}: MISMATCH")
        if s_set - c_set:
            print(f"    Only in egglog: {s_set - c_set}")
        if c_set - s_set:
            print(f"    Only in Clingo: {c_set - s_set}")


if __name__ == "__main__":
    main()
