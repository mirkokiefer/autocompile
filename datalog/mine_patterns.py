"""
mine_patterns.py — The monotonic fragment of mine_patterns.lp, expressed in Datalog.

Port of rules/mine_patterns.lp to pure-Python Datalog. Covers everything
EXCEPT choice rules and #minimize (those stay in Clingo).

This is the "Layer 1" from the Datalog experiment:
  Layer 1 (Datalog): computes all facts — incremental, explainable
  Layer 2 (Clingo):  makes decisions over those facts — choice/optimization
"""

from .engine import DatalogEngine, Rule


GLUE_TOOLS = {"llm_generate", "notify_user", "prompt_user"}


def build_rules(engine: DatalogEngine):
    """Register all monotonic pattern-mining rules."""

    # --- actionable_call(J, C, T) ---
    # call(J, C, T, "completed"), not glue_tool(T)  →  actionable_call(J, C, T)
    engine.add_rule(Rule(
        head_relation="actionable_call",
        head_vars=("J", "C", "T"),
        body=[
            ("call", ("J", "C", "T", "completed"), False),
            ("glue_tool", ("T",), True),
        ],
        constraints=[],
    ))
    engine.add_rule(Rule(
        head_relation="actionable_call",
        head_vars=("J", "C", "T"),
        body=[
            ("call", ("J", "C", "T", "failed"), False),
            ("glue_tool", ("T",), True),
        ],
        constraints=[],
    ))

    # --- tool_in_job(J, T) :- actionable_call(J, C, T) ---
    engine.add_rule(Rule(
        head_relation="tool_in_job",
        head_vars=("J", "T"),
        body=[("actionable_call", ("J", "C", "T"), False)],
        constraints=[],
    ))

    # --- known_tool(T) :- tool_in_job(_, T) ---
    engine.add_rule(Rule(
        head_relation="known_tool",
        head_vars=("T",),
        body=[("tool_in_job", ("J", "T"), False)],
        constraints=[],
    ))

    # --- job_count(N) — aggregate: count distinct jobs ---
    # Handled specially after solve (see compute_aggregates)

    # --- tool_job_count(T, N) — aggregate: count jobs per tool ---
    # Handled specially after solve

    # --- Transitive dependency ---
    # depends_transitive(J, A, B) :- depends(J, A, B).
    engine.add_rule(Rule(
        head_relation="depends_transitive",
        head_vars=("J", "A", "B"),
        body=[("depends", ("J", "A", "B"), False)],
        constraints=[],
    ))
    # depends_transitive(J, A, C) :- depends(J, A, B), depends_transitive(J, B, C).
    engine.add_rule(Rule(
        head_relation="depends_transitive",
        head_vars=("J", "A", "C"),
        body=[
            ("depends", ("J", "A", "B"), False),
            ("depends_transitive", ("J", "B", "C"), False),
        ],
        constraints=[],
    ))

    # --- precedes_in_job(J, T1, T2) ---
    # Via spawned_by parents with transitive dependency
    engine.add_rule(Rule(
        head_relation="precedes_in_job",
        head_vars=("J", "T1", "T2"),
        body=[
            ("actionable_call", ("J", "C1", "T1"), False),
            ("actionable_call", ("J", "C2", "T2"), False),
            ("spawned_by", ("J", "C1", "P1"), False),
            ("spawned_by", ("J", "C2", "P2"), False),
            ("depends_transitive", ("J", "P2", "P1"), False),
            ("core_tool", ("T1",), False),
            ("core_tool", ("T2",), False),
        ],
        constraints=[("!=", "P1", "P2"), ("!=", "T1", "T2")],
    ))
    # Via direct dependency: depends(J, P2, C1)
    engine.add_rule(Rule(
        head_relation="precedes_in_job",
        head_vars=("J", "T1", "T2"),
        body=[
            ("actionable_call", ("J", "C1", "T1"), False),
            ("actionable_call", ("J", "C2", "T2"), False),
            ("spawned_by", ("J", "C2", "P2"), False),
            ("depends", ("J", "P2", "C1"), False),
            ("core_tool", ("T1",), False),
            ("core_tool", ("T2",), False),
        ],
        constraints=[("!=", "T1", "T2")],
    ))

    # --- parallel_pair(T1, T2) ---
    engine.add_rule(Rule(
        head_relation="parallel_pair",
        head_vars=("T1", "T2"),
        body=[
            ("actionable_call", ("J", "C1", "T1"), False),
            ("actionable_call", ("J", "C2", "T2"), False),
            ("spawned_by", ("J", "C1", "P"), False),
            ("spawned_by", ("J", "C2", "P"), False),
            ("core_tool", ("T1",), False),
            ("core_tool", ("T2",), False),
        ],
        constraints=[("<", "T1", "T2")],
    ))

    # --- b_without_a(B, A) — B appears in a job without A ---
    engine.add_rule(Rule(
        head_relation="b_without_a",
        head_vars=("B", "A"),
        body=[
            ("tool_in_job", ("J", "B"), False),
            ("known_tool", ("A",), False),
            ("known_tool", ("B",), False),
            ("tool_in_job", ("J", "A"), True),
        ],
        constraints=[],
    ))

    # --- split_occurrence(A, B) ---
    engine.add_rule(Rule(
        head_relation="split_occurrence",
        head_vars=("A", "B"),
        body=[
            ("tool_in_job", ("J", "A"), False),
            ("known_tool", ("A",), False),
            ("known_tool", ("B",), False),
            ("tool_in_job", ("J", "B"), True),
        ],
        constraints=[("<", "A", "B")],
    ))
    engine.add_rule(Rule(
        head_relation="split_occurrence",
        head_vars=("A", "B"),
        body=[
            ("tool_in_job", ("J", "B"), False),
            ("known_tool", ("A",), False),
            ("known_tool", ("B",), False),
            ("tool_in_job", ("J", "A"), True),
        ],
        constraints=[("<", "A", "B")],
    ))

    # --- both_in_any_job(A, B) ---
    engine.add_rule(Rule(
        head_relation="both_in_any_job",
        head_vars=("A", "B"),
        body=[
            ("tool_in_job", ("J", "A"), False),
            ("tool_in_job", ("J", "B"), False),
        ],
        constraints=[("<", "A", "B")],
    ))

    # --- has_intermediate(A, B) ---
    # (computed post-aggregates since it depends on consistent_order which depends on precedes_in_job)

    # --- known_param(T, K, V) ---
    engine.add_rule(Rule(
        head_relation="known_param",
        head_vars=("T", "K", "V"),
        body=[
            ("actionable_call", ("J", "C", "T"), False),
            ("param", ("J", "C", "K", "V"), False),
            ("core_tool", ("T",), False),
        ],
        constraints=[],
    ))

    # --- has_stable_value(T, K) — computed post-aggregates ---

    # --- has_core_predecessor(T) — computed post-aggregates ---


def compute_aggregates(engine: DatalogEngine):
    """Compute aggregate-dependent relations that pure Datalog can't express.

    This is the seam where we do the counting and thresholding that ASP
    does with #count{} and arithmetic. In a production system this would
    be Soufflé's aggregate support or egglog's merge functions.
    """
    facts = engine.facts

    # --- job_count ---
    jobs = facts.get("job", set())
    job_count = len(jobs)
    facts["job_count"] = {(job_count,)}

    # --- tool_job_count(T, N) ---
    tool_jobs = {}
    for (j, t) in facts.get("tool_in_job", set()):
        tool_jobs.setdefault(t, set()).add(j)
    for t, js in tool_jobs.items():
        facts.setdefault("tool_job_count", set()).add((t, len(js)))

    # --- core_tool(T) — appears in >= 50% of runs ---
    for t, js in tool_jobs.items():
        if len(js) * 2 >= job_count:
            facts.setdefault("core_tool", set()).add((t,))

    # --- consistent_order / conflicting_order ---
    precedes = facts.get("precedes_in_job", set())
    # Collect which (T1, T2) pairs have evidence
    pair_evidence = {}
    for (j, t1, t2) in precedes:
        pair_evidence.setdefault((t1, t2), set()).add(j)

    for (t1, t2), evidence in pair_evidence.items():
        reverse = pair_evidence.get((t2, t1), set())
        if not reverse:
            facts.setdefault("consistent_order", set()).add((t1, t2))
        elif t1 < t2:
            facts.setdefault("conflicting_order", set()).add((t1, t2))
            facts.setdefault("order_evidence", set()).add((t1, t2, len(evidence)))
            facts.setdefault("order_evidence", set()).add((t2, t1, len(reverse)))

    # --- conditional_tool(B, A) ---
    tool_counts = {t: n for (t, n) in facts.get("tool_job_count", set())}
    for a in facts.get("known_tool", set()):
        a = a[0]
        na = tool_counts.get(a, 0)
        for b in facts.get("known_tool", set()):
            b = b[0]
            if a == b:
                continue
            nb = tool_counts.get(b, 0)
            if nb >= na or nb == 0:
                continue
            if nb * 10 < na:
                continue
            # Check: every run with B also has A
            if (b, a) not in facts.get("b_without_a", set()):
                facts.setdefault("conditional_tool", set()).add((b, a))
                facts.setdefault("conditional_rate", set()).add((b, a, nb, na))

    # --- co_occurs(A, B) ---
    for a in facts.get("known_tool", set()):
        a = a[0]
        for b in facts.get("known_tool", set()):
            b = b[0]
            if a >= b:
                continue
            na = tool_counts.get(a, 0)
            nb = tool_counts.get(b, 0)
            if na == nb and (a, b) not in facts.get("split_occurrence", set()):
                facts.setdefault("co_occurs", set()).add((a, b))

    # --- mutually_exclusive(A, B) ---
    for a in facts.get("known_tool", set()):
        a = a[0]
        for b in facts.get("known_tool", set()):
            b = b[0]
            if a >= b:
                continue
            has_a = bool(tool_jobs.get(a))
            has_b = bool(tool_jobs.get(b))
            if has_a and has_b and (a, b) not in facts.get("both_in_any_job", set()):
                facts.setdefault("mutually_exclusive", set()).add((a, b))

    # --- fusion_candidate(A, B) ---
    consistent = facts.get("consistent_order", set())
    split = facts.get("split_occurrence", set())
    core = {t for (t,) in facts.get("core_tool", set())}

    # has_intermediate(A, B): exists X s.t. A->X and X->B, X is core
    has_intermediate = set()
    for (a, x) in consistent:
        if x not in core:
            continue
        for (x2, b) in consistent:
            if x2 == x and a != x and x != b:
                has_intermediate.add((a, b))

    for (a, b) in consistent:
        if a in core and b in core:
            if (a, b) not in split and (a, b) not in has_intermediate:
                facts.setdefault("fusion_candidate", set()).add((a, b))

    # --- stable_param(T, K, V) ---
    param_counts = {}
    for (t, k, v) in facts.get("known_param", set()):
        param_counts.setdefault((t, k, v), set())
    for (j, c, t) in facts.get("actionable_call", set()):
        for (j2, c2, k, v) in facts.get("param", set()):
            if j2 == j and c2 == c and (t,) in facts.get("core_tool", set()):
                param_counts.setdefault((t, k, v), set()).add(j)

    for (t, k, v), js in param_counts.items():
        tc = tool_counts.get(t, 0)
        if tc > 0 and len(js) * 2 >= tc:
            facts.setdefault("stable_param", set()).add((t, k, v))
            facts.setdefault("has_stable_value", set()).add((t, k))

    # --- variable_param(T, K) ---
    for (t, k, v) in facts.get("known_param", set()):
        if (t, k) not in facts.get("has_stable_value", set()):
            facts.setdefault("variable_param", set()).add((t, k))

    # --- phase_zero(T) ---
    for (t,) in facts.get("core_tool", set()):
        has_pred = False
        for (t2, t3) in consistent:
            if t3 == t and (t2,) in facts.get("core_tool", set()):
                has_pred = True
                break
        # Also check chosen_order (but that's Layer 2 — skip for now)
        if not has_pred:
            facts.setdefault("phase_zero", set()).add((t,))
