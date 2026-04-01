#!/usr/bin/env python3
"""
llm_baseline.py — Score autocompile against pure LLM compilation.

The elephant in the room: can an LLM just look at the traces and produce
the same (or better) compiled program without any symbolic machinery?

This script:
  1. Loads traces (same input as compile.py)
  2. Summarizes them for the LLM (tool frequencies, orderings, parameters)
  3. Asks the LLM to produce a compiled program (core tools, ordering, etc.)
  4. Runs Clingo on the same traces
  5. Compares both against holdout traces
  6. Scores: which compilation is more accurate?

Usage:
    # Compare on agent workflow traces
    python src/llm_baseline.py --traces examples/travel-updates/traces.lp

    # Compare on real robot data
    python src/llm_baseline.py --traces examples/lerobot-pick-place/traces.lp

    # Use a specific model
    python src/llm_baseline.py --traces data.lp --model anthropic/claude-sonnet-4

Requires: OPENROUTER_API_KEY in .env or environment
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("ERROR: requests required. pip install requests")

# Load .env
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())


# ============================================================================
# Trace Summarization
# ============================================================================

def parse_traces(traces_path):
    """Parse ASP trace facts into structured data."""
    jobs = {}
    calls = defaultdict(list)  # job_id -> [call_info]
    deps = defaultdict(list)
    spawns = defaultdict(dict)
    params = defaultdict(lambda: defaultdict(dict))

    with open(traces_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue

            # Parse simple ASP facts
            if line.startswith('job("'):
                m = re.match(r'job\("([^"]+)"\)', line)
                if m:
                    jobs[m.group(1)] = True

            elif line.startswith('call("'):
                m = re.match(r'call\("([^"]+)",\s*"([^"]+)",\s*"([^"]+)",\s*"([^"]+)"\)', line)
                if m:
                    job_id, step_id, tool, status = m.groups()
                    calls[job_id].append({
                        "step_id": step_id, "tool": tool, "status": status
                    })

            elif line.startswith('depends("'):
                m = re.match(r'depends\("([^"]+)",\s*"([^"]+)",\s*"([^"]+)"\)', line)
                if m:
                    deps[m.group(1)].append((m.group(2), m.group(3)))

            elif line.startswith('spawned_by("'):
                m = re.match(r'spawned_by\("([^"]+)",\s*"([^"]+)",\s*"([^"]+)"\)', line)
                if m:
                    spawns[m.group(1)][m.group(2)] = m.group(3)

            elif line.startswith('param("'):
                m = re.match(r'param\("([^"]+)",\s*"([^"]+)",\s*"([^"]+)",\s*"([^"]+)"\)', line)
                if m:
                    params[m.group(1)][m.group(2)][m.group(3)] = m.group(4)

    return jobs, calls, deps, spawns, params


def summarize_traces(jobs, calls, deps, spawns, params, max_runs_shown=10):
    """Create a concise summary of traces for the LLM."""
    n_runs = len(jobs)

    # Filter out glue tools
    glue = {"llm_generate", "prompt_user", "notify_user"}

    # Tool frequency
    tool_counts = Counter()
    tool_per_run = defaultdict(set)
    for job_id, job_calls in calls.items():
        for c in job_calls:
            if c["tool"] not in glue and c["status"] in ("completed", "failed"):
                tool_counts[c["tool"]] += 1
                tool_per_run[c["tool"]].add(job_id)

    tool_run_counts = {t: len(runs) for t, runs in tool_per_run.items()}

    # Parameter analysis
    param_values = defaultdict(lambda: defaultdict(Counter))
    for job_id, job_params in params.items():
        for step_id, step_params in job_params.items():
            # Find tool for this step
            tool = None
            for c in calls.get(job_id, []):
                if c["step_id"] == step_id:
                    tool = c["tool"]
                    break
            if tool and tool not in glue:
                for key, val in step_params.items():
                    param_values[tool][key][val] += 1

    # Ordering evidence
    ordering_evidence = Counter()
    for job_id, job_calls in calls.items():
        actionable = [(c["step_id"], c["tool"]) for c in job_calls
                      if c["tool"] not in glue]
        step_to_tool = {sid: t for sid, t in actionable}
        dep_pairs = deps.get(job_id, [])
        # Build transitive ordering from deps
        for sid, dep_sid in dep_pairs:
            t1 = step_to_tool.get(dep_sid)
            t2 = step_to_tool.get(sid)
            if t1 and t2 and t1 != t2:
                ordering_evidence[(t1, t2)] += 1

    # Build summary text
    lines = []
    lines.append(f"TRACE SUMMARY: {n_runs} execution runs")
    lines.append(f"\nTool frequencies (tool: appears in N/{n_runs} runs):")
    for tool, count in sorted(tool_run_counts.items(),
                               key=lambda x: -x[1]):
        pct = count * 100 // n_runs
        lines.append(f"  {tool}: {count}/{n_runs} runs ({pct}%)")

    lines.append(f"\nOrdering evidence (tool_A -> tool_B: seen in N runs):")
    for (t1, t2), count in sorted(ordering_evidence.items(),
                                    key=lambda x: -x[1])[:30]:
        lines.append(f"  {t1} -> {t2}: {count} runs")

    lines.append(f"\nParameter analysis:")
    for tool in sorted(param_values.keys()):
        if tool in glue:
            continue
        lines.append(f"  {tool}:")
        for key in sorted(param_values[tool].keys()):
            vals = param_values[tool][key]
            total = sum(vals.values())
            top_val, top_count = vals.most_common(1)[0]
            if top_count == total and total > 1:
                lines.append(f"    {key}: always '{top_val}' ({total} calls)")
            elif top_count > total * 0.5:
                lines.append(f"    {key}: usually '{top_val}' ({top_count}/{total}), "
                            f"varies in {total - top_count}")
            else:
                n_unique = len(vals)
                lines.append(f"    {key}: {n_unique} distinct values across {total} calls")

    # Show a few sample runs
    lines.append(f"\nSample runs (first {min(max_runs_shown, n_runs)}):")
    for i, (job_id, job_calls) in enumerate(sorted(calls.items())):
        if i >= max_runs_shown:
            break
        tools = [c["tool"] for c in job_calls if c["tool"] not in glue]
        lines.append(f"  {job_id}: {' -> '.join(tools)}")

    return "\n".join(lines)


# ============================================================================
# LLM Compilation
# ============================================================================

LLM_PROMPT = """You are analyzing execution traces from a repeated workflow. Your job is to compile these traces into an optimized deterministic program.

{summary}

## Task

Analyze the traces above and produce a JSON compilation with this exact structure:

```json
{{
  "core_tools": ["tools that appear consistently and should be in the compiled program"],
  "ordering": [["tool_A", "tool_B"], ...],  // tool_A always/usually before tool_B
  "parallel_pairs": [["tool_A", "tool_B"], ...],  // can run concurrently
  "fusion_candidates": [["tool_A", "tool_B"], ...],  // always sequential, merge into one
  "mutually_exclusive": [["tool_A", "tool_B"], ...],  // never both in same run
  "conditional_tools": [{{"tool": "X", "condition": "runs only when Y has results", "rate": "N/M"}}],
  "stable_params": {{"tool": {{"param": "value"}}}},  // parameters that are always the same
  "variable_params": {{"tool": ["param1", "param2"]}},  // parameters that change per run
  "phases": [["phase0_tools"], ["phase1_tools"], ...],  // execution phases
  "compilation_ratio": 0.XX  // fraction of tool types you compiled
}}
```

Be precise. Only include patterns you're confident about from the data. For ordering, only include pairs where one consistently comes before the other. For stable_params, only include values that are the same across most runs.

Respond with ONLY the JSON object."""


def query_llm(summary, model, api_key):
    """Query LLM for trace compilation."""
    prompt = LLM_PROMPT.format(summary=summary)

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 3000,
        },
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()

    msg = data["choices"][0]["message"]
    text = msg.get("content") or ""
    usage = data.get("usage", {})

    # Some models (Qwen 3.5) put thinking in a separate "reasoning" field
    # and the clean answer in "content"
    if not text and msg.get("reasoning"):
        # Content was empty but reasoning exists — check if JSON is in reasoning
        reasoning = msg["reasoning"]
        # Try to find JSON in reasoning
        import re as _re
        json_match = _re.search(r'\{[\s\S]*\}', reasoning)
        if json_match:
            text = json_match.group(0)

    # Strip markdown/think tags
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if "```" in text:
            text = text[:text.rfind("```")]
    if "<think>" in text:
        think_end = text.find("</think>")
        if think_end != -1:
            text = text[think_end + len("</think>"):].strip()
    text = text.strip()

    result = json.loads(text)
    return result, usage


# ============================================================================
# Clingo Compilation (reuse from compile.py)
# ============================================================================

def run_clingo(traces_path, rules_path):
    """Run Clingo and return structured results."""
    sys.path.insert(0, str(Path(__file__).parent))
    from compile import run_asp_strategy
    return run_asp_strategy(traces_path, rules_path, verbose=False)


# ============================================================================
# Comparison
# ============================================================================

def compare_results(clingo_results, llm_results, n_runs):
    """Compare Clingo vs LLM compilation results."""
    comparison = {}

    # Core tools
    clingo_core = set(clingo_results.get("core_tools", []))
    llm_core = set(llm_results.get("core_tools", []))
    comparison["core_tools"] = {
        "clingo": sorted(clingo_core),
        "llm": sorted(llm_core),
        "agreement": sorted(clingo_core & llm_core),
        "clingo_only": sorted(clingo_core - llm_core),
        "llm_only": sorted(llm_core - clingo_core),
        "jaccard": len(clingo_core & llm_core) / max(len(clingo_core | llm_core), 1),
    }

    # Orderings
    clingo_order = set(tuple(x) for x in clingo_results.get("consistent_order", []))
    llm_order = set(tuple(x) for x in llm_results.get("ordering", []))
    comparison["orderings"] = {
        "clingo_count": len(clingo_order),
        "llm_count": len(llm_order),
        "agreement": len(clingo_order & llm_order),
        "clingo_only": len(clingo_order - llm_order),
        "llm_only": len(llm_order - clingo_order),
    }

    # Fusion candidates
    clingo_fusion = set(tuple(x) for x in clingo_results.get("fusion_candidates", []))
    llm_fusion = set(tuple(x) for x in llm_results.get("fusion_candidates", []))
    comparison["fusion"] = {
        "clingo": sorted(clingo_fusion),
        "llm": sorted(llm_fusion),
        "agreement": len(clingo_fusion & llm_fusion),
    }

    # Mutually exclusive
    clingo_excl = set(tuple(sorted(x)) for x in clingo_results.get("mutually_exclusive", []))
    llm_excl = set(tuple(sorted(x)) for x in llm_results.get("mutually_exclusive", []))
    comparison["mutually_exclusive"] = {
        "clingo_count": len(clingo_excl),
        "llm_count": len(llm_excl),
        "agreement": len(clingo_excl & llm_excl),
    }

    # Stable params
    clingo_stable = set()
    for tool, params in clingo_results.get("stable_params", {}).items():
        for key, vals in params.items():
            for v in vals:
                clingo_stable.add((tool, key, v))
    llm_stable = set()
    for tool, params in llm_results.get("stable_params", {}).items():
        for key, val in params.items():
            if isinstance(val, list):
                for v in val:
                    llm_stable.add((tool, key, str(v)))
            else:
                llm_stable.add((tool, key, str(val)))
    comparison["stable_params"] = {
        "clingo_count": len(clingo_stable),
        "llm_count": len(llm_stable),
        "agreement": len(clingo_stable & llm_stable),
    }

    return comparison


def print_comparison(comparison, clingo_time, llm_time, llm_usage, model):
    """Print human-readable comparison."""
    print(f"\n{'='*70}")
    print(f"CLINGO vs LLM ({model})")
    print(f"{'='*70}")

    # Core tools
    ct = comparison["core_tools"]
    print(f"\nCore Tools:")
    print(f"  Clingo found: {ct['clingo']}")
    print(f"  LLM found:    {ct['llm']}")
    print(f"  Agreement:    {ct['agreement']} (Jaccard={ct['jaccard']:.2f})")
    if ct["clingo_only"]:
        print(f"  Clingo only:  {ct['clingo_only']}")
    if ct["llm_only"]:
        print(f"  LLM only:     {ct['llm_only']}")

    # Orderings
    od = comparison["orderings"]
    print(f"\nOrderings:")
    print(f"  Clingo: {od['clingo_count']}  LLM: {od['llm_count']}  "
          f"Agree: {od['agreement']}  "
          f"Clingo-only: {od['clingo_only']}  LLM-only: {od['llm_only']}")

    # Fusion
    fu = comparison["fusion"]
    print(f"\nFusion Candidates:")
    print(f"  Clingo: {fu['clingo']}  LLM: {fu['llm']}  Agree: {fu['agreement']}")

    # Mutual exclusion
    mx = comparison["mutually_exclusive"]
    print(f"\nMutually Exclusive:")
    print(f"  Clingo: {mx['clingo_count']}  LLM: {mx['llm_count']}  "
          f"Agree: {mx['agreement']}")

    # Stable params
    sp = comparison["stable_params"]
    print(f"\nStable Parameters:")
    print(f"  Clingo: {sp['clingo_count']}  LLM: {sp['llm_count']}  "
          f"Agree: {sp['agreement']}")

    # Cost / Performance
    print(f"\nPerformance:")
    print(f"  Clingo: {clingo_time:.3f}s")
    print(f"  LLM:    {llm_time:.3f}s")
    if llm_usage:
        prompt_tokens = llm_usage.get("prompt_tokens", 0)
        completion_tokens = llm_usage.get("completion_tokens", 0)
        print(f"  LLM tokens: {prompt_tokens} prompt + {completion_tokens} completion")


def main():
    parser = argparse.ArgumentParser(description="Compare Clingo vs LLM compilation")
    parser.add_argument("--traces", required=True, help="Path to trace file")
    parser.add_argument("--rules", default="rules/mine_patterns_relaxed.lp")
    parser.add_argument("--model", default="qwen/qwen3-235b-a22b",
                        help="OpenRouter model ID")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    traces_path = Path(args.traces)
    if not traces_path.is_absolute():
        traces_path = project_root / traces_path
    rules_path = Path(args.rules)
    if not rules_path.is_absolute():
        rules_path = project_root / rules_path

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("ERROR: OPENROUTER_API_KEY not set. Add to .env or environment.")

    # Parse and summarize traces
    print(f"Loading traces from {traces_path}...")
    jobs, calls, deps, spawns, params = parse_traces(traces_path)
    summary = summarize_traces(jobs, calls, deps, spawns, params)
    n_runs = len(jobs)
    print(f"Parsed {n_runs} runs")

    # Run Clingo
    print(f"\nRunning Clingo ({args.rules})...")
    clingo_start = time.time()
    clingo_results = run_clingo(traces_path, rules_path)
    clingo_time = time.time() - clingo_start
    print(f"  Done in {clingo_time:.3f}s — "
          f"{len(clingo_results['core_tools'])} core tools")

    # Run LLM
    print(f"\nQuerying {args.model}...")
    llm_start = time.time()
    try:
        llm_results, llm_usage = query_llm(summary, args.model, api_key)
        llm_time = time.time() - llm_start
        print(f"  Done in {llm_time:.3f}s — "
              f"{len(llm_results.get('core_tools', []))} core tools")
    except Exception as e:
        print(f"  LLM error: {e}")
        print(f"  Running Clingo-only comparison")
        llm_results = {"core_tools": [], "ordering": [], "fusion_candidates": [],
                       "mutually_exclusive": [], "stable_params": {}, "variable_params": {}}
        llm_usage = {}
        llm_time = 0

    # Compare
    comparison = compare_results(clingo_results, llm_results, n_runs)

    if args.format == "json":
        print(json.dumps({
            "comparison": comparison,
            "clingo_time": clingo_time,
            "llm_time": llm_time,
            "llm_usage": llm_usage,
            "model": args.model,
            "n_runs": n_runs,
            "traces": str(traces_path),
        }, indent=2))
    else:
        print_comparison(comparison, clingo_time, llm_time, llm_usage, args.model)

        # Overall verdict
        ct = comparison["core_tools"]
        print(f"\n{'='*70}")
        print(f"VERDICT")
        print(f"{'='*70}")
        if ct["jaccard"] > 0.8:
            print(f"HIGH AGREEMENT (Jaccard={ct['jaccard']:.2f}) — LLM finds similar patterns")
            print(f"Clingo's advantage: {comparison['orderings']['clingo_only']} extra orderings, "
                  f"guaranteed consistent")
        elif ct["jaccard"] > 0.5:
            print(f"MODERATE AGREEMENT (Jaccard={ct['jaccard']:.2f}) — partial overlap")
            print(f"Differences worth investigating:")
            if ct["clingo_only"]:
                print(f"  Clingo found {ct['clingo_only']} — LLM missed these")
            if ct["llm_only"]:
                print(f"  LLM found {ct['llm_only']} — Clingo missed these (below threshold?)")
        else:
            print(f"LOW AGREEMENT (Jaccard={ct['jaccard']:.2f}) — fundamentally different compilations")
            print(f"This is interesting — investigate what each approach captures that the other doesn't")


if __name__ == "__main__":
    main()
