#!/usr/bin/env python3
"""
codegen.py — Generate an executable job spec from a compiled workflow.

Takes the output of compile.py and emits a runnable program. The output
format depends on the target runtime. Currently supports:
  - daslab: Daslab direct DAG job spec (JSON)
  - pseudo: Human-readable pseudocode (for inspection)

Usage:
    python codegen.py --compiled result.json --target daslab --output job.json
    python codegen.py --compiled result.json --target pseudo
"""

import sys
import json
import argparse
from pathlib import Path


# API field name mapping: compiled format uses short names, APIs use full names
FIELD_MAP = {
    "gmail_search": {"account": "accountId", "query": "query", "max_results": "maxResults"},
    "gmail_list_threads": {"account": "accountId", "labels": "labelIds"},
    "gmail_get_thread": {"account": "accountId", "thread_id": "threadId"},
    "gmail_get_attachment": {"account": "accountId"},
    "sheets_read": {"account": "accountId", "spreadsheet": "spreadsheetId", "sheet_name": "sheetName", "max_rows": "maxRows"},
    "sheets_get_spreadsheet": {"account": "accountId", "spreadsheet": "spreadsheetId"},
    "sheets_get_values": {"account": "accountId", "spreadsheet": "spreadsheetId", "range": "range"},
    "sheets_update_values": {"account": "accountId", "spreadsheet": "spreadsheetId", "range": "range"},
    "tracking_get_status": {"account": "accountId", "tracking_number": "trackingNumber"},
    "tracking_get_list": {"account": "accountId"},
    "tracking_register": {"account": "accountId"},
}

# Type coercions for specific API fields
TYPE_COERCE = {
    "maxResults": int,
    "maxRows": int,
    "labelIds": lambda v: v.split(",") if isinstance(v, str) else v,
}


def main():
    parser = argparse.ArgumentParser(description="Generate executable code from compiled workflow")
    parser.add_argument("--compiled", required=True, help="Path to compiled JSON (output of compile.py)")
    parser.add_argument("--target", default="daslab", choices=["daslab", "pseudo"], help="Output target")
    parser.add_argument("--output", help="Output file (default: stdout for pseudo, job.json for daslab)")
    parser.add_argument("--title", default=None, help="Job title (daslab target)")
    parser.add_argument("--scene-id", default=None, help="Scene ID (daslab target)")
    parser.add_argument("--schedule", default=None, help="Cron expression for scheduled execution")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    compiled_path = Path(args.compiled) if Path(args.compiled).is_absolute() else project_root / args.compiled
    compiled = json.loads(compiled_path.read_text())

    if args.target == "daslab":
        result = generate_daslab(compiled, args)
        output = json.dumps(result, indent=2)
    elif args.target == "pseudo":
        output = generate_pseudo(compiled)

    if args.output:
        out_path = Path(args.output) if Path(args.output).is_absolute() else project_root / args.output
        out_path.write_text(output + "\n")
        print(f"Written to {out_path}")
    else:
        print(output)


def map_input(tool, input_params):
    """Map compiled parameter names to API field names with type coercion."""
    mapping = FIELD_MAP.get(tool, {})
    mapped = {}
    for key, val in input_params.items():
        # Skip analysis-only fields
        if key in ("value_type", "value_constant", "target_sheet", "target_col_start",
                    "target_col_end", "target_row_start", "target_row_end",
                    "target_col_count", "target_row_count", "value_row_count", "value_col_count"):
            continue
        api_key = mapping.get(key, key)
        if api_key in TYPE_COERCE:
            val = TYPE_COERCE[api_key](val)
        mapped[api_key] = val
    return mapped


def expand_fanout(call):
    """Expand a call with array parameters into parallel calls."""
    input_params = call.get("input", {})
    array_keys = [(k, v) for k, v in input_params.items() if isinstance(v, list)]

    if not array_keys:
        return [call]

    # Fan out on the first array key
    array_key, array_vals = array_keys[0]
    base_input = {k: v for k, v in input_params.items() if k != array_key}
    expanded = []

    for i, val in enumerate(array_vals):
        expanded.append({
            **call,
            "id": f"{call['id']}_{i}",
            "input": {**base_input, array_key: val},
        })

    return expanded


def generate_daslab(compiled, args):
    """Generate a Daslab direct DAG job spec."""
    analysis = compiled.get("_analysis", {})
    meta = compiled.get("_autocompile", {})
    conditionals = {c["tool"]: c for c in analysis.get("conditionals", [])}
    variable_params = analysis.get("variable_params", {})

    calls = []
    expanded_id_map = {}  # original_id -> [expanded_ids]

    for call in compiled.get("calls", []):
        tool = call["tool"]
        compilation = call.get("compilation", "compiled")

        # Expand fan-out calls
        expanded = expand_fanout(call)

        for exp_call in expanded:
            mapped_input = map_input(tool, exp_call.get("input", {}))

            job_call = {
                "id": exp_call["id"],
                "tool": tool,
                "input": mapped_input,
            }

            # Add waits_for, resolving any fan-out references
            waits_for = exp_call.get("waits_for", [])
            resolved_waits = []
            for dep in waits_for:
                if dep in expanded_id_map:
                    resolved_waits.extend(expanded_id_map[dep])
                else:
                    resolved_waits.append(dep)
            if resolved_waits:
                job_call["waits_for"] = sorted(set(resolved_waits))

            # Mark variable params as runtime bindings
            tool_vars = variable_params.get(tool, [])
            if tool_vars:
                job_call["_runtime_params"] = tool_vars
                job_call["_note"] = f"Params {tool_vars} are data-dependent — bind at runtime from previous step outputs"

            # Mark conditional execution
            if tool in conditionals:
                cond = conditionals[tool]
                job_call["_conditional"] = f"Runs when {cond['depends_on']} produces results ({cond['rate']})"

            calls.append(job_call)

        # Track expanded IDs for dependency resolution
        if len(expanded) > 1:
            expanded_id_map[call["id"]] = [e["id"] for e in expanded]

    job = {
        "_generated_by": "autocompile codegen",
        "_source": {
            "strategy": meta.get("strategy", "asp"),
            "source_runs": meta.get("source_runs", 0),
            "compilation_ratio": compiled.get("_boundary", {}).get("compilation_ratio", 0),
        },
        "title": args.title or f"Compiled Workflow ({meta.get('source_runs', '?')} runs analyzed)",
        "calls": calls,
    }

    if args.scene_id:
        job["scene_id"] = args.scene_id

    if args.schedule:
        job["schedule"] = {
            "trigger": "cron",
            "cron": args.schedule,
            "timezone": "UTC",
            "enabled": True,
        }

    return job


def generate_pseudo(compiled):
    """Generate human-readable pseudocode showing the compiled program."""
    meta = compiled.get("_autocompile", {})
    analysis = compiled.get("_analysis", {})
    boundary = compiled.get("_boundary", {})
    phases = meta.get("phases", {})
    conditionals = {c["tool"]: c for c in analysis.get("conditionals", [])}
    variable_params = analysis.get("variable_params", {})
    fusions = analysis.get("fusion_candidates", [])

    lines = [
        f"# Compiled workflow — synthesized from {meta.get('source_runs', '?')} observed runs",
        f"# Compilation ratio: {boundary.get('compilation_ratio', '?')} "
        f"({boundary.get('core_tool_types', '?')}/{boundary.get('total_observed_tool_types', '?')} tool types compiled)",
        f"# Strategy: {meta.get('strategy', 'unknown')}",
        "",
    ]

    for phase_num in sorted(phases.keys()):
        tools = phases[phase_num]
        lines.append(f"# --- Phase {phase_num} ---")

        # Check if tools in this phase are parallel
        if len(tools) > 1:
            lines.append(f"parallel {{")
            indent = "    "
        else:
            indent = ""

        for tool in tools:
            # Find the call spec
            call = next((c for c in compiled.get("calls", []) if c["tool"] == tool), None)
            if not call:
                continue

            tool_vars = variable_params.get(tool, [])
            params = call.get("input", {})

            # Build parameter string
            param_parts = []
            for k, v in sorted(params.items()):
                if k in tool_vars:
                    param_parts.append(f"{k}=<runtime>")
                elif isinstance(v, list):
                    param_parts.append(f"{k}=[{', '.join(repr(x) for x in v)}]  # fan-out: runs once per value")
                else:
                    param_parts.append(f"{k}={repr(v)}")

            param_str = ", ".join(param_parts)

            # Conditional?
            cond = conditionals.get(tool)
            if cond:
                lines.append(f"{indent}if {cond['depends_on']}.has_results:  # {cond['rate']} of runs")
                indent_inner = indent + "    "
            else:
                indent_inner = indent

            lines.append(f"{indent_inner}{tool}({param_str})")

            if cond:
                lines.append(f"{indent}# else: skip (no results from {cond['depends_on']})")

            lines.append("")

        if len(tools) > 1:
            lines.append(f"}}")
            lines.append("")

    # Show what remains as LLM calls
    lines.append("# --- Steps requiring LLM ---")
    lines.append("# The following steps were not compilable:")
    lines.append("# They appear inconsistently or require LLM reasoning.")
    lines.append("# In execution, these are handled by llm_invoke().")

    # Fusion opportunities
    if fusions:
        lines.append("")
        lines.append("# --- Fusion opportunities ---")
        for a, b in fusions:
            lines.append(f"# {a} + {b} can be merged into a single operation")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
