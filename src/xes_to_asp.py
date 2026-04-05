#!/usr/bin/env python3
"""
xes_to_asp.py — Convert XES event logs to autocompile ASP traces.

XES (eXtensible Event Stream) is the IEEE standard for process mining.
This converter handles any XES file — healthcare, manufacturing, etc.

Usage:
    python src/xes_to_asp.py --input data-private/sepsis_cases.xes --output examples/medical-ed/traces.lp
    python src/xes_to_asp.py --input data.xes.gz --max-cases 500 --output traces.lp
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

try:
    import pm4py
    import pandas as pd
except ImportError:
    sys.exit("ERROR: pm4py required. Run: uv add pm4py")


def load_xes(path: Path, max_cases: int | None = None) -> pd.DataFrame:
    """Load XES file into a DataFrame."""
    print(f"Loading {path}...")
    log = pm4py.read_xes(str(path))
    df = pm4py.convert_to_dataframe(log)

    case_col = "case:concept:name"
    activity_col = "concept:name"
    time_col = "time:timestamp"

    print(f"  {len(df)} events, {df[case_col].nunique()} cases")

    if max_cases:
        case_ids = df[case_col].unique()[:max_cases]
        df = df[df[case_col].isin(case_ids)]
        print(f"  Limited to {max_cases} cases: {len(df)} events")

    # Sort by case then timestamp
    df = df.sort_values([case_col, time_col]).reset_index(drop=True)
    return df


def clean_asp(s: str) -> str:
    """Clean string for ASP fact embedding."""
    if pd.isna(s):
        return ""
    s = str(s).replace('"', '\\"').replace("\n", " ").replace("\r", "")
    if len(s) > 100:
        s = s[:97] + "..."
    return s


def tool_name(activity: str) -> str:
    """Convert activity name to a clean tool identifier."""
    return activity.strip().lower().replace(" ", "_").replace("-", "_")


def df_to_asp(df: pd.DataFrame) -> list[str]:
    """Convert a DataFrame of events to ASP facts."""
    case_col = "case:concept:name"
    activity_col = "concept:name"
    time_col = "time:timestamp"
    resource_col = "org:group"

    # Identify parameter columns (non-standard columns with actual data)
    standard_cols = {case_col, activity_col, time_col, resource_col,
                     "lifecycle:transition", "@@index"}
    param_cols = []
    for col in df.columns:
        if col in standard_cols or col.startswith("case:"):
            continue
        # Only include columns that have meaningful data (not all NaN)
        if df[col].notna().any():
            param_cols.append(col)

    # Identify case-level attributes (columns starting with "case:" or that
    # have the same value for all events in a case)
    case_attr_cols = [c for c in df.columns if c.startswith("case:") and c != case_col]
    # Also include boolean/flag columns that are case-level (same value per case)
    for col in df.columns:
        if col in standard_cols or col in case_attr_cols or col.startswith("case:"):
            continue
        # Check if column has same value for all events in each case
        grouped = df.groupby(case_col)[col].nunique()
        if (grouped <= 1).all() and df[col].notna().any():
            case_attr_cols.append(col)

    # Event-level param columns (vary within a case)
    event_param_cols = [c for c in param_cols if c not in case_attr_cols]

    lines = []
    cases = df.groupby(case_col)
    total_cases = len(cases)

    lines.append(f"% === Clinical Event Log Traces ({total_cases} cases) ===")
    lines.append(f"% Activities: {', '.join(sorted(df[activity_col].unique()))}")
    lines.append(f"% Event-level params: {', '.join(event_param_cols)}")
    lines.append(f"% Case-level attributes: {', '.join(case_attr_cols)}")
    lines.append("")

    for case_idx, (case_id, group) in enumerate(cases):
        run_id = f"run_{case_idx + 1}"

        # Case metadata
        lines.append(f'% === Case {run_id} ({len(group)} events) ===')
        lines.append(f'job("{run_id}").')

        # Case-level attributes as job metadata
        first_row = group.iloc[0]
        for col in case_attr_cols:
            val = first_row.get(col)
            if pd.notna(val) and str(val).strip():
                val_clean = clean_asp(str(val))
                col_clean = col.replace("case:", "").replace(":", "_")
                lines.append(f'param("{run_id}", "{run_id}", "{col_clean}", "{val_clean}").')

        # Events
        prev_step = None
        first_step = f"{run_id}_step_1"
        triage_step = None

        for event_idx, (_, row) in enumerate(group.iterrows()):
            step_id = f"{run_id}_step_{event_idx + 1}"
            tool = tool_name(row[activity_col])

            lines.append(f'call("{run_id}", "{step_id}", "{tool}", "completed").')

            # Sequential dependency
            if prev_step:
                lines.append(f'depends("{run_id}", "{step_id}", "{prev_step}").')

            # Spawned_by: first event is root, subsequent spawned by first
            # (or by triage if it exists)
            if event_idx == 0:
                pass  # root
            else:
                parent = triage_step or first_step
                lines.append(f'spawned_by("{run_id}", "{step_id}", "{parent}").')

            # Track triage step for spawned_by
            if "triage" in tool:
                triage_step = step_id

            # Resource as parameter
            if resource_col in row and pd.notna(row[resource_col]):
                lines.append(f'param("{run_id}", "{step_id}", "department", "{clean_asp(row[resource_col])}").')

            # Event-level parameters (non-null values)
            for col in event_param_cols:
                val = row.get(col)
                if pd.notna(val) and str(val).strip():
                    col_clean = col.replace(":", "_").replace(" ", "_")
                    val_clean = clean_asp(str(val))
                    if val_clean:
                        lines.append(f'param("{run_id}", "{step_id}", "{col_clean}", "{val_clean}").')

            prev_step = step_id

        lines.append("")

    return lines


def main():
    parser = argparse.ArgumentParser(description="Convert XES event logs to autocompile traces")
    parser.add_argument("--input", required=True, help="Path to .xes or .xes.gz file")
    parser.add_argument("--max-cases", type=int, help="Maximum number of cases")
    parser.add_argument("--output", default="traces.lp", help="Output .lp file")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    input_path = Path(args.input) if Path(args.input).is_absolute() else project_root / args.input
    output_path = Path(args.output) if Path(args.output).is_absolute() else project_root / args.output

    if not input_path.exists():
        sys.exit(f"ERROR: {input_path} not found")

    df = load_xes(input_path, args.max_cases)

    lines = df_to_asp(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))

    # Stats
    case_col = "case:concept:name"
    activity_col = "concept:name"
    n_cases = df[case_col].nunique()
    n_events = len(df)
    print(f"\nWritten {n_cases} cases ({n_events} events) to {output_path}")
    print(f"\nActivity distribution:")
    for act, count in df[activity_col].value_counts().items():
        pct = count * 100 // n_events
        print(f"  {act}: {count} ({pct}%)")


if __name__ == "__main__":
    main()
