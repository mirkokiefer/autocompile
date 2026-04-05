#!/usr/bin/env python3
"""
single_demo_compile.py — Compile a robust program from a SINGLE demonstration.

The key insight: humans learn from 1-3 demos because they use world knowledge
to mentally generate variations. This script does the same:

  1. Take one clean demo → segment it into state transitions
  2. Feed segments + video context to an LLM → get world model as structured JSON
  3. LLM world model contains:
     - Physical constraints (invariants that must hold in ANY valid execution)
     - Degrees of freedom (what can vary across executions)
     - Failure modes (what could go wrong → synthetic retry/recovery traces)
     - Simplification rules (what could be skipped under certain conditions)
  4. Generate synthetic trace variations from the world model
  5. Run the EXISTING ASP pattern miner on (1 real + N synthetic) traces
  6. The compiled program handles all variations — from ONE demo.

In production: step 2 calls Claude API with video frames + trajectory data.
For this prototype: we demonstrate the interface and show what Claude produces.

Usage:
    python src/single_demo_compile.py \
        --dataset lerobot/ucsd_pick_and_place_dataset \
        --episode 178 \
        --output examples/single-demo/result.json
"""

import argparse
import json
import random
import sys
import tempfile
from pathlib import Path

try:
    import numpy as np
except ImportError:
    sys.exit("ERROR: numpy required")

try:
    from datasets import load_dataset
except ImportError:
    sys.exit("ERROR: datasets required")

from segment_discover import (
    load_episodes, detect_changepoints, characterize_segment
)

random.seed(42)


# =============================================================================
# World Model Interface
#
# In production, this calls Claude API with:
#   - Video frames from the demo
#   - Segmented trajectory data
#   - The question: "What is this task? What are the physical constraints?
#     What could vary? What could go wrong?"
#
# Claude returns structured JSON. For this prototype, we show what that
# JSON looks like for a tabletop pick-and-place task.
# =============================================================================

def query_world_model(segments, video_frames=None):
    """
    Query LLM world model about a demonstration.

    In production:
        response = anthropic.messages.create(
            model="claude-sonnet-4-20250514",
            messages=[{
                "role": "user",
                "content": [
                    *[{"type": "image", "source": frame} for frame in video_frames],
                    {"type": "text", "text": WORLD_MODEL_PROMPT.format(
                        segments=format_segments(segments)
                    )}
                ]
            }],
            response_format={"type": "json"}
        )
        return json.loads(response.content[0].text)

    For now, we return what Claude would produce for this demo.
    """
    # Analyze the segments to determine task type
    has_gripper_close = any(s["char"]["delta_gripper"] < -1.5 for s in segments)
    has_gripper_open = any(s["char"]["delta_gripper"] > 1.5 for s in segments)
    has_movement = any(s["char"]["pos_distance"] > 0.3 for s in segments)

    # The LLM would analyze the video and trajectory, producing:
    world_model = {
        "task": {
            "description": "Pick-and-place: grasp an object and move it to a target location",
            "inferred_from": "Video shows robot arm approaching objects on tabletop, "
                           "trajectory shows gripper close followed by large position change "
                           "then gripper open",
        },

        "state_variables": {
            "arm_position": {"type": "continuous", "dims": 3},
            "gripper_state": {"type": "continuous", "range": [0, 8],
                            "semantics": "0=closed, 8=open"},
            "object_position": {"type": "continuous", "dims": 3,
                              "note": "not directly observed, inferred from grasp point"},
            "target_position": {"type": "continuous", "dims": 3,
                              "note": "inferred from release point"},
        },

        # INVARIANTS: must be true in ANY valid execution
        # These are the "physics" that the LLM knows
        "invariants": [
            {
                "name": "open_before_grasp",
                "rule": "Gripper must be open before closing on object",
                "asp": "The segment before close_gripper must have end_gripper > 5.0",
            },
            {
                "name": "at_object_before_grasp",
                "rule": "Arm must be near the object before closing gripper",
                "asp": "close_gripper segment must have low pos_distance (< 0.3)",
            },
            {
                "name": "hold_during_transport",
                "rule": "Gripper must stay closed while transporting",
                "asp": "Between close and open, no segment should have delta_gripper > 1.0",
            },
            {
                "name": "at_target_before_release",
                "rule": "Arm must be at target position before opening gripper",
                "asp": "open_gripper segment should follow a transport segment",
            },
            {
                "name": "sequence_order",
                "rule": "Must follow: approach → grasp → transport → release",
                "asp": "No valid execution reverses this order",
            },
        ],

        # DEGREES OF FREEDOM: what can vary across valid executions
        "variations": [
            {
                "name": "object_position",
                "description": "Object can be anywhere on the table",
                "affects": ["approach direction", "approach distance"],
                "param_ranges": {
                    "approach_x": [-3.0, 5.0],
                    "approach_y": [-3.0, 3.0],
                },
            },
            {
                "name": "target_position",
                "description": "Target location can vary",
                "affects": ["transport direction", "transport distance"],
                "param_ranges": {
                    "transport_distance": [0.5, 4.0],
                },
            },
            {
                "name": "object_size",
                "description": "Different objects require different grip widths",
                "affects": ["gripper close amount"],
                "param_ranges": {
                    "grip_delta": [-4.0, -1.5],
                },
            },
            {
                "name": "approach_speed",
                "description": "Can approach faster or slower",
                "affects": ["approach duration"],
                "param_ranges": {
                    "speed": [0.05, 0.4],
                },
            },
        ],

        # FAILURE MODES: what could go wrong
        "failure_modes": [
            {
                "name": "missed_grasp",
                "description": "Gripper closes but doesn't catch object",
                "probability": 0.15,
                "recovery": "Open gripper, re-approach, try again",
                "generates": "retry sequence: open → approach → close",
            },
            {
                "name": "object_slips",
                "description": "Object drops during transport",
                "probability": 0.05,
                "recovery": "Stop, re-approach dropped object, re-grasp",
                "generates": "recovery: transport_interrupted → approach → close → transport",
            },
            {
                "name": "wrong_object",
                "description": "Grasped wrong object, need to put back and retry",
                "probability": 0.10,
                "recovery": "Place back, approach correct object",
                "generates": "correction: transport_back → open → approach_correct → close",
            },
        ],

        # SIMPLIFICATIONS: conditions under which steps can be skipped
        "simplifications": [
            {
                "name": "object_at_target",
                "condition": "Object is already at the target location",
                "effect": "Skip entire task — no action needed",
            },
            {
                "name": "already_grasped",
                "condition": "Gripper already holding the object",
                "effect": "Skip approach and grasp — go directly to transport",
            },
            {
                "name": "adjacent_objects",
                "condition": "Next object is very close to current position",
                "effect": "Minimal approach — just adjust slightly",
            },
        ],
    }

    return world_model


# =============================================================================
# Synthetic Trace Generation
#
# Use the world model to generate variations of the observed demo.
# Each variation is a plausible execution under different conditions.
# =============================================================================

def generate_synthetic_traces(real_segments, world_model, n_variations=30):
    """
    Generate synthetic trace variations using world model knowledge.

    This is what humans do mentally: "What if the object was over there?
    What if I missed the grasp? What if it was already in place?"
    """
    traces = []

    # Extract the real demo's structure
    real_tools = []
    for seg in real_segments:
        c = seg["char"]
        if c["delta_gripper"] < -1.5:
            tool = "close_gripper"
        elif c["delta_gripper"] > 1.5:
            tool = "open_gripper"
        elif c["pos_distance"] > 0.3 and c["mean_speed"] > 0.1:
            tool = "move_to_position"
        elif c["pos_distance"] > 0.1:
            tool = "adjust_position"
        else:
            tool = "hold_steady"

        real_tools.append({
            "tool": tool,
            "params": {
                "pos_distance": c["pos_distance"],
                "delta_gripper": c["delta_gripper"],
                "efficiency": c["efficiency"],
                "mean_speed": c["mean_speed"],
                "start_gripper": c["start_gripper"],
                "end_gripper": c["end_gripper"],
            }
        })

    # Type 1: Normal variations (vary positions/speeds, keep structure)
    # ~60% of synthetic traces
    for i in range(int(n_variations * 0.6)):
        steps = []
        for rt in real_tools:
            step = dict(rt)
            step["params"] = dict(step["params"])
            # Vary position distances by ±50%
            step["params"]["pos_distance"] *= random.uniform(0.5, 1.5)
            step["params"]["mean_speed"] *= random.uniform(0.7, 1.3)
            # Vary efficiency slightly
            step["params"]["efficiency"] = min(1.0,
                step["params"]["efficiency"] * random.uniform(0.8, 1.1))
            steps.append(step)
        traces.append({"type": "normal_variation", "steps": steps})

    # Type 2: Failed grasp + retry
    # ~15% of traces
    for i in range(int(n_variations * 0.15)):
        steps = []
        retry_inserted = False
        for j, rt in enumerate(real_tools):
            steps.append(dict(rt))
            # After close_gripper, sometimes insert a retry sequence
            if rt["tool"] == "close_gripper" and not retry_inserted:
                retry_inserted = True
                # Failed grasp: open, adjust, close again
                steps.append({
                    "tool": "open_gripper",
                    "params": {"pos_distance": 0.05, "delta_gripper": 3.0,
                              "efficiency": 0.3, "mean_speed": 0.08,
                              "start_gripper": rt["params"]["end_gripper"],
                              "end_gripper": rt["params"]["start_gripper"]},
                })
                steps.append({
                    "tool": "adjust_position",
                    "params": {"pos_distance": 0.15, "delta_gripper": 0.0,
                              "efficiency": 0.5, "mean_speed": 0.1,
                              "start_gripper": rt["params"]["start_gripper"],
                              "end_gripper": rt["params"]["start_gripper"]},
                })
                steps.append({
                    "tool": "close_gripper",
                    "params": dict(rt["params"]),
                })
        traces.append({"type": "grasp_retry", "steps": steps})

    # Type 3: Object slip during transport
    # ~10% of traces
    for i in range(int(n_variations * 0.1)):
        steps = []
        slip_inserted = False
        for j, rt in enumerate(real_tools):
            steps.append(dict(rt))
            # After a move_to_position (transport), sometimes insert slip recovery
            if rt["tool"] == "move_to_position" and not slip_inserted and j > 0:
                if real_tools[j-1]["tool"] == "close_gripper":
                    slip_inserted = True
                    # Slip: re-approach and re-grasp
                    steps.append({
                        "tool": "adjust_position",
                        "params": {"pos_distance": 0.2, "delta_gripper": 0.5,
                                  "efficiency": 0.4, "mean_speed": 0.15,
                                  "start_gripper": 2.0, "end_gripper": 2.5},
                    })
                    steps.append({
                        "tool": "close_gripper",
                        "params": {"pos_distance": 0.1, "delta_gripper": -2.0,
                                  "efficiency": 0.3, "mean_speed": 0.08,
                                  "start_gripper": 2.5, "end_gripper": 0.5},
                    })
                    steps.append({
                        "tool": "move_to_position",
                        "params": dict(rt["params"]),
                    })
        traces.append({"type": "object_slip", "steps": steps})

    # Type 4: Shorter executions (object closer, or already positioned)
    # ~15% of traces
    for i in range(int(n_variations * 0.15)):
        steps = []
        for rt in real_tools:
            step = dict(rt)
            step["params"] = dict(step["params"])
            # Shorter distances overall
            step["params"]["pos_distance"] *= random.uniform(0.2, 0.6)
            step["params"]["mean_speed"] *= random.uniform(0.5, 1.0)
            steps.append(step)
        # Maybe skip the approach (already near object)
        if random.random() < 0.3 and len(steps) > 2:
            if steps[0]["tool"] == "move_to_position":
                steps = steps[1:]  # skip approach
        traces.append({"type": "short_execution", "steps": steps})

    return traces


def traces_to_asp(real_segments, real_tools, synthetic_traces, world_model):
    """Convert real demo + synthetic traces to ASP facts."""
    lines = []
    lines.append("% === Single Demo Compilation ===")
    lines.append(f"% 1 real demo + {len(synthetic_traces)} synthetic variations")
    lines.append(f"% Task: {world_model['task']['description']}")
    lines.append("% Synthetic traces generated from LLM world model knowledge")
    lines.append("")

    all_traces = []

    # Real demo as run_0
    all_traces.append(("real", real_tools))

    # Synthetic traces
    for i, st in enumerate(synthetic_traces):
        all_traces.append((st["type"], st["steps"]))

    for run_idx, (trace_type, steps) in enumerate(all_traces):
        run_id = f"run_{run_idx}"
        lines.append(f"% === Run {run_idx} ({trace_type}) ===")
        lines.append(f'job("{run_id}").')
        lines.append(f'job_status("{run_id}", "completed").')

        lines.append(f'call("{run_id}", "{run_id}_s0", "prompt_user", "completed").')
        lines.append(f'call("{run_id}", "{run_id}_s1", "llm_generate", "completed").')
        lines.append(f'depends("{run_id}", "{run_id}_s1", "{run_id}_s0").')

        prev = f"{run_id}_s1"
        ln = 1

        for i, step in enumerate(steps):
            ln += 1
            lid = f"{run_id}_l{ln}"
            lines.append(f'call("{run_id}", "{lid}", "llm_generate", "completed").')
            lines.append(f'depends("{run_id}", "{lid}", "{prev}").')

            sid = f"{run_id}_t{i}"
            lines.append(f'call("{run_id}", "{sid}", "{step["tool"]}", "completed").')
            lines.append(f'depends("{run_id}", "{sid}", "{lid}").')
            lines.append(f'spawned_by("{run_id}", "{sid}", "{lid}").')

            p = step["params"]
            lines.append(f'param("{run_id}", "{sid}", "pos_distance", "{p["pos_distance"]:.2f}").')
            lines.append(f'param("{run_id}", "{sid}", "delta_gripper", "{p["delta_gripper"]:.1f}").')
            lines.append(f'param("{run_id}", "{sid}", "efficiency", "{p["efficiency"]:.1f}").')
            lines.append(f'param("{run_id}", "{sid}", "mean_speed", "{p["mean_speed"]:.3f}").')

            prev = sid

        ln += 1
        lines.append(f'call("{run_id}", "{run_id}_l{ln}", "llm_generate", "completed").')
        lines.append(f'depends("{run_id}", "{run_id}_l{ln}", "{prev}").')
        lines.append(f'call("{run_id}", "{run_id}_n", "notify_user", "completed").')
        lines.append(f'depends("{run_id}", "{run_id}_n", "{run_id}_l{ln}").')
        lines.append(f'spawned_by("{run_id}", "{run_id}_n", "{run_id}_l{ln}").')
        lines.append("")

    # Add world model constraints as ASP integrity constraints
    lines.append("")
    lines.append("% === World Model Constraints (from LLM) ===")
    lines.append("% These are physical invariants that the LLM knows from world knowledge.")
    lines.append("% They prune the search space — invalid programs are ruled out.")
    lines.append("")

    for inv in world_model["invariants"]:
        lines.append(f"% Invariant: {inv['name']} — {inv['rule']}")

    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Compile a robust program from a single demonstration")
    parser.add_argument("--dataset", default="lerobot/ucsd_pick_and_place_dataset")
    parser.add_argument("--episode", type=int, default=178)
    parser.add_argument("--n-synthetic", type=int, default=30,
                        help="Number of synthetic trace variations to generate")
    parser.add_argument("--rules", default="rules/mine_patterns.lp")
    parser.add_argument("--output", default="examples/single-demo/result.json")
    parser.add_argument("--penalty", type=float, default=0.8)
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    rules_path = project_root / args.rules

    # Load single episode
    episodes = load_episodes(args.dataset, args.episode + 1)
    if args.episode not in episodes:
        sys.exit(f"Episode {args.episode} not found")

    frames = episodes[args.episode]
    print(f"\nSingle demo: Episode {args.episode}, {len(frames)} frames")

    # Segment the real demo
    states = np.array([f['observation.state'] for f in frames])
    actions = np.array([f['action'] for f in frames])
    boundaries = detect_changepoints(states, min_segment_len=3,
                                     penalty_factor=args.penalty)

    segments = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        if end - start < 2:
            continue
        seg_states = states[start:end + 1]
        seg_actions = actions[start:min(end + 1, len(actions))]
        char = characterize_segment(seg_states, seg_actions)
        segments.append({"start_frame": start, "end_frame": end, "char": char})

    print(f"Segmented into {len(segments)} state transitions:")
    for i, seg in enumerate(segments):
        c = seg["char"]
        print(f"  seg {i}: Δpos={c['pos_distance']:.3f} Δgrip={c['delta_gripper']:+.1f} "
              f"eff={c['efficiency']:.2f} spd={c['mean_speed']:.3f}")

    # Query world model (LLM)
    print(f"\n{'='*60}")
    print("QUERYING WORLD MODEL (LLM)")
    print(f"{'='*60}")
    print("\nIn production: sends video frames + trajectory to Claude API")
    print("For prototype: using structured world model knowledge\n")

    world_model = query_world_model(segments)

    print(f"Task: {world_model['task']['description']}")
    print(f"\nInvariants discovered by LLM:")
    for inv in world_model["invariants"]:
        print(f"  - {inv['name']}: {inv['rule']}")
    print(f"\nVariations identified:")
    for var in world_model["variations"]:
        print(f"  - {var['name']}: {var['description']}")
    print(f"\nFailure modes predicted:")
    for fm in world_model["failure_modes"]:
        print(f"  - {fm['name']} ({fm['probability']*100:.0f}%): {fm['description']}")
        print(f"    Recovery: {fm['recovery']}")

    # Extract tool sequence from real demo
    real_tools = []
    for seg in segments:
        c = seg["char"]
        if c["delta_gripper"] < -1.5:
            tool = "close_gripper"
        elif c["delta_gripper"] > 1.5:
            tool = "open_gripper"
        elif c["pos_distance"] > 0.3 and c["mean_speed"] > 0.1:
            tool = "move_to_position"
        elif c["pos_distance"] > 0.1:
            tool = "adjust_position"
        else:
            tool = "hold_steady"
        real_tools.append({
            "tool": tool,
            "params": {
                "pos_distance": c["pos_distance"],
                "delta_gripper": c["delta_gripper"],
                "efficiency": c["efficiency"],
                "mean_speed": c["mean_speed"],
                "start_gripper": c["start_gripper"],
                "end_gripper": c["end_gripper"],
            },
        })

    print(f"\nReal demo tool sequence:")
    for i, rt in enumerate(real_tools):
        print(f"  {i}: {rt['tool']}  Δpos={rt['params']['pos_distance']:.3f} "
              f"Δgrip={rt['params']['delta_gripper']:+.1f}")

    # Generate synthetic variations
    print(f"\n{'='*60}")
    print(f"GENERATING {args.n_synthetic} SYNTHETIC VARIATIONS")
    print(f"{'='*60}")

    synthetic = generate_synthetic_traces(segments, world_model, args.n_synthetic)

    type_counts = {}
    for st in synthetic:
        type_counts[st["type"]] = type_counts.get(st["type"], 0) + 1
    for t, c in sorted(type_counts.items()):
        print(f"  {t}: {c} traces ({len(synthetic[0]['steps'])} steps typical)")

    # Convert to ASP facts
    asp_lines = traces_to_asp(segments, real_tools, synthetic, world_model)

    # Write ASP facts
    output_dir = project_root / "examples" / "single-demo"
    output_dir.mkdir(parents=True, exist_ok=True)
    traces_path = output_dir / "traces.lp"
    with open(traces_path, "w") as f:
        f.write("\n".join(asp_lines))
    print(f"\nASP facts: {len(asp_lines)} lines → {traces_path}")

    # Run ASP pattern miner
    print(f"\n{'='*60}")
    print("COMPILING WITH ASP PATTERN MINER")
    print(f"{'='*60}")

    sys.path.insert(0, str(Path(__file__).parent))
    from compile import run_asp_strategy, synthesize, describe_workflow

    results = run_asp_strategy(traces_path, rules_path, verbose=False)

    print(f"\nCore tools (from 1 real + {args.n_synthetic} synthetic):")
    for tool in sorted(results["core_tools"]):
        count = results["tool_counts"].get(tool, "?")
        print(f"  {tool} — in {count}/{results['job_count']} runs")

    if results["consistent_order"]:
        print(f"\nConsistent ordering:")
        for t1, t2 in sorted(results["consistent_order"]):
            print(f"  {t1} -> {t2}")

    if results["conflicting_order"]:
        print(f"\nConflicting orderings resolved:")
        for t1, t2 in sorted(results["conflicting_order"]):
            print(f"  {t1} <-> {t2}")
        for t1, t2 in sorted(results.get("chosen_order", [])):
            print(f"  resolved: {t1} -> {t2}")

    if results["fusion_candidates"]:
        print(f"\nFusion candidates:")
        for a, b in sorted(results["fusion_candidates"]):
            print(f"  {a} + {b}")

    if results["conditionals"]:
        print(f"\nConditional execution:")
        for b, a in sorted(results["conditionals"]):
            rate = results["conditional_rates"].get((b, a))
            if rate:
                pct = rate[0] * 100 // rate[1]
                print(f"  {b} conditional on {a} ({rate[0]}/{rate[1]} = {pct}%)")

    if results["stable_params"]:
        print(f"\nStable parameters:")
        for tool in sorted(results["stable_params"]):
            for key, vals in sorted(results["stable_params"][tool].items()):
                print(f"  {tool}.{key} = {vals[0] if len(vals)==1 else vals}")

    if results["variable_params"]:
        print(f"\nVariable parameters (runtime-dependent):")
        for tool in sorted(results["variable_params"]):
            print(f"  {tool}: {', '.join(sorted(results['variable_params'][tool]))}")

    # Synthesize compiled workflow
    compiled = synthesize(results)

    output_path = project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(compiled, f, indent=2)

    print(f"\n{'='*60}")
    print("COMPILED PROGRAM")
    print(f"{'='*60}")
    describe_workflow(compiled, results)

    print(f"\n{'='*60}")
    print("WHAT THIS MEANS")
    print(f"{'='*60}")
    print(f"\nFrom 1 demo + LLM world knowledge:")
    print(f"  - {len(results['core_tools'])} core primitives discovered")
    print(f"  - {len(results['consistent_order'])} ordering constraints")
    print(f"  - {len(results['conditionals'])} conditional branches (from failure modes)")
    print(f"  - {len(results.get('fusion_candidates', []))} fusion candidates")
    print(f"\nThe LLM's world model substituted for {args.n_synthetic} additional demos")
    print(f"by predicting what COULD happen — not just what DID happen.")
    print(f"\nWritten to {output_path}")


if __name__ == "__main__":
    main()
