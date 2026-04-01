#!/usr/bin/env python3
"""
trajectory_rewrite.py — Discover and apply rewrite rules to robot trajectories.

Given start/end states and multiple noisy demonstrations, discovers:
  1. NOISE: segments with near-zero state change (hesitation, shaking)
  2. CANCELLING PAIRS: segments whose effects get undone later (fumbles)
  3. REDUNDANT PATHS: multi-segment paths replaceable by shorter sequences
  4. ESSENTIAL TRANSITIONS: the minimal set of state changes to reach the goal

This is the "scene diff" applied to trajectories:
  State_A → [what are the minimal essential actions?] → State_B

Usage:
    python src/trajectory_rewrite.py \
        --dataset lerobot/ucsd_pick_and_place_dataset \
        --episodes 50

Outputs analysis of which segments are essential vs prunable, and
what the "compiled" (minimal) trajectory would look like.
"""

import argparse
import sys
from collections import defaultdict

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy required")
    sys.exit(1)

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: datasets required")
    sys.exit(1)

from segment_discover import (
    load_episodes, detect_changepoints, characterize_segment
)


# --- Rewrite rules ---

def detect_noise_segments(segments, noise_threshold=0.02):
    """
    Rule 1: NOISE — segments where almost nothing happens.
    Hesitation, shaking, drift. Δstate ≈ 0.
    These can be pruned without changing the outcome.
    """
    noise = []
    for i, seg in enumerate(segments):
        c = seg["char"]
        if c["pos_distance"] < noise_threshold and abs(c["delta_gripper"]) < 0.5:
            noise.append({
                "index": i,
                "type": "noise",
                "reason": f"near-zero state change (Δpos={c['pos_distance']:.4f}, Δgrip={c['delta_gripper']:.2f})",
                "frames": (seg["start_frame"], seg["end_frame"]),
            })
    return noise


def detect_cancelling_pairs(segments, cancel_threshold=0.15):
    """
    Rule 2: CANCELLING PAIRS — segment A followed (eventually) by segment B
    where B undoes A's state change. These are fumbles/corrections.

    Example: move right 0.3 ... then move left 0.3 = net zero.
    Example: close gripper ... then open gripper ... then close again.
    """
    pairs = []
    for i in range(len(segments)):
        ci = segments[i]["char"]
        # Look for a later segment that undoes this one
        cumulative_delta = np.zeros_like(ci["delta_pos"])
        cumulative_grip = 0
        for j in range(i + 1, len(segments)):
            cj = segments[j]["char"]
            cumulative_delta += cj["delta_pos"]
            cumulative_grip += cj["delta_gripper"]

            # Check if segment j cancels segment i
            net_pos = ci["delta_pos"] + cumulative_delta
            net_grip = ci["delta_gripper"] + cumulative_grip

            if (np.linalg.norm(net_pos) < cancel_threshold and
                    abs(net_grip) < 1.0):
                # Segments i through j cancel out
                if j - i <= 3:  # Only flag short cancellation sequences
                    pairs.append({
                        "start_idx": i,
                        "end_idx": j,
                        "type": "cancelling",
                        "reason": f"segments {i}-{j} cancel out "
                                  f"(net Δpos={np.linalg.norm(net_pos):.3f}, "
                                  f"net Δgrip={net_grip:.1f})",
                        "net_state_change": np.linalg.norm(net_pos) + abs(net_grip) * 0.1,
                    })
                break
    return pairs


def detect_redundant_paths(segments, shortcut_threshold=0.5):
    """
    Rule 3: REDUNDANT PATHS — multiple segments that could be replaced
    by a single segment achieving the same net state change.

    A→B→C where the direct A→C path is much shorter than going through B.
    """
    redundant = []
    for i in range(len(segments)):
        for window in [2, 3, 4]:  # Check windows of 2-4 segments
            j = i + window
            if j > len(segments):
                break

            # Net state change across the window
            sub = segments[i:j]
            total_pos = sum(s["char"]["delta_pos"] for s in sub)
            total_grip = sum(s["char"]["delta_gripper"] for s in sub)
            total_frames = sum(s["char"]["n_frames"] for s in sub)

            # Direct distance
            direct_dist = np.linalg.norm(total_pos)

            # Actual path length
            actual_dist = sum(s["char"]["pos_distance"] for s in sub)

            # Efficiency: how much longer is the actual path vs direct?
            if direct_dist > 0.01:
                path_ratio = actual_dist / direct_dist
                if path_ratio > 1.5 and window > 1:
                    redundant.append({
                        "start_idx": i,
                        "end_idx": j - 1,
                        "type": "redundant_path",
                        "reason": f"segments {i}-{j-1}: path is {path_ratio:.1f}x "
                                  f"longer than direct "
                                  f"(actual={actual_dist:.3f}, direct={direct_dist:.3f})",
                        "path_ratio": path_ratio,
                        "could_save_frames": total_frames - max(s["char"]["n_frames"] for s in sub),
                    })
    return redundant


def compute_essential_transitions(segments, episodes_segments):
    """
    Rule 4: ESSENTIAL TRANSITIONS — state changes that appear consistently
    across multiple demonstrations of the same task.

    A state transition is "essential" if removing it would make it impossible
    to reach the goal state from the start state.
    """
    # For each episode, compute cumulative state changes
    essential = []

    # Find the mandatory state changes by looking at start/end states
    # across all episodes
    start_states = []
    end_states = []
    for ep_id, segs in episodes_segments.items():
        if not segs:
            continue
        start_states.append(np.concatenate([
            segs[0]["char"]["delta_pos"] * 0 + segs[0]["states"][0][:3],
            [segs[0]["states"][0][-1]]
        ]))
        end_states.append(np.concatenate([
            segs[-1]["states"][-1][:3],
            [segs[-1]["states"][-1][-1]]
        ]))

    if not start_states:
        return []

    # The "goal diff" is what MUST happen between start and end
    goal_diffs = [end - start for start, end in zip(start_states, end_states)]
    mean_goal = np.mean(goal_diffs, axis=0)
    std_goal = np.std(goal_diffs, axis=0)

    return {
        "mean_goal_diff": mean_goal,
        "std_goal_diff": std_goal,
        "n_episodes": len(start_states),
        "position_change_required": np.linalg.norm(mean_goal[:3]),
        "gripper_change_required": mean_goal[3],
    }


def analyze_episode(frames, penalty_factor=0.8):
    """Full analysis of one episode: segment, characterize, find rewrites."""
    states = np.array([f['observation.state'] for f in frames])
    actions = np.array([f['action'] for f in frames])

    boundaries = detect_changepoints(states, min_segment_len=3,
                                     penalty_factor=penalty_factor)

    segments = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        if end - start < 2:
            continue
        seg_states = states[start:end + 1]
        seg_actions = actions[start:min(end + 1, len(actions))]
        char = characterize_segment(seg_states, seg_actions)
        segments.append({
            "start_frame": start,
            "end_frame": end,
            "char": char,
            "states": seg_states,
        })

    # Apply rewrite rules
    noise = detect_noise_segments(segments)
    cancelling = detect_cancelling_pairs(segments)
    redundant = detect_redundant_paths(segments)

    # Classify each segment
    prunable_indices = set()
    for n in noise:
        prunable_indices.add(n["index"])
    for c in cancelling:
        for idx in range(c["start_idx"], c["end_idx"] + 1):
            prunable_indices.add(idx)

    essential_indices = set(range(len(segments))) - prunable_indices

    return {
        "segments": segments,
        "noise": noise,
        "cancelling": cancelling,
        "redundant": redundant,
        "essential_indices": essential_indices,
        "prunable_indices": prunable_indices,
        "total_frames": len(frames),
        "essential_frames": sum(
            segments[i]["char"]["n_frames"] for i in essential_indices
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Discover rewrite rules for robot trajectories")
    parser.add_argument("--dataset", default="lerobot/ucsd_pick_and_place_dataset")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--penalty", type=float, default=0.8)
    args = parser.parse_args()

    episodes = load_episodes(args.dataset, args.episodes)

    print(f"\n{'='*70}")
    print("TRAJECTORY REWRITE ANALYSIS")
    print(f"{'='*70}\n")

    total_noise = 0
    total_cancelling = 0
    total_redundant = 0
    total_segments = 0
    total_essential = 0
    total_frames = 0
    total_essential_frames = 0

    episode_results = {}
    for ep_id, frames in sorted(episodes.items()):
        result = analyze_episode(frames, args.penalty)
        episode_results[ep_id] = result

        n_segs = len(result["segments"])
        n_essential = len(result["essential_indices"])
        n_prunable = len(result["prunable_indices"])

        total_segments += n_segs
        total_essential += n_essential
        total_noise += len(result["noise"])
        total_cancelling += len(result["cancelling"])
        total_redundant += len(result["redundant"])
        total_frames += result["total_frames"]
        total_essential_frames += result["essential_frames"]

    # Summary
    print(f"Episodes analyzed: {len(episodes)}")
    print(f"Total segments: {total_segments}")
    print(f"Essential segments: {total_essential} ({total_essential*100//max(total_segments,1)}%)")
    print(f"Prunable segments: {total_segments - total_essential} ({(total_segments-total_essential)*100//max(total_segments,1)}%)")
    print(f"  - Noise (near-zero Δstate): {total_noise}")
    print(f"  - Cancelling pairs (fumbles): {total_cancelling}")
    print(f"Redundant paths found: {total_redundant}")
    print(f"\nFrame reduction: {total_frames} → {total_essential_frames} "
          f"({total_essential_frames*100//max(total_frames,1)}% of original)")
    print()

    # Detailed per-episode breakdown
    print(f"{'Ep':>3} {'Segs':>5} {'Essnt':>5} {'Noise':>5} {'Cancel':>6} {'Redund':>6} {'Frames':>8} {'→':>3} {'EssFrm':>7} {'%':>5}")
    print("-" * 70)
    for ep_id in sorted(episode_results.keys()):
        r = episode_results[ep_id]
        n = len(r["segments"])
        e = len(r["essential_indices"])
        pct = r["essential_frames"] * 100 // max(r["total_frames"], 1)
        print(f"{ep_id:3d} {n:5d} {e:5d} {len(r['noise']):5d} "
              f"{len(r['cancelling']):6d} {len(r['redundant']):6d} "
              f"{r['total_frames']:8d}     {r['essential_frames']:7d} {pct:4d}%")

    # Show some interesting rewrite examples
    print(f"\n{'='*70}")
    print("EXAMPLE REWRITES")
    print(f"{'='*70}")

    examples_shown = 0
    for ep_id in sorted(episode_results.keys()):
        r = episode_results[ep_id]
        if r["cancelling"] or (r["noise"] and len(r["segments"]) > 3):
            print(f"\nEpisode {ep_id}:")
            for seg_idx, seg in enumerate(r["segments"]):
                c = seg["char"]
                status = "✓ essential" if seg_idx in r["essential_indices"] else "✗ prunable"
                reason = ""
                for n in r["noise"]:
                    if n["index"] == seg_idx:
                        reason = f" [{n['reason']}]"
                for cancel in r["cancelling"]:
                    if cancel["start_idx"] <= seg_idx <= cancel["end_idx"]:
                        reason = f" [{cancel['reason']}]"
                print(f"  seg {seg_idx}: Δpos={c['pos_distance']:.3f} "
                      f"Δgrip={c['delta_gripper']:+.1f} "
                      f"eff={c['efficiency']:.2f} "
                      f"spd={c['mean_speed']:.3f} "
                      f"frames={c['n_frames']:2d}  "
                      f"{status}{reason}")

            for red in r["redundant"]:
                print(f"  → rewrite: {red['reason']}")

            examples_shown += 1
            if examples_shown >= 8:
                break

    # Output the "compiled" trajectory concept
    print(f"\n{'='*70}")
    print("COMPILED TRAJECTORY (essential segments only)")
    print(f"{'='*70}")
    print(f"\nIf we keep only essential segments across all episodes:")
    print(f"  Segments: {total_segments} → {total_essential} ({100 - total_essential*100//max(total_segments,1)}% reduction)")
    print(f"  Frames: {total_frames} → {total_essential_frames} ({100 - total_essential_frames*100//max(total_frames,1)}% reduction)")
    print(f"\nThe pruned segments are hesitation, shaking, and fumble-recovery —")
    print(f"exactly what you'd remove to get a clean demonstration for policy learning.")


if __name__ == "__main__":
    main()
