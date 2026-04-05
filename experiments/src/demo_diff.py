#!/usr/bin/env python3
"""
demo_diff.py — Discover rewrite rules by diffing good vs bad demonstrations.

Instead of hand-coding rewrite rules, we:
  1. Score each demo by efficiency (path ratio, completion, segment count)
  2. Split into "clean" and "messy" groups
  3. Run the EXISTING ASP pattern miner on each group separately
  4. Diff the results: what structure exists in clean but not messy = essential
     What exists in messy but not clean = noise patterns (rewrite candidates)

This uses Clingo to discover rewrite rules, not hand-coded Python heuristics.

Additionally: ranks demos by quality for dataset curation — identifying
which clips are "interesting" (clean technique) vs "boring" (fumbling).

Usage:
    python src/demo_diff.py \
        --dataset lerobot/ucsd_pick_and_place_dataset \
        --episodes 200
"""

import argparse
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

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
    load_episodes, detect_changepoints, characterize_segment,
    cluster_transitions, name_cluster
)


def score_episode(frames):
    """
    Score a demo's quality purely from trajectory data.
    Higher = cleaner demo, lower = messier.
    """
    states = np.array([f['observation.state'] for f in frames])
    actions = np.array([f['action'] for f in frames])

    # Overall start-to-end state change
    start = states[0]
    end = states[-1]
    goal_dist = np.linalg.norm(end[:3] - start[:3])
    gripper_change = abs(end[-1] - start[-1])

    # Actual path length
    path_lengths = np.linalg.norm(np.diff(states[:, :3], axis=0), axis=1)
    actual_path = np.sum(path_lengths)

    # Path efficiency
    efficiency = goal_dist / max(actual_path, 0.001)

    # Speed consistency (low variance = smooth motion)
    speeds = np.linalg.norm(np.diff(states[:, :3], axis=0), axis=1)
    speed_cv = np.std(speeds) / max(np.mean(speeds), 0.001)  # coefficient of variation

    # Direction consistency (how often does the arm reverse?)
    if len(speeds) > 2:
        vel = np.diff(states[:, :3], axis=0)
        direction_changes = 0
        for i in range(1, len(vel)):
            cos_sim = np.dot(vel[i], vel[i-1]) / (
                max(np.linalg.norm(vel[i]), 1e-8) *
                max(np.linalg.norm(vel[i-1]), 1e-8))
            if cos_sim < 0:  # direction reversal
                direction_changes += 1
        reversal_rate = direction_changes / len(vel)
    else:
        reversal_rate = 0

    # Number of gripper state changes (fewer = more decisive)
    gripper_cmd = actions[:, -1]
    gripper_sign_changes = np.sum(np.abs(np.diff(np.sign(gripper_cmd))) > 0)

    # Composite score (higher = better)
    score = (
        efficiency * 2.0                    # path efficiency (0-1, higher better)
        - speed_cv * 0.3                    # penalize jerky motion
        - reversal_rate * 1.0               # penalize direction changes
        - gripper_sign_changes * 0.1        # penalize gripper fumbling
        + (goal_dist > 0.3) * 0.5          # bonus for actually moving something
        + (gripper_change > 1.0) * 0.5     # bonus for meaningful gripper action
    )

    return {
        "score": score,
        "efficiency": efficiency,
        "actual_path": actual_path,
        "goal_dist": goal_dist,
        "gripper_change": gripper_change,
        "speed_cv": speed_cv,
        "reversal_rate": reversal_rate,
        "gripper_sign_changes": int(gripper_sign_changes),
    }


def episodes_to_asp_facts(episodes, episode_ids, segments_by_ep, cluster_labels,
                          cluster_names, global_seg_offset):
    """Convert a subset of episodes to ASP facts."""
    lines = []
    seg_idx = global_seg_offset

    for ep_id in episode_ids:
        frames = episodes[ep_id]
        segs = segments_by_ep[ep_id]
        if not segs:
            continue

        run_id = f"run_{ep_id}"
        lines.append(f'job("{run_id}").')
        lines.append(f'job_status("{run_id}", "completed").')

        lines.append(f'call("{run_id}", "{run_id}_s0", "prompt_user", "completed").')
        lines.append(f'call("{run_id}", "{run_id}_s1", "llm_generate", "completed").')
        lines.append(f'depends("{run_id}", "{run_id}_s1", "{run_id}_s0").')

        prev = f"{run_id}_s1"
        llm_n = 1

        for i, seg in enumerate(segs):
            label = cluster_labels[seg_idx]
            tool = cluster_names[label]

            llm_n += 1
            llm_id = f"{run_id}_l{llm_n}"
            lines.append(f'call("{run_id}", "{llm_id}", "llm_generate", "completed").')
            lines.append(f'depends("{run_id}", "{llm_id}", "{prev}").')

            step_id = f"{run_id}_t{i}"
            lines.append(f'call("{run_id}", "{step_id}", "{tool}", "completed").')
            lines.append(f'depends("{run_id}", "{step_id}", "{llm_id}").')
            lines.append(f'spawned_by("{run_id}", "{step_id}", "{llm_id}").')

            c = seg["char"]
            lines.append(f'param("{run_id}", "{step_id}", "pos_distance", "{c["pos_distance"]:.2f}").')
            lines.append(f'param("{run_id}", "{step_id}", "delta_gripper", "{c["delta_gripper"]:.1f}").')
            lines.append(f'param("{run_id}", "{step_id}", "efficiency", "{c["efficiency"]:.1f}").')

            prev = step_id
            seg_idx += 1

        llm_n += 1
        lines.append(f'call("{run_id}", "{run_id}_l{llm_n}", "llm_generate", "completed").')
        lines.append(f'depends("{run_id}", "{run_id}_l{llm_n}", "{prev}").')
        lines.append(f'call("{run_id}", "{run_id}_n", "notify_user", "completed").')
        lines.append(f'depends("{run_id}", "{run_id}_n", "{run_id}_l{llm_n}").')
        lines.append(f'spawned_by("{run_id}", "{run_id}_n", "{run_id}_l{llm_n}").')

    return lines


def run_asp_compile(asp_lines, rules_path):
    """Run the ASP pattern miner and return results."""
    # Import compile module
    sys.path.insert(0, str(Path(__file__).parent))
    from compile import run_asp_strategy

    with tempfile.NamedTemporaryFile(mode='w', suffix='.lp', delete=False) as f:
        f.write("\n".join(asp_lines))
        traces_path = Path(f.name)

    try:
        results = run_asp_strategy(traces_path, rules_path, verbose=False)
    finally:
        traces_path.unlink()

    return results


def diff_results(clean_results, messy_results):
    """Diff pattern mining results between clean and messy demos."""
    diff = {
        "clean_only_core": [],
        "messy_only_core": [],
        "shared_core": [],
        "clean_only_order": [],
        "messy_only_order": [],
        "clean_only_fusion": [],
        "messy_only_fusion": [],
        "clean_only_exclusive": [],
        "messy_only_exclusive": [],
    }

    clean_core = set(clean_results["core_tools"])
    messy_core = set(messy_results["core_tools"])
    diff["clean_only_core"] = sorted(clean_core - messy_core)
    diff["messy_only_core"] = sorted(messy_core - clean_core)
    diff["shared_core"] = sorted(clean_core & messy_core)

    clean_order = set(clean_results["consistent_order"])
    messy_order = set(messy_results["consistent_order"])
    diff["clean_only_order"] = sorted(clean_order - messy_order)
    diff["messy_only_order"] = sorted(messy_order - clean_order)

    clean_fusion = set(tuple(x) for x in clean_results["fusion_candidates"])
    messy_fusion = set(tuple(x) for x in messy_results["fusion_candidates"])
    diff["clean_only_fusion"] = sorted(clean_fusion - messy_fusion)
    diff["messy_only_fusion"] = sorted(messy_fusion - clean_fusion)

    clean_excl = set(tuple(x) for x in clean_results["mutually_exclusive"])
    messy_excl = set(tuple(x) for x in messy_results["mutually_exclusive"])
    diff["clean_only_exclusive"] = sorted(clean_excl - messy_excl)
    diff["messy_only_exclusive"] = sorted(messy_excl - clean_excl)

    return diff


def main():
    parser = argparse.ArgumentParser(
        description="Discover rewrite rules by diffing good vs bad demos")
    parser.add_argument("--dataset", default="lerobot/ucsd_pick_and_place_dataset")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--rules", default="rules/mine_patterns_relaxed.lp")
    parser.add_argument("--split-percentile", type=int, default=50,
                        help="Score percentile to split clean/messy")
    parser.add_argument("--cluster-threshold", type=float, default=2.0)
    parser.add_argument("--penalty", type=float, default=0.8)
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    rules_path = project_root / args.rules

    # Load
    episodes = load_episodes(args.dataset, args.episodes)

    # Score all demos
    print("\nScoring demos...")
    scores = {}
    for ep_id, frames in sorted(episodes.items()):
        scores[ep_id] = score_episode(frames)

    # Rank
    ranked = sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)

    print(f"\n{'='*70}")
    print("DEMO QUALITY RANKING")
    print(f"{'='*70}")
    print(f"\n{'Rank':>4} {'Ep':>4} {'Score':>6} {'Eff':>5} {'GoalΔ':>6} "
          f"{'GripΔ':>6} {'Revsls':>6} {'GripSw':>6}")
    print("-" * 55)
    for rank, (ep_id, sc) in enumerate(ranked[:15]):
        print(f"{rank+1:4d} {ep_id:4d} {sc['score']:6.2f} {sc['efficiency']:5.2f} "
              f"{sc['goal_dist']:6.3f} {sc['gripper_change']:6.1f} "
              f"{sc['reversal_rate']:6.2f} {sc['gripper_sign_changes']:6d}")
    print("  ...")
    for rank_offset, (ep_id, sc) in enumerate(ranked[-10:]):
        rank = len(ranked) - 10 + rank_offset + 1
        print(f"{rank:4d} {ep_id:4d} {sc['score']:6.2f} {sc['efficiency']:5.2f} "
              f"{sc['goal_dist']:6.3f} {sc['gripper_change']:6.1f} "
              f"{sc['reversal_rate']:6.2f} {sc['gripper_sign_changes']:6d}")

    # Split into clean and messy
    split_idx = len(ranked) * args.split_percentile // 100
    clean_ids = [ep_id for ep_id, _ in ranked[:split_idx]]
    messy_ids = [ep_id for ep_id, _ in ranked[split_idx:]]

    clean_scores = [scores[e]["score"] for e in clean_ids]
    messy_scores = [scores[e]["score"] for e in messy_ids]
    print(f"\nClean group: {len(clean_ids)} episodes "
          f"(score {min(clean_scores):.2f} - {max(clean_scores):.2f})")
    print(f"Messy group: {len(messy_ids)} episodes "
          f"(score {min(messy_scores):.2f} - {max(messy_scores):.2f})")

    # Segment all episodes
    print("\nSegmenting all episodes...")
    segments_by_ep = {}
    all_segments = []
    for ep_id, frames in sorted(episodes.items()):
        states = np.array([f['observation.state'] for f in frames])
        actions = np.array([f['action'] for f in frames])
        boundaries = detect_changepoints(states, min_segment_len=3,
                                         penalty_factor=args.penalty)
        segs = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            if end - start < 2:
                continue
            seg_states = states[start:end + 1]
            seg_actions = actions[start:min(end + 1, len(actions))]
            char = characterize_segment(seg_states, seg_actions)
            segs.append({"start_frame": start, "end_frame": end, "char": char})
        segments_by_ep[ep_id] = segs
        all_segments.extend(segs)

    # Cluster
    print(f"Clustering {len(all_segments)} segments...")
    labels, n_clusters = cluster_transitions(all_segments,
                                             distance_threshold=args.cluster_threshold)
    cluster_members = defaultdict(list)
    for i, label in enumerate(labels):
        cluster_members[label].append(all_segments[i])
    cluster_names = {}
    name_counts = defaultdict(int)
    for cid in sorted(cluster_members.keys()):
        base_name = name_cluster(cluster_members[cid])
        name_counts[base_name] += 1
        if name_counts[base_name] > 1:
            cluster_names[cid] = f"{base_name}_{name_counts[base_name]}"
        else:
            cluster_names[cid] = base_name

    print(f"Discovered {n_clusters} primitive types")

    # Build ASP facts for each group
    # We need to track segment indices per episode
    ep_seg_offsets = {}
    offset = 0
    for ep_id in sorted(episodes.keys()):
        ep_seg_offsets[ep_id] = offset
        offset += len(segments_by_ep[ep_id])

    print("\nCompiling clean demos...")
    clean_asp = episodes_to_asp_facts(episodes, clean_ids, segments_by_ep,
                                       labels, cluster_names, 0)
    # Need to recalculate offsets for clean group only
    clean_offset_map = {}
    off = 0
    for ep_id in sorted(episodes.keys()):
        clean_offset_map[ep_id] = off
        off += len(segments_by_ep[ep_id])

    clean_asp_lines = []
    seg_idx = 0
    for ep_id in sorted(clean_ids):
        segs = segments_by_ep[ep_id]
        if not segs:
            seg_idx += len(segs)
            continue
        run_id = f"run_{ep_id}"
        clean_asp_lines.append(f'job("{run_id}").')
        clean_asp_lines.append(f'call("{run_id}", "{run_id}_s0", "prompt_user", "completed").')
        clean_asp_lines.append(f'call("{run_id}", "{run_id}_s1", "llm_generate", "completed").')
        clean_asp_lines.append(f'depends("{run_id}", "{run_id}_s1", "{run_id}_s0").')
        prev = f"{run_id}_s1"
        ln = 1
        for i, seg in enumerate(segs):
            gidx = ep_seg_offsets[ep_id] + i
            label = labels[gidx]
            tool = cluster_names[label]
            ln += 1
            lid = f"{run_id}_l{ln}"
            clean_asp_lines.append(f'call("{run_id}", "{lid}", "llm_generate", "completed").')
            clean_asp_lines.append(f'depends("{run_id}", "{lid}", "{prev}").')
            sid = f"{run_id}_t{i}"
            clean_asp_lines.append(f'call("{run_id}", "{sid}", "{tool}", "completed").')
            clean_asp_lines.append(f'depends("{run_id}", "{sid}", "{lid}").')
            clean_asp_lines.append(f'spawned_by("{run_id}", "{sid}", "{lid}").')
            c = seg["char"]
            clean_asp_lines.append(f'param("{run_id}", "{sid}", "eff", "{c["efficiency"]:.1f}").')
            prev = sid
        ln += 1
        clean_asp_lines.append(f'call("{run_id}", "{run_id}_l{ln}", "llm_generate", "completed").')
        clean_asp_lines.append(f'depends("{run_id}", "{run_id}_l{ln}", "{prev}").')
        clean_asp_lines.append(f'call("{run_id}", "{run_id}_n", "notify_user", "completed").')
        clean_asp_lines.append(f'depends("{run_id}", "{run_id}_n", "{run_id}_l{ln}").')
        clean_asp_lines.append(f'spawned_by("{run_id}", "{run_id}_n", "{run_id}_l{ln}").')

    print(f"  {len(clean_asp_lines)} ASP facts for {len(clean_ids)} clean demos")

    messy_asp_lines = []
    for ep_id in sorted(messy_ids):
        segs = segments_by_ep[ep_id]
        if not segs:
            continue
        run_id = f"run_{ep_id}"
        messy_asp_lines.append(f'job("{run_id}").')
        messy_asp_lines.append(f'call("{run_id}", "{run_id}_s0", "prompt_user", "completed").')
        messy_asp_lines.append(f'call("{run_id}", "{run_id}_s1", "llm_generate", "completed").')
        messy_asp_lines.append(f'depends("{run_id}", "{run_id}_s1", "{run_id}_s0").')
        prev = f"{run_id}_s1"
        ln = 1
        for i, seg in enumerate(segs):
            gidx = ep_seg_offsets[ep_id] + i
            label = labels[gidx]
            tool = cluster_names[label]
            ln += 1
            lid = f"{run_id}_l{ln}"
            messy_asp_lines.append(f'call("{run_id}", "{lid}", "llm_generate", "completed").')
            messy_asp_lines.append(f'depends("{run_id}", "{lid}", "{prev}").')
            sid = f"{run_id}_t{i}"
            messy_asp_lines.append(f'call("{run_id}", "{sid}", "{tool}", "completed").')
            messy_asp_lines.append(f'depends("{run_id}", "{sid}", "{lid}").')
            messy_asp_lines.append(f'spawned_by("{run_id}", "{sid}", "{lid}").')
            c = seg["char"]
            messy_asp_lines.append(f'param("{run_id}", "{sid}", "eff", "{c["efficiency"]:.1f}").')
            prev = sid
        ln += 1
        messy_asp_lines.append(f'call("{run_id}", "{run_id}_l{ln}", "llm_generate", "completed").')
        messy_asp_lines.append(f'depends("{run_id}", "{run_id}_l{ln}", "{prev}").')
        messy_asp_lines.append(f'call("{run_id}", "{run_id}_n", "notify_user", "completed").')
        messy_asp_lines.append(f'depends("{run_id}", "{run_id}_n", "{run_id}_l{ln}").')
        messy_asp_lines.append(f'spawned_by("{run_id}", "{run_id}_n", "{run_id}_l{ln}").')

    print(f"  {len(messy_asp_lines)} ASP facts for {len(messy_ids)} messy demos")

    # Run Clingo on each
    print("\nRunning ASP pattern miner on clean group...")
    clean_results = run_asp_compile(clean_asp_lines, rules_path)
    print(f"  Core tools: {len(clean_results['core_tools'])}")

    print("Running ASP pattern miner on messy group...")
    messy_results = run_asp_compile(messy_asp_lines, rules_path)
    print(f"  Core tools: {len(messy_results['core_tools'])}")

    # Diff
    diff = diff_results(clean_results, messy_results)

    print(f"\n{'='*70}")
    print("DISCOVERED DIFFERENCES (Clingo-derived rewrite rules)")
    print(f"{'='*70}")

    print(f"\nShared core primitives (appear in both clean and messy):")
    for t in diff["shared_core"]:
        c_count = clean_results["tool_counts"].get(t, 0)
        m_count = messy_results["tool_counts"].get(t, 0)
        print(f"  {t:30s}  clean: {c_count}/{len(clean_ids)}  messy: {m_count}/{len(messy_ids)}")

    if diff["clean_only_core"]:
        print(f"\nCore in CLEAN only (essential structure — messy demos lack these):")
        for t in diff["clean_only_core"]:
            count = clean_results["tool_counts"].get(t, 0)
            print(f"  {t:30s}  in {count}/{len(clean_ids)} clean demos")

    if diff["messy_only_core"]:
        print(f"\nCore in MESSY only (noise patterns — these are what to prune):")
        for t in diff["messy_only_core"]:
            count = messy_results["tool_counts"].get(t, 0)
            print(f"  {t:30s}  in {count}/{len(messy_ids)} messy demos")

    if diff["clean_only_order"]:
        print(f"\nConsistent orderings in CLEAN only (clean demos have this structure):")
        for t1, t2 in diff["clean_only_order"][:10]:
            print(f"  {t1} -> {t2}")

    if diff["messy_only_order"]:
        print(f"\nConsistent orderings in MESSY only (messy demos have extra sequencing):")
        for t1, t2 in diff["messy_only_order"][:10]:
            print(f"  {t1} -> {t2}")

    if diff["clean_only_fusion"]:
        print(f"\nFusion candidates in CLEAN only (clean demos fuse these):")
        for a, b in diff["clean_only_fusion"]:
            print(f"  {a} + {b}")

    if diff["messy_only_fusion"]:
        print(f"\nFusion candidates in MESSY only:")
        for a, b in diff["messy_only_fusion"]:
            print(f"  {a} + {b}")

    if diff["clean_only_exclusive"]:
        print(f"\nMutually exclusive in CLEAN only:")
        for a, b in diff["clean_only_exclusive"][:10]:
            print(f"  {a} XOR {b}")

    if diff["messy_only_exclusive"]:
        print(f"\nMutually exclusive in MESSY only:")
        for a, b in diff["messy_only_exclusive"][:10]:
            print(f"  {a} XOR {b}")

    # The key insight: conflicting orders
    print(f"\nConflicting orderings (where clean and messy disagree):")
    print(f"  Clean: {len(clean_results['conflicting_order'])} conflicts, "
          f"resolved: {len(clean_results.get('chosen_order', []))}")
    print(f"  Messy: {len(messy_results['conflicting_order'])} conflicts, "
          f"resolved: {len(messy_results.get('chosen_order', []))}")

    if messy_results['conflicting_order']:
        print(f"\n  Messy group conflicting orderings (evidence of fumbling):")
        for t1, t2 in sorted(messy_results['conflicting_order'])[:10]:
            print(f"    {t1} <-> {t2}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: DISCOVERED REWRITE RULES")
    print(f"{'='*70}")
    print(f"\nFrom comparing {len(clean_ids)} clean vs {len(messy_ids)} messy demos:")
    print(f"  - {len(diff['messy_only_core'])} noise primitive types "
          f"(core in messy, absent in clean)")
    print(f"  - {len(diff['clean_only_core'])} essential primitive types "
          f"(core in clean, absent in messy)")
    print(f"  - {len(diff['messy_only_order'])} extra sequencing patterns in messy demos")
    print(f"  - {len(messy_results['conflicting_order'])} conflicting orderings in messy "
          f"vs {len(clean_results['conflicting_order'])} in clean")
    print(f"\nRewrite rule: if a demo contains primitives from the 'messy-only' set,")
    print(f"those segments are candidates for pruning. The 'clean-only' primitives")
    print(f"define the essential structure that good demos share.")
    print(f"\nThis was discovered by Clingo, not hand-coded.")


if __name__ == "__main__":
    main()
