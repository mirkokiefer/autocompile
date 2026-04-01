#!/usr/bin/env python3
"""
segment_discover.py — Discover manipulation primitives from raw trajectories.

Instead of hardcoding segment labels (approach, grasp, etc.), this discovers
segments purely from state transitions. A "tool" is defined by what state
change it achieves, not by a human label.

Pipeline:
  1. Changepoint detection on raw trajectories (ruptures/velocity)
  2. Characterize each segment by its state transition (Δposition, Δgripper)
  3. Cluster similar state transitions across episodes
  4. Each cluster = a discovered primitive ("tool")
  5. Output ASP facts for autocompile

The key insight: noise (hesitation, shaking, suboptimal paths) shows up as
non-core segments that appear in few demos. The compiled program IS the
minimal consistent path from start state to end state.

Usage:
    python src/segment_discover.py \
        --dataset lerobot/ucsd_pick_and_place_dataset \
        --episodes 50 \
        --output examples/lerobot-discovered/traces.lp
"""

import argparse
import sys
from collections import defaultdict

try:
    import numpy as np
    from scipy.spatial.distance import cdist
except ImportError:
    print("ERROR: numpy and scipy required. Run: pip install numpy scipy")
    sys.exit(1)

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: datasets required. Run: pip install datasets")
    sys.exit(1)


# --- Changepoint detection (no external dependency) ---

def detect_changepoints(trajectory, min_segment_len=3, penalty_factor=1.0):
    """
    Detect changepoints using velocity-based cost function.
    Returns list of boundary indices.

    Uses a simplified PELT-like approach: find points where the
    trajectory's local behavior changes significantly.
    """
    n = len(trajectory)
    if n < min_segment_len * 2:
        return [0, n - 1]

    # Compute frame-to-frame velocity
    vel = np.diff(trajectory, axis=0)
    speed = np.linalg.norm(vel, axis=1)

    # Compute acceleration (change in velocity direction + magnitude)
    if len(vel) > 1:
        accel = np.diff(vel, axis=0)
        accel_mag = np.linalg.norm(accel, axis=1)
    else:
        return [0, n - 1]

    # Detect changepoints: frames where acceleration is unusually high
    # OR where speed crosses from high to low (or vice versa)
    median_accel = np.median(accel_mag)
    threshold = median_accel + penalty_factor * np.std(accel_mag)

    # Also detect gripper state changes (last dim of trajectory)
    gripper = trajectory[:, -1]
    gripper_vel = np.abs(np.diff(gripper))
    gripper_threshold = np.median(gripper_vel) + np.std(gripper_vel)

    boundaries = [0]
    for i in range(len(accel_mag)):
        frame = i + 1  # offset for diff
        if frame - boundaries[-1] < min_segment_len:
            continue
        if n - 1 - frame < min_segment_len:
            break

        is_accel_change = accel_mag[i] > threshold
        is_gripper_change = (i < len(gripper_vel) and
                             gripper_vel[i] > gripper_threshold)
        # Speed regime change: was fast, now slow (or vice versa)
        if i > 0 and i < len(speed) - 1:
            local_before = np.mean(speed[max(0, i-2):i+1])
            local_after = np.mean(speed[i+1:min(len(speed), i+4)])
            is_speed_regime = abs(local_before - local_after) > np.std(speed) * 0.5
        else:
            is_speed_regime = False

        if is_accel_change or is_gripper_change or is_speed_regime:
            boundaries.append(frame)

    boundaries.append(n - 1)
    return sorted(set(boundaries))


# --- State transition characterization ---

def characterize_segment(states, actions):
    """
    Characterize a segment purely by its state transition.
    Returns a feature vector describing WHAT the segment achieves.
    """
    start_state = states[0]
    end_state = states[-1]

    # Position change (first 3 dims)
    delta_pos = end_state[:3] - start_state[:3]
    pos_distance = np.linalg.norm(delta_pos)

    # Gripper change (last dim)
    delta_gripper = end_state[-1] - start_state[-1]

    # Direction of movement (normalized)
    if pos_distance > 0.01:
        direction = delta_pos / pos_distance
    else:
        direction = np.zeros(3)

    # Path efficiency: straight-line distance / actual path length
    path_lengths = np.linalg.norm(np.diff(states[:, :3], axis=0), axis=1)
    actual_path = np.sum(path_lengths)
    efficiency = pos_distance / max(actual_path, 0.001)

    # Duration and speed stats
    n_frames = len(states)
    speeds = path_lengths if len(path_lengths) > 0 else np.array([0])
    mean_speed = np.mean(speeds)
    speed_variance = np.var(speeds)

    return {
        "delta_pos": delta_pos,
        "pos_distance": pos_distance,
        "delta_gripper": delta_gripper,
        "direction": direction,
        "efficiency": efficiency,
        "n_frames": n_frames,
        "mean_speed": mean_speed,
        "speed_variance": speed_variance,
        "start_gripper": start_state[-1],
        "end_gripper": end_state[-1],
        # Feature vector for clustering
        "feature": np.array([
            delta_pos[0], delta_pos[1], delta_pos[2],
            delta_gripper / 8.0,  # normalize gripper to similar scale
            pos_distance,
            efficiency,
            mean_speed,
        ]),
    }


# --- Clustering state transitions ---

def cluster_transitions(all_segments, distance_threshold=0.3):
    """
    Cluster segment state transitions across all episodes.
    Uses simple agglomerative clustering based on feature similarity.

    Each cluster becomes a discovered "primitive" / "tool".
    """
    features = np.array([s["char"]["feature"] for s in all_segments])

    # Normalize features
    std = np.std(features, axis=0)
    std[std < 1e-8] = 1.0
    mean = np.mean(features, axis=0)
    features_norm = (features - mean) / std

    # Simple greedy clustering
    n = len(features_norm)
    labels = np.full(n, -1, dtype=int)
    cluster_centers = []
    cluster_id = 0

    for i in range(n):
        if labels[i] >= 0:
            continue

        # Start new cluster
        center = features_norm[i]
        members = [i]

        # Find all similar segments
        for j in range(i + 1, n):
            if labels[j] >= 0:
                continue
            dist = np.linalg.norm(features_norm[j] - center)
            if dist < distance_threshold:
                members.append(j)

        # Assign cluster
        for m in members:
            labels[m] = cluster_id

        # Recompute center
        cluster_centers.append(np.mean(features_norm[members], axis=0))
        cluster_id += 1

    return labels, cluster_id


def name_cluster(segments_in_cluster):
    """
    Generate a descriptive name for a cluster based on its state transitions.
    No hardcoded labels — names are derived from behavior.
    """
    chars = [s["char"] for s in segments_in_cluster]

    avg_delta_gripper = np.mean([c["delta_gripper"] for c in chars])
    avg_distance = np.mean([c["pos_distance"] for c in chars])
    avg_efficiency = np.mean([c["efficiency"] for c in chars])
    avg_speed = np.mean([c["mean_speed"] for c in chars])
    avg_direction = np.mean([c["direction"] for c in chars], axis=0)

    # Name based on dominant behavior
    if avg_delta_gripper < -1.5:
        name = "close_gripper"
    elif avg_delta_gripper > 1.5:
        name = "open_gripper"
    elif avg_distance > 0.3 and avg_speed > 0.15:
        # Moving significantly — name by dominant direction
        dominant = np.argmax(np.abs(avg_direction))
        sign = "pos" if avg_direction[dominant] > 0 else "neg"
        axis = ["x", "y", "z"][dominant]
        name = f"move_{axis}_{sign}"
    elif avg_distance > 0.1:
        name = "adjust_position"
    elif avg_speed < 0.03:
        name = "hold_steady"
    else:
        name = "micro_adjust"

    return name


# --- Main pipeline ---

def load_episodes(dataset_name, max_episodes):
    """Load episodes from HuggingFace."""
    print(f"Loading dataset: {dataset_name} (streaming)...")
    ds = load_dataset(dataset_name, split='train', streaming=True)

    episodes = defaultdict(list)
    for sample in ds:
        ep = sample['episode_index']
        if ep >= max_episodes:
            break
        episodes[ep].append(sample)

    print(f"Loaded {len(episodes)} episodes")
    return dict(episodes)


def process_episode(frames, penalty_factor=1.0):
    """Segment one episode and characterize each segment."""
    states = np.array([f['observation.state'] for f in frames])
    actions = np.array([f['action'] for f in frames])

    # Detect changepoints on full state vector
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
        })

    return segments


def segments_to_asp(episodes_segments, cluster_labels, cluster_names,
                    all_segments):
    """Convert clustered segments to ASP facts."""
    lines = []
    lines.append("% === LeRobot Pick-and-Place: Discovered Primitives ===")
    lines.append("% Segments discovered via changepoint detection + state-transition clustering")
    lines.append("% NO hardcoded labels — primitive names derived from behavior")
    lines.append("% Source: lerobot/ucsd_pick_and_place_dataset (HuggingFace)")
    lines.append("")

    seg_idx = 0
    for ep_idx, (ep_id, ep_segs) in enumerate(sorted(episodes_segments.items())):
        if not ep_segs:
            continue

        run_id = f"run_{ep_idx + 1}"
        lines.append(f"% === Episode {ep_id} ({len(ep_segs)} discovered segments) ===")
        lines.append(f'job("{run_id}").')
        lines.append(f'job_status("{run_id}", "completed").')

        # Glue: prompt_user + llm_generate
        lines.append(f'call("{run_id}", "{run_id}_step_0", "prompt_user", "completed").')
        lines.append(f'call("{run_id}", "{run_id}_step_1", "llm_generate", "completed").')
        lines.append(f'depends("{run_id}", "{run_id}_step_1", "{run_id}_step_0").')

        prev_step = f"{run_id}_step_1"
        llm_counter = 1

        for i, seg in enumerate(ep_segs):
            label = cluster_labels[seg_idx]
            tool_name = cluster_names[label]
            char = seg["char"]

            # LLM orchestrator
            llm_counter += 1
            llm_id = f"{run_id}_llm_{llm_counter}"
            lines.append(f'call("{run_id}", "{llm_id}", "llm_generate", "completed").')
            lines.append(f'depends("{run_id}", "{llm_id}", "{prev_step}").')

            # The discovered primitive
            step_id = f"{run_id}_step_{i + 2}"
            lines.append(f'call("{run_id}", "{step_id}", "{tool_name}", "completed").')
            lines.append(f'depends("{run_id}", "{step_id}", "{llm_id}").')
            lines.append(f'spawned_by("{run_id}", "{step_id}", "{llm_id}").')

            # Parameters from state transition characterization
            lines.append(f'param("{run_id}", "{step_id}", "pos_distance", "{char["pos_distance"]:.3f}").')
            lines.append(f'param("{run_id}", "{step_id}", "delta_gripper", "{char["delta_gripper"]:.2f}").')
            lines.append(f'param("{run_id}", "{step_id}", "efficiency", "{char["efficiency"]:.2f}").')
            lines.append(f'param("{run_id}", "{step_id}", "mean_speed", "{char["mean_speed"]:.4f}").')
            lines.append(f'param("{run_id}", "{step_id}", "n_frames", "{char["n_frames"]}").')
            lines.append(f'param("{run_id}", "{step_id}", "start_gripper", "{char["start_gripper"]:.1f}").')
            lines.append(f'param("{run_id}", "{step_id}", "end_gripper", "{char["end_gripper"]:.1f}").')

            prev_step = step_id
            seg_idx += 1

        # Glue: notify
        llm_counter += 1
        llm_end = f"{run_id}_llm_{llm_counter}"
        lines.append(f'call("{run_id}", "{llm_end}", "llm_generate", "completed").')
        lines.append(f'depends("{run_id}", "{llm_end}", "{prev_step}").')
        notify_id = f"{run_id}_notify"
        lines.append(f'call("{run_id}", "{notify_id}", "notify_user", "completed").')
        lines.append(f'depends("{run_id}", "{notify_id}", "{llm_end}").')
        lines.append(f'spawned_by("{run_id}", "{notify_id}", "{llm_end}").')
        lines.append("")

    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Discover manipulation primitives from raw trajectories")
    parser.add_argument("--dataset", default="lerobot/ucsd_pick_and_place_dataset")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--output", default="examples/lerobot-discovered/traces.lp")
    parser.add_argument("--cluster-threshold", type=float, default=0.8,
                        help="Distance threshold for clustering (higher = fewer clusters)")
    parser.add_argument("--penalty", type=float, default=1.0,
                        help="Changepoint detection sensitivity (lower = more segments)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Load
    episodes = load_episodes(args.dataset, args.episodes)

    # Segment each episode
    print("Segmenting trajectories...")
    episodes_segments = {}
    all_segments = []
    for ep_id, frames in sorted(episodes.items()):
        segs = process_episode(frames, penalty_factor=args.penalty)
        episodes_segments[ep_id] = segs
        all_segments.extend(segs)

    print(f"Total segments across all episodes: {len(all_segments)}")

    if not all_segments:
        print("ERROR: No segments found")
        sys.exit(1)

    # Cluster state transitions
    print("Clustering state transitions...")
    labels, n_clusters = cluster_transitions(all_segments,
                                             distance_threshold=args.cluster_threshold)

    # Name clusters based on behavior
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

    # Report
    print(f"\nDiscovered {n_clusters} primitive types:")
    for cid in sorted(cluster_members.keys()):
        members = cluster_members[cid]
        name = cluster_names[cid]
        chars = [m["char"] for m in members]
        avg_dist = np.mean([c["pos_distance"] for c in chars])
        avg_dg = np.mean([c["delta_gripper"] for c in chars])
        avg_eff = np.mean([c["efficiency"] for c in chars])
        avg_spd = np.mean([c["mean_speed"] for c in chars])
        ep_count = len(set(
            ep_id for ep_id, ep_segs in episodes_segments.items()
            for seg in ep_segs
            if any(np.array_equal(seg["char"]["feature"], m["char"]["feature"])
                   for m in members)
        ))
        print(f"  {name:25s}  count={len(members):3d}  "
              f"Δpos={avg_dist:.3f}  Δgrip={avg_dg:+.1f}  "
              f"eff={avg_eff:.2f}  spd={avg_spd:.3f}  "
              f"in {ep_count}/{len(episodes)} eps")

    if args.verbose:
        for ep_id, segs in sorted(episodes_segments.items()):
            print(f"\n  Episode {ep_id}:")
            ep_start_idx = sum(len(episodes_segments[eid])
                               for eid in sorted(episodes_segments.keys())
                               if eid < ep_id)
            for i, seg in enumerate(segs):
                cid = labels[ep_start_idx + i]
                name = cluster_names[cid]
                c = seg["char"]
                print(f"    {name:25s}  frames {seg['start_frame']:3d}-{seg['end_frame']:3d}  "
                      f"Δpos={c['pos_distance']:.3f}  Δgrip={c['delta_gripper']:+.1f}  "
                      f"eff={c['efficiency']:.2f}  spd={c['mean_speed']:.3f}")

    # Convert to ASP
    asp_lines = segments_to_asp(episodes_segments, labels, cluster_names,
                                all_segments)

    # Write
    from pathlib import Path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(__file__).parent.parent / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write("\n".join(asp_lines))

    print(f"\nWritten {len(asp_lines)} lines to {output_path}")


if __name__ == "__main__":
    main()
