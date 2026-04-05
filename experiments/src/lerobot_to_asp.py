#!/usr/bin/env python3
"""
lerobot_to_asp.py — Convert LeRobot datasets to ASP trace facts.

Loads a HuggingFace LeRobot dataset, segments continuous trajectories
into discrete manipulation primitives using gripper state changes and
velocity thresholds, then outputs ASP facts compatible with compile.py.

Segmentation strategy:
  1. Gripper transitions (open→close = grasp, close→open = release)
  2. Velocity drops between segments (movement phases)
  3. Labels: reach → approach → grasp → transport → release → retract

Usage:
    python src/lerobot_to_asp.py \
        --dataset lerobot/ucsd_pick_and_place_dataset \
        --episodes 50 \
        --output examples/lerobot-pick-place/traces.lp

Requires: pip install numpy scipy datasets huggingface_hub
"""

import argparse
import sys
from collections import defaultdict

try:
    import numpy as np
    from scipy.signal import find_peaks
except ImportError:
    print("ERROR: numpy and scipy required. Run: pip install numpy scipy")
    sys.exit(1)

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: datasets required. Run: pip install datasets huggingface_hub")
    sys.exit(1)


# Segmentation parameters
GRIPPER_CLOSE_THRESHOLD = 0.0   # action[3] < this = closing
SPEED_LOW_PERCENTILE = 25       # frames below this speed percentile = pauses
MIN_SEGMENT_FRAMES = 3          # ignore segments shorter than this


def load_episodes(dataset_name, max_episodes, action_dim=4, state_dim=7):
    """Load episodes from a LeRobot dataset on HuggingFace."""
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


def segment_episode(frames):
    """
    Segment a continuous trajectory into discrete manipulation primitives.

    Uses gripper state transitions as primary boundaries, velocity changes as secondary.
    Returns list of segments: [{"label": str, "start": int, "end": int, "params": dict}]
    """
    actions = np.array([f['action'] for f in frames])
    states = np.array([f['observation.state'] for f in frames])
    n_frames = len(frames)

    if n_frames < 5:
        return []

    # Gripper command is action[-1], gripper state is state[-1]
    gripper_cmd = actions[:, -1]
    gripper_state = states[:, -1]

    # Position is first 3 state dims
    pos = states[:, :3]
    vel = np.diff(pos, axis=0)
    speed = np.linalg.norm(vel, axis=1)

    # Find gripper transitions
    gripper_sign = np.sign(gripper_cmd)
    sign_changes = np.where(np.diff(gripper_sign) != 0)[0]

    # Build boundaries: [0, ...sign_changes..., n_frames-1]
    boundaries = [0]
    for sc in sign_changes:
        if sc - boundaries[-1] >= MIN_SEGMENT_FRAMES:
            boundaries.append(sc)
    if n_frames - 1 - boundaries[-1] >= MIN_SEGMENT_FRAMES:
        boundaries.append(n_frames - 1)
    elif boundaries[-1] != n_frames - 1:
        boundaries[-1] = n_frames - 1

    # Within each gripper phase, split on velocity drops
    refined_boundaries = [boundaries[0]]
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        seg_speed = speed[start:min(end, len(speed))]
        if len(seg_speed) > MIN_SEGMENT_FRAMES * 2:
            threshold = np.percentile(seg_speed, SPEED_LOW_PERCENTILE)
            # Find sustained low-speed regions (pauses)
            low_speed = seg_speed < threshold
            # Look for transitions from fast to slow
            for j in range(1, len(low_speed) - 1):
                if low_speed[j] and not low_speed[j-1]:
                    frame_idx = start + j
                    if (frame_idx - refined_boundaries[-1] >= MIN_SEGMENT_FRAMES and
                        end - frame_idx >= MIN_SEGMENT_FRAMES):
                        refined_boundaries.append(frame_idx)
                        break  # Only one split per gripper phase
        refined_boundaries.append(end)

    # Remove duplicates and sort
    refined_boundaries = sorted(set(refined_boundaries))

    # Label each segment
    segments = []
    for i in range(len(refined_boundaries) - 1):
        start = refined_boundaries[i]
        end = refined_boundaries[i + 1]
        if end - start < MIN_SEGMENT_FRAMES:
            continue

        seg_actions = actions[start:end]
        seg_states = states[start:end]
        seg_gripper_cmd = gripper_cmd[start:end]
        seg_speed = speed[start:min(end, len(speed))]

        # Determine label from gripper state and position
        gripper_closing = np.mean(seg_gripper_cmd) < GRIPPER_CLOSE_THRESHOLD
        gripper_was_open = gripper_state[start] > np.median(gripper_state)
        gripper_was_closed = gripper_state[start] < np.median(gripper_state)

        # Mean speed for the segment
        mean_speed = np.mean(seg_speed) if len(seg_speed) > 0 else 0

        # Is gripper transitioning?
        g_start = gripper_state[start]
        g_end = gripper_state[min(end, len(gripper_state) - 1)]
        g_delta = g_end - g_start

        if g_delta < -1.0:
            label = "grasp"
        elif g_delta > 1.0:
            label = "release"
        elif gripper_closing and mean_speed > 0.05:
            label = "transport"
        elif gripper_closing and mean_speed <= 0.05:
            label = "hold"
        elif not gripper_closing and i == 0:
            label = "reach"
        elif not gripper_closing and mean_speed > 0.05:
            label = "move"
        else:
            label = "idle"

        # Extract parameters
        start_pos = seg_states[0, :3]
        end_pos = seg_states[-1, :3]
        duration = frames[min(end, n_frames-1)]['timestamp'] - frames[start]['timestamp']

        params = {
            "start_x": f"{start_pos[0]:.3f}",
            "start_y": f"{start_pos[1]:.3f}",
            "start_z": f"{start_pos[2]:.3f}",
            "end_x": f"{end_pos[0]:.3f}",
            "end_y": f"{end_pos[1]:.3f}",
            "end_z": f"{end_pos[2]:.3f}",
            "duration_sec": f"{duration:.2f}",
            "mean_speed": f"{mean_speed:.4f}",
            "gripper_start": f"{g_start:.2f}",
            "gripper_end": f"{g_end:.2f}",
        }

        segments.append({
            "label": label,
            "start": start,
            "end": end,
            "params": params,
        })

    # Post-process: merge consecutive same-label segments
    if len(segments) > 1:
        merged = [segments[0]]
        for seg in segments[1:]:
            if seg["label"] == merged[-1]["label"]:
                merged[-1]["end"] = seg["end"]
                merged[-1]["params"]["end_x"] = seg["params"]["end_x"]
                merged[-1]["params"]["end_y"] = seg["params"]["end_y"]
                merged[-1]["params"]["end_z"] = seg["params"]["end_z"]
                merged[-1]["params"]["duration_sec"] = f"{float(merged[-1]['params']['duration_sec']) + float(seg['params']['duration_sec']):.2f}"
            else:
                merged.append(seg)
        segments = merged

    # Relabel based on position in sequence for cleaner primitives
    has_grasp = any(s["label"] == "grasp" for s in segments)
    has_release = any(s["label"] == "release" for s in segments)
    if has_grasp:
        grasp_idx = next(i for i, s in enumerate(segments) if s["label"] == "grasp")
        # Everything before grasp that's moving = approach
        for i in range(grasp_idx):
            if segments[i]["label"] in ("reach", "move"):
                segments[i]["label"] = "approach"
        # First moving segment after grasp = transport
        for i in range(grasp_idx + 1, len(segments)):
            if segments[i]["label"] in ("move", "transport"):
                segments[i]["label"] = "transport"
                break
        # After release = retract
        if has_release:
            release_idx = next(i for i, s in enumerate(segments) if s["label"] == "release")
            for i in range(release_idx + 1, len(segments)):
                if segments[i]["label"] in ("move", "idle"):
                    segments[i]["label"] = "retract"

    return segments


def episodes_to_asp(episodes, task_labels=None):
    """Convert segmented episodes to ASP facts."""
    lines = []
    lines.append("% === LeRobot Pick-and-Place: Real Robot Demonstrations ===")
    lines.append(f"% {len(episodes)} episodes converted from continuous trajectories")
    lines.append("% Segmentation: gripper transitions + velocity changepoints")
    lines.append("% Source: lerobot/ucsd_pick_and_place_dataset (HuggingFace)")
    lines.append("")

    for ep_idx, (ep_id, frames) in enumerate(sorted(episodes.items())):
        segments = segment_episode(frames)
        if not segments:
            continue

        run_id = f"run_{ep_idx + 1}"
        task_idx = frames[0]['task_index']
        task_label = task_labels.get(task_idx, f"task_{task_idx}") if task_labels else f"task_{task_idx}"

        lines.append(f"% === Episode {ep_id} ({task_label}, {len(segments)} segments) ===")
        lines.append(f'job("{run_id}").')
        lines.append(f'job_status("{run_id}", "completed").')

        # Glue: prompt_user + llm_generate at start
        lines.append(f'call("{run_id}", "{run_id}_step_0", "prompt_user", "completed").')
        lines.append(f'call("{run_id}", "{run_id}_step_1", "llm_generate", "completed").')
        lines.append(f'depends("{run_id}", "{run_id}_step_1", "{run_id}_step_0").')

        prev_step = f"{run_id}_step_1"
        llm_step_counter = 1

        for i, seg in enumerate(segments):
            # LLM orchestrator between each primitive
            llm_step_counter += 1
            llm_id = f"{run_id}_llm_{llm_step_counter}"
            lines.append(f'call("{run_id}", "{llm_id}", "llm_generate", "completed").')
            lines.append(f'depends("{run_id}", "{llm_id}", "{prev_step}").')

            # The actual primitive
            step_id = f"{run_id}_step_{i + 2}"
            lines.append(f'call("{run_id}", "{step_id}", "{seg["label"]}", "completed").')
            lines.append(f'depends("{run_id}", "{step_id}", "{llm_id}").')
            lines.append(f'spawned_by("{run_id}", "{step_id}", "{llm_id}").')

            for key, value in seg["params"].items():
                lines.append(f'param("{run_id}", "{step_id}", "{key}", "{value}").')

            prev_step = step_id

        # Glue: notify at end
        llm_step_counter += 1
        llm_end = f"{run_id}_llm_{llm_step_counter}"
        lines.append(f'call("{run_id}", "{llm_end}", "llm_generate", "completed").')
        lines.append(f'depends("{run_id}", "{llm_end}", "{prev_step}").')
        notify_id = f"{run_id}_notify"
        lines.append(f'call("{run_id}", "{notify_id}", "notify_user", "completed").')
        lines.append(f'depends("{run_id}", "{notify_id}", "{llm_end}").')
        lines.append(f'spawned_by("{run_id}", "{notify_id}", "{llm_end}").')
        lines.append("")

    return lines


def main():
    parser = argparse.ArgumentParser(description="Convert LeRobot dataset to ASP trace facts")
    parser.add_argument("--dataset", default="lerobot/ucsd_pick_and_place_dataset",
                        help="HuggingFace dataset name")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Number of episodes to convert")
    parser.add_argument("--output", default="examples/lerobot-pick-place/traces.lp",
                        help="Output ASP file")
    parser.add_argument("--verbose", action="store_true",
                        help="Print segment details for each episode")
    args = parser.parse_args()

    # Load
    episodes = load_episodes(args.dataset, args.episodes)

    # Task labels for UCSD dataset
    task_labels = {0: "pick_and_place", 1: "pick_and_place", 2: "pick_and_place"}

    if args.verbose:
        for ep_id, frames in sorted(episodes.items()):
            segments = segment_episode(frames)
            print(f"\nEpisode {ep_id} ({len(frames)} frames → {len(segments)} segments):")
            for s in segments:
                print(f"  {s['label']:12s} frames {s['start']:3d}-{s['end']:3d} "
                      f"speed={s['params']['mean_speed']} "
                      f"gripper={s['params']['gripper_start']}→{s['params']['gripper_end']}")

    # Convert
    asp_lines = episodes_to_asp(episodes, task_labels)

    # Write
    from pathlib import Path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(__file__).parent.parent / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write("\n".join(asp_lines))

    print(f"\nWritten {len(asp_lines)} lines to {output_path}")
    print(f"Episodes: {len(episodes)}")

    # Summary stats
    all_segments = []
    for ep_id, frames in episodes.items():
        all_segments.extend(segment_episode(frames))
    labels = [s["label"] for s in all_segments]
    from collections import Counter
    counts = Counter(labels)
    print(f"\nSegment labels:")
    for label, count in counts.most_common():
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
