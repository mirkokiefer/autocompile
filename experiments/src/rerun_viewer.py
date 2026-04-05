#!/usr/bin/env python3
"""
rerun_viewer.py — Visualize robot episodes with compilation claims.

Logs to Rerun:
  - Video frames (synchronized timeline)
  - Trajectory (arm position over time)
  - Gripper state (open/close curve)
  - Speed profile
  - Segment boundaries with labels
  - Clingo claims as text annotations
  - Rewrite analysis (essential vs prunable segments)

Usage:
    uv run python src/rerun_viewer.py --episodes 0 7 15 36
    uv run python src/rerun_viewer.py --episodes 0 1 2 3 4 5 6 7 8 9
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import rerun as rr

from segment_discover import detect_changepoints, characterize_segment
from trajectory_rewrite import (
    detect_noise_segments, detect_cancelling_pairs, detect_redundant_paths
)

# Video frames location
VIDEO_DIR = Path(__file__).parent.parent / "examples" / "lerobot-discovered" / "video"


def load_episode_data(dataset_name, episode_id):
    """Load one episode from HuggingFace."""
    from datasets import load_dataset
    ds = load_dataset(dataset_name, split='train', streaming=True)
    frames = []
    for sample in ds:
        if sample['episode_index'] == episode_id:
            frames.append(sample)
        elif sample['episode_index'] > episode_id:
            break
    return frames


def load_video_frames(episode_id):
    """Load extracted PNG frames for an episode."""
    frame_dir = VIDEO_DIR / f"ep_{episode_id}"
    if not frame_dir.exists():
        return []
    paths = sorted(frame_dir.glob("frame_*.png"))
    return paths


def analyze_episode(frames):
    """Segment and analyze one episode."""
    states = np.array([f['observation.state'] for f in frames])
    actions = np.array([f['action'] for f in frames])

    # Changepoint detection
    boundaries = detect_changepoints(states, min_segment_len=3, penalty_factor=0.8)

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

    # Label segments
    for seg in segments:
        c = seg["char"]
        if c["delta_gripper"] < -1.5:
            seg["label"] = "GRASP"
        elif c["delta_gripper"] > 1.5:
            seg["label"] = "RELEASE"
        elif c["pos_distance"] > 0.3 and c["mean_speed"] > 0.1:
            seg["label"] = "MOVE"
        elif c["pos_distance"] > 0.1:
            seg["label"] = "ADJUST"
        elif c["mean_speed"] < 0.03:
            seg["label"] = "HOLD"
        else:
            seg["label"] = "MICRO"

    # Rewrite analysis
    noise = detect_noise_segments(segments)
    cancelling = detect_cancelling_pairs(segments)
    redundant = detect_redundant_paths(segments)

    prunable = set()
    for n in noise:
        prunable.add(n["index"])
    for c in cancelling:
        for idx in range(c["start_idx"], c["end_idx"] + 1):
            prunable.add(idx)

    for i, seg in enumerate(segments):
        seg["essential"] = i not in prunable
        seg["prunable_reason"] = None
        for n in noise:
            if n["index"] == i:
                seg["prunable_reason"] = "noise (near-zero Δstate)"
        for c in cancelling:
            if c["start_idx"] <= i <= c["end_idx"]:
                seg["prunable_reason"] = f"cancelling (net Δ≈0 over segs {c['start_idx']}-{c['end_idx']})"

    return states, actions, segments, redundant


def log_episode(episode_id, frames, states, actions, segments, redundant,
                video_frame_paths):
    """Log one episode to Rerun."""
    entity_base = f"episode_{episode_id}"
    n_frames = len(frames)

    # Set time for each frame
    for frame_idx in range(n_frames):
        timestamp = frames[frame_idx]['timestamp']
        rr.set_time("frame", sequence=frame_idx)
        rr.set_time("time", duration=timestamp)

        # Video frame
        if frame_idx < len(video_frame_paths):
            rr.log(f"{entity_base}/video",
                   rr.EncodedImage(path=video_frame_paths[frame_idx]))

        # Arm position (first 3 state dims)
        pos = states[frame_idx, :3]
        rr.log(f"{entity_base}/trajectory",
               rr.Points3D([pos], radii=[0.02]))

        # Gripper state
        gripper = states[frame_idx, -1]
        rr.log(f"{entity_base}/gripper_state",
               rr.Scalars(gripper))

        # Speed (frame-to-frame)
        if frame_idx > 0:
            vel = np.linalg.norm(states[frame_idx, :3] - states[frame_idx - 1, :3])
            rr.log(f"{entity_base}/speed", rr.Scalars(vel))

        # Gripper command (action[-1])
        rr.log(f"{entity_base}/gripper_cmd",
               rr.Scalars(actions[min(frame_idx, len(actions) - 1), -1]))

        # Segment annotations
        for seg_idx, seg in enumerate(segments):
            if seg["start_frame"] <= frame_idx <= seg["end_frame"]:
                c = seg["char"]
                color = [0, 200, 0] if seg["essential"] else [200, 0, 0]
                status = "ESSENTIAL" if seg["essential"] else "PRUNABLE"
                reason = f" — {seg['prunable_reason']}" if seg["prunable_reason"] else ""

                rr.log(f"{entity_base}/segment_label",
                       rr.TextLog(
                           f"Seg {seg_idx}: {seg['label']} [{status}]{reason} | "
                           f"Δpos={c['pos_distance']:.3f} Δgrip={c['delta_gripper']:+.1f} "
                           f"eff={c['efficiency']:.2f} spd={c['mean_speed']:.3f}"))
                break

    # Log full trajectory as a line strip
    rr.set_time("frame", sequence=0)
    rr.log(f"{entity_base}/full_trajectory",
           rr.LineStrips3D([states[:, :3].tolist()],
                           colors=[[100, 100, 255]]))

    # Log segment boundaries as colored regions on trajectory
    for seg_idx, seg in enumerate(segments):
        start = seg["start_frame"]
        end = min(seg["end_frame"], n_frames - 1)
        seg_positions = states[start:end + 1, :3]
        if len(seg_positions) < 2:
            continue
        color = [0, 220, 0, 180] if seg["essential"] else [220, 0, 0, 180]
        rr.set_time("frame", sequence=start)
        rr.log(f"{entity_base}/segments/seg_{seg_idx}_{seg['label']}",
               rr.LineStrips3D([seg_positions.tolist()],
                               colors=[color],
                               radii=[0.01]))


def main():
    parser = argparse.ArgumentParser(description="Visualize episodes in Rerun")
    parser.add_argument("--dataset", default="lerobot/ucsd_pick_and_place_dataset")
    parser.add_argument("--episodes", nargs="+", type=int,
                        default=[0, 7, 15, 36],
                        help="Episode IDs to visualize")
    args = parser.parse_args()

    rr.init("autocompile_verify", spawn=True)

    # Log a text overview
    rr.log("README", rr.TextDocument("""# Autocompile Verification

## What to look at:

**Per episode:**
- `video`: Camera view of the robot
- `gripper_state`: Gripper position over time (high=open, low=closed)
- `gripper_cmd`: Gripper command from actions (positive=open, negative=close)
- `speed`: Frame-to-frame arm velocity
- `segment_label`: Current segment with ESSENTIAL/PRUNABLE classification
- `trajectory`: 3D arm position (dot)
- `full_trajectory`: Complete path (blue line)
- `segments/`: Colored path segments (green=essential, red=prunable)

## Episodes to verify:

- **Episode 0**: Clean pick-and-place (5 segments, all essential). Baseline.
- **Episode 7**: Fidgety (13 segments, 6 essential). Late segments should be red (micro-adjustments that cancel out).
- **Episode 15**: Wanderer (14 segments, 3 essential). Most segments should be red (arm wanders without picking).
- **Episode 36**: Clean with post-fidget (9 segments, 7 essential). Small red section near the end.

## Claims to verify:

1. GRASP segments always start with gripper_state ≈ 7.96 (fully open)
2. RELEASE segments always end with gripper_state ≈ 7.96
3. Red (prunable) segments in Ep 7 are genuinely unnecessary micro-adjustments
4. Red segments in Ep 15 are genuinely the arm wandering (no gripper action)
5. The ordering release → approach → grasp → transport matches what you see
""", media_type=rr.MediaType.MARKDOWN))

    for ep_id in args.episodes:
        print(f"\nLoading episode {ep_id}...")
        frames = load_episode_data(args.dataset, ep_id)
        if not frames:
            print(f"  Episode {ep_id} not found, skipping")
            continue

        video_frames = load_video_frames(ep_id)
        print(f"  {len(frames)} data frames, {len(video_frames)} video frames")

        states, actions, segments, redundant = analyze_episode(frames)
        n_essential = sum(1 for s in segments if s["essential"])
        n_prunable = sum(1 for s in segments if not s["essential"])
        print(f"  {len(segments)} segments: {n_essential} essential, {n_prunable} prunable")

        log_episode(ep_id, frames, states, actions, segments, redundant,
                    video_frames)

    print("\nRerun viewer should open. Check the README panel for what to verify.")


if __name__ == "__main__":
    main()
