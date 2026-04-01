#!/usr/bin/env python3
"""
world_model.py — LLM-powered world model for single-demo compilation.

Takes segmented trajectory data (and optionally video frames) and queries
an LLM to produce structured world knowledge: invariants, variations,
failure modes, and synthetic trace specifications.

Supports OpenRouter API (Claude, Qwen, etc.) or falls back to
hardcoded heuristics for offline use.

Usage:
    from world_model import WorldModel

    wm = WorldModel(api_key="sk-or-...", model="qwen/qwen3-235b-a22b")
    knowledge = wm.query(segments, video_frames=None)
"""

import json
import os
import sys
import base64
from pathlib import Path

try:
    import requests
except ImportError:
    requests = None


WORLD_MODEL_PROMPT = """You are analyzing a robot demonstration trajectory. Given the segmented state transitions below, produce a structured world model.

## Trajectory Segments

{segments_text}

## Task

Analyze these state transitions and produce a JSON object with this exact structure:

```json
{{
  "task": {{
    "description": "One sentence describing what task this trajectory performs"
  }},
  "invariants": [
    {{
      "name": "short_name",
      "rule": "Physical constraint that must hold in ANY valid execution of this task",
      "always_before": "primitive that must come before another (or null)",
      "always_after": "primitive that must come after another (or null)"
    }}
  ],
  "variations": [
    {{
      "name": "what_varies",
      "description": "What could be different in another execution",
      "affected_primitives": ["list of primitive names affected"],
      "param_ranges": {{"param_name": [min, max]}}
    }}
  ],
  "failure_modes": [
    {{
      "name": "failure_type",
      "description": "What could go wrong",
      "probability": 0.1,
      "recovery_primitives": ["list of primitives for recovery"],
      "inserts_after": "primitive name after which recovery would be needed"
    }}
  ],
  "can_skip": [
    {{
      "primitive": "primitive_name",
      "condition": "When this primitive can be skipped entirely"
    }}
  ]
}}
```

Be specific and grounded in physics. Only include invariants you're confident about from the trajectory data. For failure modes, think about what commonly goes wrong in manipulation tasks.

Respond with ONLY the JSON object, no markdown fences or explanation."""


def format_segments_for_prompt(segments):
    """Format segment data for the LLM prompt."""
    lines = []
    for i, seg in enumerate(segments):
        c = seg["char"]
        # Determine a descriptive label from the data
        if c["delta_gripper"] < -1.5:
            label = "CLOSE_GRIPPER"
        elif c["delta_gripper"] > 1.5:
            label = "OPEN_GRIPPER"
        elif c["pos_distance"] > 0.3 and c["mean_speed"] > 0.1:
            label = "LARGE_MOVEMENT"
        elif c["pos_distance"] > 0.1:
            label = "SMALL_ADJUSTMENT"
        else:
            label = "HOLD/PAUSE"

        lines.append(
            f"Segment {i} [{label}]: "
            f"position_change={c['pos_distance']:.3f}, "
            f"gripper_change={c['delta_gripper']:+.1f}, "
            f"path_efficiency={c['efficiency']:.2f}, "
            f"speed={c['mean_speed']:.3f}, "
            f"gripper_start={c['start_gripper']:.1f}, "
            f"gripper_end={c['end_gripper']:.1f}, "
            f"frames={c['n_frames']}"
        )
    return "\n".join(lines)


class WorldModel:
    """LLM-powered world model for trajectory analysis."""

    def __init__(self, api_key=None, model=None, base_url=None):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.model = model or "qwen/qwen3-235b-a22b"
        self.base_url = base_url or "https://openrouter.ai/api/v1"

    def query(self, segments, video_frames=None):
        """
        Query the LLM world model.

        Args:
            segments: List of segment dicts with "char" field
            video_frames: Optional list of image file paths

        Returns:
            Structured world model dict
        """
        if not self.api_key:
            print("  [No API key — using offline heuristic world model]")
            return self._offline_heuristic(segments)

        segments_text = format_segments_for_prompt(segments)
        prompt = WORLD_MODEL_PROMPT.format(segments_text=segments_text)

        # Build messages
        content = []

        # Add video frames if provided (for vision models)
        if video_frames:
            for frame_path in video_frames[:4]:  # Max 4 frames
                try:
                    with open(frame_path, "rb") as f:
                        img_data = base64.b64encode(f.read()).decode()
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_data}"
                        }
                    })
                except Exception as e:
                    print(f"  Warning: couldn't load frame {frame_path}: {e}")

        content.append({"type": "text", "text": prompt})

        print(f"  Querying {self.model} via OpenRouter...")

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": content}],
                    "temperature": 0.3,
                    "max_tokens": 2000,
                },
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()

            text = data["choices"][0]["message"]["content"]

            # Strip markdown fences if present
            text = text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                if text.endswith("```"):
                    text = text[:-3]
                elif "```" in text:
                    text = text[:text.rfind("```")]
            text = text.strip()

            # Strip <think> tags if present (Qwen reasoning)
            if "<think>" in text:
                think_end = text.find("</think>")
                if think_end != -1:
                    text = text[think_end + len("</think>"):].strip()

            result = json.loads(text)
            print(f"  Got world model from {self.model}")
            return self._normalize(result, segments)

        except requests.exceptions.RequestException as e:
            print(f"  API error: {e}")
            print(f"  Falling back to offline heuristic")
            return self._offline_heuristic(segments)
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"  Parse error: {e}")
            print(f"  Raw response: {text[:500]}")
            print(f"  Falling back to offline heuristic")
            return self._offline_heuristic(segments)

    def _normalize(self, result, segments):
        """Normalize LLM output to expected schema."""
        normalized = {
            "task": result.get("task", {"description": "Unknown task"}),
            "invariants": result.get("invariants", []),
            "variations": result.get("variations", []),
            "failure_modes": result.get("failure_modes", []),
            "can_skip": result.get("can_skip", []),
            "source": "llm",
            "model": self.model,
        }
        return normalized

    def _offline_heuristic(self, segments):
        """Fallback: generate world model from trajectory statistics alone."""
        has_gripper_close = any(s["char"]["delta_gripper"] < -1.5 for s in segments)
        has_gripper_open = any(s["char"]["delta_gripper"] > 1.5 for s in segments)
        has_large_move = any(s["char"]["pos_distance"] > 0.5 for s in segments)

        invariants = []
        failure_modes = []

        if has_gripper_close and has_gripper_open:
            invariants.append({
                "name": "grasp_release_order",
                "rule": "Gripper close must precede gripper open in a pick-place task",
            })
            failure_modes.append({
                "name": "missed_grasp",
                "description": "Gripper closes but misses the object",
                "probability": 0.15,
                "recovery_primitives": ["open_gripper", "adjust_position", "close_gripper"],
                "inserts_after": "close_gripper",
            })

        if has_large_move:
            invariants.append({
                "name": "move_requires_grasp",
                "rule": "Large movement while holding requires prior successful grasp",
            })
            failure_modes.append({
                "name": "object_slip",
                "description": "Object drops during transport",
                "probability": 0.05,
                "recovery_primitives": ["adjust_position", "close_gripper", "move_to_position"],
                "inserts_after": "move_to_position",
            })

        variations = [
            {
                "name": "position_variation",
                "description": "Object/target position varies",
                "affected_primitives": ["move_to_position", "adjust_position"],
                "param_ranges": {"pos_distance": [0.1, 3.0]},
            },
            {
                "name": "speed_variation",
                "description": "Execution speed varies",
                "affected_primitives": ["move_to_position", "adjust_position"],
                "param_ranges": {"mean_speed": [0.03, 0.4]},
            },
        ]

        return {
            "task": {"description": "Manipulation task (inferred from trajectory statistics)"},
            "invariants": invariants,
            "variations": variations,
            "failure_modes": failure_modes,
            "can_skip": [],
            "source": "offline_heuristic",
        }


# =============================================================================
# Synthetic trace generation from world model
# =============================================================================

def generate_from_world_model(real_tools, world_model, n_variations=30):
    """
    Generate synthetic trace variations using LLM world model knowledge.

    Uses the structured failure_modes and variations from the LLM to
    create realistic synthetic demonstrations.
    """
    import random
    random.seed(42)
    traces = []

    # Type 1: Normal variations (vary parameters within ranges)
    n_normal = int(n_variations * 0.55)
    for i in range(n_normal):
        steps = []
        for rt in real_tools:
            step = {"tool": rt["tool"], "params": dict(rt["params"])}
            # Apply variation ranges from world model
            for var in world_model.get("variations", []):
                if rt["tool"] in var.get("affected_primitives", []):
                    for param, (lo, hi) in var.get("param_ranges", {}).items():
                        if param in step["params"]:
                            step["params"][param] = round(
                                random.uniform(lo, hi), 3)
            # Also add general noise
            if "pos_distance" in step["params"]:
                step["params"]["pos_distance"] *= random.uniform(0.5, 1.5)
            if "mean_speed" in step["params"]:
                step["params"]["mean_speed"] *= random.uniform(0.7, 1.3)
            steps.append(step)
        traces.append({"type": "normal_variation", "steps": steps})

    # Type 2: Failure mode recovery traces
    for fm in world_model.get("failure_modes", []):
        n_fm = max(1, int(n_variations * fm.get("probability", 0.1)))
        for i in range(n_fm):
            steps = []
            recovery_inserted = False
            for rt in real_tools:
                step = {"tool": rt["tool"], "params": dict(rt["params"])}
                # Add parameter noise
                if "pos_distance" in step["params"]:
                    step["params"]["pos_distance"] *= random.uniform(0.6, 1.4)
                steps.append(step)

                # Insert recovery after the specified primitive
                trigger = fm.get("inserts_after", "")
                if rt["tool"] == trigger and not recovery_inserted:
                    recovery_inserted = True
                    for rec_tool in fm.get("recovery_primitives", []):
                        # Generate recovery step with reasonable params
                        rec_params = dict(rt["params"])
                        rec_params["pos_distance"] = round(
                            random.uniform(0.05, 0.3), 3)
                        rec_params["efficiency"] = round(
                            random.uniform(0.2, 0.5), 2)
                        if "gripper" in rec_tool:
                            rec_params["delta_gripper"] = (
                                3.0 if "open" in rec_tool else -2.0)
                        else:
                            rec_params["delta_gripper"] = 0.0
                        steps.append({
                            "tool": rec_tool,
                            "params": rec_params,
                        })

            traces.append({"type": f"recovery_{fm['name']}", "steps": steps})

    # Type 3: Short executions (some steps skippable)
    n_short = max(1, int(n_variations * 0.15))
    for i in range(n_short):
        skippable = {s.get("primitive", "") for s in world_model.get("can_skip", [])}
        steps = []
        for rt in real_tools:
            if rt["tool"] in skippable and random.random() < 0.3:
                continue  # Skip this step
            step = {"tool": rt["tool"], "params": dict(rt["params"])}
            step["params"]["pos_distance"] = round(
                step["params"].get("pos_distance", 0.1) * random.uniform(0.3, 0.7), 3)
            steps.append(step)
        if steps:  # Don't add empty traces
            traces.append({"type": "short_execution", "steps": steps})

    return traces


if __name__ == "__main__":
    # Quick test
    test_segments = [
        {"char": {"pos_distance": 0.5, "delta_gripper": 1.6, "efficiency": 1.0,
                  "mean_speed": 0.15, "start_gripper": 2.0, "end_gripper": 7.9,
                  "n_frames": 5, "direction": [1, 0, 0], "speed_variance": 0.01,
                  "delta_pos": [0.4, 0.1, 0.0]}},
        {"char": {"pos_distance": 0.4, "delta_gripper": -2.5, "efficiency": 0.9,
                  "mean_speed": 0.12, "start_gripper": 7.9, "end_gripper": 0.5,
                  "n_frames": 4, "direction": [0, -1, 0], "speed_variance": 0.02,
                  "delta_pos": [0.0, -0.3, 0.0]}},
        {"char": {"pos_distance": 1.2, "delta_gripper": 0.0, "efficiency": 0.85,
                  "mean_speed": 0.2, "start_gripper": 0.5, "end_gripper": 0.5,
                  "n_frames": 8, "direction": [-1, 0, 0], "speed_variance": 0.03,
                  "delta_pos": [-1.0, 0.0, 0.0]}},
    ]

    api_key = os.environ.get("OPENROUTER_API_KEY")
    wm = WorldModel(api_key=api_key)
    result = wm.query(test_segments)
    print(json.dumps(result, indent=2))
