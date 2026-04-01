#!/usr/bin/env python3
"""
Generate realistic pick-and-place robot demonstration traces.

Simulates 20 teleoperated demonstrations of a single-arm robot picking
objects from a workspace and placing them at target locations.

Demonstration structure:
  Phase 0: sense_object || sense_target  (parallel perception)
  Phase 1: move_arm                      (approach object)
  Phase 2: close_gripper                 (grasp)
  Phase 3: [check_grasp]                 (conditional verification)
           [adjust_grip]                 (rare recovery)
  Phase 4: move_to_target               (transport)
  Phase 5: open_gripper                 (release)
  Phase 6: retract_arm                  (retreat to home)

Expected discoveries by autocompile:
  - core_tool: sense_object, sense_target, move_arm, close_gripper,
               move_to_target, open_gripper, retract_arm
  - parallel_pair: (sense_object, sense_target)
  - fusion_candidate: (move_arm, close_gripper), (open_gripper, retract_arm)
  - stable_param: approach_height=0.15, gripper_force=0.8, retract_height=0.35, ...
  - variable_param: object_x, object_y, target_x, target_y
  - conditional_tool: check_grasp (40%), adjust_grip conditional on check_grasp
"""

import random
import sys

random.seed(42)

NUM_RUNS = 20

# Object positions (variable across demos)
OBJECTS = [
    ("0.25", "0.10"), ("0.30", "0.15"), ("0.20", "0.20"),
    ("0.35", "0.05"), ("0.28", "0.18"), ("0.22", "0.12"),
    ("0.32", "0.08"), ("0.26", "0.22"), ("0.18", "0.14"),
    ("0.34", "0.16"), ("0.24", "0.06"), ("0.29", "0.19"),
    ("0.21", "0.11"), ("0.33", "0.13"), ("0.27", "0.21"),
    ("0.23", "0.07"), ("0.31", "0.17"), ("0.19", "0.09"),
    ("0.36", "0.20"), ("0.25", "0.15"),
]

TARGETS = [
    ("0.50", "0.40"), ("0.55", "0.45"), ("0.48", "0.42"),
    ("0.52", "0.38"), ("0.50", "0.44"), ("0.53", "0.41"),
    ("0.49", "0.43"), ("0.51", "0.39"), ("0.54", "0.40"),
    ("0.50", "0.42"), ("0.52", "0.44"), ("0.48", "0.38"),
    ("0.55", "0.41"), ("0.50", "0.43"), ("0.53", "0.39"),
    ("0.51", "0.45"), ("0.49", "0.40"), ("0.54", "0.42"),
    ("0.52", "0.41"), ("0.50", "0.40"),
]

# Target zones (variable)
ZONES = ["zone_a", "zone_b", "zone_c"]

# Grip widths (variable per object size)
GRIP_WIDTHS = ["0.04", "0.05", "0.06", "0.03", "0.07"]


def generate_run(run_idx):
    """Generate one demonstration trace."""
    run_id = f"run_{run_idx}"
    step_num = 0
    lines = []

    has_check_grasp = run_idx in range(13, 21)  # runs 13-20: 40%
    has_adjust_grip = run_idx in [19, 20]  # runs 19-20: 10%
    obj_x, obj_y = OBJECTS[run_idx - 1]
    tgt_x, tgt_y = TARGETS[run_idx - 1]
    zone = ZONES[(run_idx - 1) % len(ZONES)]
    grip_w = GRIP_WIDTHS[(run_idx - 1) % len(GRIP_WIDTHS)]

    # Determine status — runs 7 and 14 fail (sensor or grasp failure)
    run_status = "completed"
    if run_idx == 7:
        run_status = "failed"
    if run_idx == 14:
        run_status = "failed"

    def sid():
        nonlocal step_num
        step_num += 1
        return f"{run_id}_step_{step_num}"

    def call(step_id, tool, status="completed"):
        lines.append(f'call("{run_id}", "{step_id}", "{tool}", "{status}").')

    def dep(step_id, dep_id):
        lines.append(f'depends("{run_id}", "{step_id}", "{dep_id}").')

    def spawn(step_id, parent_id):
        lines.append(f'spawned_by("{run_id}", "{step_id}", "{parent_id}").')

    def param(step_id, key, value):
        lines.append(f'param("{run_id}", "{step_id}", "{key}", "{value}").')

    lines.append(f'job("{run_id}").')
    lines.append(f'job_status("{run_id}", "{run_status}").')

    # Step 1: prompt_user (task input)
    s1 = sid()
    call(s1, "prompt_user")

    # Step 2: llm_generate (plan the task)
    s2 = sid()
    call(s2, "llm_generate")
    dep(s2, s1)

    # Step 3: sense_object (perceive object in workspace)
    s3 = sid()
    status = "failed" if run_idx == 7 else "completed"
    call(s3, "sense_object", status)
    dep(s3, s2)
    spawn(s3, s2)
    param(s3, "sensor_type", "rgbd")
    param(s3, "workspace_id", "ws_main")

    # Step 4: sense_target (perceive target location) — PARALLEL with sense_object
    s4 = sid()
    call(s4, "sense_target")
    dep(s4, s2)
    spawn(s4, s2)
    param(s4, "target_zone", zone)
    param(s4, "sensor_type", "rgbd")

    # If run 7, sensing failed — short run
    if run_idx == 7:
        s_llm = sid()
        call(s_llm, "llm_generate")
        dep(s_llm, s3)
        dep(s_llm, s4)
        s_notify = sid()
        call(s_notify, "notify_user")
        dep(s_notify, s_llm)
        spawn(s_notify, s_llm)
        return lines

    # Step 5: llm_generate (plan approach based on perception)
    s5 = sid()
    call(s5, "llm_generate")
    dep(s5, s3)
    dep(s5, s4)

    # Step 6: move_arm (approach object)
    s6 = sid()
    call(s6, "move_arm")
    dep(s6, s5)
    spawn(s6, s5)
    param(s6, "target_x", obj_x)
    param(s6, "target_y", obj_y)
    param(s6, "target_z", "0.15")
    param(s6, "approach_height", "0.15")
    param(s6, "speed", "0.5")

    # Step 7: llm_generate
    s7 = sid()
    call(s7, "llm_generate")
    dep(s7, s6)

    # Step 8: close_gripper
    s8 = sid()
    call(s8, "close_gripper")
    dep(s8, s7)
    spawn(s8, s7)
    param(s8, "gripper_force", "0.8")
    param(s8, "grip_width", grip_w)

    if has_check_grasp:
        # Step: llm_generate
        s_llm_cg = sid()
        call(s_llm_cg, "llm_generate")
        dep(s_llm_cg, s8)

        # Step: check_grasp
        s_cg = sid()
        cg_status = "completed"
        call(s_cg, "check_grasp", cg_status)
        dep(s_cg, s_llm_cg)
        spawn(s_cg, s_llm_cg)
        param(s_cg, "method", "force_feedback")
        param(s_cg, "threshold", "0.3")

        if has_adjust_grip:
            # Step: llm_generate
            s_llm_ag = sid()
            call(s_llm_ag, "llm_generate")
            dep(s_llm_ag, s_cg)

            # Step: adjust_grip
            s_ag = sid()
            call(s_ag, "adjust_grip")
            dep(s_ag, s_llm_ag)
            spawn(s_ag, s_llm_ag)
            param(s_ag, "adjustment", "tighten")
            param(s_ag, "force_delta", "0.1")

            s_prev = s_ag
        else:
            s_prev = s_cg

        # If run 14, grasp check "fails" — short run
        if run_idx == 14:
            s_llm_f = sid()
            call(s_llm_f, "llm_generate")
            dep(s_llm_f, s_prev)
            s_notify = sid()
            call(s_notify, "notify_user")
            dep(s_notify, s_llm_f)
            spawn(s_notify, s_llm_f)
            return lines
    else:
        s_prev = s8

    # Step: llm_generate (plan transport)
    s_llm_t = sid()
    call(s_llm_t, "llm_generate")
    dep(s_llm_t, s_prev)

    # Step: move_to_target
    s_mt = sid()
    call(s_mt, "move_to_target")
    dep(s_mt, s_llm_t)
    spawn(s_mt, s_llm_t)
    param(s_mt, "target_x", tgt_x)
    param(s_mt, "target_y", tgt_y)
    param(s_mt, "carry_height", "0.25")
    param(s_mt, "speed", "0.3")

    # Step: llm_generate
    s_llm_r = sid()
    call(s_llm_r, "llm_generate")
    dep(s_llm_r, s_mt)

    # Step: open_gripper
    s_og = sid()
    call(s_og, "open_gripper")
    dep(s_og, s_llm_r)
    spawn(s_og, s_llm_r)
    param(s_og, "release_speed", "0.2")

    # Step: llm_generate
    s_llm_ret = sid()
    call(s_llm_ret, "llm_generate")
    dep(s_llm_ret, s_og)

    # Step: retract_arm
    s_ra = sid()
    call(s_ra, "retract_arm")
    dep(s_ra, s_llm_ret)
    spawn(s_ra, s_llm_ret)
    param(s_ra, "retract_height", "0.35")
    param(s_ra, "speed", "0.4")

    # Step: notify_user
    s_llm_end = sid()
    call(s_llm_end, "llm_generate")
    dep(s_llm_end, s_ra)
    s_notify = sid()
    call(s_notify, "notify_user")
    dep(s_notify, s_llm_end)
    spawn(s_notify, s_llm_end)

    return lines


def main():
    output = []
    output.append("% === Robotics Pick-and-Place Demonstrations ===")
    output.append(f"% {NUM_RUNS} teleoperated demos of single-arm pick-and-place")
    output.append("% Generated by generate_traces.py")
    output.append("")

    for i in range(1, NUM_RUNS + 1):
        desc = "standard"
        if i in range(13, 21):
            desc = "with grasp check"
        if i in [19, 20]:
            desc = "with grasp check + adjust"
        if i == 7:
            desc = "sensing failed"
        if i == 14:
            desc = "grasp failed"
        output.append(f"% === Demo {i}: {desc} ===")
        output.extend(generate_run(i))
        output.append("")

    print("\n".join(output))


if __name__ == "__main__":
    main()
