#!/usr/bin/env python3
"""
Generate distributed edge compute pipeline traces.

Simulates 25 runs of an anomaly detection pipeline across a heterogeneous
grid of devices: ESP32 sensors, Raspberry Pi, NVIDIA Jetson, Mac Studio.

Pipeline structure:
  Phase 0: read_sensor (x3 ESP32s, parallel)
  Phase 1: preprocess_data (on Pi or Mac Studio)
  Phase 2: run_inference (always on Jetson)
  Phase 3: aggregate_results (on Mac Studio)
  Phase 4: [send_alert]  (conditional — only on anomaly detection)
  Phase 4: store_results  (always — to persistent storage)

Expected discoveries by autocompile:
  - core_tool: read_sensor, preprocess_data, run_inference,
               aggregate_results, store_results
  - parallel_pair: multiple read_sensor calls
  - fusion_candidate: (aggregate_results, store_results) — always sequential
  - stable_param: device affinity (run_inference -> jetson_1),
                  model=yolov8n, sensor_type=temperature
  - variable_param: threshold, batch_id
  - conditional_tool: send_alert (~36%), recalibrate_sensor (~16%)
"""

import random

random.seed(42)

NUM_RUNS = 25

# Thresholds vary per pipeline run
THRESHOLDS = ["0.85", "0.90", "0.80", "0.75", "0.95", "0.88", "0.82", "0.92"]

# Batch IDs
BATCH_IDS = [f"batch_{i:04d}" for i in range(1, NUM_RUNS + 1)]

# Device assignments for preprocessing (varies — load balanced)
PREPROCESS_DEVICES = [
    "pi_1", "mac_studio_1", "pi_1", "pi_1", "mac_studio_1",
    "pi_1", "mac_studio_1", "pi_1", "pi_1", "mac_studio_1",
    "mac_studio_1", "pi_1", "pi_1", "mac_studio_1", "pi_1",
    "mac_studio_1", "pi_1", "pi_1", "mac_studio_1", "pi_1",
    "mac_studio_1", "pi_1", "mac_studio_1", "pi_1", "pi_1",
]


def generate_run(run_idx):
    """Generate one pipeline execution trace."""
    run_id = f"run_{run_idx}"
    step_num = 0
    lines = []

    threshold = THRESHOLDS[(run_idx - 1) % len(THRESHOLDS)]
    batch_id = BATCH_IDS[run_idx - 1]
    preprocess_device = PREPROCESS_DEVICES[run_idx - 1]

    # Anomaly detected in ~36% of runs (9/25)
    has_anomaly = run_idx in [2, 5, 8, 11, 14, 17, 20, 23, 25]
    # Sensor recalibration needed in ~16% (4/25)
    needs_recalibrate = run_idx in [3, 10, 16, 22]
    # Run 6: sensor failure, run 19: inference timeout
    run_status = "completed"
    fail_at = None
    if run_idx == 6:
        run_status = "failed"
        fail_at = "sensor"
    if run_idx == 19:
        run_status = "failed"
        fail_at = "inference"

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

    # Step 1: prompt_user (pipeline trigger)
    s1 = sid()
    call(s1, "prompt_user")

    # Step 2: llm_generate (orchestrate pipeline)
    s2 = sid()
    call(s2, "llm_generate")
    dep(s2, s1)

    # Steps 3-5: read_sensor on three ESP32s (PARALLEL)
    sensor_steps = []
    sensors = [
        ("esp32_1", "temperature"),
        ("esp32_2", "vibration"),
        ("esp32_3", "pressure"),
    ]
    for device, sensor_type in sensors:
        s = sid()
        status = "failed" if (fail_at == "sensor" and device == "esp32_2") else "completed"
        call(s, "read_sensor", status)
        dep(s, s2)
        spawn(s, s2)
        param(s, "device", device)
        param(s, "sensor_type", sensor_type)
        param(s, "sample_rate_hz", "100")
        param(s, "duration_sec", "10")
        sensor_steps.append(s)

    if fail_at == "sensor":
        s_llm = sid()
        call(s_llm, "llm_generate")
        for ss in sensor_steps:
            dep(s_llm, ss)
        s_notify = sid()
        call(s_notify, "notify_user")
        dep(s_notify, s_llm)
        spawn(s_notify, s_llm)
        return lines

    # Step 6: llm_generate (sensors done)
    s6 = sid()
    call(s6, "llm_generate")
    for ss in sensor_steps:
        dep(s6, ss)

    # Step 7: preprocess_data (on Pi or Mac Studio — varies)
    s7 = sid()
    call(s7, "preprocess_data")
    dep(s7, s6)
    spawn(s7, s6)
    param(s7, "device", preprocess_device)
    param(s7, "batch_id", batch_id)
    param(s7, "normalize", "true")
    param(s7, "window_size", "256")

    # Step 8: llm_generate
    s8 = sid()
    call(s8, "llm_generate")
    dep(s8, s7)

    # Step 9: run_inference (ALWAYS on Jetson)
    s9 = sid()
    status = "failed" if fail_at == "inference" else "completed"
    call(s9, "run_inference", status)
    dep(s9, s8)
    spawn(s9, s8)
    param(s9, "device", "jetson_1")
    param(s9, "model", "anomaly_detector_v3")
    param(s9, "threshold", threshold)
    param(s9, "batch_id", batch_id)
    param(s9, "precision", "fp16")

    if fail_at == "inference":
        s_llm = sid()
        call(s_llm, "llm_generate")
        dep(s_llm, s9)
        s_notify = sid()
        call(s_notify, "notify_user")
        dep(s_notify, s_llm)
        spawn(s_notify, s_llm)
        return lines

    # Step 10: llm_generate
    s10 = sid()
    call(s10, "llm_generate")
    dep(s10, s9)

    # Step 11: aggregate_results (on Mac Studio)
    s11 = sid()
    call(s11, "aggregate_results")
    dep(s11, s10)
    spawn(s11, s10)
    param(s11, "device", "mac_studio_1")
    param(s11, "batch_id", batch_id)
    param(s11, "output_format", "json")

    # Step 12: llm_generate
    s12 = sid()
    call(s12, "llm_generate")
    dep(s12, s11)

    # Step 13: store_results (always)
    s13 = sid()
    call(s13, "store_results")
    dep(s13, s12)
    spawn(s13, s12)
    param(s13, "device", "mac_studio_1")
    param(s13, "storage", "timescaledb")
    param(s13, "batch_id", batch_id)
    param(s13, "retention_days", "90")

    if has_anomaly:
        # Step: llm_generate
        s_llm_alert = sid()
        call(s_llm_alert, "llm_generate")
        dep(s_llm_alert, s13)

        # Step: send_alert
        s_alert = sid()
        call(s_alert, "send_alert")
        dep(s_alert, s_llm_alert)
        spawn(s_alert, s_llm_alert)
        param(s_alert, "channel", "ops_slack")
        param(s_alert, "severity", "warning")
        param(s_alert, "batch_id", batch_id)

        s_prev = s_alert
    else:
        s_prev = s13

    if needs_recalibrate:
        # Step: llm_generate
        s_llm_rc = sid()
        call(s_llm_rc, "llm_generate")
        dep(s_llm_rc, s_prev)

        # Step: recalibrate_sensor
        s_rc = sid()
        call(s_rc, "recalibrate_sensor")
        dep(s_rc, s_llm_rc)
        spawn(s_rc, s_llm_rc)
        param(s_rc, "device", "esp32_2")
        param(s_rc, "method", "baseline_reset")

        s_prev = s_rc

    # Final: notify_user
    s_llm_end = sid()
    call(s_llm_end, "llm_generate")
    dep(s_llm_end, s_prev)
    s_notify = sid()
    call(s_notify, "notify_user")
    dep(s_notify, s_llm_end)
    spawn(s_notify, s_llm_end)

    return lines


def main():
    output = []
    output.append("% === Distributed Edge Compute: Anomaly Detection Pipeline ===")
    output.append(f"% {NUM_RUNS} pipeline runs across ESP32 / Raspberry Pi / Jetson / Mac Studio")
    output.append("% Pipeline: sensor reads -> preprocess -> inference -> aggregate -> alert")
    output.append("% Generated by generate_traces.py")
    output.append("")

    for i in range(1, NUM_RUNS + 1):
        desc = "normal"
        if i in [2, 5, 8, 11, 14, 17, 20, 23, 25]:
            desc = "anomaly detected → alert"
        if i in [3, 10, 16, 22]:
            desc += " + recalibrate"
        if i == 6:
            desc = "sensor failure"
        if i == 19:
            desc = "inference timeout"
        output.append(f"% === Pipeline run {i}: {desc} ===")
        output.extend(generate_run(i))
        output.append("")

    print("\n".join(output))


if __name__ == "__main__":
    main()
