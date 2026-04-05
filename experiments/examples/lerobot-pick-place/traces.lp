% === LeRobot Pick-and-Place: Real Robot Demonstrations ===
% 50 episodes converted from continuous trajectories
% Segmentation: gripper transitions + velocity changepoints
% Source: lerobot/ucsd_pick_and_place_dataset (HuggingFace)

% === Episode 0 (pick_and_place, 5 segments) ===
job("run_1").
job_status("run_1", "completed").
call("run_1", "run_1_step_0", "prompt_user", "completed").
call("run_1", "run_1_step_1", "llm_generate", "completed").
depends("run_1", "run_1_step_1", "run_1_step_0").
call("run_1", "run_1_llm_2", "llm_generate", "completed").
depends("run_1", "run_1_llm_2", "run_1_step_1").
call("run_1", "run_1_step_2", "approach", "completed").
depends("run_1", "run_1_step_2", "run_1_llm_2").
spawned_by("run_1", "run_1_step_2", "run_1_llm_2").
param("run_1", "run_1_step_2", "start_x", "0.900").
param("run_1", "run_1_step_2", "start_y", "2.000").
param("run_1", "run_1_step_2", "start_z", "2.000").
param("run_1", "run_1_step_2", "end_x", "1.476").
param("run_1", "run_1_step_2", "end_y", "1.902").
param("run_1", "run_1_step_2", "end_z", "1.644").
param("run_1", "run_1_step_2", "duration_sec", "0.80").
param("run_1", "run_1_step_2", "mean_speed", "0.2022").
param("run_1", "run_1_step_2", "gripper_start", "3.98").
param("run_1", "run_1_step_2", "gripper_end", "3.99").
call("run_1", "run_1_llm_3", "llm_generate", "completed").
depends("run_1", "run_1_llm_3", "run_1_step_2").
call("run_1", "run_1_step_3", "grasp", "completed").
depends("run_1", "run_1_step_3", "run_1_llm_3").
spawned_by("run_1", "run_1_step_3", "run_1_llm_3").
param("run_1", "run_1_step_3", "start_x", "1.525").
param("run_1", "run_1_step_3", "start_y", "1.888").
param("run_1", "run_1_step_3", "start_z", "1.540").
param("run_1", "run_1_step_3", "end_x", "1.760").
param("run_1", "run_1_step_3", "end_y", "1.928").
param("run_1", "run_1_step_3", "end_z", "1.215").
param("run_1", "run_1_step_3", "duration_sec", "1.00").
param("run_1", "run_1_step_3", "mean_speed", "0.1476").
param("run_1", "run_1_step_3", "gripper_start", "3.99").
param("run_1", "run_1_step_3", "gripper_end", "0.27").
call("run_1", "run_1_llm_4", "llm_generate", "completed").
depends("run_1", "run_1_llm_4", "run_1_step_3").
call("run_1", "run_1_step_4", "transport", "completed").
depends("run_1", "run_1_step_4", "run_1_llm_4").
spawned_by("run_1", "run_1_step_4", "run_1_llm_4").
param("run_1", "run_1_step_4", "start_x", "1.596").
param("run_1", "run_1_step_4", "start_y", "1.966").
param("run_1", "run_1_step_4", "start_z", "1.279").
param("run_1", "run_1_step_4", "end_x", "3.535").
param("run_1", "run_1_step_4", "end_y", "2.495").
param("run_1", "run_1_step_4", "end_z", "1.649").
param("run_1", "run_1_step_4", "duration_sec", "2.60").
param("run_1", "run_1_step_4", "mean_speed", "0.2464").
param("run_1", "run_1_step_4", "gripper_start", "0.27").
param("run_1", "run_1_step_4", "gripper_end", "0.27").
call("run_1", "run_1_llm_5", "llm_generate", "completed").
depends("run_1", "run_1_llm_5", "run_1_step_4").
call("run_1", "run_1_step_5", "release", "completed").
depends("run_1", "run_1_step_5", "run_1_llm_5").
spawned_by("run_1", "run_1_step_5", "run_1_llm_5").
param("run_1", "run_1_step_5", "start_x", "3.720").
param("run_1", "run_1_step_5", "start_y", "2.582").
param("run_1", "run_1_step_5", "start_z", "1.557").
param("run_1", "run_1_step_5", "end_x", "4.919").
param("run_1", "run_1_step_5", "end_y", "2.986").
param("run_1", "run_1_step_5", "end_z", "1.170").
param("run_1", "run_1_step_5", "duration_sec", "2.20").
param("run_1", "run_1_step_5", "mean_speed", "0.1560").
param("run_1", "run_1_step_5", "gripper_start", "0.27").
param("run_1", "run_1_step_5", "gripper_end", "3.96").
call("run_1", "run_1_llm_6", "llm_generate", "completed").
depends("run_1", "run_1_llm_6", "run_1_step_5").
call("run_1", "run_1_step_6", "retract", "completed").
depends("run_1", "run_1_step_6", "run_1_llm_6").
spawned_by("run_1", "run_1_step_6", "run_1_llm_6").
param("run_1", "run_1_step_6", "start_x", "4.906").
param("run_1", "run_1_step_6", "start_y", "3.019").
param("run_1", "run_1_step_6", "start_z", "1.192").
param("run_1", "run_1_step_6", "end_x", "5.044").
param("run_1", "run_1_step_6", "end_y", "3.000").
param("run_1", "run_1_step_6", "end_z", "1.200").
param("run_1", "run_1_step_6", "duration_sec", "3.20").
param("run_1", "run_1_step_6", "mean_speed", "0.0504").
param("run_1", "run_1_step_6", "gripper_start", "3.96").
param("run_1", "run_1_step_6", "gripper_end", "3.97").
call("run_1", "run_1_llm_7", "llm_generate", "completed").
depends("run_1", "run_1_llm_7", "run_1_step_6").
call("run_1", "run_1_notify", "notify_user", "completed").
depends("run_1", "run_1_notify", "run_1_llm_7").
spawned_by("run_1", "run_1_notify", "run_1_llm_7").

% === Episode 1 (pick_and_place, 2 segments) ===
job("run_2").
job_status("run_2", "completed").
call("run_2", "run_2_step_0", "prompt_user", "completed").
call("run_2", "run_2_step_1", "llm_generate", "completed").
depends("run_2", "run_2_step_1", "run_2_step_0").
call("run_2", "run_2_llm_2", "llm_generate", "completed").
depends("run_2", "run_2_llm_2", "run_2_step_1").
call("run_2", "run_2_step_2", "release", "completed").
depends("run_2", "run_2_step_2", "run_2_llm_2").
spawned_by("run_2", "run_2_step_2", "run_2_llm_2").
param("run_2", "run_2_step_2", "start_x", "2.500").
param("run_2", "run_2_step_2", "start_y", "-0.000").
param("run_2", "run_2_step_2", "start_z", "0.200").
param("run_2", "run_2_step_2", "end_x", "3.293").
param("run_2", "run_2_step_2", "end_y", "-1.832").
param("run_2", "run_2_step_2", "end_z", "0.158").
param("run_2", "run_2_step_2", "duration_sec", "1.60").
param("run_2", "run_2_step_2", "mean_speed", "0.2794").
param("run_2", "run_2_step_2", "gripper_start", "6.04").
param("run_2", "run_2_step_2", "gripper_end", "7.96").
call("run_2", "run_2_llm_3", "llm_generate", "completed").
depends("run_2", "run_2_llm_3", "run_2_step_2").
call("run_2", "run_2_step_3", "grasp", "completed").
depends("run_2", "run_2_step_3", "run_2_llm_3").
spawned_by("run_2", "run_2_step_3", "run_2_llm_3").
param("run_2", "run_2_step_3", "start_x", "3.329").
param("run_2", "run_2_step_3", "start_y", "-2.011").
param("run_2", "run_2_step_3", "start_z", "0.157").
param("run_2", "run_2_step_3", "end_x", "3.524").
param("run_2", "run_2_step_3", "end_y", "-2.016").
param("run_2", "run_2_step_3", "end_z", "1.904").
param("run_2", "run_2_step_3", "duration_sec", "8.20").
param("run_2", "run_2_step_3", "mean_speed", "0.1646").
param("run_2", "run_2_step_3", "gripper_start", "7.96").
param("run_2", "run_2_step_3", "gripper_end", "4.69").
call("run_2", "run_2_llm_4", "llm_generate", "completed").
depends("run_2", "run_2_llm_4", "run_2_step_3").
call("run_2", "run_2_notify", "notify_user", "completed").
depends("run_2", "run_2_notify", "run_2_llm_4").
spawned_by("run_2", "run_2_notify", "run_2_llm_4").

% === Episode 2 (pick_and_place, 2 segments) ===
job("run_3").
job_status("run_3", "completed").
call("run_3", "run_3_step_0", "prompt_user", "completed").
call("run_3", "run_3_step_1", "llm_generate", "completed").
depends("run_3", "run_3_step_1", "run_3_step_0").
call("run_3", "run_3_llm_2", "llm_generate", "completed").
depends("run_3", "run_3_llm_2", "run_3_step_1").
call("run_3", "run_3_step_2", "release", "completed").
depends("run_3", "run_3_step_2", "run_3_llm_2").
spawned_by("run_3", "run_3_step_2", "run_3_llm_2").
param("run_3", "run_3_step_2", "start_x", "2.500").
param("run_3", "run_3_step_2", "start_y", "-0.000").
param("run_3", "run_3_step_2", "start_z", "0.200").
param("run_3", "run_3_step_2", "end_x", "4.548").
param("run_3", "run_3_step_2", "end_y", "-2.014").
param("run_3", "run_3_step_2", "end_z", "0.152").
param("run_3", "run_3_step_2", "duration_sec", "2.60").
param("run_3", "run_3_step_2", "mean_speed", "0.2380").
param("run_3", "run_3_step_2", "gripper_start", "6.03").
param("run_3", "run_3_step_2", "gripper_end", "7.96").
call("run_3", "run_3_llm_3", "llm_generate", "completed").
depends("run_3", "run_3_llm_3", "run_3_step_2").
call("run_3", "run_3_step_3", "grasp", "completed").
depends("run_3", "run_3_step_3", "run_3_llm_3").
spawned_by("run_3", "run_3_step_3", "run_3_llm_3").
param("run_3", "run_3_step_3", "start_x", "4.569").
param("run_3", "run_3_step_3", "start_y", "-2.156").
param("run_3", "run_3_step_3", "start_z", "0.151").
param("run_3", "run_3_step_3", "end_x", "4.576").
param("run_3", "run_3_step_3", "end_y", "-2.479").
param("run_3", "run_3_step_3", "end_z", "1.505").
param("run_3", "run_3_step_3", "duration_sec", "7.20").
param("run_3", "run_3_step_3", "mean_speed", "0.0937").
param("run_3", "run_3_step_3", "gripper_start", "7.96").
param("run_3", "run_3_step_3", "gripper_end", "5.42").
call("run_3", "run_3_llm_4", "llm_generate", "completed").
depends("run_3", "run_3_llm_4", "run_3_step_3").
call("run_3", "run_3_notify", "notify_user", "completed").
depends("run_3", "run_3_notify", "run_3_llm_4").
spawned_by("run_3", "run_3_notify", "run_3_llm_4").

% === Episode 3 (pick_and_place, 3 segments) ===
job("run_4").
job_status("run_4", "completed").
call("run_4", "run_4_step_0", "prompt_user", "completed").
call("run_4", "run_4_step_1", "llm_generate", "completed").
depends("run_4", "run_4_step_1", "run_4_step_0").
call("run_4", "run_4_llm_2", "llm_generate", "completed").
depends("run_4", "run_4_llm_2", "run_4_step_1").
call("run_4", "run_4_step_2", "release", "completed").
depends("run_4", "run_4_step_2", "run_4_llm_2").
spawned_by("run_4", "run_4_step_2", "run_4_llm_2").
param("run_4", "run_4_step_2", "start_x", "2.500").
param("run_4", "run_4_step_2", "start_y", "-0.000").
param("run_4", "run_4_step_2", "start_z", "0.200").
param("run_4", "run_4_step_2", "end_x", "4.050").
param("run_4", "run_4_step_2", "end_y", "-0.547").
param("run_4", "run_4_step_2", "end_z", "0.156").
param("run_4", "run_4_step_2", "duration_sec", "1.60").
param("run_4", "run_4_step_2", "mean_speed", "0.2499").
param("run_4", "run_4_step_2", "gripper_start", "6.03").
param("run_4", "run_4_step_2", "gripper_end", "7.96").
call("run_4", "run_4_llm_3", "llm_generate", "completed").
depends("run_4", "run_4_llm_3", "run_4_step_2").
call("run_4", "run_4_step_3", "grasp", "completed").
depends("run_4", "run_4_step_3", "run_4_llm_3").
spawned_by("run_4", "run_4_step_3", "run_4_llm_3").
param("run_4", "run_4_step_3", "start_x", "4.221").
param("run_4", "run_4_step_3", "start_y", "-0.832").
param("run_4", "run_4_step_3", "start_z", "0.160").
param("run_4", "run_4_step_3", "end_x", "4.284").
param("run_4", "run_4_step_3", "end_y", "-1.515").
param("run_4", "run_4_step_3", "end_z", "1.467").
param("run_4", "run_4_step_3", "duration_sec", "4.60").
param("run_4", "run_4_step_3", "mean_speed", "0.0818").
param("run_4", "run_4_step_3", "gripper_start", "7.96").
param("run_4", "run_4_step_3", "gripper_end", "3.68").
call("run_4", "run_4_llm_4", "llm_generate", "completed").
depends("run_4", "run_4_llm_4", "run_4_step_3").
call("run_4", "run_4_step_4", "hold", "completed").
depends("run_4", "run_4_step_4", "run_4_llm_4").
spawned_by("run_4", "run_4_step_4", "run_4_llm_4").
param("run_4", "run_4_step_4", "start_x", "4.313").
param("run_4", "run_4_step_4", "start_y", "-1.505").
param("run_4", "run_4_step_4", "start_z", "1.470").
param("run_4", "run_4_step_4", "end_x", "4.364").
param("run_4", "run_4_step_4", "end_y", "-1.443").
param("run_4", "run_4_step_4", "end_z", "1.519").
param("run_4", "run_4_step_4", "duration_sec", "3.60").
param("run_4", "run_4_step_4", "mean_speed", "0.0157").
param("run_4", "run_4_step_4", "gripper_start", "3.68").
param("run_4", "run_4_step_4", "gripper_end", "3.68").
call("run_4", "run_4_llm_5", "llm_generate", "completed").
depends("run_4", "run_4_llm_5", "run_4_step_4").
call("run_4", "run_4_notify", "notify_user", "completed").
depends("run_4", "run_4_notify", "run_4_llm_5").
spawned_by("run_4", "run_4_notify", "run_4_llm_5").

% === Episode 4 (pick_and_place, 3 segments) ===
job("run_5").
job_status("run_5", "completed").
call("run_5", "run_5_step_0", "prompt_user", "completed").
call("run_5", "run_5_step_1", "llm_generate", "completed").
depends("run_5", "run_5_step_1", "run_5_step_0").
call("run_5", "run_5_llm_2", "llm_generate", "completed").
depends("run_5", "run_5_llm_2", "run_5_step_1").
call("run_5", "run_5_step_2", "release", "completed").
depends("run_5", "run_5_step_2", "run_5_llm_2").
spawned_by("run_5", "run_5_step_2", "run_5_llm_2").
param("run_5", "run_5_step_2", "start_x", "2.500").
param("run_5", "run_5_step_2", "start_y", "-0.000").
param("run_5", "run_5_step_2", "start_z", "0.200").
param("run_5", "run_5_step_2", "end_x", "4.135").
param("run_5", "run_5_step_2", "end_y", "-1.904").
param("run_5", "run_5_step_2", "end_z", "0.233").
param("run_5", "run_5_step_2", "duration_sec", "3.20").
param("run_5", "run_5_step_2", "mean_speed", "0.1761").
param("run_5", "run_5_step_2", "gripper_start", "6.04").
param("run_5", "run_5_step_2", "gripper_end", "7.96").
call("run_5", "run_5_llm_3", "llm_generate", "completed").
depends("run_5", "run_5_llm_3", "run_5_step_2").
call("run_5", "run_5_step_3", "retract", "completed").
depends("run_5", "run_5_step_3", "run_5_llm_3").
spawned_by("run_5", "run_5_step_3", "run_5_llm_3").
param("run_5", "run_5_step_3", "start_x", "4.107").
param("run_5", "run_5_step_3", "start_y", "-2.005").
param("run_5", "run_5_step_3", "start_z", "0.221").
param("run_5", "run_5_step_3", "end_x", "3.765").
param("run_5", "run_5_step_3", "end_y", "-1.874").
param("run_5", "run_5_step_3", "end_z", "0.150").
param("run_5", "run_5_step_3", "duration_sec", "3.60").
param("run_5", "run_5_step_3", "mean_speed", "0.0368").
param("run_5", "run_5_step_3", "gripper_start", "7.96").
param("run_5", "run_5_step_3", "gripper_end", "7.96").
call("run_5", "run_5_llm_4", "llm_generate", "completed").
depends("run_5", "run_5_llm_4", "run_5_step_3").
call("run_5", "run_5_step_4", "grasp", "completed").
depends("run_5", "run_5_step_4", "run_5_llm_4").
spawned_by("run_5", "run_5_step_4", "run_5_llm_4").
param("run_5", "run_5_step_4", "start_x", "3.759").
param("run_5", "run_5_step_4", "start_y", "-1.863").
param("run_5", "run_5_step_4", "start_z", "0.150").
param("run_5", "run_5_step_4", "end_x", "3.904").
param("run_5", "run_5_step_4", "end_y", "-1.784").
param("run_5", "run_5_step_4", "end_z", "1.005").
param("run_5", "run_5_step_4", "duration_sec", "3.00").
param("run_5", "run_5_step_4", "mean_speed", "0.0397").
param("run_5", "run_5_step_4", "gripper_start", "7.96").
param("run_5", "run_5_step_4", "gripper_end", "6.58").
call("run_5", "run_5_llm_5", "llm_generate", "completed").
depends("run_5", "run_5_llm_5", "run_5_step_4").
call("run_5", "run_5_notify", "notify_user", "completed").
depends("run_5", "run_5_notify", "run_5_llm_5").
spawned_by("run_5", "run_5_notify", "run_5_llm_5").

% === Episode 5 (pick_and_place, 6 segments) ===
job("run_6").
job_status("run_6", "completed").
call("run_6", "run_6_step_0", "prompt_user", "completed").
call("run_6", "run_6_step_1", "llm_generate", "completed").
depends("run_6", "run_6_step_1", "run_6_step_0").
call("run_6", "run_6_llm_2", "llm_generate", "completed").
depends("run_6", "run_6_llm_2", "run_6_step_1").
call("run_6", "run_6_step_2", "transport", "completed").
depends("run_6", "run_6_step_2", "run_6_llm_2").
spawned_by("run_6", "run_6_step_2", "run_6_llm_2").
param("run_6", "run_6_step_2", "start_x", "2.500").
param("run_6", "run_6_step_2", "start_y", "-0.000").
param("run_6", "run_6_step_2", "start_z", "0.200").
param("run_6", "run_6_step_2", "end_x", "2.483").
param("run_6", "run_6_step_2", "end_y", "-0.087").
param("run_6", "run_6_step_2", "end_z", "0.298").
param("run_6", "run_6_step_2", "duration_sec", "1.40").
param("run_6", "run_6_step_2", "mean_speed", "0.0999").
param("run_6", "run_6_step_2", "gripper_start", "5.97").
param("run_6", "run_6_step_2", "gripper_end", "5.17").
call("run_6", "run_6_llm_3", "llm_generate", "completed").
depends("run_6", "run_6_llm_3", "run_6_step_2").
call("run_6", "run_6_step_3", "approach", "completed").
depends("run_6", "run_6_step_3", "run_6_llm_3").
spawned_by("run_6", "run_6_step_3", "run_6_llm_3").
param("run_6", "run_6_step_3", "start_x", "2.599").
param("run_6", "run_6_step_3", "start_y", "-0.205").
param("run_6", "run_6_step_3", "start_z", "0.228").
param("run_6", "run_6_step_3", "end_x", "2.822").
param("run_6", "run_6_step_3", "end_y", "-0.291").
param("run_6", "run_6_step_3", "end_z", "0.266").
param("run_6", "run_6_step_3", "duration_sec", "0.60").
param("run_6", "run_6_step_3", "mean_speed", "0.1333").
param("run_6", "run_6_step_3", "gripper_start", "4.93").
param("run_6", "run_6_step_3", "gripper_end", "5.57").
call("run_6", "run_6_llm_4", "llm_generate", "completed").
depends("run_6", "run_6_llm_4", "run_6_step_3").
call("run_6", "run_6_step_4", "grasp", "completed").
depends("run_6", "run_6_step_4", "run_6_llm_4").
spawned_by("run_6", "run_6_step_4", "run_6_llm_4").
param("run_6", "run_6_step_4", "start_x", "2.952").
param("run_6", "run_6_step_4", "start_y", "-0.262").
param("run_6", "run_6_step_4", "start_z", "0.305").
param("run_6", "run_6_step_4", "end_x", "3.666").
param("run_6", "run_6_step_4", "end_y", "-0.501").
param("run_6", "run_6_step_4", "end_z", "0.078").
param("run_6", "run_6_step_4", "duration_sec", "1.00").
param("run_6", "run_6_step_4", "mean_speed", "0.1985").
param("run_6", "run_6_step_4", "gripper_start", "5.57").
param("run_6", "run_6_step_4", "gripper_end", "3.66").
call("run_6", "run_6_llm_5", "llm_generate", "completed").
depends("run_6", "run_6_llm_5", "run_6_step_4").
call("run_6", "run_6_step_5", "transport", "completed").
depends("run_6", "run_6_step_5", "run_6_llm_5").
spawned_by("run_6", "run_6_step_5", "run_6_llm_5").
param("run_6", "run_6_step_5", "start_x", "3.752").
param("run_6", "run_6_step_5", "start_y", "-0.622").
param("run_6", "run_6_step_5", "start_z", "0.129").
param("run_6", "run_6_step_5", "end_x", "3.848").
param("run_6", "run_6_step_5", "end_y", "-0.278").
param("run_6", "run_6_step_5", "end_z", "0.424").
param("run_6", "run_6_step_5", "duration_sec", "2.20").
param("run_6", "run_6_step_5", "mean_speed", "0.1222").
param("run_6", "run_6_step_5", "gripper_start", "3.66").
param("run_6", "run_6_step_5", "gripper_end", "4.05").
call("run_6", "run_6_llm_6", "llm_generate", "completed").
depends("run_6", "run_6_llm_6", "run_6_step_5").
call("run_6", "run_6_step_6", "grasp", "completed").
depends("run_6", "run_6_step_6", "run_6_llm_6").
spawned_by("run_6", "run_6_step_6", "run_6_llm_6").
param("run_6", "run_6_step_6", "start_x", "3.896").
param("run_6", "run_6_step_6", "start_y", "-0.338").
param("run_6", "run_6_step_6", "start_z", "0.457").
param("run_6", "run_6_step_6", "end_x", "2.973").
param("run_6", "run_6_step_6", "end_y", "-0.201").
param("run_6", "run_6_step_6", "end_z", "0.419").
param("run_6", "run_6_step_6", "duration_sec", "2.40").
param("run_6", "run_6_step_6", "mean_speed", "0.1276").
param("run_6", "run_6_step_6", "gripper_start", "4.86").
param("run_6", "run_6_step_6", "gripper_end", "3.38").
call("run_6", "run_6_llm_7", "llm_generate", "completed").
depends("run_6", "run_6_llm_7", "run_6_step_6").
call("run_6", "run_6_step_7", "move", "completed").
depends("run_6", "run_6_step_7", "run_6_llm_7").
spawned_by("run_6", "run_6_step_7", "run_6_llm_7").
param("run_6", "run_6_step_7", "start_x", "2.929").
param("run_6", "run_6_step_7", "start_y", "-0.172").
param("run_6", "run_6_step_7", "start_z", "0.390").
param("run_6", "run_6_step_7", "end_x", "2.438").
param("run_6", "run_6_step_7", "end_y", "-0.544").
param("run_6", "run_6_step_7", "end_z", "0.458").
param("run_6", "run_6_step_7", "duration_sec", "2.20").
param("run_6", "run_6_step_7", "mean_speed", "0.1642").
param("run_6", "run_6_step_7", "gripper_start", "0.14").
param("run_6", "run_6_step_7", "gripper_end", "0.26").
call("run_6", "run_6_llm_8", "llm_generate", "completed").
depends("run_6", "run_6_llm_8", "run_6_step_7").
call("run_6", "run_6_notify", "notify_user", "completed").
depends("run_6", "run_6_notify", "run_6_llm_8").
spawned_by("run_6", "run_6_notify", "run_6_llm_8").

% === Episode 6 (pick_and_place, 4 segments) ===
job("run_7").
job_status("run_7", "completed").
call("run_7", "run_7_step_0", "prompt_user", "completed").
call("run_7", "run_7_step_1", "llm_generate", "completed").
depends("run_7", "run_7_step_1", "run_7_step_0").
call("run_7", "run_7_llm_2", "llm_generate", "completed").
depends("run_7", "run_7_llm_2", "run_7_step_1").
call("run_7", "run_7_step_2", "release", "completed").
depends("run_7", "run_7_step_2", "run_7_llm_2").
spawned_by("run_7", "run_7_step_2", "run_7_llm_2").
param("run_7", "run_7_step_2", "start_x", "2.500").
param("run_7", "run_7_step_2", "start_y", "-0.000").
param("run_7", "run_7_step_2", "start_z", "0.200").
param("run_7", "run_7_step_2", "end_x", "4.750").
param("run_7", "run_7_step_2", "end_y", "-0.204").
param("run_7", "run_7_step_2", "end_z", "0.103").
param("run_7", "run_7_step_2", "duration_sec", "2.20").
param("run_7", "run_7_step_2", "mean_speed", "0.2308").
param("run_7", "run_7_step_2", "gripper_start", "6.04").
param("run_7", "run_7_step_2", "gripper_end", "7.96").
call("run_7", "run_7_llm_3", "llm_generate", "completed").
depends("run_7", "run_7_llm_3", "run_7_step_2").
call("run_7", "run_7_step_3", "approach", "completed").
depends("run_7", "run_7_step_3", "run_7_llm_3").
spawned_by("run_7", "run_7_step_3", "run_7_llm_3").
param("run_7", "run_7_step_3", "start_x", "4.949").
param("run_7", "run_7_step_3", "start_y", "-0.213").
param("run_7", "run_7_step_3", "start_z", "0.089").
param("run_7", "run_7_step_3", "end_x", "5.126").
param("run_7", "run_7_step_3", "end_y", "-0.299").
param("run_7", "run_7_step_3", "end_z", "0.097").
param("run_7", "run_7_step_3", "duration_sec", "1.00").
param("run_7", "run_7_step_3", "mean_speed", "0.0558").
param("run_7", "run_7_step_3", "gripper_start", "7.96").
param("run_7", "run_7_step_3", "gripper_end", "7.96").
call("run_7", "run_7_llm_4", "llm_generate", "completed").
depends("run_7", "run_7_llm_4", "run_7_step_3").
call("run_7", "run_7_step_4", "hold", "completed").
depends("run_7", "run_7_step_4", "run_7_llm_4").
spawned_by("run_7", "run_7_step_4", "run_7_llm_4").
param("run_7", "run_7_step_4", "start_x", "5.127").
param("run_7", "run_7_step_4", "start_y", "-0.325").
param("run_7", "run_7_step_4", "start_z", "0.094").
param("run_7", "run_7_step_4", "end_x", "5.130").
param("run_7", "run_7_step_4", "end_y", "-0.345").
param("run_7", "run_7_step_4", "end_z", "0.101").
param("run_7", "run_7_step_4", "duration_sec", "0.60").
param("run_7", "run_7_step_4", "mean_speed", "0.0128").
param("run_7", "run_7_step_4", "gripper_start", "7.96").
param("run_7", "run_7_step_4", "gripper_end", "7.38").
call("run_7", "run_7_llm_5", "llm_generate", "completed").
depends("run_7", "run_7_llm_5", "run_7_step_4").
call("run_7", "run_7_step_5", "grasp", "completed").
depends("run_7", "run_7_step_5", "run_7_llm_5").
spawned_by("run_7", "run_7_step_5", "run_7_llm_5").
param("run_7", "run_7_step_5", "start_x", "5.140").
param("run_7", "run_7_step_5", "start_y", "-0.357").
param("run_7", "run_7_step_5", "start_z", "0.100").
param("run_7", "run_7_step_5", "end_x", "5.028").
param("run_7", "run_7_step_5", "end_y", "-1.941").
param("run_7", "run_7_step_5", "end_z", "1.066").
param("run_7", "run_7_step_5", "duration_sec", "6.00").
param("run_7", "run_7_step_5", "mean_speed", "0.0672").
param("run_7", "run_7_step_5", "gripper_start", "7.38").
param("run_7", "run_7_step_5", "gripper_end", "0.02").
call("run_7", "run_7_llm_6", "llm_generate", "completed").
depends("run_7", "run_7_llm_6", "run_7_step_5").
call("run_7", "run_7_notify", "notify_user", "completed").
depends("run_7", "run_7_notify", "run_7_llm_6").
spawned_by("run_7", "run_7_notify", "run_7_llm_6").

% === Episode 7 (pick_and_place, 6 segments) ===
job("run_8").
job_status("run_8", "completed").
call("run_8", "run_8_step_0", "prompt_user", "completed").
call("run_8", "run_8_step_1", "llm_generate", "completed").
depends("run_8", "run_8_step_1", "run_8_step_0").
call("run_8", "run_8_llm_2", "llm_generate", "completed").
depends("run_8", "run_8_llm_2", "run_8_step_1").
call("run_8", "run_8_step_2", "release", "completed").
depends("run_8", "run_8_step_2", "run_8_llm_2").
spawned_by("run_8", "run_8_step_2", "run_8_llm_2").
param("run_8", "run_8_step_2", "start_x", "2.500").
param("run_8", "run_8_step_2", "start_y", "-0.000").
param("run_8", "run_8_step_2", "start_z", "0.200").
param("run_8", "run_8_step_2", "end_x", "3.905").
param("run_8", "run_8_step_2", "end_y", "-1.901").
param("run_8", "run_8_step_2", "end_z", "0.160").
param("run_8", "run_8_step_2", "duration_sec", "1.80").
param("run_8", "run_8_step_2", "mean_speed", "0.3088").
param("run_8", "run_8_step_2", "gripper_start", "6.04").
param("run_8", "run_8_step_2", "gripper_end", "7.97").
call("run_8", "run_8_llm_3", "llm_generate", "completed").
depends("run_8", "run_8_llm_3", "run_8_step_2").
call("run_8", "run_8_step_3", "grasp", "completed").
depends("run_8", "run_8_step_3", "run_8_llm_3").
spawned_by("run_8", "run_8_step_3", "run_8_llm_3").
param("run_8", "run_8_step_3", "start_x", "4.071").
param("run_8", "run_8_step_3", "start_y", "-2.263").
param("run_8", "run_8_step_3", "start_z", "0.147").
param("run_8", "run_8_step_3", "end_x", "4.152").
param("run_8", "run_8_step_3", "end_y", "-2.549").
param("run_8", "run_8_step_3", "end_z", "0.440").
param("run_8", "run_8_step_3", "duration_sec", "1.40").
param("run_8", "run_8_step_3", "mean_speed", "0.1621").
param("run_8", "run_8_step_3", "gripper_start", "7.97").
param("run_8", "run_8_step_3", "gripper_end", "3.25").
call("run_8", "run_8_llm_4", "llm_generate", "completed").
depends("run_8", "run_8_llm_4", "run_8_step_3").
call("run_8", "run_8_step_4", "transport", "completed").
depends("run_8", "run_8_step_4", "run_8_llm_4").
spawned_by("run_8", "run_8_step_4", "run_8_llm_4").
param("run_8", "run_8_step_4", "start_x", "4.275").
param("run_8", "run_8_step_4", "start_y", "-2.507").
param("run_8", "run_8_step_4", "start_z", "0.507").
param("run_8", "run_8_step_4", "end_x", "5.393").
param("run_8", "run_8_step_4", "end_y", "-2.934").
param("run_8", "run_8_step_4", "end_z", "1.952").
param("run_8", "run_8_step_4", "duration_sec", "2.20").
param("run_8", "run_8_step_4", "mean_speed", "0.3584").
param("run_8", "run_8_step_4", "gripper_start", "3.25").
param("run_8", "run_8_step_4", "gripper_end", "2.63").
call("run_8", "run_8_llm_5", "llm_generate", "completed").
depends("run_8", "run_8_llm_5", "run_8_step_4").
call("run_8", "run_8_step_5", "retract", "completed").
depends("run_8", "run_8_step_5", "run_8_llm_5").
spawned_by("run_8", "run_8_step_5", "run_8_llm_5").
param("run_8", "run_8_step_5", "start_x", "5.425").
param("run_8", "run_8_step_5", "start_y", "-2.983").
param("run_8", "run_8_step_5", "start_z", "2.056").
param("run_8", "run_8_step_5", "end_x", "5.560").
param("run_8", "run_8_step_5", "end_y", "-3.018").
param("run_8", "run_8_step_5", "end_z", "1.969").
param("run_8", "run_8_step_5", "duration_sec", "0.60").
param("run_8", "run_8_step_5", "mean_speed", "0.0794").
param("run_8", "run_8_step_5", "gripper_start", "2.11").
param("run_8", "run_8_step_5", "gripper_end", "2.88").
call("run_8", "run_8_llm_6", "llm_generate", "completed").
depends("run_8", "run_8_llm_6", "run_8_step_5").
call("run_8", "run_8_step_6", "grasp", "completed").
depends("run_8", "run_8_step_6", "run_8_llm_6").
spawned_by("run_8", "run_8_step_6", "run_8_llm_6").
param("run_8", "run_8_step_6", "start_x", "5.536").
param("run_8", "run_8_step_6", "start_y", "-3.002").
param("run_8", "run_8_step_6", "start_z", "1.955").
param("run_8", "run_8_step_6", "end_x", "5.520").
param("run_8", "run_8_step_6", "end_y", "-3.006").
param("run_8", "run_8_step_6", "end_z", "1.989").
param("run_8", "run_8_step_6", "duration_sec", "0.80").
param("run_8", "run_8_step_6", "mean_speed", "0.0424").
param("run_8", "run_8_step_6", "gripper_start", "2.88").
param("run_8", "run_8_step_6", "gripper_end", "1.68").
call("run_8", "run_8_llm_7", "llm_generate", "completed").
depends("run_8", "run_8_llm_7", "run_8_step_6").
call("run_8", "run_8_step_7", "retract", "completed").
depends("run_8", "run_8_step_7", "run_8_llm_7").
spawned_by("run_8", "run_8_step_7", "run_8_llm_7").
param("run_8", "run_8_step_7", "start_x", "5.508").
param("run_8", "run_8_step_7", "start_y", "-3.002").
param("run_8", "run_8_step_7", "start_z", "1.995").
param("run_8", "run_8_step_7", "end_x", "5.500").
param("run_8", "run_8_step_7", "end_y", "-3.000").
param("run_8", "run_8_step_7", "end_z", "2.000").
param("run_8", "run_8_step_7", "duration_sec", "3.00").
param("run_8", "run_8_step_7", "mean_speed", "0.0264").
param("run_8", "run_8_step_7", "gripper_start", "1.68").
param("run_8", "run_8_step_7", "gripper_end", "1.77").
call("run_8", "run_8_llm_8", "llm_generate", "completed").
depends("run_8", "run_8_llm_8", "run_8_step_7").
call("run_8", "run_8_notify", "notify_user", "completed").
depends("run_8", "run_8_notify", "run_8_llm_8").
spawned_by("run_8", "run_8_notify", "run_8_llm_8").

% === Episode 8 (pick_and_place, 5 segments) ===
job("run_9").
job_status("run_9", "completed").
call("run_9", "run_9_step_0", "prompt_user", "completed").
call("run_9", "run_9_step_1", "llm_generate", "completed").
depends("run_9", "run_9_step_1", "run_9_step_0").
call("run_9", "run_9_llm_2", "llm_generate", "completed").
depends("run_9", "run_9_llm_2", "run_9_step_1").
call("run_9", "run_9_step_2", "release", "completed").
depends("run_9", "run_9_step_2", "run_9_llm_2").
spawned_by("run_9", "run_9_step_2", "run_9_llm_2").
param("run_9", "run_9_step_2", "start_x", "2.500").
param("run_9", "run_9_step_2", "start_y", "-0.000").
param("run_9", "run_9_step_2", "start_z", "0.200").
param("run_9", "run_9_step_2", "end_x", "3.743").
param("run_9", "run_9_step_2", "end_y", "-0.389").
param("run_9", "run_9_step_2", "end_z", "0.194").
param("run_9", "run_9_step_2", "duration_sec", "1.20").
param("run_9", "run_9_step_2", "mean_speed", "0.2833").
param("run_9", "run_9_step_2", "gripper_start", "6.04").
param("run_9", "run_9_step_2", "gripper_end", "7.95").
call("run_9", "run_9_llm_3", "llm_generate", "completed").
depends("run_9", "run_9_llm_3", "run_9_step_2").
call("run_9", "run_9_step_3", "approach", "completed").
depends("run_9", "run_9_step_3", "run_9_llm_3").
spawned_by("run_9", "run_9_step_3", "run_9_llm_3").
param("run_9", "run_9_step_3", "start_x", "4.064").
param("run_9", "run_9_step_3", "start_y", "-0.464").
param("run_9", "run_9_step_3", "start_z", "0.190").
param("run_9", "run_9_step_3", "end_x", "4.513").
param("run_9", "run_9_step_3", "end_y", "-0.594").
param("run_9", "run_9_step_3", "end_z", "0.145").
param("run_9", "run_9_step_3", "duration_sec", "0.60").
param("run_9", "run_9_step_3", "mean_speed", "0.2534").
param("run_9", "run_9_step_3", "gripper_start", "7.95").
param("run_9", "run_9_step_3", "gripper_end", "7.96").
call("run_9", "run_9_llm_4", "llm_generate", "completed").
depends("run_9", "run_9_llm_4", "run_9_step_3").
call("run_9", "run_9_step_4", "grasp", "completed").
depends("run_9", "run_9_step_4", "run_9_llm_4").
spawned_by("run_9", "run_9_step_4", "run_9_llm_4").
param("run_9", "run_9_step_4", "start_x", "4.770").
param("run_9", "run_9_step_4", "start_y", "-0.682").
param("run_9", "run_9_step_4", "start_z", "0.134").
param("run_9", "run_9_step_4", "end_x", "5.217").
param("run_9", "run_9_step_4", "end_y", "-1.075").
param("run_9", "run_9_step_4", "end_z", "0.168").
param("run_9", "run_9_step_4", "duration_sec", "0.60").
param("run_9", "run_9_step_4", "mean_speed", "0.2516").
param("run_9", "run_9_step_4", "gripper_start", "7.96").
param("run_9", "run_9_step_4", "gripper_end", "6.53").
call("run_9", "run_9_llm_5", "llm_generate", "completed").
depends("run_9", "run_9_llm_5", "run_9_step_4").
call("run_9", "run_9_step_5", "release", "completed").
depends("run_9", "run_9_step_5", "run_9_llm_5").
spawned_by("run_9", "run_9_step_5", "run_9_llm_5").
param("run_9", "run_9_step_5", "start_x", "5.291").
param("run_9", "run_9_step_5", "start_y", "-1.214").
param("run_9", "run_9_step_5", "start_z", "0.152").
param("run_9", "run_9_step_5", "end_x", "5.394").
param("run_9", "run_9_step_5", "end_y", "-2.206").
param("run_9", "run_9_step_5", "end_z", "0.210").
param("run_9", "run_9_step_5", "duration_sec", "1.60").
param("run_9", "run_9_step_5", "mean_speed", "0.1686").
param("run_9", "run_9_step_5", "gripper_start", "6.53").
param("run_9", "run_9_step_5", "gripper_end", "7.96").
call("run_9", "run_9_llm_6", "llm_generate", "completed").
depends("run_9", "run_9_llm_6", "run_9_step_5").
call("run_9", "run_9_step_6", "transport", "completed").
depends("run_9", "run_9_step_6", "run_9_llm_6").
spawned_by("run_9", "run_9_step_6", "run_9_llm_6").
param("run_9", "run_9_step_6", "start_x", "5.378").
param("run_9", "run_9_step_6", "start_y", "-2.345").
param("run_9", "run_9_step_6", "start_z", "0.189").
param("run_9", "run_9_step_6", "end_x", "5.498").
param("run_9", "run_9_step_6", "end_y", "-2.427").
param("run_9", "run_9_step_6", "end_z", "0.152").
param("run_9", "run_9_step_6", "duration_sec", "5.80").
param("run_9", "run_9_step_6", "mean_speed", "0.0997").
param("run_9", "run_9_step_6", "gripper_start", "7.96").
param("run_9", "run_9_step_6", "gripper_end", "7.82").
call("run_9", "run_9_llm_7", "llm_generate", "completed").
depends("run_9", "run_9_llm_7", "run_9_step_6").
call("run_9", "run_9_notify", "notify_user", "completed").
depends("run_9", "run_9_notify", "run_9_llm_7").
spawned_by("run_9", "run_9_notify", "run_9_llm_7").

% === Episode 9 (pick_and_place, 6 segments) ===
job("run_10").
job_status("run_10", "completed").
call("run_10", "run_10_step_0", "prompt_user", "completed").
call("run_10", "run_10_step_1", "llm_generate", "completed").
depends("run_10", "run_10_step_1", "run_10_step_0").
call("run_10", "run_10_llm_2", "llm_generate", "completed").
depends("run_10", "run_10_llm_2", "run_10_step_1").
call("run_10", "run_10_step_2", "approach", "completed").
depends("run_10", "run_10_step_2", "run_10_llm_2").
spawned_by("run_10", "run_10_step_2", "run_10_llm_2").
param("run_10", "run_10_step_2", "start_x", "2.500").
param("run_10", "run_10_step_2", "start_y", "-0.000").
param("run_10", "run_10_step_2", "start_z", "0.200").
param("run_10", "run_10_step_2", "end_x", "2.533").
param("run_10", "run_10_step_2", "end_y", "-0.326").
param("run_10", "run_10_step_2", "end_z", "0.227").
param("run_10", "run_10_step_2", "duration_sec", "0.60").
param("run_10", "run_10_step_2", "mean_speed", "0.2491").
param("run_10", "run_10_step_2", "gripper_start", "6.04").
param("run_10", "run_10_step_2", "gripper_end", "6.89").
call("run_10", "run_10_llm_3", "llm_generate", "completed").
depends("run_10", "run_10_llm_3", "run_10_step_2").
call("run_10", "run_10_step_3", "release", "completed").
depends("run_10", "run_10_step_3", "run_10_llm_3").
spawned_by("run_10", "run_10_step_3", "run_10_llm_3").
param("run_10", "run_10_step_3", "start_x", "2.690").
param("run_10", "run_10_step_3", "start_y", "-0.705").
param("run_10", "run_10_step_3", "start_z", "0.151").
param("run_10", "run_10_step_3", "end_x", "3.740").
param("run_10", "run_10_step_3", "end_y", "-1.491").
param("run_10", "run_10_step_3", "end_z", "0.273").
param("run_10", "run_10_step_3", "duration_sec", "1.00").
param("run_10", "run_10_step_3", "mean_speed", "0.2998").
param("run_10", "run_10_step_3", "gripper_start", "6.89").
param("run_10", "run_10_step_3", "gripper_end", "7.94").
call("run_10", "run_10_llm_4", "llm_generate", "completed").
depends("run_10", "run_10_llm_4", "run_10_step_3").
call("run_10", "run_10_step_4", "grasp", "completed").
depends("run_10", "run_10_step_4", "run_10_llm_4").
spawned_by("run_10", "run_10_step_4", "run_10_llm_4").
param("run_10", "run_10_step_4", "start_x", "3.871").
param("run_10", "run_10_step_4", "start_y", "-1.458").
param("run_10", "run_10_step_4", "start_z", "0.340").
param("run_10", "run_10_step_4", "end_x", "3.646").
param("run_10", "run_10_step_4", "end_y", "-1.607").
param("run_10", "run_10_step_4", "end_z", "0.230").
param("run_10", "run_10_step_4", "duration_sec", "0.60").
param("run_10", "run_10_step_4", "mean_speed", "0.2067").
param("run_10", "run_10_step_4", "gripper_start", "7.94").
param("run_10", "run_10_step_4", "gripper_end", "6.53").
call("run_10", "run_10_llm_5", "llm_generate", "completed").
depends("run_10", "run_10_llm_5", "run_10_step_4").
call("run_10", "run_10_step_5", "release", "completed").
depends("run_10", "run_10_step_5", "run_10_llm_5").
spawned_by("run_10", "run_10_step_5", "run_10_llm_5").
param("run_10", "run_10_step_5", "start_x", "3.518").
param("run_10", "run_10_step_5", "start_y", "-1.621").
param("run_10", "run_10_step_5", "start_z", "0.145").
param("run_10", "run_10_step_5", "end_x", "3.505").
param("run_10", "run_10_step_5", "end_y", "-2.049").
param("run_10", "run_10_step_5", "end_z", "0.241").
param("run_10", "run_10_step_5", "duration_sec", "1.00").
param("run_10", "run_10_step_5", "mean_speed", "0.2187").
param("run_10", "run_10_step_5", "gripper_start", "6.53").
param("run_10", "run_10_step_5", "gripper_end", "7.95").
call("run_10", "run_10_llm_6", "llm_generate", "completed").
depends("run_10", "run_10_llm_6", "run_10_step_5").
call("run_10", "run_10_step_6", "grasp", "completed").
depends("run_10", "run_10_step_6", "run_10_llm_6").
spawned_by("run_10", "run_10_step_6", "run_10_llm_6").
param("run_10", "run_10_step_6", "start_x", "3.438").
param("run_10", "run_10_step_6", "start_y", "-2.202").
param("run_10", "run_10_step_6", "start_z", "0.221").
param("run_10", "run_10_step_6", "end_x", "4.116").
param("run_10", "run_10_step_6", "end_y", "-2.554").
param("run_10", "run_10_step_6", "end_z", "0.742").
param("run_10", "run_10_step_6", "duration_sec", "1.80").
param("run_10", "run_10_step_6", "mean_speed", "0.2099").
param("run_10", "run_10_step_6", "gripper_start", "7.95").
param("run_10", "run_10_step_6", "gripper_end", "3.33").
call("run_10", "run_10_llm_7", "llm_generate", "completed").
depends("run_10", "run_10_llm_7", "run_10_step_6").
call("run_10", "run_10_step_7", "transport", "completed").
depends("run_10", "run_10_step_7", "run_10_llm_7").
spawned_by("run_10", "run_10_step_7", "run_10_llm_7").
param("run_10", "run_10_step_7", "start_x", "4.161").
param("run_10", "run_10_step_7", "start_y", "-2.577").
param("run_10", "run_10_step_7", "start_z", "0.899").
param("run_10", "run_10_step_7", "end_x", "3.813").
param("run_10", "run_10_step_7", "end_y", "-2.821").
param("run_10", "run_10_step_7", "end_z", "1.600").
param("run_10", "run_10_step_7", "duration_sec", "4.80").
param("run_10", "run_10_step_7", "mean_speed", "0.1411").
param("run_10", "run_10_step_7", "gripper_start", "3.33").
param("run_10", "run_10_step_7", "gripper_end", "3.33").
call("run_10", "run_10_llm_8", "llm_generate", "completed").
depends("run_10", "run_10_llm_8", "run_10_step_7").
call("run_10", "run_10_notify", "notify_user", "completed").
depends("run_10", "run_10_notify", "run_10_llm_8").
spawned_by("run_10", "run_10_notify", "run_10_llm_8").

% === Episode 10 (pick_and_place, 3 segments) ===
job("run_11").
job_status("run_11", "completed").
call("run_11", "run_11_step_0", "prompt_user", "completed").
call("run_11", "run_11_step_1", "llm_generate", "completed").
depends("run_11", "run_11_step_1", "run_11_step_0").
call("run_11", "run_11_llm_2", "llm_generate", "completed").
depends("run_11", "run_11_llm_2", "run_11_step_1").
call("run_11", "run_11_step_2", "release", "completed").
depends("run_11", "run_11_step_2", "run_11_llm_2").
spawned_by("run_11", "run_11_step_2", "run_11_llm_2").
param("run_11", "run_11_step_2", "start_x", "2.500").
param("run_11", "run_11_step_2", "start_y", "-0.000").
param("run_11", "run_11_step_2", "start_z", "0.200").
param("run_11", "run_11_step_2", "end_x", "2.940").
param("run_11", "run_11_step_2", "end_y", "-0.690").
param("run_11", "run_11_step_2", "end_z", "0.195").
param("run_11", "run_11_step_2", "duration_sec", "1.20").
param("run_11", "run_11_step_2", "mean_speed", "0.1869").
param("run_11", "run_11_step_2", "gripper_start", "6.03").
param("run_11", "run_11_step_2", "gripper_end", "7.96").
call("run_11", "run_11_llm_3", "llm_generate", "completed").
depends("run_11", "run_11_llm_3", "run_11_step_2").
call("run_11", "run_11_step_3", "grasp", "completed").
depends("run_11", "run_11_step_3", "run_11_llm_3").
spawned_by("run_11", "run_11_step_3", "run_11_llm_3").
param("run_11", "run_11_step_3", "start_x", "3.153").
param("run_11", "run_11_step_3", "start_y", "-0.899").
param("run_11", "run_11_step_3", "start_z", "0.184").
param("run_11", "run_11_step_3", "end_x", "3.563").
param("run_11", "run_11_step_3", "end_y", "-1.364").
param("run_11", "run_11_step_3", "end_z", "1.364").
param("run_11", "run_11_step_3", "duration_sec", "3.80").
param("run_11", "run_11_step_3", "mean_speed", "0.0838").
param("run_11", "run_11_step_3", "gripper_start", "7.96").
param("run_11", "run_11_step_3", "gripper_end", "3.38").
call("run_11", "run_11_llm_4", "llm_generate", "completed").
depends("run_11", "run_11_llm_4", "run_11_step_3").
call("run_11", "run_11_step_4", "hold", "completed").
depends("run_11", "run_11_step_4", "run_11_llm_4").
spawned_by("run_11", "run_11_step_4", "run_11_llm_4").
param("run_11", "run_11_step_4", "start_x", "3.580").
param("run_11", "run_11_step_4", "start_y", "-1.363").
param("run_11", "run_11_step_4", "start_z", "1.371").
param("run_11", "run_11_step_4", "end_x", "3.665").
param("run_11", "run_11_step_4", "end_y", "-1.483").
param("run_11", "run_11_step_4", "end_z", "1.449").
param("run_11", "run_11_step_4", "duration_sec", "4.80").
param("run_11", "run_11_step_4", "mean_speed", "0.0153").
param("run_11", "run_11_step_4", "gripper_start", "3.38").
param("run_11", "run_11_step_4", "gripper_end", "3.38").
call("run_11", "run_11_llm_5", "llm_generate", "completed").
depends("run_11", "run_11_llm_5", "run_11_step_4").
call("run_11", "run_11_notify", "notify_user", "completed").
depends("run_11", "run_11_notify", "run_11_llm_5").
spawned_by("run_11", "run_11_notify", "run_11_llm_5").

% === Episode 11 (pick_and_place, 4 segments) ===
job("run_12").
job_status("run_12", "completed").
call("run_12", "run_12_step_0", "prompt_user", "completed").
call("run_12", "run_12_step_1", "llm_generate", "completed").
depends("run_12", "run_12_step_1", "run_12_step_0").
call("run_12", "run_12_llm_2", "llm_generate", "completed").
depends("run_12", "run_12_llm_2", "run_12_step_1").
call("run_12", "run_12_step_2", "release", "completed").
depends("run_12", "run_12_step_2", "run_12_llm_2").
spawned_by("run_12", "run_12_step_2", "run_12_llm_2").
param("run_12", "run_12_step_2", "start_x", "2.500").
param("run_12", "run_12_step_2", "start_y", "-0.000").
param("run_12", "run_12_step_2", "start_z", "0.200").
param("run_12", "run_12_step_2", "end_x", "2.801").
param("run_12", "run_12_step_2", "end_y", "-0.118").
param("run_12", "run_12_step_2", "end_z", "0.189").
param("run_12", "run_12_step_2", "duration_sec", "0.60").
param("run_12", "run_12_step_2", "mean_speed", "0.2413").
param("run_12", "run_12_step_2", "gripper_start", "6.04").
param("run_12", "run_12_step_2", "gripper_end", "7.71").
call("run_12", "run_12_llm_3", "llm_generate", "completed").
depends("run_12", "run_12_llm_3", "run_12_step_2").
call("run_12", "run_12_step_3", "approach", "completed").
depends("run_12", "run_12_step_3", "run_12_llm_3").
spawned_by("run_12", "run_12_step_3", "run_12_llm_3").
param("run_12", "run_12_step_3", "start_x", "3.195").
param("run_12", "run_12_step_3", "start_y", "-0.109").
param("run_12", "run_12_step_3", "start_z", "0.118").
param("run_12", "run_12_step_3", "end_x", "5.352").
param("run_12", "run_12_step_3", "end_y", "-1.418").
param("run_12", "run_12_step_3", "end_z", "0.147").
param("run_12", "run_12_step_3", "duration_sec", "1.80").
param("run_12", "run_12_step_3", "mean_speed", "0.3139").
param("run_12", "run_12_step_3", "gripper_start", "7.71").
param("run_12", "run_12_step_3", "gripper_end", "7.96").
call("run_12", "run_12_llm_4", "llm_generate", "completed").
depends("run_12", "run_12_llm_4", "run_12_step_3").
call("run_12", "run_12_step_4", "grasp", "completed").
depends("run_12", "run_12_step_4", "run_12_llm_4").
spawned_by("run_12", "run_12_step_4", "run_12_llm_4").
param("run_12", "run_12_step_4", "start_x", "5.201").
param("run_12", "run_12_step_4", "start_y", "-1.506").
param("run_12", "run_12_step_4", "start_z", "0.149").
param("run_12", "run_12_step_4", "end_x", "4.760").
param("run_12", "run_12_step_4", "end_y", "-1.908").
param("run_12", "run_12_step_4", "end_z", "0.340").
param("run_12", "run_12_step_4", "duration_sec", "1.40").
param("run_12", "run_12_step_4", "mean_speed", "0.1588").
param("run_12", "run_12_step_4", "gripper_start", "7.96").
param("run_12", "run_12_step_4", "gripper_end", "3.31").
call("run_12", "run_12_llm_5", "llm_generate", "completed").
depends("run_12", "run_12_llm_5", "run_12_step_4").
call("run_12", "run_12_step_5", "transport", "completed").
depends("run_12", "run_12_step_5", "run_12_llm_5").
spawned_by("run_12", "run_12_step_5", "run_12_llm_5").
param("run_12", "run_12_step_5", "start_x", "4.672").
param("run_12", "run_12_step_5", "start_y", "-1.962").
param("run_12", "run_12_step_5", "start_z", "0.462").
param("run_12", "run_12_step_5", "end_x", "4.320").
param("run_12", "run_12_step_5", "end_y", "-1.758").
param("run_12", "run_12_step_5", "end_z", "1.721").
param("run_12", "run_12_step_5", "duration_sec", "6.00").
param("run_12", "run_12_step_5", "mean_speed", "0.1153").
param("run_12", "run_12_step_5", "gripper_start", "3.31").
param("run_12", "run_12_step_5", "gripper_end", "3.31").
call("run_12", "run_12_llm_6", "llm_generate", "completed").
depends("run_12", "run_12_llm_6", "run_12_step_5").
call("run_12", "run_12_notify", "notify_user", "completed").
depends("run_12", "run_12_notify", "run_12_llm_6").
spawned_by("run_12", "run_12_notify", "run_12_llm_6").

% === Episode 12 (pick_and_place, 2 segments) ===
job("run_13").
job_status("run_13", "completed").
call("run_13", "run_13_step_0", "prompt_user", "completed").
call("run_13", "run_13_step_1", "llm_generate", "completed").
depends("run_13", "run_13_step_1", "run_13_step_0").
call("run_13", "run_13_llm_2", "llm_generate", "completed").
depends("run_13", "run_13_llm_2", "run_13_step_1").
call("run_13", "run_13_step_2", "release", "completed").
depends("run_13", "run_13_step_2", "run_13_llm_2").
spawned_by("run_13", "run_13_step_2", "run_13_llm_2").
param("run_13", "run_13_step_2", "start_x", "2.500").
param("run_13", "run_13_step_2", "start_y", "-0.000").
param("run_13", "run_13_step_2", "start_z", "0.200").
param("run_13", "run_13_step_2", "end_x", "4.804").
param("run_13", "run_13_step_2", "end_y", "-2.971").
param("run_13", "run_13_step_2", "end_z", "0.852").
param("run_13", "run_13_step_2", "duration_sec", "6.80").
param("run_13", "run_13_step_2", "mean_speed", "0.1529").
param("run_13", "run_13_step_2", "gripper_start", "6.03").
param("run_13", "run_13_step_2", "gripper_end", "7.97").
call("run_13", "run_13_llm_3", "llm_generate", "completed").
depends("run_13", "run_13_llm_3", "run_13_step_2").
call("run_13", "run_13_step_3", "idle", "completed").
depends("run_13", "run_13_step_3", "run_13_llm_3").
spawned_by("run_13", "run_13_step_3", "run_13_llm_3").
param("run_13", "run_13_step_3", "start_x", "4.802").
param("run_13", "run_13_step_3", "start_y", "-3.003").
param("run_13", "run_13_step_3", "start_z", "0.865").
param("run_13", "run_13_step_3", "end_x", "4.826").
param("run_13", "run_13_step_3", "end_y", "-3.000").
param("run_13", "run_13_step_3", "end_z", "0.869").
param("run_13", "run_13_step_3", "duration_sec", "3.00").
param("run_13", "run_13_step_3", "mean_speed", "0.0113").
param("run_13", "run_13_step_3", "gripper_start", "7.97").
param("run_13", "run_13_step_3", "gripper_end", "7.98").
call("run_13", "run_13_llm_4", "llm_generate", "completed").
depends("run_13", "run_13_llm_4", "run_13_step_3").
call("run_13", "run_13_notify", "notify_user", "completed").
depends("run_13", "run_13_notify", "run_13_llm_4").
spawned_by("run_13", "run_13_notify", "run_13_llm_4").

% === Episode 13 (pick_and_place, 6 segments) ===
job("run_14").
job_status("run_14", "completed").
call("run_14", "run_14_step_0", "prompt_user", "completed").
call("run_14", "run_14_step_1", "llm_generate", "completed").
depends("run_14", "run_14_step_1", "run_14_step_0").
call("run_14", "run_14_llm_2", "llm_generate", "completed").
depends("run_14", "run_14_llm_2", "run_14_step_1").
call("run_14", "run_14_step_2", "release", "completed").
depends("run_14", "run_14_step_2", "run_14_llm_2").
spawned_by("run_14", "run_14_step_2", "run_14_llm_2").
param("run_14", "run_14_step_2", "start_x", "2.500").
param("run_14", "run_14_step_2", "start_y", "-0.000").
param("run_14", "run_14_step_2", "start_z", "0.200").
param("run_14", "run_14_step_2", "end_x", "4.078").
param("run_14", "run_14_step_2", "end_y", "-1.246").
param("run_14", "run_14_step_2", "end_z", "0.150").
param("run_14", "run_14_step_2", "duration_sec", "1.60").
param("run_14", "run_14_step_2", "mean_speed", "0.2885").
param("run_14", "run_14_step_2", "gripper_start", "6.03").
param("run_14", "run_14_step_2", "gripper_end", "7.96").
call("run_14", "run_14_llm_3", "llm_generate", "completed").
depends("run_14", "run_14_llm_3", "run_14_step_2").
call("run_14", "run_14_step_3", "approach", "completed").
depends("run_14", "run_14_step_3", "run_14_llm_3").
spawned_by("run_14", "run_14_step_3", "run_14_llm_3").
param("run_14", "run_14_step_3", "start_x", "4.172").
param("run_14", "run_14_step_3", "start_y", "-1.418").
param("run_14", "run_14_step_3", "start_z", "0.129").
param("run_14", "run_14_step_3", "end_x", "4.328").
param("run_14", "run_14_step_3", "end_y", "-2.424").
param("run_14", "run_14_step_3", "end_z", "0.149").
param("run_14", "run_14_step_3", "duration_sec", "2.60").
param("run_14", "run_14_step_3", "mean_speed", "0.1153").
param("run_14", "run_14_step_3", "gripper_start", "7.96").
param("run_14", "run_14_step_3", "gripper_end", "7.96").
call("run_14", "run_14_llm_4", "llm_generate", "completed").
depends("run_14", "run_14_llm_4", "run_14_step_3").
call("run_14", "run_14_step_4", "transport", "completed").
depends("run_14", "run_14_step_4", "run_14_llm_4").
spawned_by("run_14", "run_14_step_4", "run_14_llm_4").
param("run_14", "run_14_step_4", "start_x", "4.108").
param("run_14", "run_14_step_4", "start_y", "-2.223").
param("run_14", "run_14_step_4", "start_z", "0.146").
param("run_14", "run_14_step_4", "end_x", "3.905").
param("run_14", "run_14_step_4", "end_y", "-1.825").
param("run_14", "run_14_step_4", "end_z", "0.210").
param("run_14", "run_14_step_4", "duration_sec", "0.60").
param("run_14", "run_14_step_4", "mean_speed", "0.1730").
param("run_14", "run_14_step_4", "gripper_start", "7.90").
param("run_14", "run_14_step_4", "gripper_end", "7.15").
call("run_14", "run_14_llm_5", "llm_generate", "completed").
depends("run_14", "run_14_llm_5", "run_14_step_4").
call("run_14", "run_14_step_5", "grasp", "completed").
depends("run_14", "run_14_step_5", "run_14_llm_5").
spawned_by("run_14", "run_14_step_5", "run_14_llm_5").
param("run_14", "run_14_step_5", "start_x", "3.906").
param("run_14", "run_14_step_5", "start_y", "-1.873").
param("run_14", "run_14_step_5", "start_z", "0.184").
param("run_14", "run_14_step_5", "end_x", "4.146").
param("run_14", "run_14_step_5", "end_y", "-2.108").
param("run_14", "run_14_step_5", "end_z", "0.233").
param("run_14", "run_14_step_5", "duration_sec", "1.20").
param("run_14", "run_14_step_5", "mean_speed", "0.1316").
param("run_14", "run_14_step_5", "gripper_start", "7.15").
param("run_14", "run_14_step_5", "gripper_end", "3.27").
call("run_14", "run_14_llm_6", "llm_generate", "completed").
depends("run_14", "run_14_llm_6", "run_14_step_5").
call("run_14", "run_14_step_6", "transport", "completed").
depends("run_14", "run_14_step_6", "run_14_llm_6").
spawned_by("run_14", "run_14_step_6", "run_14_llm_6").
param("run_14", "run_14_step_6", "start_x", "4.278").
param("run_14", "run_14_step_6", "start_y", "-2.044").
param("run_14", "run_14_step_6", "start_z", "0.338").
param("run_14", "run_14_step_6", "end_x", "4.917").
param("run_14", "run_14_step_6", "end_y", "-2.152").
param("run_14", "run_14_step_6", "end_z", "1.160").
param("run_14", "run_14_step_6", "duration_sec", "2.80").
param("run_14", "run_14_step_6", "mean_speed", "0.2182").
param("run_14", "run_14_step_6", "gripper_start", "3.27").
param("run_14", "run_14_step_6", "gripper_end", "3.27").
call("run_14", "run_14_llm_7", "llm_generate", "completed").
depends("run_14", "run_14_llm_7", "run_14_step_6").
call("run_14", "run_14_step_7", "grasp", "completed").
depends("run_14", "run_14_step_7", "run_14_llm_7").
spawned_by("run_14", "run_14_step_7", "run_14_llm_7").
param("run_14", "run_14_step_7", "start_x", "5.153").
param("run_14", "run_14_step_7", "start_y", "-2.148").
param("run_14", "run_14_step_7", "start_z", "1.191").
param("run_14", "run_14_step_7", "end_x", "5.685").
param("run_14", "run_14_step_7", "end_y", "-2.378").
param("run_14", "run_14_step_7", "end_z", "1.551").
param("run_14", "run_14_step_7", "duration_sec", "1.00").
param("run_14", "run_14_step_7", "mean_speed", "0.1788").
param("run_14", "run_14_step_7", "gripper_start", "2.84").
param("run_14", "run_14_step_7", "gripper_end", "0.62").
call("run_14", "run_14_llm_8", "llm_generate", "completed").
depends("run_14", "run_14_llm_8", "run_14_step_7").
call("run_14", "run_14_notify", "notify_user", "completed").
depends("run_14", "run_14_notify", "run_14_llm_8").
spawned_by("run_14", "run_14_notify", "run_14_llm_8").

% === Episode 14 (pick_and_place, 3 segments) ===
job("run_15").
job_status("run_15", "completed").
call("run_15", "run_15_step_0", "prompt_user", "completed").
call("run_15", "run_15_step_1", "llm_generate", "completed").
depends("run_15", "run_15_step_1", "run_15_step_0").
call("run_15", "run_15_llm_2", "llm_generate", "completed").
depends("run_15", "run_15_llm_2", "run_15_step_1").
call("run_15", "run_15_step_2", "release", "completed").
depends("run_15", "run_15_step_2", "run_15_llm_2").
spawned_by("run_15", "run_15_step_2", "run_15_llm_2").
param("run_15", "run_15_step_2", "start_x", "2.500").
param("run_15", "run_15_step_2", "start_y", "-0.000").
param("run_15", "run_15_step_2", "start_z", "0.200").
param("run_15", "run_15_step_2", "end_x", "4.449").
param("run_15", "run_15_step_2", "end_y", "-1.457").
param("run_15", "run_15_step_2", "end_z", "0.157").
param("run_15", "run_15_step_2", "duration_sec", "2.40").
param("run_15", "run_15_step_2", "mean_speed", "0.2256").
param("run_15", "run_15_step_2", "gripper_start", "6.03").
param("run_15", "run_15_step_2", "gripper_end", "7.96").
call("run_15", "run_15_llm_3", "llm_generate", "completed").
depends("run_15", "run_15_llm_3", "run_15_step_2").
call("run_15", "run_15_step_3", "approach", "completed").
depends("run_15", "run_15_step_3", "run_15_llm_3").
spawned_by("run_15", "run_15_step_3", "run_15_llm_3").
param("run_15", "run_15_step_3", "start_x", "4.396").
param("run_15", "run_15_step_3", "start_y", "-1.555").
param("run_15", "run_15_step_3", "start_z", "0.160").
param("run_15", "run_15_step_3", "end_x", "4.639").
param("run_15", "run_15_step_3", "end_y", "-1.663").
param("run_15", "run_15_step_3", "end_z", "0.155").
param("run_15", "run_15_step_3", "duration_sec", "1.20").
param("run_15", "run_15_step_3", "mean_speed", "0.0568").
param("run_15", "run_15_step_3", "gripper_start", "7.96").
param("run_15", "run_15_step_3", "gripper_end", "7.97").
call("run_15", "run_15_llm_4", "llm_generate", "completed").
depends("run_15", "run_15_llm_4", "run_15_step_3").
call("run_15", "run_15_step_4", "grasp", "completed").
depends("run_15", "run_15_step_4", "run_15_llm_4").
spawned_by("run_15", "run_15_step_4", "run_15_llm_4").
param("run_15", "run_15_step_4", "start_x", "4.680").
param("run_15", "run_15_step_4", "start_y", "-1.669").
param("run_15", "run_15_step_4", "start_z", "0.150").
param("run_15", "run_15_step_4", "end_x", "5.007").
param("run_15", "run_15_step_4", "end_y", "-2.362").
param("run_15", "run_15_step_4", "end_z", "0.725").
param("run_15", "run_15_step_4", "duration_sec", "6.20").
param("run_15", "run_15_step_4", "mean_speed", "0.0259").
param("run_15", "run_15_step_4", "gripper_start", "7.97").
param("run_15", "run_15_step_4", "gripper_end", "6.05").
call("run_15", "run_15_llm_5", "llm_generate", "completed").
depends("run_15", "run_15_llm_5", "run_15_step_4").
call("run_15", "run_15_notify", "notify_user", "completed").
depends("run_15", "run_15_notify", "run_15_llm_5").
spawned_by("run_15", "run_15_notify", "run_15_llm_5").

% === Episode 15 (pick_and_place, 2 segments) ===
job("run_16").
job_status("run_16", "completed").
call("run_16", "run_16_step_0", "prompt_user", "completed").
call("run_16", "run_16_step_1", "llm_generate", "completed").
depends("run_16", "run_16_step_1", "run_16_step_0").
call("run_16", "run_16_llm_2", "llm_generate", "completed").
depends("run_16", "run_16_llm_2", "run_16_step_1").
call("run_16", "run_16_step_2", "reach", "completed").
depends("run_16", "run_16_step_2", "run_16_llm_2").
spawned_by("run_16", "run_16_step_2", "run_16_llm_2").
param("run_16", "run_16_step_2", "start_x", "0.900").
param("run_16", "run_16_step_2", "start_y", "2.000").
param("run_16", "run_16_step_2", "start_z", "2.000").
param("run_16", "run_16_step_2", "end_x", "1.340").
param("run_16", "run_16_step_2", "end_y", "2.053").
param("run_16", "run_16_step_2", "end_z", "2.000").
param("run_16", "run_16_step_2", "duration_sec", "0.60").
param("run_16", "run_16_step_2", "mean_speed", "0.2413").
param("run_16", "run_16_step_2", "gripper_start", "4.02").
param("run_16", "run_16_step_2", "gripper_end", "4.01").
call("run_16", "run_16_llm_3", "llm_generate", "completed").
depends("run_16", "run_16_llm_3", "run_16_step_2").
call("run_16", "run_16_step_3", "move", "completed").
depends("run_16", "run_16_step_3", "run_16_llm_3").
spawned_by("run_16", "run_16_step_3", "run_16_llm_3").
param("run_16", "run_16_step_3", "start_x", "1.616").
param("run_16", "run_16_step_3", "start_y", "2.096").
param("run_16", "run_16_step_3", "start_z", "1.999").
param("run_16", "run_16_step_3", "end_x", "1.586").
param("run_16", "run_16_step_3", "end_y", "1.815").
param("run_16", "run_16_step_3", "end_z", "1.979").
param("run_16", "run_16_step_3", "duration_sec", "9.20").
param("run_16", "run_16_step_3", "mean_speed", "0.2059").
param("run_16", "run_16_step_3", "gripper_start", "4.01").
param("run_16", "run_16_step_3", "gripper_end", "4.00").
call("run_16", "run_16_llm_4", "llm_generate", "completed").
depends("run_16", "run_16_llm_4", "run_16_step_3").
call("run_16", "run_16_notify", "notify_user", "completed").
depends("run_16", "run_16_notify", "run_16_llm_4").
spawned_by("run_16", "run_16_notify", "run_16_llm_4").

% === Episode 16 (pick_and_place, 5 segments) ===
job("run_17").
job_status("run_17", "completed").
call("run_17", "run_17_step_0", "prompt_user", "completed").
call("run_17", "run_17_step_1", "llm_generate", "completed").
depends("run_17", "run_17_step_1", "run_17_step_0").
call("run_17", "run_17_llm_2", "llm_generate", "completed").
depends("run_17", "run_17_llm_2", "run_17_step_1").
call("run_17", "run_17_step_2", "release", "completed").
depends("run_17", "run_17_step_2", "run_17_llm_2").
spawned_by("run_17", "run_17_step_2", "run_17_llm_2").
param("run_17", "run_17_step_2", "start_x", "2.500").
param("run_17", "run_17_step_2", "start_y", "-0.000").
param("run_17", "run_17_step_2", "start_z", "0.200").
param("run_17", "run_17_step_2", "end_x", "4.159").
param("run_17", "run_17_step_2", "end_y", "-2.018").
param("run_17", "run_17_step_2", "end_z", "0.145").
param("run_17", "run_17_step_2", "duration_sec", "1.80").
param("run_17", "run_17_step_2", "mean_speed", "0.3381").
param("run_17", "run_17_step_2", "gripper_start", "6.04").
param("run_17", "run_17_step_2", "gripper_end", "7.96").
call("run_17", "run_17_llm_3", "llm_generate", "completed").
depends("run_17", "run_17_llm_3", "run_17_step_2").
call("run_17", "run_17_step_3", "grasp", "completed").
depends("run_17", "run_17_step_3", "run_17_llm_3").
spawned_by("run_17", "run_17_step_3", "run_17_llm_3").
param("run_17", "run_17_step_3", "start_x", "4.227").
param("run_17", "run_17_step_3", "start_y", "-2.412").
param("run_17", "run_17_step_3", "start_z", "0.139").
param("run_17", "run_17_step_3", "end_x", "4.143").
param("run_17", "run_17_step_3", "end_y", "-2.884").
param("run_17", "run_17_step_3", "end_z", "0.147").
param("run_17", "run_17_step_3", "duration_sec", "1.60").
param("run_17", "run_17_step_3", "mean_speed", "0.2002").
param("run_17", "run_17_step_3", "gripper_start", "7.96").
param("run_17", "run_17_step_3", "gripper_end", "5.97").
call("run_17", "run_17_llm_4", "llm_generate", "completed").
depends("run_17", "run_17_llm_4", "run_17_step_3").
call("run_17", "run_17_step_4", "transport", "completed").
depends("run_17", "run_17_step_4", "run_17_llm_4").
spawned_by("run_17", "run_17_step_4", "run_17_llm_4").
param("run_17", "run_17_step_4", "start_x", "4.142").
param("run_17", "run_17_step_4", "start_y", "-2.821").
param("run_17", "run_17_step_4", "start_z", "0.131").
param("run_17", "run_17_step_4", "end_x", "4.207").
param("run_17", "run_17_step_4", "end_y", "-1.414").
param("run_17", "run_17_step_4", "end_z", "0.430").
param("run_17", "run_17_step_4", "duration_sec", "1.80").
param("run_17", "run_17_step_4", "mean_speed", "0.2601").
param("run_17", "run_17_step_4", "gripper_start", "3.22").
param("run_17", "run_17_step_4", "gripper_end", "3.73").
call("run_17", "run_17_llm_5", "llm_generate", "completed").
depends("run_17", "run_17_llm_5", "run_17_step_4").
call("run_17", "run_17_step_5", "release", "completed").
depends("run_17", "run_17_step_5", "run_17_llm_5").
spawned_by("run_17", "run_17_step_5", "run_17_llm_5").
param("run_17", "run_17_step_5", "start_x", "3.896").
param("run_17", "run_17_step_5", "start_y", "-1.102").
param("run_17", "run_17_step_5", "start_z", "0.443").
param("run_17", "run_17_step_5", "end_x", "4.149").
param("run_17", "run_17_step_5", "end_y", "-0.972").
param("run_17", "run_17_step_5", "end_z", "0.115").
param("run_17", "run_17_step_5", "duration_sec", "2.20").
param("run_17", "run_17_step_5", "mean_speed", "0.1289").
param("run_17", "run_17_step_5", "gripper_start", "4.16").
param("run_17", "run_17_step_5", "gripper_end", "6.25").
call("run_17", "run_17_llm_6", "llm_generate", "completed").
depends("run_17", "run_17_llm_6", "run_17_step_5").
call("run_17", "run_17_step_6", "grasp", "completed").
depends("run_17", "run_17_step_6", "run_17_llm_6").
spawned_by("run_17", "run_17_step_6", "run_17_llm_6").
param("run_17", "run_17_step_6", "start_x", "4.371").
param("run_17", "run_17_step_6", "start_y", "-1.281").
param("run_17", "run_17_step_6", "start_z", "0.101").
param("run_17", "run_17_step_6", "end_x", "3.275").
param("run_17", "run_17_step_6", "end_y", "-1.386").
param("run_17", "run_17_step_6", "end_z", "0.648").
param("run_17", "run_17_step_6", "duration_sec", "2.40").
param("run_17", "run_17_step_6", "mean_speed", "0.1325").
param("run_17", "run_17_step_6", "gripper_start", "7.96").
param("run_17", "run_17_step_6", "gripper_end", "6.53").
call("run_17", "run_17_llm_7", "llm_generate", "completed").
depends("run_17", "run_17_llm_7", "run_17_step_6").
call("run_17", "run_17_notify", "notify_user", "completed").
depends("run_17", "run_17_notify", "run_17_llm_7").
spawned_by("run_17", "run_17_notify", "run_17_llm_7").

% === Episode 17 (pick_and_place, 8 segments) ===
job("run_18").
job_status("run_18", "completed").
call("run_18", "run_18_step_0", "prompt_user", "completed").
call("run_18", "run_18_step_1", "llm_generate", "completed").
depends("run_18", "run_18_step_1", "run_18_step_0").
call("run_18", "run_18_llm_2", "llm_generate", "completed").
depends("run_18", "run_18_llm_2", "run_18_step_1").
call("run_18", "run_18_step_2", "transport", "completed").
depends("run_18", "run_18_step_2", "run_18_llm_2").
spawned_by("run_18", "run_18_step_2", "run_18_llm_2").
param("run_18", "run_18_step_2", "start_x", "0.900").
param("run_18", "run_18_step_2", "start_y", "2.000").
param("run_18", "run_18_step_2", "start_z", "2.000").
param("run_18", "run_18_step_2", "end_x", "0.988").
param("run_18", "run_18_step_2", "end_y", "2.253").
param("run_18", "run_18_step_2", "end_z", "1.911").
param("run_18", "run_18_step_2", "duration_sec", "0.60").
param("run_18", "run_18_step_2", "mean_speed", "0.1398").
param("run_18", "run_18_step_2", "gripper_start", "4.00").
param("run_18", "run_18_step_2", "gripper_end", "3.34").
call("run_18", "run_18_llm_3", "llm_generate", "completed").
depends("run_18", "run_18_llm_3", "run_18_step_2").
call("run_18", "run_18_step_3", "grasp", "completed").
depends("run_18", "run_18_step_3", "run_18_llm_3").
spawned_by("run_18", "run_18_step_3", "run_18_llm_3").
param("run_18", "run_18_step_3", "start_x", "0.929").
param("run_18", "run_18_step_3", "start_y", "2.137").
param("run_18", "run_18_step_3", "start_z", "1.878").
param("run_18", "run_18_step_3", "end_x", "1.187").
param("run_18", "run_18_step_3", "end_y", "2.350").
param("run_18", "run_18_step_3", "end_z", "1.958").
param("run_18", "run_18_step_3", "duration_sec", "1.40").
param("run_18", "run_18_step_3", "mean_speed", "0.1224").
param("run_18", "run_18_step_3", "gripper_start", "3.34").
param("run_18", "run_18_step_3", "gripper_end", "2.23").
call("run_18", "run_18_llm_4", "llm_generate", "completed").
depends("run_18", "run_18_llm_4", "run_18_step_3").
call("run_18", "run_18_step_4", "transport", "completed").
depends("run_18", "run_18_step_4", "run_18_llm_4").
spawned_by("run_18", "run_18_step_4", "run_18_llm_4").
param("run_18", "run_18_step_4", "start_x", "0.972").
param("run_18", "run_18_step_4", "start_y", "2.273").
param("run_18", "run_18_step_4", "start_z", "2.002").
param("run_18", "run_18_step_4", "end_x", "0.774").
param("run_18", "run_18_step_4", "end_y", "2.013").
param("run_18", "run_18_step_4", "end_z", "1.921").
param("run_18", "run_18_step_4", "duration_sec", "2.00").
param("run_18", "run_18_step_4", "mean_speed", "0.1617").
param("run_18", "run_18_step_4", "gripper_start", "1.22").
param("run_18", "run_18_step_4", "gripper_end", "1.15").
call("run_18", "run_18_llm_5", "llm_generate", "completed").
depends("run_18", "run_18_llm_5", "run_18_step_4").
call("run_18", "run_18_step_5", "release", "completed").
depends("run_18", "run_18_step_5", "run_18_llm_5").
spawned_by("run_18", "run_18_step_5", "run_18_llm_5").
param("run_18", "run_18_step_5", "start_x", "0.846").
param("run_18", "run_18_step_5", "start_y", "2.053").
param("run_18", "run_18_step_5", "start_z", "1.833").
param("run_18", "run_18_step_5", "end_x", "1.015").
param("run_18", "run_18_step_5", "end_y", "1.965").
param("run_18", "run_18_step_5", "end_z", "1.658").
param("run_18", "run_18_step_5", "duration_sec", "1.20").
param("run_18", "run_18_step_5", "mean_speed", "0.1495").
param("run_18", "run_18_step_5", "gripper_start", "0.58").
param("run_18", "run_18_step_5", "gripper_end", "1.96").
call("run_18", "run_18_llm_6", "llm_generate", "completed").
depends("run_18", "run_18_llm_6", "run_18_step_5").
call("run_18", "run_18_step_6", "retract", "completed").
depends("run_18", "run_18_step_6", "run_18_llm_6").
spawned_by("run_18", "run_18_step_6", "run_18_llm_6").
param("run_18", "run_18_step_6", "start_x", "1.017").
param("run_18", "run_18_step_6", "start_y", "1.801").
param("run_18", "run_18_step_6", "start_z", "1.575").
param("run_18", "run_18_step_6", "end_x", "1.091").
param("run_18", "run_18_step_6", "end_y", "2.160").
param("run_18", "run_18_step_6", "end_z", "1.413").
param("run_18", "run_18_step_6", "duration_sec", "1.20").
param("run_18", "run_18_step_6", "mean_speed", "0.1857").
param("run_18", "run_18_step_6", "gripper_start", "1.96").
param("run_18", "run_18_step_6", "gripper_end", "2.24").
call("run_18", "run_18_llm_7", "llm_generate", "completed").
depends("run_18", "run_18_llm_7", "run_18_step_6").
call("run_18", "run_18_step_7", "transport", "completed").
depends("run_18", "run_18_step_7", "run_18_llm_7").
spawned_by("run_18", "run_18_step_7", "run_18_llm_7").
param("run_18", "run_18_step_7", "start_x", "0.909").
param("run_18", "run_18_step_7", "start_y", "2.027").
param("run_18", "run_18_step_7", "start_z", "1.443").
param("run_18", "run_18_step_7", "end_x", "0.888").
param("run_18", "run_18_step_7", "end_y", "2.056").
param("run_18", "run_18_step_7", "end_z", "1.512").
param("run_18", "run_18_step_7", "duration_sec", "0.60").
param("run_18", "run_18_step_7", "mean_speed", "0.1102").
param("run_18", "run_18_step_7", "gripper_start", "2.86").
param("run_18", "run_18_step_7", "gripper_end", "2.22").
call("run_18", "run_18_llm_8", "llm_generate", "completed").
depends("run_18", "run_18_llm_8", "run_18_step_7").
call("run_18", "run_18_step_8", "grasp", "completed").
depends("run_18", "run_18_step_8", "run_18_llm_8").
spawned_by("run_18", "run_18_step_8", "run_18_llm_8").
param("run_18", "run_18_step_8", "start_x", "0.966").
param("run_18", "run_18_step_8", "start_y", "2.012").
param("run_18", "run_18_step_8", "start_z", "1.627").
param("run_18", "run_18_step_8", "end_x", "1.195").
param("run_18", "run_18_step_8", "end_y", "2.059").
param("run_18", "run_18_step_8", "end_z", "1.755").
param("run_18", "run_18_step_8", "duration_sec", "1.80").
param("run_18", "run_18_step_8", "mean_speed", "0.1216").
param("run_18", "run_18_step_8", "gripper_start", "2.22").
param("run_18", "run_18_step_8", "gripper_end", "0.04").
call("run_18", "run_18_llm_9", "llm_generate", "completed").
depends("run_18", "run_18_llm_9", "run_18_step_8").
call("run_18", "run_18_step_9", "retract", "completed").
depends("run_18", "run_18_step_9", "run_18_llm_9").
spawned_by("run_18", "run_18_step_9", "run_18_llm_9").
param("run_18", "run_18_step_9", "start_x", "1.132").
param("run_18", "run_18_step_9", "start_y", "2.110").
param("run_18", "run_18_step_9", "start_z", "1.685").
param("run_18", "run_18_step_9", "end_x", "1.157").
param("run_18", "run_18_step_9", "end_y", "1.925").
param("run_18", "run_18_step_9", "end_z", "1.900").
param("run_18", "run_18_step_9", "duration_sec", "1.00").
param("run_18", "run_18_step_9", "mean_speed", "0.1023").
param("run_18", "run_18_step_9", "gripper_start", "0.04").
param("run_18", "run_18_step_9", "gripper_end", "0.74").
call("run_18", "run_18_llm_10", "llm_generate", "completed").
depends("run_18", "run_18_llm_10", "run_18_step_9").
call("run_18", "run_18_notify", "notify_user", "completed").
depends("run_18", "run_18_notify", "run_18_llm_10").
spawned_by("run_18", "run_18_notify", "run_18_llm_10").

% === Episode 18 (pick_and_place, 6 segments) ===
job("run_19").
job_status("run_19", "completed").
call("run_19", "run_19_step_0", "prompt_user", "completed").
call("run_19", "run_19_step_1", "llm_generate", "completed").
depends("run_19", "run_19_step_1", "run_19_step_0").
call("run_19", "run_19_llm_2", "llm_generate", "completed").
depends("run_19", "run_19_llm_2", "run_19_step_1").
call("run_19", "run_19_step_2", "approach", "completed").
depends("run_19", "run_19_step_2", "run_19_llm_2").
spawned_by("run_19", "run_19_step_2", "run_19_llm_2").
param("run_19", "run_19_step_2", "start_x", "0.900").
param("run_19", "run_19_step_2", "start_y", "2.000").
param("run_19", "run_19_step_2", "start_z", "2.000").
param("run_19", "run_19_step_2", "end_x", "1.641").
param("run_19", "run_19_step_2", "end_y", "2.351").
param("run_19", "run_19_step_2", "end_z", "1.683").
param("run_19", "run_19_step_2", "duration_sec", "1.60").
param("run_19", "run_19_step_2", "mean_speed", "0.1285").
param("run_19", "run_19_step_2", "gripper_start", "3.98").
param("run_19", "run_19_step_2", "gripper_end", "3.98").
call("run_19", "run_19_llm_3", "llm_generate", "completed").
depends("run_19", "run_19_llm_3", "run_19_step_2").
call("run_19", "run_19_step_3", "grasp", "completed").
depends("run_19", "run_19_step_3", "run_19_llm_3").
spawned_by("run_19", "run_19_step_3", "run_19_llm_3").
param("run_19", "run_19_step_3", "start_x", "1.680").
param("run_19", "run_19_step_3", "start_y", "2.416").
param("run_19", "run_19_step_3", "start_z", "1.562").
param("run_19", "run_19_step_3", "end_x", "3.574").
param("run_19", "run_19_step_3", "end_y", "2.556").
param("run_19", "run_19_step_3", "end_z", "1.523").
param("run_19", "run_19_step_3", "duration_sec", "3.40").
param("run_19", "run_19_step_3", "mean_speed", "0.1652").
param("run_19", "run_19_step_3", "gripper_start", "3.98").
param("run_19", "run_19_step_3", "gripper_end", "0.03").
call("run_19", "run_19_llm_4", "llm_generate", "completed").
depends("run_19", "run_19_llm_4", "run_19_step_3").
call("run_19", "run_19_step_4", "transport", "completed").
depends("run_19", "run_19_step_4", "run_19_llm_4").
spawned_by("run_19", "run_19_step_4", "run_19_llm_4").
param("run_19", "run_19_step_4", "start_x", "3.863").
param("run_19", "run_19_step_4", "start_y", "2.563").
param("run_19", "run_19_step_4", "start_z", "1.430").
param("run_19", "run_19_step_4", "end_x", "4.115").
param("run_19", "run_19_step_4", "end_y", "2.591").
param("run_19", "run_19_step_4", "end_z", "1.194").
param("run_19", "run_19_step_4", "duration_sec", "0.80").
param("run_19", "run_19_step_4", "mean_speed", "0.1025").
param("run_19", "run_19_step_4", "gripper_start", "0.03").
param("run_19", "run_19_step_4", "gripper_end", "0.06").
call("run_19", "run_19_llm_5", "llm_generate", "completed").
depends("run_19", "run_19_llm_5", "run_19_step_4").
call("run_19", "run_19_step_5", "hold", "completed").
depends("run_19", "run_19_step_5", "run_19_llm_5").
spawned_by("run_19", "run_19_step_5", "run_19_llm_5").
param("run_19", "run_19_step_5", "start_x", "4.137").
param("run_19", "run_19_step_5", "start_y", "2.616").
param("run_19", "run_19_step_5", "start_z", "1.198").
param("run_19", "run_19_step_5", "end_x", "4.158").
param("run_19", "run_19_step_5", "end_y", "2.683").
param("run_19", "run_19_step_5", "end_z", "1.201").
param("run_19", "run_19_step_5", "duration_sec", "0.80").
param("run_19", "run_19_step_5", "mean_speed", "0.0244").
param("run_19", "run_19_step_5", "gripper_start", "0.06").
param("run_19", "run_19_step_5", "gripper_end", "0.05").
call("run_19", "run_19_llm_6", "llm_generate", "completed").
depends("run_19", "run_19_llm_6", "run_19_step_5").
call("run_19", "run_19_step_6", "idle", "completed").
depends("run_19", "run_19_step_6", "run_19_llm_6").
spawned_by("run_19", "run_19_step_6", "run_19_llm_6").
param("run_19", "run_19_step_6", "start_x", "4.163").
param("run_19", "run_19_step_6", "start_y", "2.708").
param("run_19", "run_19_step_6", "start_z", "1.200").
param("run_19", "run_19_step_6", "end_x", "4.168").
param("run_19", "run_19_step_6", "end_y", "2.737").
param("run_19", "run_19_step_6", "end_z", "1.200").
param("run_19", "run_19_step_6", "duration_sec", "0.60").
param("run_19", "run_19_step_6", "mean_speed", "0.0194").
param("run_19", "run_19_step_6", "gripper_start", "0.05").
param("run_19", "run_19_step_6", "gripper_end", "0.05").
call("run_19", "run_19_llm_7", "llm_generate", "completed").
depends("run_19", "run_19_llm_7", "run_19_step_6").
call("run_19", "run_19_step_7", "release", "completed").
depends("run_19", "run_19_step_7", "run_19_llm_7").
spawned_by("run_19", "run_19_step_7", "run_19_llm_7").
param("run_19", "run_19_step_7", "start_x", "4.173").
param("run_19", "run_19_step_7", "start_y", "2.765").
param("run_19", "run_19_step_7", "start_z", "1.200").
param("run_19", "run_19_step_7", "end_x", "3.721").
param("run_19", "run_19_step_7", "end_y", "2.756").
param("run_19", "run_19_step_7", "end_z", "1.200").
param("run_19", "run_19_step_7", "duration_sec", "2.60").
param("run_19", "run_19_step_7", "mean_speed", "0.0421").
param("run_19", "run_19_step_7", "gripper_start", "0.05").
param("run_19", "run_19_step_7", "gripper_end", "3.96").
call("run_19", "run_19_llm_8", "llm_generate", "completed").
depends("run_19", "run_19_llm_8", "run_19_step_7").
call("run_19", "run_19_notify", "notify_user", "completed").
depends("run_19", "run_19_notify", "run_19_llm_8").
spawned_by("run_19", "run_19_notify", "run_19_llm_8").

% === Episode 19 (pick_and_place, 2 segments) ===
job("run_20").
job_status("run_20", "completed").
call("run_20", "run_20_step_0", "prompt_user", "completed").
call("run_20", "run_20_step_1", "llm_generate", "completed").
depends("run_20", "run_20_step_1", "run_20_step_0").
call("run_20", "run_20_llm_2", "llm_generate", "completed").
depends("run_20", "run_20_llm_2", "run_20_step_1").
call("run_20", "run_20_step_2", "release", "completed").
depends("run_20", "run_20_step_2", "run_20_llm_2").
spawned_by("run_20", "run_20_step_2", "run_20_llm_2").
param("run_20", "run_20_step_2", "start_x", "2.500").
param("run_20", "run_20_step_2", "start_y", "-0.000").
param("run_20", "run_20_step_2", "start_z", "0.200").
param("run_20", "run_20_step_2", "end_x", "4.676").
param("run_20", "run_20_step_2", "end_y", "-1.520").
param("run_20", "run_20_step_2", "end_z", "0.214").
param("run_20", "run_20_step_2", "duration_sec", "2.60").
param("run_20", "run_20_step_2", "mean_speed", "0.2270").
param("run_20", "run_20_step_2", "gripper_start", "6.04").
param("run_20", "run_20_step_2", "gripper_end", "7.96").
call("run_20", "run_20_llm_3", "llm_generate", "completed").
depends("run_20", "run_20_llm_3", "run_20_step_2").
call("run_20", "run_20_step_3", "grasp", "completed").
depends("run_20", "run_20_step_3", "run_20_llm_3").
spawned_by("run_20", "run_20_step_3", "run_20_llm_3").
param("run_20", "run_20_step_3", "start_x", "4.758").
param("run_20", "run_20_step_3", "start_y", "-1.668").
param("run_20", "run_20_step_3", "start_z", "0.255").
param("run_20", "run_20_step_3", "end_x", "4.242").
param("run_20", "run_20_step_3", "end_y", "-1.931").
param("run_20", "run_20_step_3", "end_z", "1.370").
param("run_20", "run_20_step_3", "duration_sec", "7.20").
param("run_20", "run_20_step_3", "mean_speed", "0.0668").
param("run_20", "run_20_step_3", "gripper_start", "7.96").
param("run_20", "run_20_step_3", "gripper_end", "5.13").
call("run_20", "run_20_llm_4", "llm_generate", "completed").
depends("run_20", "run_20_llm_4", "run_20_step_3").
call("run_20", "run_20_notify", "notify_user", "completed").
depends("run_20", "run_20_notify", "run_20_llm_4").
spawned_by("run_20", "run_20_notify", "run_20_llm_4").

% === Episode 20 (pick_and_place, 3 segments) ===
job("run_21").
job_status("run_21", "completed").
call("run_21", "run_21_step_0", "prompt_user", "completed").
call("run_21", "run_21_step_1", "llm_generate", "completed").
depends("run_21", "run_21_step_1", "run_21_step_0").
call("run_21", "run_21_llm_2", "llm_generate", "completed").
depends("run_21", "run_21_llm_2", "run_21_step_1").
call("run_21", "run_21_step_2", "release", "completed").
depends("run_21", "run_21_step_2", "run_21_llm_2").
spawned_by("run_21", "run_21_step_2", "run_21_llm_2").
param("run_21", "run_21_step_2", "start_x", "2.500").
param("run_21", "run_21_step_2", "start_y", "-0.000").
param("run_21", "run_21_step_2", "start_z", "0.200").
param("run_21", "run_21_step_2", "end_x", "2.779").
param("run_21", "run_21_step_2", "end_y", "-0.782").
param("run_21", "run_21_step_2", "end_z", "0.699").
param("run_21", "run_21_step_2", "duration_sec", "1.20").
param("run_21", "run_21_step_2", "mean_speed", "0.2088").
param("run_21", "run_21_step_2", "gripper_start", "6.03").
param("run_21", "run_21_step_2", "gripper_end", "7.96").
call("run_21", "run_21_llm_3", "llm_generate", "completed").
depends("run_21", "run_21_llm_3", "run_21_step_2").
call("run_21", "run_21_step_3", "move", "completed").
depends("run_21", "run_21_step_3", "run_21_llm_3").
spawned_by("run_21", "run_21_step_3", "run_21_llm_3").
param("run_21", "run_21_step_3", "start_x", "2.871").
param("run_21", "run_21_step_3", "start_y", "-0.981").
param("run_21", "run_21_step_3", "start_z", "0.878").
param("run_21", "run_21_step_3", "end_x", "2.500").
param("run_21", "run_21_step_3", "end_y", "-2.087").
param("run_21", "run_21_step_3", "end_z", "1.999").
param("run_21", "run_21_step_3", "duration_sec", "5.80").
param("run_21", "run_21_step_3", "mean_speed", "0.0810").
param("run_21", "run_21_step_3", "gripper_start", "7.96").
param("run_21", "run_21_step_3", "gripper_end", "7.97").
call("run_21", "run_21_llm_4", "llm_generate", "completed").
depends("run_21", "run_21_llm_4", "run_21_step_3").
call("run_21", "run_21_step_4", "idle", "completed").
depends("run_21", "run_21_step_4", "run_21_llm_4").
spawned_by("run_21", "run_21_step_4", "run_21_llm_4").
param("run_21", "run_21_step_4", "start_x", "2.500").
param("run_21", "run_21_step_4", "start_y", "-2.119").
param("run_21", "run_21_step_4", "start_z", "2.000").
param("run_21", "run_21_step_4", "end_x", "2.501").
param("run_21", "run_21_step_4", "end_y", "-2.183").
param("run_21", "run_21_step_4", "end_z", "2.000").
param("run_21", "run_21_step_4", "duration_sec", "2.80").
param("run_21", "run_21_step_4", "mean_speed", "0.0090").
param("run_21", "run_21_step_4", "gripper_start", "7.97").
param("run_21", "run_21_step_4", "gripper_end", "7.98").
call("run_21", "run_21_llm_5", "llm_generate", "completed").
depends("run_21", "run_21_llm_5", "run_21_step_4").
call("run_21", "run_21_notify", "notify_user", "completed").
depends("run_21", "run_21_notify", "run_21_llm_5").
spawned_by("run_21", "run_21_notify", "run_21_llm_5").

% === Episode 21 (pick_and_place, 3 segments) ===
job("run_22").
job_status("run_22", "completed").
call("run_22", "run_22_step_0", "prompt_user", "completed").
call("run_22", "run_22_step_1", "llm_generate", "completed").
depends("run_22", "run_22_step_1", "run_22_step_0").
call("run_22", "run_22_llm_2", "llm_generate", "completed").
depends("run_22", "run_22_llm_2", "run_22_step_1").
call("run_22", "run_22_step_2", "release", "completed").
depends("run_22", "run_22_step_2", "run_22_llm_2").
spawned_by("run_22", "run_22_step_2", "run_22_llm_2").
param("run_22", "run_22_step_2", "start_x", "2.500").
param("run_22", "run_22_step_2", "start_y", "-0.000").
param("run_22", "run_22_step_2", "start_z", "0.200").
param("run_22", "run_22_step_2", "end_x", "3.181").
param("run_22", "run_22_step_2", "end_y", "-0.972").
param("run_22", "run_22_step_2", "end_z", "0.154").
param("run_22", "run_22_step_2", "duration_sec", "1.00").
param("run_22", "run_22_step_2", "mean_speed", "0.2803").
param("run_22", "run_22_step_2", "gripper_start", "6.03").
param("run_22", "run_22_step_2", "gripper_end", "7.96").
call("run_22", "run_22_llm_3", "llm_generate", "completed").
depends("run_22", "run_22_llm_3", "run_22_step_2").
call("run_22", "run_22_step_3", "grasp", "completed").
depends("run_22", "run_22_step_3", "run_22_llm_3").
spawned_by("run_22", "run_22_step_3", "run_22_llm_3").
param("run_22", "run_22_step_3", "start_x", "3.237").
param("run_22", "run_22_step_3", "start_y", "-1.167").
param("run_22", "run_22_step_3", "start_z", "0.182").
param("run_22", "run_22_step_3", "end_x", "3.355").
param("run_22", "run_22_step_3", "end_y", "-0.225").
param("run_22", "run_22_step_3", "end_z", "1.179").
param("run_22", "run_22_step_3", "duration_sec", "2.60").
param("run_22", "run_22_step_3", "mean_speed", "0.2343").
param("run_22", "run_22_step_3", "gripper_start", "7.96").
param("run_22", "run_22_step_3", "gripper_end", "3.32").
call("run_22", "run_22_llm_4", "llm_generate", "completed").
depends("run_22", "run_22_llm_4", "run_22_step_3").
call("run_22", "run_22_step_4", "transport", "completed").
depends("run_22", "run_22_step_4", "run_22_llm_4").
spawned_by("run_22", "run_22_step_4", "run_22_llm_4").
param("run_22", "run_22_step_4", "start_x", "3.256").
param("run_22", "run_22_step_4", "start_y", "-0.125").
param("run_22", "run_22_step_4", "start_z", "1.370").
param("run_22", "run_22_step_4", "end_x", "2.744").
param("run_22", "run_22_step_4", "end_y", "-0.976").
param("run_22", "run_22_step_4", "end_z", "1.984").
param("run_22", "run_22_step_4", "duration_sec", "6.20").
param("run_22", "run_22_step_4", "mean_speed", "0.0747").
param("run_22", "run_22_step_4", "gripper_start", "3.32").
param("run_22", "run_22_step_4", "gripper_end", "3.32").
call("run_22", "run_22_llm_5", "llm_generate", "completed").
depends("run_22", "run_22_llm_5", "run_22_step_4").
call("run_22", "run_22_notify", "notify_user", "completed").
depends("run_22", "run_22_notify", "run_22_llm_5").
spawned_by("run_22", "run_22_notify", "run_22_llm_5").

% === Episode 22 (pick_and_place, 3 segments) ===
job("run_23").
job_status("run_23", "completed").
call("run_23", "run_23_step_0", "prompt_user", "completed").
call("run_23", "run_23_step_1", "llm_generate", "completed").
depends("run_23", "run_23_step_1", "run_23_step_0").
call("run_23", "run_23_llm_2", "llm_generate", "completed").
depends("run_23", "run_23_llm_2", "run_23_step_1").
call("run_23", "run_23_step_2", "release", "completed").
depends("run_23", "run_23_step_2", "run_23_llm_2").
spawned_by("run_23", "run_23_step_2", "run_23_llm_2").
param("run_23", "run_23_step_2", "start_x", "2.500").
param("run_23", "run_23_step_2", "start_y", "-0.000").
param("run_23", "run_23_step_2", "start_z", "0.200").
param("run_23", "run_23_step_2", "end_x", "2.828").
param("run_23", "run_23_step_2", "end_y", "-0.894").
param("run_23", "run_23_step_2", "end_z", "0.217").
param("run_23", "run_23_step_2", "duration_sec", "1.20").
param("run_23", "run_23_step_2", "mean_speed", "0.2268").
param("run_23", "run_23_step_2", "gripper_start", "6.03").
param("run_23", "run_23_step_2", "gripper_end", "7.96").
call("run_23", "run_23_llm_3", "llm_generate", "completed").
depends("run_23", "run_23_llm_3", "run_23_step_2").
call("run_23", "run_23_step_3", "grasp", "completed").
depends("run_23", "run_23_step_3", "run_23_llm_3").
spawned_by("run_23", "run_23_step_3", "run_23_llm_3").
param("run_23", "run_23_step_3", "start_x", "2.966").
param("run_23", "run_23_step_3", "start_y", "-1.271").
param("run_23", "run_23_step_3", "start_z", "0.228").
param("run_23", "run_23_step_3", "end_x", "3.514").
param("run_23", "run_23_step_3", "end_y", "-1.483").
param("run_23", "run_23_step_3", "end_z", "1.257").
param("run_23", "run_23_step_3", "duration_sec", "4.20").
param("run_23", "run_23_step_3", "mean_speed", "0.0861").
param("run_23", "run_23_step_3", "gripper_start", "7.96").
param("run_23", "run_23_step_3", "gripper_end", "3.38").
call("run_23", "run_23_llm_4", "llm_generate", "completed").
depends("run_23", "run_23_llm_4", "run_23_step_3").
call("run_23", "run_23_step_4", "hold", "completed").
depends("run_23", "run_23_step_4", "run_23_llm_4").
spawned_by("run_23", "run_23_step_4", "run_23_llm_4").
param("run_23", "run_23_step_4", "start_x", "3.513").
param("run_23", "run_23_step_4", "start_y", "-1.498").
param("run_23", "run_23_step_4", "start_z", "1.250").
param("run_23", "run_23_step_4", "end_x", "3.643").
param("run_23", "run_23_step_4", "end_y", "-1.652").
param("run_23", "run_23_step_4", "end_z", "1.287").
param("run_23", "run_23_step_4", "duration_sec", "4.40").
param("run_23", "run_23_step_4", "mean_speed", "0.0155").
param("run_23", "run_23_step_4", "gripper_start", "3.38").
param("run_23", "run_23_step_4", "gripper_end", "3.38").
call("run_23", "run_23_llm_5", "llm_generate", "completed").
depends("run_23", "run_23_llm_5", "run_23_step_4").
call("run_23", "run_23_notify", "notify_user", "completed").
depends("run_23", "run_23_notify", "run_23_llm_5").
spawned_by("run_23", "run_23_notify", "run_23_llm_5").

% === Episode 23 (pick_and_place, 4 segments) ===
job("run_24").
job_status("run_24", "completed").
call("run_24", "run_24_step_0", "prompt_user", "completed").
call("run_24", "run_24_step_1", "llm_generate", "completed").
depends("run_24", "run_24_step_1", "run_24_step_0").
call("run_24", "run_24_llm_2", "llm_generate", "completed").
depends("run_24", "run_24_llm_2", "run_24_step_1").
call("run_24", "run_24_step_2", "release", "completed").
depends("run_24", "run_24_step_2", "run_24_llm_2").
spawned_by("run_24", "run_24_step_2", "run_24_llm_2").
param("run_24", "run_24_step_2", "start_x", "2.500").
param("run_24", "run_24_step_2", "start_y", "-0.000").
param("run_24", "run_24_step_2", "start_z", "0.200").
param("run_24", "run_24_step_2", "end_x", "3.662").
param("run_24", "run_24_step_2", "end_y", "-0.531").
param("run_24", "run_24_step_2", "end_z", "0.182").
param("run_24", "run_24_step_2", "duration_sec", "1.20").
param("run_24", "run_24_step_2", "mean_speed", "0.2807").
param("run_24", "run_24_step_2", "gripper_start", "6.04").
param("run_24", "run_24_step_2", "gripper_end", "7.96").
call("run_24", "run_24_llm_3", "llm_generate", "completed").
depends("run_24", "run_24_llm_3", "run_24_step_2").
call("run_24", "run_24_step_3", "approach", "completed").
depends("run_24", "run_24_step_3", "run_24_llm_3").
spawned_by("run_24", "run_24_step_3", "run_24_llm_3").
param("run_24", "run_24_step_3", "start_x", "3.871").
param("run_24", "run_24_step_3", "start_y", "-0.848").
param("run_24", "run_24_step_3", "start_z", "0.182").
param("run_24", "run_24_step_3", "end_x", "3.590").
param("run_24", "run_24_step_3", "end_y", "-1.275").
param("run_24", "run_24_step_3", "end_z", "0.106").
param("run_24", "run_24_step_3", "duration_sec", "0.60").
param("run_24", "run_24_step_3", "mean_speed", "0.2190").
param("run_24", "run_24_step_3", "gripper_start", "7.96").
param("run_24", "run_24_step_3", "gripper_end", "7.77").
call("run_24", "run_24_llm_4", "llm_generate", "completed").
depends("run_24", "run_24_llm_4", "run_24_step_3").
call("run_24", "run_24_step_4", "grasp", "completed").
depends("run_24", "run_24_step_4", "run_24_llm_4").
spawned_by("run_24", "run_24_step_4", "run_24_llm_4").
param("run_24", "run_24_step_4", "start_x", "3.509").
param("run_24", "run_24_step_4", "start_y", "-1.374").
param("run_24", "run_24_step_4", "start_z", "0.130").
param("run_24", "run_24_step_4", "end_x", "3.563").
param("run_24", "run_24_step_4", "end_y", "-1.663").
param("run_24", "run_24_step_4", "end_z", "0.113").
param("run_24", "run_24_step_4", "duration_sec", "1.40").
param("run_24", "run_24_step_4", "mean_speed", "0.1060").
param("run_24", "run_24_step_4", "gripper_start", "7.77").
param("run_24", "run_24_step_4", "gripper_end", "3.67").
call("run_24", "run_24_llm_5", "llm_generate", "completed").
depends("run_24", "run_24_llm_5", "run_24_step_4").
call("run_24", "run_24_step_5", "transport", "completed").
depends("run_24", "run_24_step_5", "run_24_llm_5").
spawned_by("run_24", "run_24_step_5", "run_24_llm_5").
param("run_24", "run_24_step_5", "start_x", "3.652").
param("run_24", "run_24_step_5", "start_y", "-1.672").
param("run_24", "run_24_step_5", "start_z", "0.148").
param("run_24", "run_24_step_5", "end_x", "4.772").
param("run_24", "run_24_step_5", "end_y", "-2.077").
param("run_24", "run_24_step_5", "end_z", "1.561").
param("run_24", "run_24_step_5", "duration_sec", "6.60").
param("run_24", "run_24_step_5", "mean_speed", "0.1933").
param("run_24", "run_24_step_5", "gripper_start", "3.67").
param("run_24", "run_24_step_5", "gripper_end", "3.38").
call("run_24", "run_24_llm_6", "llm_generate", "completed").
depends("run_24", "run_24_llm_6", "run_24_step_5").
call("run_24", "run_24_notify", "notify_user", "completed").
depends("run_24", "run_24_notify", "run_24_llm_6").
spawned_by("run_24", "run_24_notify", "run_24_llm_6").

% === Episode 24 (pick_and_place, 5 segments) ===
job("run_25").
job_status("run_25", "completed").
call("run_25", "run_25_step_0", "prompt_user", "completed").
call("run_25", "run_25_step_1", "llm_generate", "completed").
depends("run_25", "run_25_step_1", "run_25_step_0").
call("run_25", "run_25_llm_2", "llm_generate", "completed").
depends("run_25", "run_25_llm_2", "run_25_step_1").
call("run_25", "run_25_step_2", "release", "completed").
depends("run_25", "run_25_step_2", "run_25_llm_2").
spawned_by("run_25", "run_25_step_2", "run_25_llm_2").
param("run_25", "run_25_step_2", "start_x", "2.500").
param("run_25", "run_25_step_2", "start_y", "-0.000").
param("run_25", "run_25_step_2", "start_z", "0.200").
param("run_25", "run_25_step_2", "end_x", "4.248").
param("run_25", "run_25_step_2", "end_y", "-0.228").
param("run_25", "run_25_step_2", "end_z", "0.241").
param("run_25", "run_25_step_2", "duration_sec", "1.60").
param("run_25", "run_25_step_2", "mean_speed", "0.2881").
param("run_25", "run_25_step_2", "gripper_start", "6.04").
param("run_25", "run_25_step_2", "gripper_end", "7.96").
call("run_25", "run_25_llm_3", "llm_generate", "completed").
depends("run_25", "run_25_llm_3", "run_25_step_2").
call("run_25", "run_25_step_3", "grasp", "completed").
depends("run_25", "run_25_step_3", "run_25_llm_3").
spawned_by("run_25", "run_25_step_3", "run_25_llm_3").
param("run_25", "run_25_step_3", "start_x", "4.642").
param("run_25", "run_25_step_3", "start_y", "-0.513").
param("run_25", "run_25_step_3", "start_z", "0.273").
param("run_25", "run_25_step_3", "end_x", "4.890").
param("run_25", "run_25_step_3", "end_y", "-0.620").
param("run_25", "run_25_step_3", "end_z", "0.150").
param("run_25", "run_25_step_3", "duration_sec", "1.00").
param("run_25", "run_25_step_3", "mean_speed", "0.2015").
param("run_25", "run_25_step_3", "gripper_start", "7.96").
param("run_25", "run_25_step_3", "gripper_end", "4.46").
call("run_25", "run_25_llm_4", "llm_generate", "completed").
depends("run_25", "run_25_llm_4", "run_25_step_3").
call("run_25", "run_25_step_4", "transport", "completed").
depends("run_25", "run_25_step_4", "run_25_llm_4").
spawned_by("run_25", "run_25_step_4", "run_25_llm_4").
param("run_25", "run_25_step_4", "start_x", "4.804").
param("run_25", "run_25_step_4", "start_y", "-0.588").
param("run_25", "run_25_step_4", "start_z", "0.169").
param("run_25", "run_25_step_4", "end_x", "4.390").
param("run_25", "run_25_step_4", "end_y", "-0.898").
param("run_25", "run_25_step_4", "end_z", "0.161").
param("run_25", "run_25_step_4", "duration_sec", "0.60").
param("run_25", "run_25_step_4", "mean_speed", "0.2444").
param("run_25", "run_25_step_4", "gripper_start", "4.46").
param("run_25", "run_25_step_4", "gripper_end", "4.95").
call("run_25", "run_25_llm_5", "llm_generate", "completed").
depends("run_25", "run_25_llm_5", "run_25_step_4").
call("run_25", "run_25_step_5", "grasp", "completed").
depends("run_25", "run_25_step_5", "run_25_llm_5").
spawned_by("run_25", "run_25_step_5", "run_25_llm_5").
param("run_25", "run_25_step_5", "start_x", "4.354").
param("run_25", "run_25_step_5", "start_y", "-1.093").
param("run_25", "run_25_step_5", "start_z", "0.160").
param("run_25", "run_25_step_5", "end_x", "3.798").
param("run_25", "run_25_step_5", "end_y", "-1.263").
param("run_25", "run_25_step_5", "end_z", "0.155").
param("run_25", "run_25_step_5", "duration_sec", "1.00").
param("run_25", "run_25_step_5", "mean_speed", "0.1897").
param("run_25", "run_25_step_5", "gripper_start", "4.95").
param("run_25", "run_25_step_5", "gripper_end", "3.32").
call("run_25", "run_25_llm_6", "llm_generate", "completed").
depends("run_25", "run_25_llm_6", "run_25_step_5").
call("run_25", "run_25_step_6", "transport", "completed").
depends("run_25", "run_25_step_6", "run_25_llm_6").
spawned_by("run_25", "run_25_step_6", "run_25_llm_6").
param("run_25", "run_25_step_6", "start_x", "3.633").
param("run_25", "run_25_step_6", "start_y", "-1.286").
param("run_25", "run_25_step_6", "start_z", "0.181").
param("run_25", "run_25_step_6", "end_x", "4.204").
param("run_25", "run_25_step_6", "end_y", "-1.934").
param("run_25", "run_25_step_6", "end_z", "1.913").
param("run_25", "run_25_step_6", "duration_sec", "5.60").
param("run_25", "run_25_step_6", "mean_speed", "0.1483").
param("run_25", "run_25_step_6", "gripper_start", "3.32").
param("run_25", "run_25_step_6", "gripper_end", "3.32").
call("run_25", "run_25_llm_7", "llm_generate", "completed").
depends("run_25", "run_25_llm_7", "run_25_step_6").
call("run_25", "run_25_notify", "notify_user", "completed").
depends("run_25", "run_25_notify", "run_25_llm_7").
spawned_by("run_25", "run_25_notify", "run_25_llm_7").

% === Episode 25 (pick_and_place, 5 segments) ===
job("run_26").
job_status("run_26", "completed").
call("run_26", "run_26_step_0", "prompt_user", "completed").
call("run_26", "run_26_step_1", "llm_generate", "completed").
depends("run_26", "run_26_step_1", "run_26_step_0").
call("run_26", "run_26_llm_2", "llm_generate", "completed").
depends("run_26", "run_26_llm_2", "run_26_step_1").
call("run_26", "run_26_step_2", "release", "completed").
depends("run_26", "run_26_step_2", "run_26_llm_2").
spawned_by("run_26", "run_26_step_2", "run_26_llm_2").
param("run_26", "run_26_step_2", "start_x", "2.500").
param("run_26", "run_26_step_2", "start_y", "-0.000").
param("run_26", "run_26_step_2", "start_z", "0.200").
param("run_26", "run_26_step_2", "end_x", "4.293").
param("run_26", "run_26_step_2", "end_y", "-1.651").
param("run_26", "run_26_step_2", "end_z", "0.152").
param("run_26", "run_26_step_2", "duration_sec", "1.80").
param("run_26", "run_26_step_2", "mean_speed", "0.3065").
param("run_26", "run_26_step_2", "gripper_start", "6.03").
param("run_26", "run_26_step_2", "gripper_end", "7.96").
call("run_26", "run_26_llm_3", "llm_generate", "completed").
depends("run_26", "run_26_llm_3", "run_26_step_2").
call("run_26", "run_26_step_3", "approach", "completed").
depends("run_26", "run_26_step_3", "run_26_llm_3").
spawned_by("run_26", "run_26_step_3", "run_26_llm_3").
param("run_26", "run_26_step_3", "start_x", "4.312").
param("run_26", "run_26_step_3", "start_y", "-1.943").
param("run_26", "run_26_step_3", "start_z", "0.130").
param("run_26", "run_26_step_3", "end_x", "4.373").
param("run_26", "run_26_step_3", "end_y", "-2.072").
param("run_26", "run_26_step_3", "end_z", "0.136").
param("run_26", "run_26_step_3", "duration_sec", "1.80").
param("run_26", "run_26_step_3", "mean_speed", "0.0925").
param("run_26", "run_26_step_3", "gripper_start", "7.96").
param("run_26", "run_26_step_3", "gripper_end", "7.05").
call("run_26", "run_26_llm_4", "llm_generate", "completed").
depends("run_26", "run_26_llm_4", "run_26_step_3").
call("run_26", "run_26_step_4", "grasp", "completed").
depends("run_26", "run_26_step_4", "run_26_llm_4").
spawned_by("run_26", "run_26_step_4", "run_26_llm_4").
param("run_26", "run_26_step_4", "start_x", "4.116").
param("run_26", "run_26_step_4", "start_y", "-1.919").
param("run_26", "run_26_step_4", "start_z", "0.224").
param("run_26", "run_26_step_4", "end_x", "4.098").
param("run_26", "run_26_step_4", "end_y", "-2.032").
param("run_26", "run_26_step_4", "end_z", "0.172").
param("run_26", "run_26_step_4", "duration_sec", "1.00").
param("run_26", "run_26_step_4", "mean_speed", "0.1448").
param("run_26", "run_26_step_4", "gripper_start", "7.64").
param("run_26", "run_26_step_4", "gripper_end", "3.33").
call("run_26", "run_26_llm_5", "llm_generate", "completed").
depends("run_26", "run_26_llm_5", "run_26_step_4").
call("run_26", "run_26_step_5", "transport", "completed").
depends("run_26", "run_26_step_5", "run_26_llm_5").
spawned_by("run_26", "run_26_step_5", "run_26_llm_5").
param("run_26", "run_26_step_5", "start_x", "4.314").
param("run_26", "run_26_step_5", "start_y", "-1.944").
param("run_26", "run_26_step_5", "start_z", "0.199").
param("run_26", "run_26_step_5", "end_x", "4.435").
param("run_26", "run_26_step_5", "end_y", "-1.397").
param("run_26", "run_26_step_5", "end_z", "1.416").
param("run_26", "run_26_step_5", "duration_sec", "2.20").
param("run_26", "run_26_step_5", "mean_speed", "0.1595").
param("run_26", "run_26_step_5", "gripper_start", "3.33").
param("run_26", "run_26_step_5", "gripper_end", "3.32").
call("run_26", "run_26_llm_6", "llm_generate", "completed").
depends("run_26", "run_26_llm_6", "run_26_step_5").
call("run_26", "run_26_step_6", "hold", "completed").
depends("run_26", "run_26_step_6", "run_26_llm_6").
spawned_by("run_26", "run_26_step_6", "run_26_llm_6").
param("run_26", "run_26_step_6", "start_x", "4.551").
param("run_26", "run_26_step_6", "start_y", "-1.378").
param("run_26", "run_26_step_6", "start_z", "1.396").
param("run_26", "run_26_step_6", "end_x", "4.939").
param("run_26", "run_26_step_6", "end_y", "-1.398").
param("run_26", "run_26_step_6", "end_z", "1.442").
param("run_26", "run_26_step_6", "duration_sec", "3.00").
param("run_26", "run_26_step_6", "mean_speed", "0.0405").
param("run_26", "run_26_step_6", "gripper_start", "3.32").
param("run_26", "run_26_step_6", "gripper_end", "3.32").
call("run_26", "run_26_llm_7", "llm_generate", "completed").
depends("run_26", "run_26_llm_7", "run_26_step_6").
call("run_26", "run_26_notify", "notify_user", "completed").
depends("run_26", "run_26_notify", "run_26_llm_7").
spawned_by("run_26", "run_26_notify", "run_26_llm_7").

% === Episode 26 (pick_and_place, 2 segments) ===
job("run_27").
job_status("run_27", "completed").
call("run_27", "run_27_step_0", "prompt_user", "completed").
call("run_27", "run_27_step_1", "llm_generate", "completed").
depends("run_27", "run_27_step_1", "run_27_step_0").
call("run_27", "run_27_llm_2", "llm_generate", "completed").
depends("run_27", "run_27_llm_2", "run_27_step_1").
call("run_27", "run_27_step_2", "release", "completed").
depends("run_27", "run_27_step_2", "run_27_llm_2").
spawned_by("run_27", "run_27_step_2", "run_27_llm_2").
param("run_27", "run_27_step_2", "start_x", "2.500").
param("run_27", "run_27_step_2", "start_y", "-0.000").
param("run_27", "run_27_step_2", "start_z", "0.200").
param("run_27", "run_27_step_2", "end_x", "5.493").
param("run_27", "run_27_step_2", "end_y", "-0.005").
param("run_27", "run_27_step_2", "end_z", "0.147").
param("run_27", "run_27_step_2", "duration_sec", "6.40").
param("run_27", "run_27_step_2", "mean_speed", "0.1202").
param("run_27", "run_27_step_2", "gripper_start", "6.03").
param("run_27", "run_27_step_2", "gripper_end", "7.97").
call("run_27", "run_27_llm_3", "llm_generate", "completed").
depends("run_27", "run_27_llm_3", "run_27_step_2").
call("run_27", "run_27_step_3", "idle", "completed").
depends("run_27", "run_27_step_3", "run_27_llm_3").
spawned_by("run_27", "run_27_step_3", "run_27_llm_3").
param("run_27", "run_27_step_3", "start_x", "5.507").
param("run_27", "run_27_step_3", "start_y", "0.004").
param("run_27", "run_27_step_3", "start_z", "0.153").
param("run_27", "run_27_step_3", "end_x", "5.500").
param("run_27", "run_27_step_3", "end_y", "0.000").
param("run_27", "run_27_step_3", "end_z", "0.150").
param("run_27", "run_27_step_3", "duration_sec", "3.40").
param("run_27", "run_27_step_3", "mean_speed", "0.0013").
param("run_27", "run_27_step_3", "gripper_start", "7.97").
param("run_27", "run_27_step_3", "gripper_end", "7.98").
call("run_27", "run_27_llm_4", "llm_generate", "completed").
depends("run_27", "run_27_llm_4", "run_27_step_3").
call("run_27", "run_27_notify", "notify_user", "completed").
depends("run_27", "run_27_notify", "run_27_llm_4").
spawned_by("run_27", "run_27_notify", "run_27_llm_4").

% === Episode 27 (pick_and_place, 2 segments) ===
job("run_28").
job_status("run_28", "completed").
call("run_28", "run_28_step_0", "prompt_user", "completed").
call("run_28", "run_28_step_1", "llm_generate", "completed").
depends("run_28", "run_28_step_1", "run_28_step_0").
call("run_28", "run_28_llm_2", "llm_generate", "completed").
depends("run_28", "run_28_llm_2", "run_28_step_1").
call("run_28", "run_28_step_2", "release", "completed").
depends("run_28", "run_28_step_2", "run_28_llm_2").
spawned_by("run_28", "run_28_step_2", "run_28_llm_2").
param("run_28", "run_28_step_2", "start_x", "2.500").
param("run_28", "run_28_step_2", "start_y", "-0.000").
param("run_28", "run_28_step_2", "start_z", "0.200").
param("run_28", "run_28_step_2", "end_x", "3.328").
param("run_28", "run_28_step_2", "end_y", "-1.631").
param("run_28", "run_28_step_2", "end_z", "0.256").
param("run_28", "run_28_step_2", "duration_sec", "2.40").
param("run_28", "run_28_step_2", "mean_speed", "0.1788").
param("run_28", "run_28_step_2", "gripper_start", "6.03").
param("run_28", "run_28_step_2", "gripper_end", "7.96").
call("run_28", "run_28_llm_3", "llm_generate", "completed").
depends("run_28", "run_28_llm_3", "run_28_step_2").
call("run_28", "run_28_step_3", "grasp", "completed").
depends("run_28", "run_28_step_3", "run_28_llm_3").
spawned_by("run_28", "run_28_step_3", "run_28_llm_3").
param("run_28", "run_28_step_3", "start_x", "3.352").
param("run_28", "run_28_step_3", "start_y", "-1.888").
param("run_28", "run_28_step_3", "start_z", "0.254").
param("run_28", "run_28_step_3", "end_x", "3.762").
param("run_28", "run_28_step_3", "end_y", "-1.841").
param("run_28", "run_28_step_3", "end_z", "1.468").
param("run_28", "run_28_step_3", "duration_sec", "7.40").
param("run_28", "run_28_step_3", "mean_speed", "0.0496").
param("run_28", "run_28_step_3", "gripper_start", "7.96").
param("run_28", "run_28_step_3", "gripper_end", "4.92").
call("run_28", "run_28_llm_4", "llm_generate", "completed").
depends("run_28", "run_28_llm_4", "run_28_step_3").
call("run_28", "run_28_notify", "notify_user", "completed").
depends("run_28", "run_28_notify", "run_28_llm_4").
spawned_by("run_28", "run_28_notify", "run_28_llm_4").

% === Episode 28 (pick_and_place, 2 segments) ===
job("run_29").
job_status("run_29", "completed").
call("run_29", "run_29_step_0", "prompt_user", "completed").
call("run_29", "run_29_step_1", "llm_generate", "completed").
depends("run_29", "run_29_step_1", "run_29_step_0").
call("run_29", "run_29_llm_2", "llm_generate", "completed").
depends("run_29", "run_29_llm_2", "run_29_step_1").
call("run_29", "run_29_step_2", "release", "completed").
depends("run_29", "run_29_step_2", "run_29_llm_2").
spawned_by("run_29", "run_29_step_2", "run_29_llm_2").
param("run_29", "run_29_step_2", "start_x", "2.500").
param("run_29", "run_29_step_2", "start_y", "-0.000").
param("run_29", "run_29_step_2", "start_z", "0.200").
param("run_29", "run_29_step_2", "end_x", "3.064").
param("run_29", "run_29_step_2", "end_y", "-0.484").
param("run_29", "run_29_step_2", "end_z", "0.134").
param("run_29", "run_29_step_2", "duration_sec", "0.80").
param("run_29", "run_29_step_2", "mean_speed", "0.3243").
param("run_29", "run_29_step_2", "gripper_start", "6.04").
param("run_29", "run_29_step_2", "gripper_end", "7.95").
call("run_29", "run_29_llm_3", "llm_generate", "completed").
depends("run_29", "run_29_llm_3", "run_29_step_2").
call("run_29", "run_29_step_3", "grasp", "completed").
depends("run_29", "run_29_step_3", "run_29_llm_3").
spawned_by("run_29", "run_29_step_3", "run_29_llm_3").
param("run_29", "run_29_step_3", "start_x", "3.454").
param("run_29", "run_29_step_3", "start_y", "-0.870").
param("run_29", "run_29_step_3", "start_z", "0.151").
param("run_29", "run_29_step_3", "end_x", "4.229").
param("run_29", "run_29_step_3", "end_y", "-2.055").
param("run_29", "run_29_step_3", "end_z", "2.001").
param("run_29", "run_29_step_3", "duration_sec", "9.00").
param("run_29", "run_29_step_3", "mean_speed", "0.1905").
param("run_29", "run_29_step_3", "gripper_start", "7.95").
param("run_29", "run_29_step_3", "gripper_end", "5.55").
call("run_29", "run_29_llm_4", "llm_generate", "completed").
depends("run_29", "run_29_llm_4", "run_29_step_3").
call("run_29", "run_29_notify", "notify_user", "completed").
depends("run_29", "run_29_notify", "run_29_llm_4").
spawned_by("run_29", "run_29_notify", "run_29_llm_4").

% === Episode 29 (pick_and_place, 2 segments) ===
job("run_30").
job_status("run_30", "completed").
call("run_30", "run_30_step_0", "prompt_user", "completed").
call("run_30", "run_30_step_1", "llm_generate", "completed").
depends("run_30", "run_30_step_1", "run_30_step_0").
call("run_30", "run_30_llm_2", "llm_generate", "completed").
depends("run_30", "run_30_llm_2", "run_30_step_1").
call("run_30", "run_30_step_2", "release", "completed").
depends("run_30", "run_30_step_2", "run_30_llm_2").
spawned_by("run_30", "run_30_step_2", "run_30_llm_2").
param("run_30", "run_30_step_2", "start_x", "2.500").
param("run_30", "run_30_step_2", "start_y", "-0.000").
param("run_30", "run_30_step_2", "start_z", "0.200").
param("run_30", "run_30_step_2", "end_x", "4.841").
param("run_30", "run_30_step_2", "end_y", "-2.483").
param("run_30", "run_30_step_2", "end_z", "0.544").
param("run_30", "run_30_step_2", "duration_sec", "4.20").
param("run_30", "run_30_step_2", "mean_speed", "0.1876").
param("run_30", "run_30_step_2", "gripper_start", "6.05").
param("run_30", "run_30_step_2", "gripper_end", "7.97").
call("run_30", "run_30_llm_3", "llm_generate", "completed").
depends("run_30", "run_30_llm_3", "run_30_step_2").
call("run_30", "run_30_step_3", "idle", "completed").
depends("run_30", "run_30_step_3", "run_30_llm_3").
spawned_by("run_30", "run_30_step_3", "run_30_llm_3").
param("run_30", "run_30_step_3", "start_x", "4.880").
param("run_30", "run_30_step_3", "start_y", "-2.542").
param("run_30", "run_30_step_3", "start_z", "0.549").
param("run_30", "run_30_step_3", "end_x", "4.850").
param("run_30", "run_30_step_3", "end_y", "-2.615").
param("run_30", "run_30_step_3", "end_z", "0.632").
param("run_30", "run_30_step_3", "duration_sec", "5.60").
param("run_30", "run_30_step_3", "mean_speed", "0.0282").
param("run_30", "run_30_step_3", "gripper_start", "7.97").
param("run_30", "run_30_step_3", "gripper_end", "7.98").
call("run_30", "run_30_llm_4", "llm_generate", "completed").
depends("run_30", "run_30_llm_4", "run_30_step_3").
call("run_30", "run_30_notify", "notify_user", "completed").
depends("run_30", "run_30_notify", "run_30_llm_4").
spawned_by("run_30", "run_30_notify", "run_30_llm_4").

% === Episode 30 (pick_and_place, 2 segments) ===
job("run_31").
job_status("run_31", "completed").
call("run_31", "run_31_step_0", "prompt_user", "completed").
call("run_31", "run_31_step_1", "llm_generate", "completed").
depends("run_31", "run_31_step_1", "run_31_step_0").
call("run_31", "run_31_llm_2", "llm_generate", "completed").
depends("run_31", "run_31_llm_2", "run_31_step_1").
call("run_31", "run_31_step_2", "release", "completed").
depends("run_31", "run_31_step_2", "run_31_llm_2").
spawned_by("run_31", "run_31_step_2", "run_31_llm_2").
param("run_31", "run_31_step_2", "start_x", "2.500").
param("run_31", "run_31_step_2", "start_y", "-0.000").
param("run_31", "run_31_step_2", "start_z", "0.200").
param("run_31", "run_31_step_2", "end_x", "5.494").
param("run_31", "run_31_step_2", "end_y", "-0.001").
param("run_31", "run_31_step_2", "end_z", "0.143").
param("run_31", "run_31_step_2", "duration_sec", "5.80").
param("run_31", "run_31_step_2", "mean_speed", "0.1266").
param("run_31", "run_31_step_2", "gripper_start", "6.04").
param("run_31", "run_31_step_2", "gripper_end", "7.97").
call("run_31", "run_31_llm_3", "llm_generate", "completed").
depends("run_31", "run_31_llm_3", "run_31_step_2").
call("run_31", "run_31_step_3", "idle", "completed").
depends("run_31", "run_31_step_3", "run_31_llm_3").
spawned_by("run_31", "run_31_step_3", "run_31_llm_3").
param("run_31", "run_31_step_3", "start_x", "5.506").
param("run_31", "run_31_step_3", "start_y", "-0.000").
param("run_31", "run_31_step_3", "start_z", "0.157").
param("run_31", "run_31_step_3", "end_x", "5.500").
param("run_31", "run_31_step_3", "end_y", "0.000").
param("run_31", "run_31_step_3", "end_z", "0.150").
param("run_31", "run_31_step_3", "duration_sec", "4.00").
param("run_31", "run_31_step_3", "mean_speed", "0.0014").
param("run_31", "run_31_step_3", "gripper_start", "7.97").
param("run_31", "run_31_step_3", "gripper_end", "7.99").
call("run_31", "run_31_llm_4", "llm_generate", "completed").
depends("run_31", "run_31_llm_4", "run_31_step_3").
call("run_31", "run_31_notify", "notify_user", "completed").
depends("run_31", "run_31_notify", "run_31_llm_4").
spawned_by("run_31", "run_31_notify", "run_31_llm_4").

% === Episode 31 (pick_and_place, 9 segments) ===
job("run_32").
job_status("run_32", "completed").
call("run_32", "run_32_step_0", "prompt_user", "completed").
call("run_32", "run_32_step_1", "llm_generate", "completed").
depends("run_32", "run_32_step_1", "run_32_step_0").
call("run_32", "run_32_llm_2", "llm_generate", "completed").
depends("run_32", "run_32_llm_2", "run_32_step_1").
call("run_32", "run_32_step_2", "release", "completed").
depends("run_32", "run_32_step_2", "run_32_llm_2").
spawned_by("run_32", "run_32_step_2", "run_32_llm_2").
param("run_32", "run_32_step_2", "start_x", "2.500").
param("run_32", "run_32_step_2", "start_y", "-0.000").
param("run_32", "run_32_step_2", "start_z", "0.200").
param("run_32", "run_32_step_2", "end_x", "2.918").
param("run_32", "run_32_step_2", "end_y", "-1.701").
param("run_32", "run_32_step_2", "end_z", "0.162").
param("run_32", "run_32_step_2", "duration_sec", "1.60").
param("run_32", "run_32_step_2", "mean_speed", "0.2526").
param("run_32", "run_32_step_2", "gripper_start", "6.04").
param("run_32", "run_32_step_2", "gripper_end", "7.96").
call("run_32", "run_32_llm_3", "llm_generate", "completed").
depends("run_32", "run_32_llm_3", "run_32_step_2").
call("run_32", "run_32_step_3", "approach", "completed").
depends("run_32", "run_32_step_3", "run_32_llm_3").
spawned_by("run_32", "run_32_step_3", "run_32_llm_3").
param("run_32", "run_32_step_3", "start_x", "3.074").
param("run_32", "run_32_step_3", "start_y", "-1.837").
param("run_32", "run_32_step_3", "start_z", "0.183").
param("run_32", "run_32_step_3", "end_x", "3.706").
param("run_32", "run_32_step_3", "end_y", "-3.138").
param("run_32", "run_32_step_3", "end_z", "0.312").
param("run_32", "run_32_step_3", "duration_sec", "1.40").
param("run_32", "run_32_step_3", "mean_speed", "0.2714").
param("run_32", "run_32_step_3", "gripper_start", "7.96").
param("run_32", "run_32_step_3", "gripper_end", "7.96").
call("run_32", "run_32_llm_4", "llm_generate", "completed").
depends("run_32", "run_32_llm_4", "run_32_step_3").
call("run_32", "run_32_step_4", "grasp", "completed").
depends("run_32", "run_32_step_4", "run_32_llm_4").
spawned_by("run_32", "run_32_step_4", "run_32_llm_4").
param("run_32", "run_32_step_4", "start_x", "3.998").
param("run_32", "run_32_step_4", "start_y", "-2.982").
param("run_32", "run_32_step_4", "start_z", "0.357").
param("run_32", "run_32_step_4", "end_x", "3.366").
param("run_32", "run_32_step_4", "end_y", "-2.997").
param("run_32", "run_32_step_4", "end_z", "0.653").
param("run_32", "run_32_step_4", "duration_sec", "1.20").
param("run_32", "run_32_step_4", "mean_speed", "0.1784").
param("run_32", "run_32_step_4", "gripper_start", "7.96").
param("run_32", "run_32_step_4", "gripper_end", "5.59").
call("run_32", "run_32_llm_5", "llm_generate", "completed").
depends("run_32", "run_32_llm_5", "run_32_step_4").
call("run_32", "run_32_step_5", "transport", "completed").
depends("run_32", "run_32_step_5", "run_32_llm_5").
spawned_by("run_32", "run_32_step_5", "run_32_llm_5").
param("run_32", "run_32_step_5", "start_x", "3.469").
param("run_32", "run_32_step_5", "start_y", "-2.999").
param("run_32", "run_32_step_5", "start_z", "0.752").
param("run_32", "run_32_step_5", "end_x", "3.590").
param("run_32", "run_32_step_5", "end_y", "-3.000").
param("run_32", "run_32_step_5", "end_z", "0.872").
param("run_32", "run_32_step_5", "duration_sec", "0.60").
param("run_32", "run_32_step_5", "mean_speed", "0.1420").
param("run_32", "run_32_step_5", "gripper_start", "5.59").
param("run_32", "run_32_step_5", "gripper_end", "6.23").
call("run_32", "run_32_llm_6", "llm_generate", "completed").
depends("run_32", "run_32_llm_6", "run_32_step_5").
call("run_32", "run_32_step_6", "grasp", "completed").
depends("run_32", "run_32_step_6", "run_32_llm_6").
spawned_by("run_32", "run_32_step_6", "run_32_llm_6").
param("run_32", "run_32_step_6", "start_x", "3.785").
param("run_32", "run_32_step_6", "start_y", "-3.000").
param("run_32", "run_32_step_6", "start_z", "0.774").
param("run_32", "run_32_step_6", "end_x", "3.990").
param("run_32", "run_32_step_6", "end_y", "-2.969").
param("run_32", "run_32_step_6", "end_z", "0.760").
param("run_32", "run_32_step_6", "duration_sec", "0.60").
param("run_32", "run_32_step_6", "mean_speed", "0.2536").
param("run_32", "run_32_step_6", "gripper_start", "6.23").
param("run_32", "run_32_step_6", "gripper_end", "4.57").
call("run_32", "run_32_llm_7", "llm_generate", "completed").
depends("run_32", "run_32_llm_7", "run_32_step_6").
call("run_32", "run_32_step_7", "retract", "completed").
depends("run_32", "run_32_step_7", "run_32_llm_7").
spawned_by("run_32", "run_32_step_7", "run_32_llm_7").
param("run_32", "run_32_step_7", "start_x", "3.791").
param("run_32", "run_32_step_7", "start_y", "-2.937").
param("run_32", "run_32_step_7", "start_z", "0.938").
param("run_32", "run_32_step_7", "end_x", "4.057").
param("run_32", "run_32_step_7", "end_y", "-3.046").
param("run_32", "run_32_step_7", "end_z", "0.801").
param("run_32", "run_32_step_7", "duration_sec", "0.80").
param("run_32", "run_32_step_7", "mean_speed", "0.1254").
param("run_32", "run_32_step_7", "gripper_start", "4.57").
param("run_32", "run_32_step_7", "gripper_end", "4.40").
call("run_32", "run_32_llm_8", "llm_generate", "completed").
depends("run_32", "run_32_llm_8", "run_32_step_7").
call("run_32", "run_32_step_8", "transport", "completed").
depends("run_32", "run_32_step_8", "run_32_llm_8").
spawned_by("run_32", "run_32_step_8", "run_32_llm_8").
param("run_32", "run_32_step_8", "start_x", "4.012").
param("run_32", "run_32_step_8", "start_y", "-2.960").
param("run_32", "run_32_step_8", "start_z", "0.940").
param("run_32", "run_32_step_8", "end_x", "3.780").
param("run_32", "run_32_step_8", "end_y", "-2.767").
param("run_32", "run_32_step_8", "end_z", "1.216").
param("run_32", "run_32_step_8", "duration_sec", "0.80").
param("run_32", "run_32_step_8", "mean_speed", "0.1295").
param("run_32", "run_32_step_8", "gripper_start", "4.40").
param("run_32", "run_32_step_8", "gripper_end", "4.41").
call("run_32", "run_32_llm_9", "llm_generate", "completed").
depends("run_32", "run_32_llm_9", "run_32_step_8").
call("run_32", "run_32_step_9", "grasp", "completed").
depends("run_32", "run_32_step_9", "run_32_llm_9").
spawned_by("run_32", "run_32_step_9", "run_32_llm_9").
param("run_32", "run_32_step_9", "start_x", "3.733").
param("run_32", "run_32_step_9", "start_y", "-2.723").
param("run_32", "run_32_step_9", "start_z", "1.273").
param("run_32", "run_32_step_9", "end_x", "3.508").
param("run_32", "run_32_step_9", "end_y", "-3.028").
param("run_32", "run_32_step_9", "end_z", "1.338").
param("run_32", "run_32_step_9", "duration_sec", "0.80").
param("run_32", "run_32_step_9", "mean_speed", "0.2603").
param("run_32", "run_32_step_9", "gripper_start", "4.41").
param("run_32", "run_32_step_9", "gripper_end", "1.61").
call("run_32", "run_32_llm_10", "llm_generate", "completed").
depends("run_32", "run_32_llm_10", "run_32_step_9").
call("run_32", "run_32_step_10", "retract", "completed").
depends("run_32", "run_32_step_10", "run_32_llm_10").
spawned_by("run_32", "run_32_step_10", "run_32_llm_10").
param("run_32", "run_32_step_10", "start_x", "3.240").
param("run_32", "run_32_step_10", "start_y", "-2.880").
param("run_32", "run_32_step_10", "start_z", "1.250").
param("run_32", "run_32_step_10", "end_x", "2.512").
param("run_32", "run_32_step_10", "end_y", "-2.999").
param("run_32", "run_32_step_10", "end_z", "1.277").
param("run_32", "run_32_step_10", "duration_sec", "2.00").
param("run_32", "run_32_step_10", "mean_speed", "0.2835").
param("run_32", "run_32_step_10", "gripper_start", "1.61").
param("run_32", "run_32_step_10", "gripper_end", "2.43").
call("run_32", "run_32_llm_11", "llm_generate", "completed").
depends("run_32", "run_32_llm_11", "run_32_step_10").
call("run_32", "run_32_notify", "notify_user", "completed").
depends("run_32", "run_32_notify", "run_32_llm_11").
spawned_by("run_32", "run_32_notify", "run_32_llm_11").

% === Episode 32 (pick_and_place, 4 segments) ===
job("run_33").
job_status("run_33", "completed").
call("run_33", "run_33_step_0", "prompt_user", "completed").
call("run_33", "run_33_step_1", "llm_generate", "completed").
depends("run_33", "run_33_step_1", "run_33_step_0").
call("run_33", "run_33_llm_2", "llm_generate", "completed").
depends("run_33", "run_33_llm_2", "run_33_step_1").
call("run_33", "run_33_step_2", "release", "completed").
depends("run_33", "run_33_step_2", "run_33_llm_2").
spawned_by("run_33", "run_33_step_2", "run_33_llm_2").
param("run_33", "run_33_step_2", "start_x", "2.500").
param("run_33", "run_33_step_2", "start_y", "-0.000").
param("run_33", "run_33_step_2", "start_z", "0.200").
param("run_33", "run_33_step_2", "end_x", "4.175").
param("run_33", "run_33_step_2", "end_y", "-0.283").
param("run_33", "run_33_step_2", "end_z", "0.178").
param("run_33", "run_33_step_2", "duration_sec", "1.60").
param("run_33", "run_33_step_2", "mean_speed", "0.2662").
param("run_33", "run_33_step_2", "gripper_start", "6.03").
param("run_33", "run_33_step_2", "gripper_end", "7.96").
call("run_33", "run_33_llm_3", "llm_generate", "completed").
depends("run_33", "run_33_llm_3", "run_33_step_2").
call("run_33", "run_33_step_3", "approach", "completed").
depends("run_33", "run_33_step_3", "run_33_llm_3").
spawned_by("run_33", "run_33_step_3", "run_33_llm_3").
param("run_33", "run_33_step_3", "start_x", "4.572").
param("run_33", "run_33_step_3", "start_y", "-0.401").
param("run_33", "run_33_step_3", "start_z", "0.168").
param("run_33", "run_33_step_3", "end_x", "4.924").
param("run_33", "run_33_step_3", "end_y", "-0.553").
param("run_33", "run_33_step_3", "end_z", "0.137").
param("run_33", "run_33_step_3", "duration_sec", "0.60").
param("run_33", "run_33_step_3", "mean_speed", "0.2651").
param("run_33", "run_33_step_3", "gripper_start", "7.96").
param("run_33", "run_33_step_3", "gripper_end", "7.96").
call("run_33", "run_33_llm_4", "llm_generate", "completed").
depends("run_33", "run_33_llm_4", "run_33_step_3").
call("run_33", "run_33_step_4", "grasp", "completed").
depends("run_33", "run_33_step_4", "run_33_llm_4").
spawned_by("run_33", "run_33_step_4", "run_33_llm_4").
param("run_33", "run_33_step_4", "start_x", "5.070").
param("run_33", "run_33_step_4", "start_y", "-0.937").
param("run_33", "run_33_step_4", "start_z", "0.143").
param("run_33", "run_33_step_4", "end_x", "4.922").
param("run_33", "run_33_step_4", "end_y", "-1.501").
param("run_33", "run_33_step_4", "end_z", "0.175").
param("run_33", "run_33_step_4", "duration_sec", "1.80").
param("run_33", "run_33_step_4", "mean_speed", "0.2547").
param("run_33", "run_33_step_4", "gripper_start", "7.96").
param("run_33", "run_33_step_4", "gripper_end", "5.91").
call("run_33", "run_33_llm_5", "llm_generate", "completed").
depends("run_33", "run_33_llm_5", "run_33_step_4").
call("run_33", "run_33_step_5", "transport", "completed").
depends("run_33", "run_33_step_5", "run_33_llm_5").
spawned_by("run_33", "run_33_step_5", "run_33_llm_5").
param("run_33", "run_33_step_5", "start_x", "4.890").
param("run_33", "run_33_step_5", "start_y", "-1.463").
param("run_33", "run_33_step_5", "start_z", "0.290").
param("run_33", "run_33_step_5", "end_x", "4.285").
param("run_33", "run_33_step_5", "end_y", "-2.413").
param("run_33", "run_33_step_5", "end_z", "1.304").
param("run_33", "run_33_step_5", "duration_sec", "5.80").
param("run_33", "run_33_step_5", "mean_speed", "0.1427").
param("run_33", "run_33_step_5", "gripper_start", "3.40").
param("run_33", "run_33_step_5", "gripper_end", "3.40").
call("run_33", "run_33_llm_6", "llm_generate", "completed").
depends("run_33", "run_33_llm_6", "run_33_step_5").
call("run_33", "run_33_notify", "notify_user", "completed").
depends("run_33", "run_33_notify", "run_33_llm_6").
spawned_by("run_33", "run_33_notify", "run_33_llm_6").

% === Episode 33 (pick_and_place, 8 segments) ===
job("run_34").
job_status("run_34", "completed").
call("run_34", "run_34_step_0", "prompt_user", "completed").
call("run_34", "run_34_step_1", "llm_generate", "completed").
depends("run_34", "run_34_step_1", "run_34_step_0").
call("run_34", "run_34_llm_2", "llm_generate", "completed").
depends("run_34", "run_34_llm_2", "run_34_step_1").
call("run_34", "run_34_step_2", "transport", "completed").
depends("run_34", "run_34_step_2", "run_34_llm_2").
spawned_by("run_34", "run_34_step_2", "run_34_llm_2").
param("run_34", "run_34_step_2", "start_x", "0.900").
param("run_34", "run_34_step_2", "start_y", "2.000").
param("run_34", "run_34_step_2", "start_z", "2.000").
param("run_34", "run_34_step_2", "end_x", "0.959").
param("run_34", "run_34_step_2", "end_y", "2.193").
param("run_34", "run_34_step_2", "end_z", "1.974").
param("run_34", "run_34_step_2", "duration_sec", "0.60").
param("run_34", "run_34_step_2", "mean_speed", "0.0995").
param("run_34", "run_34_step_2", "gripper_start", "4.00").
param("run_34", "run_34_step_2", "gripper_end", "3.84").
call("run_34", "run_34_llm_3", "llm_generate", "completed").
depends("run_34", "run_34_llm_3", "run_34_step_2").
call("run_34", "run_34_step_3", "approach", "completed").
depends("run_34", "run_34_step_3", "run_34_llm_3").
spawned_by("run_34", "run_34_step_3", "run_34_llm_3").
param("run_34", "run_34_step_3", "start_x", "0.950").
param("run_34", "run_34_step_3", "start_y", "2.278").
param("run_34", "run_34_step_3", "start_z", "1.953").
param("run_34", "run_34_step_3", "end_x", "0.874").
param("run_34", "run_34_step_3", "end_y", "2.543").
param("run_34", "run_34_step_3", "end_z", "2.002").
param("run_34", "run_34_step_3", "duration_sec", "0.60").
param("run_34", "run_34_step_3", "mean_speed", "0.1126").
param("run_34", "run_34_step_3", "gripper_start", "3.84").
param("run_34", "run_34_step_3", "gripper_end", "3.13").
call("run_34", "run_34_llm_4", "llm_generate", "completed").
depends("run_34", "run_34_llm_4", "run_34_step_3").
call("run_34", "run_34_step_4", "transport", "completed").
depends("run_34", "run_34_step_4", "run_34_llm_4").
spawned_by("run_34", "run_34_step_4", "run_34_llm_4").
param("run_34", "run_34_step_4", "start_x", "0.878").
param("run_34", "run_34_step_4", "start_y", "2.583").
param("run_34", "run_34_step_4", "start_z", "2.001").
param("run_34", "run_34_step_4", "end_x", "0.918").
param("run_34", "run_34_step_4", "end_y", "2.540").
param("run_34", "run_34_step_4", "end_z", "1.998").
param("run_34", "run_34_step_4", "duration_sec", "0.60").
param("run_34", "run_34_step_4", "mean_speed", "0.0901").
param("run_34", "run_34_step_4", "gripper_start", "3.13").
param("run_34", "run_34_step_4", "gripper_end", "2.53").
call("run_34", "run_34_llm_5", "llm_generate", "completed").
depends("run_34", "run_34_llm_5", "run_34_step_4").
call("run_34", "run_34_step_5", "grasp", "completed").
depends("run_34", "run_34_step_5", "run_34_llm_5").
spawned_by("run_34", "run_34_step_5", "run_34_llm_5").
param("run_34", "run_34_step_5", "start_x", "0.904").
param("run_34", "run_34_step_5", "start_y", "2.336").
param("run_34", "run_34_step_5", "start_z", "1.976").
param("run_34", "run_34_step_5", "end_x", "0.961").
param("run_34", "run_34_step_5", "end_y", "2.083").
param("run_34", "run_34_step_5", "end_z", "1.979").
param("run_34", "run_34_step_5", "duration_sec", "0.80").
param("run_34", "run_34_step_5", "mean_speed", "0.1217").
param("run_34", "run_34_step_5", "gripper_start", "2.53").
param("run_34", "run_34_step_5", "gripper_end", "0.87").
call("run_34", "run_34_llm_6", "llm_generate", "completed").
depends("run_34", "run_34_llm_6", "run_34_step_5").
call("run_34", "run_34_step_6", "release", "completed").
depends("run_34", "run_34_step_6", "run_34_llm_6").
spawned_by("run_34", "run_34_step_6", "run_34_llm_6").
param("run_34", "run_34_step_6", "start_x", "0.933").
param("run_34", "run_34_step_6", "start_y", "1.884").
param("run_34", "run_34_step_6", "start_z", "1.995").
param("run_34", "run_34_step_6", "end_x", "0.914").
param("run_34", "run_34_step_6", "end_y", "1.900").
param("run_34", "run_34_step_6", "end_z", "2.007").
param("run_34", "run_34_step_6", "duration_sec", "0.60").
param("run_34", "run_34_step_6", "mean_speed", "0.0686").
param("run_34", "run_34_step_6", "gripper_start", "0.87").
param("run_34", "run_34_step_6", "gripper_end", "2.19").
call("run_34", "run_34_llm_7", "llm_generate", "completed").
depends("run_34", "run_34_llm_7", "run_34_step_6").
call("run_34", "run_34_step_7", "transport", "completed").
depends("run_34", "run_34_step_7", "run_34_llm_7").
spawned_by("run_34", "run_34_step_7", "run_34_llm_7").
param("run_34", "run_34_step_7", "start_x", "1.066").
param("run_34", "run_34_step_7", "start_y", "1.936").
param("run_34", "run_34_step_7", "start_z", "1.996").
param("run_34", "run_34_step_7", "end_x", "1.209").
param("run_34", "run_34_step_7", "end_y", "1.821").
param("run_34", "run_34_step_7", "end_z", "1.964").
param("run_34", "run_34_step_7", "duration_sec", "0.60").
param("run_34", "run_34_step_7", "mean_speed", "0.0670").
param("run_34", "run_34_step_7", "gripper_start", "2.19").
param("run_34", "run_34_step_7", "gripper_end", "1.87").
call("run_34", "run_34_llm_8", "llm_generate", "completed").
depends("run_34", "run_34_llm_8", "run_34_step_7").
call("run_34", "run_34_step_8", "grasp", "completed").
depends("run_34", "run_34_step_8", "run_34_llm_8").
spawned_by("run_34", "run_34_step_8", "run_34_llm_8").
param("run_34", "run_34_step_8", "start_x", "1.213").
param("run_34", "run_34_step_8", "start_y", "1.818").
param("run_34", "run_34_step_8", "start_z", "1.964").
param("run_34", "run_34_step_8", "end_x", "1.099").
param("run_34", "run_34_step_8", "end_y", "1.753").
param("run_34", "run_34_step_8", "end_z", "1.889").
param("run_34", "run_34_step_8", "duration_sec", "1.40").
param("run_34", "run_34_step_8", "mean_speed", "0.1779").
param("run_34", "run_34_step_8", "gripper_start", "1.87").
param("run_34", "run_34_step_8", "gripper_end", "0.13").
call("run_34", "run_34_llm_9", "llm_generate", "completed").
depends("run_34", "run_34_llm_9", "run_34_step_8").
call("run_34", "run_34_step_9", "transport", "completed").
depends("run_34", "run_34_step_9", "run_34_llm_9").
spawned_by("run_34", "run_34_step_9", "run_34_llm_9").
param("run_34", "run_34_step_9", "start_x", "1.261").
param("run_34", "run_34_step_9", "start_y", "1.695").
param("run_34", "run_34_step_9", "start_z", "1.847").
param("run_34", "run_34_step_9", "end_x", "1.541").
param("run_34", "run_34_step_9", "end_y", "2.615").
param("run_34", "run_34_step_9", "end_z", "2.002").
param("run_34", "run_34_step_9", "duration_sec", "4.60").
param("run_34", "run_34_step_9", "mean_speed", "0.1589").
param("run_34", "run_34_step_9", "gripper_start", "0.13").
param("run_34", "run_34_step_9", "gripper_end", "0.04").
call("run_34", "run_34_llm_10", "llm_generate", "completed").
depends("run_34", "run_34_llm_10", "run_34_step_9").
call("run_34", "run_34_notify", "notify_user", "completed").
depends("run_34", "run_34_notify", "run_34_llm_10").
spawned_by("run_34", "run_34_notify", "run_34_llm_10").

% === Episode 34 (pick_and_place, 4 segments) ===
job("run_35").
job_status("run_35", "completed").
call("run_35", "run_35_step_0", "prompt_user", "completed").
call("run_35", "run_35_step_1", "llm_generate", "completed").
depends("run_35", "run_35_step_1", "run_35_step_0").
call("run_35", "run_35_llm_2", "llm_generate", "completed").
depends("run_35", "run_35_llm_2", "run_35_step_1").
call("run_35", "run_35_step_2", "release", "completed").
depends("run_35", "run_35_step_2", "run_35_llm_2").
spawned_by("run_35", "run_35_step_2", "run_35_llm_2").
param("run_35", "run_35_step_2", "start_x", "2.500").
param("run_35", "run_35_step_2", "start_y", "-0.000").
param("run_35", "run_35_step_2", "start_z", "0.200").
param("run_35", "run_35_step_2", "end_x", "4.116").
param("run_35", "run_35_step_2", "end_y", "-1.265").
param("run_35", "run_35_step_2", "end_z", "0.141").
param("run_35", "run_35_step_2", "duration_sec", "1.60").
param("run_35", "run_35_step_2", "mean_speed", "0.3159").
param("run_35", "run_35_step_2", "gripper_start", "6.04").
param("run_35", "run_35_step_2", "gripper_end", "7.96").
call("run_35", "run_35_llm_3", "llm_generate", "completed").
depends("run_35", "run_35_llm_3", "run_35_step_2").
call("run_35", "run_35_step_3", "approach", "completed").
depends("run_35", "run_35_step_3", "run_35_llm_3").
spawned_by("run_35", "run_35_step_3", "run_35_llm_3").
param("run_35", "run_35_step_3", "start_x", "4.421").
param("run_35", "run_35_step_3", "start_y", "-1.606").
param("run_35", "run_35_step_3", "start_z", "0.155").
param("run_35", "run_35_step_3", "end_x", "4.543").
param("run_35", "run_35_step_3", "end_y", "-2.368").
param("run_35", "run_35_step_3", "end_z", "0.143").
param("run_35", "run_35_step_3", "duration_sec", "0.80").
param("run_35", "run_35_step_3", "mean_speed", "0.2451").
param("run_35", "run_35_step_3", "gripper_start", "7.96").
param("run_35", "run_35_step_3", "gripper_end", "7.96").
call("run_35", "run_35_llm_4", "llm_generate", "completed").
depends("run_35", "run_35_llm_4", "run_35_step_3").
call("run_35", "run_35_step_4", "grasp", "completed").
depends("run_35", "run_35_step_4", "run_35_llm_4").
spawned_by("run_35", "run_35_step_4", "run_35_llm_4").
param("run_35", "run_35_step_4", "start_x", "4.604").
param("run_35", "run_35_step_4", "start_y", "-2.567").
param("run_35", "run_35_step_4", "start_z", "0.144").
param("run_35", "run_35_step_4", "end_x", "4.630").
param("run_35", "run_35_step_4", "end_y", "-2.864").
param("run_35", "run_35_step_4", "end_z", "0.259").
param("run_35", "run_35_step_4", "duration_sec", "1.20").
param("run_35", "run_35_step_4", "mean_speed", "0.1879").
param("run_35", "run_35_step_4", "gripper_start", "7.96").
param("run_35", "run_35_step_4", "gripper_end", "3.41").
call("run_35", "run_35_llm_5", "llm_generate", "completed").
depends("run_35", "run_35_llm_5", "run_35_step_4").
call("run_35", "run_35_step_5", "transport", "completed").
depends("run_35", "run_35_step_5", "run_35_llm_5").
spawned_by("run_35", "run_35_step_5", "run_35_llm_5").
param("run_35", "run_35_step_5", "start_x", "4.744").
param("run_35", "run_35_step_5", "start_y", "-2.699").
param("run_35", "run_35_step_5", "start_z", "0.413").
param("run_35", "run_35_step_5", "end_x", "3.034").
param("run_35", "run_35_step_5", "end_y", "-2.835").
param("run_35", "run_35_step_5", "end_z", "1.993").
param("run_35", "run_35_step_5", "duration_sec", "6.20").
param("run_35", "run_35_step_5", "mean_speed", "0.2425").
param("run_35", "run_35_step_5", "gripper_start", "3.41").
param("run_35", "run_35_step_5", "gripper_end", "3.40").
call("run_35", "run_35_llm_6", "llm_generate", "completed").
depends("run_35", "run_35_llm_6", "run_35_step_5").
call("run_35", "run_35_notify", "notify_user", "completed").
depends("run_35", "run_35_notify", "run_35_llm_6").
spawned_by("run_35", "run_35_notify", "run_35_llm_6").

% === Episode 35 (pick_and_place, 2 segments) ===
job("run_36").
job_status("run_36", "completed").
call("run_36", "run_36_step_0", "prompt_user", "completed").
call("run_36", "run_36_step_1", "llm_generate", "completed").
depends("run_36", "run_36_step_1", "run_36_step_0").
call("run_36", "run_36_llm_2", "llm_generate", "completed").
depends("run_36", "run_36_llm_2", "run_36_step_1").
call("run_36", "run_36_step_2", "release", "completed").
depends("run_36", "run_36_step_2", "run_36_llm_2").
spawned_by("run_36", "run_36_step_2", "run_36_llm_2").
param("run_36", "run_36_step_2", "start_x", "2.500").
param("run_36", "run_36_step_2", "start_y", "-0.000").
param("run_36", "run_36_step_2", "start_z", "0.200").
param("run_36", "run_36_step_2", "end_x", "2.500").
param("run_36", "run_36_step_2", "end_y", "-0.000").
param("run_36", "run_36_step_2", "end_z", "0.215").
param("run_36", "run_36_step_2", "duration_sec", "0.80").
param("run_36", "run_36_step_2", "mean_speed", "0.0073").
param("run_36", "run_36_step_2", "gripper_start", "6.03").
param("run_36", "run_36_step_2", "gripper_end", "7.95").
call("run_36", "run_36_llm_3", "llm_generate", "completed").
depends("run_36", "run_36_llm_3", "run_36_step_2").
call("run_36", "run_36_step_3", "move", "completed").
depends("run_36", "run_36_step_3", "run_36_llm_3").
spawned_by("run_36", "run_36_step_3", "run_36_llm_3").
param("run_36", "run_36_step_3", "start_x", "2.514").
param("run_36", "run_36_step_3", "start_y", "-0.000").
param("run_36", "run_36_step_3", "start_z", "0.215").
param("run_36", "run_36_step_3", "end_x", "3.478").
param("run_36", "run_36_step_3", "end_y", "-1.448").
param("run_36", "run_36_step_3", "end_z", "1.034").
param("run_36", "run_36_step_3", "duration_sec", "9.00").
param("run_36", "run_36_step_3", "mean_speed", "0.0627").
param("run_36", "run_36_step_3", "gripper_start", "7.95").
param("run_36", "run_36_step_3", "gripper_end", "7.98").
call("run_36", "run_36_llm_4", "llm_generate", "completed").
depends("run_36", "run_36_llm_4", "run_36_step_3").
call("run_36", "run_36_notify", "notify_user", "completed").
depends("run_36", "run_36_notify", "run_36_llm_4").
spawned_by("run_36", "run_36_notify", "run_36_llm_4").

% === Episode 36 (pick_and_place, 4 segments) ===
job("run_37").
job_status("run_37", "completed").
call("run_37", "run_37_step_0", "prompt_user", "completed").
call("run_37", "run_37_step_1", "llm_generate", "completed").
depends("run_37", "run_37_step_1", "run_37_step_0").
call("run_37", "run_37_llm_2", "llm_generate", "completed").
depends("run_37", "run_37_llm_2", "run_37_step_1").
call("run_37", "run_37_step_2", "approach", "completed").
depends("run_37", "run_37_step_2", "run_37_llm_2").
spawned_by("run_37", "run_37_step_2", "run_37_llm_2").
param("run_37", "run_37_step_2", "start_x", "0.900").
param("run_37", "run_37_step_2", "start_y", "2.000").
param("run_37", "run_37_step_2", "start_z", "2.000").
param("run_37", "run_37_step_2", "end_x", "2.007").
param("run_37", "run_37_step_2", "end_y", "2.405").
param("run_37", "run_37_step_2", "end_z", "1.844").
param("run_37", "run_37_step_2", "duration_sec", "1.80").
param("run_37", "run_37_step_2", "mean_speed", "0.1447").
param("run_37", "run_37_step_2", "gripper_start", "4.02").
param("run_37", "run_37_step_2", "gripper_end", "4.01").
call("run_37", "run_37_llm_3", "llm_generate", "completed").
depends("run_37", "run_37_llm_3", "run_37_step_2").
call("run_37", "run_37_step_3", "approach", "completed").
depends("run_37", "run_37_step_3", "run_37_llm_3").
spawned_by("run_37", "run_37_step_3", "run_37_llm_3").
param("run_37", "run_37_step_3", "start_x", "2.053").
param("run_37", "run_37_step_3", "start_y", "2.452").
param("run_37", "run_37_step_3", "start_z", "1.786").
param("run_37", "run_37_step_3", "end_x", "2.077").
param("run_37", "run_37_step_3", "end_y", "2.501").
param("run_37", "run_37_step_3", "end_z", "1.638").
param("run_37", "run_37_step_3", "duration_sec", "0.60").
param("run_37", "run_37_step_3", "mean_speed", "0.0847").
param("run_37", "run_37_step_3", "gripper_start", "4.01").
param("run_37", "run_37_step_3", "gripper_end", "4.01").
call("run_37", "run_37_llm_4", "llm_generate", "completed").
depends("run_37", "run_37_llm_4", "run_37_step_3").
call("run_37", "run_37_step_4", "grasp", "completed").
depends("run_37", "run_37_step_4", "run_37_llm_4").
spawned_by("run_37", "run_37_step_4", "run_37_llm_4").
param("run_37", "run_37_step_4", "start_x", "2.063").
param("run_37", "run_37_step_4", "start_y", "2.512").
param("run_37", "run_37_step_4", "start_z", "1.546").
param("run_37", "run_37_step_4", "end_x", "4.413").
param("run_37", "run_37_step_4", "end_y", "2.636").
param("run_37", "run_37_step_4", "end_z", "1.200").
param("run_37", "run_37_step_4", "duration_sec", "3.60").
param("run_37", "run_37_step_4", "mean_speed", "0.1007").
param("run_37", "run_37_step_4", "gripper_start", "4.01").
param("run_37", "run_37_step_4", "gripper_end", "2.41").
call("run_37", "run_37_llm_5", "llm_generate", "completed").
depends("run_37", "run_37_llm_5", "run_37_step_4").
call("run_37", "run_37_step_5", "release", "completed").
depends("run_37", "run_37_step_5", "run_37_llm_5").
spawned_by("run_37", "run_37_step_5", "run_37_llm_5").
param("run_37", "run_37_step_5", "start_x", "4.426").
param("run_37", "run_37_step_5", "start_y", "2.644").
param("run_37", "run_37_step_5", "start_z", "1.200").
param("run_37", "run_37_step_5", "end_x", "3.476").
param("run_37", "run_37_step_5", "end_y", "2.484").
param("run_37", "run_37_step_5", "end_z", "1.200").
param("run_37", "run_37_step_5", "duration_sec", "3.80").
param("run_37", "run_37_step_5", "mean_speed", "0.0579").
param("run_37", "run_37_step_5", "gripper_start", "0.25").
param("run_37", "run_37_step_5", "gripper_end", "3.97").
call("run_37", "run_37_llm_6", "llm_generate", "completed").
depends("run_37", "run_37_llm_6", "run_37_step_5").
call("run_37", "run_37_notify", "notify_user", "completed").
depends("run_37", "run_37_notify", "run_37_llm_6").
spawned_by("run_37", "run_37_notify", "run_37_llm_6").

% === Episode 37 (pick_and_place, 3 segments) ===
job("run_38").
job_status("run_38", "completed").
call("run_38", "run_38_step_0", "prompt_user", "completed").
call("run_38", "run_38_step_1", "llm_generate", "completed").
depends("run_38", "run_38_step_1", "run_38_step_0").
call("run_38", "run_38_llm_2", "llm_generate", "completed").
depends("run_38", "run_38_llm_2", "run_38_step_1").
call("run_38", "run_38_step_2", "release", "completed").
depends("run_38", "run_38_step_2", "run_38_llm_2").
spawned_by("run_38", "run_38_step_2", "run_38_llm_2").
param("run_38", "run_38_step_2", "start_x", "2.500").
param("run_38", "run_38_step_2", "start_y", "-0.000").
param("run_38", "run_38_step_2", "start_z", "0.200").
param("run_38", "run_38_step_2", "end_x", "4.655").
param("run_38", "run_38_step_2", "end_y", "-2.275").
param("run_38", "run_38_step_2", "end_z", "0.161").
param("run_38", "run_38_step_2", "duration_sec", "3.00").
param("run_38", "run_38_step_2", "mean_speed", "0.2266").
param("run_38", "run_38_step_2", "gripper_start", "6.03").
param("run_38", "run_38_step_2", "gripper_end", "7.96").
call("run_38", "run_38_llm_3", "llm_generate", "completed").
depends("run_38", "run_38_llm_3", "run_38_step_2").
call("run_38", "run_38_step_3", "grasp", "completed").
depends("run_38", "run_38_step_3", "run_38_llm_3").
spawned_by("run_38", "run_38_step_3", "run_38_llm_3").
param("run_38", "run_38_step_3", "start_x", "4.703").
param("run_38", "run_38_step_3", "start_y", "-2.398").
param("run_38", "run_38_step_3", "start_z", "0.171").
param("run_38", "run_38_step_3", "end_x", "4.129").
param("run_38", "run_38_step_3", "end_y", "-2.037").
param("run_38", "run_38_step_3", "end_z", "1.316").
param("run_38", "run_38_step_3", "duration_sec", "3.40").
param("run_38", "run_38_step_3", "mean_speed", "0.1001").
param("run_38", "run_38_step_3", "gripper_start", "7.96").
param("run_38", "run_38_step_3", "gripper_end", "3.36").
call("run_38", "run_38_llm_4", "llm_generate", "completed").
depends("run_38", "run_38_llm_4", "run_38_step_3").
call("run_38", "run_38_step_4", "hold", "completed").
depends("run_38", "run_38_step_4", "run_38_llm_4").
spawned_by("run_38", "run_38_step_4", "run_38_llm_4").
param("run_38", "run_38_step_4", "start_x", "4.126").
param("run_38", "run_38_step_4", "start_y", "-2.014").
param("run_38", "run_38_step_4", "start_z", "1.349").
param("run_38", "run_38_step_4", "end_x", "4.072").
param("run_38", "run_38_step_4", "end_y", "-2.197").
param("run_38", "run_38_step_4", "end_z", "1.555").
param("run_38", "run_38_step_4", "duration_sec", "3.40").
param("run_38", "run_38_step_4", "mean_speed", "0.0210").
param("run_38", "run_38_step_4", "gripper_start", "3.36").
param("run_38", "run_38_step_4", "gripper_end", "3.36").
call("run_38", "run_38_llm_5", "llm_generate", "completed").
depends("run_38", "run_38_llm_5", "run_38_step_4").
call("run_38", "run_38_notify", "notify_user", "completed").
depends("run_38", "run_38_notify", "run_38_llm_5").
spawned_by("run_38", "run_38_notify", "run_38_llm_5").

% === Episode 38 (pick_and_place, 4 segments) ===
job("run_39").
job_status("run_39", "completed").
call("run_39", "run_39_step_0", "prompt_user", "completed").
call("run_39", "run_39_step_1", "llm_generate", "completed").
depends("run_39", "run_39_step_1", "run_39_step_0").
call("run_39", "run_39_llm_2", "llm_generate", "completed").
depends("run_39", "run_39_llm_2", "run_39_step_1").
call("run_39", "run_39_step_2", "approach", "completed").
depends("run_39", "run_39_step_2", "run_39_llm_2").
spawned_by("run_39", "run_39_step_2", "run_39_llm_2").
param("run_39", "run_39_step_2", "start_x", "0.900").
param("run_39", "run_39_step_2", "start_y", "2.000").
param("run_39", "run_39_step_2", "start_z", "2.000").
param("run_39", "run_39_step_2", "end_x", "1.733").
param("run_39", "run_39_step_2", "end_y", "2.146").
param("run_39", "run_39_step_2", "end_z", "1.954").
param("run_39", "run_39_step_2", "duration_sec", "1.20").
param("run_39", "run_39_step_2", "mean_speed", "0.1666").
param("run_39", "run_39_step_2", "gripper_start", "3.98").
param("run_39", "run_39_step_2", "gripper_end", "3.98").
call("run_39", "run_39_llm_3", "llm_generate", "completed").
depends("run_39", "run_39_llm_3", "run_39_step_2").
call("run_39", "run_39_step_3", "approach", "completed").
depends("run_39", "run_39_step_3", "run_39_llm_3").
spawned_by("run_39", "run_39_step_3", "run_39_llm_3").
param("run_39", "run_39_step_3", "start_x", "1.875").
param("run_39", "run_39_step_3", "start_y", "2.185").
param("run_39", "run_39_step_3", "start_z", "1.920").
param("run_39", "run_39_step_3", "end_x", "1.778").
param("run_39", "run_39_step_3", "end_y", "2.222").
param("run_39", "run_39_step_3", "end_z", "1.663").
param("run_39", "run_39_step_3", "duration_sec", "0.60").
param("run_39", "run_39_step_3", "mean_speed", "0.1324").
param("run_39", "run_39_step_3", "gripper_start", "3.98").
param("run_39", "run_39_step_3", "gripper_end", "3.99").
call("run_39", "run_39_llm_4", "llm_generate", "completed").
depends("run_39", "run_39_llm_4", "run_39_step_3").
call("run_39", "run_39_step_4", "grasp", "completed").
depends("run_39", "run_39_step_4", "run_39_llm_4").
spawned_by("run_39", "run_39_step_4", "run_39_llm_4").
param("run_39", "run_39_step_4", "start_x", "1.736").
param("run_39", "run_39_step_4", "start_y", "2.226").
param("run_39", "run_39_step_4", "start_z", "1.575").
param("run_39", "run_39_step_4", "end_x", "4.177").
param("run_39", "run_39_step_4", "end_y", "2.434").
param("run_39", "run_39_step_4", "end_z", "1.362").
param("run_39", "run_39_step_4", "duration_sec", "2.80").
param("run_39", "run_39_step_4", "mean_speed", "0.2164").
param("run_39", "run_39_step_4", "gripper_start", "3.99").
param("run_39", "run_39_step_4", "gripper_end", "0.20").
call("run_39", "run_39_llm_5", "llm_generate", "completed").
depends("run_39", "run_39_llm_5", "run_39_step_4").
call("run_39", "run_39_step_5", "release", "completed").
depends("run_39", "run_39_step_5", "run_39_llm_5").
spawned_by("run_39", "run_39_step_5", "run_39_llm_5").
param("run_39", "run_39_step_5", "start_x", "4.436").
param("run_39", "run_39_step_5", "start_y", "2.437").
param("run_39", "run_39_step_5", "start_z", "1.263").
param("run_39", "run_39_step_5", "end_x", "4.049").
param("run_39", "run_39_step_5", "end_y", "2.304").
param("run_39", "run_39_step_5", "end_z", "1.231").
param("run_39", "run_39_step_5", "duration_sec", "5.20").
param("run_39", "run_39_step_5", "mean_speed", "0.0784").
param("run_39", "run_39_step_5", "gripper_start", "0.20").
param("run_39", "run_39_step_5", "gripper_end", "1.44").
call("run_39", "run_39_llm_6", "llm_generate", "completed").
depends("run_39", "run_39_llm_6", "run_39_step_5").
call("run_39", "run_39_notify", "notify_user", "completed").
depends("run_39", "run_39_notify", "run_39_llm_6").
spawned_by("run_39", "run_39_notify", "run_39_llm_6").

% === Episode 39 (pick_and_place, 3 segments) ===
job("run_40").
job_status("run_40", "completed").
call("run_40", "run_40_step_0", "prompt_user", "completed").
call("run_40", "run_40_step_1", "llm_generate", "completed").
depends("run_40", "run_40_step_1", "run_40_step_0").
call("run_40", "run_40_llm_2", "llm_generate", "completed").
depends("run_40", "run_40_llm_2", "run_40_step_1").
call("run_40", "run_40_step_2", "release", "completed").
depends("run_40", "run_40_step_2", "run_40_llm_2").
spawned_by("run_40", "run_40_step_2", "run_40_llm_2").
param("run_40", "run_40_step_2", "start_x", "2.500").
param("run_40", "run_40_step_2", "start_y", "-0.000").
param("run_40", "run_40_step_2", "start_z", "0.200").
param("run_40", "run_40_step_2", "end_x", "4.204").
param("run_40", "run_40_step_2", "end_y", "-1.141").
param("run_40", "run_40_step_2", "end_z", "0.087").
param("run_40", "run_40_step_2", "duration_sec", "2.00").
param("run_40", "run_40_step_2", "mean_speed", "0.2498").
param("run_40", "run_40_step_2", "gripper_start", "6.03").
param("run_40", "run_40_step_2", "gripper_end", "7.96").
call("run_40", "run_40_llm_3", "llm_generate", "completed").
depends("run_40", "run_40_llm_3", "run_40_step_2").
call("run_40", "run_40_step_3", "grasp", "completed").
depends("run_40", "run_40_step_3", "run_40_llm_3").
spawned_by("run_40", "run_40_step_3", "run_40_llm_3").
param("run_40", "run_40_step_3", "start_x", "4.519").
param("run_40", "run_40_step_3", "start_y", "-1.401").
param("run_40", "run_40_step_3", "start_z", "0.086").
param("run_40", "run_40_step_3", "end_x", "4.427").
param("run_40", "run_40_step_3", "end_y", "-2.183").
param("run_40", "run_40_step_3", "end_z", "1.446").
param("run_40", "run_40_step_3", "duration_sec", "4.20").
param("run_40", "run_40_step_3", "mean_speed", "0.1044").
param("run_40", "run_40_step_3", "gripper_start", "7.96").
param("run_40", "run_40_step_3", "gripper_end", "3.31").
call("run_40", "run_40_llm_4", "llm_generate", "completed").
depends("run_40", "run_40_llm_4", "run_40_step_3").
call("run_40", "run_40_step_4", "hold", "completed").
depends("run_40", "run_40_step_4", "run_40_llm_4").
spawned_by("run_40", "run_40_step_4", "run_40_llm_4").
param("run_40", "run_40_step_4", "start_x", "4.385").
param("run_40", "run_40_step_4", "start_y", "-2.204").
param("run_40", "run_40_step_4", "start_z", "1.457").
param("run_40", "run_40_step_4", "end_x", "4.122").
param("run_40", "run_40_step_4", "end_y", "-2.194").
param("run_40", "run_40_step_4", "end_z", "1.586").
param("run_40", "run_40_step_4", "duration_sec", "3.60").
param("run_40", "run_40_step_4", "mean_speed", "0.0227").
param("run_40", "run_40_step_4", "gripper_start", "3.31").
param("run_40", "run_40_step_4", "gripper_end", "3.31").
call("run_40", "run_40_llm_5", "llm_generate", "completed").
depends("run_40", "run_40_llm_5", "run_40_step_4").
call("run_40", "run_40_notify", "notify_user", "completed").
depends("run_40", "run_40_notify", "run_40_llm_5").
spawned_by("run_40", "run_40_notify", "run_40_llm_5").

% === Episode 40 (pick_and_place, 4 segments) ===
job("run_41").
job_status("run_41", "completed").
call("run_41", "run_41_step_0", "prompt_user", "completed").
call("run_41", "run_41_step_1", "llm_generate", "completed").
depends("run_41", "run_41_step_1", "run_41_step_0").
call("run_41", "run_41_llm_2", "llm_generate", "completed").
depends("run_41", "run_41_llm_2", "run_41_step_1").
call("run_41", "run_41_step_2", "release", "completed").
depends("run_41", "run_41_step_2", "run_41_llm_2").
spawned_by("run_41", "run_41_step_2", "run_41_llm_2").
param("run_41", "run_41_step_2", "start_x", "2.500").
param("run_41", "run_41_step_2", "start_y", "-0.000").
param("run_41", "run_41_step_2", "start_z", "0.200").
param("run_41", "run_41_step_2", "end_x", "4.495").
param("run_41", "run_41_step_2", "end_y", "-0.806").
param("run_41", "run_41_step_2", "end_z", "0.155").
param("run_41", "run_41_step_2", "duration_sec", "2.20").
param("run_41", "run_41_step_2", "mean_speed", "0.2299").
param("run_41", "run_41_step_2", "gripper_start", "6.04").
param("run_41", "run_41_step_2", "gripper_end", "7.96").
call("run_41", "run_41_llm_3", "llm_generate", "completed").
depends("run_41", "run_41_llm_3", "run_41_step_2").
call("run_41", "run_41_step_3", "approach", "completed").
depends("run_41", "run_41_step_3", "run_41_llm_3").
spawned_by("run_41", "run_41_step_3", "run_41_llm_3").
param("run_41", "run_41_step_3", "start_x", "4.699").
param("run_41", "run_41_step_3", "start_y", "-1.091").
param("run_41", "run_41_step_3", "start_z", "0.209").
param("run_41", "run_41_step_3", "end_x", "4.753").
param("run_41", "run_41_step_3", "end_y", "-1.405").
param("run_41", "run_41_step_3", "end_z", "0.227").
param("run_41", "run_41_step_3", "duration_sec", "0.60").
param("run_41", "run_41_step_3", "mean_speed", "0.1838").
param("run_41", "run_41_step_3", "gripper_start", "7.96").
param("run_41", "run_41_step_3", "gripper_end", "7.96").
call("run_41", "run_41_llm_4", "llm_generate", "completed").
depends("run_41", "run_41_llm_4", "run_41_step_3").
call("run_41", "run_41_step_4", "grasp", "completed").
depends("run_41", "run_41_step_4", "run_41_llm_4").
spawned_by("run_41", "run_41_step_4", "run_41_llm_4").
param("run_41", "run_41_step_4", "start_x", "4.737").
param("run_41", "run_41_step_4", "start_y", "-1.634").
param("run_41", "run_41_step_4", "start_z", "0.199").
param("run_41", "run_41_step_4", "end_x", "4.529").
param("run_41", "run_41_step_4", "end_y", "-2.358").
param("run_41", "run_41_step_4", "end_z", "1.792").
param("run_41", "run_41_step_4", "duration_sec", "6.20").
param("run_41", "run_41_step_4", "mean_speed", "0.0938").
param("run_41", "run_41_step_4", "gripper_start", "7.96").
param("run_41", "run_41_step_4", "gripper_end", "4.50").
call("run_41", "run_41_llm_5", "llm_generate", "completed").
depends("run_41", "run_41_llm_5", "run_41_step_4").
call("run_41", "run_41_step_5", "transport", "completed").
depends("run_41", "run_41_step_5", "run_41_llm_5").
spawned_by("run_41", "run_41_step_5", "run_41_llm_5").
param("run_41", "run_41_step_5", "start_x", "4.540").
param("run_41", "run_41_step_5", "start_y", "-2.365").
param("run_41", "run_41_step_5", "start_z", "1.796").
param("run_41", "run_41_step_5", "end_x", "4.639").
param("run_41", "run_41_step_5", "end_y", "-2.473").
param("run_41", "run_41_step_5", "end_z", "1.847").
param("run_41", "run_41_step_5", "duration_sec", "0.80").
param("run_41", "run_41_step_5", "mean_speed", "0.0608").
param("run_41", "run_41_step_5", "gripper_start", "3.28").
param("run_41", "run_41_step_5", "gripper_end", "3.14").
call("run_41", "run_41_llm_6", "llm_generate", "completed").
depends("run_41", "run_41_llm_6", "run_41_step_5").
call("run_41", "run_41_notify", "notify_user", "completed").
depends("run_41", "run_41_notify", "run_41_llm_6").
spawned_by("run_41", "run_41_notify", "run_41_llm_6").

% === Episode 41 (pick_and_place, 3 segments) ===
job("run_42").
job_status("run_42", "completed").
call("run_42", "run_42_step_0", "prompt_user", "completed").
call("run_42", "run_42_step_1", "llm_generate", "completed").
depends("run_42", "run_42_step_1", "run_42_step_0").
call("run_42", "run_42_llm_2", "llm_generate", "completed").
depends("run_42", "run_42_llm_2", "run_42_step_1").
call("run_42", "run_42_step_2", "release", "completed").
depends("run_42", "run_42_step_2", "run_42_llm_2").
spawned_by("run_42", "run_42_step_2", "run_42_llm_2").
param("run_42", "run_42_step_2", "start_x", "2.500").
param("run_42", "run_42_step_2", "start_y", "-0.000").
param("run_42", "run_42_step_2", "start_z", "0.200").
param("run_42", "run_42_step_2", "end_x", "3.754").
param("run_42", "run_42_step_2", "end_y", "-0.374").
param("run_42", "run_42_step_2", "end_z", "0.277").
param("run_42", "run_42_step_2", "duration_sec", "2.00").
param("run_42", "run_42_step_2", "mean_speed", "0.1581").
param("run_42", "run_42_step_2", "gripper_start", "6.04").
param("run_42", "run_42_step_2", "gripper_end", "7.96").
call("run_42", "run_42_llm_3", "llm_generate", "completed").
depends("run_42", "run_42_llm_3", "run_42_step_2").
call("run_42", "run_42_step_3", "grasp", "completed").
depends("run_42", "run_42_step_3", "run_42_llm_3").
spawned_by("run_42", "run_42_step_3", "run_42_llm_3").
param("run_42", "run_42_step_3", "start_x", "3.741").
param("run_42", "run_42_step_3", "start_y", "-0.499").
param("run_42", "run_42_step_3", "start_z", "0.270").
param("run_42", "run_42_step_3", "end_x", "3.704").
param("run_42", "run_42_step_3", "end_y", "-0.913").
param("run_42", "run_42_step_3", "end_z", "1.150").
param("run_42", "run_42_step_3", "duration_sec", "3.00").
param("run_42", "run_42_step_3", "mean_speed", "0.0953").
param("run_42", "run_42_step_3", "gripper_start", "7.96").
param("run_42", "run_42_step_3", "gripper_end", "0.04").
call("run_42", "run_42_llm_4", "llm_generate", "completed").
depends("run_42", "run_42_llm_4", "run_42_step_3").
call("run_42", "run_42_step_4", "hold", "completed").
depends("run_42", "run_42_step_4", "run_42_llm_4").
spawned_by("run_42", "run_42_step_4", "run_42_llm_4").
param("run_42", "run_42_step_4", "start_x", "3.691").
param("run_42", "run_42_step_4", "start_y", "-0.879").
param("run_42", "run_42_step_4", "start_z", "1.195").
param("run_42", "run_42_step_4", "end_x", "3.635").
param("run_42", "run_42_step_4", "end_y", "-0.788").
param("run_42", "run_42_step_4", "end_z", "1.111").
param("run_42", "run_42_step_4", "duration_sec", "4.80").
param("run_42", "run_42_step_4", "mean_speed", "0.0212").
param("run_42", "run_42_step_4", "gripper_start", "0.04").
param("run_42", "run_42_step_4", "gripper_end", "0.02").
call("run_42", "run_42_llm_5", "llm_generate", "completed").
depends("run_42", "run_42_llm_5", "run_42_step_4").
call("run_42", "run_42_notify", "notify_user", "completed").
depends("run_42", "run_42_notify", "run_42_llm_5").
spawned_by("run_42", "run_42_notify", "run_42_llm_5").

% === Episode 42 (pick_and_place, 8 segments) ===
job("run_43").
job_status("run_43", "completed").
call("run_43", "run_43_step_0", "prompt_user", "completed").
call("run_43", "run_43_step_1", "llm_generate", "completed").
depends("run_43", "run_43_step_1", "run_43_step_0").
call("run_43", "run_43_llm_2", "llm_generate", "completed").
depends("run_43", "run_43_llm_2", "run_43_step_1").
call("run_43", "run_43_step_2", "release", "completed").
depends("run_43", "run_43_step_2", "run_43_llm_2").
spawned_by("run_43", "run_43_step_2", "run_43_llm_2").
param("run_43", "run_43_step_2", "start_x", "2.500").
param("run_43", "run_43_step_2", "start_y", "-0.000").
param("run_43", "run_43_step_2", "start_z", "0.200").
param("run_43", "run_43_step_2", "end_x", "5.528").
param("run_43", "run_43_step_2", "end_y", "-0.228").
param("run_43", "run_43_step_2", "end_z", "0.277").
param("run_43", "run_43_step_2", "duration_sec", "2.60").
param("run_43", "run_43_step_2", "mean_speed", "0.2590").
param("run_43", "run_43_step_2", "gripper_start", "6.04").
param("run_43", "run_43_step_2", "gripper_end", "7.96").
call("run_43", "run_43_llm_3", "llm_generate", "completed").
depends("run_43", "run_43_llm_3", "run_43_step_2").
call("run_43", "run_43_step_3", "approach", "completed").
depends("run_43", "run_43_step_3", "run_43_llm_3").
spawned_by("run_43", "run_43_step_3", "run_43_llm_3").
param("run_43", "run_43_step_3", "start_x", "5.626").
param("run_43", "run_43_step_3", "start_y", "-0.298").
param("run_43", "run_43_step_3", "start_z", "0.269").
param("run_43", "run_43_step_3", "end_x", "5.453").
param("run_43", "run_43_step_3", "end_y", "-0.451").
param("run_43", "run_43_step_3", "end_z", "0.299").
param("run_43", "run_43_step_3", "duration_sec", "1.60").
param("run_43", "run_43_step_3", "mean_speed", "0.0792").
param("run_43", "run_43_step_3", "gripper_start", "7.96").
param("run_43", "run_43_step_3", "gripper_end", "7.97").
call("run_43", "run_43_llm_4", "llm_generate", "completed").
depends("run_43", "run_43_llm_4", "run_43_step_3").
call("run_43", "run_43_step_4", "grasp", "completed").
depends("run_43", "run_43_step_4", "run_43_llm_4").
spawned_by("run_43", "run_43_step_4", "run_43_llm_4").
param("run_43", "run_43_step_4", "start_x", "5.458").
param("run_43", "run_43_step_4", "start_y", "-0.385").
param("run_43", "run_43_step_4", "start_z", "0.335").
param("run_43", "run_43_step_4", "end_x", "5.500").
param("run_43", "run_43_step_4", "end_y", "-0.209").
param("run_43", "run_43_step_4", "end_z", "0.450").
param("run_43", "run_43_step_4", "duration_sec", "2.20").
param("run_43", "run_43_step_4", "mean_speed", "0.0553").
param("run_43", "run_43_step_4", "gripper_start", "7.97").
param("run_43", "run_43_step_4", "gripper_end", "6.44").
call("run_43", "run_43_llm_5", "llm_generate", "completed").
depends("run_43", "run_43_llm_5", "run_43_step_4").
call("run_43", "run_43_step_5", "retract", "completed").
depends("run_43", "run_43_step_5", "run_43_llm_5").
spawned_by("run_43", "run_43_step_5", "run_43_llm_5").
param("run_43", "run_43_step_5", "start_x", "5.500").
param("run_43", "run_43_step_5", "start_y", "-0.182").
param("run_43", "run_43_step_5", "start_z", "0.461").
param("run_43", "run_43_step_5", "end_x", "5.500").
param("run_43", "run_43_step_5", "end_y", "-0.124").
param("run_43", "run_43_step_5", "end_z", "0.476").
param("run_43", "run_43_step_5", "duration_sec", "0.60").
param("run_43", "run_43_step_5", "mean_speed", "0.0250").
param("run_43", "run_43_step_5", "gripper_start", "6.44").
param("run_43", "run_43_step_5", "gripper_end", "6.61").
call("run_43", "run_43_llm_6", "llm_generate", "completed").
depends("run_43", "run_43_llm_6", "run_43_step_5").
call("run_43", "run_43_step_6", "grasp", "completed").
depends("run_43", "run_43_step_6", "run_43_llm_6").
spawned_by("run_43", "run_43_step_6", "run_43_llm_6").
param("run_43", "run_43_step_6", "start_x", "5.500").
param("run_43", "run_43_step_6", "start_y", "-0.120").
param("run_43", "run_43_step_6", "start_z", "0.491").
param("run_43", "run_43_step_6", "end_x", "5.499").
param("run_43", "run_43_step_6", "end_y", "-0.098").
param("run_43", "run_43_step_6", "end_z", "0.538").
param("run_43", "run_43_step_6", "duration_sec", "0.60").
param("run_43", "run_43_step_6", "mean_speed", "0.0658").
param("run_43", "run_43_step_6", "gripper_start", "6.61").
param("run_43", "run_43_step_6", "gripper_end", "4.54").
call("run_43", "run_43_llm_7", "llm_generate", "completed").
depends("run_43", "run_43_llm_7", "run_43_step_6").
call("run_43", "run_43_step_7", "transport", "completed").
depends("run_43", "run_43_step_7", "run_43_llm_7").
spawned_by("run_43", "run_43_step_7", "run_43_llm_7").
param("run_43", "run_43_step_7", "start_x", "5.499").
param("run_43", "run_43_step_7", "start_y", "-0.208").
param("run_43", "run_43_step_7", "start_z", "0.565").
param("run_43", "run_43_step_7", "end_x", "5.500").
param("run_43", "run_43_step_7", "end_y", "-0.214").
param("run_43", "run_43_step_7", "end_z", "0.668").
param("run_43", "run_43_step_7", "duration_sec", "0.60").
param("run_43", "run_43_step_7", "mean_speed", "0.0843").
param("run_43", "run_43_step_7", "gripper_start", "4.54").
param("run_43", "run_43_step_7", "gripper_end", "4.45").
call("run_43", "run_43_llm_8", "llm_generate", "completed").
depends("run_43", "run_43_llm_8", "run_43_step_7").
call("run_43", "run_43_step_8", "retract", "completed").
depends("run_43", "run_43_step_8", "run_43_llm_8").
spawned_by("run_43", "run_43_step_8", "run_43_llm_8").
param("run_43", "run_43_step_8", "start_x", "5.500").
param("run_43", "run_43_step_8", "start_y", "-0.213").
param("run_43", "run_43_step_8", "start_z", "0.808").
param("run_43", "run_43_step_8", "end_x", "5.500").
param("run_43", "run_43_step_8", "end_y", "0.098").
param("run_43", "run_43_step_8", "end_z", "1.065").
param("run_43", "run_43_step_8", "duration_sec", "0.80").
param("run_43", "run_43_step_8", "mean_speed", "0.1230").
param("run_43", "run_43_step_8", "gripper_start", "4.45").
param("run_43", "run_43_step_8", "gripper_end", "4.19").
call("run_43", "run_43_llm_9", "llm_generate", "completed").
depends("run_43", "run_43_llm_9", "run_43_step_8").
call("run_43", "run_43_step_9", "grasp", "completed").
depends("run_43", "run_43_step_9", "run_43_llm_9").
spawned_by("run_43", "run_43_step_9", "run_43_llm_9").
param("run_43", "run_43_step_9", "start_x", "5.500").
param("run_43", "run_43_step_9", "start_y", "0.047").
param("run_43", "run_43_step_9", "start_z", "1.136").
param("run_43", "run_43_step_9", "end_x", "5.500").
param("run_43", "run_43_step_9", "end_y", "-0.073").
param("run_43", "run_43_step_9", "end_z", "1.413").
param("run_43", "run_43_step_9", "duration_sec", "0.80").
param("run_43", "run_43_step_9", "mean_speed", "0.1025").
param("run_43", "run_43_step_9", "gripper_start", "4.19").
param("run_43", "run_43_step_9", "gripper_end", "2.89").
call("run_43", "run_43_llm_10", "llm_generate", "completed").
depends("run_43", "run_43_llm_10", "run_43_step_9").
call("run_43", "run_43_notify", "notify_user", "completed").
depends("run_43", "run_43_notify", "run_43_llm_10").
spawned_by("run_43", "run_43_notify", "run_43_llm_10").

% === Episode 43 (pick_and_place, 4 segments) ===
job("run_44").
job_status("run_44", "completed").
call("run_44", "run_44_step_0", "prompt_user", "completed").
call("run_44", "run_44_step_1", "llm_generate", "completed").
depends("run_44", "run_44_step_1", "run_44_step_0").
call("run_44", "run_44_llm_2", "llm_generate", "completed").
depends("run_44", "run_44_llm_2", "run_44_step_1").
call("run_44", "run_44_step_2", "release", "completed").
depends("run_44", "run_44_step_2", "run_44_llm_2").
spawned_by("run_44", "run_44_step_2", "run_44_llm_2").
param("run_44", "run_44_step_2", "start_x", "2.500").
param("run_44", "run_44_step_2", "start_y", "-0.000").
param("run_44", "run_44_step_2", "start_z", "0.200").
param("run_44", "run_44_step_2", "end_x", "3.433").
param("run_44", "run_44_step_2", "end_y", "-0.289").
param("run_44", "run_44_step_2", "end_z", "0.177").
param("run_44", "run_44_step_2", "duration_sec", "1.00").
param("run_44", "run_44_step_2", "mean_speed", "0.2726").
param("run_44", "run_44_step_2", "gripper_start", "6.04").
param("run_44", "run_44_step_2", "gripper_end", "7.96").
call("run_44", "run_44_llm_3", "llm_generate", "completed").
depends("run_44", "run_44_llm_3", "run_44_step_2").
call("run_44", "run_44_step_3", "grasp", "completed").
depends("run_44", "run_44_step_3", "run_44_llm_3").
spawned_by("run_44", "run_44_step_3", "run_44_llm_3").
param("run_44", "run_44_step_3", "start_x", "3.798").
param("run_44", "run_44_step_3", "start_y", "-0.403").
param("run_44", "run_44_step_3", "start_z", "0.144").
param("run_44", "run_44_step_3", "end_x", "3.931").
param("run_44", "run_44_step_3", "end_y", "-0.514").
param("run_44", "run_44_step_3", "end_z", "0.151").
param("run_44", "run_44_step_3", "duration_sec", "0.80").
param("run_44", "run_44_step_3", "mean_speed", "0.0632").
param("run_44", "run_44_step_3", "gripper_start", "7.96").
param("run_44", "run_44_step_3", "gripper_end", "4.20").
call("run_44", "run_44_llm_4", "llm_generate", "completed").
depends("run_44", "run_44_llm_4", "run_44_step_3").
call("run_44", "run_44_step_4", "transport", "completed").
depends("run_44", "run_44_step_4", "run_44_llm_4").
spawned_by("run_44", "run_44_step_4", "run_44_llm_4").
param("run_44", "run_44_step_4", "start_x", "3.955").
param("run_44", "run_44_step_4", "start_y", "-0.453").
param("run_44", "run_44_step_4", "start_z", "0.152").
param("run_44", "run_44_step_4", "end_x", "3.767").
param("run_44", "run_44_step_4", "end_y", "-0.997").
param("run_44", "run_44_step_4", "end_z", "1.612").
param("run_44", "run_44_step_4", "duration_sec", "3.60").
param("run_44", "run_44_step_4", "mean_speed", "0.1290").
param("run_44", "run_44_step_4", "gripper_start", "4.20").
param("run_44", "run_44_step_4", "gripper_end", "3.39").
call("run_44", "run_44_llm_5", "llm_generate", "completed").
depends("run_44", "run_44_llm_5", "run_44_step_4").
call("run_44", "run_44_step_5", "hold", "completed").
depends("run_44", "run_44_step_5", "run_44_llm_5").
spawned_by("run_44", "run_44_step_5", "run_44_llm_5").
param("run_44", "run_44_step_5", "start_x", "3.702").
param("run_44", "run_44_step_5", "start_y", "-1.025").
param("run_44", "run_44_step_5", "start_z", "1.621").
param("run_44", "run_44_step_5", "end_x", "3.240").
param("run_44", "run_44_step_5", "end_y", "-1.177").
param("run_44", "run_44_step_5", "end_z", "1.927").
param("run_44", "run_44_step_5", "duration_sec", "4.40").
param("run_44", "run_44_step_5", "mean_speed", "0.0422").
param("run_44", "run_44_step_5", "gripper_start", "3.39").
param("run_44", "run_44_step_5", "gripper_end", "3.36").
call("run_44", "run_44_llm_6", "llm_generate", "completed").
depends("run_44", "run_44_llm_6", "run_44_step_5").
call("run_44", "run_44_notify", "notify_user", "completed").
depends("run_44", "run_44_notify", "run_44_llm_6").
spawned_by("run_44", "run_44_notify", "run_44_llm_6").

% === Episode 44 (pick_and_place, 3 segments) ===
job("run_45").
job_status("run_45", "completed").
call("run_45", "run_45_step_0", "prompt_user", "completed").
call("run_45", "run_45_step_1", "llm_generate", "completed").
depends("run_45", "run_45_step_1", "run_45_step_0").
call("run_45", "run_45_llm_2", "llm_generate", "completed").
depends("run_45", "run_45_llm_2", "run_45_step_1").
call("run_45", "run_45_step_2", "release", "completed").
depends("run_45", "run_45_step_2", "run_45_llm_2").
spawned_by("run_45", "run_45_step_2", "run_45_llm_2").
param("run_45", "run_45_step_2", "start_x", "2.500").
param("run_45", "run_45_step_2", "start_y", "-0.000").
param("run_45", "run_45_step_2", "start_z", "0.200").
param("run_45", "run_45_step_2", "end_x", "2.650").
param("run_45", "run_45_step_2", "end_y", "-0.231").
param("run_45", "run_45_step_2", "end_z", "0.100").
param("run_45", "run_45_step_2", "duration_sec", "1.60").
param("run_45", "run_45_step_2", "mean_speed", "0.0549").
param("run_45", "run_45_step_2", "gripper_start", "6.03").
param("run_45", "run_45_step_2", "gripper_end", "7.96").
call("run_45", "run_45_llm_3", "llm_generate", "completed").
depends("run_45", "run_45_llm_3", "run_45_step_2").
call("run_45", "run_45_step_3", "approach", "completed").
depends("run_45", "run_45_step_3", "run_45_llm_3").
spawned_by("run_45", "run_45_step_3", "run_45_llm_3").
param("run_45", "run_45_step_3", "start_x", "2.693").
param("run_45", "run_45_step_3", "start_y", "-0.324").
param("run_45", "run_45_step_3", "start_z", "0.103").
param("run_45", "run_45_step_3", "end_x", "3.584").
param("run_45", "run_45_step_3", "end_y", "-2.366").
param("run_45", "run_45_step_3", "end_z", "0.637").
param("run_45", "run_45_step_3", "duration_sec", "7.20").
param("run_45", "run_45_step_3", "mean_speed", "0.0743").
param("run_45", "run_45_step_3", "gripper_start", "7.96").
param("run_45", "run_45_step_3", "gripper_end", "7.98").
call("run_45", "run_45_llm_4", "llm_generate", "completed").
depends("run_45", "run_45_llm_4", "run_45_step_3").
call("run_45", "run_45_step_4", "grasp", "completed").
depends("run_45", "run_45_step_4", "run_45_llm_4").
spawned_by("run_45", "run_45_step_4", "run_45_llm_4").
param("run_45", "run_45_step_4", "start_x", "3.582").
param("run_45", "run_45_step_4", "start_y", "-2.404").
param("run_45", "run_45_step_4", "start_z", "0.722").
param("run_45", "run_45_step_4", "end_x", "3.531").
param("run_45", "run_45_step_4", "end_y", "-2.851").
param("run_45", "run_45_step_4", "end_z", "1.144").
param("run_45", "run_45_step_4", "duration_sec", "1.00").
param("run_45", "run_45_step_4", "mean_speed", "0.1413").
param("run_45", "run_45_step_4", "gripper_start", "7.96").
param("run_45", "run_45_step_4", "gripper_end", "5.38").
call("run_45", "run_45_llm_5", "llm_generate", "completed").
depends("run_45", "run_45_llm_5", "run_45_step_4").
call("run_45", "run_45_notify", "notify_user", "completed").
depends("run_45", "run_45_notify", "run_45_llm_5").
spawned_by("run_45", "run_45_notify", "run_45_llm_5").

% === Episode 45 (pick_and_place, 6 segments) ===
job("run_46").
job_status("run_46", "completed").
call("run_46", "run_46_step_0", "prompt_user", "completed").
call("run_46", "run_46_step_1", "llm_generate", "completed").
depends("run_46", "run_46_step_1", "run_46_step_0").
call("run_46", "run_46_llm_2", "llm_generate", "completed").
depends("run_46", "run_46_llm_2", "run_46_step_1").
call("run_46", "run_46_step_2", "grasp", "completed").
depends("run_46", "run_46_step_2", "run_46_llm_2").
spawned_by("run_46", "run_46_step_2", "run_46_llm_2").
param("run_46", "run_46_step_2", "start_x", "2.500").
param("run_46", "run_46_step_2", "start_y", "-0.000").
param("run_46", "run_46_step_2", "start_z", "0.200").
param("run_46", "run_46_step_2", "end_x", "3.085").
param("run_46", "run_46_step_2", "end_y", "-0.604").
param("run_46", "run_46_step_2", "end_z", "0.098").
param("run_46", "run_46_step_2", "duration_sec", "0.80").
param("run_46", "run_46_step_2", "mean_speed", "0.2547").
param("run_46", "run_46_step_2", "gripper_start", "6.03").
param("run_46", "run_46_step_2", "gripper_end", "2.70").
call("run_46", "run_46_llm_3", "llm_generate", "completed").
depends("run_46", "run_46_llm_3", "run_46_step_2").
call("run_46", "run_46_step_3", "transport", "completed").
depends("run_46", "run_46_step_3", "run_46_llm_3").
spawned_by("run_46", "run_46_step_3", "run_46_llm_3").
param("run_46", "run_46_step_3", "start_x", "3.047").
param("run_46", "run_46_step_3", "start_y", "-0.759").
param("run_46", "run_46_step_3", "start_z", "0.125").
param("run_46", "run_46_step_3", "end_x", "3.908").
param("run_46", "run_46_step_3", "end_y", "-0.707").
param("run_46", "run_46_step_3", "end_z", "0.915").
param("run_46", "run_46_step_3", "duration_sec", "5.40").
param("run_46", "run_46_step_3", "mean_speed", "0.2967").
param("run_46", "run_46_step_3", "gripper_start", "2.70").
param("run_46", "run_46_step_3", "gripper_end", "2.91").
call("run_46", "run_46_llm_4", "llm_generate", "completed").
depends("run_46", "run_46_llm_4", "run_46_step_3").
call("run_46", "run_46_step_4", "release", "completed").
depends("run_46", "run_46_step_4", "run_46_llm_4").
spawned_by("run_46", "run_46_step_4", "run_46_llm_4").
param("run_46", "run_46_step_4", "start_x", "4.017").
param("run_46", "run_46_step_4", "start_y", "-0.700").
param("run_46", "run_46_step_4", "start_z", "0.927").
param("run_46", "run_46_step_4", "end_x", "3.663").
param("run_46", "run_46_step_4", "end_y", "-0.263").
param("run_46", "run_46_step_4", "end_z", "1.119").
param("run_46", "run_46_step_4", "duration_sec", "0.80").
param("run_46", "run_46_step_4", "mean_speed", "0.2744").
param("run_46", "run_46_step_4", "gripper_start", "3.52").
param("run_46", "run_46_step_4", "gripper_end", "6.04").
call("run_46", "run_46_llm_5", "llm_generate", "completed").
depends("run_46", "run_46_llm_5", "run_46_step_4").
call("run_46", "run_46_step_5", "retract", "completed").
depends("run_46", "run_46_step_5", "run_46_llm_5").
spawned_by("run_46", "run_46_step_5", "run_46_llm_5").
param("run_46", "run_46_step_5", "start_x", "3.495").
param("run_46", "run_46_step_5", "start_y", "0.065").
param("run_46", "run_46_step_5", "start_z", "1.315").
param("run_46", "run_46_step_5", "end_x", "3.206").
param("run_46", "run_46_step_5", "end_y", "0.115").
param("run_46", "run_46_step_5", "end_z", "1.604").
param("run_46", "run_46_step_5", "duration_sec", "0.60").
param("run_46", "run_46_step_5", "mean_speed", "0.2126").
param("run_46", "run_46_step_5", "gripper_start", "6.04").
param("run_46", "run_46_step_5", "gripper_end", "6.97").
call("run_46", "run_46_llm_6", "llm_generate", "completed").
depends("run_46", "run_46_llm_6", "run_46_step_5").
call("run_46", "run_46_step_6", "grasp", "completed").
depends("run_46", "run_46_step_6", "run_46_llm_6").
spawned_by("run_46", "run_46_step_6", "run_46_llm_6").
param("run_46", "run_46_step_6", "start_x", "3.111").
param("run_46", "run_46_step_6", "start_y", "-0.027").
param("run_46", "run_46_step_6", "start_z", "1.701").
param("run_46", "run_46_step_6", "end_x", "2.724").
param("run_46", "run_46_step_6", "end_y", "-0.402").
param("run_46", "run_46_step_6", "end_z", "2.075").
param("run_46", "run_46_step_6", "duration_sec", "0.60").
param("run_46", "run_46_step_6", "mean_speed", "0.2523").
param("run_46", "run_46_step_6", "gripper_start", "6.97").
param("run_46", "run_46_step_6", "gripper_end", "5.76").
call("run_46", "run_46_llm_7", "llm_generate", "completed").
depends("run_46", "run_46_llm_7", "run_46_step_6").
call("run_46", "run_46_step_7", "release", "completed").
depends("run_46", "run_46_step_7", "run_46_llm_7").
spawned_by("run_46", "run_46_step_7", "run_46_llm_7").
param("run_46", "run_46_step_7", "start_x", "2.646").
param("run_46", "run_46_step_7", "start_y", "-0.430").
param("run_46", "run_46_step_7", "start_z", "2.113").
param("run_46", "run_46_step_7", "end_x", "2.560").
param("run_46", "run_46_step_7", "end_y", "-0.016").
param("run_46", "run_46_step_7", "end_z", "2.008").
param("run_46", "run_46_step_7", "duration_sec", "1.60").
param("run_46", "run_46_step_7", "mean_speed", "0.1483").
param("run_46", "run_46_step_7", "gripper_start", "5.76").
param("run_46", "run_46_step_7", "gripper_end", "7.84").
call("run_46", "run_46_llm_8", "llm_generate", "completed").
depends("run_46", "run_46_llm_8", "run_46_step_7").
call("run_46", "run_46_notify", "notify_user", "completed").
depends("run_46", "run_46_notify", "run_46_llm_8").
spawned_by("run_46", "run_46_notify", "run_46_llm_8").

% === Episode 46 (pick_and_place, 4 segments) ===
job("run_47").
job_status("run_47", "completed").
call("run_47", "run_47_step_0", "prompt_user", "completed").
call("run_47", "run_47_step_1", "llm_generate", "completed").
depends("run_47", "run_47_step_1", "run_47_step_0").
call("run_47", "run_47_llm_2", "llm_generate", "completed").
depends("run_47", "run_47_llm_2", "run_47_step_1").
call("run_47", "run_47_step_2", "approach", "completed").
depends("run_47", "run_47_step_2", "run_47_llm_2").
spawned_by("run_47", "run_47_step_2", "run_47_llm_2").
param("run_47", "run_47_step_2", "start_x", "0.900").
param("run_47", "run_47_step_2", "start_y", "2.000").
param("run_47", "run_47_step_2", "start_z", "2.000").
param("run_47", "run_47_step_2", "end_x", "1.639").
param("run_47", "run_47_step_2", "end_y", "2.365").
param("run_47", "run_47_step_2", "end_z", "1.696").
param("run_47", "run_47_step_2", "duration_sec", "1.60").
param("run_47", "run_47_step_2", "mean_speed", "0.1266").
param("run_47", "run_47_step_2", "gripper_start", "4.04").
param("run_47", "run_47_step_2", "gripper_end", "4.03").
call("run_47", "run_47_llm_3", "llm_generate", "completed").
depends("run_47", "run_47_llm_3", "run_47_step_2").
call("run_47", "run_47_step_3", "grasp", "completed").
depends("run_47", "run_47_step_3", "run_47_llm_3").
spawned_by("run_47", "run_47_step_3", "run_47_llm_3").
param("run_47", "run_47_step_3", "start_x", "1.611").
param("run_47", "run_47_step_3", "start_y", "2.432").
param("run_47", "run_47_step_3", "start_z", "1.589").
param("run_47", "run_47_step_3", "end_x", "4.173").
param("run_47", "run_47_step_3", "end_y", "2.728").
param("run_47", "run_47_step_3", "end_z", "1.200").
param("run_47", "run_47_step_3", "duration_sec", "5.60").
param("run_47", "run_47_step_3", "mean_speed", "0.0744").
param("run_47", "run_47_step_3", "gripper_start", "4.03").
param("run_47", "run_47_step_3", "gripper_end", "1.86").
call("run_47", "run_47_llm_4", "llm_generate", "completed").
depends("run_47", "run_47_llm_4", "run_47_step_3").
call("run_47", "run_47_step_4", "idle", "completed").
depends("run_47", "run_47_step_4", "run_47_llm_4").
spawned_by("run_47", "run_47_step_4", "run_47_llm_4").
param("run_47", "run_47_step_4", "start_x", "4.177").
param("run_47", "run_47_step_4", "start_y", "2.751").
param("run_47", "run_47_step_4", "start_z", "1.200").
param("run_47", "run_47_step_4", "end_x", "4.180").
param("run_47", "run_47_step_4", "end_y", "2.800").
param("run_47", "run_47_step_4", "end_z", "1.200").
param("run_47", "run_47_step_4", "duration_sec", "0.80").
param("run_47", "run_47_step_4", "mean_speed", "0.0151").
param("run_47", "run_47_step_4", "gripper_start", "0.04").
param("run_47", "run_47_step_4", "gripper_end", "0.05").
call("run_47", "run_47_llm_5", "llm_generate", "completed").
depends("run_47", "run_47_llm_5", "run_47_step_4").
call("run_47", "run_47_step_5", "release", "completed").
depends("run_47", "run_47_step_5", "run_47_llm_5").
spawned_by("run_47", "run_47_step_5", "run_47_llm_5").
param("run_47", "run_47_step_5", "start_x", "4.182").
param("run_47", "run_47_step_5", "start_y", "2.811").
param("run_47", "run_47_step_5", "start_z", "1.200").
param("run_47", "run_47_step_5", "end_x", "3.958").
param("run_47", "run_47_step_5", "end_y", "2.803").
param("run_47", "run_47_step_5", "end_z", "1.200").
param("run_47", "run_47_step_5", "duration_sec", "1.80").
param("run_47", "run_47_step_5", "mean_speed", "0.0334").
param("run_47", "run_47_step_5", "gripper_start", "0.05").
param("run_47", "run_47_step_5", "gripper_end", "3.96").
call("run_47", "run_47_llm_6", "llm_generate", "completed").
depends("run_47", "run_47_llm_6", "run_47_step_5").
call("run_47", "run_47_notify", "notify_user", "completed").
depends("run_47", "run_47_notify", "run_47_llm_6").
spawned_by("run_47", "run_47_notify", "run_47_llm_6").

% === Episode 47 (pick_and_place, 4 segments) ===
job("run_48").
job_status("run_48", "completed").
call("run_48", "run_48_step_0", "prompt_user", "completed").
call("run_48", "run_48_step_1", "llm_generate", "completed").
depends("run_48", "run_48_step_1", "run_48_step_0").
call("run_48", "run_48_llm_2", "llm_generate", "completed").
depends("run_48", "run_48_llm_2", "run_48_step_1").
call("run_48", "run_48_step_2", "release", "completed").
depends("run_48", "run_48_step_2", "run_48_llm_2").
spawned_by("run_48", "run_48_step_2", "run_48_llm_2").
param("run_48", "run_48_step_2", "start_x", "2.500").
param("run_48", "run_48_step_2", "start_y", "-0.000").
param("run_48", "run_48_step_2", "start_z", "0.200").
param("run_48", "run_48_step_2", "end_x", "4.666").
param("run_48", "run_48_step_2", "end_y", "-1.626").
param("run_48", "run_48_step_2", "end_z", "0.278").
param("run_48", "run_48_step_2", "duration_sec", "3.40").
param("run_48", "run_48_step_2", "mean_speed", "0.1802").
param("run_48", "run_48_step_2", "gripper_start", "6.03").
param("run_48", "run_48_step_2", "gripper_end", "7.97").
call("run_48", "run_48_llm_3", "llm_generate", "completed").
depends("run_48", "run_48_llm_3", "run_48_step_2").
call("run_48", "run_48_step_3", "approach", "completed").
depends("run_48", "run_48_step_3", "run_48_llm_3").
spawned_by("run_48", "run_48_step_3", "run_48_llm_3").
param("run_48", "run_48_step_3", "start_x", "4.709").
param("run_48", "run_48_step_3", "start_y", "-1.752").
param("run_48", "run_48_step_3", "start_z", "0.263").
param("run_48", "run_48_step_3", "end_x", "4.845").
param("run_48", "run_48_step_3", "end_y", "-1.743").
param("run_48", "run_48_step_3", "end_z", "0.219").
param("run_48", "run_48_step_3", "duration_sec", "0.60").
param("run_48", "run_48_step_3", "mean_speed", "0.0619").
param("run_48", "run_48_step_3", "gripper_start", "7.97").
param("run_48", "run_48_step_3", "gripper_end", "7.97").
call("run_48", "run_48_llm_4", "llm_generate", "completed").
depends("run_48", "run_48_llm_4", "run_48_step_3").
call("run_48", "run_48_step_4", "hold", "completed").
depends("run_48", "run_48_step_4", "run_48_llm_4").
spawned_by("run_48", "run_48_step_4", "run_48_llm_4").
param("run_48", "run_48_step_4", "start_x", "4.872").
param("run_48", "run_48_step_4", "start_y", "-1.729").
param("run_48", "run_48_step_4", "start_z", "0.200").
param("run_48", "run_48_step_4", "end_x", "4.904").
param("run_48", "run_48_step_4", "end_y", "-1.677").
param("run_48", "run_48_step_4", "end_z", "0.141").
param("run_48", "run_48_step_4", "duration_sec", "0.60").
param("run_48", "run_48_step_4", "mean_speed", "0.0480").
param("run_48", "run_48_step_4", "gripper_start", "7.97").
param("run_48", "run_48_step_4", "gripper_end", "7.25").
call("run_48", "run_48_llm_5", "llm_generate", "completed").
depends("run_48", "run_48_llm_5", "run_48_step_4").
call("run_48", "run_48_step_5", "grasp", "completed").
depends("run_48", "run_48_step_5", "run_48_llm_5").
spawned_by("run_48", "run_48_step_5", "run_48_llm_5").
param("run_48", "run_48_step_5", "start_x", "4.880").
param("run_48", "run_48_step_5", "start_y", "-1.625").
param("run_48", "run_48_step_5", "start_z", "0.150").
param("run_48", "run_48_step_5", "end_x", "5.022").
param("run_48", "run_48_step_5", "end_y", "-2.390").
param("run_48", "run_48_step_5", "end_z", "0.719").
param("run_48", "run_48_step_5", "duration_sec", "5.20").
param("run_48", "run_48_step_5", "mean_speed", "0.0476").
param("run_48", "run_48_step_5", "gripper_start", "7.25").
param("run_48", "run_48_step_5", "gripper_end", "0.03").
call("run_48", "run_48_llm_6", "llm_generate", "completed").
depends("run_48", "run_48_llm_6", "run_48_step_5").
call("run_48", "run_48_notify", "notify_user", "completed").
depends("run_48", "run_48_notify", "run_48_llm_6").
spawned_by("run_48", "run_48_notify", "run_48_llm_6").

% === Episode 48 (pick_and_place, 3 segments) ===
job("run_49").
job_status("run_49", "completed").
call("run_49", "run_49_step_0", "prompt_user", "completed").
call("run_49", "run_49_step_1", "llm_generate", "completed").
depends("run_49", "run_49_step_1", "run_49_step_0").
call("run_49", "run_49_llm_2", "llm_generate", "completed").
depends("run_49", "run_49_llm_2", "run_49_step_1").
call("run_49", "run_49_step_2", "release", "completed").
depends("run_49", "run_49_step_2", "run_49_llm_2").
spawned_by("run_49", "run_49_step_2", "run_49_llm_2").
param("run_49", "run_49_step_2", "start_x", "2.500").
param("run_49", "run_49_step_2", "start_y", "-0.000").
param("run_49", "run_49_step_2", "start_z", "0.200").
param("run_49", "run_49_step_2", "end_x", "3.762").
param("run_49", "run_49_step_2", "end_y", "-1.255").
param("run_49", "run_49_step_2", "end_z", "0.172").
param("run_49", "run_49_step_2", "duration_sec", "1.20").
param("run_49", "run_49_step_2", "mean_speed", "0.3422").
param("run_49", "run_49_step_2", "gripper_start", "6.04").
param("run_49", "run_49_step_2", "gripper_end", "7.96").
call("run_49", "run_49_llm_3", "llm_generate", "completed").
depends("run_49", "run_49_llm_3", "run_49_step_2").
call("run_49", "run_49_step_3", "grasp", "completed").
depends("run_49", "run_49_step_3", "run_49_llm_3").
spawned_by("run_49", "run_49_step_3", "run_49_llm_3").
param("run_49", "run_49_step_3", "start_x", "3.956").
param("run_49", "run_49_step_3", "start_y", "-1.437").
param("run_49", "run_49_step_3", "start_z", "0.186").
param("run_49", "run_49_step_3", "end_x", "5.522").
param("run_49", "run_49_step_3", "end_y", "-1.041").
param("run_49", "run_49_step_3", "end_z", "1.441").
param("run_49", "run_49_step_3", "duration_sec", "3.20").
param("run_49", "run_49_step_3", "mean_speed", "0.2283").
param("run_49", "run_49_step_3", "gripper_start", "7.96").
param("run_49", "run_49_step_3", "gripper_end", "3.69").
call("run_49", "run_49_llm_4", "llm_generate", "completed").
depends("run_49", "run_49_llm_4", "run_49_step_3").
call("run_49", "run_49_step_4", "transport", "completed").
depends("run_49", "run_49_step_4", "run_49_llm_4").
spawned_by("run_49", "run_49_step_4", "run_49_llm_4").
param("run_49", "run_49_step_4", "start_x", "5.475").
param("run_49", "run_49_step_4", "start_y", "-0.967").
param("run_49", "run_49_step_4", "start_z", "1.591").
param("run_49", "run_49_step_4", "end_x", "4.189").
param("run_49", "run_49_step_4", "end_y", "-0.587").
param("run_49", "run_49_step_4", "end_z", "1.944").
param("run_49", "run_49_step_4", "duration_sec", "5.40").
param("run_49", "run_49_step_4", "mean_speed", "0.0737").
param("run_49", "run_49_step_4", "gripper_start", "3.69").
param("run_49", "run_49_step_4", "gripper_end", "3.68").
call("run_49", "run_49_llm_5", "llm_generate", "completed").
depends("run_49", "run_49_llm_5", "run_49_step_4").
call("run_49", "run_49_notify", "notify_user", "completed").
depends("run_49", "run_49_notify", "run_49_llm_5").
spawned_by("run_49", "run_49_notify", "run_49_llm_5").

% === Episode 49 (pick_and_place, 4 segments) ===
job("run_50").
job_status("run_50", "completed").
call("run_50", "run_50_step_0", "prompt_user", "completed").
call("run_50", "run_50_step_1", "llm_generate", "completed").
depends("run_50", "run_50_step_1", "run_50_step_0").
call("run_50", "run_50_llm_2", "llm_generate", "completed").
depends("run_50", "run_50_llm_2", "run_50_step_1").
call("run_50", "run_50_step_2", "release", "completed").
depends("run_50", "run_50_step_2", "run_50_llm_2").
spawned_by("run_50", "run_50_step_2", "run_50_llm_2").
param("run_50", "run_50_step_2", "start_x", "2.500").
param("run_50", "run_50_step_2", "start_y", "-0.000").
param("run_50", "run_50_step_2", "start_z", "0.200").
param("run_50", "run_50_step_2", "end_x", "4.440").
param("run_50", "run_50_step_2", "end_y", "-0.280").
param("run_50", "run_50_step_2", "end_z", "0.176").
param("run_50", "run_50_step_2", "duration_sec", "2.40").
param("run_50", "run_50_step_2", "mean_speed", "0.1931").
param("run_50", "run_50_step_2", "gripper_start", "6.04").
param("run_50", "run_50_step_2", "gripper_end", "7.96").
call("run_50", "run_50_llm_3", "llm_generate", "completed").
depends("run_50", "run_50_llm_3", "run_50_step_2").
call("run_50", "run_50_step_3", "approach", "completed").
depends("run_50", "run_50_step_3", "run_50_llm_3").
spawned_by("run_50", "run_50_step_3", "run_50_llm_3").
param("run_50", "run_50_step_3", "start_x", "4.654").
param("run_50", "run_50_step_3", "start_y", "-0.526").
param("run_50", "run_50_step_3", "start_z", "0.245").
param("run_50", "run_50_step_3", "end_x", "4.864").
param("run_50", "run_50_step_3", "end_y", "-1.282").
param("run_50", "run_50_step_3", "end_z", "0.217").
param("run_50", "run_50_step_3", "duration_sec", "1.00").
param("run_50", "run_50_step_3", "mean_speed", "0.2174").
param("run_50", "run_50_step_3", "gripper_start", "7.96").
param("run_50", "run_50_step_3", "gripper_end", "7.96").
call("run_50", "run_50_llm_4", "llm_generate", "completed").
depends("run_50", "run_50_llm_4", "run_50_step_3").
call("run_50", "run_50_step_4", "grasp", "completed").
depends("run_50", "run_50_step_4", "run_50_llm_4").
spawned_by("run_50", "run_50_step_4", "run_50_llm_4").
param("run_50", "run_50_step_4", "start_x", "4.947").
param("run_50", "run_50_step_4", "start_y", "-1.564").
param("run_50", "run_50_step_4", "start_z", "0.186").
param("run_50", "run_50_step_4", "end_x", "4.675").
param("run_50", "run_50_step_4", "end_y", "-1.886").
param("run_50", "run_50_step_4", "end_z", "1.335").
param("run_50", "run_50_step_4", "duration_sec", "4.20").
param("run_50", "run_50_step_4", "mean_speed", "0.0804").
param("run_50", "run_50_step_4", "gripper_start", "7.96").
param("run_50", "run_50_step_4", "gripper_end", "3.30").
call("run_50", "run_50_llm_5", "llm_generate", "completed").
depends("run_50", "run_50_llm_5", "run_50_step_4").
call("run_50", "run_50_step_5", "hold", "completed").
depends("run_50", "run_50_step_5", "run_50_llm_5").
spawned_by("run_50", "run_50_step_5", "run_50_llm_5").
param("run_50", "run_50_step_5", "start_x", "4.687").
param("run_50", "run_50_step_5", "start_y", "-1.867").
param("run_50", "run_50_step_5", "start_z", "1.369").
param("run_50", "run_50_step_5", "end_x", "4.657").
param("run_50", "run_50_step_5", "end_y", "-1.934").
param("run_50", "run_50_step_5", "end_z", "1.475").
param("run_50", "run_50_step_5", "duration_sec", "2.20").
param("run_50", "run_50_step_5", "mean_speed", "0.0181").
param("run_50", "run_50_step_5", "gripper_start", "3.30").
param("run_50", "run_50_step_5", "gripper_end", "3.30").
call("run_50", "run_50_llm_6", "llm_generate", "completed").
depends("run_50", "run_50_llm_6", "run_50_step_5").
call("run_50", "run_50_notify", "notify_user", "completed").
depends("run_50", "run_50_notify", "run_50_llm_6").
spawned_by("run_50", "run_50_notify", "run_50_llm_6").
