% === Single Demo Compilation ===
% 1 real demo + 49 synthetic variations
% Task: Pick-and-place: grasp an object and move it to a target location
% Synthetic traces generated from LLM world model knowledge

% === Run 0 (real) ===
job("run_0").
job_status("run_0", "completed").
call("run_0", "run_0_s0", "prompt_user", "completed").
call("run_0", "run_0_s1", "llm_generate", "completed").
depends("run_0", "run_0_s1", "run_0_s0").
call("run_0", "run_0_l2", "llm_generate", "completed").
depends("run_0", "run_0_l2", "run_0_s1").
call("run_0", "run_0_t0", "move_to_position", "completed").
depends("run_0", "run_0_t0", "run_0_l2").
spawned_by("run_0", "run_0_t0", "run_0_l2").
param("run_0", "run_0_t0", "pos_distance", "0.78").
param("run_0", "run_0_t0", "delta_gripper", "0.0").
param("run_0", "run_0_t0", "efficiency", "1.0").
param("run_0", "run_0_t0", "mean_speed", "0.202").
call("run_0", "run_0_l3", "llm_generate", "completed").
depends("run_0", "run_0_l3", "run_0_t0").
call("run_0", "run_0_t1", "close_gripper", "completed").
depends("run_0", "run_0_t1", "run_0_l3").
spawned_by("run_0", "run_0_t1", "run_0_l3").
param("run_0", "run_0_t1", "pos_distance", "0.46").
param("run_0", "run_0_t1", "delta_gripper", "-2.0").
param("run_0", "run_0_t1", "efficiency", "0.9").
param("run_0", "run_0_t1", "mean_speed", "0.165").
call("run_0", "run_0_l4", "llm_generate", "completed").
depends("run_0", "run_0_l4", "run_0_t1").
call("run_0", "run_0_t2", "close_gripper", "completed").
depends("run_0", "run_0_t2", "run_0_l4").
spawned_by("run_0", "run_0_t2", "run_0_l4").
param("run_0", "run_0_t2", "pos_distance", "0.36").
param("run_0", "run_0_t2", "delta_gripper", "-1.8").
param("run_0", "run_0_t2", "efficiency", "1.0").
param("run_0", "run_0_t2", "mean_speed", "0.123").
call("run_0", "run_0_l5", "llm_generate", "completed").
depends("run_0", "run_0_l5", "run_0_t2").
call("run_0", "run_0_t3", "move_to_position", "completed").
depends("run_0", "run_0_t3", "run_0_l5").
spawned_by("run_0", "run_0_t3", "run_0_l5").
param("run_0", "run_0_t3", "pos_distance", "2.13").
param("run_0", "run_0_t3", "delta_gripper", "0.0").
param("run_0", "run_0_t3", "efficiency", "0.7").
param("run_0", "run_0_t3", "mean_speed", "0.259").
call("run_0", "run_0_l6", "llm_generate", "completed").
depends("run_0", "run_0_l6", "run_0_t3").
call("run_0", "run_0_t4", "move_to_position", "completed").
depends("run_0", "run_0_t4", "run_0_l6").
spawned_by("run_0", "run_0_t4", "run_0_l6").
param("run_0", "run_0_t4", "pos_distance", "0.69").
param("run_0", "run_0_t4", "delta_gripper", "1.3").
param("run_0", "run_0_t4", "efficiency", "1.0").
param("run_0", "run_0_t4", "mean_speed", "0.238").
call("run_0", "run_0_l7", "llm_generate", "completed").
depends("run_0", "run_0_l7", "run_0_t4").
call("run_0", "run_0_t5", "open_gripper", "completed").
depends("run_0", "run_0_t5", "run_0_l7").
spawned_by("run_0", "run_0_t5", "run_0_l7").
param("run_0", "run_0_t5", "pos_distance", "0.43").
param("run_0", "run_0_t5", "delta_gripper", "1.6").
param("run_0", "run_0_t5", "efficiency", "0.8").
param("run_0", "run_0_t5", "mean_speed", "0.176").
call("run_0", "run_0_l8", "llm_generate", "completed").
depends("run_0", "run_0_l8", "run_0_t5").
call("run_0", "run_0_t6", "adjust_position", "completed").
depends("run_0", "run_0_t6", "run_0_l8").
spawned_by("run_0", "run_0_t6", "run_0_l8").
param("run_0", "run_0_t6", "pos_distance", "0.25").
param("run_0", "run_0_t6", "delta_gripper", "0.8").
param("run_0", "run_0_t6", "efficiency", "0.6").
param("run_0", "run_0_t6", "mean_speed", "0.133").
call("run_0", "run_0_l9", "llm_generate", "completed").
depends("run_0", "run_0_l9", "run_0_t6").
call("run_0", "run_0_t7", "adjust_position", "completed").
depends("run_0", "run_0_t7", "run_0_l9").
spawned_by("run_0", "run_0_t7", "run_0_l9").
param("run_0", "run_0_t7", "pos_distance", "0.35").
param("run_0", "run_0_t7", "delta_gripper", "0.0").
param("run_0", "run_0_t7", "efficiency", "0.3").
param("run_0", "run_0_t7", "mean_speed", "0.058").
call("run_0", "run_0_l10", "llm_generate", "completed").
depends("run_0", "run_0_l10", "run_0_t7").
call("run_0", "run_0_n", "notify_user", "completed").
depends("run_0", "run_0_n", "run_0_l10").
spawned_by("run_0", "run_0_n", "run_0_l10").

% === Run 1 (normal_variation) ===
job("run_1").
job_status("run_1", "completed").
call("run_1", "run_1_s0", "prompt_user", "completed").
call("run_1", "run_1_s1", "llm_generate", "completed").
depends("run_1", "run_1_s1", "run_1_s0").
call("run_1", "run_1_l2", "llm_generate", "completed").
depends("run_1", "run_1_l2", "run_1_s1").
call("run_1", "run_1_t0", "move_to_position", "completed").
depends("run_1", "run_1_t0", "run_1_l2").
spawned_by("run_1", "run_1_t0", "run_1_l2").
param("run_1", "run_1_t0", "pos_distance", "0.89").
param("run_1", "run_1_t0", "delta_gripper", "0.0").
param("run_1", "run_1_t0", "efficiency", "0.9").
param("run_1", "run_1_t0", "mean_speed", "0.145").
call("run_1", "run_1_l3", "llm_generate", "completed").
depends("run_1", "run_1_l3", "run_1_t0").
call("run_1", "run_1_t1", "close_gripper", "completed").
depends("run_1", "run_1_t1", "run_1_l3").
spawned_by("run_1", "run_1_t1", "run_1_l3").
param("run_1", "run_1_t1", "pos_distance", "0.33").
param("run_1", "run_1_t1", "delta_gripper", "-2.0").
param("run_1", "run_1_t1", "efficiency", "0.9").
param("run_1", "run_1_t1", "mean_speed", "0.189").
call("run_1", "run_1_l4", "llm_generate", "completed").
depends("run_1", "run_1_l4", "run_1_t1").
call("run_1", "run_1_t2", "close_gripper", "completed").
depends("run_1", "run_1_t2", "run_1_l4").
spawned_by("run_1", "run_1_t2", "run_1_l4").
param("run_1", "run_1_t2", "pos_distance", "0.50").
param("run_1", "run_1_t2", "delta_gripper", "-1.8").
param("run_1", "run_1_t2", "efficiency", "0.9").
param("run_1", "run_1_t2", "mean_speed", "0.092").
call("run_1", "run_1_l5", "llm_generate", "completed").
depends("run_1", "run_1_l5", "run_1_t2").
call("run_1", "run_1_t3", "move_to_position", "completed").
depends("run_1", "run_1_t3", "run_1_l5").
spawned_by("run_1", "run_1_t3", "run_1_l5").
param("run_1", "run_1_t3", "pos_distance", "1.13").
param("run_1", "run_1_t3", "delta_gripper", "0.0").
param("run_1", "run_1_t3", "efficiency", "0.7").
param("run_1", "run_1_t3", "mean_speed", "0.216").
call("run_1", "run_1_l6", "llm_generate", "completed").
depends("run_1", "run_1_l6", "run_1_t3").
call("run_1", "run_1_t4", "move_to_position", "completed").
depends("run_1", "run_1_t4", "run_1_l6").
spawned_by("run_1", "run_1_t4", "run_1_l6").
param("run_1", "run_1_t4", "pos_distance", "0.36").
param("run_1", "run_1_t4", "delta_gripper", "1.3").
param("run_1", "run_1_t4", "efficiency", "1.0").
param("run_1", "run_1_t4", "mean_speed", "0.195").
call("run_1", "run_1_l7", "llm_generate", "completed").
depends("run_1", "run_1_l7", "run_1_t4").
call("run_1", "run_1_t5", "open_gripper", "completed").
depends("run_1", "run_1_t5", "run_1_l7").
spawned_by("run_1", "run_1_t5", "run_1_l7").
param("run_1", "run_1_t5", "pos_distance", "0.45").
param("run_1", "run_1_t5", "delta_gripper", "1.6").
param("run_1", "run_1_t5", "efficiency", "0.8").
param("run_1", "run_1_t5", "mean_speed", "0.146").
call("run_1", "run_1_l8", "llm_generate", "completed").
depends("run_1", "run_1_l8", "run_1_t5").
call("run_1", "run_1_t6", "adjust_position", "completed").
depends("run_1", "run_1_t6", "run_1_l8").
spawned_by("run_1", "run_1_t6", "run_1_l8").
param("run_1", "run_1_t6", "pos_distance", "0.33").
param("run_1", "run_1_t6", "delta_gripper", "0.8").
param("run_1", "run_1_t6", "efficiency", "0.7").
param("run_1", "run_1_t6", "mean_speed", "0.094").
call("run_1", "run_1_l9", "llm_generate", "completed").
depends("run_1", "run_1_l9", "run_1_t6").
call("run_1", "run_1_t7", "adjust_position", "completed").
depends("run_1", "run_1_t7", "run_1_l9").
spawned_by("run_1", "run_1_t7", "run_1_l9").
param("run_1", "run_1_t7", "pos_distance", "0.42").
param("run_1", "run_1_t7", "delta_gripper", "0.0").
param("run_1", "run_1_t7", "efficiency", "0.3").
param("run_1", "run_1_t7", "mean_speed", "0.053").
call("run_1", "run_1_l10", "llm_generate", "completed").
depends("run_1", "run_1_l10", "run_1_t7").
call("run_1", "run_1_n", "notify_user", "completed").
depends("run_1", "run_1_n", "run_1_l10").
spawned_by("run_1", "run_1_n", "run_1_l10").

% === Run 2 (normal_variation) ===
job("run_2").
job_status("run_2", "completed").
call("run_2", "run_2_s0", "prompt_user", "completed").
call("run_2", "run_2_s1", "llm_generate", "completed").
depends("run_2", "run_2_s1", "run_2_s0").
call("run_2", "run_2_l2", "llm_generate", "completed").
depends("run_2", "run_2_l2", "run_2_s1").
call("run_2", "run_2_t0", "move_to_position", "completed").
depends("run_2", "run_2_t0", "run_2_l2").
spawned_by("run_2", "run_2_t0", "run_2_l2").
param("run_2", "run_2_t0", "pos_distance", "1.14").
param("run_2", "run_2_t0", "delta_gripper", "0.0").
param("run_2", "run_2_t0", "efficiency", "0.8").
param("run_2", "run_2_t0", "mean_speed", "0.182").
call("run_2", "run_2_l3", "llm_generate", "completed").
depends("run_2", "run_2_l3", "run_2_t0").
call("run_2", "run_2_t1", "close_gripper", "completed").
depends("run_2", "run_2_t1", "run_2_l3").
spawned_by("run_2", "run_2_t1", "run_2_l3").
param("run_2", "run_2_t1", "pos_distance", "0.28").
param("run_2", "run_2_t1", "delta_gripper", "-2.0").
param("run_2", "run_2_t1", "efficiency", "0.9").
param("run_2", "run_2_t1", "mean_speed", "0.200").
call("run_2", "run_2_l4", "llm_generate", "completed").
depends("run_2", "run_2_l4", "run_2_t1").
call("run_2", "run_2_t2", "close_gripper", "completed").
depends("run_2", "run_2_t2", "run_2_l4").
spawned_by("run_2", "run_2_t2", "run_2_l4").
param("run_2", "run_2_t2", "pos_distance", "0.47").
param("run_2", "run_2_t2", "delta_gripper", "-1.8").
param("run_2", "run_2_t2", "efficiency", "0.9").
param("run_2", "run_2_t2", "mean_speed", "0.140").
call("run_2", "run_2_l5", "llm_generate", "completed").
depends("run_2", "run_2_l5", "run_2_t2").
call("run_2", "run_2_t3", "move_to_position", "completed").
depends("run_2", "run_2_t3", "run_2_l5").
spawned_by("run_2", "run_2_t3", "run_2_l5").
param("run_2", "run_2_t3", "pos_distance", "3.14").
param("run_2", "run_2_t3", "delta_gripper", "0.0").
param("run_2", "run_2_t3", "efficiency", "0.7").
param("run_2", "run_2_t3", "mean_speed", "0.240").
call("run_2", "run_2_l6", "llm_generate", "completed").
depends("run_2", "run_2_l6", "run_2_t3").
call("run_2", "run_2_t4", "move_to_position", "completed").
depends("run_2", "run_2_t4", "run_2_l6").
spawned_by("run_2", "run_2_t4", "run_2_l6").
param("run_2", "run_2_t4", "pos_distance", "0.92").
param("run_2", "run_2_t4", "delta_gripper", "1.3").
param("run_2", "run_2_t4", "efficiency", "1.0").
param("run_2", "run_2_t4", "mean_speed", "0.255").
call("run_2", "run_2_l7", "llm_generate", "completed").
depends("run_2", "run_2_l7", "run_2_t4").
call("run_2", "run_2_t5", "open_gripper", "completed").
depends("run_2", "run_2_t5", "run_2_l7").
spawned_by("run_2", "run_2_t5", "run_2_l7").
param("run_2", "run_2_t5", "pos_distance", "0.47").
param("run_2", "run_2_t5", "delta_gripper", "1.6").
param("run_2", "run_2_t5", "efficiency", "0.7").
param("run_2", "run_2_t5", "mean_speed", "0.197").
call("run_2", "run_2_l8", "llm_generate", "completed").
depends("run_2", "run_2_l8", "run_2_t5").
call("run_2", "run_2_t6", "adjust_position", "completed").
depends("run_2", "run_2_t6", "run_2_l8").
spawned_by("run_2", "run_2_t6", "run_2_l8").
param("run_2", "run_2_t6", "pos_distance", "0.18").
param("run_2", "run_2_t6", "delta_gripper", "0.8").
param("run_2", "run_2_t6", "efficiency", "0.5").
param("run_2", "run_2_t6", "mean_speed", "0.116").
call("run_2", "run_2_l9", "llm_generate", "completed").
depends("run_2", "run_2_l9", "run_2_t6").
call("run_2", "run_2_t7", "adjust_position", "completed").
depends("run_2", "run_2_t7", "run_2_l9").
spawned_by("run_2", "run_2_t7", "run_2_l9").
param("run_2", "run_2_t7", "pos_distance", "0.25").
param("run_2", "run_2_t7", "delta_gripper", "0.0").
param("run_2", "run_2_t7", "efficiency", "0.3").
param("run_2", "run_2_t7", "mean_speed", "0.044").
call("run_2", "run_2_l10", "llm_generate", "completed").
depends("run_2", "run_2_l10", "run_2_t7").
call("run_2", "run_2_n", "notify_user", "completed").
depends("run_2", "run_2_n", "run_2_l10").
spawned_by("run_2", "run_2_n", "run_2_l10").

% === Run 3 (normal_variation) ===
job("run_3").
job_status("run_3", "completed").
call("run_3", "run_3_s0", "prompt_user", "completed").
call("run_3", "run_3_s1", "llm_generate", "completed").
depends("run_3", "run_3_s1", "run_3_s0").
call("run_3", "run_3_l2", "llm_generate", "completed").
depends("run_3", "run_3_l2", "run_3_s1").
call("run_3", "run_3_t0", "move_to_position", "completed").
depends("run_3", "run_3_t0", "run_3_l2").
spawned_by("run_3", "run_3_t0", "run_3_l2").
param("run_3", "run_3_t0", "pos_distance", "0.89").
param("run_3", "run_3_t0", "delta_gripper", "0.0").
param("run_3", "run_3_t0", "efficiency", "0.9").
param("run_3", "run_3_t0", "mean_speed", "0.186").
call("run_3", "run_3_l3", "llm_generate", "completed").
depends("run_3", "run_3_l3", "run_3_t0").
call("run_3", "run_3_t1", "close_gripper", "completed").
depends("run_3", "run_3_t1", "run_3_l3").
spawned_by("run_3", "run_3_t1", "run_3_l3").
param("run_3", "run_3_t1", "pos_distance", "0.33").
param("run_3", "run_3_t1", "delta_gripper", "-2.0").
param("run_3", "run_3_t1", "efficiency", "1.0").
param("run_3", "run_3_t1", "mean_speed", "0.142").
call("run_3", "run_3_l4", "llm_generate", "completed").
depends("run_3", "run_3_l4", "run_3_t1").
call("run_3", "run_3_t2", "close_gripper", "completed").
depends("run_3", "run_3_t2", "run_3_l4").
spawned_by("run_3", "run_3_t2", "run_3_l4").
param("run_3", "run_3_t2", "pos_distance", "0.41").
param("run_3", "run_3_t2", "delta_gripper", "-1.8").
param("run_3", "run_3_t2", "efficiency", "0.8").
param("run_3", "run_3_t2", "mean_speed", "0.131").
call("run_3", "run_3_l5", "llm_generate", "completed").
depends("run_3", "run_3_l5", "run_3_t2").
call("run_3", "run_3_t3", "move_to_position", "completed").
depends("run_3", "run_3_t3", "run_3_l5").
spawned_by("run_3", "run_3_t3", "run_3_l5").
param("run_3", "run_3_t3", "pos_distance", "2.62").
param("run_3", "run_3_t3", "delta_gripper", "0.0").
param("run_3", "run_3_t3", "efficiency", "0.7").
param("run_3", "run_3_t3", "mean_speed", "0.207").
call("run_3", "run_3_l6", "llm_generate", "completed").
depends("run_3", "run_3_l6", "run_3_t3").
call("run_3", "run_3_t4", "move_to_position", "completed").
depends("run_3", "run_3_t4", "run_3_l6").
spawned_by("run_3", "run_3_t4", "run_3_l6").
param("run_3", "run_3_t4", "pos_distance", "1.03").
param("run_3", "run_3_t4", "delta_gripper", "1.3").
param("run_3", "run_3_t4", "efficiency", "0.9").
param("run_3", "run_3_t4", "mean_speed", "0.258").
call("run_3", "run_3_l7", "llm_generate", "completed").
depends("run_3", "run_3_l7", "run_3_t4").
call("run_3", "run_3_t5", "open_gripper", "completed").
depends("run_3", "run_3_t5", "run_3_l7").
spawned_by("run_3", "run_3_t5", "run_3_l7").
param("run_3", "run_3_t5", "pos_distance", "0.51").
param("run_3", "run_3_t5", "delta_gripper", "1.6").
param("run_3", "run_3_t5", "efficiency", "0.8").
param("run_3", "run_3_t5", "mean_speed", "0.212").
call("run_3", "run_3_l8", "llm_generate", "completed").
depends("run_3", "run_3_l8", "run_3_t5").
call("run_3", "run_3_t6", "adjust_position", "completed").
depends("run_3", "run_3_t6", "run_3_l8").
spawned_by("run_3", "run_3_t6", "run_3_l8").
param("run_3", "run_3_t6", "pos_distance", "0.19").
param("run_3", "run_3_t6", "delta_gripper", "0.8").
param("run_3", "run_3_t6", "efficiency", "0.6").
param("run_3", "run_3_t6", "mean_speed", "0.096").
call("run_3", "run_3_l9", "llm_generate", "completed").
depends("run_3", "run_3_l9", "run_3_t6").
call("run_3", "run_3_t7", "adjust_position", "completed").
depends("run_3", "run_3_t7", "run_3_l9").
spawned_by("run_3", "run_3_t7", "run_3_l9").
param("run_3", "run_3_t7", "pos_distance", "0.27").
param("run_3", "run_3_t7", "delta_gripper", "0.0").
param("run_3", "run_3_t7", "efficiency", "0.3").
param("run_3", "run_3_t7", "mean_speed", "0.048").
call("run_3", "run_3_l10", "llm_generate", "completed").
depends("run_3", "run_3_l10", "run_3_t7").
call("run_3", "run_3_n", "notify_user", "completed").
depends("run_3", "run_3_n", "run_3_l10").
spawned_by("run_3", "run_3_n", "run_3_l10").

% === Run 4 (normal_variation) ===
job("run_4").
job_status("run_4", "completed").
call("run_4", "run_4_s0", "prompt_user", "completed").
call("run_4", "run_4_s1", "llm_generate", "completed").
depends("run_4", "run_4_s1", "run_4_s0").
call("run_4", "run_4_l2", "llm_generate", "completed").
depends("run_4", "run_4_l2", "run_4_s1").
call("run_4", "run_4_t0", "move_to_position", "completed").
depends("run_4", "run_4_t0", "run_4_l2").
spawned_by("run_4", "run_4_t0", "run_4_l2").
param("run_4", "run_4_t0", "pos_distance", "1.08").
param("run_4", "run_4_t0", "delta_gripper", "0.0").
param("run_4", "run_4_t0", "efficiency", "1.0").
param("run_4", "run_4_t0", "mean_speed", "0.180").
call("run_4", "run_4_l3", "llm_generate", "completed").
depends("run_4", "run_4_l3", "run_4_t0").
call("run_4", "run_4_t1", "close_gripper", "completed").
depends("run_4", "run_4_t1", "run_4_l3").
spawned_by("run_4", "run_4_t1", "run_4_l3").
param("run_4", "run_4_t1", "pos_distance", "0.41").
param("run_4", "run_4_t1", "delta_gripper", "-2.0").
param("run_4", "run_4_t1", "efficiency", "0.9").
param("run_4", "run_4_t1", "mean_speed", "0.206").
call("run_4", "run_4_l4", "llm_generate", "completed").
depends("run_4", "run_4_l4", "run_4_t1").
call("run_4", "run_4_t2", "close_gripper", "completed").
depends("run_4", "run_4_t2", "run_4_l4").
spawned_by("run_4", "run_4_t2", "run_4_l4").
param("run_4", "run_4_t2", "pos_distance", "0.27").
param("run_4", "run_4_t2", "delta_gripper", "-1.8").
param("run_4", "run_4_t2", "efficiency", "0.9").
param("run_4", "run_4_t2", "mean_speed", "0.104").
call("run_4", "run_4_l5", "llm_generate", "completed").
depends("run_4", "run_4_l5", "run_4_t2").
call("run_4", "run_4_t3", "move_to_position", "completed").
depends("run_4", "run_4_t3", "run_4_l5").
spawned_by("run_4", "run_4_t3", "run_4_l5").
param("run_4", "run_4_t3", "pos_distance", "1.62").
param("run_4", "run_4_t3", "delta_gripper", "0.0").
param("run_4", "run_4_t3", "efficiency", "0.8").
param("run_4", "run_4_t3", "mean_speed", "0.273").
call("run_4", "run_4_l6", "llm_generate", "completed").
depends("run_4", "run_4_l6", "run_4_t3").
call("run_4", "run_4_t4", "move_to_position", "completed").
depends("run_4", "run_4_t4", "run_4_l6").
spawned_by("run_4", "run_4_t4", "run_4_l6").
param("run_4", "run_4_t4", "pos_distance", "0.62").
param("run_4", "run_4_t4", "delta_gripper", "1.3").
param("run_4", "run_4_t4", "efficiency", "1.0").
param("run_4", "run_4_t4", "mean_speed", "0.198").
call("run_4", "run_4_l7", "llm_generate", "completed").
depends("run_4", "run_4_l7", "run_4_t4").
call("run_4", "run_4_t5", "open_gripper", "completed").
depends("run_4", "run_4_t5", "run_4_l7").
spawned_by("run_4", "run_4_t5", "run_4_l7").
param("run_4", "run_4_t5", "pos_distance", "0.44").
param("run_4", "run_4_t5", "delta_gripper", "1.6").
param("run_4", "run_4_t5", "efficiency", "0.7").
param("run_4", "run_4_t5", "mean_speed", "0.132").
call("run_4", "run_4_l8", "llm_generate", "completed").
depends("run_4", "run_4_l8", "run_4_t5").
call("run_4", "run_4_t6", "adjust_position", "completed").
depends("run_4", "run_4_t6", "run_4_l8").
spawned_by("run_4", "run_4_t6", "run_4_l8").
param("run_4", "run_4_t6", "pos_distance", "0.15").
param("run_4", "run_4_t6", "delta_gripper", "0.8").
param("run_4", "run_4_t6", "efficiency", "0.7").
param("run_4", "run_4_t6", "mean_speed", "0.143").
call("run_4", "run_4_l9", "llm_generate", "completed").
depends("run_4", "run_4_l9", "run_4_t6").
call("run_4", "run_4_t7", "adjust_position", "completed").
depends("run_4", "run_4_t7", "run_4_l9").
spawned_by("run_4", "run_4_t7", "run_4_l9").
param("run_4", "run_4_t7", "pos_distance", "0.32").
param("run_4", "run_4_t7", "delta_gripper", "0.0").
param("run_4", "run_4_t7", "efficiency", "0.3").
param("run_4", "run_4_t7", "mean_speed", "0.043").
call("run_4", "run_4_l10", "llm_generate", "completed").
depends("run_4", "run_4_l10", "run_4_t7").
call("run_4", "run_4_n", "notify_user", "completed").
depends("run_4", "run_4_n", "run_4_l10").
spawned_by("run_4", "run_4_n", "run_4_l10").

% === Run 5 (normal_variation) ===
job("run_5").
job_status("run_5", "completed").
call("run_5", "run_5_s0", "prompt_user", "completed").
call("run_5", "run_5_s1", "llm_generate", "completed").
depends("run_5", "run_5_s1", "run_5_s0").
call("run_5", "run_5_l2", "llm_generate", "completed").
depends("run_5", "run_5_l2", "run_5_s1").
call("run_5", "run_5_t0", "move_to_position", "completed").
depends("run_5", "run_5_t0", "run_5_l2").
spawned_by("run_5", "run_5_t0", "run_5_l2").
param("run_5", "run_5_t0", "pos_distance", "1.17").
param("run_5", "run_5_t0", "delta_gripper", "0.0").
param("run_5", "run_5_t0", "efficiency", "1.0").
param("run_5", "run_5_t0", "mean_speed", "0.206").
call("run_5", "run_5_l3", "llm_generate", "completed").
depends("run_5", "run_5_l3", "run_5_t0").
call("run_5", "run_5_t1", "close_gripper", "completed").
depends("run_5", "run_5_t1", "run_5_l3").
spawned_by("run_5", "run_5_t1", "run_5_l3").
param("run_5", "run_5_t1", "pos_distance", "0.63").
param("run_5", "run_5_t1", "delta_gripper", "-2.0").
param("run_5", "run_5_t1", "efficiency", "0.9").
param("run_5", "run_5_t1", "mean_speed", "0.117").
call("run_5", "run_5_l4", "llm_generate", "completed").
depends("run_5", "run_5_l4", "run_5_t1").
call("run_5", "run_5_t2", "close_gripper", "completed").
depends("run_5", "run_5_t2", "run_5_l4").
spawned_by("run_5", "run_5_t2", "run_5_l4").
param("run_5", "run_5_t2", "pos_distance", "0.42").
param("run_5", "run_5_t2", "delta_gripper", "-1.8").
param("run_5", "run_5_t2", "efficiency", "0.9").
param("run_5", "run_5_t2", "mean_speed", "0.125").
call("run_5", "run_5_l5", "llm_generate", "completed").
depends("run_5", "run_5_l5", "run_5_t2").
call("run_5", "run_5_t3", "move_to_position", "completed").
depends("run_5", "run_5_t3", "run_5_l5").
spawned_by("run_5", "run_5_t3", "run_5_l5").
param("run_5", "run_5_t3", "pos_distance", "2.43").
param("run_5", "run_5_t3", "delta_gripper", "0.0").
param("run_5", "run_5_t3", "efficiency", "0.7").
param("run_5", "run_5_t3", "mean_speed", "0.199").
call("run_5", "run_5_l6", "llm_generate", "completed").
depends("run_5", "run_5_l6", "run_5_t3").
call("run_5", "run_5_t4", "move_to_position", "completed").
depends("run_5", "run_5_t4", "run_5_l6").
spawned_by("run_5", "run_5_t4", "run_5_l6").
param("run_5", "run_5_t4", "pos_distance", "0.66").
param("run_5", "run_5_t4", "delta_gripper", "1.3").
param("run_5", "run_5_t4", "efficiency", "1.0").
param("run_5", "run_5_t4", "mean_speed", "0.302").
call("run_5", "run_5_l7", "llm_generate", "completed").
depends("run_5", "run_5_l7", "run_5_t4").
call("run_5", "run_5_t5", "open_gripper", "completed").
depends("run_5", "run_5_t5", "run_5_l7").
spawned_by("run_5", "run_5_t5", "run_5_l7").
param("run_5", "run_5_t5", "pos_distance", "0.33").
param("run_5", "run_5_t5", "delta_gripper", "1.6").
param("run_5", "run_5_t5", "efficiency", "0.7").
param("run_5", "run_5_t5", "mean_speed", "0.176").
call("run_5", "run_5_l8", "llm_generate", "completed").
depends("run_5", "run_5_l8", "run_5_t5").
call("run_5", "run_5_t6", "adjust_position", "completed").
depends("run_5", "run_5_t6", "run_5_l8").
spawned_by("run_5", "run_5_t6", "run_5_l8").
param("run_5", "run_5_t6", "pos_distance", "0.36").
param("run_5", "run_5_t6", "delta_gripper", "0.8").
param("run_5", "run_5_t6", "efficiency", "0.6").
param("run_5", "run_5_t6", "mean_speed", "0.163").
call("run_5", "run_5_l9", "llm_generate", "completed").
depends("run_5", "run_5_l9", "run_5_t6").
call("run_5", "run_5_t7", "adjust_position", "completed").
depends("run_5", "run_5_t7", "run_5_l9").
spawned_by("run_5", "run_5_t7", "run_5_l9").
param("run_5", "run_5_t7", "pos_distance", "0.40").
param("run_5", "run_5_t7", "delta_gripper", "0.0").
param("run_5", "run_5_t7", "efficiency", "0.3").
param("run_5", "run_5_t7", "mean_speed", "0.062").
call("run_5", "run_5_l10", "llm_generate", "completed").
depends("run_5", "run_5_l10", "run_5_t7").
call("run_5", "run_5_n", "notify_user", "completed").
depends("run_5", "run_5_n", "run_5_l10").
spawned_by("run_5", "run_5_n", "run_5_l10").

% === Run 6 (normal_variation) ===
job("run_6").
job_status("run_6", "completed").
call("run_6", "run_6_s0", "prompt_user", "completed").
call("run_6", "run_6_s1", "llm_generate", "completed").
depends("run_6", "run_6_s1", "run_6_s0").
call("run_6", "run_6_l2", "llm_generate", "completed").
depends("run_6", "run_6_l2", "run_6_s1").
call("run_6", "run_6_t0", "move_to_position", "completed").
depends("run_6", "run_6_t0", "run_6_l2").
spawned_by("run_6", "run_6_t0", "run_6_l2").
param("run_6", "run_6_t0", "pos_distance", "0.99").
param("run_6", "run_6_t0", "delta_gripper", "0.0").
param("run_6", "run_6_t0", "efficiency", "1.0").
param("run_6", "run_6_t0", "mean_speed", "0.207").
call("run_6", "run_6_l3", "llm_generate", "completed").
depends("run_6", "run_6_l3", "run_6_t0").
call("run_6", "run_6_t1", "close_gripper", "completed").
depends("run_6", "run_6_t1", "run_6_l3").
spawned_by("run_6", "run_6_t1", "run_6_l3").
param("run_6", "run_6_t1", "pos_distance", "0.48").
param("run_6", "run_6_t1", "delta_gripper", "-2.0").
param("run_6", "run_6_t1", "efficiency", "0.8").
param("run_6", "run_6_t1", "mean_speed", "0.116").
call("run_6", "run_6_l4", "llm_generate", "completed").
depends("run_6", "run_6_l4", "run_6_t1").
call("run_6", "run_6_t2", "close_gripper", "completed").
depends("run_6", "run_6_t2", "run_6_l4").
spawned_by("run_6", "run_6_t2", "run_6_l4").
param("run_6", "run_6_t2", "pos_distance", "0.19").
param("run_6", "run_6_t2", "delta_gripper", "-1.8").
param("run_6", "run_6_t2", "efficiency", "1.0").
param("run_6", "run_6_t2", "mean_speed", "0.154").
call("run_6", "run_6_l5", "llm_generate", "completed").
depends("run_6", "run_6_l5", "run_6_t2").
call("run_6", "run_6_t3", "move_to_position", "completed").
depends("run_6", "run_6_t3", "run_6_l5").
spawned_by("run_6", "run_6_t3", "run_6_l5").
param("run_6", "run_6_t3", "pos_distance", "2.84").
param("run_6", "run_6_t3", "delta_gripper", "0.0").
param("run_6", "run_6_t3", "efficiency", "0.6").
param("run_6", "run_6_t3", "mean_speed", "0.229").
call("run_6", "run_6_l6", "llm_generate", "completed").
depends("run_6", "run_6_l6", "run_6_t3").
call("run_6", "run_6_t4", "move_to_position", "completed").
depends("run_6", "run_6_t4", "run_6_l6").
spawned_by("run_6", "run_6_t4", "run_6_l6").
param("run_6", "run_6_t4", "pos_distance", "0.95").
param("run_6", "run_6_t4", "delta_gripper", "1.3").
param("run_6", "run_6_t4", "efficiency", "0.8").
param("run_6", "run_6_t4", "mean_speed", "0.301").
call("run_6", "run_6_l7", "llm_generate", "completed").
depends("run_6", "run_6_l7", "run_6_t4").
call("run_6", "run_6_t5", "open_gripper", "completed").
depends("run_6", "run_6_t5", "run_6_l7").
spawned_by("run_6", "run_6_t5", "run_6_l7").
param("run_6", "run_6_t5", "pos_distance", "0.43").
param("run_6", "run_6_t5", "delta_gripper", "1.6").
param("run_6", "run_6_t5", "efficiency", "0.8").
param("run_6", "run_6_t5", "mean_speed", "0.130").
call("run_6", "run_6_l8", "llm_generate", "completed").
depends("run_6", "run_6_l8", "run_6_t5").
call("run_6", "run_6_t6", "adjust_position", "completed").
depends("run_6", "run_6_t6", "run_6_l8").
spawned_by("run_6", "run_6_t6", "run_6_l8").
param("run_6", "run_6_t6", "pos_distance", "0.32").
param("run_6", "run_6_t6", "delta_gripper", "0.8").
param("run_6", "run_6_t6", "efficiency", "0.6").
param("run_6", "run_6_t6", "mean_speed", "0.103").
call("run_6", "run_6_l9", "llm_generate", "completed").
depends("run_6", "run_6_l9", "run_6_t6").
call("run_6", "run_6_t7", "adjust_position", "completed").
depends("run_6", "run_6_t7", "run_6_l9").
spawned_by("run_6", "run_6_t7", "run_6_l9").
param("run_6", "run_6_t7", "pos_distance", "0.36").
param("run_6", "run_6_t7", "delta_gripper", "0.0").
param("run_6", "run_6_t7", "efficiency", "0.3").
param("run_6", "run_6_t7", "mean_speed", "0.050").
call("run_6", "run_6_l10", "llm_generate", "completed").
depends("run_6", "run_6_l10", "run_6_t7").
call("run_6", "run_6_n", "notify_user", "completed").
depends("run_6", "run_6_n", "run_6_l10").
spawned_by("run_6", "run_6_n", "run_6_l10").

% === Run 7 (normal_variation) ===
job("run_7").
job_status("run_7", "completed").
call("run_7", "run_7_s0", "prompt_user", "completed").
call("run_7", "run_7_s1", "llm_generate", "completed").
depends("run_7", "run_7_s1", "run_7_s0").
call("run_7", "run_7_l2", "llm_generate", "completed").
depends("run_7", "run_7_l2", "run_7_s1").
call("run_7", "run_7_t0", "move_to_position", "completed").
depends("run_7", "run_7_t0", "run_7_l2").
spawned_by("run_7", "run_7_t0", "run_7_l2").
param("run_7", "run_7_t0", "pos_distance", "0.72").
param("run_7", "run_7_t0", "delta_gripper", "0.0").
param("run_7", "run_7_t0", "efficiency", "0.9").
param("run_7", "run_7_t0", "mean_speed", "0.167").
call("run_7", "run_7_l3", "llm_generate", "completed").
depends("run_7", "run_7_l3", "run_7_t0").
call("run_7", "run_7_t1", "close_gripper", "completed").
depends("run_7", "run_7_t1", "run_7_l3").
spawned_by("run_7", "run_7_t1", "run_7_l3").
param("run_7", "run_7_t1", "pos_distance", "0.57").
param("run_7", "run_7_t1", "delta_gripper", "-2.0").
param("run_7", "run_7_t1", "efficiency", "0.8").
param("run_7", "run_7_t1", "mean_speed", "0.136").
call("run_7", "run_7_l4", "llm_generate", "completed").
depends("run_7", "run_7_l4", "run_7_t1").
call("run_7", "run_7_t2", "close_gripper", "completed").
depends("run_7", "run_7_t2", "run_7_l4").
spawned_by("run_7", "run_7_t2", "run_7_l4").
param("run_7", "run_7_t2", "pos_distance", "0.53").
param("run_7", "run_7_t2", "delta_gripper", "-1.8").
param("run_7", "run_7_t2", "efficiency", "0.9").
param("run_7", "run_7_t2", "mean_speed", "0.134").
call("run_7", "run_7_l5", "llm_generate", "completed").
depends("run_7", "run_7_l5", "run_7_t2").
call("run_7", "run_7_t3", "move_to_position", "completed").
depends("run_7", "run_7_t3", "run_7_l5").
spawned_by("run_7", "run_7_t3", "run_7_l5").
param("run_7", "run_7_t3", "pos_distance", "2.17").
param("run_7", "run_7_t3", "delta_gripper", "0.0").
param("run_7", "run_7_t3", "efficiency", "0.6").
param("run_7", "run_7_t3", "mean_speed", "0.200").
call("run_7", "run_7_l6", "llm_generate", "completed").
depends("run_7", "run_7_l6", "run_7_t3").
call("run_7", "run_7_t4", "move_to_position", "completed").
depends("run_7", "run_7_t4", "run_7_l6").
spawned_by("run_7", "run_7_t4", "run_7_l6").
param("run_7", "run_7_t4", "pos_distance", "0.58").
param("run_7", "run_7_t4", "delta_gripper", "1.3").
param("run_7", "run_7_t4", "efficiency", "0.8").
param("run_7", "run_7_t4", "mean_speed", "0.250").
call("run_7", "run_7_l7", "llm_generate", "completed").
depends("run_7", "run_7_l7", "run_7_t4").
call("run_7", "run_7_t5", "open_gripper", "completed").
depends("run_7", "run_7_t5", "run_7_l7").
spawned_by("run_7", "run_7_t5", "run_7_l7").
param("run_7", "run_7_t5", "pos_distance", "0.31").
param("run_7", "run_7_t5", "delta_gripper", "1.6").
param("run_7", "run_7_t5", "efficiency", "0.8").
param("run_7", "run_7_t5", "mean_speed", "0.130").
call("run_7", "run_7_l8", "llm_generate", "completed").
depends("run_7", "run_7_l8", "run_7_t5").
call("run_7", "run_7_t6", "adjust_position", "completed").
depends("run_7", "run_7_t6", "run_7_l8").
spawned_by("run_7", "run_7_t6", "run_7_l8").
param("run_7", "run_7_t6", "pos_distance", "0.19").
param("run_7", "run_7_t6", "delta_gripper", "0.8").
param("run_7", "run_7_t6", "efficiency", "0.7").
param("run_7", "run_7_t6", "mean_speed", "0.165").
call("run_7", "run_7_l9", "llm_generate", "completed").
depends("run_7", "run_7_l9", "run_7_t6").
call("run_7", "run_7_t7", "adjust_position", "completed").
depends("run_7", "run_7_t7", "run_7_l9").
spawned_by("run_7", "run_7_t7", "run_7_l9").
param("run_7", "run_7_t7", "pos_distance", "0.20").
param("run_7", "run_7_t7", "delta_gripper", "0.0").
param("run_7", "run_7_t7", "efficiency", "0.3").
param("run_7", "run_7_t7", "mean_speed", "0.049").
call("run_7", "run_7_l10", "llm_generate", "completed").
depends("run_7", "run_7_l10", "run_7_t7").
call("run_7", "run_7_n", "notify_user", "completed").
depends("run_7", "run_7_n", "run_7_l10").
spawned_by("run_7", "run_7_n", "run_7_l10").

% === Run 8 (normal_variation) ===
job("run_8").
job_status("run_8", "completed").
call("run_8", "run_8_s0", "prompt_user", "completed").
call("run_8", "run_8_s1", "llm_generate", "completed").
depends("run_8", "run_8_s1", "run_8_s0").
call("run_8", "run_8_l2", "llm_generate", "completed").
depends("run_8", "run_8_l2", "run_8_s1").
call("run_8", "run_8_t0", "move_to_position", "completed").
depends("run_8", "run_8_t0", "run_8_l2").
spawned_by("run_8", "run_8_t0", "run_8_l2").
param("run_8", "run_8_t0", "pos_distance", "0.56").
param("run_8", "run_8_t0", "delta_gripper", "0.0").
param("run_8", "run_8_t0", "efficiency", "1.0").
param("run_8", "run_8_t0", "mean_speed", "0.158").
call("run_8", "run_8_l3", "llm_generate", "completed").
depends("run_8", "run_8_l3", "run_8_t0").
call("run_8", "run_8_t1", "close_gripper", "completed").
depends("run_8", "run_8_t1", "run_8_l3").
spawned_by("run_8", "run_8_t1", "run_8_l3").
param("run_8", "run_8_t1", "pos_distance", "0.50").
param("run_8", "run_8_t1", "delta_gripper", "-2.0").
param("run_8", "run_8_t1", "efficiency", "1.0").
param("run_8", "run_8_t1", "mean_speed", "0.162").
call("run_8", "run_8_l4", "llm_generate", "completed").
depends("run_8", "run_8_l4", "run_8_t1").
call("run_8", "run_8_t2", "close_gripper", "completed").
depends("run_8", "run_8_t2", "run_8_l4").
spawned_by("run_8", "run_8_t2", "run_8_l4").
param("run_8", "run_8_t2", "pos_distance", "0.47").
param("run_8", "run_8_t2", "delta_gripper", "-1.8").
param("run_8", "run_8_t2", "efficiency", "0.8").
param("run_8", "run_8_t2", "mean_speed", "0.100").
call("run_8", "run_8_l5", "llm_generate", "completed").
depends("run_8", "run_8_l5", "run_8_t2").
call("run_8", "run_8_t3", "move_to_position", "completed").
depends("run_8", "run_8_t3", "run_8_l5").
spawned_by("run_8", "run_8_t3", "run_8_l5").
param("run_8", "run_8_t3", "pos_distance", "1.98").
param("run_8", "run_8_t3", "delta_gripper", "0.0").
param("run_8", "run_8_t3", "efficiency", "0.7").
param("run_8", "run_8_t3", "mean_speed", "0.247").
call("run_8", "run_8_l6", "llm_generate", "completed").
depends("run_8", "run_8_l6", "run_8_t3").
call("run_8", "run_8_t4", "move_to_position", "completed").
depends("run_8", "run_8_t4", "run_8_l6").
spawned_by("run_8", "run_8_t4", "run_8_l6").
param("run_8", "run_8_t4", "pos_distance", "0.85").
param("run_8", "run_8_t4", "delta_gripper", "1.3").
param("run_8", "run_8_t4", "efficiency", "1.0").
param("run_8", "run_8_t4", "mean_speed", "0.262").
call("run_8", "run_8_l7", "llm_generate", "completed").
depends("run_8", "run_8_l7", "run_8_t4").
call("run_8", "run_8_t5", "open_gripper", "completed").
depends("run_8", "run_8_t5", "run_8_l7").
spawned_by("run_8", "run_8_t5", "run_8_l7").
param("run_8", "run_8_t5", "pos_distance", "0.26").
param("run_8", "run_8_t5", "delta_gripper", "1.6").
param("run_8", "run_8_t5", "efficiency", "0.7").
param("run_8", "run_8_t5", "mean_speed", "0.165").
call("run_8", "run_8_l8", "llm_generate", "completed").
depends("run_8", "run_8_l8", "run_8_t5").
call("run_8", "run_8_t6", "adjust_position", "completed").
depends("run_8", "run_8_t6", "run_8_l8").
spawned_by("run_8", "run_8_t6", "run_8_l8").
param("run_8", "run_8_t6", "pos_distance", "0.35").
param("run_8", "run_8_t6", "delta_gripper", "0.8").
param("run_8", "run_8_t6", "efficiency", "0.5").
param("run_8", "run_8_t6", "mean_speed", "0.113").
call("run_8", "run_8_l9", "llm_generate", "completed").
depends("run_8", "run_8_l9", "run_8_t6").
call("run_8", "run_8_t7", "adjust_position", "completed").
depends("run_8", "run_8_t7", "run_8_l9").
spawned_by("run_8", "run_8_t7", "run_8_l9").
param("run_8", "run_8_t7", "pos_distance", "0.33").
param("run_8", "run_8_t7", "delta_gripper", "0.0").
param("run_8", "run_8_t7", "efficiency", "0.3").
param("run_8", "run_8_t7", "mean_speed", "0.056").
call("run_8", "run_8_l10", "llm_generate", "completed").
depends("run_8", "run_8_l10", "run_8_t7").
call("run_8", "run_8_n", "notify_user", "completed").
depends("run_8", "run_8_n", "run_8_l10").
spawned_by("run_8", "run_8_n", "run_8_l10").

% === Run 9 (normal_variation) ===
job("run_9").
job_status("run_9", "completed").
call("run_9", "run_9_s0", "prompt_user", "completed").
call("run_9", "run_9_s1", "llm_generate", "completed").
depends("run_9", "run_9_s1", "run_9_s0").
call("run_9", "run_9_l2", "llm_generate", "completed").
depends("run_9", "run_9_l2", "run_9_s1").
call("run_9", "run_9_t0", "move_to_position", "completed").
depends("run_9", "run_9_t0", "run_9_l2").
spawned_by("run_9", "run_9_t0", "run_9_l2").
param("run_9", "run_9_t0", "pos_distance", "0.59").
param("run_9", "run_9_t0", "delta_gripper", "0.0").
param("run_9", "run_9_t0", "efficiency", "0.9").
param("run_9", "run_9_t0", "mean_speed", "0.253").
call("run_9", "run_9_l3", "llm_generate", "completed").
depends("run_9", "run_9_l3", "run_9_t0").
call("run_9", "run_9_t1", "close_gripper", "completed").
depends("run_9", "run_9_t1", "run_9_l3").
spawned_by("run_9", "run_9_t1", "run_9_l3").
param("run_9", "run_9_t1", "pos_distance", "0.63").
param("run_9", "run_9_t1", "delta_gripper", "-2.0").
param("run_9", "run_9_t1", "efficiency", "0.8").
param("run_9", "run_9_t1", "mean_speed", "0.170").
call("run_9", "run_9_l4", "llm_generate", "completed").
depends("run_9", "run_9_l4", "run_9_t1").
call("run_9", "run_9_t2", "close_gripper", "completed").
depends("run_9", "run_9_t2", "run_9_l4").
spawned_by("run_9", "run_9_t2", "run_9_l4").
param("run_9", "run_9_t2", "pos_distance", "0.54").
param("run_9", "run_9_t2", "delta_gripper", "-1.8").
param("run_9", "run_9_t2", "efficiency", "1.0").
param("run_9", "run_9_t2", "mean_speed", "0.147").
call("run_9", "run_9_l5", "llm_generate", "completed").
depends("run_9", "run_9_l5", "run_9_t2").
call("run_9", "run_9_t3", "move_to_position", "completed").
depends("run_9", "run_9_t3", "run_9_l5").
spawned_by("run_9", "run_9_t3", "run_9_l5").
param("run_9", "run_9_t3", "pos_distance", "3.04").
param("run_9", "run_9_t3", "delta_gripper", "0.0").
param("run_9", "run_9_t3", "efficiency", "0.6").
param("run_9", "run_9_t3", "mean_speed", "0.314").
call("run_9", "run_9_l6", "llm_generate", "completed").
depends("run_9", "run_9_l6", "run_9_t3").
call("run_9", "run_9_t4", "move_to_position", "completed").
depends("run_9", "run_9_t4", "run_9_l6").
spawned_by("run_9", "run_9_t4", "run_9_l6").
param("run_9", "run_9_t4", "pos_distance", "0.68").
param("run_9", "run_9_t4", "delta_gripper", "1.3").
param("run_9", "run_9_t4", "efficiency", "0.9").
param("run_9", "run_9_t4", "mean_speed", "0.197").
call("run_9", "run_9_l7", "llm_generate", "completed").
depends("run_9", "run_9_l7", "run_9_t4").
call("run_9", "run_9_t5", "open_gripper", "completed").
depends("run_9", "run_9_t5", "run_9_l7").
spawned_by("run_9", "run_9_t5", "run_9_l7").
param("run_9", "run_9_t5", "pos_distance", "0.24").
param("run_9", "run_9_t5", "delta_gripper", "1.6").
param("run_9", "run_9_t5", "efficiency", "0.9").
param("run_9", "run_9_t5", "mean_speed", "0.163").
call("run_9", "run_9_l8", "llm_generate", "completed").
depends("run_9", "run_9_l8", "run_9_t5").
call("run_9", "run_9_t6", "adjust_position", "completed").
depends("run_9", "run_9_t6", "run_9_l8").
spawned_by("run_9", "run_9_t6", "run_9_l8").
param("run_9", "run_9_t6", "pos_distance", "0.19").
param("run_9", "run_9_t6", "delta_gripper", "0.8").
param("run_9", "run_9_t6", "efficiency", "0.6").
param("run_9", "run_9_t6", "mean_speed", "0.156").
call("run_9", "run_9_l9", "llm_generate", "completed").
depends("run_9", "run_9_l9", "run_9_t6").
call("run_9", "run_9_t7", "adjust_position", "completed").
depends("run_9", "run_9_t7", "run_9_l9").
spawned_by("run_9", "run_9_t7", "run_9_l9").
param("run_9", "run_9_t7", "pos_distance", "0.32").
param("run_9", "run_9_t7", "delta_gripper", "0.0").
param("run_9", "run_9_t7", "efficiency", "0.3").
param("run_9", "run_9_t7", "mean_speed", "0.074").
call("run_9", "run_9_l10", "llm_generate", "completed").
depends("run_9", "run_9_l10", "run_9_t7").
call("run_9", "run_9_n", "notify_user", "completed").
depends("run_9", "run_9_n", "run_9_l10").
spawned_by("run_9", "run_9_n", "run_9_l10").

% === Run 10 (normal_variation) ===
job("run_10").
job_status("run_10", "completed").
call("run_10", "run_10_s0", "prompt_user", "completed").
call("run_10", "run_10_s1", "llm_generate", "completed").
depends("run_10", "run_10_s1", "run_10_s0").
call("run_10", "run_10_l2", "llm_generate", "completed").
depends("run_10", "run_10_l2", "run_10_s1").
call("run_10", "run_10_t0", "move_to_position", "completed").
depends("run_10", "run_10_t0", "run_10_l2").
spawned_by("run_10", "run_10_t0", "run_10_l2").
param("run_10", "run_10_t0", "pos_distance", "0.83").
param("run_10", "run_10_t0", "delta_gripper", "0.0").
param("run_10", "run_10_t0", "efficiency", "0.8").
param("run_10", "run_10_t0", "mean_speed", "0.229").
call("run_10", "run_10_l3", "llm_generate", "completed").
depends("run_10", "run_10_l3", "run_10_t0").
call("run_10", "run_10_t1", "close_gripper", "completed").
depends("run_10", "run_10_t1", "run_10_l3").
spawned_by("run_10", "run_10_t1", "run_10_l3").
param("run_10", "run_10_t1", "pos_distance", "0.37").
param("run_10", "run_10_t1", "delta_gripper", "-2.0").
param("run_10", "run_10_t1", "efficiency", "0.9").
param("run_10", "run_10_t1", "mean_speed", "0.212").
call("run_10", "run_10_l4", "llm_generate", "completed").
depends("run_10", "run_10_l4", "run_10_t1").
call("run_10", "run_10_t2", "close_gripper", "completed").
depends("run_10", "run_10_t2", "run_10_l4").
spawned_by("run_10", "run_10_t2", "run_10_l4").
param("run_10", "run_10_t2", "pos_distance", "0.37").
param("run_10", "run_10_t2", "delta_gripper", "-1.8").
param("run_10", "run_10_t2", "efficiency", "0.8").
param("run_10", "run_10_t2", "mean_speed", "0.141").
call("run_10", "run_10_l5", "llm_generate", "completed").
depends("run_10", "run_10_l5", "run_10_t2").
call("run_10", "run_10_t3", "move_to_position", "completed").
depends("run_10", "run_10_t3", "run_10_l5").
spawned_by("run_10", "run_10_t3", "run_10_l5").
param("run_10", "run_10_t3", "pos_distance", "2.31").
param("run_10", "run_10_t3", "delta_gripper", "0.0").
param("run_10", "run_10_t3", "efficiency", "0.8").
param("run_10", "run_10_t3", "mean_speed", "0.260").
call("run_10", "run_10_l6", "llm_generate", "completed").
depends("run_10", "run_10_l6", "run_10_t3").
call("run_10", "run_10_t4", "move_to_position", "completed").
depends("run_10", "run_10_t4", "run_10_l6").
spawned_by("run_10", "run_10_t4", "run_10_l6").
param("run_10", "run_10_t4", "pos_distance", "0.45").
param("run_10", "run_10_t4", "delta_gripper", "1.3").
param("run_10", "run_10_t4", "efficiency", "0.8").
param("run_10", "run_10_t4", "mean_speed", "0.303").
call("run_10", "run_10_l7", "llm_generate", "completed").
depends("run_10", "run_10_l7", "run_10_t4").
call("run_10", "run_10_t5", "open_gripper", "completed").
depends("run_10", "run_10_t5", "run_10_l7").
spawned_by("run_10", "run_10_t5", "run_10_l7").
param("run_10", "run_10_t5", "pos_distance", "0.30").
param("run_10", "run_10_t5", "delta_gripper", "1.6").
param("run_10", "run_10_t5", "efficiency", "0.8").
param("run_10", "run_10_t5", "mean_speed", "0.186").
call("run_10", "run_10_l8", "llm_generate", "completed").
depends("run_10", "run_10_l8", "run_10_t5").
call("run_10", "run_10_t6", "adjust_position", "completed").
depends("run_10", "run_10_t6", "run_10_l8").
spawned_by("run_10", "run_10_t6", "run_10_l8").
param("run_10", "run_10_t6", "pos_distance", "0.19").
param("run_10", "run_10_t6", "delta_gripper", "0.8").
param("run_10", "run_10_t6", "efficiency", "0.7").
param("run_10", "run_10_t6", "mean_speed", "0.103").
call("run_10", "run_10_l9", "llm_generate", "completed").
depends("run_10", "run_10_l9", "run_10_t6").
call("run_10", "run_10_t7", "adjust_position", "completed").
depends("run_10", "run_10_t7", "run_10_l9").
spawned_by("run_10", "run_10_t7", "run_10_l9").
param("run_10", "run_10_t7", "pos_distance", "0.26").
param("run_10", "run_10_t7", "delta_gripper", "0.0").
param("run_10", "run_10_t7", "efficiency", "0.3").
param("run_10", "run_10_t7", "mean_speed", "0.062").
call("run_10", "run_10_l10", "llm_generate", "completed").
depends("run_10", "run_10_l10", "run_10_t7").
call("run_10", "run_10_n", "notify_user", "completed").
depends("run_10", "run_10_n", "run_10_l10").
spawned_by("run_10", "run_10_n", "run_10_l10").

% === Run 11 (normal_variation) ===
job("run_11").
job_status("run_11", "completed").
call("run_11", "run_11_s0", "prompt_user", "completed").
call("run_11", "run_11_s1", "llm_generate", "completed").
depends("run_11", "run_11_s1", "run_11_s0").
call("run_11", "run_11_l2", "llm_generate", "completed").
depends("run_11", "run_11_l2", "run_11_s1").
call("run_11", "run_11_t0", "move_to_position", "completed").
depends("run_11", "run_11_t0", "run_11_l2").
spawned_by("run_11", "run_11_t0", "run_11_l2").
param("run_11", "run_11_t0", "pos_distance", "0.72").
param("run_11", "run_11_t0", "delta_gripper", "0.0").
param("run_11", "run_11_t0", "efficiency", "0.9").
param("run_11", "run_11_t0", "mean_speed", "0.212").
call("run_11", "run_11_l3", "llm_generate", "completed").
depends("run_11", "run_11_l3", "run_11_t0").
call("run_11", "run_11_t1", "close_gripper", "completed").
depends("run_11", "run_11_t1", "run_11_l3").
spawned_by("run_11", "run_11_t1", "run_11_l3").
param("run_11", "run_11_t1", "pos_distance", "0.66").
param("run_11", "run_11_t1", "delta_gripper", "-2.0").
param("run_11", "run_11_t1", "efficiency", "0.9").
param("run_11", "run_11_t1", "mean_speed", "0.136").
call("run_11", "run_11_l4", "llm_generate", "completed").
depends("run_11", "run_11_l4", "run_11_t1").
call("run_11", "run_11_t2", "close_gripper", "completed").
depends("run_11", "run_11_t2", "run_11_l4").
spawned_by("run_11", "run_11_t2", "run_11_l4").
param("run_11", "run_11_t2", "pos_distance", "0.26").
param("run_11", "run_11_t2", "delta_gripper", "-1.8").
param("run_11", "run_11_t2", "efficiency", "1.0").
param("run_11", "run_11_t2", "mean_speed", "0.115").
call("run_11", "run_11_l5", "llm_generate", "completed").
depends("run_11", "run_11_l5", "run_11_t2").
call("run_11", "run_11_t3", "move_to_position", "completed").
depends("run_11", "run_11_t3", "run_11_l5").
spawned_by("run_11", "run_11_t3", "run_11_l5").
param("run_11", "run_11_t3", "pos_distance", "1.70").
param("run_11", "run_11_t3", "delta_gripper", "0.0").
param("run_11", "run_11_t3", "efficiency", "0.8").
param("run_11", "run_11_t3", "mean_speed", "0.231").
call("run_11", "run_11_l6", "llm_generate", "completed").
depends("run_11", "run_11_l6", "run_11_t3").
call("run_11", "run_11_t4", "move_to_position", "completed").
depends("run_11", "run_11_t4", "run_11_l6").
spawned_by("run_11", "run_11_t4", "run_11_l6").
param("run_11", "run_11_t4", "pos_distance", "0.39").
param("run_11", "run_11_t4", "delta_gripper", "1.3").
param("run_11", "run_11_t4", "efficiency", "1.0").
param("run_11", "run_11_t4", "mean_speed", "0.232").
call("run_11", "run_11_l7", "llm_generate", "completed").
depends("run_11", "run_11_l7", "run_11_t4").
call("run_11", "run_11_t5", "open_gripper", "completed").
depends("run_11", "run_11_t5", "run_11_l7").
spawned_by("run_11", "run_11_t5", "run_11_l7").
param("run_11", "run_11_t5", "pos_distance", "0.65").
param("run_11", "run_11_t5", "delta_gripper", "1.6").
param("run_11", "run_11_t5", "efficiency", "0.7").
param("run_11", "run_11_t5", "mean_speed", "0.131").
call("run_11", "run_11_l8", "llm_generate", "completed").
depends("run_11", "run_11_l8", "run_11_t5").
call("run_11", "run_11_t6", "adjust_position", "completed").
depends("run_11", "run_11_t6", "run_11_l8").
spawned_by("run_11", "run_11_t6", "run_11_l8").
param("run_11", "run_11_t6", "pos_distance", "0.19").
param("run_11", "run_11_t6", "delta_gripper", "0.8").
param("run_11", "run_11_t6", "efficiency", "0.7").
param("run_11", "run_11_t6", "mean_speed", "0.168").
call("run_11", "run_11_l9", "llm_generate", "completed").
depends("run_11", "run_11_l9", "run_11_t6").
call("run_11", "run_11_t7", "adjust_position", "completed").
depends("run_11", "run_11_t7", "run_11_l9").
spawned_by("run_11", "run_11_t7", "run_11_l9").
param("run_11", "run_11_t7", "pos_distance", "0.48").
param("run_11", "run_11_t7", "delta_gripper", "0.0").
param("run_11", "run_11_t7", "efficiency", "0.3").
param("run_11", "run_11_t7", "mean_speed", "0.054").
call("run_11", "run_11_l10", "llm_generate", "completed").
depends("run_11", "run_11_l10", "run_11_t7").
call("run_11", "run_11_n", "notify_user", "completed").
depends("run_11", "run_11_n", "run_11_l10").
spawned_by("run_11", "run_11_n", "run_11_l10").

% === Run 12 (normal_variation) ===
job("run_12").
job_status("run_12", "completed").
call("run_12", "run_12_s0", "prompt_user", "completed").
call("run_12", "run_12_s1", "llm_generate", "completed").
depends("run_12", "run_12_s1", "run_12_s0").
call("run_12", "run_12_l2", "llm_generate", "completed").
depends("run_12", "run_12_l2", "run_12_s1").
call("run_12", "run_12_t0", "move_to_position", "completed").
depends("run_12", "run_12_t0", "run_12_l2").
spawned_by("run_12", "run_12_t0", "run_12_l2").
param("run_12", "run_12_t0", "pos_distance", "1.05").
param("run_12", "run_12_t0", "delta_gripper", "0.0").
param("run_12", "run_12_t0", "efficiency", "1.0").
param("run_12", "run_12_t0", "mean_speed", "0.227").
call("run_12", "run_12_l3", "llm_generate", "completed").
depends("run_12", "run_12_l3", "run_12_t0").
call("run_12", "run_12_t1", "close_gripper", "completed").
depends("run_12", "run_12_t1", "run_12_l3").
spawned_by("run_12", "run_12_t1", "run_12_l3").
param("run_12", "run_12_t1", "pos_distance", "0.69").
param("run_12", "run_12_t1", "delta_gripper", "-2.0").
param("run_12", "run_12_t1", "efficiency", "0.7").
param("run_12", "run_12_t1", "mean_speed", "0.180").
call("run_12", "run_12_l4", "llm_generate", "completed").
depends("run_12", "run_12_l4", "run_12_t1").
call("run_12", "run_12_t2", "close_gripper", "completed").
depends("run_12", "run_12_t2", "run_12_l4").
spawned_by("run_12", "run_12_t2", "run_12_l4").
param("run_12", "run_12_t2", "pos_distance", "0.47").
param("run_12", "run_12_t2", "delta_gripper", "-1.8").
param("run_12", "run_12_t2", "efficiency", "1.0").
param("run_12", "run_12_t2", "mean_speed", "0.108").
call("run_12", "run_12_l5", "llm_generate", "completed").
depends("run_12", "run_12_l5", "run_12_t2").
call("run_12", "run_12_t3", "move_to_position", "completed").
depends("run_12", "run_12_t3", "run_12_l5").
spawned_by("run_12", "run_12_t3", "run_12_l5").
param("run_12", "run_12_t3", "pos_distance", "3.06").
param("run_12", "run_12_t3", "delta_gripper", "0.0").
param("run_12", "run_12_t3", "efficiency", "0.6").
param("run_12", "run_12_t3", "mean_speed", "0.202").
call("run_12", "run_12_l6", "llm_generate", "completed").
depends("run_12", "run_12_l6", "run_12_t3").
call("run_12", "run_12_t4", "move_to_position", "completed").
depends("run_12", "run_12_t4", "run_12_l6").
spawned_by("run_12", "run_12_t4", "run_12_l6").
param("run_12", "run_12_t4", "pos_distance", "0.42").
param("run_12", "run_12_t4", "delta_gripper", "1.3").
param("run_12", "run_12_t4", "efficiency", "0.9").
param("run_12", "run_12_t4", "mean_speed", "0.245").
call("run_12", "run_12_l7", "llm_generate", "completed").
depends("run_12", "run_12_l7", "run_12_t4").
call("run_12", "run_12_t5", "open_gripper", "completed").
depends("run_12", "run_12_t5", "run_12_l7").
spawned_by("run_12", "run_12_t5", "run_12_l7").
param("run_12", "run_12_t5", "pos_distance", "0.48").
param("run_12", "run_12_t5", "delta_gripper", "1.6").
param("run_12", "run_12_t5", "efficiency", "0.7").
param("run_12", "run_12_t5", "mean_speed", "0.198").
call("run_12", "run_12_l8", "llm_generate", "completed").
depends("run_12", "run_12_l8", "run_12_t5").
call("run_12", "run_12_t6", "adjust_position", "completed").
depends("run_12", "run_12_t6", "run_12_l8").
spawned_by("run_12", "run_12_t6", "run_12_l8").
param("run_12", "run_12_t6", "pos_distance", "0.29").
param("run_12", "run_12_t6", "delta_gripper", "0.8").
param("run_12", "run_12_t6", "efficiency", "0.6").
param("run_12", "run_12_t6", "mean_speed", "0.114").
call("run_12", "run_12_l9", "llm_generate", "completed").
depends("run_12", "run_12_l9", "run_12_t6").
call("run_12", "run_12_t7", "adjust_position", "completed").
depends("run_12", "run_12_t7", "run_12_l9").
spawned_by("run_12", "run_12_t7", "run_12_l9").
param("run_12", "run_12_t7", "pos_distance", "0.49").
param("run_12", "run_12_t7", "delta_gripper", "0.0").
param("run_12", "run_12_t7", "efficiency", "0.3").
param("run_12", "run_12_t7", "mean_speed", "0.070").
call("run_12", "run_12_l10", "llm_generate", "completed").
depends("run_12", "run_12_l10", "run_12_t7").
call("run_12", "run_12_n", "notify_user", "completed").
depends("run_12", "run_12_n", "run_12_l10").
spawned_by("run_12", "run_12_n", "run_12_l10").

% === Run 13 (normal_variation) ===
job("run_13").
job_status("run_13", "completed").
call("run_13", "run_13_s0", "prompt_user", "completed").
call("run_13", "run_13_s1", "llm_generate", "completed").
depends("run_13", "run_13_s1", "run_13_s0").
call("run_13", "run_13_l2", "llm_generate", "completed").
depends("run_13", "run_13_l2", "run_13_s1").
call("run_13", "run_13_t0", "move_to_position", "completed").
depends("run_13", "run_13_t0", "run_13_l2").
spawned_by("run_13", "run_13_t0", "run_13_l2").
param("run_13", "run_13_t0", "pos_distance", "0.72").
param("run_13", "run_13_t0", "delta_gripper", "0.0").
param("run_13", "run_13_t0", "efficiency", "0.8").
param("run_13", "run_13_t0", "mean_speed", "0.175").
call("run_13", "run_13_l3", "llm_generate", "completed").
depends("run_13", "run_13_l3", "run_13_t0").
call("run_13", "run_13_t1", "close_gripper", "completed").
depends("run_13", "run_13_t1", "run_13_l3").
spawned_by("run_13", "run_13_t1", "run_13_l3").
param("run_13", "run_13_t1", "pos_distance", "0.59").
param("run_13", "run_13_t1", "delta_gripper", "-2.0").
param("run_13", "run_13_t1", "efficiency", "0.8").
param("run_13", "run_13_t1", "mean_speed", "0.179").
call("run_13", "run_13_l4", "llm_generate", "completed").
depends("run_13", "run_13_l4", "run_13_t1").
call("run_13", "run_13_t2", "close_gripper", "completed").
depends("run_13", "run_13_t2", "run_13_l4").
spawned_by("run_13", "run_13_t2", "run_13_l4").
param("run_13", "run_13_t2", "pos_distance", "0.44").
param("run_13", "run_13_t2", "delta_gripper", "-1.8").
param("run_13", "run_13_t2", "efficiency", "0.9").
param("run_13", "run_13_t2", "mean_speed", "0.126").
call("run_13", "run_13_l5", "llm_generate", "completed").
depends("run_13", "run_13_l5", "run_13_t2").
call("run_13", "run_13_t3", "move_to_position", "completed").
depends("run_13", "run_13_t3", "run_13_l5").
spawned_by("run_13", "run_13_t3", "run_13_l5").
param("run_13", "run_13_t3", "pos_distance", "1.09").
param("run_13", "run_13_t3", "delta_gripper", "0.0").
param("run_13", "run_13_t3", "efficiency", "0.8").
param("run_13", "run_13_t3", "mean_speed", "0.193").
call("run_13", "run_13_l6", "llm_generate", "completed").
depends("run_13", "run_13_l6", "run_13_t3").
call("run_13", "run_13_t4", "move_to_position", "completed").
depends("run_13", "run_13_t4", "run_13_l6").
spawned_by("run_13", "run_13_t4", "run_13_l6").
param("run_13", "run_13_t4", "pos_distance", "0.97").
param("run_13", "run_13_t4", "delta_gripper", "1.3").
param("run_13", "run_13_t4", "efficiency", "1.0").
param("run_13", "run_13_t4", "mean_speed", "0.244").
call("run_13", "run_13_l7", "llm_generate", "completed").
depends("run_13", "run_13_l7", "run_13_t4").
call("run_13", "run_13_t5", "open_gripper", "completed").
depends("run_13", "run_13_t5", "run_13_l7").
spawned_by("run_13", "run_13_t5", "run_13_l7").
param("run_13", "run_13_t5", "pos_distance", "0.47").
param("run_13", "run_13_t5", "delta_gripper", "1.6").
param("run_13", "run_13_t5", "efficiency", "0.7").
param("run_13", "run_13_t5", "mean_speed", "0.138").
call("run_13", "run_13_l8", "llm_generate", "completed").
depends("run_13", "run_13_l8", "run_13_t5").
call("run_13", "run_13_t6", "adjust_position", "completed").
depends("run_13", "run_13_t6", "run_13_l8").
spawned_by("run_13", "run_13_t6", "run_13_l8").
param("run_13", "run_13_t6", "pos_distance", "0.21").
param("run_13", "run_13_t6", "delta_gripper", "0.8").
param("run_13", "run_13_t6", "efficiency", "0.7").
param("run_13", "run_13_t6", "mean_speed", "0.165").
call("run_13", "run_13_l9", "llm_generate", "completed").
depends("run_13", "run_13_l9", "run_13_t6").
call("run_13", "run_13_t7", "adjust_position", "completed").
depends("run_13", "run_13_t7", "run_13_l9").
spawned_by("run_13", "run_13_t7", "run_13_l9").
param("run_13", "run_13_t7", "pos_distance", "0.47").
param("run_13", "run_13_t7", "delta_gripper", "0.0").
param("run_13", "run_13_t7", "efficiency", "0.3").
param("run_13", "run_13_t7", "mean_speed", "0.072").
call("run_13", "run_13_l10", "llm_generate", "completed").
depends("run_13", "run_13_l10", "run_13_t7").
call("run_13", "run_13_n", "notify_user", "completed").
depends("run_13", "run_13_n", "run_13_l10").
spawned_by("run_13", "run_13_n", "run_13_l10").

% === Run 14 (normal_variation) ===
job("run_14").
job_status("run_14", "completed").
call("run_14", "run_14_s0", "prompt_user", "completed").
call("run_14", "run_14_s1", "llm_generate", "completed").
depends("run_14", "run_14_s1", "run_14_s0").
call("run_14", "run_14_l2", "llm_generate", "completed").
depends("run_14", "run_14_l2", "run_14_s1").
call("run_14", "run_14_t0", "move_to_position", "completed").
depends("run_14", "run_14_t0", "run_14_l2").
spawned_by("run_14", "run_14_t0", "run_14_l2").
param("run_14", "run_14_t0", "pos_distance", "0.59").
param("run_14", "run_14_t0", "delta_gripper", "0.0").
param("run_14", "run_14_t0", "efficiency", "1.0").
param("run_14", "run_14_t0", "mean_speed", "0.154").
call("run_14", "run_14_l3", "llm_generate", "completed").
depends("run_14", "run_14_l3", "run_14_t0").
call("run_14", "run_14_t1", "close_gripper", "completed").
depends("run_14", "run_14_t1", "run_14_l3").
spawned_by("run_14", "run_14_t1", "run_14_l3").
param("run_14", "run_14_t1", "pos_distance", "0.64").
param("run_14", "run_14_t1", "delta_gripper", "-2.0").
param("run_14", "run_14_t1", "efficiency", "0.9").
param("run_14", "run_14_t1", "mean_speed", "0.156").
call("run_14", "run_14_l4", "llm_generate", "completed").
depends("run_14", "run_14_l4", "run_14_t1").
call("run_14", "run_14_t2", "close_gripper", "completed").
depends("run_14", "run_14_t2", "run_14_l4").
spawned_by("run_14", "run_14_t2", "run_14_l4").
param("run_14", "run_14_t2", "pos_distance", "0.23").
param("run_14", "run_14_t2", "delta_gripper", "-1.8").
param("run_14", "run_14_t2", "efficiency", "1.0").
param("run_14", "run_14_t2", "mean_speed", "0.154").
call("run_14", "run_14_l5", "llm_generate", "completed").
depends("run_14", "run_14_l5", "run_14_t2").
call("run_14", "run_14_t3", "move_to_position", "completed").
depends("run_14", "run_14_t3", "run_14_l5").
spawned_by("run_14", "run_14_t3", "run_14_l5").
param("run_14", "run_14_t3", "pos_distance", "3.14").
param("run_14", "run_14_t3", "delta_gripper", "0.0").
param("run_14", "run_14_t3", "efficiency", "0.8").
param("run_14", "run_14_t3", "mean_speed", "0.308").
call("run_14", "run_14_l6", "llm_generate", "completed").
depends("run_14", "run_14_l6", "run_14_t3").
call("run_14", "run_14_t4", "move_to_position", "completed").
depends("run_14", "run_14_t4", "run_14_l6").
spawned_by("run_14", "run_14_t4", "run_14_l6").
param("run_14", "run_14_t4", "pos_distance", "0.36").
param("run_14", "run_14_t4", "delta_gripper", "1.3").
param("run_14", "run_14_t4", "efficiency", "0.9").
param("run_14", "run_14_t4", "mean_speed", "0.271").
call("run_14", "run_14_l7", "llm_generate", "completed").
depends("run_14", "run_14_l7", "run_14_t4").
call("run_14", "run_14_t5", "open_gripper", "completed").
depends("run_14", "run_14_t5", "run_14_l7").
spawned_by("run_14", "run_14_t5", "run_14_l7").
param("run_14", "run_14_t5", "pos_distance", "0.62").
param("run_14", "run_14_t5", "delta_gripper", "1.6").
param("run_14", "run_14_t5", "efficiency", "0.9").
param("run_14", "run_14_t5", "mean_speed", "0.207").
call("run_14", "run_14_l8", "llm_generate", "completed").
depends("run_14", "run_14_l8", "run_14_t5").
call("run_14", "run_14_t6", "adjust_position", "completed").
depends("run_14", "run_14_t6", "run_14_l8").
spawned_by("run_14", "run_14_t6", "run_14_l8").
param("run_14", "run_14_t6", "pos_distance", "0.33").
param("run_14", "run_14_t6", "delta_gripper", "0.8").
param("run_14", "run_14_t6", "efficiency", "0.7").
param("run_14", "run_14_t6", "mean_speed", "0.114").
call("run_14", "run_14_l9", "llm_generate", "completed").
depends("run_14", "run_14_l9", "run_14_t6").
call("run_14", "run_14_t7", "adjust_position", "completed").
depends("run_14", "run_14_t7", "run_14_l9").
spawned_by("run_14", "run_14_t7", "run_14_l9").
param("run_14", "run_14_t7", "pos_distance", "0.21").
param("run_14", "run_14_t7", "delta_gripper", "0.0").
param("run_14", "run_14_t7", "efficiency", "0.3").
param("run_14", "run_14_t7", "mean_speed", "0.071").
call("run_14", "run_14_l10", "llm_generate", "completed").
depends("run_14", "run_14_l10", "run_14_t7").
call("run_14", "run_14_n", "notify_user", "completed").
depends("run_14", "run_14_n", "run_14_l10").
spawned_by("run_14", "run_14_n", "run_14_l10").

% === Run 15 (normal_variation) ===
job("run_15").
job_status("run_15", "completed").
call("run_15", "run_15_s0", "prompt_user", "completed").
call("run_15", "run_15_s1", "llm_generate", "completed").
depends("run_15", "run_15_s1", "run_15_s0").
call("run_15", "run_15_l2", "llm_generate", "completed").
depends("run_15", "run_15_l2", "run_15_s1").
call("run_15", "run_15_t0", "move_to_position", "completed").
depends("run_15", "run_15_t0", "run_15_l2").
spawned_by("run_15", "run_15_t0", "run_15_l2").
param("run_15", "run_15_t0", "pos_distance", "0.57").
param("run_15", "run_15_t0", "delta_gripper", "0.0").
param("run_15", "run_15_t0", "efficiency", "0.9").
param("run_15", "run_15_t0", "mean_speed", "0.241").
call("run_15", "run_15_l3", "llm_generate", "completed").
depends("run_15", "run_15_l3", "run_15_t0").
call("run_15", "run_15_t1", "close_gripper", "completed").
depends("run_15", "run_15_t1", "run_15_l3").
spawned_by("run_15", "run_15_t1", "run_15_l3").
param("run_15", "run_15_t1", "pos_distance", "0.37").
param("run_15", "run_15_t1", "delta_gripper", "-2.0").
param("run_15", "run_15_t1", "efficiency", "0.8").
param("run_15", "run_15_t1", "mean_speed", "0.194").
call("run_15", "run_15_l4", "llm_generate", "completed").
depends("run_15", "run_15_l4", "run_15_t1").
call("run_15", "run_15_t2", "close_gripper", "completed").
depends("run_15", "run_15_t2", "run_15_l4").
spawned_by("run_15", "run_15_t2", "run_15_l4").
param("run_15", "run_15_t2", "pos_distance", "0.19").
param("run_15", "run_15_t2", "delta_gripper", "-1.8").
param("run_15", "run_15_t2", "efficiency", "0.9").
param("run_15", "run_15_t2", "mean_speed", "0.100").
call("run_15", "run_15_l5", "llm_generate", "completed").
depends("run_15", "run_15_l5", "run_15_t2").
call("run_15", "run_15_t3", "move_to_position", "completed").
depends("run_15", "run_15_t3", "run_15_l5").
spawned_by("run_15", "run_15_t3", "run_15_l5").
param("run_15", "run_15_t3", "pos_distance", "2.90").
param("run_15", "run_15_t3", "delta_gripper", "0.0").
param("run_15", "run_15_t3", "efficiency", "0.7").
param("run_15", "run_15_t3", "mean_speed", "0.332").
call("run_15", "run_15_l6", "llm_generate", "completed").
depends("run_15", "run_15_l6", "run_15_t3").
call("run_15", "run_15_t4", "move_to_position", "completed").
depends("run_15", "run_15_t4", "run_15_l6").
spawned_by("run_15", "run_15_t4", "run_15_l6").
param("run_15", "run_15_t4", "pos_distance", "0.79").
param("run_15", "run_15_t4", "delta_gripper", "1.3").
param("run_15", "run_15_t4", "efficiency", "1.0").
param("run_15", "run_15_t4", "mean_speed", "0.223").
call("run_15", "run_15_l7", "llm_generate", "completed").
depends("run_15", "run_15_l7", "run_15_t4").
call("run_15", "run_15_t5", "open_gripper", "completed").
depends("run_15", "run_15_t5", "run_15_l7").
spawned_by("run_15", "run_15_t5", "run_15_l7").
param("run_15", "run_15_t5", "pos_distance", "0.45").
param("run_15", "run_15_t5", "delta_gripper", "1.6").
param("run_15", "run_15_t5", "efficiency", "0.7").
param("run_15", "run_15_t5", "mean_speed", "0.222").
call("run_15", "run_15_l8", "llm_generate", "completed").
depends("run_15", "run_15_l8", "run_15_t5").
call("run_15", "run_15_t6", "adjust_position", "completed").
depends("run_15", "run_15_t6", "run_15_l8").
spawned_by("run_15", "run_15_t6", "run_15_l8").
param("run_15", "run_15_t6", "pos_distance", "0.37").
param("run_15", "run_15_t6", "delta_gripper", "0.8").
param("run_15", "run_15_t6", "efficiency", "0.7").
param("run_15", "run_15_t6", "mean_speed", "0.107").
call("run_15", "run_15_l9", "llm_generate", "completed").
depends("run_15", "run_15_l9", "run_15_t6").
call("run_15", "run_15_t7", "adjust_position", "completed").
depends("run_15", "run_15_t7", "run_15_l9").
spawned_by("run_15", "run_15_t7", "run_15_l9").
param("run_15", "run_15_t7", "pos_distance", "0.27").
param("run_15", "run_15_t7", "delta_gripper", "0.0").
param("run_15", "run_15_t7", "efficiency", "0.3").
param("run_15", "run_15_t7", "mean_speed", "0.045").
call("run_15", "run_15_l10", "llm_generate", "completed").
depends("run_15", "run_15_l10", "run_15_t7").
call("run_15", "run_15_n", "notify_user", "completed").
depends("run_15", "run_15_n", "run_15_l10").
spawned_by("run_15", "run_15_n", "run_15_l10").

% === Run 16 (normal_variation) ===
job("run_16").
job_status("run_16", "completed").
call("run_16", "run_16_s0", "prompt_user", "completed").
call("run_16", "run_16_s1", "llm_generate", "completed").
depends("run_16", "run_16_s1", "run_16_s0").
call("run_16", "run_16_l2", "llm_generate", "completed").
depends("run_16", "run_16_l2", "run_16_s1").
call("run_16", "run_16_t0", "move_to_position", "completed").
depends("run_16", "run_16_t0", "run_16_l2").
spawned_by("run_16", "run_16_t0", "run_16_l2").
param("run_16", "run_16_t0", "pos_distance", "0.96").
param("run_16", "run_16_t0", "delta_gripper", "0.0").
param("run_16", "run_16_t0", "efficiency", "1.0").
param("run_16", "run_16_t0", "mean_speed", "0.180").
call("run_16", "run_16_l3", "llm_generate", "completed").
depends("run_16", "run_16_l3", "run_16_t0").
call("run_16", "run_16_t1", "close_gripper", "completed").
depends("run_16", "run_16_t1", "run_16_l3").
spawned_by("run_16", "run_16_t1", "run_16_l3").
param("run_16", "run_16_t1", "pos_distance", "0.47").
param("run_16", "run_16_t1", "delta_gripper", "-2.0").
param("run_16", "run_16_t1", "efficiency", "0.9").
param("run_16", "run_16_t1", "mean_speed", "0.154").
call("run_16", "run_16_l4", "llm_generate", "completed").
depends("run_16", "run_16_l4", "run_16_t1").
call("run_16", "run_16_t2", "close_gripper", "completed").
depends("run_16", "run_16_t2", "run_16_l4").
spawned_by("run_16", "run_16_t2", "run_16_l4").
param("run_16", "run_16_t2", "pos_distance", "0.27").
param("run_16", "run_16_t2", "delta_gripper", "-1.8").
param("run_16", "run_16_t2", "efficiency", "0.8").
param("run_16", "run_16_t2", "mean_speed", "0.138").
call("run_16", "run_16_l5", "llm_generate", "completed").
depends("run_16", "run_16_l5", "run_16_t2").
call("run_16", "run_16_t3", "move_to_position", "completed").
depends("run_16", "run_16_t3", "run_16_l5").
spawned_by("run_16", "run_16_t3", "run_16_l5").
param("run_16", "run_16_t3", "pos_distance", "3.04").
param("run_16", "run_16_t3", "delta_gripper", "0.0").
param("run_16", "run_16_t3", "efficiency", "0.8").
param("run_16", "run_16_t3", "mean_speed", "0.265").
call("run_16", "run_16_l6", "llm_generate", "completed").
depends("run_16", "run_16_l6", "run_16_t3").
call("run_16", "run_16_t4", "move_to_position", "completed").
depends("run_16", "run_16_t4", "run_16_l6").
spawned_by("run_16", "run_16_t4", "run_16_l6").
param("run_16", "run_16_t4", "pos_distance", "0.86").
param("run_16", "run_16_t4", "delta_gripper", "1.3").
param("run_16", "run_16_t4", "efficiency", "0.9").
param("run_16", "run_16_t4", "mean_speed", "0.262").
call("run_16", "run_16_l7", "llm_generate", "completed").
depends("run_16", "run_16_l7", "run_16_t4").
call("run_16", "run_16_t5", "open_gripper", "completed").
depends("run_16", "run_16_t5", "run_16_l7").
spawned_by("run_16", "run_16_t5", "run_16_l7").
param("run_16", "run_16_t5", "pos_distance", "0.25").
param("run_16", "run_16_t5", "delta_gripper", "1.6").
param("run_16", "run_16_t5", "efficiency", "0.7").
param("run_16", "run_16_t5", "mean_speed", "0.193").
call("run_16", "run_16_l8", "llm_generate", "completed").
depends("run_16", "run_16_l8", "run_16_t5").
call("run_16", "run_16_t6", "adjust_position", "completed").
depends("run_16", "run_16_t6", "run_16_l8").
spawned_by("run_16", "run_16_t6", "run_16_l8").
param("run_16", "run_16_t6", "pos_distance", "0.21").
param("run_16", "run_16_t6", "delta_gripper", "0.8").
param("run_16", "run_16_t6", "efficiency", "0.6").
param("run_16", "run_16_t6", "mean_speed", "0.161").
call("run_16", "run_16_l9", "llm_generate", "completed").
depends("run_16", "run_16_l9", "run_16_t6").
call("run_16", "run_16_t7", "adjust_position", "completed").
depends("run_16", "run_16_t7", "run_16_l9").
spawned_by("run_16", "run_16_t7", "run_16_l9").
param("run_16", "run_16_t7", "pos_distance", "0.28").
param("run_16", "run_16_t7", "delta_gripper", "0.0").
param("run_16", "run_16_t7", "efficiency", "0.3").
param("run_16", "run_16_t7", "mean_speed", "0.052").
call("run_16", "run_16_l10", "llm_generate", "completed").
depends("run_16", "run_16_l10", "run_16_t7").
call("run_16", "run_16_n", "notify_user", "completed").
depends("run_16", "run_16_n", "run_16_l10").
spawned_by("run_16", "run_16_n", "run_16_l10").

% === Run 17 (normal_variation) ===
job("run_17").
job_status("run_17", "completed").
call("run_17", "run_17_s0", "prompt_user", "completed").
call("run_17", "run_17_s1", "llm_generate", "completed").
depends("run_17", "run_17_s1", "run_17_s0").
call("run_17", "run_17_l2", "llm_generate", "completed").
depends("run_17", "run_17_l2", "run_17_s1").
call("run_17", "run_17_t0", "move_to_position", "completed").
depends("run_17", "run_17_t0", "run_17_l2").
spawned_by("run_17", "run_17_t0", "run_17_l2").
param("run_17", "run_17_t0", "pos_distance", "0.71").
param("run_17", "run_17_t0", "delta_gripper", "0.0").
param("run_17", "run_17_t0", "efficiency", "0.8").
param("run_17", "run_17_t0", "mean_speed", "0.177").
call("run_17", "run_17_l3", "llm_generate", "completed").
depends("run_17", "run_17_l3", "run_17_t0").
call("run_17", "run_17_t1", "close_gripper", "completed").
depends("run_17", "run_17_t1", "run_17_l3").
spawned_by("run_17", "run_17_t1", "run_17_l3").
param("run_17", "run_17_t1", "pos_distance", "0.43").
param("run_17", "run_17_t1", "delta_gripper", "-2.0").
param("run_17", "run_17_t1", "efficiency", "0.9").
param("run_17", "run_17_t1", "mean_speed", "0.209").
call("run_17", "run_17_l4", "llm_generate", "completed").
depends("run_17", "run_17_l4", "run_17_t1").
call("run_17", "run_17_t2", "close_gripper", "completed").
depends("run_17", "run_17_t2", "run_17_l4").
spawned_by("run_17", "run_17_t2", "run_17_l4").
param("run_17", "run_17_t2", "pos_distance", "0.50").
param("run_17", "run_17_t2", "delta_gripper", "-1.8").
param("run_17", "run_17_t2", "efficiency", "0.9").
param("run_17", "run_17_t2", "mean_speed", "0.131").
call("run_17", "run_17_l5", "llm_generate", "completed").
depends("run_17", "run_17_l5", "run_17_t2").
call("run_17", "run_17_t3", "move_to_position", "completed").
depends("run_17", "run_17_t3", "run_17_l5").
spawned_by("run_17", "run_17_t3", "run_17_l5").
param("run_17", "run_17_t3", "pos_distance", "2.23").
param("run_17", "run_17_t3", "delta_gripper", "0.0").
param("run_17", "run_17_t3", "efficiency", "0.7").
param("run_17", "run_17_t3", "mean_speed", "0.182").
call("run_17", "run_17_l6", "llm_generate", "completed").
depends("run_17", "run_17_l6", "run_17_t3").
call("run_17", "run_17_t4", "move_to_position", "completed").
depends("run_17", "run_17_t4", "run_17_l6").
spawned_by("run_17", "run_17_t4", "run_17_l6").
param("run_17", "run_17_t4", "pos_distance", "0.64").
param("run_17", "run_17_t4", "delta_gripper", "1.3").
param("run_17", "run_17_t4", "efficiency", "1.0").
param("run_17", "run_17_t4", "mean_speed", "0.249").
call("run_17", "run_17_l7", "llm_generate", "completed").
depends("run_17", "run_17_l7", "run_17_t4").
call("run_17", "run_17_t5", "open_gripper", "completed").
depends("run_17", "run_17_t5", "run_17_l7").
spawned_by("run_17", "run_17_t5", "run_17_l7").
param("run_17", "run_17_t5", "pos_distance", "0.42").
param("run_17", "run_17_t5", "delta_gripper", "1.6").
param("run_17", "run_17_t5", "efficiency", "0.7").
param("run_17", "run_17_t5", "mean_speed", "0.169").
call("run_17", "run_17_l8", "llm_generate", "completed").
depends("run_17", "run_17_l8", "run_17_t5").
call("run_17", "run_17_t6", "adjust_position", "completed").
depends("run_17", "run_17_t6", "run_17_l8").
spawned_by("run_17", "run_17_t6", "run_17_l8").
param("run_17", "run_17_t6", "pos_distance", "0.25").
param("run_17", "run_17_t6", "delta_gripper", "0.8").
param("run_17", "run_17_t6", "efficiency", "0.7").
param("run_17", "run_17_t6", "mean_speed", "0.165").
call("run_17", "run_17_l9", "llm_generate", "completed").
depends("run_17", "run_17_l9", "run_17_t6").
call("run_17", "run_17_t7", "adjust_position", "completed").
depends("run_17", "run_17_t7", "run_17_l9").
spawned_by("run_17", "run_17_t7", "run_17_l9").
param("run_17", "run_17_t7", "pos_distance", "0.23").
param("run_17", "run_17_t7", "delta_gripper", "0.0").
param("run_17", "run_17_t7", "efficiency", "0.3").
param("run_17", "run_17_t7", "mean_speed", "0.044").
call("run_17", "run_17_l10", "llm_generate", "completed").
depends("run_17", "run_17_l10", "run_17_t7").
call("run_17", "run_17_n", "notify_user", "completed").
depends("run_17", "run_17_n", "run_17_l10").
spawned_by("run_17", "run_17_n", "run_17_l10").

% === Run 18 (normal_variation) ===
job("run_18").
job_status("run_18", "completed").
call("run_18", "run_18_s0", "prompt_user", "completed").
call("run_18", "run_18_s1", "llm_generate", "completed").
depends("run_18", "run_18_s1", "run_18_s0").
call("run_18", "run_18_l2", "llm_generate", "completed").
depends("run_18", "run_18_l2", "run_18_s1").
call("run_18", "run_18_t0", "move_to_position", "completed").
depends("run_18", "run_18_t0", "run_18_l2").
spawned_by("run_18", "run_18_t0", "run_18_l2").
param("run_18", "run_18_t0", "pos_distance", "0.89").
param("run_18", "run_18_t0", "delta_gripper", "0.0").
param("run_18", "run_18_t0", "efficiency", "1.0").
param("run_18", "run_18_t0", "mean_speed", "0.182").
call("run_18", "run_18_l3", "llm_generate", "completed").
depends("run_18", "run_18_l3", "run_18_t0").
call("run_18", "run_18_t1", "close_gripper", "completed").
depends("run_18", "run_18_t1", "run_18_l3").
spawned_by("run_18", "run_18_t1", "run_18_l3").
param("run_18", "run_18_t1", "pos_distance", "0.58").
param("run_18", "run_18_t1", "delta_gripper", "-2.0").
param("run_18", "run_18_t1", "efficiency", "0.8").
param("run_18", "run_18_t1", "mean_speed", "0.182").
call("run_18", "run_18_l4", "llm_generate", "completed").
depends("run_18", "run_18_l4", "run_18_t1").
call("run_18", "run_18_t2", "close_gripper", "completed").
depends("run_18", "run_18_t2", "run_18_l4").
spawned_by("run_18", "run_18_t2", "run_18_l4").
param("run_18", "run_18_t2", "pos_distance", "0.25").
param("run_18", "run_18_t2", "delta_gripper", "-1.8").
param("run_18", "run_18_t2", "efficiency", "0.8").
param("run_18", "run_18_t2", "mean_speed", "0.088").
call("run_18", "run_18_l5", "llm_generate", "completed").
depends("run_18", "run_18_l5", "run_18_t2").
call("run_18", "run_18_t3", "move_to_position", "completed").
depends("run_18", "run_18_t3", "run_18_l5").
spawned_by("run_18", "run_18_t3", "run_18_l5").
param("run_18", "run_18_t3", "pos_distance", "2.08").
param("run_18", "run_18_t3", "delta_gripper", "0.0").
param("run_18", "run_18_t3", "efficiency", "0.6").
param("run_18", "run_18_t3", "mean_speed", "0.314").
call("run_18", "run_18_l6", "llm_generate", "completed").
depends("run_18", "run_18_l6", "run_18_t3").
call("run_18", "run_18_t4", "move_to_position", "completed").
depends("run_18", "run_18_t4", "run_18_l6").
spawned_by("run_18", "run_18_t4", "run_18_l6").
param("run_18", "run_18_t4", "pos_distance", "0.63").
param("run_18", "run_18_t4", "delta_gripper", "1.3").
param("run_18", "run_18_t4", "efficiency", "0.8").
param("run_18", "run_18_t4", "mean_speed", "0.256").
call("run_18", "run_18_l7", "llm_generate", "completed").
depends("run_18", "run_18_l7", "run_18_t4").
call("run_18", "run_18_t5", "open_gripper", "completed").
depends("run_18", "run_18_t5", "run_18_l7").
spawned_by("run_18", "run_18_t5", "run_18_l7").
param("run_18", "run_18_t5", "pos_distance", "0.52").
param("run_18", "run_18_t5", "delta_gripper", "1.6").
param("run_18", "run_18_t5", "efficiency", "0.7").
param("run_18", "run_18_t5", "mean_speed", "0.175").
call("run_18", "run_18_l8", "llm_generate", "completed").
depends("run_18", "run_18_l8", "run_18_t5").
call("run_18", "run_18_t6", "adjust_position", "completed").
depends("run_18", "run_18_t6", "run_18_l8").
spawned_by("run_18", "run_18_t6", "run_18_l8").
param("run_18", "run_18_t6", "pos_distance", "0.29").
param("run_18", "run_18_t6", "delta_gripper", "0.8").
param("run_18", "run_18_t6", "efficiency", "0.7").
param("run_18", "run_18_t6", "mean_speed", "0.094").
call("run_18", "run_18_l9", "llm_generate", "completed").
depends("run_18", "run_18_l9", "run_18_t6").
call("run_18", "run_18_t7", "adjust_position", "completed").
depends("run_18", "run_18_t7", "run_18_l9").
spawned_by("run_18", "run_18_t7", "run_18_l9").
param("run_18", "run_18_t7", "pos_distance", "0.44").
param("run_18", "run_18_t7", "delta_gripper", "0.0").
param("run_18", "run_18_t7", "efficiency", "0.3").
param("run_18", "run_18_t7", "mean_speed", "0.045").
call("run_18", "run_18_l10", "llm_generate", "completed").
depends("run_18", "run_18_l10", "run_18_t7").
call("run_18", "run_18_n", "notify_user", "completed").
depends("run_18", "run_18_n", "run_18_l10").
spawned_by("run_18", "run_18_n", "run_18_l10").

% === Run 19 (normal_variation) ===
job("run_19").
job_status("run_19", "completed").
call("run_19", "run_19_s0", "prompt_user", "completed").
call("run_19", "run_19_s1", "llm_generate", "completed").
depends("run_19", "run_19_s1", "run_19_s0").
call("run_19", "run_19_l2", "llm_generate", "completed").
depends("run_19", "run_19_l2", "run_19_s1").
call("run_19", "run_19_t0", "move_to_position", "completed").
depends("run_19", "run_19_t0", "run_19_l2").
spawned_by("run_19", "run_19_t0", "run_19_l2").
param("run_19", "run_19_t0", "pos_distance", "0.53").
param("run_19", "run_19_t0", "delta_gripper", "0.0").
param("run_19", "run_19_t0", "efficiency", "0.9").
param("run_19", "run_19_t0", "mean_speed", "0.258").
call("run_19", "run_19_l3", "llm_generate", "completed").
depends("run_19", "run_19_l3", "run_19_t0").
call("run_19", "run_19_t1", "close_gripper", "completed").
depends("run_19", "run_19_t1", "run_19_l3").
spawned_by("run_19", "run_19_t1", "run_19_l3").
param("run_19", "run_19_t1", "pos_distance", "0.25").
param("run_19", "run_19_t1", "delta_gripper", "-2.0").
param("run_19", "run_19_t1", "efficiency", "1.0").
param("run_19", "run_19_t1", "mean_speed", "0.140").
call("run_19", "run_19_l4", "llm_generate", "completed").
depends("run_19", "run_19_l4", "run_19_t1").
call("run_19", "run_19_t2", "close_gripper", "completed").
depends("run_19", "run_19_t2", "run_19_l4").
spawned_by("run_19", "run_19_t2", "run_19_l4").
param("run_19", "run_19_t2", "pos_distance", "0.34").
param("run_19", "run_19_t2", "delta_gripper", "-1.8").
param("run_19", "run_19_t2", "efficiency", "1.0").
param("run_19", "run_19_t2", "mean_speed", "0.145").
call("run_19", "run_19_l5", "llm_generate", "completed").
depends("run_19", "run_19_l5", "run_19_t2").
call("run_19", "run_19_t3", "move_to_position", "completed").
depends("run_19", "run_19_t3", "run_19_l5").
spawned_by("run_19", "run_19_t3", "run_19_l5").
param("run_19", "run_19_t3", "pos_distance", "3.17").
param("run_19", "run_19_t3", "delta_gripper", "0.0").
param("run_19", "run_19_t3", "efficiency", "0.8").
param("run_19", "run_19_t3", "mean_speed", "0.274").
call("run_19", "run_19_l6", "llm_generate", "completed").
depends("run_19", "run_19_l6", "run_19_t3").
call("run_19", "run_19_t4", "move_to_position", "completed").
depends("run_19", "run_19_t4", "run_19_l6").
spawned_by("run_19", "run_19_t4", "run_19_l6").
param("run_19", "run_19_t4", "pos_distance", "0.96").
param("run_19", "run_19_t4", "delta_gripper", "1.3").
param("run_19", "run_19_t4", "efficiency", "1.0").
param("run_19", "run_19_t4", "mean_speed", "0.254").
call("run_19", "run_19_l7", "llm_generate", "completed").
depends("run_19", "run_19_l7", "run_19_t4").
call("run_19", "run_19_t5", "open_gripper", "completed").
depends("run_19", "run_19_t5", "run_19_l7").
spawned_by("run_19", "run_19_t5", "run_19_l7").
param("run_19", "run_19_t5", "pos_distance", "0.44").
param("run_19", "run_19_t5", "delta_gripper", "1.6").
param("run_19", "run_19_t5", "efficiency", "0.8").
param("run_19", "run_19_t5", "mean_speed", "0.210").
call("run_19", "run_19_l8", "llm_generate", "completed").
depends("run_19", "run_19_l8", "run_19_t5").
call("run_19", "run_19_t6", "adjust_position", "completed").
depends("run_19", "run_19_t6", "run_19_l8").
spawned_by("run_19", "run_19_t6", "run_19_l8").
param("run_19", "run_19_t6", "pos_distance", "0.35").
param("run_19", "run_19_t6", "delta_gripper", "0.8").
param("run_19", "run_19_t6", "efficiency", "0.6").
param("run_19", "run_19_t6", "mean_speed", "0.152").
call("run_19", "run_19_l9", "llm_generate", "completed").
depends("run_19", "run_19_l9", "run_19_t6").
call("run_19", "run_19_t7", "adjust_position", "completed").
depends("run_19", "run_19_t7", "run_19_l9").
spawned_by("run_19", "run_19_t7", "run_19_l9").
param("run_19", "run_19_t7", "pos_distance", "0.26").
param("run_19", "run_19_t7", "delta_gripper", "0.0").
param("run_19", "run_19_t7", "efficiency", "0.3").
param("run_19", "run_19_t7", "mean_speed", "0.049").
call("run_19", "run_19_l10", "llm_generate", "completed").
depends("run_19", "run_19_l10", "run_19_t7").
call("run_19", "run_19_n", "notify_user", "completed").
depends("run_19", "run_19_n", "run_19_l10").
spawned_by("run_19", "run_19_n", "run_19_l10").

% === Run 20 (normal_variation) ===
job("run_20").
job_status("run_20", "completed").
call("run_20", "run_20_s0", "prompt_user", "completed").
call("run_20", "run_20_s1", "llm_generate", "completed").
depends("run_20", "run_20_s1", "run_20_s0").
call("run_20", "run_20_l2", "llm_generate", "completed").
depends("run_20", "run_20_l2", "run_20_s1").
call("run_20", "run_20_t0", "move_to_position", "completed").
depends("run_20", "run_20_t0", "run_20_l2").
spawned_by("run_20", "run_20_t0", "run_20_l2").
param("run_20", "run_20_t0", "pos_distance", "0.99").
param("run_20", "run_20_t0", "delta_gripper", "0.0").
param("run_20", "run_20_t0", "efficiency", "1.0").
param("run_20", "run_20_t0", "mean_speed", "0.205").
call("run_20", "run_20_l3", "llm_generate", "completed").
depends("run_20", "run_20_l3", "run_20_t0").
call("run_20", "run_20_t1", "close_gripper", "completed").
depends("run_20", "run_20_t1", "run_20_l3").
spawned_by("run_20", "run_20_t1", "run_20_l3").
param("run_20", "run_20_t1", "pos_distance", "0.36").
param("run_20", "run_20_t1", "delta_gripper", "-2.0").
param("run_20", "run_20_t1", "efficiency", "0.8").
param("run_20", "run_20_t1", "mean_speed", "0.123").
call("run_20", "run_20_l4", "llm_generate", "completed").
depends("run_20", "run_20_l4", "run_20_t1").
call("run_20", "run_20_t2", "close_gripper", "completed").
depends("run_20", "run_20_t2", "run_20_l4").
spawned_by("run_20", "run_20_t2", "run_20_l4").
param("run_20", "run_20_t2", "pos_distance", "0.28").
param("run_20", "run_20_t2", "delta_gripper", "-1.8").
param("run_20", "run_20_t2", "efficiency", "0.9").
param("run_20", "run_20_t2", "mean_speed", "0.109").
call("run_20", "run_20_l5", "llm_generate", "completed").
depends("run_20", "run_20_l5", "run_20_t2").
call("run_20", "run_20_t3", "move_to_position", "completed").
depends("run_20", "run_20_t3", "run_20_l5").
spawned_by("run_20", "run_20_t3", "run_20_l5").
param("run_20", "run_20_t3", "pos_distance", "1.36").
param("run_20", "run_20_t3", "delta_gripper", "0.0").
param("run_20", "run_20_t3", "efficiency", "0.8").
param("run_20", "run_20_t3", "mean_speed", "0.218").
call("run_20", "run_20_l6", "llm_generate", "completed").
depends("run_20", "run_20_l6", "run_20_t3").
call("run_20", "run_20_t4", "move_to_position", "completed").
depends("run_20", "run_20_t4", "run_20_l6").
spawned_by("run_20", "run_20_t4", "run_20_l6").
param("run_20", "run_20_t4", "pos_distance", "0.83").
param("run_20", "run_20_t4", "delta_gripper", "1.3").
param("run_20", "run_20_t4", "efficiency", "0.9").
param("run_20", "run_20_t4", "mean_speed", "0.176").
call("run_20", "run_20_l7", "llm_generate", "completed").
depends("run_20", "run_20_l7", "run_20_t4").
call("run_20", "run_20_t5", "open_gripper", "completed").
depends("run_20", "run_20_t5", "run_20_l7").
spawned_by("run_20", "run_20_t5", "run_20_l7").
param("run_20", "run_20_t5", "pos_distance", "0.45").
param("run_20", "run_20_t5", "delta_gripper", "1.6").
param("run_20", "run_20_t5", "efficiency", "0.7").
param("run_20", "run_20_t5", "mean_speed", "0.167").
call("run_20", "run_20_l8", "llm_generate", "completed").
depends("run_20", "run_20_l8", "run_20_t5").
call("run_20", "run_20_t6", "adjust_position", "completed").
depends("run_20", "run_20_t6", "run_20_l8").
spawned_by("run_20", "run_20_t6", "run_20_l8").
param("run_20", "run_20_t6", "pos_distance", "0.23").
param("run_20", "run_20_t6", "delta_gripper", "0.8").
param("run_20", "run_20_t6", "efficiency", "0.6").
param("run_20", "run_20_t6", "mean_speed", "0.165").
call("run_20", "run_20_l9", "llm_generate", "completed").
depends("run_20", "run_20_l9", "run_20_t6").
call("run_20", "run_20_t7", "adjust_position", "completed").
depends("run_20", "run_20_t7", "run_20_l9").
spawned_by("run_20", "run_20_t7", "run_20_l9").
param("run_20", "run_20_t7", "pos_distance", "0.41").
param("run_20", "run_20_t7", "delta_gripper", "0.0").
param("run_20", "run_20_t7", "efficiency", "0.3").
param("run_20", "run_20_t7", "mean_speed", "0.071").
call("run_20", "run_20_l10", "llm_generate", "completed").
depends("run_20", "run_20_l10", "run_20_t7").
call("run_20", "run_20_n", "notify_user", "completed").
depends("run_20", "run_20_n", "run_20_l10").
spawned_by("run_20", "run_20_n", "run_20_l10").

% === Run 21 (normal_variation) ===
job("run_21").
job_status("run_21", "completed").
call("run_21", "run_21_s0", "prompt_user", "completed").
call("run_21", "run_21_s1", "llm_generate", "completed").
depends("run_21", "run_21_s1", "run_21_s0").
call("run_21", "run_21_l2", "llm_generate", "completed").
depends("run_21", "run_21_l2", "run_21_s1").
call("run_21", "run_21_t0", "move_to_position", "completed").
depends("run_21", "run_21_t0", "run_21_l2").
spawned_by("run_21", "run_21_t0", "run_21_l2").
param("run_21", "run_21_t0", "pos_distance", "0.69").
param("run_21", "run_21_t0", "delta_gripper", "0.0").
param("run_21", "run_21_t0", "efficiency", "0.9").
param("run_21", "run_21_t0", "mean_speed", "0.142").
call("run_21", "run_21_l3", "llm_generate", "completed").
depends("run_21", "run_21_l3", "run_21_t0").
call("run_21", "run_21_t1", "close_gripper", "completed").
depends("run_21", "run_21_t1", "run_21_l3").
spawned_by("run_21", "run_21_t1", "run_21_l3").
param("run_21", "run_21_t1", "pos_distance", "0.58").
param("run_21", "run_21_t1", "delta_gripper", "-2.0").
param("run_21", "run_21_t1", "efficiency", "1.0").
param("run_21", "run_21_t1", "mean_speed", "0.200").
call("run_21", "run_21_l4", "llm_generate", "completed").
depends("run_21", "run_21_l4", "run_21_t1").
call("run_21", "run_21_t2", "close_gripper", "completed").
depends("run_21", "run_21_t2", "run_21_l4").
spawned_by("run_21", "run_21_t2", "run_21_l4").
param("run_21", "run_21_t2", "pos_distance", "0.33").
param("run_21", "run_21_t2", "delta_gripper", "-1.8").
param("run_21", "run_21_t2", "efficiency", "0.9").
param("run_21", "run_21_t2", "mean_speed", "0.141").
call("run_21", "run_21_l5", "llm_generate", "completed").
depends("run_21", "run_21_l5", "run_21_t2").
call("run_21", "run_21_t3", "move_to_position", "completed").
depends("run_21", "run_21_t3", "run_21_l5").
spawned_by("run_21", "run_21_t3", "run_21_l5").
param("run_21", "run_21_t3", "pos_distance", "2.35").
param("run_21", "run_21_t3", "delta_gripper", "0.0").
param("run_21", "run_21_t3", "efficiency", "0.6").
param("run_21", "run_21_t3", "mean_speed", "0.216").
call("run_21", "run_21_l6", "llm_generate", "completed").
depends("run_21", "run_21_l6", "run_21_t3").
call("run_21", "run_21_t4", "move_to_position", "completed").
depends("run_21", "run_21_t4", "run_21_l6").
spawned_by("run_21", "run_21_t4", "run_21_l6").
param("run_21", "run_21_t4", "pos_distance", "0.65").
param("run_21", "run_21_t4", "delta_gripper", "1.3").
param("run_21", "run_21_t4", "efficiency", "0.9").
param("run_21", "run_21_t4", "mean_speed", "0.171").
call("run_21", "run_21_l7", "llm_generate", "completed").
depends("run_21", "run_21_l7", "run_21_t4").
call("run_21", "run_21_t5", "open_gripper", "completed").
depends("run_21", "run_21_t5", "run_21_l7").
spawned_by("run_21", "run_21_t5", "run_21_l7").
param("run_21", "run_21_t5", "pos_distance", "0.51").
param("run_21", "run_21_t5", "delta_gripper", "1.6").
param("run_21", "run_21_t5", "efficiency", "0.7").
param("run_21", "run_21_t5", "mean_speed", "0.165").
call("run_21", "run_21_l8", "llm_generate", "completed").
depends("run_21", "run_21_l8", "run_21_t5").
call("run_21", "run_21_t6", "adjust_position", "completed").
depends("run_21", "run_21_t6", "run_21_l8").
spawned_by("run_21", "run_21_t6", "run_21_l8").
param("run_21", "run_21_t6", "pos_distance", "0.25").
param("run_21", "run_21_t6", "delta_gripper", "0.8").
param("run_21", "run_21_t6", "efficiency", "0.6").
param("run_21", "run_21_t6", "mean_speed", "0.103").
call("run_21", "run_21_l9", "llm_generate", "completed").
depends("run_21", "run_21_l9", "run_21_t6").
call("run_21", "run_21_t7", "adjust_position", "completed").
depends("run_21", "run_21_t7", "run_21_l9").
spawned_by("run_21", "run_21_t7", "run_21_l9").
param("run_21", "run_21_t7", "pos_distance", "0.18").
param("run_21", "run_21_t7", "delta_gripper", "0.0").
param("run_21", "run_21_t7", "efficiency", "0.3").
param("run_21", "run_21_t7", "mean_speed", "0.055").
call("run_21", "run_21_l10", "llm_generate", "completed").
depends("run_21", "run_21_l10", "run_21_t7").
call("run_21", "run_21_n", "notify_user", "completed").
depends("run_21", "run_21_n", "run_21_l10").
spawned_by("run_21", "run_21_n", "run_21_l10").

% === Run 22 (normal_variation) ===
job("run_22").
job_status("run_22", "completed").
call("run_22", "run_22_s0", "prompt_user", "completed").
call("run_22", "run_22_s1", "llm_generate", "completed").
depends("run_22", "run_22_s1", "run_22_s0").
call("run_22", "run_22_l2", "llm_generate", "completed").
depends("run_22", "run_22_l2", "run_22_s1").
call("run_22", "run_22_t0", "move_to_position", "completed").
depends("run_22", "run_22_t0", "run_22_l2").
spawned_by("run_22", "run_22_t0", "run_22_l2").
param("run_22", "run_22_t0", "pos_distance", "0.41").
param("run_22", "run_22_t0", "delta_gripper", "0.0").
param("run_22", "run_22_t0", "efficiency", "0.8").
param("run_22", "run_22_t0", "mean_speed", "0.219").
call("run_22", "run_22_l3", "llm_generate", "completed").
depends("run_22", "run_22_l3", "run_22_t0").
call("run_22", "run_22_t1", "close_gripper", "completed").
depends("run_22", "run_22_t1", "run_22_l3").
spawned_by("run_22", "run_22_t1", "run_22_l3").
param("run_22", "run_22_t1", "pos_distance", "0.44").
param("run_22", "run_22_t1", "delta_gripper", "-2.0").
param("run_22", "run_22_t1", "efficiency", "0.9").
param("run_22", "run_22_t1", "mean_speed", "0.121").
call("run_22", "run_22_l4", "llm_generate", "completed").
depends("run_22", "run_22_l4", "run_22_t1").
call("run_22", "run_22_t2", "close_gripper", "completed").
depends("run_22", "run_22_t2", "run_22_l4").
spawned_by("run_22", "run_22_t2", "run_22_l4").
param("run_22", "run_22_t2", "pos_distance", "0.25").
param("run_22", "run_22_t2", "delta_gripper", "-1.8").
param("run_22", "run_22_t2", "efficiency", "1.0").
param("run_22", "run_22_t2", "mean_speed", "0.110").
call("run_22", "run_22_l5", "llm_generate", "completed").
depends("run_22", "run_22_l5", "run_22_t2").
call("run_22", "run_22_t3", "move_to_position", "completed").
depends("run_22", "run_22_t3", "run_22_l5").
spawned_by("run_22", "run_22_t3", "run_22_l5").
param("run_22", "run_22_t3", "pos_distance", "1.87").
param("run_22", "run_22_t3", "delta_gripper", "0.0").
param("run_22", "run_22_t3", "efficiency", "0.8").
param("run_22", "run_22_t3", "mean_speed", "0.299").
call("run_22", "run_22_l6", "llm_generate", "completed").
depends("run_22", "run_22_l6", "run_22_t3").
call("run_22", "run_22_t4", "move_to_position", "completed").
depends("run_22", "run_22_t4", "run_22_l6").
spawned_by("run_22", "run_22_t4", "run_22_l6").
param("run_22", "run_22_t4", "pos_distance", "0.52").
param("run_22", "run_22_t4", "delta_gripper", "1.3").
param("run_22", "run_22_t4", "efficiency", "0.8").
param("run_22", "run_22_t4", "mean_speed", "0.178").
call("run_22", "run_22_l7", "llm_generate", "completed").
depends("run_22", "run_22_l7", "run_22_t4").
call("run_22", "run_22_t5", "open_gripper", "completed").
depends("run_22", "run_22_t5", "run_22_l7").
spawned_by("run_22", "run_22_t5", "run_22_l7").
param("run_22", "run_22_t5", "pos_distance", "0.45").
param("run_22", "run_22_t5", "delta_gripper", "1.6").
param("run_22", "run_22_t5", "efficiency", "0.7").
param("run_22", "run_22_t5", "mean_speed", "0.228").
call("run_22", "run_22_l8", "llm_generate", "completed").
depends("run_22", "run_22_l8", "run_22_t5").
call("run_22", "run_22_t6", "adjust_position", "completed").
depends("run_22", "run_22_t6", "run_22_l8").
spawned_by("run_22", "run_22_t6", "run_22_l8").
param("run_22", "run_22_t6", "pos_distance", "0.29").
param("run_22", "run_22_t6", "delta_gripper", "0.8").
param("run_22", "run_22_t6", "efficiency", "0.6").
param("run_22", "run_22_t6", "mean_speed", "0.155").
call("run_22", "run_22_l9", "llm_generate", "completed").
depends("run_22", "run_22_l9", "run_22_t6").
call("run_22", "run_22_t7", "adjust_position", "completed").
depends("run_22", "run_22_t7", "run_22_l9").
spawned_by("run_22", "run_22_t7", "run_22_l9").
param("run_22", "run_22_t7", "pos_distance", "0.44").
param("run_22", "run_22_t7", "delta_gripper", "0.0").
param("run_22", "run_22_t7", "efficiency", "0.3").
param("run_22", "run_22_t7", "mean_speed", "0.074").
call("run_22", "run_22_l10", "llm_generate", "completed").
depends("run_22", "run_22_l10", "run_22_t7").
call("run_22", "run_22_n", "notify_user", "completed").
depends("run_22", "run_22_n", "run_22_l10").
spawned_by("run_22", "run_22_n", "run_22_l10").

% === Run 23 (normal_variation) ===
job("run_23").
job_status("run_23", "completed").
call("run_23", "run_23_s0", "prompt_user", "completed").
call("run_23", "run_23_s1", "llm_generate", "completed").
depends("run_23", "run_23_s1", "run_23_s0").
call("run_23", "run_23_l2", "llm_generate", "completed").
depends("run_23", "run_23_l2", "run_23_s1").
call("run_23", "run_23_t0", "move_to_position", "completed").
depends("run_23", "run_23_t0", "run_23_l2").
spawned_by("run_23", "run_23_t0", "run_23_l2").
param("run_23", "run_23_t0", "pos_distance", "0.41").
param("run_23", "run_23_t0", "delta_gripper", "0.0").
param("run_23", "run_23_t0", "efficiency", "0.8").
param("run_23", "run_23_t0", "mean_speed", "0.160").
call("run_23", "run_23_l3", "llm_generate", "completed").
depends("run_23", "run_23_l3", "run_23_t0").
call("run_23", "run_23_t1", "close_gripper", "completed").
depends("run_23", "run_23_t1", "run_23_l3").
spawned_by("run_23", "run_23_t1", "run_23_l3").
param("run_23", "run_23_t1", "pos_distance", "0.54").
param("run_23", "run_23_t1", "delta_gripper", "-2.0").
param("run_23", "run_23_t1", "efficiency", "0.8").
param("run_23", "run_23_t1", "mean_speed", "0.172").
call("run_23", "run_23_l4", "llm_generate", "completed").
depends("run_23", "run_23_l4", "run_23_t1").
call("run_23", "run_23_t2", "close_gripper", "completed").
depends("run_23", "run_23_t2", "run_23_l4").
spawned_by("run_23", "run_23_t2", "run_23_l4").
param("run_23", "run_23_t2", "pos_distance", "0.43").
param("run_23", "run_23_t2", "delta_gripper", "-1.8").
param("run_23", "run_23_t2", "efficiency", "0.8").
param("run_23", "run_23_t2", "mean_speed", "0.142").
call("run_23", "run_23_l5", "llm_generate", "completed").
depends("run_23", "run_23_l5", "run_23_t2").
call("run_23", "run_23_t3", "move_to_position", "completed").
depends("run_23", "run_23_t3", "run_23_l5").
spawned_by("run_23", "run_23_t3", "run_23_l5").
param("run_23", "run_23_t3", "pos_distance", "2.36").
param("run_23", "run_23_t3", "delta_gripper", "0.0").
param("run_23", "run_23_t3", "efficiency", "0.6").
param("run_23", "run_23_t3", "mean_speed", "0.298").
call("run_23", "run_23_l6", "llm_generate", "completed").
depends("run_23", "run_23_l6", "run_23_t3").
call("run_23", "run_23_t4", "move_to_position", "completed").
depends("run_23", "run_23_t4", "run_23_l6").
spawned_by("run_23", "run_23_t4", "run_23_l6").
param("run_23", "run_23_t4", "pos_distance", "0.91").
param("run_23", "run_23_t4", "delta_gripper", "1.3").
param("run_23", "run_23_t4", "efficiency", "0.8").
param("run_23", "run_23_t4", "mean_speed", "0.304").
call("run_23", "run_23_l7", "llm_generate", "completed").
depends("run_23", "run_23_l7", "run_23_t4").
call("run_23", "run_23_t5", "open_gripper", "completed").
depends("run_23", "run_23_t5", "run_23_l7").
spawned_by("run_23", "run_23_t5", "run_23_l7").
param("run_23", "run_23_t5", "pos_distance", "0.23").
param("run_23", "run_23_t5", "delta_gripper", "1.6").
param("run_23", "run_23_t5", "efficiency", "0.8").
param("run_23", "run_23_t5", "mean_speed", "0.156").
call("run_23", "run_23_l8", "llm_generate", "completed").
depends("run_23", "run_23_l8", "run_23_t5").
call("run_23", "run_23_t6", "adjust_position", "completed").
depends("run_23", "run_23_t6", "run_23_l8").
spawned_by("run_23", "run_23_t6", "run_23_l8").
param("run_23", "run_23_t6", "pos_distance", "0.37").
param("run_23", "run_23_t6", "delta_gripper", "0.8").
param("run_23", "run_23_t6", "efficiency", "0.6").
param("run_23", "run_23_t6", "mean_speed", "0.125").
call("run_23", "run_23_l9", "llm_generate", "completed").
depends("run_23", "run_23_l9", "run_23_t6").
call("run_23", "run_23_t7", "adjust_position", "completed").
depends("run_23", "run_23_t7", "run_23_l9").
spawned_by("run_23", "run_23_t7", "run_23_l9").
param("run_23", "run_23_t7", "pos_distance", "0.20").
param("run_23", "run_23_t7", "delta_gripper", "0.0").
param("run_23", "run_23_t7", "efficiency", "0.3").
param("run_23", "run_23_t7", "mean_speed", "0.065").
call("run_23", "run_23_l10", "llm_generate", "completed").
depends("run_23", "run_23_l10", "run_23_t7").
call("run_23", "run_23_n", "notify_user", "completed").
depends("run_23", "run_23_n", "run_23_l10").
spawned_by("run_23", "run_23_n", "run_23_l10").

% === Run 24 (normal_variation) ===
job("run_24").
job_status("run_24", "completed").
call("run_24", "run_24_s0", "prompt_user", "completed").
call("run_24", "run_24_s1", "llm_generate", "completed").
depends("run_24", "run_24_s1", "run_24_s0").
call("run_24", "run_24_l2", "llm_generate", "completed").
depends("run_24", "run_24_l2", "run_24_s1").
call("run_24", "run_24_t0", "move_to_position", "completed").
depends("run_24", "run_24_t0", "run_24_l2").
spawned_by("run_24", "run_24_t0", "run_24_l2").
param("run_24", "run_24_t0", "pos_distance", "0.47").
param("run_24", "run_24_t0", "delta_gripper", "0.0").
param("run_24", "run_24_t0", "efficiency", "1.0").
param("run_24", "run_24_t0", "mean_speed", "0.235").
call("run_24", "run_24_l3", "llm_generate", "completed").
depends("run_24", "run_24_l3", "run_24_t0").
call("run_24", "run_24_t1", "close_gripper", "completed").
depends("run_24", "run_24_t1", "run_24_l3").
spawned_by("run_24", "run_24_t1", "run_24_l3").
param("run_24", "run_24_t1", "pos_distance", "0.51").
param("run_24", "run_24_t1", "delta_gripper", "-2.0").
param("run_24", "run_24_t1", "efficiency", "1.0").
param("run_24", "run_24_t1", "mean_speed", "0.128").
call("run_24", "run_24_l4", "llm_generate", "completed").
depends("run_24", "run_24_l4", "run_24_t1").
call("run_24", "run_24_t2", "close_gripper", "completed").
depends("run_24", "run_24_t2", "run_24_l4").
spawned_by("run_24", "run_24_t2", "run_24_l4").
param("run_24", "run_24_t2", "pos_distance", "0.46").
param("run_24", "run_24_t2", "delta_gripper", "-1.8").
param("run_24", "run_24_t2", "efficiency", "0.9").
param("run_24", "run_24_t2", "mean_speed", "0.111").
call("run_24", "run_24_l5", "llm_generate", "completed").
depends("run_24", "run_24_l5", "run_24_t2").
call("run_24", "run_24_t3", "move_to_position", "completed").
depends("run_24", "run_24_t3", "run_24_l5").
spawned_by("run_24", "run_24_t3", "run_24_l5").
param("run_24", "run_24_t3", "pos_distance", "1.85").
param("run_24", "run_24_t3", "delta_gripper", "0.0").
param("run_24", "run_24_t3", "efficiency", "0.7").
param("run_24", "run_24_t3", "mean_speed", "0.260").
call("run_24", "run_24_l6", "llm_generate", "completed").
depends("run_24", "run_24_l6", "run_24_t3").
call("run_24", "run_24_t4", "move_to_position", "completed").
depends("run_24", "run_24_t4", "run_24_l6").
spawned_by("run_24", "run_24_t4", "run_24_l6").
param("run_24", "run_24_t4", "pos_distance", "0.93").
param("run_24", "run_24_t4", "delta_gripper", "1.3").
param("run_24", "run_24_t4", "efficiency", "0.8").
param("run_24", "run_24_t4", "mean_speed", "0.284").
call("run_24", "run_24_l7", "llm_generate", "completed").
depends("run_24", "run_24_l7", "run_24_t4").
call("run_24", "run_24_t5", "open_gripper", "completed").
depends("run_24", "run_24_t5", "run_24_l7").
spawned_by("run_24", "run_24_t5", "run_24_l7").
param("run_24", "run_24_t5", "pos_distance", "0.63").
param("run_24", "run_24_t5", "delta_gripper", "1.6").
param("run_24", "run_24_t5", "efficiency", "0.9").
param("run_24", "run_24_t5", "mean_speed", "0.190").
call("run_24", "run_24_l8", "llm_generate", "completed").
depends("run_24", "run_24_l8", "run_24_t5").
call("run_24", "run_24_t6", "adjust_position", "completed").
depends("run_24", "run_24_t6", "run_24_l8").
spawned_by("run_24", "run_24_t6", "run_24_l8").
param("run_24", "run_24_t6", "pos_distance", "0.31").
param("run_24", "run_24_t6", "delta_gripper", "0.8").
param("run_24", "run_24_t6", "efficiency", "0.6").
param("run_24", "run_24_t6", "mean_speed", "0.128").
call("run_24", "run_24_l9", "llm_generate", "completed").
depends("run_24", "run_24_l9", "run_24_t6").
call("run_24", "run_24_t7", "adjust_position", "completed").
depends("run_24", "run_24_t7", "run_24_l9").
spawned_by("run_24", "run_24_t7", "run_24_l9").
param("run_24", "run_24_t7", "pos_distance", "0.51").
param("run_24", "run_24_t7", "delta_gripper", "0.0").
param("run_24", "run_24_t7", "efficiency", "0.3").
param("run_24", "run_24_t7", "mean_speed", "0.050").
call("run_24", "run_24_l10", "llm_generate", "completed").
depends("run_24", "run_24_l10", "run_24_t7").
call("run_24", "run_24_n", "notify_user", "completed").
depends("run_24", "run_24_n", "run_24_l10").
spawned_by("run_24", "run_24_n", "run_24_l10").

% === Run 25 (normal_variation) ===
job("run_25").
job_status("run_25", "completed").
call("run_25", "run_25_s0", "prompt_user", "completed").
call("run_25", "run_25_s1", "llm_generate", "completed").
depends("run_25", "run_25_s1", "run_25_s0").
call("run_25", "run_25_l2", "llm_generate", "completed").
depends("run_25", "run_25_l2", "run_25_s1").
call("run_25", "run_25_t0", "move_to_position", "completed").
depends("run_25", "run_25_t0", "run_25_l2").
spawned_by("run_25", "run_25_t0", "run_25_l2").
param("run_25", "run_25_t0", "pos_distance", "0.81").
param("run_25", "run_25_t0", "delta_gripper", "0.0").
param("run_25", "run_25_t0", "efficiency", "0.9").
param("run_25", "run_25_t0", "mean_speed", "0.200").
call("run_25", "run_25_l3", "llm_generate", "completed").
depends("run_25", "run_25_l3", "run_25_t0").
call("run_25", "run_25_t1", "close_gripper", "completed").
depends("run_25", "run_25_t1", "run_25_l3").
spawned_by("run_25", "run_25_t1", "run_25_l3").
param("run_25", "run_25_t1", "pos_distance", "0.57").
param("run_25", "run_25_t1", "delta_gripper", "-2.0").
param("run_25", "run_25_t1", "efficiency", "1.0").
param("run_25", "run_25_t1", "mean_speed", "0.142").
call("run_25", "run_25_l4", "llm_generate", "completed").
depends("run_25", "run_25_l4", "run_25_t1").
call("run_25", "run_25_t2", "close_gripper", "completed").
depends("run_25", "run_25_t2", "run_25_l4").
spawned_by("run_25", "run_25_t2", "run_25_l4").
param("run_25", "run_25_t2", "pos_distance", "0.48").
param("run_25", "run_25_t2", "delta_gripper", "-1.8").
param("run_25", "run_25_t2", "efficiency", "1.0").
param("run_25", "run_25_t2", "mean_speed", "0.092").
call("run_25", "run_25_l5", "llm_generate", "completed").
depends("run_25", "run_25_l5", "run_25_t2").
call("run_25", "run_25_t3", "move_to_position", "completed").
depends("run_25", "run_25_t3", "run_25_l5").
spawned_by("run_25", "run_25_t3", "run_25_l5").
param("run_25", "run_25_t3", "pos_distance", "1.58").
param("run_25", "run_25_t3", "delta_gripper", "0.0").
param("run_25", "run_25_t3", "efficiency", "0.7").
param("run_25", "run_25_t3", "mean_speed", "0.254").
call("run_25", "run_25_l6", "llm_generate", "completed").
depends("run_25", "run_25_l6", "run_25_t3").
call("run_25", "run_25_t4", "move_to_position", "completed").
depends("run_25", "run_25_t4", "run_25_l6").
spawned_by("run_25", "run_25_t4", "run_25_l6").
param("run_25", "run_25_t4", "pos_distance", "0.61").
param("run_25", "run_25_t4", "delta_gripper", "1.3").
param("run_25", "run_25_t4", "efficiency", "1.0").
param("run_25", "run_25_t4", "mean_speed", "0.170").
call("run_25", "run_25_l7", "llm_generate", "completed").
depends("run_25", "run_25_l7", "run_25_t4").
call("run_25", "run_25_t5", "open_gripper", "completed").
depends("run_25", "run_25_t5", "run_25_l7").
spawned_by("run_25", "run_25_t5", "run_25_l7").
param("run_25", "run_25_t5", "pos_distance", "0.30").
param("run_25", "run_25_t5", "delta_gripper", "1.6").
param("run_25", "run_25_t5", "efficiency", "0.9").
param("run_25", "run_25_t5", "mean_speed", "0.145").
call("run_25", "run_25_l8", "llm_generate", "completed").
depends("run_25", "run_25_l8", "run_25_t5").
call("run_25", "run_25_t6", "adjust_position", "completed").
depends("run_25", "run_25_t6", "run_25_l8").
spawned_by("run_25", "run_25_t6", "run_25_l8").
param("run_25", "run_25_t6", "pos_distance", "0.21").
param("run_25", "run_25_t6", "delta_gripper", "0.8").
param("run_25", "run_25_t6", "efficiency", "0.6").
param("run_25", "run_25_t6", "mean_speed", "0.163").
call("run_25", "run_25_l9", "llm_generate", "completed").
depends("run_25", "run_25_l9", "run_25_t6").
call("run_25", "run_25_t7", "adjust_position", "completed").
depends("run_25", "run_25_t7", "run_25_l9").
spawned_by("run_25", "run_25_t7", "run_25_l9").
param("run_25", "run_25_t7", "pos_distance", "0.27").
param("run_25", "run_25_t7", "delta_gripper", "0.0").
param("run_25", "run_25_t7", "efficiency", "0.3").
param("run_25", "run_25_t7", "mean_speed", "0.041").
call("run_25", "run_25_l10", "llm_generate", "completed").
depends("run_25", "run_25_l10", "run_25_t7").
call("run_25", "run_25_n", "notify_user", "completed").
depends("run_25", "run_25_n", "run_25_l10").
spawned_by("run_25", "run_25_n", "run_25_l10").

% === Run 26 (normal_variation) ===
job("run_26").
job_status("run_26", "completed").
call("run_26", "run_26_s0", "prompt_user", "completed").
call("run_26", "run_26_s1", "llm_generate", "completed").
depends("run_26", "run_26_s1", "run_26_s0").
call("run_26", "run_26_l2", "llm_generate", "completed").
depends("run_26", "run_26_l2", "run_26_s1").
call("run_26", "run_26_t0", "move_to_position", "completed").
depends("run_26", "run_26_t0", "run_26_l2").
spawned_by("run_26", "run_26_t0", "run_26_l2").
param("run_26", "run_26_t0", "pos_distance", "0.46").
param("run_26", "run_26_t0", "delta_gripper", "0.0").
param("run_26", "run_26_t0", "efficiency", "0.9").
param("run_26", "run_26_t0", "mean_speed", "0.229").
call("run_26", "run_26_l3", "llm_generate", "completed").
depends("run_26", "run_26_l3", "run_26_t0").
call("run_26", "run_26_t1", "close_gripper", "completed").
depends("run_26", "run_26_t1", "run_26_l3").
spawned_by("run_26", "run_26_t1", "run_26_l3").
param("run_26", "run_26_t1", "pos_distance", "0.58").
param("run_26", "run_26_t1", "delta_gripper", "-2.0").
param("run_26", "run_26_t1", "efficiency", "0.9").
param("run_26", "run_26_t1", "mean_speed", "0.184").
call("run_26", "run_26_l4", "llm_generate", "completed").
depends("run_26", "run_26_l4", "run_26_t1").
call("run_26", "run_26_t2", "close_gripper", "completed").
depends("run_26", "run_26_t2", "run_26_l4").
spawned_by("run_26", "run_26_t2", "run_26_l4").
param("run_26", "run_26_t2", "pos_distance", "0.35").
param("run_26", "run_26_t2", "delta_gripper", "-1.8").
param("run_26", "run_26_t2", "efficiency", "0.8").
param("run_26", "run_26_t2", "mean_speed", "0.144").
call("run_26", "run_26_l5", "llm_generate", "completed").
depends("run_26", "run_26_l5", "run_26_t2").
call("run_26", "run_26_t3", "move_to_position", "completed").
depends("run_26", "run_26_t3", "run_26_l5").
spawned_by("run_26", "run_26_t3", "run_26_l5").
param("run_26", "run_26_t3", "pos_distance", "1.54").
param("run_26", "run_26_t3", "delta_gripper", "0.0").
param("run_26", "run_26_t3", "efficiency", "0.7").
param("run_26", "run_26_t3", "mean_speed", "0.289").
call("run_26", "run_26_l6", "llm_generate", "completed").
depends("run_26", "run_26_l6", "run_26_t3").
call("run_26", "run_26_t4", "move_to_position", "completed").
depends("run_26", "run_26_t4", "run_26_l6").
spawned_by("run_26", "run_26_t4", "run_26_l6").
param("run_26", "run_26_t4", "pos_distance", "0.75").
param("run_26", "run_26_t4", "delta_gripper", "1.3").
param("run_26", "run_26_t4", "efficiency", "0.9").
param("run_26", "run_26_t4", "mean_speed", "0.234").
call("run_26", "run_26_l7", "llm_generate", "completed").
depends("run_26", "run_26_l7", "run_26_t4").
call("run_26", "run_26_t5", "open_gripper", "completed").
depends("run_26", "run_26_t5", "run_26_l7").
spawned_by("run_26", "run_26_t5", "run_26_l7").
param("run_26", "run_26_t5", "pos_distance", "0.40").
param("run_26", "run_26_t5", "delta_gripper", "1.6").
param("run_26", "run_26_t5", "efficiency", "0.7").
param("run_26", "run_26_t5", "mean_speed", "0.201").
call("run_26", "run_26_l8", "llm_generate", "completed").
depends("run_26", "run_26_l8", "run_26_t5").
call("run_26", "run_26_t6", "adjust_position", "completed").
depends("run_26", "run_26_t6", "run_26_l8").
spawned_by("run_26", "run_26_t6", "run_26_l8").
param("run_26", "run_26_t6", "pos_distance", "0.31").
param("run_26", "run_26_t6", "delta_gripper", "0.8").
param("run_26", "run_26_t6", "efficiency", "0.6").
param("run_26", "run_26_t6", "mean_speed", "0.115").
call("run_26", "run_26_l9", "llm_generate", "completed").
depends("run_26", "run_26_l9", "run_26_t6").
call("run_26", "run_26_t7", "adjust_position", "completed").
depends("run_26", "run_26_t7", "run_26_l9").
spawned_by("run_26", "run_26_t7", "run_26_l9").
param("run_26", "run_26_t7", "pos_distance", "0.22").
param("run_26", "run_26_t7", "delta_gripper", "0.0").
param("run_26", "run_26_t7", "efficiency", "0.3").
param("run_26", "run_26_t7", "mean_speed", "0.048").
call("run_26", "run_26_l10", "llm_generate", "completed").
depends("run_26", "run_26_l10", "run_26_t7").
call("run_26", "run_26_n", "notify_user", "completed").
depends("run_26", "run_26_n", "run_26_l10").
spawned_by("run_26", "run_26_n", "run_26_l10").

% === Run 27 (normal_variation) ===
job("run_27").
job_status("run_27", "completed").
call("run_27", "run_27_s0", "prompt_user", "completed").
call("run_27", "run_27_s1", "llm_generate", "completed").
depends("run_27", "run_27_s1", "run_27_s0").
call("run_27", "run_27_l2", "llm_generate", "completed").
depends("run_27", "run_27_l2", "run_27_s1").
call("run_27", "run_27_t0", "move_to_position", "completed").
depends("run_27", "run_27_t0", "run_27_l2").
spawned_by("run_27", "run_27_t0", "run_27_l2").
param("run_27", "run_27_t0", "pos_distance", "0.81").
param("run_27", "run_27_t0", "delta_gripper", "0.0").
param("run_27", "run_27_t0", "efficiency", "0.8").
param("run_27", "run_27_t0", "mean_speed", "0.234").
call("run_27", "run_27_l3", "llm_generate", "completed").
depends("run_27", "run_27_l3", "run_27_t0").
call("run_27", "run_27_t1", "close_gripper", "completed").
depends("run_27", "run_27_t1", "run_27_l3").
spawned_by("run_27", "run_27_t1", "run_27_l3").
param("run_27", "run_27_t1", "pos_distance", "0.33").
param("run_27", "run_27_t1", "delta_gripper", "-2.0").
param("run_27", "run_27_t1", "efficiency", "0.9").
param("run_27", "run_27_t1", "mean_speed", "0.164").
call("run_27", "run_27_l4", "llm_generate", "completed").
depends("run_27", "run_27_l4", "run_27_t1").
call("run_27", "run_27_t2", "close_gripper", "completed").
depends("run_27", "run_27_t2", "run_27_l4").
spawned_by("run_27", "run_27_t2", "run_27_l4").
param("run_27", "run_27_t2", "pos_distance", "0.53").
param("run_27", "run_27_t2", "delta_gripper", "-1.8").
param("run_27", "run_27_t2", "efficiency", "0.9").
param("run_27", "run_27_t2", "mean_speed", "0.125").
call("run_27", "run_27_l5", "llm_generate", "completed").
depends("run_27", "run_27_l5", "run_27_t2").
call("run_27", "run_27_t3", "move_to_position", "completed").
depends("run_27", "run_27_t3", "run_27_l5").
spawned_by("run_27", "run_27_t3", "run_27_l5").
param("run_27", "run_27_t3", "pos_distance", "1.28").
param("run_27", "run_27_t3", "delta_gripper", "0.0").
param("run_27", "run_27_t3", "efficiency", "0.6").
param("run_27", "run_27_t3", "mean_speed", "0.212").
call("run_27", "run_27_l6", "llm_generate", "completed").
depends("run_27", "run_27_l6", "run_27_t3").
call("run_27", "run_27_t4", "move_to_position", "completed").
depends("run_27", "run_27_t4", "run_27_l6").
spawned_by("run_27", "run_27_t4", "run_27_l6").
param("run_27", "run_27_t4", "pos_distance", "0.47").
param("run_27", "run_27_t4", "delta_gripper", "1.3").
param("run_27", "run_27_t4", "efficiency", "0.9").
param("run_27", "run_27_t4", "mean_speed", "0.168").
call("run_27", "run_27_l7", "llm_generate", "completed").
depends("run_27", "run_27_l7", "run_27_t4").
call("run_27", "run_27_t5", "open_gripper", "completed").
depends("run_27", "run_27_t5", "run_27_l7").
spawned_by("run_27", "run_27_t5", "run_27_l7").
param("run_27", "run_27_t5", "pos_distance", "0.34").
param("run_27", "run_27_t5", "delta_gripper", "1.6").
param("run_27", "run_27_t5", "efficiency", "0.8").
param("run_27", "run_27_t5", "mean_speed", "0.226").
call("run_27", "run_27_l8", "llm_generate", "completed").
depends("run_27", "run_27_l8", "run_27_t5").
call("run_27", "run_27_t6", "adjust_position", "completed").
depends("run_27", "run_27_t6", "run_27_l8").
spawned_by("run_27", "run_27_t6", "run_27_l8").
param("run_27", "run_27_t6", "pos_distance", "0.30").
param("run_27", "run_27_t6", "delta_gripper", "0.8").
param("run_27", "run_27_t6", "efficiency", "0.7").
param("run_27", "run_27_t6", "mean_speed", "0.103").
call("run_27", "run_27_l9", "llm_generate", "completed").
depends("run_27", "run_27_l9", "run_27_t6").
call("run_27", "run_27_t7", "adjust_position", "completed").
depends("run_27", "run_27_t7", "run_27_l9").
spawned_by("run_27", "run_27_t7", "run_27_l9").
param("run_27", "run_27_t7", "pos_distance", "0.34").
param("run_27", "run_27_t7", "delta_gripper", "0.0").
param("run_27", "run_27_t7", "efficiency", "0.3").
param("run_27", "run_27_t7", "mean_speed", "0.071").
call("run_27", "run_27_l10", "llm_generate", "completed").
depends("run_27", "run_27_l10", "run_27_t7").
call("run_27", "run_27_n", "notify_user", "completed").
depends("run_27", "run_27_n", "run_27_l10").
spawned_by("run_27", "run_27_n", "run_27_l10").

% === Run 28 (normal_variation) ===
job("run_28").
job_status("run_28", "completed").
call("run_28", "run_28_s0", "prompt_user", "completed").
call("run_28", "run_28_s1", "llm_generate", "completed").
depends("run_28", "run_28_s1", "run_28_s0").
call("run_28", "run_28_l2", "llm_generate", "completed").
depends("run_28", "run_28_l2", "run_28_s1").
call("run_28", "run_28_t0", "move_to_position", "completed").
depends("run_28", "run_28_t0", "run_28_l2").
spawned_by("run_28", "run_28_t0", "run_28_l2").
param("run_28", "run_28_t0", "pos_distance", "0.76").
param("run_28", "run_28_t0", "delta_gripper", "0.0").
param("run_28", "run_28_t0", "efficiency", "0.8").
param("run_28", "run_28_t0", "mean_speed", "0.195").
call("run_28", "run_28_l3", "llm_generate", "completed").
depends("run_28", "run_28_l3", "run_28_t0").
call("run_28", "run_28_t1", "close_gripper", "completed").
depends("run_28", "run_28_t1", "run_28_l3").
spawned_by("run_28", "run_28_t1", "run_28_l3").
param("run_28", "run_28_t1", "pos_distance", "0.26").
param("run_28", "run_28_t1", "delta_gripper", "-2.0").
param("run_28", "run_28_t1", "efficiency", "0.9").
param("run_28", "run_28_t1", "mean_speed", "0.209").
call("run_28", "run_28_l4", "llm_generate", "completed").
depends("run_28", "run_28_l4", "run_28_t1").
call("run_28", "run_28_t2", "close_gripper", "completed").
depends("run_28", "run_28_t2", "run_28_l4").
spawned_by("run_28", "run_28_t2", "run_28_l4").
param("run_28", "run_28_t2", "pos_distance", "0.47").
param("run_28", "run_28_t2", "delta_gripper", "-1.8").
param("run_28", "run_28_t2", "efficiency", "0.8").
param("run_28", "run_28_t2", "mean_speed", "0.115").
call("run_28", "run_28_l5", "llm_generate", "completed").
depends("run_28", "run_28_l5", "run_28_t2").
call("run_28", "run_28_t3", "move_to_position", "completed").
depends("run_28", "run_28_t3", "run_28_l5").
spawned_by("run_28", "run_28_t3", "run_28_l5").
param("run_28", "run_28_t3", "pos_distance", "2.40").
param("run_28", "run_28_t3", "delta_gripper", "0.0").
param("run_28", "run_28_t3", "efficiency", "0.6").
param("run_28", "run_28_t3", "mean_speed", "0.190").
call("run_28", "run_28_l6", "llm_generate", "completed").
depends("run_28", "run_28_l6", "run_28_t3").
call("run_28", "run_28_t4", "move_to_position", "completed").
depends("run_28", "run_28_t4", "run_28_l6").
spawned_by("run_28", "run_28_t4", "run_28_l6").
param("run_28", "run_28_t4", "pos_distance", "0.73").
param("run_28", "run_28_t4", "delta_gripper", "1.3").
param("run_28", "run_28_t4", "efficiency", "1.0").
param("run_28", "run_28_t4", "mean_speed", "0.210").
call("run_28", "run_28_l7", "llm_generate", "completed").
depends("run_28", "run_28_l7", "run_28_t4").
call("run_28", "run_28_t5", "open_gripper", "completed").
depends("run_28", "run_28_t5", "run_28_l7").
spawned_by("run_28", "run_28_t5", "run_28_l7").
param("run_28", "run_28_t5", "pos_distance", "0.27").
param("run_28", "run_28_t5", "delta_gripper", "1.6").
param("run_28", "run_28_t5", "efficiency", "0.8").
param("run_28", "run_28_t5", "mean_speed", "0.203").
call("run_28", "run_28_l8", "llm_generate", "completed").
depends("run_28", "run_28_l8", "run_28_t5").
call("run_28", "run_28_t6", "adjust_position", "completed").
depends("run_28", "run_28_t6", "run_28_l8").
spawned_by("run_28", "run_28_t6", "run_28_l8").
param("run_28", "run_28_t6", "pos_distance", "0.33").
param("run_28", "run_28_t6", "delta_gripper", "0.8").
param("run_28", "run_28_t6", "efficiency", "0.6").
param("run_28", "run_28_t6", "mean_speed", "0.111").
call("run_28", "run_28_l9", "llm_generate", "completed").
depends("run_28", "run_28_l9", "run_28_t6").
call("run_28", "run_28_t7", "adjust_position", "completed").
depends("run_28", "run_28_t7", "run_28_l9").
spawned_by("run_28", "run_28_t7", "run_28_l9").
param("run_28", "run_28_t7", "pos_distance", "0.33").
param("run_28", "run_28_t7", "delta_gripper", "0.0").
param("run_28", "run_28_t7", "efficiency", "0.3").
param("run_28", "run_28_t7", "mean_speed", "0.056").
call("run_28", "run_28_l10", "llm_generate", "completed").
depends("run_28", "run_28_l10", "run_28_t7").
call("run_28", "run_28_n", "notify_user", "completed").
depends("run_28", "run_28_n", "run_28_l10").
spawned_by("run_28", "run_28_n", "run_28_l10").

% === Run 29 (normal_variation) ===
job("run_29").
job_status("run_29", "completed").
call("run_29", "run_29_s0", "prompt_user", "completed").
call("run_29", "run_29_s1", "llm_generate", "completed").
depends("run_29", "run_29_s1", "run_29_s0").
call("run_29", "run_29_l2", "llm_generate", "completed").
depends("run_29", "run_29_l2", "run_29_s1").
call("run_29", "run_29_t0", "move_to_position", "completed").
depends("run_29", "run_29_t0", "run_29_l2").
spawned_by("run_29", "run_29_t0", "run_29_l2").
param("run_29", "run_29_t0", "pos_distance", "1.17").
param("run_29", "run_29_t0", "delta_gripper", "0.0").
param("run_29", "run_29_t0", "efficiency", "1.0").
param("run_29", "run_29_t0", "mean_speed", "0.179").
call("run_29", "run_29_l3", "llm_generate", "completed").
depends("run_29", "run_29_l3", "run_29_t0").
call("run_29", "run_29_t1", "close_gripper", "completed").
depends("run_29", "run_29_t1", "run_29_l3").
spawned_by("run_29", "run_29_t1", "run_29_l3").
param("run_29", "run_29_t1", "pos_distance", "0.51").
param("run_29", "run_29_t1", "delta_gripper", "-2.0").
param("run_29", "run_29_t1", "efficiency", "1.0").
param("run_29", "run_29_t1", "mean_speed", "0.189").
call("run_29", "run_29_l4", "llm_generate", "completed").
depends("run_29", "run_29_l4", "run_29_t1").
call("run_29", "run_29_t2", "close_gripper", "completed").
depends("run_29", "run_29_t2", "run_29_l4").
spawned_by("run_29", "run_29_t2", "run_29_l4").
param("run_29", "run_29_t2", "pos_distance", "0.25").
param("run_29", "run_29_t2", "delta_gripper", "-1.8").
param("run_29", "run_29_t2", "efficiency", "1.0").
param("run_29", "run_29_t2", "mean_speed", "0.101").
call("run_29", "run_29_l5", "llm_generate", "completed").
depends("run_29", "run_29_l5", "run_29_t2").
call("run_29", "run_29_t3", "move_to_position", "completed").
depends("run_29", "run_29_t3", "run_29_l5").
spawned_by("run_29", "run_29_t3", "run_29_l5").
param("run_29", "run_29_t3", "pos_distance", "1.40").
param("run_29", "run_29_t3", "delta_gripper", "0.0").
param("run_29", "run_29_t3", "efficiency", "0.6").
param("run_29", "run_29_t3", "mean_speed", "0.209").
call("run_29", "run_29_l6", "llm_generate", "completed").
depends("run_29", "run_29_l6", "run_29_t3").
call("run_29", "run_29_t4", "move_to_position", "completed").
depends("run_29", "run_29_t4", "run_29_l6").
spawned_by("run_29", "run_29_t4", "run_29_l6").
param("run_29", "run_29_t4", "pos_distance", "0.35").
param("run_29", "run_29_t4", "delta_gripper", "1.3").
param("run_29", "run_29_t4", "efficiency", "0.9").
param("run_29", "run_29_t4", "mean_speed", "0.231").
call("run_29", "run_29_l7", "llm_generate", "completed").
depends("run_29", "run_29_l7", "run_29_t4").
call("run_29", "run_29_t5", "open_gripper", "completed").
depends("run_29", "run_29_t5", "run_29_l7").
spawned_by("run_29", "run_29_t5", "run_29_l7").
param("run_29", "run_29_t5", "pos_distance", "0.34").
param("run_29", "run_29_t5", "delta_gripper", "1.6").
param("run_29", "run_29_t5", "efficiency", "0.8").
param("run_29", "run_29_t5", "mean_speed", "0.147").
call("run_29", "run_29_l8", "llm_generate", "completed").
depends("run_29", "run_29_l8", "run_29_t5").
call("run_29", "run_29_t6", "adjust_position", "completed").
depends("run_29", "run_29_t6", "run_29_l8").
spawned_by("run_29", "run_29_t6", "run_29_l8").
param("run_29", "run_29_t6", "pos_distance", "0.31").
param("run_29", "run_29_t6", "delta_gripper", "0.8").
param("run_29", "run_29_t6", "efficiency", "0.6").
param("run_29", "run_29_t6", "mean_speed", "0.129").
call("run_29", "run_29_l9", "llm_generate", "completed").
depends("run_29", "run_29_l9", "run_29_t6").
call("run_29", "run_29_t7", "adjust_position", "completed").
depends("run_29", "run_29_t7", "run_29_l9").
spawned_by("run_29", "run_29_t7", "run_29_l9").
param("run_29", "run_29_t7", "pos_distance", "0.49").
param("run_29", "run_29_t7", "delta_gripper", "0.0").
param("run_29", "run_29_t7", "efficiency", "0.3").
param("run_29", "run_29_t7", "mean_speed", "0.068").
call("run_29", "run_29_l10", "llm_generate", "completed").
depends("run_29", "run_29_l10", "run_29_t7").
call("run_29", "run_29_n", "notify_user", "completed").
depends("run_29", "run_29_n", "run_29_l10").
spawned_by("run_29", "run_29_n", "run_29_l10").

% === Run 30 (normal_variation) ===
job("run_30").
job_status("run_30", "completed").
call("run_30", "run_30_s0", "prompt_user", "completed").
call("run_30", "run_30_s1", "llm_generate", "completed").
depends("run_30", "run_30_s1", "run_30_s0").
call("run_30", "run_30_l2", "llm_generate", "completed").
depends("run_30", "run_30_l2", "run_30_s1").
call("run_30", "run_30_t0", "move_to_position", "completed").
depends("run_30", "run_30_t0", "run_30_l2").
spawned_by("run_30", "run_30_t0", "run_30_l2").
param("run_30", "run_30_t0", "pos_distance", "0.91").
param("run_30", "run_30_t0", "delta_gripper", "0.0").
param("run_30", "run_30_t0", "efficiency", "0.9").
param("run_30", "run_30_t0", "mean_speed", "0.255").
call("run_30", "run_30_l3", "llm_generate", "completed").
depends("run_30", "run_30_l3", "run_30_t0").
call("run_30", "run_30_t1", "close_gripper", "completed").
depends("run_30", "run_30_t1", "run_30_l3").
spawned_by("run_30", "run_30_t1", "run_30_l3").
param("run_30", "run_30_t1", "pos_distance", "0.48").
param("run_30", "run_30_t1", "delta_gripper", "-2.0").
param("run_30", "run_30_t1", "efficiency", "1.0").
param("run_30", "run_30_t1", "mean_speed", "0.180").
call("run_30", "run_30_l4", "llm_generate", "completed").
depends("run_30", "run_30_l4", "run_30_t1").
call("run_30", "run_30_t2", "close_gripper", "completed").
depends("run_30", "run_30_t2", "run_30_l4").
spawned_by("run_30", "run_30_t2", "run_30_l4").
param("run_30", "run_30_t2", "pos_distance", "0.47").
param("run_30", "run_30_t2", "delta_gripper", "-1.8").
param("run_30", "run_30_t2", "efficiency", "0.8").
param("run_30", "run_30_t2", "mean_speed", "0.091").
call("run_30", "run_30_l5", "llm_generate", "completed").
depends("run_30", "run_30_l5", "run_30_t2").
call("run_30", "run_30_t3", "move_to_position", "completed").
depends("run_30", "run_30_t3", "run_30_l5").
spawned_by("run_30", "run_30_t3", "run_30_l5").
param("run_30", "run_30_t3", "pos_distance", "1.72").
param("run_30", "run_30_t3", "delta_gripper", "0.0").
param("run_30", "run_30_t3", "efficiency", "0.7").
param("run_30", "run_30_t3", "mean_speed", "0.298").
call("run_30", "run_30_l6", "llm_generate", "completed").
depends("run_30", "run_30_l6", "run_30_t3").
call("run_30", "run_30_t4", "move_to_position", "completed").
depends("run_30", "run_30_t4", "run_30_l6").
spawned_by("run_30", "run_30_t4", "run_30_l6").
param("run_30", "run_30_t4", "pos_distance", "0.54").
param("run_30", "run_30_t4", "delta_gripper", "1.3").
param("run_30", "run_30_t4", "efficiency", "1.0").
param("run_30", "run_30_t4", "mean_speed", "0.184").
call("run_30", "run_30_l7", "llm_generate", "completed").
depends("run_30", "run_30_l7", "run_30_t4").
call("run_30", "run_30_t5", "open_gripper", "completed").
depends("run_30", "run_30_t5", "run_30_l7").
spawned_by("run_30", "run_30_t5", "run_30_l7").
param("run_30", "run_30_t5", "pos_distance", "0.52").
param("run_30", "run_30_t5", "delta_gripper", "1.6").
param("run_30", "run_30_t5", "efficiency", "0.8").
param("run_30", "run_30_t5", "mean_speed", "0.222").
call("run_30", "run_30_l8", "llm_generate", "completed").
depends("run_30", "run_30_l8", "run_30_t5").
call("run_30", "run_30_t6", "adjust_position", "completed").
depends("run_30", "run_30_t6", "run_30_l8").
spawned_by("run_30", "run_30_t6", "run_30_l8").
param("run_30", "run_30_t6", "pos_distance", "0.25").
param("run_30", "run_30_t6", "delta_gripper", "0.8").
param("run_30", "run_30_t6", "efficiency", "0.5").
param("run_30", "run_30_t6", "mean_speed", "0.099").
call("run_30", "run_30_l9", "llm_generate", "completed").
depends("run_30", "run_30_l9", "run_30_t6").
call("run_30", "run_30_t7", "adjust_position", "completed").
depends("run_30", "run_30_t7", "run_30_l9").
spawned_by("run_30", "run_30_t7", "run_30_l9").
param("run_30", "run_30_t7", "pos_distance", "0.32").
param("run_30", "run_30_t7", "delta_gripper", "0.0").
param("run_30", "run_30_t7", "efficiency", "0.3").
param("run_30", "run_30_t7", "mean_speed", "0.052").
call("run_30", "run_30_l10", "llm_generate", "completed").
depends("run_30", "run_30_l10", "run_30_t7").
call("run_30", "run_30_n", "notify_user", "completed").
depends("run_30", "run_30_n", "run_30_l10").
spawned_by("run_30", "run_30_n", "run_30_l10").

% === Run 31 (grasp_retry) ===
job("run_31").
job_status("run_31", "completed").
call("run_31", "run_31_s0", "prompt_user", "completed").
call("run_31", "run_31_s1", "llm_generate", "completed").
depends("run_31", "run_31_s1", "run_31_s0").
call("run_31", "run_31_l2", "llm_generate", "completed").
depends("run_31", "run_31_l2", "run_31_s1").
call("run_31", "run_31_t0", "move_to_position", "completed").
depends("run_31", "run_31_t0", "run_31_l2").
spawned_by("run_31", "run_31_t0", "run_31_l2").
param("run_31", "run_31_t0", "pos_distance", "0.78").
param("run_31", "run_31_t0", "delta_gripper", "0.0").
param("run_31", "run_31_t0", "efficiency", "1.0").
param("run_31", "run_31_t0", "mean_speed", "0.202").
call("run_31", "run_31_l3", "llm_generate", "completed").
depends("run_31", "run_31_l3", "run_31_t0").
call("run_31", "run_31_t1", "close_gripper", "completed").
depends("run_31", "run_31_t1", "run_31_l3").
spawned_by("run_31", "run_31_t1", "run_31_l3").
param("run_31", "run_31_t1", "pos_distance", "0.46").
param("run_31", "run_31_t1", "delta_gripper", "-2.0").
param("run_31", "run_31_t1", "efficiency", "0.9").
param("run_31", "run_31_t1", "mean_speed", "0.165").
call("run_31", "run_31_l4", "llm_generate", "completed").
depends("run_31", "run_31_l4", "run_31_t1").
call("run_31", "run_31_t2", "open_gripper", "completed").
depends("run_31", "run_31_t2", "run_31_l4").
spawned_by("run_31", "run_31_t2", "run_31_l4").
param("run_31", "run_31_t2", "pos_distance", "0.05").
param("run_31", "run_31_t2", "delta_gripper", "3.0").
param("run_31", "run_31_t2", "efficiency", "0.3").
param("run_31", "run_31_t2", "mean_speed", "0.080").
call("run_31", "run_31_l5", "llm_generate", "completed").
depends("run_31", "run_31_l5", "run_31_t2").
call("run_31", "run_31_t3", "adjust_position", "completed").
depends("run_31", "run_31_t3", "run_31_l5").
spawned_by("run_31", "run_31_t3", "run_31_l5").
param("run_31", "run_31_t3", "pos_distance", "0.15").
param("run_31", "run_31_t3", "delta_gripper", "0.0").
param("run_31", "run_31_t3", "efficiency", "0.5").
param("run_31", "run_31_t3", "mean_speed", "0.100").
call("run_31", "run_31_l6", "llm_generate", "completed").
depends("run_31", "run_31_l6", "run_31_t3").
call("run_31", "run_31_t4", "close_gripper", "completed").
depends("run_31", "run_31_t4", "run_31_l6").
spawned_by("run_31", "run_31_t4", "run_31_l6").
param("run_31", "run_31_t4", "pos_distance", "0.46").
param("run_31", "run_31_t4", "delta_gripper", "-2.0").
param("run_31", "run_31_t4", "efficiency", "0.9").
param("run_31", "run_31_t4", "mean_speed", "0.165").
call("run_31", "run_31_l7", "llm_generate", "completed").
depends("run_31", "run_31_l7", "run_31_t4").
call("run_31", "run_31_t5", "close_gripper", "completed").
depends("run_31", "run_31_t5", "run_31_l7").
spawned_by("run_31", "run_31_t5", "run_31_l7").
param("run_31", "run_31_t5", "pos_distance", "0.36").
param("run_31", "run_31_t5", "delta_gripper", "-1.8").
param("run_31", "run_31_t5", "efficiency", "1.0").
param("run_31", "run_31_t5", "mean_speed", "0.123").
call("run_31", "run_31_l8", "llm_generate", "completed").
depends("run_31", "run_31_l8", "run_31_t5").
call("run_31", "run_31_t6", "move_to_position", "completed").
depends("run_31", "run_31_t6", "run_31_l8").
spawned_by("run_31", "run_31_t6", "run_31_l8").
param("run_31", "run_31_t6", "pos_distance", "2.13").
param("run_31", "run_31_t6", "delta_gripper", "0.0").
param("run_31", "run_31_t6", "efficiency", "0.7").
param("run_31", "run_31_t6", "mean_speed", "0.259").
call("run_31", "run_31_l9", "llm_generate", "completed").
depends("run_31", "run_31_l9", "run_31_t6").
call("run_31", "run_31_t7", "move_to_position", "completed").
depends("run_31", "run_31_t7", "run_31_l9").
spawned_by("run_31", "run_31_t7", "run_31_l9").
param("run_31", "run_31_t7", "pos_distance", "0.69").
param("run_31", "run_31_t7", "delta_gripper", "1.3").
param("run_31", "run_31_t7", "efficiency", "1.0").
param("run_31", "run_31_t7", "mean_speed", "0.238").
call("run_31", "run_31_l10", "llm_generate", "completed").
depends("run_31", "run_31_l10", "run_31_t7").
call("run_31", "run_31_t8", "open_gripper", "completed").
depends("run_31", "run_31_t8", "run_31_l10").
spawned_by("run_31", "run_31_t8", "run_31_l10").
param("run_31", "run_31_t8", "pos_distance", "0.43").
param("run_31", "run_31_t8", "delta_gripper", "1.6").
param("run_31", "run_31_t8", "efficiency", "0.8").
param("run_31", "run_31_t8", "mean_speed", "0.176").
call("run_31", "run_31_l11", "llm_generate", "completed").
depends("run_31", "run_31_l11", "run_31_t8").
call("run_31", "run_31_t9", "adjust_position", "completed").
depends("run_31", "run_31_t9", "run_31_l11").
spawned_by("run_31", "run_31_t9", "run_31_l11").
param("run_31", "run_31_t9", "pos_distance", "0.25").
param("run_31", "run_31_t9", "delta_gripper", "0.8").
param("run_31", "run_31_t9", "efficiency", "0.6").
param("run_31", "run_31_t9", "mean_speed", "0.133").
call("run_31", "run_31_l12", "llm_generate", "completed").
depends("run_31", "run_31_l12", "run_31_t9").
call("run_31", "run_31_t10", "adjust_position", "completed").
depends("run_31", "run_31_t10", "run_31_l12").
spawned_by("run_31", "run_31_t10", "run_31_l12").
param("run_31", "run_31_t10", "pos_distance", "0.35").
param("run_31", "run_31_t10", "delta_gripper", "0.0").
param("run_31", "run_31_t10", "efficiency", "0.3").
param("run_31", "run_31_t10", "mean_speed", "0.058").
call("run_31", "run_31_l13", "llm_generate", "completed").
depends("run_31", "run_31_l13", "run_31_t10").
call("run_31", "run_31_n", "notify_user", "completed").
depends("run_31", "run_31_n", "run_31_l13").
spawned_by("run_31", "run_31_n", "run_31_l13").

% === Run 32 (grasp_retry) ===
job("run_32").
job_status("run_32", "completed").
call("run_32", "run_32_s0", "prompt_user", "completed").
call("run_32", "run_32_s1", "llm_generate", "completed").
depends("run_32", "run_32_s1", "run_32_s0").
call("run_32", "run_32_l2", "llm_generate", "completed").
depends("run_32", "run_32_l2", "run_32_s1").
call("run_32", "run_32_t0", "move_to_position", "completed").
depends("run_32", "run_32_t0", "run_32_l2").
spawned_by("run_32", "run_32_t0", "run_32_l2").
param("run_32", "run_32_t0", "pos_distance", "0.78").
param("run_32", "run_32_t0", "delta_gripper", "0.0").
param("run_32", "run_32_t0", "efficiency", "1.0").
param("run_32", "run_32_t0", "mean_speed", "0.202").
call("run_32", "run_32_l3", "llm_generate", "completed").
depends("run_32", "run_32_l3", "run_32_t0").
call("run_32", "run_32_t1", "close_gripper", "completed").
depends("run_32", "run_32_t1", "run_32_l3").
spawned_by("run_32", "run_32_t1", "run_32_l3").
param("run_32", "run_32_t1", "pos_distance", "0.46").
param("run_32", "run_32_t1", "delta_gripper", "-2.0").
param("run_32", "run_32_t1", "efficiency", "0.9").
param("run_32", "run_32_t1", "mean_speed", "0.165").
call("run_32", "run_32_l4", "llm_generate", "completed").
depends("run_32", "run_32_l4", "run_32_t1").
call("run_32", "run_32_t2", "open_gripper", "completed").
depends("run_32", "run_32_t2", "run_32_l4").
spawned_by("run_32", "run_32_t2", "run_32_l4").
param("run_32", "run_32_t2", "pos_distance", "0.05").
param("run_32", "run_32_t2", "delta_gripper", "3.0").
param("run_32", "run_32_t2", "efficiency", "0.3").
param("run_32", "run_32_t2", "mean_speed", "0.080").
call("run_32", "run_32_l5", "llm_generate", "completed").
depends("run_32", "run_32_l5", "run_32_t2").
call("run_32", "run_32_t3", "adjust_position", "completed").
depends("run_32", "run_32_t3", "run_32_l5").
spawned_by("run_32", "run_32_t3", "run_32_l5").
param("run_32", "run_32_t3", "pos_distance", "0.15").
param("run_32", "run_32_t3", "delta_gripper", "0.0").
param("run_32", "run_32_t3", "efficiency", "0.5").
param("run_32", "run_32_t3", "mean_speed", "0.100").
call("run_32", "run_32_l6", "llm_generate", "completed").
depends("run_32", "run_32_l6", "run_32_t3").
call("run_32", "run_32_t4", "close_gripper", "completed").
depends("run_32", "run_32_t4", "run_32_l6").
spawned_by("run_32", "run_32_t4", "run_32_l6").
param("run_32", "run_32_t4", "pos_distance", "0.46").
param("run_32", "run_32_t4", "delta_gripper", "-2.0").
param("run_32", "run_32_t4", "efficiency", "0.9").
param("run_32", "run_32_t4", "mean_speed", "0.165").
call("run_32", "run_32_l7", "llm_generate", "completed").
depends("run_32", "run_32_l7", "run_32_t4").
call("run_32", "run_32_t5", "close_gripper", "completed").
depends("run_32", "run_32_t5", "run_32_l7").
spawned_by("run_32", "run_32_t5", "run_32_l7").
param("run_32", "run_32_t5", "pos_distance", "0.36").
param("run_32", "run_32_t5", "delta_gripper", "-1.8").
param("run_32", "run_32_t5", "efficiency", "1.0").
param("run_32", "run_32_t5", "mean_speed", "0.123").
call("run_32", "run_32_l8", "llm_generate", "completed").
depends("run_32", "run_32_l8", "run_32_t5").
call("run_32", "run_32_t6", "move_to_position", "completed").
depends("run_32", "run_32_t6", "run_32_l8").
spawned_by("run_32", "run_32_t6", "run_32_l8").
param("run_32", "run_32_t6", "pos_distance", "2.13").
param("run_32", "run_32_t6", "delta_gripper", "0.0").
param("run_32", "run_32_t6", "efficiency", "0.7").
param("run_32", "run_32_t6", "mean_speed", "0.259").
call("run_32", "run_32_l9", "llm_generate", "completed").
depends("run_32", "run_32_l9", "run_32_t6").
call("run_32", "run_32_t7", "move_to_position", "completed").
depends("run_32", "run_32_t7", "run_32_l9").
spawned_by("run_32", "run_32_t7", "run_32_l9").
param("run_32", "run_32_t7", "pos_distance", "0.69").
param("run_32", "run_32_t7", "delta_gripper", "1.3").
param("run_32", "run_32_t7", "efficiency", "1.0").
param("run_32", "run_32_t7", "mean_speed", "0.238").
call("run_32", "run_32_l10", "llm_generate", "completed").
depends("run_32", "run_32_l10", "run_32_t7").
call("run_32", "run_32_t8", "open_gripper", "completed").
depends("run_32", "run_32_t8", "run_32_l10").
spawned_by("run_32", "run_32_t8", "run_32_l10").
param("run_32", "run_32_t8", "pos_distance", "0.43").
param("run_32", "run_32_t8", "delta_gripper", "1.6").
param("run_32", "run_32_t8", "efficiency", "0.8").
param("run_32", "run_32_t8", "mean_speed", "0.176").
call("run_32", "run_32_l11", "llm_generate", "completed").
depends("run_32", "run_32_l11", "run_32_t8").
call("run_32", "run_32_t9", "adjust_position", "completed").
depends("run_32", "run_32_t9", "run_32_l11").
spawned_by("run_32", "run_32_t9", "run_32_l11").
param("run_32", "run_32_t9", "pos_distance", "0.25").
param("run_32", "run_32_t9", "delta_gripper", "0.8").
param("run_32", "run_32_t9", "efficiency", "0.6").
param("run_32", "run_32_t9", "mean_speed", "0.133").
call("run_32", "run_32_l12", "llm_generate", "completed").
depends("run_32", "run_32_l12", "run_32_t9").
call("run_32", "run_32_t10", "adjust_position", "completed").
depends("run_32", "run_32_t10", "run_32_l12").
spawned_by("run_32", "run_32_t10", "run_32_l12").
param("run_32", "run_32_t10", "pos_distance", "0.35").
param("run_32", "run_32_t10", "delta_gripper", "0.0").
param("run_32", "run_32_t10", "efficiency", "0.3").
param("run_32", "run_32_t10", "mean_speed", "0.058").
call("run_32", "run_32_l13", "llm_generate", "completed").
depends("run_32", "run_32_l13", "run_32_t10").
call("run_32", "run_32_n", "notify_user", "completed").
depends("run_32", "run_32_n", "run_32_l13").
spawned_by("run_32", "run_32_n", "run_32_l13").

% === Run 33 (grasp_retry) ===
job("run_33").
job_status("run_33", "completed").
call("run_33", "run_33_s0", "prompt_user", "completed").
call("run_33", "run_33_s1", "llm_generate", "completed").
depends("run_33", "run_33_s1", "run_33_s0").
call("run_33", "run_33_l2", "llm_generate", "completed").
depends("run_33", "run_33_l2", "run_33_s1").
call("run_33", "run_33_t0", "move_to_position", "completed").
depends("run_33", "run_33_t0", "run_33_l2").
spawned_by("run_33", "run_33_t0", "run_33_l2").
param("run_33", "run_33_t0", "pos_distance", "0.78").
param("run_33", "run_33_t0", "delta_gripper", "0.0").
param("run_33", "run_33_t0", "efficiency", "1.0").
param("run_33", "run_33_t0", "mean_speed", "0.202").
call("run_33", "run_33_l3", "llm_generate", "completed").
depends("run_33", "run_33_l3", "run_33_t0").
call("run_33", "run_33_t1", "close_gripper", "completed").
depends("run_33", "run_33_t1", "run_33_l3").
spawned_by("run_33", "run_33_t1", "run_33_l3").
param("run_33", "run_33_t1", "pos_distance", "0.46").
param("run_33", "run_33_t1", "delta_gripper", "-2.0").
param("run_33", "run_33_t1", "efficiency", "0.9").
param("run_33", "run_33_t1", "mean_speed", "0.165").
call("run_33", "run_33_l4", "llm_generate", "completed").
depends("run_33", "run_33_l4", "run_33_t1").
call("run_33", "run_33_t2", "open_gripper", "completed").
depends("run_33", "run_33_t2", "run_33_l4").
spawned_by("run_33", "run_33_t2", "run_33_l4").
param("run_33", "run_33_t2", "pos_distance", "0.05").
param("run_33", "run_33_t2", "delta_gripper", "3.0").
param("run_33", "run_33_t2", "efficiency", "0.3").
param("run_33", "run_33_t2", "mean_speed", "0.080").
call("run_33", "run_33_l5", "llm_generate", "completed").
depends("run_33", "run_33_l5", "run_33_t2").
call("run_33", "run_33_t3", "adjust_position", "completed").
depends("run_33", "run_33_t3", "run_33_l5").
spawned_by("run_33", "run_33_t3", "run_33_l5").
param("run_33", "run_33_t3", "pos_distance", "0.15").
param("run_33", "run_33_t3", "delta_gripper", "0.0").
param("run_33", "run_33_t3", "efficiency", "0.5").
param("run_33", "run_33_t3", "mean_speed", "0.100").
call("run_33", "run_33_l6", "llm_generate", "completed").
depends("run_33", "run_33_l6", "run_33_t3").
call("run_33", "run_33_t4", "close_gripper", "completed").
depends("run_33", "run_33_t4", "run_33_l6").
spawned_by("run_33", "run_33_t4", "run_33_l6").
param("run_33", "run_33_t4", "pos_distance", "0.46").
param("run_33", "run_33_t4", "delta_gripper", "-2.0").
param("run_33", "run_33_t4", "efficiency", "0.9").
param("run_33", "run_33_t4", "mean_speed", "0.165").
call("run_33", "run_33_l7", "llm_generate", "completed").
depends("run_33", "run_33_l7", "run_33_t4").
call("run_33", "run_33_t5", "close_gripper", "completed").
depends("run_33", "run_33_t5", "run_33_l7").
spawned_by("run_33", "run_33_t5", "run_33_l7").
param("run_33", "run_33_t5", "pos_distance", "0.36").
param("run_33", "run_33_t5", "delta_gripper", "-1.8").
param("run_33", "run_33_t5", "efficiency", "1.0").
param("run_33", "run_33_t5", "mean_speed", "0.123").
call("run_33", "run_33_l8", "llm_generate", "completed").
depends("run_33", "run_33_l8", "run_33_t5").
call("run_33", "run_33_t6", "move_to_position", "completed").
depends("run_33", "run_33_t6", "run_33_l8").
spawned_by("run_33", "run_33_t6", "run_33_l8").
param("run_33", "run_33_t6", "pos_distance", "2.13").
param("run_33", "run_33_t6", "delta_gripper", "0.0").
param("run_33", "run_33_t6", "efficiency", "0.7").
param("run_33", "run_33_t6", "mean_speed", "0.259").
call("run_33", "run_33_l9", "llm_generate", "completed").
depends("run_33", "run_33_l9", "run_33_t6").
call("run_33", "run_33_t7", "move_to_position", "completed").
depends("run_33", "run_33_t7", "run_33_l9").
spawned_by("run_33", "run_33_t7", "run_33_l9").
param("run_33", "run_33_t7", "pos_distance", "0.69").
param("run_33", "run_33_t7", "delta_gripper", "1.3").
param("run_33", "run_33_t7", "efficiency", "1.0").
param("run_33", "run_33_t7", "mean_speed", "0.238").
call("run_33", "run_33_l10", "llm_generate", "completed").
depends("run_33", "run_33_l10", "run_33_t7").
call("run_33", "run_33_t8", "open_gripper", "completed").
depends("run_33", "run_33_t8", "run_33_l10").
spawned_by("run_33", "run_33_t8", "run_33_l10").
param("run_33", "run_33_t8", "pos_distance", "0.43").
param("run_33", "run_33_t8", "delta_gripper", "1.6").
param("run_33", "run_33_t8", "efficiency", "0.8").
param("run_33", "run_33_t8", "mean_speed", "0.176").
call("run_33", "run_33_l11", "llm_generate", "completed").
depends("run_33", "run_33_l11", "run_33_t8").
call("run_33", "run_33_t9", "adjust_position", "completed").
depends("run_33", "run_33_t9", "run_33_l11").
spawned_by("run_33", "run_33_t9", "run_33_l11").
param("run_33", "run_33_t9", "pos_distance", "0.25").
param("run_33", "run_33_t9", "delta_gripper", "0.8").
param("run_33", "run_33_t9", "efficiency", "0.6").
param("run_33", "run_33_t9", "mean_speed", "0.133").
call("run_33", "run_33_l12", "llm_generate", "completed").
depends("run_33", "run_33_l12", "run_33_t9").
call("run_33", "run_33_t10", "adjust_position", "completed").
depends("run_33", "run_33_t10", "run_33_l12").
spawned_by("run_33", "run_33_t10", "run_33_l12").
param("run_33", "run_33_t10", "pos_distance", "0.35").
param("run_33", "run_33_t10", "delta_gripper", "0.0").
param("run_33", "run_33_t10", "efficiency", "0.3").
param("run_33", "run_33_t10", "mean_speed", "0.058").
call("run_33", "run_33_l13", "llm_generate", "completed").
depends("run_33", "run_33_l13", "run_33_t10").
call("run_33", "run_33_n", "notify_user", "completed").
depends("run_33", "run_33_n", "run_33_l13").
spawned_by("run_33", "run_33_n", "run_33_l13").

% === Run 34 (grasp_retry) ===
job("run_34").
job_status("run_34", "completed").
call("run_34", "run_34_s0", "prompt_user", "completed").
call("run_34", "run_34_s1", "llm_generate", "completed").
depends("run_34", "run_34_s1", "run_34_s0").
call("run_34", "run_34_l2", "llm_generate", "completed").
depends("run_34", "run_34_l2", "run_34_s1").
call("run_34", "run_34_t0", "move_to_position", "completed").
depends("run_34", "run_34_t0", "run_34_l2").
spawned_by("run_34", "run_34_t0", "run_34_l2").
param("run_34", "run_34_t0", "pos_distance", "0.78").
param("run_34", "run_34_t0", "delta_gripper", "0.0").
param("run_34", "run_34_t0", "efficiency", "1.0").
param("run_34", "run_34_t0", "mean_speed", "0.202").
call("run_34", "run_34_l3", "llm_generate", "completed").
depends("run_34", "run_34_l3", "run_34_t0").
call("run_34", "run_34_t1", "close_gripper", "completed").
depends("run_34", "run_34_t1", "run_34_l3").
spawned_by("run_34", "run_34_t1", "run_34_l3").
param("run_34", "run_34_t1", "pos_distance", "0.46").
param("run_34", "run_34_t1", "delta_gripper", "-2.0").
param("run_34", "run_34_t1", "efficiency", "0.9").
param("run_34", "run_34_t1", "mean_speed", "0.165").
call("run_34", "run_34_l4", "llm_generate", "completed").
depends("run_34", "run_34_l4", "run_34_t1").
call("run_34", "run_34_t2", "open_gripper", "completed").
depends("run_34", "run_34_t2", "run_34_l4").
spawned_by("run_34", "run_34_t2", "run_34_l4").
param("run_34", "run_34_t2", "pos_distance", "0.05").
param("run_34", "run_34_t2", "delta_gripper", "3.0").
param("run_34", "run_34_t2", "efficiency", "0.3").
param("run_34", "run_34_t2", "mean_speed", "0.080").
call("run_34", "run_34_l5", "llm_generate", "completed").
depends("run_34", "run_34_l5", "run_34_t2").
call("run_34", "run_34_t3", "adjust_position", "completed").
depends("run_34", "run_34_t3", "run_34_l5").
spawned_by("run_34", "run_34_t3", "run_34_l5").
param("run_34", "run_34_t3", "pos_distance", "0.15").
param("run_34", "run_34_t3", "delta_gripper", "0.0").
param("run_34", "run_34_t3", "efficiency", "0.5").
param("run_34", "run_34_t3", "mean_speed", "0.100").
call("run_34", "run_34_l6", "llm_generate", "completed").
depends("run_34", "run_34_l6", "run_34_t3").
call("run_34", "run_34_t4", "close_gripper", "completed").
depends("run_34", "run_34_t4", "run_34_l6").
spawned_by("run_34", "run_34_t4", "run_34_l6").
param("run_34", "run_34_t4", "pos_distance", "0.46").
param("run_34", "run_34_t4", "delta_gripper", "-2.0").
param("run_34", "run_34_t4", "efficiency", "0.9").
param("run_34", "run_34_t4", "mean_speed", "0.165").
call("run_34", "run_34_l7", "llm_generate", "completed").
depends("run_34", "run_34_l7", "run_34_t4").
call("run_34", "run_34_t5", "close_gripper", "completed").
depends("run_34", "run_34_t5", "run_34_l7").
spawned_by("run_34", "run_34_t5", "run_34_l7").
param("run_34", "run_34_t5", "pos_distance", "0.36").
param("run_34", "run_34_t5", "delta_gripper", "-1.8").
param("run_34", "run_34_t5", "efficiency", "1.0").
param("run_34", "run_34_t5", "mean_speed", "0.123").
call("run_34", "run_34_l8", "llm_generate", "completed").
depends("run_34", "run_34_l8", "run_34_t5").
call("run_34", "run_34_t6", "move_to_position", "completed").
depends("run_34", "run_34_t6", "run_34_l8").
spawned_by("run_34", "run_34_t6", "run_34_l8").
param("run_34", "run_34_t6", "pos_distance", "2.13").
param("run_34", "run_34_t6", "delta_gripper", "0.0").
param("run_34", "run_34_t6", "efficiency", "0.7").
param("run_34", "run_34_t6", "mean_speed", "0.259").
call("run_34", "run_34_l9", "llm_generate", "completed").
depends("run_34", "run_34_l9", "run_34_t6").
call("run_34", "run_34_t7", "move_to_position", "completed").
depends("run_34", "run_34_t7", "run_34_l9").
spawned_by("run_34", "run_34_t7", "run_34_l9").
param("run_34", "run_34_t7", "pos_distance", "0.69").
param("run_34", "run_34_t7", "delta_gripper", "1.3").
param("run_34", "run_34_t7", "efficiency", "1.0").
param("run_34", "run_34_t7", "mean_speed", "0.238").
call("run_34", "run_34_l10", "llm_generate", "completed").
depends("run_34", "run_34_l10", "run_34_t7").
call("run_34", "run_34_t8", "open_gripper", "completed").
depends("run_34", "run_34_t8", "run_34_l10").
spawned_by("run_34", "run_34_t8", "run_34_l10").
param("run_34", "run_34_t8", "pos_distance", "0.43").
param("run_34", "run_34_t8", "delta_gripper", "1.6").
param("run_34", "run_34_t8", "efficiency", "0.8").
param("run_34", "run_34_t8", "mean_speed", "0.176").
call("run_34", "run_34_l11", "llm_generate", "completed").
depends("run_34", "run_34_l11", "run_34_t8").
call("run_34", "run_34_t9", "adjust_position", "completed").
depends("run_34", "run_34_t9", "run_34_l11").
spawned_by("run_34", "run_34_t9", "run_34_l11").
param("run_34", "run_34_t9", "pos_distance", "0.25").
param("run_34", "run_34_t9", "delta_gripper", "0.8").
param("run_34", "run_34_t9", "efficiency", "0.6").
param("run_34", "run_34_t9", "mean_speed", "0.133").
call("run_34", "run_34_l12", "llm_generate", "completed").
depends("run_34", "run_34_l12", "run_34_t9").
call("run_34", "run_34_t10", "adjust_position", "completed").
depends("run_34", "run_34_t10", "run_34_l12").
spawned_by("run_34", "run_34_t10", "run_34_l12").
param("run_34", "run_34_t10", "pos_distance", "0.35").
param("run_34", "run_34_t10", "delta_gripper", "0.0").
param("run_34", "run_34_t10", "efficiency", "0.3").
param("run_34", "run_34_t10", "mean_speed", "0.058").
call("run_34", "run_34_l13", "llm_generate", "completed").
depends("run_34", "run_34_l13", "run_34_t10").
call("run_34", "run_34_n", "notify_user", "completed").
depends("run_34", "run_34_n", "run_34_l13").
spawned_by("run_34", "run_34_n", "run_34_l13").

% === Run 35 (grasp_retry) ===
job("run_35").
job_status("run_35", "completed").
call("run_35", "run_35_s0", "prompt_user", "completed").
call("run_35", "run_35_s1", "llm_generate", "completed").
depends("run_35", "run_35_s1", "run_35_s0").
call("run_35", "run_35_l2", "llm_generate", "completed").
depends("run_35", "run_35_l2", "run_35_s1").
call("run_35", "run_35_t0", "move_to_position", "completed").
depends("run_35", "run_35_t0", "run_35_l2").
spawned_by("run_35", "run_35_t0", "run_35_l2").
param("run_35", "run_35_t0", "pos_distance", "0.78").
param("run_35", "run_35_t0", "delta_gripper", "0.0").
param("run_35", "run_35_t0", "efficiency", "1.0").
param("run_35", "run_35_t0", "mean_speed", "0.202").
call("run_35", "run_35_l3", "llm_generate", "completed").
depends("run_35", "run_35_l3", "run_35_t0").
call("run_35", "run_35_t1", "close_gripper", "completed").
depends("run_35", "run_35_t1", "run_35_l3").
spawned_by("run_35", "run_35_t1", "run_35_l3").
param("run_35", "run_35_t1", "pos_distance", "0.46").
param("run_35", "run_35_t1", "delta_gripper", "-2.0").
param("run_35", "run_35_t1", "efficiency", "0.9").
param("run_35", "run_35_t1", "mean_speed", "0.165").
call("run_35", "run_35_l4", "llm_generate", "completed").
depends("run_35", "run_35_l4", "run_35_t1").
call("run_35", "run_35_t2", "open_gripper", "completed").
depends("run_35", "run_35_t2", "run_35_l4").
spawned_by("run_35", "run_35_t2", "run_35_l4").
param("run_35", "run_35_t2", "pos_distance", "0.05").
param("run_35", "run_35_t2", "delta_gripper", "3.0").
param("run_35", "run_35_t2", "efficiency", "0.3").
param("run_35", "run_35_t2", "mean_speed", "0.080").
call("run_35", "run_35_l5", "llm_generate", "completed").
depends("run_35", "run_35_l5", "run_35_t2").
call("run_35", "run_35_t3", "adjust_position", "completed").
depends("run_35", "run_35_t3", "run_35_l5").
spawned_by("run_35", "run_35_t3", "run_35_l5").
param("run_35", "run_35_t3", "pos_distance", "0.15").
param("run_35", "run_35_t3", "delta_gripper", "0.0").
param("run_35", "run_35_t3", "efficiency", "0.5").
param("run_35", "run_35_t3", "mean_speed", "0.100").
call("run_35", "run_35_l6", "llm_generate", "completed").
depends("run_35", "run_35_l6", "run_35_t3").
call("run_35", "run_35_t4", "close_gripper", "completed").
depends("run_35", "run_35_t4", "run_35_l6").
spawned_by("run_35", "run_35_t4", "run_35_l6").
param("run_35", "run_35_t4", "pos_distance", "0.46").
param("run_35", "run_35_t4", "delta_gripper", "-2.0").
param("run_35", "run_35_t4", "efficiency", "0.9").
param("run_35", "run_35_t4", "mean_speed", "0.165").
call("run_35", "run_35_l7", "llm_generate", "completed").
depends("run_35", "run_35_l7", "run_35_t4").
call("run_35", "run_35_t5", "close_gripper", "completed").
depends("run_35", "run_35_t5", "run_35_l7").
spawned_by("run_35", "run_35_t5", "run_35_l7").
param("run_35", "run_35_t5", "pos_distance", "0.36").
param("run_35", "run_35_t5", "delta_gripper", "-1.8").
param("run_35", "run_35_t5", "efficiency", "1.0").
param("run_35", "run_35_t5", "mean_speed", "0.123").
call("run_35", "run_35_l8", "llm_generate", "completed").
depends("run_35", "run_35_l8", "run_35_t5").
call("run_35", "run_35_t6", "move_to_position", "completed").
depends("run_35", "run_35_t6", "run_35_l8").
spawned_by("run_35", "run_35_t6", "run_35_l8").
param("run_35", "run_35_t6", "pos_distance", "2.13").
param("run_35", "run_35_t6", "delta_gripper", "0.0").
param("run_35", "run_35_t6", "efficiency", "0.7").
param("run_35", "run_35_t6", "mean_speed", "0.259").
call("run_35", "run_35_l9", "llm_generate", "completed").
depends("run_35", "run_35_l9", "run_35_t6").
call("run_35", "run_35_t7", "move_to_position", "completed").
depends("run_35", "run_35_t7", "run_35_l9").
spawned_by("run_35", "run_35_t7", "run_35_l9").
param("run_35", "run_35_t7", "pos_distance", "0.69").
param("run_35", "run_35_t7", "delta_gripper", "1.3").
param("run_35", "run_35_t7", "efficiency", "1.0").
param("run_35", "run_35_t7", "mean_speed", "0.238").
call("run_35", "run_35_l10", "llm_generate", "completed").
depends("run_35", "run_35_l10", "run_35_t7").
call("run_35", "run_35_t8", "open_gripper", "completed").
depends("run_35", "run_35_t8", "run_35_l10").
spawned_by("run_35", "run_35_t8", "run_35_l10").
param("run_35", "run_35_t8", "pos_distance", "0.43").
param("run_35", "run_35_t8", "delta_gripper", "1.6").
param("run_35", "run_35_t8", "efficiency", "0.8").
param("run_35", "run_35_t8", "mean_speed", "0.176").
call("run_35", "run_35_l11", "llm_generate", "completed").
depends("run_35", "run_35_l11", "run_35_t8").
call("run_35", "run_35_t9", "adjust_position", "completed").
depends("run_35", "run_35_t9", "run_35_l11").
spawned_by("run_35", "run_35_t9", "run_35_l11").
param("run_35", "run_35_t9", "pos_distance", "0.25").
param("run_35", "run_35_t9", "delta_gripper", "0.8").
param("run_35", "run_35_t9", "efficiency", "0.6").
param("run_35", "run_35_t9", "mean_speed", "0.133").
call("run_35", "run_35_l12", "llm_generate", "completed").
depends("run_35", "run_35_l12", "run_35_t9").
call("run_35", "run_35_t10", "adjust_position", "completed").
depends("run_35", "run_35_t10", "run_35_l12").
spawned_by("run_35", "run_35_t10", "run_35_l12").
param("run_35", "run_35_t10", "pos_distance", "0.35").
param("run_35", "run_35_t10", "delta_gripper", "0.0").
param("run_35", "run_35_t10", "efficiency", "0.3").
param("run_35", "run_35_t10", "mean_speed", "0.058").
call("run_35", "run_35_l13", "llm_generate", "completed").
depends("run_35", "run_35_l13", "run_35_t10").
call("run_35", "run_35_n", "notify_user", "completed").
depends("run_35", "run_35_n", "run_35_l13").
spawned_by("run_35", "run_35_n", "run_35_l13").

% === Run 36 (grasp_retry) ===
job("run_36").
job_status("run_36", "completed").
call("run_36", "run_36_s0", "prompt_user", "completed").
call("run_36", "run_36_s1", "llm_generate", "completed").
depends("run_36", "run_36_s1", "run_36_s0").
call("run_36", "run_36_l2", "llm_generate", "completed").
depends("run_36", "run_36_l2", "run_36_s1").
call("run_36", "run_36_t0", "move_to_position", "completed").
depends("run_36", "run_36_t0", "run_36_l2").
spawned_by("run_36", "run_36_t0", "run_36_l2").
param("run_36", "run_36_t0", "pos_distance", "0.78").
param("run_36", "run_36_t0", "delta_gripper", "0.0").
param("run_36", "run_36_t0", "efficiency", "1.0").
param("run_36", "run_36_t0", "mean_speed", "0.202").
call("run_36", "run_36_l3", "llm_generate", "completed").
depends("run_36", "run_36_l3", "run_36_t0").
call("run_36", "run_36_t1", "close_gripper", "completed").
depends("run_36", "run_36_t1", "run_36_l3").
spawned_by("run_36", "run_36_t1", "run_36_l3").
param("run_36", "run_36_t1", "pos_distance", "0.46").
param("run_36", "run_36_t1", "delta_gripper", "-2.0").
param("run_36", "run_36_t1", "efficiency", "0.9").
param("run_36", "run_36_t1", "mean_speed", "0.165").
call("run_36", "run_36_l4", "llm_generate", "completed").
depends("run_36", "run_36_l4", "run_36_t1").
call("run_36", "run_36_t2", "open_gripper", "completed").
depends("run_36", "run_36_t2", "run_36_l4").
spawned_by("run_36", "run_36_t2", "run_36_l4").
param("run_36", "run_36_t2", "pos_distance", "0.05").
param("run_36", "run_36_t2", "delta_gripper", "3.0").
param("run_36", "run_36_t2", "efficiency", "0.3").
param("run_36", "run_36_t2", "mean_speed", "0.080").
call("run_36", "run_36_l5", "llm_generate", "completed").
depends("run_36", "run_36_l5", "run_36_t2").
call("run_36", "run_36_t3", "adjust_position", "completed").
depends("run_36", "run_36_t3", "run_36_l5").
spawned_by("run_36", "run_36_t3", "run_36_l5").
param("run_36", "run_36_t3", "pos_distance", "0.15").
param("run_36", "run_36_t3", "delta_gripper", "0.0").
param("run_36", "run_36_t3", "efficiency", "0.5").
param("run_36", "run_36_t3", "mean_speed", "0.100").
call("run_36", "run_36_l6", "llm_generate", "completed").
depends("run_36", "run_36_l6", "run_36_t3").
call("run_36", "run_36_t4", "close_gripper", "completed").
depends("run_36", "run_36_t4", "run_36_l6").
spawned_by("run_36", "run_36_t4", "run_36_l6").
param("run_36", "run_36_t4", "pos_distance", "0.46").
param("run_36", "run_36_t4", "delta_gripper", "-2.0").
param("run_36", "run_36_t4", "efficiency", "0.9").
param("run_36", "run_36_t4", "mean_speed", "0.165").
call("run_36", "run_36_l7", "llm_generate", "completed").
depends("run_36", "run_36_l7", "run_36_t4").
call("run_36", "run_36_t5", "close_gripper", "completed").
depends("run_36", "run_36_t5", "run_36_l7").
spawned_by("run_36", "run_36_t5", "run_36_l7").
param("run_36", "run_36_t5", "pos_distance", "0.36").
param("run_36", "run_36_t5", "delta_gripper", "-1.8").
param("run_36", "run_36_t5", "efficiency", "1.0").
param("run_36", "run_36_t5", "mean_speed", "0.123").
call("run_36", "run_36_l8", "llm_generate", "completed").
depends("run_36", "run_36_l8", "run_36_t5").
call("run_36", "run_36_t6", "move_to_position", "completed").
depends("run_36", "run_36_t6", "run_36_l8").
spawned_by("run_36", "run_36_t6", "run_36_l8").
param("run_36", "run_36_t6", "pos_distance", "2.13").
param("run_36", "run_36_t6", "delta_gripper", "0.0").
param("run_36", "run_36_t6", "efficiency", "0.7").
param("run_36", "run_36_t6", "mean_speed", "0.259").
call("run_36", "run_36_l9", "llm_generate", "completed").
depends("run_36", "run_36_l9", "run_36_t6").
call("run_36", "run_36_t7", "move_to_position", "completed").
depends("run_36", "run_36_t7", "run_36_l9").
spawned_by("run_36", "run_36_t7", "run_36_l9").
param("run_36", "run_36_t7", "pos_distance", "0.69").
param("run_36", "run_36_t7", "delta_gripper", "1.3").
param("run_36", "run_36_t7", "efficiency", "1.0").
param("run_36", "run_36_t7", "mean_speed", "0.238").
call("run_36", "run_36_l10", "llm_generate", "completed").
depends("run_36", "run_36_l10", "run_36_t7").
call("run_36", "run_36_t8", "open_gripper", "completed").
depends("run_36", "run_36_t8", "run_36_l10").
spawned_by("run_36", "run_36_t8", "run_36_l10").
param("run_36", "run_36_t8", "pos_distance", "0.43").
param("run_36", "run_36_t8", "delta_gripper", "1.6").
param("run_36", "run_36_t8", "efficiency", "0.8").
param("run_36", "run_36_t8", "mean_speed", "0.176").
call("run_36", "run_36_l11", "llm_generate", "completed").
depends("run_36", "run_36_l11", "run_36_t8").
call("run_36", "run_36_t9", "adjust_position", "completed").
depends("run_36", "run_36_t9", "run_36_l11").
spawned_by("run_36", "run_36_t9", "run_36_l11").
param("run_36", "run_36_t9", "pos_distance", "0.25").
param("run_36", "run_36_t9", "delta_gripper", "0.8").
param("run_36", "run_36_t9", "efficiency", "0.6").
param("run_36", "run_36_t9", "mean_speed", "0.133").
call("run_36", "run_36_l12", "llm_generate", "completed").
depends("run_36", "run_36_l12", "run_36_t9").
call("run_36", "run_36_t10", "adjust_position", "completed").
depends("run_36", "run_36_t10", "run_36_l12").
spawned_by("run_36", "run_36_t10", "run_36_l12").
param("run_36", "run_36_t10", "pos_distance", "0.35").
param("run_36", "run_36_t10", "delta_gripper", "0.0").
param("run_36", "run_36_t10", "efficiency", "0.3").
param("run_36", "run_36_t10", "mean_speed", "0.058").
call("run_36", "run_36_l13", "llm_generate", "completed").
depends("run_36", "run_36_l13", "run_36_t10").
call("run_36", "run_36_n", "notify_user", "completed").
depends("run_36", "run_36_n", "run_36_l13").
spawned_by("run_36", "run_36_n", "run_36_l13").

% === Run 37 (grasp_retry) ===
job("run_37").
job_status("run_37", "completed").
call("run_37", "run_37_s0", "prompt_user", "completed").
call("run_37", "run_37_s1", "llm_generate", "completed").
depends("run_37", "run_37_s1", "run_37_s0").
call("run_37", "run_37_l2", "llm_generate", "completed").
depends("run_37", "run_37_l2", "run_37_s1").
call("run_37", "run_37_t0", "move_to_position", "completed").
depends("run_37", "run_37_t0", "run_37_l2").
spawned_by("run_37", "run_37_t0", "run_37_l2").
param("run_37", "run_37_t0", "pos_distance", "0.78").
param("run_37", "run_37_t0", "delta_gripper", "0.0").
param("run_37", "run_37_t0", "efficiency", "1.0").
param("run_37", "run_37_t0", "mean_speed", "0.202").
call("run_37", "run_37_l3", "llm_generate", "completed").
depends("run_37", "run_37_l3", "run_37_t0").
call("run_37", "run_37_t1", "close_gripper", "completed").
depends("run_37", "run_37_t1", "run_37_l3").
spawned_by("run_37", "run_37_t1", "run_37_l3").
param("run_37", "run_37_t1", "pos_distance", "0.46").
param("run_37", "run_37_t1", "delta_gripper", "-2.0").
param("run_37", "run_37_t1", "efficiency", "0.9").
param("run_37", "run_37_t1", "mean_speed", "0.165").
call("run_37", "run_37_l4", "llm_generate", "completed").
depends("run_37", "run_37_l4", "run_37_t1").
call("run_37", "run_37_t2", "open_gripper", "completed").
depends("run_37", "run_37_t2", "run_37_l4").
spawned_by("run_37", "run_37_t2", "run_37_l4").
param("run_37", "run_37_t2", "pos_distance", "0.05").
param("run_37", "run_37_t2", "delta_gripper", "3.0").
param("run_37", "run_37_t2", "efficiency", "0.3").
param("run_37", "run_37_t2", "mean_speed", "0.080").
call("run_37", "run_37_l5", "llm_generate", "completed").
depends("run_37", "run_37_l5", "run_37_t2").
call("run_37", "run_37_t3", "adjust_position", "completed").
depends("run_37", "run_37_t3", "run_37_l5").
spawned_by("run_37", "run_37_t3", "run_37_l5").
param("run_37", "run_37_t3", "pos_distance", "0.15").
param("run_37", "run_37_t3", "delta_gripper", "0.0").
param("run_37", "run_37_t3", "efficiency", "0.5").
param("run_37", "run_37_t3", "mean_speed", "0.100").
call("run_37", "run_37_l6", "llm_generate", "completed").
depends("run_37", "run_37_l6", "run_37_t3").
call("run_37", "run_37_t4", "close_gripper", "completed").
depends("run_37", "run_37_t4", "run_37_l6").
spawned_by("run_37", "run_37_t4", "run_37_l6").
param("run_37", "run_37_t4", "pos_distance", "0.46").
param("run_37", "run_37_t4", "delta_gripper", "-2.0").
param("run_37", "run_37_t4", "efficiency", "0.9").
param("run_37", "run_37_t4", "mean_speed", "0.165").
call("run_37", "run_37_l7", "llm_generate", "completed").
depends("run_37", "run_37_l7", "run_37_t4").
call("run_37", "run_37_t5", "close_gripper", "completed").
depends("run_37", "run_37_t5", "run_37_l7").
spawned_by("run_37", "run_37_t5", "run_37_l7").
param("run_37", "run_37_t5", "pos_distance", "0.36").
param("run_37", "run_37_t5", "delta_gripper", "-1.8").
param("run_37", "run_37_t5", "efficiency", "1.0").
param("run_37", "run_37_t5", "mean_speed", "0.123").
call("run_37", "run_37_l8", "llm_generate", "completed").
depends("run_37", "run_37_l8", "run_37_t5").
call("run_37", "run_37_t6", "move_to_position", "completed").
depends("run_37", "run_37_t6", "run_37_l8").
spawned_by("run_37", "run_37_t6", "run_37_l8").
param("run_37", "run_37_t6", "pos_distance", "2.13").
param("run_37", "run_37_t6", "delta_gripper", "0.0").
param("run_37", "run_37_t6", "efficiency", "0.7").
param("run_37", "run_37_t6", "mean_speed", "0.259").
call("run_37", "run_37_l9", "llm_generate", "completed").
depends("run_37", "run_37_l9", "run_37_t6").
call("run_37", "run_37_t7", "move_to_position", "completed").
depends("run_37", "run_37_t7", "run_37_l9").
spawned_by("run_37", "run_37_t7", "run_37_l9").
param("run_37", "run_37_t7", "pos_distance", "0.69").
param("run_37", "run_37_t7", "delta_gripper", "1.3").
param("run_37", "run_37_t7", "efficiency", "1.0").
param("run_37", "run_37_t7", "mean_speed", "0.238").
call("run_37", "run_37_l10", "llm_generate", "completed").
depends("run_37", "run_37_l10", "run_37_t7").
call("run_37", "run_37_t8", "open_gripper", "completed").
depends("run_37", "run_37_t8", "run_37_l10").
spawned_by("run_37", "run_37_t8", "run_37_l10").
param("run_37", "run_37_t8", "pos_distance", "0.43").
param("run_37", "run_37_t8", "delta_gripper", "1.6").
param("run_37", "run_37_t8", "efficiency", "0.8").
param("run_37", "run_37_t8", "mean_speed", "0.176").
call("run_37", "run_37_l11", "llm_generate", "completed").
depends("run_37", "run_37_l11", "run_37_t8").
call("run_37", "run_37_t9", "adjust_position", "completed").
depends("run_37", "run_37_t9", "run_37_l11").
spawned_by("run_37", "run_37_t9", "run_37_l11").
param("run_37", "run_37_t9", "pos_distance", "0.25").
param("run_37", "run_37_t9", "delta_gripper", "0.8").
param("run_37", "run_37_t9", "efficiency", "0.6").
param("run_37", "run_37_t9", "mean_speed", "0.133").
call("run_37", "run_37_l12", "llm_generate", "completed").
depends("run_37", "run_37_l12", "run_37_t9").
call("run_37", "run_37_t10", "adjust_position", "completed").
depends("run_37", "run_37_t10", "run_37_l12").
spawned_by("run_37", "run_37_t10", "run_37_l12").
param("run_37", "run_37_t10", "pos_distance", "0.35").
param("run_37", "run_37_t10", "delta_gripper", "0.0").
param("run_37", "run_37_t10", "efficiency", "0.3").
param("run_37", "run_37_t10", "mean_speed", "0.058").
call("run_37", "run_37_l13", "llm_generate", "completed").
depends("run_37", "run_37_l13", "run_37_t10").
call("run_37", "run_37_n", "notify_user", "completed").
depends("run_37", "run_37_n", "run_37_l13").
spawned_by("run_37", "run_37_n", "run_37_l13").

% === Run 38 (object_slip) ===
job("run_38").
job_status("run_38", "completed").
call("run_38", "run_38_s0", "prompt_user", "completed").
call("run_38", "run_38_s1", "llm_generate", "completed").
depends("run_38", "run_38_s1", "run_38_s0").
call("run_38", "run_38_l2", "llm_generate", "completed").
depends("run_38", "run_38_l2", "run_38_s1").
call("run_38", "run_38_t0", "move_to_position", "completed").
depends("run_38", "run_38_t0", "run_38_l2").
spawned_by("run_38", "run_38_t0", "run_38_l2").
param("run_38", "run_38_t0", "pos_distance", "0.78").
param("run_38", "run_38_t0", "delta_gripper", "0.0").
param("run_38", "run_38_t0", "efficiency", "1.0").
param("run_38", "run_38_t0", "mean_speed", "0.202").
call("run_38", "run_38_l3", "llm_generate", "completed").
depends("run_38", "run_38_l3", "run_38_t0").
call("run_38", "run_38_t1", "close_gripper", "completed").
depends("run_38", "run_38_t1", "run_38_l3").
spawned_by("run_38", "run_38_t1", "run_38_l3").
param("run_38", "run_38_t1", "pos_distance", "0.46").
param("run_38", "run_38_t1", "delta_gripper", "-2.0").
param("run_38", "run_38_t1", "efficiency", "0.9").
param("run_38", "run_38_t1", "mean_speed", "0.165").
call("run_38", "run_38_l4", "llm_generate", "completed").
depends("run_38", "run_38_l4", "run_38_t1").
call("run_38", "run_38_t2", "close_gripper", "completed").
depends("run_38", "run_38_t2", "run_38_l4").
spawned_by("run_38", "run_38_t2", "run_38_l4").
param("run_38", "run_38_t2", "pos_distance", "0.36").
param("run_38", "run_38_t2", "delta_gripper", "-1.8").
param("run_38", "run_38_t2", "efficiency", "1.0").
param("run_38", "run_38_t2", "mean_speed", "0.123").
call("run_38", "run_38_l5", "llm_generate", "completed").
depends("run_38", "run_38_l5", "run_38_t2").
call("run_38", "run_38_t3", "move_to_position", "completed").
depends("run_38", "run_38_t3", "run_38_l5").
spawned_by("run_38", "run_38_t3", "run_38_l5").
param("run_38", "run_38_t3", "pos_distance", "2.13").
param("run_38", "run_38_t3", "delta_gripper", "0.0").
param("run_38", "run_38_t3", "efficiency", "0.7").
param("run_38", "run_38_t3", "mean_speed", "0.259").
call("run_38", "run_38_l6", "llm_generate", "completed").
depends("run_38", "run_38_l6", "run_38_t3").
call("run_38", "run_38_t4", "adjust_position", "completed").
depends("run_38", "run_38_t4", "run_38_l6").
spawned_by("run_38", "run_38_t4", "run_38_l6").
param("run_38", "run_38_t4", "pos_distance", "0.20").
param("run_38", "run_38_t4", "delta_gripper", "0.5").
param("run_38", "run_38_t4", "efficiency", "0.4").
param("run_38", "run_38_t4", "mean_speed", "0.150").
call("run_38", "run_38_l7", "llm_generate", "completed").
depends("run_38", "run_38_l7", "run_38_t4").
call("run_38", "run_38_t5", "close_gripper", "completed").
depends("run_38", "run_38_t5", "run_38_l7").
spawned_by("run_38", "run_38_t5", "run_38_l7").
param("run_38", "run_38_t5", "pos_distance", "0.10").
param("run_38", "run_38_t5", "delta_gripper", "-2.0").
param("run_38", "run_38_t5", "efficiency", "0.3").
param("run_38", "run_38_t5", "mean_speed", "0.080").
call("run_38", "run_38_l8", "llm_generate", "completed").
depends("run_38", "run_38_l8", "run_38_t5").
call("run_38", "run_38_t6", "move_to_position", "completed").
depends("run_38", "run_38_t6", "run_38_l8").
spawned_by("run_38", "run_38_t6", "run_38_l8").
param("run_38", "run_38_t6", "pos_distance", "2.13").
param("run_38", "run_38_t6", "delta_gripper", "0.0").
param("run_38", "run_38_t6", "efficiency", "0.7").
param("run_38", "run_38_t6", "mean_speed", "0.259").
call("run_38", "run_38_l9", "llm_generate", "completed").
depends("run_38", "run_38_l9", "run_38_t6").
call("run_38", "run_38_t7", "move_to_position", "completed").
depends("run_38", "run_38_t7", "run_38_l9").
spawned_by("run_38", "run_38_t7", "run_38_l9").
param("run_38", "run_38_t7", "pos_distance", "0.69").
param("run_38", "run_38_t7", "delta_gripper", "1.3").
param("run_38", "run_38_t7", "efficiency", "1.0").
param("run_38", "run_38_t7", "mean_speed", "0.238").
call("run_38", "run_38_l10", "llm_generate", "completed").
depends("run_38", "run_38_l10", "run_38_t7").
call("run_38", "run_38_t8", "open_gripper", "completed").
depends("run_38", "run_38_t8", "run_38_l10").
spawned_by("run_38", "run_38_t8", "run_38_l10").
param("run_38", "run_38_t8", "pos_distance", "0.43").
param("run_38", "run_38_t8", "delta_gripper", "1.6").
param("run_38", "run_38_t8", "efficiency", "0.8").
param("run_38", "run_38_t8", "mean_speed", "0.176").
call("run_38", "run_38_l11", "llm_generate", "completed").
depends("run_38", "run_38_l11", "run_38_t8").
call("run_38", "run_38_t9", "adjust_position", "completed").
depends("run_38", "run_38_t9", "run_38_l11").
spawned_by("run_38", "run_38_t9", "run_38_l11").
param("run_38", "run_38_t9", "pos_distance", "0.25").
param("run_38", "run_38_t9", "delta_gripper", "0.8").
param("run_38", "run_38_t9", "efficiency", "0.6").
param("run_38", "run_38_t9", "mean_speed", "0.133").
call("run_38", "run_38_l12", "llm_generate", "completed").
depends("run_38", "run_38_l12", "run_38_t9").
call("run_38", "run_38_t10", "adjust_position", "completed").
depends("run_38", "run_38_t10", "run_38_l12").
spawned_by("run_38", "run_38_t10", "run_38_l12").
param("run_38", "run_38_t10", "pos_distance", "0.35").
param("run_38", "run_38_t10", "delta_gripper", "0.0").
param("run_38", "run_38_t10", "efficiency", "0.3").
param("run_38", "run_38_t10", "mean_speed", "0.058").
call("run_38", "run_38_l13", "llm_generate", "completed").
depends("run_38", "run_38_l13", "run_38_t10").
call("run_38", "run_38_n", "notify_user", "completed").
depends("run_38", "run_38_n", "run_38_l13").
spawned_by("run_38", "run_38_n", "run_38_l13").

% === Run 39 (object_slip) ===
job("run_39").
job_status("run_39", "completed").
call("run_39", "run_39_s0", "prompt_user", "completed").
call("run_39", "run_39_s1", "llm_generate", "completed").
depends("run_39", "run_39_s1", "run_39_s0").
call("run_39", "run_39_l2", "llm_generate", "completed").
depends("run_39", "run_39_l2", "run_39_s1").
call("run_39", "run_39_t0", "move_to_position", "completed").
depends("run_39", "run_39_t0", "run_39_l2").
spawned_by("run_39", "run_39_t0", "run_39_l2").
param("run_39", "run_39_t0", "pos_distance", "0.78").
param("run_39", "run_39_t0", "delta_gripper", "0.0").
param("run_39", "run_39_t0", "efficiency", "1.0").
param("run_39", "run_39_t0", "mean_speed", "0.202").
call("run_39", "run_39_l3", "llm_generate", "completed").
depends("run_39", "run_39_l3", "run_39_t0").
call("run_39", "run_39_t1", "close_gripper", "completed").
depends("run_39", "run_39_t1", "run_39_l3").
spawned_by("run_39", "run_39_t1", "run_39_l3").
param("run_39", "run_39_t1", "pos_distance", "0.46").
param("run_39", "run_39_t1", "delta_gripper", "-2.0").
param("run_39", "run_39_t1", "efficiency", "0.9").
param("run_39", "run_39_t1", "mean_speed", "0.165").
call("run_39", "run_39_l4", "llm_generate", "completed").
depends("run_39", "run_39_l4", "run_39_t1").
call("run_39", "run_39_t2", "close_gripper", "completed").
depends("run_39", "run_39_t2", "run_39_l4").
spawned_by("run_39", "run_39_t2", "run_39_l4").
param("run_39", "run_39_t2", "pos_distance", "0.36").
param("run_39", "run_39_t2", "delta_gripper", "-1.8").
param("run_39", "run_39_t2", "efficiency", "1.0").
param("run_39", "run_39_t2", "mean_speed", "0.123").
call("run_39", "run_39_l5", "llm_generate", "completed").
depends("run_39", "run_39_l5", "run_39_t2").
call("run_39", "run_39_t3", "move_to_position", "completed").
depends("run_39", "run_39_t3", "run_39_l5").
spawned_by("run_39", "run_39_t3", "run_39_l5").
param("run_39", "run_39_t3", "pos_distance", "2.13").
param("run_39", "run_39_t3", "delta_gripper", "0.0").
param("run_39", "run_39_t3", "efficiency", "0.7").
param("run_39", "run_39_t3", "mean_speed", "0.259").
call("run_39", "run_39_l6", "llm_generate", "completed").
depends("run_39", "run_39_l6", "run_39_t3").
call("run_39", "run_39_t4", "adjust_position", "completed").
depends("run_39", "run_39_t4", "run_39_l6").
spawned_by("run_39", "run_39_t4", "run_39_l6").
param("run_39", "run_39_t4", "pos_distance", "0.20").
param("run_39", "run_39_t4", "delta_gripper", "0.5").
param("run_39", "run_39_t4", "efficiency", "0.4").
param("run_39", "run_39_t4", "mean_speed", "0.150").
call("run_39", "run_39_l7", "llm_generate", "completed").
depends("run_39", "run_39_l7", "run_39_t4").
call("run_39", "run_39_t5", "close_gripper", "completed").
depends("run_39", "run_39_t5", "run_39_l7").
spawned_by("run_39", "run_39_t5", "run_39_l7").
param("run_39", "run_39_t5", "pos_distance", "0.10").
param("run_39", "run_39_t5", "delta_gripper", "-2.0").
param("run_39", "run_39_t5", "efficiency", "0.3").
param("run_39", "run_39_t5", "mean_speed", "0.080").
call("run_39", "run_39_l8", "llm_generate", "completed").
depends("run_39", "run_39_l8", "run_39_t5").
call("run_39", "run_39_t6", "move_to_position", "completed").
depends("run_39", "run_39_t6", "run_39_l8").
spawned_by("run_39", "run_39_t6", "run_39_l8").
param("run_39", "run_39_t6", "pos_distance", "2.13").
param("run_39", "run_39_t6", "delta_gripper", "0.0").
param("run_39", "run_39_t6", "efficiency", "0.7").
param("run_39", "run_39_t6", "mean_speed", "0.259").
call("run_39", "run_39_l9", "llm_generate", "completed").
depends("run_39", "run_39_l9", "run_39_t6").
call("run_39", "run_39_t7", "move_to_position", "completed").
depends("run_39", "run_39_t7", "run_39_l9").
spawned_by("run_39", "run_39_t7", "run_39_l9").
param("run_39", "run_39_t7", "pos_distance", "0.69").
param("run_39", "run_39_t7", "delta_gripper", "1.3").
param("run_39", "run_39_t7", "efficiency", "1.0").
param("run_39", "run_39_t7", "mean_speed", "0.238").
call("run_39", "run_39_l10", "llm_generate", "completed").
depends("run_39", "run_39_l10", "run_39_t7").
call("run_39", "run_39_t8", "open_gripper", "completed").
depends("run_39", "run_39_t8", "run_39_l10").
spawned_by("run_39", "run_39_t8", "run_39_l10").
param("run_39", "run_39_t8", "pos_distance", "0.43").
param("run_39", "run_39_t8", "delta_gripper", "1.6").
param("run_39", "run_39_t8", "efficiency", "0.8").
param("run_39", "run_39_t8", "mean_speed", "0.176").
call("run_39", "run_39_l11", "llm_generate", "completed").
depends("run_39", "run_39_l11", "run_39_t8").
call("run_39", "run_39_t9", "adjust_position", "completed").
depends("run_39", "run_39_t9", "run_39_l11").
spawned_by("run_39", "run_39_t9", "run_39_l11").
param("run_39", "run_39_t9", "pos_distance", "0.25").
param("run_39", "run_39_t9", "delta_gripper", "0.8").
param("run_39", "run_39_t9", "efficiency", "0.6").
param("run_39", "run_39_t9", "mean_speed", "0.133").
call("run_39", "run_39_l12", "llm_generate", "completed").
depends("run_39", "run_39_l12", "run_39_t9").
call("run_39", "run_39_t10", "adjust_position", "completed").
depends("run_39", "run_39_t10", "run_39_l12").
spawned_by("run_39", "run_39_t10", "run_39_l12").
param("run_39", "run_39_t10", "pos_distance", "0.35").
param("run_39", "run_39_t10", "delta_gripper", "0.0").
param("run_39", "run_39_t10", "efficiency", "0.3").
param("run_39", "run_39_t10", "mean_speed", "0.058").
call("run_39", "run_39_l13", "llm_generate", "completed").
depends("run_39", "run_39_l13", "run_39_t10").
call("run_39", "run_39_n", "notify_user", "completed").
depends("run_39", "run_39_n", "run_39_l13").
spawned_by("run_39", "run_39_n", "run_39_l13").

% === Run 40 (object_slip) ===
job("run_40").
job_status("run_40", "completed").
call("run_40", "run_40_s0", "prompt_user", "completed").
call("run_40", "run_40_s1", "llm_generate", "completed").
depends("run_40", "run_40_s1", "run_40_s0").
call("run_40", "run_40_l2", "llm_generate", "completed").
depends("run_40", "run_40_l2", "run_40_s1").
call("run_40", "run_40_t0", "move_to_position", "completed").
depends("run_40", "run_40_t0", "run_40_l2").
spawned_by("run_40", "run_40_t0", "run_40_l2").
param("run_40", "run_40_t0", "pos_distance", "0.78").
param("run_40", "run_40_t0", "delta_gripper", "0.0").
param("run_40", "run_40_t0", "efficiency", "1.0").
param("run_40", "run_40_t0", "mean_speed", "0.202").
call("run_40", "run_40_l3", "llm_generate", "completed").
depends("run_40", "run_40_l3", "run_40_t0").
call("run_40", "run_40_t1", "close_gripper", "completed").
depends("run_40", "run_40_t1", "run_40_l3").
spawned_by("run_40", "run_40_t1", "run_40_l3").
param("run_40", "run_40_t1", "pos_distance", "0.46").
param("run_40", "run_40_t1", "delta_gripper", "-2.0").
param("run_40", "run_40_t1", "efficiency", "0.9").
param("run_40", "run_40_t1", "mean_speed", "0.165").
call("run_40", "run_40_l4", "llm_generate", "completed").
depends("run_40", "run_40_l4", "run_40_t1").
call("run_40", "run_40_t2", "close_gripper", "completed").
depends("run_40", "run_40_t2", "run_40_l4").
spawned_by("run_40", "run_40_t2", "run_40_l4").
param("run_40", "run_40_t2", "pos_distance", "0.36").
param("run_40", "run_40_t2", "delta_gripper", "-1.8").
param("run_40", "run_40_t2", "efficiency", "1.0").
param("run_40", "run_40_t2", "mean_speed", "0.123").
call("run_40", "run_40_l5", "llm_generate", "completed").
depends("run_40", "run_40_l5", "run_40_t2").
call("run_40", "run_40_t3", "move_to_position", "completed").
depends("run_40", "run_40_t3", "run_40_l5").
spawned_by("run_40", "run_40_t3", "run_40_l5").
param("run_40", "run_40_t3", "pos_distance", "2.13").
param("run_40", "run_40_t3", "delta_gripper", "0.0").
param("run_40", "run_40_t3", "efficiency", "0.7").
param("run_40", "run_40_t3", "mean_speed", "0.259").
call("run_40", "run_40_l6", "llm_generate", "completed").
depends("run_40", "run_40_l6", "run_40_t3").
call("run_40", "run_40_t4", "adjust_position", "completed").
depends("run_40", "run_40_t4", "run_40_l6").
spawned_by("run_40", "run_40_t4", "run_40_l6").
param("run_40", "run_40_t4", "pos_distance", "0.20").
param("run_40", "run_40_t4", "delta_gripper", "0.5").
param("run_40", "run_40_t4", "efficiency", "0.4").
param("run_40", "run_40_t4", "mean_speed", "0.150").
call("run_40", "run_40_l7", "llm_generate", "completed").
depends("run_40", "run_40_l7", "run_40_t4").
call("run_40", "run_40_t5", "close_gripper", "completed").
depends("run_40", "run_40_t5", "run_40_l7").
spawned_by("run_40", "run_40_t5", "run_40_l7").
param("run_40", "run_40_t5", "pos_distance", "0.10").
param("run_40", "run_40_t5", "delta_gripper", "-2.0").
param("run_40", "run_40_t5", "efficiency", "0.3").
param("run_40", "run_40_t5", "mean_speed", "0.080").
call("run_40", "run_40_l8", "llm_generate", "completed").
depends("run_40", "run_40_l8", "run_40_t5").
call("run_40", "run_40_t6", "move_to_position", "completed").
depends("run_40", "run_40_t6", "run_40_l8").
spawned_by("run_40", "run_40_t6", "run_40_l8").
param("run_40", "run_40_t6", "pos_distance", "2.13").
param("run_40", "run_40_t6", "delta_gripper", "0.0").
param("run_40", "run_40_t6", "efficiency", "0.7").
param("run_40", "run_40_t6", "mean_speed", "0.259").
call("run_40", "run_40_l9", "llm_generate", "completed").
depends("run_40", "run_40_l9", "run_40_t6").
call("run_40", "run_40_t7", "move_to_position", "completed").
depends("run_40", "run_40_t7", "run_40_l9").
spawned_by("run_40", "run_40_t7", "run_40_l9").
param("run_40", "run_40_t7", "pos_distance", "0.69").
param("run_40", "run_40_t7", "delta_gripper", "1.3").
param("run_40", "run_40_t7", "efficiency", "1.0").
param("run_40", "run_40_t7", "mean_speed", "0.238").
call("run_40", "run_40_l10", "llm_generate", "completed").
depends("run_40", "run_40_l10", "run_40_t7").
call("run_40", "run_40_t8", "open_gripper", "completed").
depends("run_40", "run_40_t8", "run_40_l10").
spawned_by("run_40", "run_40_t8", "run_40_l10").
param("run_40", "run_40_t8", "pos_distance", "0.43").
param("run_40", "run_40_t8", "delta_gripper", "1.6").
param("run_40", "run_40_t8", "efficiency", "0.8").
param("run_40", "run_40_t8", "mean_speed", "0.176").
call("run_40", "run_40_l11", "llm_generate", "completed").
depends("run_40", "run_40_l11", "run_40_t8").
call("run_40", "run_40_t9", "adjust_position", "completed").
depends("run_40", "run_40_t9", "run_40_l11").
spawned_by("run_40", "run_40_t9", "run_40_l11").
param("run_40", "run_40_t9", "pos_distance", "0.25").
param("run_40", "run_40_t9", "delta_gripper", "0.8").
param("run_40", "run_40_t9", "efficiency", "0.6").
param("run_40", "run_40_t9", "mean_speed", "0.133").
call("run_40", "run_40_l12", "llm_generate", "completed").
depends("run_40", "run_40_l12", "run_40_t9").
call("run_40", "run_40_t10", "adjust_position", "completed").
depends("run_40", "run_40_t10", "run_40_l12").
spawned_by("run_40", "run_40_t10", "run_40_l12").
param("run_40", "run_40_t10", "pos_distance", "0.35").
param("run_40", "run_40_t10", "delta_gripper", "0.0").
param("run_40", "run_40_t10", "efficiency", "0.3").
param("run_40", "run_40_t10", "mean_speed", "0.058").
call("run_40", "run_40_l13", "llm_generate", "completed").
depends("run_40", "run_40_l13", "run_40_t10").
call("run_40", "run_40_n", "notify_user", "completed").
depends("run_40", "run_40_n", "run_40_l13").
spawned_by("run_40", "run_40_n", "run_40_l13").

% === Run 41 (object_slip) ===
job("run_41").
job_status("run_41", "completed").
call("run_41", "run_41_s0", "prompt_user", "completed").
call("run_41", "run_41_s1", "llm_generate", "completed").
depends("run_41", "run_41_s1", "run_41_s0").
call("run_41", "run_41_l2", "llm_generate", "completed").
depends("run_41", "run_41_l2", "run_41_s1").
call("run_41", "run_41_t0", "move_to_position", "completed").
depends("run_41", "run_41_t0", "run_41_l2").
spawned_by("run_41", "run_41_t0", "run_41_l2").
param("run_41", "run_41_t0", "pos_distance", "0.78").
param("run_41", "run_41_t0", "delta_gripper", "0.0").
param("run_41", "run_41_t0", "efficiency", "1.0").
param("run_41", "run_41_t0", "mean_speed", "0.202").
call("run_41", "run_41_l3", "llm_generate", "completed").
depends("run_41", "run_41_l3", "run_41_t0").
call("run_41", "run_41_t1", "close_gripper", "completed").
depends("run_41", "run_41_t1", "run_41_l3").
spawned_by("run_41", "run_41_t1", "run_41_l3").
param("run_41", "run_41_t1", "pos_distance", "0.46").
param("run_41", "run_41_t1", "delta_gripper", "-2.0").
param("run_41", "run_41_t1", "efficiency", "0.9").
param("run_41", "run_41_t1", "mean_speed", "0.165").
call("run_41", "run_41_l4", "llm_generate", "completed").
depends("run_41", "run_41_l4", "run_41_t1").
call("run_41", "run_41_t2", "close_gripper", "completed").
depends("run_41", "run_41_t2", "run_41_l4").
spawned_by("run_41", "run_41_t2", "run_41_l4").
param("run_41", "run_41_t2", "pos_distance", "0.36").
param("run_41", "run_41_t2", "delta_gripper", "-1.8").
param("run_41", "run_41_t2", "efficiency", "1.0").
param("run_41", "run_41_t2", "mean_speed", "0.123").
call("run_41", "run_41_l5", "llm_generate", "completed").
depends("run_41", "run_41_l5", "run_41_t2").
call("run_41", "run_41_t3", "move_to_position", "completed").
depends("run_41", "run_41_t3", "run_41_l5").
spawned_by("run_41", "run_41_t3", "run_41_l5").
param("run_41", "run_41_t3", "pos_distance", "2.13").
param("run_41", "run_41_t3", "delta_gripper", "0.0").
param("run_41", "run_41_t3", "efficiency", "0.7").
param("run_41", "run_41_t3", "mean_speed", "0.259").
call("run_41", "run_41_l6", "llm_generate", "completed").
depends("run_41", "run_41_l6", "run_41_t3").
call("run_41", "run_41_t4", "adjust_position", "completed").
depends("run_41", "run_41_t4", "run_41_l6").
spawned_by("run_41", "run_41_t4", "run_41_l6").
param("run_41", "run_41_t4", "pos_distance", "0.20").
param("run_41", "run_41_t4", "delta_gripper", "0.5").
param("run_41", "run_41_t4", "efficiency", "0.4").
param("run_41", "run_41_t4", "mean_speed", "0.150").
call("run_41", "run_41_l7", "llm_generate", "completed").
depends("run_41", "run_41_l7", "run_41_t4").
call("run_41", "run_41_t5", "close_gripper", "completed").
depends("run_41", "run_41_t5", "run_41_l7").
spawned_by("run_41", "run_41_t5", "run_41_l7").
param("run_41", "run_41_t5", "pos_distance", "0.10").
param("run_41", "run_41_t5", "delta_gripper", "-2.0").
param("run_41", "run_41_t5", "efficiency", "0.3").
param("run_41", "run_41_t5", "mean_speed", "0.080").
call("run_41", "run_41_l8", "llm_generate", "completed").
depends("run_41", "run_41_l8", "run_41_t5").
call("run_41", "run_41_t6", "move_to_position", "completed").
depends("run_41", "run_41_t6", "run_41_l8").
spawned_by("run_41", "run_41_t6", "run_41_l8").
param("run_41", "run_41_t6", "pos_distance", "2.13").
param("run_41", "run_41_t6", "delta_gripper", "0.0").
param("run_41", "run_41_t6", "efficiency", "0.7").
param("run_41", "run_41_t6", "mean_speed", "0.259").
call("run_41", "run_41_l9", "llm_generate", "completed").
depends("run_41", "run_41_l9", "run_41_t6").
call("run_41", "run_41_t7", "move_to_position", "completed").
depends("run_41", "run_41_t7", "run_41_l9").
spawned_by("run_41", "run_41_t7", "run_41_l9").
param("run_41", "run_41_t7", "pos_distance", "0.69").
param("run_41", "run_41_t7", "delta_gripper", "1.3").
param("run_41", "run_41_t7", "efficiency", "1.0").
param("run_41", "run_41_t7", "mean_speed", "0.238").
call("run_41", "run_41_l10", "llm_generate", "completed").
depends("run_41", "run_41_l10", "run_41_t7").
call("run_41", "run_41_t8", "open_gripper", "completed").
depends("run_41", "run_41_t8", "run_41_l10").
spawned_by("run_41", "run_41_t8", "run_41_l10").
param("run_41", "run_41_t8", "pos_distance", "0.43").
param("run_41", "run_41_t8", "delta_gripper", "1.6").
param("run_41", "run_41_t8", "efficiency", "0.8").
param("run_41", "run_41_t8", "mean_speed", "0.176").
call("run_41", "run_41_l11", "llm_generate", "completed").
depends("run_41", "run_41_l11", "run_41_t8").
call("run_41", "run_41_t9", "adjust_position", "completed").
depends("run_41", "run_41_t9", "run_41_l11").
spawned_by("run_41", "run_41_t9", "run_41_l11").
param("run_41", "run_41_t9", "pos_distance", "0.25").
param("run_41", "run_41_t9", "delta_gripper", "0.8").
param("run_41", "run_41_t9", "efficiency", "0.6").
param("run_41", "run_41_t9", "mean_speed", "0.133").
call("run_41", "run_41_l12", "llm_generate", "completed").
depends("run_41", "run_41_l12", "run_41_t9").
call("run_41", "run_41_t10", "adjust_position", "completed").
depends("run_41", "run_41_t10", "run_41_l12").
spawned_by("run_41", "run_41_t10", "run_41_l12").
param("run_41", "run_41_t10", "pos_distance", "0.35").
param("run_41", "run_41_t10", "delta_gripper", "0.0").
param("run_41", "run_41_t10", "efficiency", "0.3").
param("run_41", "run_41_t10", "mean_speed", "0.058").
call("run_41", "run_41_l13", "llm_generate", "completed").
depends("run_41", "run_41_l13", "run_41_t10").
call("run_41", "run_41_n", "notify_user", "completed").
depends("run_41", "run_41_n", "run_41_l13").
spawned_by("run_41", "run_41_n", "run_41_l13").

% === Run 42 (object_slip) ===
job("run_42").
job_status("run_42", "completed").
call("run_42", "run_42_s0", "prompt_user", "completed").
call("run_42", "run_42_s1", "llm_generate", "completed").
depends("run_42", "run_42_s1", "run_42_s0").
call("run_42", "run_42_l2", "llm_generate", "completed").
depends("run_42", "run_42_l2", "run_42_s1").
call("run_42", "run_42_t0", "move_to_position", "completed").
depends("run_42", "run_42_t0", "run_42_l2").
spawned_by("run_42", "run_42_t0", "run_42_l2").
param("run_42", "run_42_t0", "pos_distance", "0.78").
param("run_42", "run_42_t0", "delta_gripper", "0.0").
param("run_42", "run_42_t0", "efficiency", "1.0").
param("run_42", "run_42_t0", "mean_speed", "0.202").
call("run_42", "run_42_l3", "llm_generate", "completed").
depends("run_42", "run_42_l3", "run_42_t0").
call("run_42", "run_42_t1", "close_gripper", "completed").
depends("run_42", "run_42_t1", "run_42_l3").
spawned_by("run_42", "run_42_t1", "run_42_l3").
param("run_42", "run_42_t1", "pos_distance", "0.46").
param("run_42", "run_42_t1", "delta_gripper", "-2.0").
param("run_42", "run_42_t1", "efficiency", "0.9").
param("run_42", "run_42_t1", "mean_speed", "0.165").
call("run_42", "run_42_l4", "llm_generate", "completed").
depends("run_42", "run_42_l4", "run_42_t1").
call("run_42", "run_42_t2", "close_gripper", "completed").
depends("run_42", "run_42_t2", "run_42_l4").
spawned_by("run_42", "run_42_t2", "run_42_l4").
param("run_42", "run_42_t2", "pos_distance", "0.36").
param("run_42", "run_42_t2", "delta_gripper", "-1.8").
param("run_42", "run_42_t2", "efficiency", "1.0").
param("run_42", "run_42_t2", "mean_speed", "0.123").
call("run_42", "run_42_l5", "llm_generate", "completed").
depends("run_42", "run_42_l5", "run_42_t2").
call("run_42", "run_42_t3", "move_to_position", "completed").
depends("run_42", "run_42_t3", "run_42_l5").
spawned_by("run_42", "run_42_t3", "run_42_l5").
param("run_42", "run_42_t3", "pos_distance", "2.13").
param("run_42", "run_42_t3", "delta_gripper", "0.0").
param("run_42", "run_42_t3", "efficiency", "0.7").
param("run_42", "run_42_t3", "mean_speed", "0.259").
call("run_42", "run_42_l6", "llm_generate", "completed").
depends("run_42", "run_42_l6", "run_42_t3").
call("run_42", "run_42_t4", "adjust_position", "completed").
depends("run_42", "run_42_t4", "run_42_l6").
spawned_by("run_42", "run_42_t4", "run_42_l6").
param("run_42", "run_42_t4", "pos_distance", "0.20").
param("run_42", "run_42_t4", "delta_gripper", "0.5").
param("run_42", "run_42_t4", "efficiency", "0.4").
param("run_42", "run_42_t4", "mean_speed", "0.150").
call("run_42", "run_42_l7", "llm_generate", "completed").
depends("run_42", "run_42_l7", "run_42_t4").
call("run_42", "run_42_t5", "close_gripper", "completed").
depends("run_42", "run_42_t5", "run_42_l7").
spawned_by("run_42", "run_42_t5", "run_42_l7").
param("run_42", "run_42_t5", "pos_distance", "0.10").
param("run_42", "run_42_t5", "delta_gripper", "-2.0").
param("run_42", "run_42_t5", "efficiency", "0.3").
param("run_42", "run_42_t5", "mean_speed", "0.080").
call("run_42", "run_42_l8", "llm_generate", "completed").
depends("run_42", "run_42_l8", "run_42_t5").
call("run_42", "run_42_t6", "move_to_position", "completed").
depends("run_42", "run_42_t6", "run_42_l8").
spawned_by("run_42", "run_42_t6", "run_42_l8").
param("run_42", "run_42_t6", "pos_distance", "2.13").
param("run_42", "run_42_t6", "delta_gripper", "0.0").
param("run_42", "run_42_t6", "efficiency", "0.7").
param("run_42", "run_42_t6", "mean_speed", "0.259").
call("run_42", "run_42_l9", "llm_generate", "completed").
depends("run_42", "run_42_l9", "run_42_t6").
call("run_42", "run_42_t7", "move_to_position", "completed").
depends("run_42", "run_42_t7", "run_42_l9").
spawned_by("run_42", "run_42_t7", "run_42_l9").
param("run_42", "run_42_t7", "pos_distance", "0.69").
param("run_42", "run_42_t7", "delta_gripper", "1.3").
param("run_42", "run_42_t7", "efficiency", "1.0").
param("run_42", "run_42_t7", "mean_speed", "0.238").
call("run_42", "run_42_l10", "llm_generate", "completed").
depends("run_42", "run_42_l10", "run_42_t7").
call("run_42", "run_42_t8", "open_gripper", "completed").
depends("run_42", "run_42_t8", "run_42_l10").
spawned_by("run_42", "run_42_t8", "run_42_l10").
param("run_42", "run_42_t8", "pos_distance", "0.43").
param("run_42", "run_42_t8", "delta_gripper", "1.6").
param("run_42", "run_42_t8", "efficiency", "0.8").
param("run_42", "run_42_t8", "mean_speed", "0.176").
call("run_42", "run_42_l11", "llm_generate", "completed").
depends("run_42", "run_42_l11", "run_42_t8").
call("run_42", "run_42_t9", "adjust_position", "completed").
depends("run_42", "run_42_t9", "run_42_l11").
spawned_by("run_42", "run_42_t9", "run_42_l11").
param("run_42", "run_42_t9", "pos_distance", "0.25").
param("run_42", "run_42_t9", "delta_gripper", "0.8").
param("run_42", "run_42_t9", "efficiency", "0.6").
param("run_42", "run_42_t9", "mean_speed", "0.133").
call("run_42", "run_42_l12", "llm_generate", "completed").
depends("run_42", "run_42_l12", "run_42_t9").
call("run_42", "run_42_t10", "adjust_position", "completed").
depends("run_42", "run_42_t10", "run_42_l12").
spawned_by("run_42", "run_42_t10", "run_42_l12").
param("run_42", "run_42_t10", "pos_distance", "0.35").
param("run_42", "run_42_t10", "delta_gripper", "0.0").
param("run_42", "run_42_t10", "efficiency", "0.3").
param("run_42", "run_42_t10", "mean_speed", "0.058").
call("run_42", "run_42_l13", "llm_generate", "completed").
depends("run_42", "run_42_l13", "run_42_t10").
call("run_42", "run_42_n", "notify_user", "completed").
depends("run_42", "run_42_n", "run_42_l13").
spawned_by("run_42", "run_42_n", "run_42_l13").

% === Run 43 (short_execution) ===
job("run_43").
job_status("run_43", "completed").
call("run_43", "run_43_s0", "prompt_user", "completed").
call("run_43", "run_43_s1", "llm_generate", "completed").
depends("run_43", "run_43_s1", "run_43_s0").
call("run_43", "run_43_l2", "llm_generate", "completed").
depends("run_43", "run_43_l2", "run_43_s1").
call("run_43", "run_43_t0", "move_to_position", "completed").
depends("run_43", "run_43_t0", "run_43_l2").
spawned_by("run_43", "run_43_t0", "run_43_l2").
param("run_43", "run_43_t0", "pos_distance", "0.19").
param("run_43", "run_43_t0", "delta_gripper", "0.0").
param("run_43", "run_43_t0", "efficiency", "1.0").
param("run_43", "run_43_t0", "mean_speed", "0.198").
call("run_43", "run_43_l3", "llm_generate", "completed").
depends("run_43", "run_43_l3", "run_43_t0").
call("run_43", "run_43_t1", "close_gripper", "completed").
depends("run_43", "run_43_t1", "run_43_l3").
spawned_by("run_43", "run_43_t1", "run_43_l3").
param("run_43", "run_43_t1", "pos_distance", "0.25").
param("run_43", "run_43_t1", "delta_gripper", "-2.0").
param("run_43", "run_43_t1", "efficiency", "0.9").
param("run_43", "run_43_t1", "mean_speed", "0.130").
call("run_43", "run_43_l4", "llm_generate", "completed").
depends("run_43", "run_43_l4", "run_43_t1").
call("run_43", "run_43_t2", "close_gripper", "completed").
depends("run_43", "run_43_t2", "run_43_l4").
spawned_by("run_43", "run_43_t2", "run_43_l4").
param("run_43", "run_43_t2", "pos_distance", "0.21").
param("run_43", "run_43_t2", "delta_gripper", "-1.8").
param("run_43", "run_43_t2", "efficiency", "1.0").
param("run_43", "run_43_t2", "mean_speed", "0.123").
call("run_43", "run_43_l5", "llm_generate", "completed").
depends("run_43", "run_43_l5", "run_43_t2").
call("run_43", "run_43_t3", "move_to_position", "completed").
depends("run_43", "run_43_t3", "run_43_l5").
spawned_by("run_43", "run_43_t3", "run_43_l5").
param("run_43", "run_43_t3", "pos_distance", "1.00").
param("run_43", "run_43_t3", "delta_gripper", "0.0").
param("run_43", "run_43_t3", "efficiency", "0.7").
param("run_43", "run_43_t3", "mean_speed", "0.165").
call("run_43", "run_43_l6", "llm_generate", "completed").
depends("run_43", "run_43_l6", "run_43_t3").
call("run_43", "run_43_t4", "move_to_position", "completed").
depends("run_43", "run_43_t4", "run_43_l6").
spawned_by("run_43", "run_43_t4", "run_43_l6").
param("run_43", "run_43_t4", "pos_distance", "0.15").
param("run_43", "run_43_t4", "delta_gripper", "1.3").
param("run_43", "run_43_t4", "efficiency", "1.0").
param("run_43", "run_43_t4", "mean_speed", "0.209").
call("run_43", "run_43_l7", "llm_generate", "completed").
depends("run_43", "run_43_l7", "run_43_t4").
call("run_43", "run_43_t5", "open_gripper", "completed").
depends("run_43", "run_43_t5", "run_43_l7").
spawned_by("run_43", "run_43_t5", "run_43_l7").
param("run_43", "run_43_t5", "pos_distance", "0.17").
param("run_43", "run_43_t5", "delta_gripper", "1.6").
param("run_43", "run_43_t5", "efficiency", "0.8").
param("run_43", "run_43_t5", "mean_speed", "0.145").
call("run_43", "run_43_l8", "llm_generate", "completed").
depends("run_43", "run_43_l8", "run_43_t5").
call("run_43", "run_43_t6", "adjust_position", "completed").
depends("run_43", "run_43_t6", "run_43_l8").
spawned_by("run_43", "run_43_t6", "run_43_l8").
param("run_43", "run_43_t6", "pos_distance", "0.14").
param("run_43", "run_43_t6", "delta_gripper", "0.8").
param("run_43", "run_43_t6", "efficiency", "0.6").
param("run_43", "run_43_t6", "mean_speed", "0.079").
call("run_43", "run_43_l9", "llm_generate", "completed").
depends("run_43", "run_43_l9", "run_43_t6").
call("run_43", "run_43_t7", "adjust_position", "completed").
depends("run_43", "run_43_t7", "run_43_l9").
spawned_by("run_43", "run_43_t7", "run_43_l9").
param("run_43", "run_43_t7", "pos_distance", "0.15").
param("run_43", "run_43_t7", "delta_gripper", "0.0").
param("run_43", "run_43_t7", "efficiency", "0.3").
param("run_43", "run_43_t7", "mean_speed", "0.048").
call("run_43", "run_43_l10", "llm_generate", "completed").
depends("run_43", "run_43_l10", "run_43_t7").
call("run_43", "run_43_n", "notify_user", "completed").
depends("run_43", "run_43_n", "run_43_l10").
spawned_by("run_43", "run_43_n", "run_43_l10").

% === Run 44 (short_execution) ===
job("run_44").
job_status("run_44", "completed").
call("run_44", "run_44_s0", "prompt_user", "completed").
call("run_44", "run_44_s1", "llm_generate", "completed").
depends("run_44", "run_44_s1", "run_44_s0").
call("run_44", "run_44_l2", "llm_generate", "completed").
depends("run_44", "run_44_l2", "run_44_s1").
call("run_44", "run_44_t0", "move_to_position", "completed").
depends("run_44", "run_44_t0", "run_44_l2").
spawned_by("run_44", "run_44_t0", "run_44_l2").
param("run_44", "run_44_t0", "pos_distance", "0.19").
param("run_44", "run_44_t0", "delta_gripper", "0.0").
param("run_44", "run_44_t0", "efficiency", "1.0").
param("run_44", "run_44_t0", "mean_speed", "0.136").
call("run_44", "run_44_l3", "llm_generate", "completed").
depends("run_44", "run_44_l3", "run_44_t0").
call("run_44", "run_44_t1", "close_gripper", "completed").
depends("run_44", "run_44_t1", "run_44_l3").
spawned_by("run_44", "run_44_t1", "run_44_l3").
param("run_44", "run_44_t1", "pos_distance", "0.15").
param("run_44", "run_44_t1", "delta_gripper", "-2.0").
param("run_44", "run_44_t1", "efficiency", "0.9").
param("run_44", "run_44_t1", "mean_speed", "0.138").
call("run_44", "run_44_l4", "llm_generate", "completed").
depends("run_44", "run_44_l4", "run_44_t1").
call("run_44", "run_44_t2", "close_gripper", "completed").
depends("run_44", "run_44_t2", "run_44_l4").
spawned_by("run_44", "run_44_t2", "run_44_l4").
param("run_44", "run_44_t2", "pos_distance", "0.19").
param("run_44", "run_44_t2", "delta_gripper", "-1.8").
param("run_44", "run_44_t2", "efficiency", "1.0").
param("run_44", "run_44_t2", "mean_speed", "0.082").
call("run_44", "run_44_l5", "llm_generate", "completed").
depends("run_44", "run_44_l5", "run_44_t2").
call("run_44", "run_44_t3", "move_to_position", "completed").
depends("run_44", "run_44_t3", "run_44_l5").
spawned_by("run_44", "run_44_t3", "run_44_l5").
param("run_44", "run_44_t3", "pos_distance", "1.02").
param("run_44", "run_44_t3", "delta_gripper", "0.0").
param("run_44", "run_44_t3", "efficiency", "0.7").
param("run_44", "run_44_t3", "mean_speed", "0.167").
call("run_44", "run_44_l6", "llm_generate", "completed").
depends("run_44", "run_44_l6", "run_44_t3").
call("run_44", "run_44_t4", "move_to_position", "completed").
depends("run_44", "run_44_t4", "run_44_l6").
spawned_by("run_44", "run_44_t4", "run_44_l6").
param("run_44", "run_44_t4", "pos_distance", "0.40").
param("run_44", "run_44_t4", "delta_gripper", "1.3").
param("run_44", "run_44_t4", "efficiency", "1.0").
param("run_44", "run_44_t4", "mean_speed", "0.216").
call("run_44", "run_44_l7", "llm_generate", "completed").
depends("run_44", "run_44_l7", "run_44_t4").
call("run_44", "run_44_t5", "open_gripper", "completed").
depends("run_44", "run_44_t5", "run_44_l7").
spawned_by("run_44", "run_44_t5", "run_44_l7").
param("run_44", "run_44_t5", "pos_distance", "0.18").
param("run_44", "run_44_t5", "delta_gripper", "1.6").
param("run_44", "run_44_t5", "efficiency", "0.8").
param("run_44", "run_44_t5", "mean_speed", "0.128").
call("run_44", "run_44_l8", "llm_generate", "completed").
depends("run_44", "run_44_l8", "run_44_t5").
call("run_44", "run_44_t6", "adjust_position", "completed").
depends("run_44", "run_44_t6", "run_44_l8").
spawned_by("run_44", "run_44_t6", "run_44_l8").
param("run_44", "run_44_t6", "pos_distance", "0.08").
param("run_44", "run_44_t6", "delta_gripper", "0.8").
param("run_44", "run_44_t6", "efficiency", "0.6").
param("run_44", "run_44_t6", "mean_speed", "0.088").
call("run_44", "run_44_l9", "llm_generate", "completed").
depends("run_44", "run_44_l9", "run_44_t6").
call("run_44", "run_44_t7", "adjust_position", "completed").
depends("run_44", "run_44_t7", "run_44_l9").
spawned_by("run_44", "run_44_t7", "run_44_l9").
param("run_44", "run_44_t7", "pos_distance", "0.20").
param("run_44", "run_44_t7", "delta_gripper", "0.0").
param("run_44", "run_44_t7", "efficiency", "0.3").
param("run_44", "run_44_t7", "mean_speed", "0.041").
call("run_44", "run_44_l10", "llm_generate", "completed").
depends("run_44", "run_44_l10", "run_44_t7").
call("run_44", "run_44_n", "notify_user", "completed").
depends("run_44", "run_44_n", "run_44_l10").
spawned_by("run_44", "run_44_n", "run_44_l10").

% === Run 45 (short_execution) ===
job("run_45").
job_status("run_45", "completed").
call("run_45", "run_45_s0", "prompt_user", "completed").
call("run_45", "run_45_s1", "llm_generate", "completed").
depends("run_45", "run_45_s1", "run_45_s0").
call("run_45", "run_45_l2", "llm_generate", "completed").
depends("run_45", "run_45_l2", "run_45_s1").
call("run_45", "run_45_t0", "move_to_position", "completed").
depends("run_45", "run_45_t0", "run_45_l2").
spawned_by("run_45", "run_45_t0", "run_45_l2").
param("run_45", "run_45_t0", "pos_distance", "0.47").
param("run_45", "run_45_t0", "delta_gripper", "0.0").
param("run_45", "run_45_t0", "efficiency", "1.0").
param("run_45", "run_45_t0", "mean_speed", "0.168").
call("run_45", "run_45_l3", "llm_generate", "completed").
depends("run_45", "run_45_l3", "run_45_t0").
call("run_45", "run_45_t1", "close_gripper", "completed").
depends("run_45", "run_45_t1", "run_45_l3").
spawned_by("run_45", "run_45_t1", "run_45_l3").
param("run_45", "run_45_t1", "pos_distance", "0.19").
param("run_45", "run_45_t1", "delta_gripper", "-2.0").
param("run_45", "run_45_t1", "efficiency", "0.9").
param("run_45", "run_45_t1", "mean_speed", "0.117").
call("run_45", "run_45_l4", "llm_generate", "completed").
depends("run_45", "run_45_l4", "run_45_t1").
call("run_45", "run_45_t2", "close_gripper", "completed").
depends("run_45", "run_45_t2", "run_45_l4").
spawned_by("run_45", "run_45_t2", "run_45_l4").
param("run_45", "run_45_t2", "pos_distance", "0.10").
param("run_45", "run_45_t2", "delta_gripper", "-1.8").
param("run_45", "run_45_t2", "efficiency", "1.0").
param("run_45", "run_45_t2", "mean_speed", "0.084").
call("run_45", "run_45_l5", "llm_generate", "completed").
depends("run_45", "run_45_l5", "run_45_t2").
call("run_45", "run_45_t3", "move_to_position", "completed").
depends("run_45", "run_45_t3", "run_45_l5").
spawned_by("run_45", "run_45_t3", "run_45_l5").
param("run_45", "run_45_t3", "pos_distance", "1.07").
param("run_45", "run_45_t3", "delta_gripper", "0.0").
param("run_45", "run_45_t3", "efficiency", "0.7").
param("run_45", "run_45_t3", "mean_speed", "0.211").
call("run_45", "run_45_l6", "llm_generate", "completed").
depends("run_45", "run_45_l6", "run_45_t3").
call("run_45", "run_45_t4", "move_to_position", "completed").
depends("run_45", "run_45_t4", "run_45_l6").
spawned_by("run_45", "run_45_t4", "run_45_l6").
param("run_45", "run_45_t4", "pos_distance", "0.35").
param("run_45", "run_45_t4", "delta_gripper", "1.3").
param("run_45", "run_45_t4", "efficiency", "1.0").
param("run_45", "run_45_t4", "mean_speed", "0.143").
call("run_45", "run_45_l7", "llm_generate", "completed").
depends("run_45", "run_45_l7", "run_45_t4").
call("run_45", "run_45_t5", "open_gripper", "completed").
depends("run_45", "run_45_t5", "run_45_l7").
spawned_by("run_45", "run_45_t5", "run_45_l7").
param("run_45", "run_45_t5", "pos_distance", "0.18").
param("run_45", "run_45_t5", "delta_gripper", "1.6").
param("run_45", "run_45_t5", "efficiency", "0.8").
param("run_45", "run_45_t5", "mean_speed", "0.169").
call("run_45", "run_45_l8", "llm_generate", "completed").
depends("run_45", "run_45_l8", "run_45_t5").
call("run_45", "run_45_t6", "adjust_position", "completed").
depends("run_45", "run_45_t6", "run_45_l8").
spawned_by("run_45", "run_45_t6", "run_45_l8").
param("run_45", "run_45_t6", "pos_distance", "0.10").
param("run_45", "run_45_t6", "delta_gripper", "0.8").
param("run_45", "run_45_t6", "efficiency", "0.6").
param("run_45", "run_45_t6", "mean_speed", "0.113").
call("run_45", "run_45_l9", "llm_generate", "completed").
depends("run_45", "run_45_l9", "run_45_t6").
call("run_45", "run_45_t7", "adjust_position", "completed").
depends("run_45", "run_45_t7", "run_45_l9").
spawned_by("run_45", "run_45_t7", "run_45_l9").
param("run_45", "run_45_t7", "pos_distance", "0.09").
param("run_45", "run_45_t7", "delta_gripper", "0.0").
param("run_45", "run_45_t7", "efficiency", "0.3").
param("run_45", "run_45_t7", "mean_speed", "0.058").
call("run_45", "run_45_l10", "llm_generate", "completed").
depends("run_45", "run_45_l10", "run_45_t7").
call("run_45", "run_45_n", "notify_user", "completed").
depends("run_45", "run_45_n", "run_45_l10").
spawned_by("run_45", "run_45_n", "run_45_l10").

% === Run 46 (short_execution) ===
job("run_46").
job_status("run_46", "completed").
call("run_46", "run_46_s0", "prompt_user", "completed").
call("run_46", "run_46_s1", "llm_generate", "completed").
depends("run_46", "run_46_s1", "run_46_s0").
call("run_46", "run_46_l2", "llm_generate", "completed").
depends("run_46", "run_46_l2", "run_46_s1").
call("run_46", "run_46_t0", "close_gripper", "completed").
depends("run_46", "run_46_t0", "run_46_l2").
spawned_by("run_46", "run_46_t0", "run_46_l2").
param("run_46", "run_46_t0", "pos_distance", "0.19").
param("run_46", "run_46_t0", "delta_gripper", "-2.0").
param("run_46", "run_46_t0", "efficiency", "0.9").
param("run_46", "run_46_t0", "mean_speed", "0.128").
call("run_46", "run_46_l3", "llm_generate", "completed").
depends("run_46", "run_46_l3", "run_46_t0").
call("run_46", "run_46_t1", "close_gripper", "completed").
depends("run_46", "run_46_t1", "run_46_l3").
spawned_by("run_46", "run_46_t1", "run_46_l3").
param("run_46", "run_46_t1", "pos_distance", "0.08").
param("run_46", "run_46_t1", "delta_gripper", "-1.8").
param("run_46", "run_46_t1", "efficiency", "1.0").
param("run_46", "run_46_t1", "mean_speed", "0.122").
call("run_46", "run_46_l4", "llm_generate", "completed").
depends("run_46", "run_46_l4", "run_46_t1").
call("run_46", "run_46_t2", "move_to_position", "completed").
depends("run_46", "run_46_t2", "run_46_l4").
spawned_by("run_46", "run_46_t2", "run_46_l4").
param("run_46", "run_46_t2", "pos_distance", "1.20").
param("run_46", "run_46_t2", "delta_gripper", "0.0").
param("run_46", "run_46_t2", "efficiency", "0.7").
param("run_46", "run_46_t2", "mean_speed", "0.190").
call("run_46", "run_46_l5", "llm_generate", "completed").
depends("run_46", "run_46_l5", "run_46_t2").
call("run_46", "run_46_t3", "move_to_position", "completed").
depends("run_46", "run_46_t3", "run_46_l5").
spawned_by("run_46", "run_46_t3", "run_46_l5").
param("run_46", "run_46_t3", "pos_distance", "0.17").
param("run_46", "run_46_t3", "delta_gripper", "1.3").
param("run_46", "run_46_t3", "efficiency", "1.0").
param("run_46", "run_46_t3", "mean_speed", "0.218").
call("run_46", "run_46_l6", "llm_generate", "completed").
depends("run_46", "run_46_l6", "run_46_t3").
call("run_46", "run_46_t4", "open_gripper", "completed").
depends("run_46", "run_46_t4", "run_46_l6").
spawned_by("run_46", "run_46_t4", "run_46_l6").
param("run_46", "run_46_t4", "pos_distance", "0.17").
param("run_46", "run_46_t4", "delta_gripper", "1.6").
param("run_46", "run_46_t4", "efficiency", "0.8").
param("run_46", "run_46_t4", "mean_speed", "0.151").
call("run_46", "run_46_l7", "llm_generate", "completed").
depends("run_46", "run_46_l7", "run_46_t4").
call("run_46", "run_46_t5", "adjust_position", "completed").
depends("run_46", "run_46_t5", "run_46_l7").
spawned_by("run_46", "run_46_t5", "run_46_l7").
param("run_46", "run_46_t5", "pos_distance", "0.10").
param("run_46", "run_46_t5", "delta_gripper", "0.8").
param("run_46", "run_46_t5", "efficiency", "0.6").
param("run_46", "run_46_t5", "mean_speed", "0.085").
call("run_46", "run_46_l8", "llm_generate", "completed").
depends("run_46", "run_46_l8", "run_46_t5").
call("run_46", "run_46_t6", "adjust_position", "completed").
depends("run_46", "run_46_t6", "run_46_l8").
spawned_by("run_46", "run_46_t6", "run_46_l8").
param("run_46", "run_46_t6", "pos_distance", "0.19").
param("run_46", "run_46_t6", "delta_gripper", "0.0").
param("run_46", "run_46_t6", "efficiency", "0.3").
param("run_46", "run_46_t6", "mean_speed", "0.058").
call("run_46", "run_46_l9", "llm_generate", "completed").
depends("run_46", "run_46_l9", "run_46_t6").
call("run_46", "run_46_n", "notify_user", "completed").
depends("run_46", "run_46_n", "run_46_l9").
spawned_by("run_46", "run_46_n", "run_46_l9").

% === Run 47 (short_execution) ===
job("run_47").
job_status("run_47", "completed").
call("run_47", "run_47_s0", "prompt_user", "completed").
call("run_47", "run_47_s1", "llm_generate", "completed").
depends("run_47", "run_47_s1", "run_47_s0").
call("run_47", "run_47_l2", "llm_generate", "completed").
depends("run_47", "run_47_l2", "run_47_s1").
call("run_47", "run_47_t0", "move_to_position", "completed").
depends("run_47", "run_47_t0", "run_47_l2").
spawned_by("run_47", "run_47_t0", "run_47_l2").
param("run_47", "run_47_t0", "pos_distance", "0.33").
param("run_47", "run_47_t0", "delta_gripper", "0.0").
param("run_47", "run_47_t0", "efficiency", "1.0").
param("run_47", "run_47_t0", "mean_speed", "0.140").
call("run_47", "run_47_l3", "llm_generate", "completed").
depends("run_47", "run_47_l3", "run_47_t0").
call("run_47", "run_47_t1", "close_gripper", "completed").
depends("run_47", "run_47_t1", "run_47_l3").
spawned_by("run_47", "run_47_t1", "run_47_l3").
param("run_47", "run_47_t1", "pos_distance", "0.26").
param("run_47", "run_47_t1", "delta_gripper", "-2.0").
param("run_47", "run_47_t1", "efficiency", "0.9").
param("run_47", "run_47_t1", "mean_speed", "0.125").
call("run_47", "run_47_l4", "llm_generate", "completed").
depends("run_47", "run_47_l4", "run_47_t1").
call("run_47", "run_47_t2", "close_gripper", "completed").
depends("run_47", "run_47_t2", "run_47_l4").
spawned_by("run_47", "run_47_t2", "run_47_l4").
param("run_47", "run_47_t2", "pos_distance", "0.20").
param("run_47", "run_47_t2", "delta_gripper", "-1.8").
param("run_47", "run_47_t2", "efficiency", "1.0").
param("run_47", "run_47_t2", "mean_speed", "0.114").
call("run_47", "run_47_l5", "llm_generate", "completed").
depends("run_47", "run_47_l5", "run_47_t2").
call("run_47", "run_47_t3", "move_to_position", "completed").
depends("run_47", "run_47_t3", "run_47_l5").
spawned_by("run_47", "run_47_t3", "run_47_l5").
param("run_47", "run_47_t3", "pos_distance", "0.66").
param("run_47", "run_47_t3", "delta_gripper", "0.0").
param("run_47", "run_47_t3", "efficiency", "0.7").
param("run_47", "run_47_t3", "mean_speed", "0.232").
call("run_47", "run_47_l6", "llm_generate", "completed").
depends("run_47", "run_47_l6", "run_47_t3").
call("run_47", "run_47_t4", "move_to_position", "completed").
depends("run_47", "run_47_t4", "run_47_l6").
spawned_by("run_47", "run_47_t4", "run_47_l6").
param("run_47", "run_47_t4", "pos_distance", "0.25").
param("run_47", "run_47_t4", "delta_gripper", "1.3").
param("run_47", "run_47_t4", "efficiency", "1.0").
param("run_47", "run_47_t4", "mean_speed", "0.230").
call("run_47", "run_47_l7", "llm_generate", "completed").
depends("run_47", "run_47_l7", "run_47_t4").
call("run_47", "run_47_t5", "open_gripper", "completed").
depends("run_47", "run_47_t5", "run_47_l7").
spawned_by("run_47", "run_47_t5", "run_47_l7").
param("run_47", "run_47_t5", "pos_distance", "0.17").
param("run_47", "run_47_t5", "delta_gripper", "1.6").
param("run_47", "run_47_t5", "efficiency", "0.8").
param("run_47", "run_47_t5", "mean_speed", "0.160").
call("run_47", "run_47_l8", "llm_generate", "completed").
depends("run_47", "run_47_l8", "run_47_t5").
call("run_47", "run_47_t6", "adjust_position", "completed").
depends("run_47", "run_47_t6", "run_47_l8").
spawned_by("run_47", "run_47_t6", "run_47_l8").
param("run_47", "run_47_t6", "pos_distance", "0.08").
param("run_47", "run_47_t6", "delta_gripper", "0.8").
param("run_47", "run_47_t6", "efficiency", "0.6").
param("run_47", "run_47_t6", "mean_speed", "0.086").
call("run_47", "run_47_l9", "llm_generate", "completed").
depends("run_47", "run_47_l9", "run_47_t6").
call("run_47", "run_47_t7", "adjust_position", "completed").
depends("run_47", "run_47_t7", "run_47_l9").
spawned_by("run_47", "run_47_t7", "run_47_l9").
param("run_47", "run_47_t7", "pos_distance", "0.15").
param("run_47", "run_47_t7", "delta_gripper", "0.0").
param("run_47", "run_47_t7", "efficiency", "0.3").
param("run_47", "run_47_t7", "mean_speed", "0.058").
call("run_47", "run_47_l10", "llm_generate", "completed").
depends("run_47", "run_47_l10", "run_47_t7").
call("run_47", "run_47_n", "notify_user", "completed").
depends("run_47", "run_47_n", "run_47_l10").
spawned_by("run_47", "run_47_n", "run_47_l10").

% === Run 48 (short_execution) ===
job("run_48").
job_status("run_48", "completed").
call("run_48", "run_48_s0", "prompt_user", "completed").
call("run_48", "run_48_s1", "llm_generate", "completed").
depends("run_48", "run_48_s1", "run_48_s0").
call("run_48", "run_48_l2", "llm_generate", "completed").
depends("run_48", "run_48_l2", "run_48_s1").
call("run_48", "run_48_t0", "move_to_position", "completed").
depends("run_48", "run_48_t0", "run_48_l2").
spawned_by("run_48", "run_48_t0", "run_48_l2").
param("run_48", "run_48_t0", "pos_distance", "0.20").
param("run_48", "run_48_t0", "delta_gripper", "0.0").
param("run_48", "run_48_t0", "efficiency", "1.0").
param("run_48", "run_48_t0", "mean_speed", "0.156").
call("run_48", "run_48_l3", "llm_generate", "completed").
depends("run_48", "run_48_l3", "run_48_t0").
call("run_48", "run_48_t1", "close_gripper", "completed").
depends("run_48", "run_48_t1", "run_48_l3").
spawned_by("run_48", "run_48_t1", "run_48_l3").
param("run_48", "run_48_t1", "pos_distance", "0.16").
param("run_48", "run_48_t1", "delta_gripper", "-2.0").
param("run_48", "run_48_t1", "efficiency", "0.9").
param("run_48", "run_48_t1", "mean_speed", "0.128").
call("run_48", "run_48_l4", "llm_generate", "completed").
depends("run_48", "run_48_l4", "run_48_t1").
call("run_48", "run_48_t2", "close_gripper", "completed").
depends("run_48", "run_48_t2", "run_48_l4").
spawned_by("run_48", "run_48_t2", "run_48_l4").
param("run_48", "run_48_t2", "pos_distance", "0.15").
param("run_48", "run_48_t2", "delta_gripper", "-1.8").
param("run_48", "run_48_t2", "efficiency", "1.0").
param("run_48", "run_48_t2", "mean_speed", "0.089").
call("run_48", "run_48_l5", "llm_generate", "completed").
depends("run_48", "run_48_l5", "run_48_t2").
call("run_48", "run_48_t3", "move_to_position", "completed").
depends("run_48", "run_48_t3", "run_48_l5").
spawned_by("run_48", "run_48_t3", "run_48_l5").
param("run_48", "run_48_t3", "pos_distance", "0.70").
param("run_48", "run_48_t3", "delta_gripper", "0.0").
param("run_48", "run_48_t3", "efficiency", "0.7").
param("run_48", "run_48_t3", "mean_speed", "0.154").
call("run_48", "run_48_l6", "llm_generate", "completed").
depends("run_48", "run_48_l6", "run_48_t3").
call("run_48", "run_48_t4", "move_to_position", "completed").
depends("run_48", "run_48_t4", "run_48_l6").
spawned_by("run_48", "run_48_t4", "run_48_l6").
param("run_48", "run_48_t4", "pos_distance", "0.33").
param("run_48", "run_48_t4", "delta_gripper", "1.3").
param("run_48", "run_48_t4", "efficiency", "1.0").
param("run_48", "run_48_t4", "mean_speed", "0.187").
call("run_48", "run_48_l7", "llm_generate", "completed").
depends("run_48", "run_48_l7", "run_48_t4").
call("run_48", "run_48_t5", "open_gripper", "completed").
depends("run_48", "run_48_t5", "run_48_l7").
spawned_by("run_48", "run_48_t5", "run_48_l7").
param("run_48", "run_48_t5", "pos_distance", "0.13").
param("run_48", "run_48_t5", "delta_gripper", "1.6").
param("run_48", "run_48_t5", "efficiency", "0.8").
param("run_48", "run_48_t5", "mean_speed", "0.156").
call("run_48", "run_48_l8", "llm_generate", "completed").
depends("run_48", "run_48_l8", "run_48_t5").
call("run_48", "run_48_t6", "adjust_position", "completed").
depends("run_48", "run_48_t6", "run_48_l8").
spawned_by("run_48", "run_48_t6", "run_48_l8").
param("run_48", "run_48_t6", "pos_distance", "0.06").
param("run_48", "run_48_t6", "delta_gripper", "0.8").
param("run_48", "run_48_t6", "efficiency", "0.6").
param("run_48", "run_48_t6", "mean_speed", "0.116").
call("run_48", "run_48_l9", "llm_generate", "completed").
depends("run_48", "run_48_l9", "run_48_t6").
call("run_48", "run_48_t7", "adjust_position", "completed").
depends("run_48", "run_48_t7", "run_48_l9").
spawned_by("run_48", "run_48_t7", "run_48_l9").
param("run_48", "run_48_t7", "pos_distance", "0.17").
param("run_48", "run_48_t7", "delta_gripper", "0.0").
param("run_48", "run_48_t7", "efficiency", "0.3").
param("run_48", "run_48_t7", "mean_speed", "0.053").
call("run_48", "run_48_l10", "llm_generate", "completed").
depends("run_48", "run_48_l10", "run_48_t7").
call("run_48", "run_48_n", "notify_user", "completed").
depends("run_48", "run_48_n", "run_48_l10").
spawned_by("run_48", "run_48_n", "run_48_l10").

% === Run 49 (short_execution) ===
job("run_49").
job_status("run_49", "completed").
call("run_49", "run_49_s0", "prompt_user", "completed").
call("run_49", "run_49_s1", "llm_generate", "completed").
depends("run_49", "run_49_s1", "run_49_s0").
call("run_49", "run_49_l2", "llm_generate", "completed").
depends("run_49", "run_49_l2", "run_49_s1").
call("run_49", "run_49_t0", "move_to_position", "completed").
depends("run_49", "run_49_t0", "run_49_l2").
spawned_by("run_49", "run_49_t0", "run_49_l2").
param("run_49", "run_49_t0", "pos_distance", "0.37").
param("run_49", "run_49_t0", "delta_gripper", "0.0").
param("run_49", "run_49_t0", "efficiency", "1.0").
param("run_49", "run_49_t0", "mean_speed", "0.184").
call("run_49", "run_49_l3", "llm_generate", "completed").
depends("run_49", "run_49_l3", "run_49_t0").
call("run_49", "run_49_t1", "close_gripper", "completed").
depends("run_49", "run_49_t1", "run_49_l3").
spawned_by("run_49", "run_49_t1", "run_49_l3").
param("run_49", "run_49_t1", "pos_distance", "0.27").
param("run_49", "run_49_t1", "delta_gripper", "-2.0").
param("run_49", "run_49_t1", "efficiency", "0.9").
param("run_49", "run_49_t1", "mean_speed", "0.123").
call("run_49", "run_49_l4", "llm_generate", "completed").
depends("run_49", "run_49_l4", "run_49_t1").
call("run_49", "run_49_t2", "close_gripper", "completed").
depends("run_49", "run_49_t2", "run_49_l4").
spawned_by("run_49", "run_49_t2", "run_49_l4").
param("run_49", "run_49_t2", "pos_distance", "0.08").
param("run_49", "run_49_t2", "delta_gripper", "-1.8").
param("run_49", "run_49_t2", "efficiency", "1.0").
param("run_49", "run_49_t2", "mean_speed", "0.092").
call("run_49", "run_49_l5", "llm_generate", "completed").
depends("run_49", "run_49_l5", "run_49_t2").
call("run_49", "run_49_t3", "move_to_position", "completed").
depends("run_49", "run_49_t3", "run_49_l5").
spawned_by("run_49", "run_49_t3", "run_49_l5").
param("run_49", "run_49_t3", "pos_distance", "0.93").
param("run_49", "run_49_t3", "delta_gripper", "0.0").
param("run_49", "run_49_t3", "efficiency", "0.7").
param("run_49", "run_49_t3", "mean_speed", "0.242").
call("run_49", "run_49_l6", "llm_generate", "completed").
depends("run_49", "run_49_l6", "run_49_t3").
call("run_49", "run_49_t4", "move_to_position", "completed").
depends("run_49", "run_49_t4", "run_49_l6").
spawned_by("run_49", "run_49_t4", "run_49_l6").
param("run_49", "run_49_t4", "pos_distance", "0.38").
param("run_49", "run_49_t4", "delta_gripper", "1.3").
param("run_49", "run_49_t4", "efficiency", "1.0").
param("run_49", "run_49_t4", "mean_speed", "0.171").
call("run_49", "run_49_l7", "llm_generate", "completed").
depends("run_49", "run_49_l7", "run_49_t4").
call("run_49", "run_49_t5", "open_gripper", "completed").
depends("run_49", "run_49_t5", "run_49_l7").
spawned_by("run_49", "run_49_t5", "run_49_l7").
param("run_49", "run_49_t5", "pos_distance", "0.18").
param("run_49", "run_49_t5", "delta_gripper", "1.6").
param("run_49", "run_49_t5", "efficiency", "0.8").
param("run_49", "run_49_t5", "mean_speed", "0.128").
call("run_49", "run_49_l8", "llm_generate", "completed").
depends("run_49", "run_49_l8", "run_49_t5").
call("run_49", "run_49_t6", "adjust_position", "completed").
depends("run_49", "run_49_t6", "run_49_l8").
spawned_by("run_49", "run_49_t6", "run_49_l8").
param("run_49", "run_49_t6", "pos_distance", "0.12").
param("run_49", "run_49_t6", "delta_gripper", "0.8").
param("run_49", "run_49_t6", "efficiency", "0.6").
param("run_49", "run_49_t6", "mean_speed", "0.094").
call("run_49", "run_49_l9", "llm_generate", "completed").
depends("run_49", "run_49_l9", "run_49_t6").
call("run_49", "run_49_t7", "adjust_position", "completed").
depends("run_49", "run_49_t7", "run_49_l9").
spawned_by("run_49", "run_49_t7", "run_49_l9").
param("run_49", "run_49_t7", "pos_distance", "0.16").
param("run_49", "run_49_t7", "delta_gripper", "0.0").
param("run_49", "run_49_t7", "efficiency", "0.3").
param("run_49", "run_49_t7", "mean_speed", "0.034").
call("run_49", "run_49_l10", "llm_generate", "completed").
depends("run_49", "run_49_l10", "run_49_t7").
call("run_49", "run_49_n", "notify_user", "completed").
depends("run_49", "run_49_n", "run_49_l10").
spawned_by("run_49", "run_49_n", "run_49_l10").


% === World Model Constraints (from LLM) ===
% These are physical invariants that the LLM knows from world knowledge.
% They prune the search space — invalid programs are ruled out.

% Invariant: open_before_grasp — Gripper must be open before closing on object
% Invariant: at_object_before_grasp — Arm must be near the object before closing gripper
% Invariant: hold_during_transport — Gripper must stay closed while transporting
% Invariant: at_target_before_release — Arm must be at target position before opening gripper
% Invariant: sequence_order — Must follow: approach → grasp → transport → release