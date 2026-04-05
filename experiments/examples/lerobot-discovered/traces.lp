% === LeRobot Pick-and-Place: Discovered Primitives ===
% Segments discovered via changepoint detection + state-transition clustering
% NO hardcoded labels — primitive names derived from behavior
% Source: lerobot/ucsd_pick_and_place_dataset (HuggingFace)

% === Episode 0 (8 discovered segments) ===
job("run_1").
job_status("run_1", "completed").
call("run_1", "run_1_step_0", "prompt_user", "completed").
call("run_1", "run_1_step_1", "llm_generate", "completed").
depends("run_1", "run_1_step_1", "run_1_step_0").
call("run_1", "run_1_llm_2", "llm_generate", "completed").
depends("run_1", "run_1_llm_2", "run_1_step_1").
call("run_1", "run_1_step_2", "move_x_pos", "completed").
depends("run_1", "run_1_step_2", "run_1_llm_2").
spawned_by("run_1", "run_1_step_2", "run_1_llm_2").
param("run_1", "run_1_step_2", "pos_distance", "0.785").
param("run_1", "run_1_step_2", "delta_gripper", "0.01").
param("run_1", "run_1_step_2", "efficiency", "0.97").
param("run_1", "run_1_step_2", "mean_speed", "0.2022").
param("run_1", "run_1_step_2", "n_frames", "5").
param("run_1", "run_1_step_2", "start_gripper", "4.0").
param("run_1", "run_1_step_2", "end_gripper", "4.0").
call("run_1", "run_1_llm_3", "llm_generate", "completed").
depends("run_1", "run_1_llm_3", "run_1_step_2").
call("run_1", "run_1_step_3", "move_x_pos", "completed").
depends("run_1", "run_1_step_3", "run_1_llm_3").
spawned_by("run_1", "run_1_step_3", "run_1_llm_3").
param("run_1", "run_1_step_3", "pos_distance", "0.463").
param("run_1", "run_1_step_3", "delta_gripper", "-1.96").
param("run_1", "run_1_step_3", "efficiency", "0.93").
param("run_1", "run_1_step_3", "mean_speed", "0.1652").
param("run_1", "run_1_step_3", "n_frames", "4").
param("run_1", "run_1_step_3", "start_gripper", "4.0").
param("run_1", "run_1_step_3", "end_gripper", "2.0").
call("run_1", "run_1_llm_4", "llm_generate", "completed").
depends("run_1", "run_1_llm_4", "run_1_step_3").
call("run_1", "run_1_step_4", "adjust_position", "completed").
depends("run_1", "run_1_step_4", "run_1_llm_4").
spawned_by("run_1", "run_1_step_4", "run_1_llm_4").
param("run_1", "run_1_step_4", "pos_distance", "0.358").
param("run_1", "run_1_step_4", "delta_gripper", "-1.76").
param("run_1", "run_1_step_4", "efficiency", "0.97").
param("run_1", "run_1_step_4", "mean_speed", "0.1227").
param("run_1", "run_1_step_4", "n_frames", "4").
param("run_1", "run_1_step_4", "start_gripper", "2.0").
param("run_1", "run_1_step_4", "end_gripper", "0.3").
call("run_1", "run_1_llm_5", "llm_generate", "completed").
depends("run_1", "run_1_llm_5", "run_1_step_4").
call("run_1", "run_1_step_5", "move_x_pos_2", "completed").
depends("run_1", "run_1_step_5", "run_1_llm_5").
spawned_by("run_1", "run_1_step_5", "run_1_llm_5").
param("run_1", "run_1_step_5", "pos_distance", "2.129").
param("run_1", "run_1_step_5", "delta_gripper", "0.00").
param("run_1", "run_1_step_5", "efficiency", "0.75").
param("run_1", "run_1_step_5", "mean_speed", "0.2594").
param("run_1", "run_1_step_5", "n_frames", "12").
param("run_1", "run_1_step_5", "start_gripper", "0.3").
param("run_1", "run_1_step_5", "end_gripper", "0.3").
call("run_1", "run_1_llm_6", "llm_generate", "completed").
depends("run_1", "run_1_llm_6", "run_1_step_5").
call("run_1", "run_1_step_6", "move_x_pos", "completed").
depends("run_1", "run_1_step_6", "run_1_llm_6").
spawned_by("run_1", "run_1_step_6", "run_1_llm_6").
param("run_1", "run_1_step_6", "pos_distance", "0.690").
param("run_1", "run_1_step_6", "delta_gripper", "1.26").
param("run_1", "run_1_step_6", "efficiency", "0.97").
param("run_1", "run_1_step_6", "mean_speed", "0.2377").
param("run_1", "run_1_step_6", "n_frames", "4").
param("run_1", "run_1_step_6", "start_gripper", "0.3").
param("run_1", "run_1_step_6", "end_gripper", "1.5").
call("run_1", "run_1_llm_7", "llm_generate", "completed").
depends("run_1", "run_1_llm_7", "run_1_step_6").
call("run_1", "run_1_step_7", "move_x_pos", "completed").
depends("run_1", "run_1_step_7", "run_1_llm_7").
spawned_by("run_1", "run_1_step_7", "run_1_llm_7").
param("run_1", "run_1_step_7", "pos_distance", "0.433").
param("run_1", "run_1_step_7", "delta_gripper", "1.64").
param("run_1", "run_1_step_7", "efficiency", "0.82").
param("run_1", "run_1_step_7", "mean_speed", "0.1756").
param("run_1", "run_1_step_7", "n_frames", "4").
param("run_1", "run_1_step_7", "start_gripper", "1.5").
param("run_1", "run_1_step_7", "end_gripper", "3.2").
call("run_1", "run_1_llm_8", "llm_generate", "completed").
depends("run_1", "run_1_llm_8", "run_1_step_7").
call("run_1", "run_1_step_8", "adjust_position_2", "completed").
depends("run_1", "run_1_step_8", "run_1_llm_8").
spawned_by("run_1", "run_1_step_8", "run_1_llm_8").
param("run_1", "run_1_step_8", "pos_distance", "0.254").
param("run_1", "run_1_step_8", "delta_gripper", "0.79").
param("run_1", "run_1_step_8", "efficiency", "0.64").
param("run_1", "run_1_step_8", "mean_speed", "0.1330").
param("run_1", "run_1_step_8", "n_frames", "4").
param("run_1", "run_1_step_8", "start_gripper", "3.2").
param("run_1", "run_1_step_8", "end_gripper", "4.0").
call("run_1", "run_1_llm_9", "llm_generate", "completed").
depends("run_1", "run_1_llm_9", "run_1_step_8").
call("run_1", "run_1_step_9", "adjust_position_2", "completed").
depends("run_1", "run_1_step_9", "run_1_llm_9").
spawned_by("run_1", "run_1_step_9", "run_1_llm_9").
param("run_1", "run_1_step_9", "pos_distance", "0.347").
param("run_1", "run_1_step_9", "delta_gripper", "0.01").
param("run_1", "run_1_step_9", "efficiency", "0.31").
param("run_1", "run_1_step_9", "mean_speed", "0.0583").
param("run_1", "run_1_step_9", "n_frames", "20").
param("run_1", "run_1_step_9", "start_gripper", "4.0").
param("run_1", "run_1_step_9", "end_gripper", "4.0").
call("run_1", "run_1_llm_10", "llm_generate", "completed").
depends("run_1", "run_1_llm_10", "run_1_step_9").
call("run_1", "run_1_notify", "notify_user", "completed").
depends("run_1", "run_1_notify", "run_1_llm_10").
spawned_by("run_1", "run_1_notify", "run_1_llm_10").

% === Episode 1 (5 discovered segments) ===
job("run_2").
job_status("run_2", "completed").
call("run_2", "run_2_step_0", "prompt_user", "completed").
call("run_2", "run_2_step_1", "llm_generate", "completed").
depends("run_2", "run_2_step_1", "run_2_step_0").
call("run_2", "run_2_llm_2", "llm_generate", "completed").
depends("run_2", "run_2_llm_2", "run_2_step_1").
call("run_2", "run_2_step_2", "move_y_neg", "completed").
depends("run_2", "run_2_step_2", "run_2_llm_2").
spawned_by("run_2", "run_2_step_2", "run_2_llm_2").
param("run_2", "run_2_step_2", "pos_distance", "0.813").
param("run_2", "run_2_step_2", "delta_gripper", "1.70").
param("run_2", "run_2_step_2", "efficiency", "0.96").
param("run_2", "run_2_step_2", "mean_speed", "0.2825").
param("run_2", "run_2_step_2", "n_frames", "4").
param("run_2", "run_2_step_2", "start_gripper", "6.0").
param("run_2", "run_2_step_2", "end_gripper", "7.7").
call("run_2", "run_2_llm_3", "llm_generate", "completed").
depends("run_2", "run_2_llm_3", "run_2_step_2").
call("run_2", "run_2_step_3", "move_y_neg", "completed").
depends("run_2", "run_2_step_3", "run_2_llm_3").
spawned_by("run_2", "run_2_step_3", "run_2_llm_3").
param("run_2", "run_2_step_3", "pos_distance", "0.813").
param("run_2", "run_2_step_3", "delta_gripper", "0.22").
param("run_2", "run_2_step_3", "efficiency", "0.99").
param("run_2", "run_2_step_3", "mean_speed", "0.2727").
param("run_2", "run_2_step_3", "n_frames", "4").
param("run_2", "run_2_step_3", "start_gripper", "7.7").
param("run_2", "run_2_step_3", "end_gripper", "8.0").
call("run_2", "run_2_llm_4", "llm_generate", "completed").
depends("run_2", "run_2_llm_4", "run_2_step_3").
call("run_2", "run_2_step_4", "move_y_neg", "completed").
depends("run_2", "run_2_step_4", "run_2_llm_4").
spawned_by("run_2", "run_2_step_4", "run_2_llm_4").
param("run_2", "run_2_step_4", "pos_distance", "0.914").
param("run_2", "run_2_step_4", "delta_gripper", "0.00").
param("run_2", "run_2_step_4", "efficiency", "1.00").
param("run_2", "run_2_step_4", "mean_speed", "0.3047").
param("run_2", "run_2_step_4", "n_frames", "4").
param("run_2", "run_2_step_4", "start_gripper", "8.0").
param("run_2", "run_2_step_4", "end_gripper", "8.0").
call("run_2", "run_2_llm_5", "llm_generate", "completed").
depends("run_2", "run_2_llm_5", "run_2_step_4").
call("run_2", "run_2_step_5", "close_gripper", "completed").
depends("run_2", "run_2_step_5", "run_2_llm_5").
spawned_by("run_2", "run_2_step_5", "run_2_llm_5").
param("run_2", "run_2_step_5", "pos_distance", "0.110").
param("run_2", "run_2_step_5", "delta_gripper", "-3.15").
param("run_2", "run_2_step_5", "efficiency", "0.30").
param("run_2", "run_2_step_5", "mean_speed", "0.1226").
param("run_2", "run_2_step_5", "n_frames", "4").
param("run_2", "run_2_step_5", "start_gripper", "8.0").
param("run_2", "run_2_step_5", "end_gripper", "4.8").
call("run_2", "run_2_llm_6", "llm_generate", "completed").
depends("run_2", "run_2_llm_6", "run_2_step_5").
call("run_2", "run_2_step_6", "adjust_position_3", "completed").
depends("run_2", "run_2_step_6", "run_2_llm_6").
spawned_by("run_2", "run_2_step_6", "run_2_llm_6").
param("run_2", "run_2_step_6", "pos_distance", "1.755").
param("run_2", "run_2_step_6", "delta_gripper", "-1.18").
param("run_2", "run_2_step_6", "efficiency", "0.50").
param("run_2", "run_2_step_6", "mean_speed", "0.0941").
param("run_2", "run_2_step_6", "n_frames", "38").
param("run_2", "run_2_step_6", "start_gripper", "4.8").
param("run_2", "run_2_step_6", "end_gripper", "3.6").
call("run_2", "run_2_llm_7", "llm_generate", "completed").
depends("run_2", "run_2_llm_7", "run_2_step_6").
call("run_2", "run_2_notify", "notify_user", "completed").
depends("run_2", "run_2_notify", "run_2_llm_7").
spawned_by("run_2", "run_2_notify", "run_2_llm_7").

% === Episode 2 (5 discovered segments) ===
job("run_3").
job_status("run_3", "completed").
call("run_3", "run_3_step_0", "prompt_user", "completed").
call("run_3", "run_3_step_1", "llm_generate", "completed").
depends("run_3", "run_3_step_1", "run_3_step_0").
call("run_3", "run_3_llm_2", "llm_generate", "completed").
depends("run_3", "run_3_llm_2", "run_3_step_1").
call("run_3", "run_3_step_2", "move_x_pos", "completed").
depends("run_3", "run_3_step_2", "run_3_llm_2").
spawned_by("run_3", "run_3_step_2", "run_3_llm_2").
param("run_3", "run_3_step_2", "pos_distance", "0.418").
param("run_3", "run_3_step_2", "delta_gripper", "1.51").
param("run_3", "run_3_step_2", "efficiency", "1.00").
param("run_3", "run_3_step_2", "mean_speed", "0.1401").
param("run_3", "run_3_step_2", "n_frames", "4").
param("run_3", "run_3_step_2", "start_gripper", "6.0").
param("run_3", "run_3_step_2", "end_gripper", "7.5").
call("run_3", "run_3_llm_3", "llm_generate", "completed").
depends("run_3", "run_3_llm_3", "run_3_step_2").
call("run_3", "run_3_step_3", "move_y_neg_2", "completed").
depends("run_3", "run_3_step_3", "run_3_llm_3").
spawned_by("run_3", "run_3_step_3", "run_3_llm_3").
param("run_3", "run_3_step_3", "pos_distance", "2.582").
param("run_3", "run_3_step_3", "delta_gripper", "0.42").
param("run_3", "run_3_step_3", "efficiency", "0.97").
param("run_3", "run_3_step_3", "mean_speed", "0.2674").
param("run_3", "run_3_step_3", "n_frames", "11").
param("run_3", "run_3_step_3", "start_gripper", "7.5").
param("run_3", "run_3_step_3", "end_gripper", "8.0").
call("run_3", "run_3_llm_4", "llm_generate", "completed").
depends("run_3", "run_3_llm_4", "run_3_step_3").
call("run_3", "run_3_step_4", "adjust_position", "completed").
depends("run_3", "run_3_step_4", "run_3_llm_4").
spawned_by("run_3", "run_3_step_4", "run_3_llm_4").
param("run_3", "run_3_step_4", "pos_distance", "0.356").
param("run_3", "run_3_step_4", "delta_gripper", "-1.36").
param("run_3", "run_3_step_4", "efficiency", "0.99").
param("run_3", "run_3_step_4", "mean_speed", "0.1194").
param("run_3", "run_3_step_4", "n_frames", "4").
param("run_3", "run_3_step_4", "start_gripper", "8.0").
param("run_3", "run_3_step_4", "end_gripper", "6.6").
call("run_3", "run_3_llm_5", "llm_generate", "completed").
depends("run_3", "run_3_llm_5", "run_3_step_4").
call("run_3", "run_3_step_5", "close_gripper", "completed").
depends("run_3", "run_3_step_5", "run_3_llm_5").
spawned_by("run_3", "run_3_step_5", "run_3_llm_5").
param("run_3", "run_3_step_5", "pos_distance", "0.014").
param("run_3", "run_3_step_5", "delta_gripper", "-3.25").
param("run_3", "run_3_step_5", "efficiency", "0.41").
param("run_3", "run_3_step_5", "mean_speed", "0.0114").
param("run_3", "run_3_step_5", "n_frames", "4").
param("run_3", "run_3_step_5", "start_gripper", "6.6").
param("run_3", "run_3_step_5", "end_gripper", "3.3").
call("run_3", "run_3_llm_6", "llm_generate", "completed").
depends("run_3", "run_3_llm_6", "run_3_step_5").
call("run_3", "run_3_step_6", "adjust_position_4", "completed").
depends("run_3", "run_3_step_6", "run_3_llm_6").
spawned_by("run_3", "run_3_step_6", "run_3_llm_6").
param("run_3", "run_3_step_6", "pos_distance", "1.381").
param("run_3", "run_3_step_6", "delta_gripper", "-0.02").
param("run_3", "run_3_step_6", "efficiency", "0.86").
param("run_3", "run_3_step_6", "mean_speed", "0.0533").
param("run_3", "run_3_step_6", "n_frames", "31").
param("run_3", "run_3_step_6", "start_gripper", "3.3").
param("run_3", "run_3_step_6", "end_gripper", "3.3").
call("run_3", "run_3_llm_7", "llm_generate", "completed").
depends("run_3", "run_3_llm_7", "run_3_step_6").
call("run_3", "run_3_notify", "notify_user", "completed").
depends("run_3", "run_3_notify", "run_3_llm_7").
spawned_by("run_3", "run_3_notify", "run_3_llm_7").

% === Episode 3 (5 discovered segments) ===
job("run_4").
job_status("run_4", "completed").
call("run_4", "run_4_step_0", "prompt_user", "completed").
call("run_4", "run_4_step_1", "llm_generate", "completed").
depends("run_4", "run_4_step_1", "run_4_step_0").
call("run_4", "run_4_llm_2", "llm_generate", "completed").
depends("run_4", "run_4_llm_2", "run_4_step_1").
call("run_4", "run_4_step_2", "move_x_pos", "completed").
depends("run_4", "run_4_step_2", "run_4_llm_2").
spawned_by("run_4", "run_4_step_2", "run_4_llm_2").
param("run_4", "run_4_step_2", "pos_distance", "0.472").
param("run_4", "run_4_step_2", "delta_gripper", "1.69").
param("run_4", "run_4_step_2", "efficiency", "1.00").
param("run_4", "run_4_step_2", "mean_speed", "0.1579").
param("run_4", "run_4_step_2", "n_frames", "4").
param("run_4", "run_4_step_2", "start_gripper", "6.0").
param("run_4", "run_4_step_2", "end_gripper", "7.7").
call("run_4", "run_4_llm_3", "llm_generate", "completed").
depends("run_4", "run_4_llm_3", "run_4_step_2").
call("run_4", "run_4_step_3", "move_y_neg", "completed").
depends("run_4", "run_4_step_3", "run_4_llm_3").
spawned_by("run_4", "run_4_step_3", "run_4_llm_3").
param("run_4", "run_4_step_3", "pos_distance", "1.182").
param("run_4", "run_4_step_3", "delta_gripper", "0.24").
param("run_4", "run_4_step_3", "efficiency", "0.99").
param("run_4", "run_4_step_3", "mean_speed", "0.2983").
param("run_4", "run_4_step_3", "n_frames", "5").
param("run_4", "run_4_step_3", "start_gripper", "7.7").
param("run_4", "run_4_step_3", "end_gripper", "8.0").
call("run_4", "run_4_llm_4", "llm_generate", "completed").
depends("run_4", "run_4_llm_4", "run_4_step_3").
call("run_4", "run_4_step_4", "move_x_pos", "completed").
depends("run_4", "run_4_step_4", "run_4_llm_4").
spawned_by("run_4", "run_4_step_4", "run_4_llm_4").
param("run_4", "run_4_step_4", "pos_distance", "0.521").
param("run_4", "run_4_step_4", "delta_gripper", "-1.15").
param("run_4", "run_4_step_4", "efficiency", "0.94").
param("run_4", "run_4_step_4", "mean_speed", "0.1838").
param("run_4", "run_4_step_4", "n_frames", "4").
param("run_4", "run_4_step_4", "start_gripper", "8.0").
param("run_4", "run_4_step_4", "end_gripper", "6.8").
call("run_4", "run_4_llm_5", "llm_generate", "completed").
depends("run_4", "run_4_llm_5", "run_4_step_4").
call("run_4", "run_4_step_5", "adjust_position", "completed").
depends("run_4", "run_4_step_5", "run_4_llm_5").
spawned_by("run_4", "run_4_step_5", "run_4_llm_5").
param("run_4", "run_4_step_5", "pos_distance", "0.138").
param("run_4", "run_4_step_5", "delta_gripper", "-3.10").
param("run_4", "run_4_step_5", "efficiency", "0.99").
param("run_4", "run_4_step_5", "mean_speed", "0.0462").
param("run_4", "run_4_step_5", "n_frames", "4").
param("run_4", "run_4_step_5", "start_gripper", "6.8").
param("run_4", "run_4_step_5", "end_gripper", "3.7").
call("run_4", "run_4_llm_6", "llm_generate", "completed").
depends("run_4", "run_4_llm_6", "run_4_step_5").
call("run_4", "run_4_step_6", "adjust_position_4", "completed").
depends("run_4", "run_4_step_6", "run_4_llm_6").
spawned_by("run_4", "run_4_step_6", "run_4_llm_6").
param("run_4", "run_4_step_6", "pos_distance", "1.361").
param("run_4", "run_4_step_6", "delta_gripper", "-0.03").
param("run_4", "run_4_step_6", "efficiency", "0.75").
param("run_4", "run_4_step_6", "mean_speed", "0.0502").
param("run_4", "run_4_step_6", "n_frames", "37").
param("run_4", "run_4_step_6", "start_gripper", "3.7").
param("run_4", "run_4_step_6", "end_gripper", "3.7").
call("run_4", "run_4_llm_7", "llm_generate", "completed").
depends("run_4", "run_4_llm_7", "run_4_step_6").
call("run_4", "run_4_notify", "notify_user", "completed").
depends("run_4", "run_4_notify", "run_4_llm_7").
spawned_by("run_4", "run_4_notify", "run_4_llm_7").

% === Episode 4 (6 discovered segments) ===
job("run_5").
job_status("run_5", "completed").
call("run_5", "run_5_step_0", "prompt_user", "completed").
call("run_5", "run_5_step_1", "llm_generate", "completed").
depends("run_5", "run_5_step_1", "run_5_step_0").
call("run_5", "run_5_llm_2", "llm_generate", "completed").
depends("run_5", "run_5_llm_2", "run_5_step_1").
call("run_5", "run_5_step_2", "adjust_position_5", "completed").
depends("run_5", "run_5_step_2", "run_5_llm_2").
spawned_by("run_5", "run_5_step_2", "run_5_llm_2").
param("run_5", "run_5_step_2", "pos_distance", "0.234").
param("run_5", "run_5_step_2", "delta_gripper", "1.92").
param("run_5", "run_5_step_2", "efficiency", "0.97").
param("run_5", "run_5_step_2", "mean_speed", "0.0601").
param("run_5", "run_5_step_2", "n_frames", "5").
param("run_5", "run_5_step_2", "start_gripper", "6.0").
param("run_5", "run_5_step_2", "end_gripper", "8.0").
call("run_5", "run_5_llm_3", "llm_generate", "completed").
depends("run_5", "run_5_llm_3", "run_5_step_2").
call("run_5", "run_5_step_3", "move_x_pos", "completed").
depends("run_5", "run_5_step_3", "run_5_llm_3").
spawned_by("run_5", "run_5_step_3", "run_5_llm_3").
param("run_5", "run_5_step_3", "pos_distance", "0.792").
param("run_5", "run_5_step_3", "delta_gripper", "0.00").
param("run_5", "run_5_step_3", "efficiency", "0.99").
param("run_5", "run_5_step_3", "mean_speed", "0.2659").
param("run_5", "run_5_step_3", "n_frames", "4").
param("run_5", "run_5_step_3", "start_gripper", "8.0").
param("run_5", "run_5_step_3", "end_gripper", "8.0").
call("run_5", "run_5_llm_4", "llm_generate", "completed").
depends("run_5", "run_5_llm_4", "run_5_step_3").
call("run_5", "run_5_step_4", "adjust_position_6", "completed").
depends("run_5", "run_5_step_4", "run_5_llm_4").
spawned_by("run_5", "run_5_step_4", "run_5_llm_4").
param("run_5", "run_5_step_4", "pos_distance", "1.327").
param("run_5", "run_5_step_4", "delta_gripper", "0.00").
param("run_5", "run_5_step_4", "efficiency", "0.59").
param("run_5", "run_5_step_4", "mean_speed", "0.0826").
param("run_5", "run_5_step_4", "n_frames", "28").
param("run_5", "run_5_step_4", "start_gripper", "8.0").
param("run_5", "run_5_step_4", "end_gripper", "8.0").
call("run_5", "run_5_llm_5", "llm_generate", "completed").
depends("run_5", "run_5_llm_5", "run_5_step_4").
call("run_5", "run_5_step_5", "adjust_position", "completed").
depends("run_5", "run_5_step_5", "run_5_llm_5").
spawned_by("run_5", "run_5_step_5", "run_5_llm_5").
param("run_5", "run_5_step_5", "pos_distance", "0.118").
param("run_5", "run_5_step_5", "delta_gripper", "-1.38").
param("run_5", "run_5_step_5", "efficiency", "0.99").
param("run_5", "run_5_step_5", "mean_speed", "0.0397").
param("run_5", "run_5_step_5", "n_frames", "4").
param("run_5", "run_5_step_5", "start_gripper", "8.0").
param("run_5", "run_5_step_5", "end_gripper", "6.6").
call("run_5", "run_5_llm_6", "llm_generate", "completed").
depends("run_5", "run_5_llm_6", "run_5_step_5").
call("run_5", "run_5_step_6", "close_gripper", "completed").
depends("run_5", "run_5_step_6", "run_5_llm_6").
spawned_by("run_5", "run_5_step_6", "run_5_llm_6").
param("run_5", "run_5_step_6", "pos_distance", "0.034").
param("run_5", "run_5_step_6", "delta_gripper", "-3.31").
param("run_5", "run_5_step_6", "efficiency", "0.49").
param("run_5", "run_5_step_6", "mean_speed", "0.0234").
param("run_5", "run_5_step_6", "n_frames", "4").
param("run_5", "run_5_step_6", "start_gripper", "6.6").
param("run_5", "run_5_step_6", "end_gripper", "3.3").
call("run_5", "run_5_llm_7", "llm_generate", "completed").
depends("run_5", "run_5_llm_7", "run_5_step_6").
call("run_5", "run_5_step_7", "adjust_position_4", "completed").
depends("run_5", "run_5_step_7", "run_5_llm_7").
spawned_by("run_5", "run_5_step_7", "run_5_llm_7").
param("run_5", "run_5_step_7", "pos_distance", "1.013").
param("run_5", "run_5_step_7", "delta_gripper", "0.00").
param("run_5", "run_5_step_7", "efficiency", "0.98").
param("run_5", "run_5_step_7", "mean_speed", "0.1144").
param("run_5", "run_5_step_7", "n_frames", "10").
param("run_5", "run_5_step_7", "start_gripper", "3.3").
param("run_5", "run_5_step_7", "end_gripper", "3.3").
call("run_5", "run_5_llm_8", "llm_generate", "completed").
depends("run_5", "run_5_llm_8", "run_5_step_7").
call("run_5", "run_5_notify", "notify_user", "completed").
depends("run_5", "run_5_notify", "run_5_llm_8").
spawned_by("run_5", "run_5_notify", "run_5_llm_8").

% === Episode 5 (8 discovered segments) ===
job("run_6").
job_status("run_6", "completed").
call("run_6", "run_6_step_0", "prompt_user", "completed").
call("run_6", "run_6_step_1", "llm_generate", "completed").
depends("run_6", "run_6_step_1", "run_6_step_0").
call("run_6", "run_6_llm_2", "llm_generate", "completed").
depends("run_6", "run_6_llm_2", "run_6_step_1").
call("run_6", "run_6_step_2", "adjust_position", "completed").
depends("run_6", "run_6_step_2", "run_6_llm_2").
spawned_by("run_6", "run_6_step_2", "run_6_llm_2").
param("run_6", "run_6_step_2", "pos_distance", "0.221").
param("run_6", "run_6_step_2", "delta_gripper", "0.14").
param("run_6", "run_6_step_2", "efficiency", "0.92").
param("run_6", "run_6_step_2", "mean_speed", "0.0799").
param("run_6", "run_6_step_2", "n_frames", "4").
param("run_6", "run_6_step_2", "start_gripper", "6.0").
param("run_6", "run_6_step_2", "end_gripper", "6.1").
call("run_6", "run_6_llm_3", "llm_generate", "completed").
depends("run_6", "run_6_llm_3", "run_6_step_2").
call("run_6", "run_6_step_3", "adjust_position_2", "completed").
depends("run_6", "run_6_step_3", "run_6_llm_3").
spawned_by("run_6", "run_6_step_3", "run_6_llm_3").
param("run_6", "run_6_step_3", "pos_distance", "0.409").
param("run_6", "run_6_step_3", "delta_gripper", "-1.49").
param("run_6", "run_6_step_3", "efficiency", "0.60").
param("run_6", "run_6_step_3", "mean_speed", "0.1144").
param("run_6", "run_6_step_3", "n_frames", "7").
param("run_6", "run_6_step_3", "start_gripper", "6.1").
param("run_6", "run_6_step_3", "end_gripper", "4.6").
call("run_6", "run_6_llm_4", "llm_generate", "completed").
depends("run_6", "run_6_llm_4", "run_6_step_3").
call("run_6", "run_6_step_4", "move_x_pos", "completed").
depends("run_6", "run_6_step_4", "run_6_llm_4").
spawned_by("run_6", "run_6_step_4", "run_6_llm_4").
param("run_6", "run_6_step_4", "pos_distance", "0.890").
param("run_6", "run_6_step_4", "delta_gripper", "0.29").
param("run_6", "run_6_step_4", "efficiency", "0.91").
param("run_6", "run_6_step_4", "mean_speed", "0.1949").
param("run_6", "run_6_step_4", "n_frames", "6").
param("run_6", "run_6_step_4", "start_gripper", "4.6").
param("run_6", "run_6_step_4", "end_gripper", "4.9").
call("run_6", "run_6_llm_5", "llm_generate", "completed").
depends("run_6", "run_6_llm_5", "run_6_step_4").
call("run_6", "run_6_step_5", "close_gripper", "completed").
depends("run_6", "run_6_step_5", "run_6_llm_5").
spawned_by("run_6", "run_6_step_5", "run_6_llm_5").
param("run_6", "run_6_step_5", "pos_distance", "0.292").
param("run_6", "run_6_step_5", "delta_gripper", "-0.77").
param("run_6", "run_6_step_5", "efficiency", "0.13").
param("run_6", "run_6_step_5", "mean_speed", "0.1424").
param("run_6", "run_6_step_5", "n_frames", "17").
param("run_6", "run_6_step_5", "start_gripper", "4.9").
param("run_6", "run_6_step_5", "end_gripper", "4.1").
call("run_6", "run_6_llm_6", "llm_generate", "completed").
depends("run_6", "run_6_llm_6", "run_6_step_5").
call("run_6", "run_6_step_6", "adjust_position", "completed").
depends("run_6", "run_6_step_6", "run_6_llm_6").
spawned_by("run_6", "run_6_step_6", "run_6_llm_6").
param("run_6", "run_6_step_6", "pos_distance", "0.408").
param("run_6", "run_6_step_6", "delta_gripper", "-2.01").
param("run_6", "run_6_step_6", "efficiency", "0.75").
param("run_6", "run_6_step_6", "mean_speed", "0.1808").
param("run_6", "run_6_step_6", "n_frames", "4").
param("run_6", "run_6_step_6", "start_gripper", "4.1").
param("run_6", "run_6_step_6", "end_gripper", "2.1").
call("run_6", "run_6_llm_7", "llm_generate", "completed").
depends("run_6", "run_6_llm_7", "run_6_step_6").
call("run_6", "run_6_step_7", "adjust_position", "completed").
depends("run_6", "run_6_step_7", "run_6_llm_7").
spawned_by("run_6", "run_6_step_7", "run_6_llm_7").
param("run_6", "run_6_step_7", "pos_distance", "0.405").
param("run_6", "run_6_step_7", "delta_gripper", "-0.47").
param("run_6", "run_6_step_7", "efficiency", "0.92").
param("run_6", "run_6_step_7", "mean_speed", "0.1472").
param("run_6", "run_6_step_7", "n_frames", "4").
param("run_6", "run_6_step_7", "start_gripper", "2.1").
param("run_6", "run_6_step_7", "end_gripper", "1.7").
call("run_6", "run_6_llm_8", "llm_generate", "completed").
depends("run_6", "run_6_llm_8", "run_6_step_7").
call("run_6", "run_6_step_8", "move_y_neg_3", "completed").
depends("run_6", "run_6_step_8", "run_6_llm_8").
spawned_by("run_6", "run_6_step_8", "run_6_llm_8").
param("run_6", "run_6_step_8", "pos_distance", "0.606").
param("run_6", "run_6_step_8", "delta_gripper", "-0.14").
param("run_6", "run_6_step_8", "efficiency", "0.63").
param("run_6", "run_6_step_8", "mean_speed", "0.1921").
param("run_6", "run_6_step_8", "n_frames", "6").
param("run_6", "run_6_step_8", "start_gripper", "1.7").
param("run_6", "run_6_step_8", "end_gripper", "1.5").
call("run_6", "run_6_llm_9", "llm_generate", "completed").
depends("run_6", "run_6_llm_9", "run_6_step_8").
call("run_6", "run_6_step_9", "close_gripper", "completed").
depends("run_6", "run_6_step_9", "run_6_llm_9").
spawned_by("run_6", "run_6_step_9", "run_6_llm_9").
param("run_6", "run_6_step_9", "pos_distance", "0.473").
param("run_6", "run_6_step_9", "delta_gripper", "-1.26").
param("run_6", "run_6_step_9", "efficiency", "0.41").
param("run_6", "run_6_step_9", "mean_speed", "0.1456").
param("run_6", "run_6_step_9", "n_frames", "9").
param("run_6", "run_6_step_9", "start_gripper", "1.5").
param("run_6", "run_6_step_9", "end_gripper", "0.3").
call("run_6", "run_6_llm_10", "llm_generate", "completed").
depends("run_6", "run_6_llm_10", "run_6_step_9").
call("run_6", "run_6_notify", "notify_user", "completed").
depends("run_6", "run_6_notify", "run_6_llm_10").
spawned_by("run_6", "run_6_notify", "run_6_llm_10").

% === Episode 6 (7 discovered segments) ===
job("run_7").
job_status("run_7", "completed").
call("run_7", "run_7_step_0", "prompt_user", "completed").
call("run_7", "run_7_step_1", "llm_generate", "completed").
depends("run_7", "run_7_step_1", "run_7_step_0").
call("run_7", "run_7_llm_2", "llm_generate", "completed").
depends("run_7", "run_7_llm_2", "run_7_step_1").
call("run_7", "run_7_step_2", "move_x_pos", "completed").
depends("run_7", "run_7_step_2", "run_7_llm_2").
spawned_by("run_7", "run_7_step_2", "run_7_llm_2").
param("run_7", "run_7_step_2", "pos_distance", "0.476").
param("run_7", "run_7_step_2", "delta_gripper", "1.34").
param("run_7", "run_7_step_2", "efficiency", "1.00").
param("run_7", "run_7_step_2", "mean_speed", "0.1588").
param("run_7", "run_7_step_2", "n_frames", "4").
param("run_7", "run_7_step_2", "start_gripper", "6.0").
param("run_7", "run_7_step_2", "end_gripper", "7.4").
call("run_7", "run_7_llm_3", "llm_generate", "completed").
depends("run_7", "run_7_llm_3", "run_7_step_2").
call("run_7", "run_7_step_3", "move_x_pos_2", "completed").
depends("run_7", "run_7_step_3", "run_7_llm_3").
spawned_by("run_7", "run_7_step_3", "run_7_llm_3").
param("run_7", "run_7_step_3", "pos_distance", "2.005").
param("run_7", "run_7_step_3", "delta_gripper", "0.58").
param("run_7", "run_7_step_3", "efficiency", "0.97").
param("run_7", "run_7_step_3", "mean_speed", "0.2579").
param("run_7", "run_7_step_3", "n_frames", "9").
param("run_7", "run_7_step_3", "start_gripper", "7.4").
param("run_7", "run_7_step_3", "end_gripper", "8.0").
call("run_7", "run_7_llm_4", "llm_generate", "completed").
depends("run_7", "run_7_llm_4", "run_7_step_3").
call("run_7", "run_7_step_4", "adjust_position_2", "completed").
depends("run_7", "run_7_step_4", "run_7_llm_4").
spawned_by("run_7", "run_7_step_4", "run_7_llm_4").
param("run_7", "run_7_step_4", "pos_distance", "0.215").
param("run_7", "run_7_step_4", "delta_gripper", "0.01").
param("run_7", "run_7_step_4", "efficiency", "0.74").
param("run_7", "run_7_step_4", "mean_speed", "0.0486").
param("run_7", "run_7_step_4", "n_frames", "7").
param("run_7", "run_7_step_4", "start_gripper", "8.0").
param("run_7", "run_7_step_4", "end_gripper", "8.0").
call("run_7", "run_7_llm_5", "llm_generate", "completed").
depends("run_7", "run_7_llm_5", "run_7_step_4").
call("run_7", "run_7_step_5", "adjust_position", "completed").
depends("run_7", "run_7_step_5", "run_7_llm_5").
spawned_by("run_7", "run_7_step_5", "run_7_llm_5").
param("run_7", "run_7_step_5", "pos_distance", "0.029").
param("run_7", "run_7_step_5", "delta_gripper", "-1.28").
param("run_7", "run_7_step_5", "efficiency", "0.96").
param("run_7", "run_7_step_5", "mean_speed", "0.0099").
param("run_7", "run_7_step_5", "n_frames", "4").
param("run_7", "run_7_step_5", "start_gripper", "8.0").
param("run_7", "run_7_step_5", "end_gripper", "6.7").
call("run_7", "run_7_llm_6", "llm_generate", "completed").
depends("run_7", "run_7_llm_6", "run_7_step_5").
call("run_7", "run_7_step_6", "adjust_position", "completed").
depends("run_7", "run_7_step_6", "run_7_llm_6").
spawned_by("run_7", "run_7_step_6", "run_7_llm_6").
param("run_7", "run_7_step_6", "pos_distance", "0.029").
param("run_7", "run_7_step_6", "delta_gripper", "-3.40").
param("run_7", "run_7_step_6", "efficiency", "0.99").
param("run_7", "run_7_step_6", "mean_speed", "0.0099").
param("run_7", "run_7_step_6", "n_frames", "4").
param("run_7", "run_7_step_6", "start_gripper", "6.7").
param("run_7", "run_7_step_6", "end_gripper", "3.3").
call("run_7", "run_7_llm_7", "llm_generate", "completed").
depends("run_7", "run_7_llm_7", "run_7_step_6").
call("run_7", "run_7_step_7", "adjust_position", "completed").
depends("run_7", "run_7_step_7", "run_7_llm_7").
spawned_by("run_7", "run_7_step_7", "run_7_llm_7").
param("run_7", "run_7_step_7", "pos_distance", "0.118").
param("run_7", "run_7_step_7", "delta_gripper", "-3.28").
param("run_7", "run_7_step_7", "efficiency", "0.97").
param("run_7", "run_7_step_7", "mean_speed", "0.0405").
param("run_7", "run_7_step_7", "n_frames", "4").
param("run_7", "run_7_step_7", "start_gripper", "3.3").
param("run_7", "run_7_step_7", "end_gripper", "0.0").
call("run_7", "run_7_llm_8", "llm_generate", "completed").
depends("run_7", "run_7_llm_8", "run_7_step_7").
call("run_7", "run_7_step_8", "adjust_position_7", "completed").
depends("run_7", "run_7_step_8", "run_7_llm_8").
spawned_by("run_7", "run_7_step_8", "run_7_llm_8").
param("run_7", "run_7_step_8", "pos_distance", "1.747").
param("run_7", "run_7_step_8", "delta_gripper", "0.01").
param("run_7", "run_7_step_8", "efficiency", "0.94").
param("run_7", "run_7_step_8", "mean_speed", "0.0809").
param("run_7", "run_7_step_8", "n_frames", "24").
param("run_7", "run_7_step_8", "start_gripper", "0.0").
param("run_7", "run_7_step_8", "end_gripper", "0.0").
call("run_7", "run_7_llm_9", "llm_generate", "completed").
depends("run_7", "run_7_llm_9", "run_7_step_8").
call("run_7", "run_7_notify", "notify_user", "completed").
depends("run_7", "run_7_notify", "run_7_llm_9").
spawned_by("run_7", "run_7_notify", "run_7_llm_9").

% === Episode 7 (13 discovered segments) ===
job("run_8").
job_status("run_8", "completed").
call("run_8", "run_8_step_0", "prompt_user", "completed").
call("run_8", "run_8_step_1", "llm_generate", "completed").
depends("run_8", "run_8_step_1", "run_8_step_0").
call("run_8", "run_8_llm_2", "llm_generate", "completed").
depends("run_8", "run_8_llm_2", "run_8_step_1").
call("run_8", "run_8_step_2", "move_x_pos", "completed").
depends("run_8", "run_8_step_2", "run_8_llm_2").
spawned_by("run_8", "run_8_step_2", "run_8_llm_2").
param("run_8", "run_8_step_2", "pos_distance", "0.625").
param("run_8", "run_8_step_2", "delta_gripper", "1.50").
param("run_8", "run_8_step_2", "efficiency", "1.00").
param("run_8", "run_8_step_2", "mean_speed", "0.2090").
param("run_8", "run_8_step_2", "n_frames", "4").
param("run_8", "run_8_step_2", "start_gripper", "6.0").
param("run_8", "run_8_step_2", "end_gripper", "7.5").
call("run_8", "run_8_llm_3", "llm_generate", "completed").
depends("run_8", "run_8_llm_3", "run_8_step_2").
call("run_8", "run_8_step_3", "move_y_neg_2", "completed").
depends("run_8", "run_8_step_3", "run_8_llm_3").
spawned_by("run_8", "run_8_step_3", "run_8_llm_3").
param("run_8", "run_8_step_3", "pos_distance", "2.135").
param("run_8", "run_8_step_3", "delta_gripper", "0.43").
param("run_8", "run_8_step_3", "efficiency", "0.99").
param("run_8", "run_8_step_3", "mean_speed", "0.3586").
param("run_8", "run_8_step_3", "n_frames", "7").
param("run_8", "run_8_step_3", "start_gripper", "7.5").
param("run_8", "run_8_step_3", "end_gripper", "8.0").
call("run_8", "run_8_llm_4", "llm_generate", "completed").
depends("run_8", "run_8_llm_4", "run_8_step_3").
call("run_8", "run_8_step_4", "adjust_position", "completed").
depends("run_8", "run_8_step_4", "run_8_llm_4").
spawned_by("run_8", "run_8_step_4", "run_8_llm_4").
param("run_8", "run_8_step_4", "pos_distance", "0.395").
param("run_8", "run_8_step_4", "delta_gripper", "-2.39").
param("run_8", "run_8_step_4", "efficiency", "0.81").
param("run_8", "run_8_step_4", "mean_speed", "0.1627").
param("run_8", "run_8_step_4", "n_frames", "4").
param("run_8", "run_8_step_4", "start_gripper", "8.0").
param("run_8", "run_8_step_4", "end_gripper", "5.6").
call("run_8", "run_8_llm_5", "llm_generate", "completed").
depends("run_8", "run_8_llm_5", "run_8_step_4").
call("run_8", "run_8_step_5", "adjust_position", "completed").
depends("run_8", "run_8_step_5", "run_8_llm_5").
spawned_by("run_8", "run_8_step_5", "run_8_llm_5").
param("run_8", "run_8_step_5", "pos_distance", "0.359").
param("run_8", "run_8_step_5", "delta_gripper", "-2.32").
param("run_8", "run_8_step_5", "efficiency", "0.72").
param("run_8", "run_8_step_5", "mean_speed", "0.1668").
param("run_8", "run_8_step_5", "n_frames", "4").
param("run_8", "run_8_step_5", "start_gripper", "5.6").
param("run_8", "run_8_step_5", "end_gripper", "3.3").
call("run_8", "run_8_llm_6", "llm_generate", "completed").
depends("run_8", "run_8_llm_6", "run_8_step_5").
call("run_8", "run_8_step_6", "move_x_pos_3", "completed").
depends("run_8", "run_8_step_6", "run_8_llm_6").
spawned_by("run_8", "run_8_step_6", "run_8_llm_6").
param("run_8", "run_8_step_6", "pos_distance", "1.964").
param("run_8", "run_8_step_6", "delta_gripper", "-0.70").
param("run_8", "run_8_step_6", "efficiency", "0.74").
param("run_8", "run_8_step_6", "mean_speed", "0.2951").
param("run_8", "run_8_step_6", "n_frames", "10").
param("run_8", "run_8_step_6", "start_gripper", "3.3").
param("run_8", "run_8_step_6", "end_gripper", "2.6").
call("run_8", "run_8_llm_7", "llm_generate", "completed").
depends("run_8", "run_8_llm_7", "run_8_step_6").
call("run_8", "run_8_step_7", "adjust_position", "completed").
depends("run_8", "run_8_step_7", "run_8_llm_7").
spawned_by("run_8", "run_8_step_7", "run_8_llm_7").
param("run_8", "run_8_step_7", "pos_distance", "0.432").
param("run_8", "run_8_step_7", "delta_gripper", "-1.20").
param("run_8", "run_8_step_7", "efficiency", "0.72").
param("run_8", "run_8_step_7", "mean_speed", "0.1502").
param("run_8", "run_8_step_7", "n_frames", "5").
param("run_8", "run_8_step_7", "start_gripper", "2.6").
param("run_8", "run_8_step_7", "end_gripper", "1.4").
call("run_8", "run_8_llm_8", "llm_generate", "completed").
depends("run_8", "run_8_llm_8", "run_8_step_7").
call("run_8", "run_8_step_8", "adjust_position_2", "completed").
depends("run_8", "run_8_step_8", "run_8_llm_8").
spawned_by("run_8", "run_8_step_8", "run_8_llm_8").
param("run_8", "run_8_step_8", "pos_distance", "0.072").
param("run_8", "run_8_step_8", "delta_gripper", "0.12").
param("run_8", "run_8_step_8", "efficiency", "0.29").
param("run_8", "run_8_step_8", "mean_speed", "0.0812").
param("run_8", "run_8_step_8", "n_frames", "4").
param("run_8", "run_8_step_8", "start_gripper", "1.4").
param("run_8", "run_8_step_8", "end_gripper", "1.5").
call("run_8", "run_8_llm_9", "llm_generate", "completed").
depends("run_8", "run_8_llm_9", "run_8_step_8").
call("run_8", "run_8_step_9", "adjust_position_2", "completed").
depends("run_8", "run_8_step_9", "run_8_llm_9").
spawned_by("run_8", "run_8_step_9", "run_8_llm_9").
param("run_8", "run_8_step_9", "pos_distance", "0.048").
param("run_8", "run_8_step_9", "delta_gripper", "0.20").
param("run_8", "run_8_step_9", "efficiency", "0.63").
param("run_8", "run_8_step_9", "mean_speed", "0.0254").
param("run_8", "run_8_step_9", "n_frames", "4").
param("run_8", "run_8_step_9", "start_gripper", "1.5").
param("run_8", "run_8_step_9", "end_gripper", "1.7").
call("run_8", "run_8_llm_10", "llm_generate", "completed").
depends("run_8", "run_8_llm_10", "run_8_step_9").
call("run_8", "run_8_step_10", "adjust_position", "completed").
depends("run_8", "run_8_step_10", "run_8_llm_10").
spawned_by("run_8", "run_8_step_10", "run_8_llm_10").
param("run_8", "run_8_step_10", "pos_distance", "0.077").
param("run_8", "run_8_step_10", "delta_gripper", "0.09").
param("run_8", "run_8_step_10", "efficiency", "0.98").
param("run_8", "run_8_step_10", "mean_speed", "0.0264").
param("run_8", "run_8_step_10", "n_frames", "4").
param("run_8", "run_8_step_10", "start_gripper", "1.7").
param("run_8", "run_8_step_10", "end_gripper", "1.8").
call("run_8", "run_8_llm_11", "llm_generate", "completed").
depends("run_8", "run_8_llm_11", "run_8_step_10").
call("run_8", "run_8_step_11", "adjust_position_5", "completed").
depends("run_8", "run_8_step_11", "run_8_llm_11").
spawned_by("run_8", "run_8_step_11", "run_8_llm_11").
param("run_8", "run_8_step_11", "pos_distance", "0.097").
param("run_8", "run_8_step_11", "delta_gripper", "0.33").
param("run_8", "run_8_step_11", "efficiency", "0.99").
param("run_8", "run_8_step_11", "mean_speed", "0.0327").
param("run_8", "run_8_step_11", "n_frames", "4").
param("run_8", "run_8_step_11", "start_gripper", "1.8").
param("run_8", "run_8_step_11", "end_gripper", "2.1").
call("run_8", "run_8_llm_12", "llm_generate", "completed").
depends("run_8", "run_8_llm_12", "run_8_step_11").
call("run_8", "run_8_step_12", "adjust_position_2", "completed").
depends("run_8", "run_8_step_12", "run_8_llm_12").
spawned_by("run_8", "run_8_step_12", "run_8_llm_12").
param("run_8", "run_8_step_12", "pos_distance", "0.040").
param("run_8", "run_8_step_12", "delta_gripper", "0.15").
param("run_8", "run_8_step_12", "efficiency", "0.84").
param("run_8", "run_8_step_12", "mean_speed", "0.0158").
param("run_8", "run_8_step_12", "n_frames", "4").
param("run_8", "run_8_step_12", "start_gripper", "2.1").
param("run_8", "run_8_step_12", "end_gripper", "2.2").
call("run_8", "run_8_llm_13", "llm_generate", "completed").
depends("run_8", "run_8_llm_13", "run_8_step_12").
call("run_8", "run_8_step_13", "adjust_position_2", "completed").
depends("run_8", "run_8_step_13", "run_8_llm_13").
spawned_by("run_8", "run_8_step_13", "run_8_llm_13").
param("run_8", "run_8_step_13", "pos_distance", "0.026").
param("run_8", "run_8_step_13", "delta_gripper", "0.22").
param("run_8", "run_8_step_13", "efficiency", "0.85").
param("run_8", "run_8_step_13", "mean_speed", "0.0100").
param("run_8", "run_8_step_13", "n_frames", "4").
param("run_8", "run_8_step_13", "start_gripper", "2.2").
param("run_8", "run_8_step_13", "end_gripper", "2.5").
call("run_8", "run_8_llm_14", "llm_generate", "completed").
depends("run_8", "run_8_llm_14", "run_8_step_13").
call("run_8", "run_8_step_14", "adjust_position_5", "completed").
depends("run_8", "run_8_step_14", "run_8_llm_14").
spawned_by("run_8", "run_8_step_14", "run_8_llm_14").
param("run_8", "run_8_step_14", "pos_distance", "0.007").
param("run_8", "run_8_step_14", "delta_gripper", "0.20").
param("run_8", "run_8_step_14", "efficiency", "0.97").
param("run_8", "run_8_step_14", "mean_speed", "0.0025").
param("run_8", "run_8_step_14", "n_frames", "4").
param("run_8", "run_8_step_14", "start_gripper", "2.5").
param("run_8", "run_8_step_14", "end_gripper", "2.7").
call("run_8", "run_8_llm_15", "llm_generate", "completed").
depends("run_8", "run_8_llm_15", "run_8_step_14").
call("run_8", "run_8_notify", "notify_user", "completed").
depends("run_8", "run_8_notify", "run_8_llm_15").
spawned_by("run_8", "run_8_notify", "run_8_llm_15").

% === Episode 8 (12 discovered segments) ===
job("run_9").
job_status("run_9", "completed").
call("run_9", "run_9_step_0", "prompt_user", "completed").
call("run_9", "run_9_step_1", "llm_generate", "completed").
depends("run_9", "run_9_step_1", "run_9_step_0").
call("run_9", "run_9_llm_2", "llm_generate", "completed").
depends("run_9", "run_9_llm_2", "run_9_step_1").
call("run_9", "run_9_step_2", "move_x_pos", "completed").
depends("run_9", "run_9_step_2", "run_9_llm_2").
spawned_by("run_9", "run_9_step_2", "run_9_llm_2").
param("run_9", "run_9_step_2", "pos_distance", "0.698").
param("run_9", "run_9_step_2", "delta_gripper", "1.81").
param("run_9", "run_9_step_2", "efficiency", "0.97").
param("run_9", "run_9_step_2", "mean_speed", "0.2393").
param("run_9", "run_9_step_2", "n_frames", "4").
param("run_9", "run_9_step_2", "start_gripper", "6.0").
param("run_9", "run_9_step_2", "end_gripper", "7.8").
call("run_9", "run_9_llm_3", "llm_generate", "completed").
depends("run_9", "run_9_llm_3", "run_9_step_2").
call("run_9", "run_9_step_3", "move_x_pos_4", "completed").
depends("run_9", "run_9_step_3", "run_9_llm_3").
spawned_by("run_9", "run_9_step_3", "run_9_llm_3").
param("run_9", "run_9_step_3", "pos_distance", "1.432").
param("run_9", "run_9_step_3", "delta_gripper", "0.11").
param("run_9", "run_9_step_3", "efficiency", "0.97").
param("run_9", "run_9_step_3", "mean_speed", "0.2939").
param("run_9", "run_9_step_3", "n_frames", "6").
param("run_9", "run_9_step_3", "start_gripper", "7.8").
param("run_9", "run_9_step_3", "end_gripper", "8.0").
call("run_9", "run_9_llm_4", "llm_generate", "completed").
depends("run_9", "run_9_llm_4", "run_9_step_3").
call("run_9", "run_9_step_4", "move_x_pos_4", "completed").
depends("run_9", "run_9_step_4", "run_9_llm_4").
spawned_by("run_9", "run_9_step_4", "run_9_llm_4").
param("run_9", "run_9_step_4", "pos_distance", "0.853").
param("run_9", "run_9_step_4", "delta_gripper", "-1.37").
param("run_9", "run_9_step_4", "efficiency", "0.98").
param("run_9", "run_9_step_4", "mean_speed", "0.2897").
param("run_9", "run_9_step_4", "n_frames", "4").
param("run_9", "run_9_step_4", "start_gripper", "8.0").
param("run_9", "run_9_step_4", "end_gripper", "6.6").
call("run_9", "run_9_llm_5", "llm_generate", "completed").
depends("run_9", "run_9_llm_5", "run_9_step_4").
call("run_9", "run_9_step_5", "adjust_position_2", "completed").
depends("run_9", "run_9_step_5", "run_9_llm_5").
spawned_by("run_9", "run_9_step_5", "run_9_llm_5").
param("run_9", "run_9_step_5", "pos_distance", "0.298").
param("run_9", "run_9_step_5", "delta_gripper", "1.06").
param("run_9", "run_9_step_5", "efficiency", "0.99").
param("run_9", "run_9_step_5", "mean_speed", "0.1006").
param("run_9", "run_9_step_5", "n_frames", "4").
param("run_9", "run_9_step_5", "start_gripper", "6.6").
param("run_9", "run_9_step_5", "end_gripper", "7.7").
call("run_9", "run_9_llm_6", "llm_generate", "completed").
depends("run_9", "run_9_llm_6", "run_9_step_5").
call("run_9", "run_9_step_6", "move_y_neg", "completed").
depends("run_9", "run_9_step_6", "run_9_llm_6").
spawned_by("run_9", "run_9_step_6", "run_9_llm_6").
param("run_9", "run_9_step_6", "pos_distance", "0.805").
param("run_9", "run_9_step_6", "delta_gripper", "0.31").
param("run_9", "run_9_step_6", "efficiency", "0.83").
param("run_9", "run_9_step_6", "mean_speed", "0.2417").
param("run_9", "run_9_step_6", "n_frames", "5").
param("run_9", "run_9_step_6", "start_gripper", "7.7").
param("run_9", "run_9_step_6", "end_gripper", "8.0").
call("run_9", "run_9_llm_7", "llm_generate", "completed").
depends("run_9", "run_9_llm_7", "run_9_step_6").
call("run_9", "run_9_step_7", "adjust_position_2", "completed").
depends("run_9", "run_9_step_7", "run_9_llm_7").
spawned_by("run_9", "run_9_step_7", "run_9_llm_7").
param("run_9", "run_9_step_7", "pos_distance", "0.228").
param("run_9", "run_9_step_7", "delta_gripper", "0.00").
param("run_9", "run_9_step_7", "efficiency", "0.71").
param("run_9", "run_9_step_7", "mean_speed", "0.1076").
param("run_9", "run_9_step_7", "n_frames", "4").
param("run_9", "run_9_step_7", "start_gripper", "8.0").
param("run_9", "run_9_step_7", "end_gripper", "8.0").
call("run_9", "run_9_llm_8", "llm_generate", "completed").
depends("run_9", "run_9_llm_8", "run_9_step_7").
call("run_9", "run_9_step_8", "adjust_position_2", "completed").
depends("run_9", "run_9_step_8", "run_9_llm_8").
spawned_by("run_9", "run_9_step_8", "run_9_llm_8").
param("run_9", "run_9_step_8", "pos_distance", "0.246").
param("run_9", "run_9_step_8", "delta_gripper", "-0.03").
param("run_9", "run_9_step_8", "efficiency", "0.59").
param("run_9", "run_9_step_8", "mean_speed", "0.1393").
param("run_9", "run_9_step_8", "n_frames", "4").
param("run_9", "run_9_step_8", "start_gripper", "8.0").
param("run_9", "run_9_step_8", "end_gripper", "7.9").
call("run_9", "run_9_llm_9", "llm_generate", "completed").
depends("run_9", "run_9_llm_9", "run_9_step_8").
call("run_9", "run_9_step_9", "adjust_position", "completed").
depends("run_9", "run_9_step_9", "run_9_llm_9").
spawned_by("run_9", "run_9_step_9", "run_9_llm_9").
param("run_9", "run_9_step_9", "pos_distance", "0.291").
param("run_9", "run_9_step_9", "delta_gripper", "-0.27").
param("run_9", "run_9_step_9", "efficiency", "0.80").
param("run_9", "run_9_step_9", "mean_speed", "0.1212").
param("run_9", "run_9_step_9", "n_frames", "4").
param("run_9", "run_9_step_9", "start_gripper", "7.9").
param("run_9", "run_9_step_9", "end_gripper", "7.7").
call("run_9", "run_9_llm_10", "llm_generate", "completed").
depends("run_9", "run_9_llm_10", "run_9_step_9").
call("run_9", "run_9_step_10", "adjust_position_8", "completed").
depends("run_9", "run_9_step_10", "run_9_llm_10").
spawned_by("run_9", "run_9_step_10", "run_9_llm_10").
param("run_9", "run_9_step_10", "pos_distance", "0.020").
param("run_9", "run_9_step_10", "delta_gripper", "0.30").
param("run_9", "run_9_step_10", "efficiency", "0.03").
param("run_9", "run_9_step_10", "mean_speed", "0.1284").
param("run_9", "run_9_step_10", "n_frames", "6").
param("run_9", "run_9_step_10", "start_gripper", "7.7").
param("run_9", "run_9_step_10", "end_gripper", "8.0").
call("run_9", "run_9_llm_11", "llm_generate", "completed").
depends("run_9", "run_9_llm_11", "run_9_step_10").
call("run_9", "run_9_step_11", "adjust_position_8", "completed").
depends("run_9", "run_9_step_11", "run_9_llm_11").
spawned_by("run_9", "run_9_step_11", "run_9_llm_11").
param("run_9", "run_9_step_11", "pos_distance", "0.180").
param("run_9", "run_9_step_11", "delta_gripper", "0.00").
param("run_9", "run_9_step_11", "efficiency", "0.19").
param("run_9", "run_9_step_11", "mean_speed", "0.1603").
param("run_9", "run_9_step_11", "n_frames", "7").
param("run_9", "run_9_step_11", "start_gripper", "8.0").
param("run_9", "run_9_step_11", "end_gripper", "8.0").
call("run_9", "run_9_llm_12", "llm_generate", "completed").
depends("run_9", "run_9_llm_12", "run_9_step_11").
call("run_9", "run_9_step_12", "adjust_position", "completed").
depends("run_9", "run_9_step_12", "run_9_llm_12").
spawned_by("run_9", "run_9_step_12", "run_9_llm_12").
param("run_9", "run_9_step_12", "pos_distance", "0.183").
param("run_9", "run_9_step_12", "delta_gripper", "-0.82").
param("run_9", "run_9_step_12", "efficiency", "0.79").
param("run_9", "run_9_step_12", "mean_speed", "0.0774").
param("run_9", "run_9_step_12", "n_frames", "4").
param("run_9", "run_9_step_12", "start_gripper", "8.0").
param("run_9", "run_9_step_12", "end_gripper", "7.1").
call("run_9", "run_9_llm_13", "llm_generate", "completed").
depends("run_9", "run_9_llm_13", "run_9_step_12").
call("run_9", "run_9_step_13", "adjust_position_2", "completed").
depends("run_9", "run_9_step_13", "run_9_llm_13").
spawned_by("run_9", "run_9_step_13", "run_9_llm_13").
param("run_9", "run_9_step_13", "pos_distance", "0.146").
param("run_9", "run_9_step_13", "delta_gripper", "0.81").
param("run_9", "run_9_step_13", "efficiency", "0.27").
param("run_9", "run_9_step_13", "mean_speed", "0.0681").
param("run_9", "run_9_step_13", "n_frames", "9").
param("run_9", "run_9_step_13", "start_gripper", "7.1").
param("run_9", "run_9_step_13", "end_gripper", "7.9").
call("run_9", "run_9_llm_14", "llm_generate", "completed").
depends("run_9", "run_9_llm_14", "run_9_step_13").
call("run_9", "run_9_notify", "notify_user", "completed").
depends("run_9", "run_9_notify", "run_9_llm_14").
spawned_by("run_9", "run_9_notify", "run_9_llm_14").

% === Episode 9 (8 discovered segments) ===
job("run_10").
job_status("run_10", "completed").
call("run_10", "run_10_step_0", "prompt_user", "completed").
call("run_10", "run_10_step_1", "llm_generate", "completed").
depends("run_10", "run_10_step_1", "run_10_step_0").
call("run_10", "run_10_llm_2", "llm_generate", "completed").
depends("run_10", "run_10_llm_2", "run_10_step_1").
call("run_10", "run_10_step_2", "move_y_neg", "completed").
depends("run_10", "run_10_step_2", "run_10_llm_2").
spawned_by("run_10", "run_10_step_2", "run_10_llm_2").
param("run_10", "run_10_step_2", "pos_distance", "0.732").
param("run_10", "run_10_step_2", "delta_gripper", "0.85").
param("run_10", "run_10_step_2", "efficiency", "0.98").
param("run_10", "run_10_step_2", "mean_speed", "0.2491").
param("run_10", "run_10_step_2", "n_frames", "4").
param("run_10", "run_10_step_2", "start_gripper", "6.0").
param("run_10", "run_10_step_2", "end_gripper", "6.9").
call("run_10", "run_10_llm_3", "llm_generate", "completed").
depends("run_10", "run_10_llm_3", "run_10_step_2").
call("run_10", "run_10_step_3", "move_y_neg", "completed").
depends("run_10", "run_10_step_3", "run_10_llm_3").
spawned_by("run_10", "run_10_step_3", "run_10_llm_3").
param("run_10", "run_10_step_3", "pos_distance", "1.010").
param("run_10", "run_10_step_3", "delta_gripper", "-0.02").
param("run_10", "run_10_step_3", "efficiency", "0.99").
param("run_10", "run_10_step_3", "mean_speed", "0.3402").
param("run_10", "run_10_step_3", "n_frames", "4").
param("run_10", "run_10_step_3", "start_gripper", "6.9").
param("run_10", "run_10_step_3", "end_gripper", "6.9").
call("run_10", "run_10_llm_4", "llm_generate", "completed").
depends("run_10", "run_10_llm_4", "run_10_step_3").
call("run_10", "run_10_step_4", "adjust_position_2", "completed").
depends("run_10", "run_10_step_4", "run_10_llm_4").
spawned_by("run_10", "run_10_step_4", "run_10_llm_4").
param("run_10", "run_10_step_4", "pos_distance", "0.519").
param("run_10", "run_10_step_4", "delta_gripper", "1.09").
param("run_10", "run_10_step_4", "efficiency", "0.92").
param("run_10", "run_10_step_4", "mean_speed", "0.1887").
param("run_10", "run_10_step_4", "n_frames", "4").
param("run_10", "run_10_step_4", "start_gripper", "6.9").
param("run_10", "run_10_step_4", "end_gripper", "8.0").
call("run_10", "run_10_llm_5", "llm_generate", "completed").
depends("run_10", "run_10_llm_5", "run_10_step_4").
call("run_10", "run_10_step_5", "move_y_neg_4", "completed").
depends("run_10", "run_10_step_5", "run_10_llm_5").
spawned_by("run_10", "run_10_step_5", "run_10_llm_5").
param("run_10", "run_10_step_5", "pos_distance", "0.315").
param("run_10", "run_10_step_5", "delta_gripper", "-1.33").
param("run_10", "run_10_step_5", "efficiency", "0.36").
param("run_10", "run_10_step_5", "mean_speed", "0.2900").
param("run_10", "run_10_step_5", "n_frames", "4").
param("run_10", "run_10_step_5", "start_gripper", "8.0").
param("run_10", "run_10_step_5", "end_gripper", "6.6").
call("run_10", "run_10_llm_6", "llm_generate", "completed").
depends("run_10", "run_10_llm_6", "run_10_step_5").
call("run_10", "run_10_step_6", "adjust_position", "completed").
depends("run_10", "run_10_step_6", "run_10_llm_6").
spawned_by("run_10", "run_10_step_6", "run_10_llm_6").
param("run_10", "run_10_step_6", "pos_distance", "0.540").
param("run_10", "run_10_step_6", "delta_gripper", "0.04").
param("run_10", "run_10_step_6", "efficiency", "0.92").
param("run_10", "run_10_step_6", "mean_speed", "0.1961").
param("run_10", "run_10_step_6", "n_frames", "4").
param("run_10", "run_10_step_6", "start_gripper", "6.6").
param("run_10", "run_10_step_6", "end_gripper", "6.7").
call("run_10", "run_10_llm_7", "llm_generate", "completed").
depends("run_10", "run_10_llm_7", "run_10_step_6").
call("run_10", "run_10_step_7", "move_x_pos", "completed").
depends("run_10", "run_10_step_7", "run_10_llm_7").
spawned_by("run_10", "run_10_step_7", "run_10_llm_7").
param("run_10", "run_10_step_7", "pos_distance", "0.445").
param("run_10", "run_10_step_7", "delta_gripper", "0.25").
param("run_10", "run_10_step_7", "efficiency", "0.81").
param("run_10", "run_10_step_7", "mean_speed", "0.1828").
param("run_10", "run_10_step_7", "n_frames", "4").
param("run_10", "run_10_step_7", "start_gripper", "6.7").
param("run_10", "run_10_step_7", "end_gripper", "6.9").
call("run_10", "run_10_llm_8", "llm_generate", "completed").
depends("run_10", "run_10_llm_8", "run_10_step_7").
call("run_10", "run_10_step_8", "adjust_position", "completed").
depends("run_10", "run_10_step_8", "run_10_llm_8").
spawned_by("run_10", "run_10_step_8", "run_10_llm_8").
param("run_10", "run_10_step_8", "pos_distance", "0.352").
param("run_10", "run_10_step_8", "delta_gripper", "-3.53").
param("run_10", "run_10_step_8", "efficiency", "0.90").
param("run_10", "run_10_step_8", "mean_speed", "0.1305").
param("run_10", "run_10_step_8", "n_frames", "4").
param("run_10", "run_10_step_8", "start_gripper", "6.9").
param("run_10", "run_10_step_8", "end_gripper", "3.4").
call("run_10", "run_10_llm_9", "llm_generate", "completed").
depends("run_10", "run_10_llm_9", "run_10_step_8").
call("run_10", "run_10_step_9", "move_z_pos", "completed").
depends("run_10", "run_10_step_9", "run_10_llm_9").
spawned_by("run_10", "run_10_step_9", "run_10_llm_9").
param("run_10", "run_10_step_9", "pos_distance", "1.413").
param("run_10", "run_10_step_9", "delta_gripper", "-0.06").
param("run_10", "run_10_step_9", "efficiency", "0.31").
param("run_10", "run_10_step_9", "mean_speed", "0.1608").
param("run_10", "run_10_step_9", "n_frames", "29").
param("run_10", "run_10_step_9", "start_gripper", "3.4").
param("run_10", "run_10_step_9", "end_gripper", "3.3").
call("run_10", "run_10_llm_10", "llm_generate", "completed").
depends("run_10", "run_10_llm_10", "run_10_step_9").
call("run_10", "run_10_notify", "notify_user", "completed").
depends("run_10", "run_10_notify", "run_10_llm_10").
spawned_by("run_10", "run_10_notify", "run_10_llm_10").

% === Episode 10 (5 discovered segments) ===
job("run_11").
job_status("run_11", "completed").
call("run_11", "run_11_step_0", "prompt_user", "completed").
call("run_11", "run_11_step_1", "llm_generate", "completed").
depends("run_11", "run_11_step_1", "run_11_step_0").
call("run_11", "run_11_llm_2", "llm_generate", "completed").
depends("run_11", "run_11_llm_2", "run_11_step_1").
call("run_11", "run_11_step_2", "move_y_neg", "completed").
depends("run_11", "run_11_step_2", "run_11_llm_2").
spawned_by("run_11", "run_11_step_2", "run_11_llm_2").
param("run_11", "run_11_step_2", "pos_distance", "0.427").
param("run_11", "run_11_step_2", "delta_gripper", "1.47").
param("run_11", "run_11_step_2", "efficiency", "0.99").
param("run_11", "run_11_step_2", "mean_speed", "0.1433").
param("run_11", "run_11_step_2", "n_frames", "4").
param("run_11", "run_11_step_2", "start_gripper", "6.0").
param("run_11", "run_11_step_2", "end_gripper", "7.5").
call("run_11", "run_11_llm_3", "llm_generate", "completed").
depends("run_11", "run_11_llm_3", "run_11_step_2").
call("run_11", "run_11_step_3", "move_x_pos", "completed").
depends("run_11", "run_11_step_3", "run_11_llm_3").
spawned_by("run_11", "run_11_step_3", "run_11_llm_3").
param("run_11", "run_11_step_3", "pos_distance", "0.690").
param("run_11", "run_11_step_3", "delta_gripper", "0.46").
param("run_11", "run_11_step_3", "efficiency", "1.00").
param("run_11", "run_11_step_3", "mean_speed", "0.2305").
param("run_11", "run_11_step_3", "n_frames", "4").
param("run_11", "run_11_step_3", "start_gripper", "7.5").
param("run_11", "run_11_step_3", "end_gripper", "8.0").
call("run_11", "run_11_llm_4", "llm_generate", "completed").
depends("run_11", "run_11_llm_4", "run_11_step_3").
call("run_11", "run_11_step_4", "adjust_position", "completed").
depends("run_11", "run_11_step_4", "run_11_llm_4").
spawned_by("run_11", "run_11_step_4", "run_11_llm_4").
param("run_11", "run_11_step_4", "pos_distance", "0.129").
param("run_11", "run_11_step_4", "delta_gripper", "-2.27").
param("run_11", "run_11_step_4", "efficiency", "0.74").
param("run_11", "run_11_step_4", "mean_speed", "0.0580").
param("run_11", "run_11_step_4", "n_frames", "4").
param("run_11", "run_11_step_4", "start_gripper", "8.0").
param("run_11", "run_11_step_4", "end_gripper", "5.7").
call("run_11", "run_11_llm_5", "llm_generate", "completed").
depends("run_11", "run_11_llm_5", "run_11_step_4").
call("run_11", "run_11_step_5", "adjust_position", "completed").
depends("run_11", "run_11_step_5", "run_11_llm_5").
spawned_by("run_11", "run_11_step_5", "run_11_llm_5").
param("run_11", "run_11_step_5", "pos_distance", "0.077").
param("run_11", "run_11_step_5", "delta_gripper", "-2.31").
param("run_11", "run_11_step_5", "efficiency", "0.97").
param("run_11", "run_11_step_5", "mean_speed", "0.0265").
param("run_11", "run_11_step_5", "n_frames", "4").
param("run_11", "run_11_step_5", "start_gripper", "5.7").
param("run_11", "run_11_step_5", "end_gripper", "3.4").
call("run_11", "run_11_llm_6", "llm_generate", "completed").
depends("run_11", "run_11_llm_6", "run_11_step_5").
call("run_11", "run_11_step_6", "adjust_position_4", "completed").
depends("run_11", "run_11_step_6", "run_11_llm_6").
spawned_by("run_11", "run_11_step_6", "run_11_llm_6").
param("run_11", "run_11_step_6", "pos_distance", "1.426").
param("run_11", "run_11_step_6", "delta_gripper", "0.00").
param("run_11", "run_11_step_6", "efficiency", "0.84").
param("run_11", "run_11_step_6", "mean_speed", "0.0461").
param("run_11", "run_11_step_6", "n_frames", "38").
param("run_11", "run_11_step_6", "start_gripper", "3.4").
param("run_11", "run_11_step_6", "end_gripper", "3.4").
call("run_11", "run_11_llm_7", "llm_generate", "completed").
depends("run_11", "run_11_llm_7", "run_11_step_6").
call("run_11", "run_11_notify", "notify_user", "completed").
depends("run_11", "run_11_notify", "run_11_llm_7").
spawned_by("run_11", "run_11_notify", "run_11_llm_7").

% === Episode 11 (7 discovered segments) ===
job("run_12").
job_status("run_12", "completed").
call("run_12", "run_12_step_0", "prompt_user", "completed").
call("run_12", "run_12_step_1", "llm_generate", "completed").
depends("run_12", "run_12_step_1", "run_12_step_0").
call("run_12", "run_12_llm_2", "llm_generate", "completed").
depends("run_12", "run_12_llm_2", "run_12_step_1").
call("run_12", "run_12_step_2", "move_x_pos", "completed").
depends("run_12", "run_12_step_2", "run_12_llm_2").
spawned_by("run_12", "run_12_step_2", "run_12_llm_2").
param("run_12", "run_12_step_2", "pos_distance", "0.708").
param("run_12", "run_12_step_2", "delta_gripper", "1.67").
param("run_12", "run_12_step_2", "efficiency", "0.98").
param("run_12", "run_12_step_2", "mean_speed", "0.2413").
param("run_12", "run_12_step_2", "n_frames", "4").
param("run_12", "run_12_step_2", "start_gripper", "6.0").
param("run_12", "run_12_step_2", "end_gripper", "7.7").
call("run_12", "run_12_llm_3", "llm_generate", "completed").
depends("run_12", "run_12_llm_3", "run_12_step_2").
call("run_12", "run_12_step_3", "move_y_neg_2", "completed").
depends("run_12", "run_12_step_3", "run_12_llm_3").
spawned_by("run_12", "run_12_step_3", "run_12_llm_3").
param("run_12", "run_12_step_3", "pos_distance", "2.524").
param("run_12", "run_12_step_3", "delta_gripper", "0.25").
param("run_12", "run_12_step_3", "efficiency", "0.95").
param("run_12", "run_12_step_3", "mean_speed", "0.3312").
param("run_12", "run_12_step_3", "n_frames", "9").
param("run_12", "run_12_step_3", "start_gripper", "7.7").
param("run_12", "run_12_step_3", "end_gripper", "8.0").
call("run_12", "run_12_llm_4", "llm_generate", "completed").
depends("run_12", "run_12_llm_4", "run_12_step_3").
call("run_12", "run_12_step_4", "move_y_neg_3", "completed").
depends("run_12", "run_12_step_4", "run_12_llm_4").
spawned_by("run_12", "run_12_step_4", "run_12_llm_4").
param("run_12", "run_12_step_4", "pos_distance", "0.750").
param("run_12", "run_12_step_4", "delta_gripper", "-0.09").
param("run_12", "run_12_step_4", "efficiency", "0.98").
param("run_12", "run_12_step_4", "mean_speed", "0.2543").
param("run_12", "run_12_step_4", "n_frames", "4").
param("run_12", "run_12_step_4", "start_gripper", "8.0").
param("run_12", "run_12_step_4", "end_gripper", "7.9").
call("run_12", "run_12_llm_5", "llm_generate", "completed").
depends("run_12", "run_12_llm_5", "run_12_step_4").
call("run_12", "run_12_step_5", "close_gripper", "completed").
depends("run_12", "run_12_step_5", "run_12_llm_5").
spawned_by("run_12", "run_12_step_5", "run_12_llm_5").
param("run_12", "run_12_step_5", "pos_distance", "0.127").
param("run_12", "run_12_step_5", "delta_gripper", "-3.40").
param("run_12", "run_12_step_5", "efficiency", "0.56").
param("run_12", "run_12_step_5", "mean_speed", "0.0750").
param("run_12", "run_12_step_5", "n_frames", "4").
param("run_12", "run_12_step_5", "start_gripper", "7.9").
param("run_12", "run_12_step_5", "end_gripper", "4.5").
call("run_12", "run_12_llm_6", "llm_generate", "completed").
depends("run_12", "run_12_llm_6", "run_12_step_5").
call("run_12", "run_12_step_6", "adjust_position", "completed").
depends("run_12", "run_12_step_6", "run_12_llm_6").
spawned_by("run_12", "run_12_step_6", "run_12_llm_6").
param("run_12", "run_12_step_6", "pos_distance", "0.326").
param("run_12", "run_12_step_6", "delta_gripper", "-1.17").
param("run_12", "run_12_step_6", "efficiency", "0.96").
param("run_12", "run_12_step_6", "mean_speed", "0.1136").
param("run_12", "run_12_step_6", "n_frames", "4").
param("run_12", "run_12_step_6", "start_gripper", "4.5").
param("run_12", "run_12_step_6", "end_gripper", "3.3").
call("run_12", "run_12_llm_7", "llm_generate", "completed").
depends("run_12", "run_12_llm_7", "run_12_step_6").
call("run_12", "run_12_step_7", "move_z_pos_2", "completed").
depends("run_12", "run_12_step_7", "run_12_llm_7").
spawned_by("run_12", "run_12_step_7", "run_12_llm_7").
param("run_12", "run_12_step_7", "pos_distance", "1.073").
param("run_12", "run_12_step_7", "delta_gripper", "0.01").
param("run_12", "run_12_step_7", "efficiency", "0.80").
param("run_12", "run_12_step_7", "mean_speed", "0.2675").
param("run_12", "run_12_step_7", "n_frames", "6").
param("run_12", "run_12_step_7", "start_gripper", "3.3").
param("run_12", "run_12_step_7", "end_gripper", "3.3").
call("run_12", "run_12_llm_8", "llm_generate", "completed").
depends("run_12", "run_12_llm_8", "run_12_step_7").
call("run_12", "run_12_step_8", "adjust_position_9", "completed").
depends("run_12", "run_12_step_8", "run_12_llm_8").
spawned_by("run_12", "run_12_step_8", "run_12_llm_8").
param("run_12", "run_12_step_8", "pos_distance", "1.159").
param("run_12", "run_12_step_8", "delta_gripper", "0.00").
param("run_12", "run_12_step_8", "efficiency", "0.56").
param("run_12", "run_12_step_8", "mean_speed", "0.0866").
param("run_12", "run_12_step_8", "n_frames", "25").
param("run_12", "run_12_step_8", "start_gripper", "3.3").
param("run_12", "run_12_step_8", "end_gripper", "3.3").
call("run_12", "run_12_llm_9", "llm_generate", "completed").
depends("run_12", "run_12_llm_9", "run_12_step_8").
call("run_12", "run_12_notify", "notify_user", "completed").
depends("run_12", "run_12_notify", "run_12_llm_9").
spawned_by("run_12", "run_12_notify", "run_12_llm_9").

% === Episode 12 (3 discovered segments) ===
job("run_13").
job_status("run_13", "completed").
call("run_13", "run_13_step_0", "prompt_user", "completed").
call("run_13", "run_13_step_1", "llm_generate", "completed").
depends("run_13", "run_13_step_1", "run_13_step_0").
call("run_13", "run_13_llm_2", "llm_generate", "completed").
depends("run_13", "run_13_llm_2", "run_13_step_1").
call("run_13", "run_13_step_2", "adjust_position_2", "completed").
depends("run_13", "run_13_step_2", "run_13_llm_2").
spawned_by("run_13", "run_13_step_2", "run_13_llm_2").
param("run_13", "run_13_step_2", "pos_distance", "0.183").
param("run_13", "run_13_step_2", "delta_gripper", "1.75").
param("run_13", "run_13_step_2", "efficiency", "0.99").
param("run_13", "run_13_step_2", "mean_speed", "0.0616").
param("run_13", "run_13_step_2", "n_frames", "4").
param("run_13", "run_13_step_2", "start_gripper", "6.0").
param("run_13", "run_13_step_2", "end_gripper", "7.8").
call("run_13", "run_13_llm_3", "llm_generate", "completed").
depends("run_13", "run_13_llm_3", "run_13_step_2").
call("run_13", "run_13_step_3", "move_y_neg", "completed").
depends("run_13", "run_13_step_3", "run_13_llm_3").
spawned_by("run_13", "run_13_step_3", "run_13_llm_3").
param("run_13", "run_13_step_3", "pos_distance", "1.067").
param("run_13", "run_13_step_3", "delta_gripper", "0.18").
param("run_13", "run_13_step_3", "efficiency", "1.00").
param("run_13", "run_13_step_3", "mean_speed", "0.3562").
param("run_13", "run_13_step_3", "n_frames", "4").
param("run_13", "run_13_step_3", "start_gripper", "7.8").
param("run_13", "run_13_step_3", "end_gripper", "8.0").
call("run_13", "run_13_llm_4", "llm_generate", "completed").
depends("run_13", "run_13_llm_4", "run_13_step_3").
call("run_13", "run_13_step_4", "adjust_position_10", "completed").
depends("run_13", "run_13_step_4", "run_13_llm_4").
spawned_by("run_13", "run_13_step_4", "run_13_llm_4").
param("run_13", "run_13_step_4", "pos_distance", "2.617").
param("run_13", "run_13_step_4", "delta_gripper", "0.02").
param("run_13", "run_13_step_4", "efficiency", "0.64").
param("run_13", "run_13_step_4", "mean_speed", "0.0957").
param("run_13", "run_13_step_4", "n_frames", "44").
param("run_13", "run_13_step_4", "start_gripper", "8.0").
param("run_13", "run_13_step_4", "end_gripper", "8.0").
call("run_13", "run_13_llm_5", "llm_generate", "completed").
depends("run_13", "run_13_llm_5", "run_13_step_4").
call("run_13", "run_13_notify", "notify_user", "completed").
depends("run_13", "run_13_notify", "run_13_llm_5").
spawned_by("run_13", "run_13_notify", "run_13_llm_5").

% === Episode 13 (10 discovered segments) ===
job("run_14").
job_status("run_14", "completed").
call("run_14", "run_14_step_0", "prompt_user", "completed").
call("run_14", "run_14_step_1", "llm_generate", "completed").
depends("run_14", "run_14_step_1", "run_14_step_0").
call("run_14", "run_14_llm_2", "llm_generate", "completed").
depends("run_14", "run_14_llm_2", "run_14_step_1").
call("run_14", "run_14_step_2", "move_x_pos", "completed").
depends("run_14", "run_14_step_2", "run_14_llm_2").
spawned_by("run_14", "run_14_step_2", "run_14_llm_2").
param("run_14", "run_14_step_2", "pos_distance", "0.606").
param("run_14", "run_14_step_2", "delta_gripper", "1.75").
param("run_14", "run_14_step_2", "efficiency", "0.99").
param("run_14", "run_14_step_2", "mean_speed", "0.2044").
param("run_14", "run_14_step_2", "n_frames", "4").
param("run_14", "run_14_step_2", "start_gripper", "6.0").
param("run_14", "run_14_step_2", "end_gripper", "7.8").
call("run_14", "run_14_llm_3", "llm_generate", "completed").
depends("run_14", "run_14_llm_3", "run_14_step_2").
call("run_14", "run_14_step_3", "move_y_neg_5", "completed").
depends("run_14", "run_14_step_3", "run_14_llm_3").
spawned_by("run_14", "run_14_step_3", "run_14_llm_3").
param("run_14", "run_14_step_3", "pos_distance", "1.734").
param("run_14", "run_14_step_3", "delta_gripper", "0.18").
param("run_14", "run_14_step_3", "efficiency", "0.85").
param("run_14", "run_14_step_3", "mean_speed", "0.3388").
param("run_14", "run_14_step_3", "n_frames", "7").
param("run_14", "run_14_step_3", "start_gripper", "7.8").
param("run_14", "run_14_step_3", "end_gripper", "8.0").
call("run_14", "run_14_llm_4", "llm_generate", "completed").
depends("run_14", "run_14_llm_4", "run_14_step_3").
call("run_14", "run_14_step_4", "adjust_position_6", "completed").
depends("run_14", "run_14_step_4", "run_14_llm_4").
spawned_by("run_14", "run_14_step_4", "run_14_llm_4").
param("run_14", "run_14_step_4", "pos_distance", "0.855").
param("run_14", "run_14_step_4", "delta_gripper", "-0.06").
param("run_14", "run_14_step_4", "efficiency", "0.77").
param("run_14", "run_14_step_4", "mean_speed", "0.1007").
param("run_14", "run_14_step_4", "n_frames", "12").
param("run_14", "run_14_step_4", "start_gripper", "8.0").
param("run_14", "run_14_step_4", "end_gripper", "7.9").
call("run_14", "run_14_llm_5", "llm_generate", "completed").
depends("run_14", "run_14_llm_5", "run_14_step_4").
call("run_14", "run_14_step_5", "move_y_pos", "completed").
depends("run_14", "run_14_step_5", "run_14_llm_5").
spawned_by("run_14", "run_14_step_5", "run_14_llm_5").
param("run_14", "run_14_step_5", "pos_distance", "0.736").
param("run_14", "run_14_step_5", "delta_gripper", "-0.90").
param("run_14", "run_14_step_5", "efficiency", "0.97").
param("run_14", "run_14_step_5", "mean_speed", "0.2537").
param("run_14", "run_14_step_5", "n_frames", "4").
param("run_14", "run_14_step_5", "start_gripper", "7.9").
param("run_14", "run_14_step_5", "end_gripper", "7.0").
call("run_14", "run_14_llm_6", "llm_generate", "completed").
depends("run_14", "run_14_llm_6", "run_14_step_5").
call("run_14", "run_14_step_6", "adjust_position", "completed").
depends("run_14", "run_14_step_6", "run_14_llm_6").
spawned_by("run_14", "run_14_step_6", "run_14_llm_6").
param("run_14", "run_14_step_6", "pos_distance", "0.339").
param("run_14", "run_14_step_6", "delta_gripper", "-0.69").
param("run_14", "run_14_step_6", "efficiency", "0.95").
param("run_14", "run_14_step_6", "mean_speed", "0.1185").
param("run_14", "run_14_step_6", "n_frames", "4").
param("run_14", "run_14_step_6", "start_gripper", "7.0").
param("run_14", "run_14_step_6", "end_gripper", "6.3").
call("run_14", "run_14_llm_7", "llm_generate", "completed").
depends("run_14", "run_14_llm_7", "run_14_step_6").
call("run_14", "run_14_step_7", "adjust_position", "completed").
depends("run_14", "run_14_step_7", "run_14_llm_7").
spawned_by("run_14", "run_14_step_7", "run_14_llm_7").
param("run_14", "run_14_step_7", "pos_distance", "0.214").
param("run_14", "run_14_step_7", "delta_gripper", "-3.03").
param("run_14", "run_14_step_7", "efficiency", "0.69").
param("run_14", "run_14_step_7", "mean_speed", "0.1031").
param("run_14", "run_14_step_7", "n_frames", "4").
param("run_14", "run_14_step_7", "start_gripper", "6.3").
param("run_14", "run_14_step_7", "end_gripper", "3.3").
call("run_14", "run_14_llm_8", "llm_generate", "completed").
depends("run_14", "run_14_llm_8", "run_14_step_7").
call("run_14", "run_14_step_8", "move_z_pos_2", "completed").
depends("run_14", "run_14_step_8", "run_14_llm_8").
spawned_by("run_14", "run_14_step_8", "run_14_llm_8").
param("run_14", "run_14_step_8", "pos_distance", "1.616").
param("run_14", "run_14_step_8", "delta_gripper", "-0.01").
param("run_14", "run_14_step_8", "efficiency", "0.84").
param("run_14", "run_14_step_8", "mean_speed", "0.2140").
param("run_14", "run_14_step_8", "n_frames", "10").
param("run_14", "run_14_step_8", "start_gripper", "3.3").
param("run_14", "run_14_step_8", "end_gripper", "3.3").
call("run_14", "run_14_llm_9", "llm_generate", "completed").
depends("run_14", "run_14_llm_9", "run_14_step_8").
call("run_14", "run_14_step_9", "move_y_neg", "completed").
depends("run_14", "run_14_step_9", "run_14_llm_9").
spawned_by("run_14", "run_14_step_9", "run_14_llm_9").
param("run_14", "run_14_step_9", "pos_distance", "0.686").
param("run_14", "run_14_step_9", "delta_gripper", "0.69").
param("run_14", "run_14_step_9", "efficiency", "0.92").
param("run_14", "run_14_step_9", "mean_speed", "0.2500").
param("run_14", "run_14_step_9", "n_frames", "4").
param("run_14", "run_14_step_9", "start_gripper", "3.3").
param("run_14", "run_14_step_9", "end_gripper", "4.0").
call("run_14", "run_14_llm_10", "llm_generate", "completed").
depends("run_14", "run_14_llm_10", "run_14_step_9").
call("run_14", "run_14_step_10", "move_y_neg_4", "completed").
depends("run_14", "run_14_step_10", "run_14_llm_10").
spawned_by("run_14", "run_14_step_10", "run_14_llm_10").
param("run_14", "run_14_step_10", "pos_distance", "0.500").
param("run_14", "run_14_step_10", "delta_gripper", "-1.12").
param("run_14", "run_14_step_10", "efficiency", "0.58").
param("run_14", "run_14_step_10", "mean_speed", "0.2858").
param("run_14", "run_14_step_10", "n_frames", "4").
param("run_14", "run_14_step_10", "start_gripper", "4.0").
param("run_14", "run_14_step_10", "end_gripper", "2.8").
call("run_14", "run_14_llm_11", "llm_generate", "completed").
depends("run_14", "run_14_llm_11", "run_14_step_10").
call("run_14", "run_14_step_11", "close_gripper_2", "completed").
depends("run_14", "run_14_step_11", "run_14_llm_11").
spawned_by("run_14", "run_14_step_11", "run_14_llm_11").
param("run_14", "run_14_step_11", "pos_distance", "0.567").
param("run_14", "run_14_step_11", "delta_gripper", "-2.22").
param("run_14", "run_14_step_11", "efficiency", "0.63").
param("run_14", "run_14_step_11", "mean_speed", "0.1788").
param("run_14", "run_14_step_11", "n_frames", "6").
param("run_14", "run_14_step_11", "start_gripper", "2.8").
param("run_14", "run_14_step_11", "end_gripper", "0.6").
call("run_14", "run_14_llm_12", "llm_generate", "completed").
depends("run_14", "run_14_llm_12", "run_14_step_11").
call("run_14", "run_14_notify", "notify_user", "completed").
depends("run_14", "run_14_notify", "run_14_llm_12").
spawned_by("run_14", "run_14_notify", "run_14_llm_12").

% === Episode 14 (7 discovered segments) ===
job("run_15").
job_status("run_15", "completed").
call("run_15", "run_15_step_0", "prompt_user", "completed").
call("run_15", "run_15_step_1", "llm_generate", "completed").
depends("run_15", "run_15_step_1", "run_15_step_0").
call("run_15", "run_15_llm_2", "llm_generate", "completed").
depends("run_15", "run_15_llm_2", "run_15_step_1").
call("run_15", "run_15_step_2", "adjust_position_2", "completed").
depends("run_15", "run_15_step_2", "run_15_llm_2").
spawned_by("run_15", "run_15_step_2", "run_15_llm_2").
param("run_15", "run_15_step_2", "pos_distance", "0.355").
param("run_15", "run_15_step_2", "delta_gripper", "1.74").
param("run_15", "run_15_step_2", "efficiency", "1.00").
param("run_15", "run_15_step_2", "mean_speed", "0.1188").
param("run_15", "run_15_step_2", "n_frames", "4").
param("run_15", "run_15_step_2", "start_gripper", "6.0").
param("run_15", "run_15_step_2", "end_gripper", "7.8").
call("run_15", "run_15_llm_3", "llm_generate", "completed").
depends("run_15", "run_15_llm_3", "run_15_step_2").
call("run_15", "run_15_step_3", "move_y_neg", "completed").
depends("run_15", "run_15_step_3", "run_15_llm_3").
spawned_by("run_15", "run_15_step_3", "run_15_llm_3").
param("run_15", "run_15_step_3", "pos_distance", "1.123").
param("run_15", "run_15_step_3", "delta_gripper", "0.19").
param("run_15", "run_15_step_3", "efficiency", "1.00").
param("run_15", "run_15_step_3", "mean_speed", "0.3747").
param("run_15", "run_15_step_3", "n_frames", "4").
param("run_15", "run_15_step_3", "start_gripper", "7.8").
param("run_15", "run_15_step_3", "end_gripper", "8.0").
call("run_15", "run_15_llm_4", "llm_generate", "completed").
depends("run_15", "run_15_llm_4", "run_15_step_3").
call("run_15", "run_15_step_4", "adjust_position_6", "completed").
depends("run_15", "run_15_step_4", "run_15_llm_4").
spawned_by("run_15", "run_15_step_4", "run_15_llm_4").
param("run_15", "run_15_step_4", "pos_distance", "1.343").
param("run_15", "run_15_step_4", "delta_gripper", "0.01").
param("run_15", "run_15_step_4", "efficiency", "0.86").
param("run_15", "run_15_step_4", "mean_speed", "0.1306").
param("run_15", "run_15_step_4", "n_frames", "13").
param("run_15", "run_15_step_4", "start_gripper", "8.0").
param("run_15", "run_15_step_4", "end_gripper", "8.0").
call("run_15", "run_15_llm_5", "llm_generate", "completed").
depends("run_15", "run_15_llm_5", "run_15_step_4").
call("run_15", "run_15_step_5", "adjust_position", "completed").
depends("run_15", "run_15_step_5", "run_15_llm_5").
spawned_by("run_15", "run_15_step_5", "run_15_llm_5").
param("run_15", "run_15_step_5", "pos_distance", "0.061").
param("run_15", "run_15_step_5", "delta_gripper", "-0.74").
param("run_15", "run_15_step_5", "efficiency", "0.90").
param("run_15", "run_15_step_5", "mean_speed", "0.0226").
param("run_15", "run_15_step_5", "n_frames", "4").
param("run_15", "run_15_step_5", "start_gripper", "8.0").
param("run_15", "run_15_step_5", "end_gripper", "7.2").
call("run_15", "run_15_llm_6", "llm_generate", "completed").
depends("run_15", "run_15_llm_6", "run_15_step_5").
call("run_15", "run_15_step_6", "close_gripper_3", "completed").
depends("run_15", "run_15_step_6", "run_15_llm_6").
spawned_by("run_15", "run_15_step_6", "run_15_llm_6").
param("run_15", "run_15_step_6", "pos_distance", "0.046").
param("run_15", "run_15_step_6", "delta_gripper", "-3.63").
param("run_15", "run_15_step_6", "efficiency", "0.75").
param("run_15", "run_15_step_6", "mean_speed", "0.0206").
param("run_15", "run_15_step_6", "n_frames", "4").
param("run_15", "run_15_step_6", "start_gripper", "7.2").
param("run_15", "run_15_step_6", "end_gripper", "3.6").
call("run_15", "run_15_llm_7", "llm_generate", "completed").
depends("run_15", "run_15_llm_7", "run_15_step_6").
call("run_15", "run_15_step_7", "adjust_position", "completed").
depends("run_15", "run_15_step_7", "run_15_llm_7").
spawned_by("run_15", "run_15_step_7", "run_15_llm_7").
param("run_15", "run_15_step_7", "pos_distance", "0.094").
param("run_15", "run_15_step_7", "delta_gripper", "-3.53").
param("run_15", "run_15_step_7", "efficiency", "0.99").
param("run_15", "run_15_step_7", "mean_speed", "0.0315").
param("run_15", "run_15_step_7", "n_frames", "4").
param("run_15", "run_15_step_7", "start_gripper", "3.6").
param("run_15", "run_15_step_7", "end_gripper", "0.1").
call("run_15", "run_15_llm_8", "llm_generate", "completed").
depends("run_15", "run_15_llm_8", "run_15_step_7").
call("run_15", "run_15_step_8", "adjust_position_11", "completed").
depends("run_15", "run_15_step_8", "run_15_llm_8").
spawned_by("run_15", "run_15_step_8", "run_15_llm_8").
param("run_15", "run_15_step_8", "pos_distance", "0.867").
param("run_15", "run_15_step_8", "delta_gripper", "-0.04").
param("run_15", "run_15_step_8", "efficiency", "0.91").
param("run_15", "run_15_step_8", "mean_speed", "0.0433").
param("run_15", "run_15_step_8", "n_frames", "23").
param("run_15", "run_15_step_8", "start_gripper", "0.1").
param("run_15", "run_15_step_8", "end_gripper", "0.0").
call("run_15", "run_15_llm_9", "llm_generate", "completed").
depends("run_15", "run_15_llm_9", "run_15_step_8").
call("run_15", "run_15_notify", "notify_user", "completed").
depends("run_15", "run_15_notify", "run_15_llm_9").
spawned_by("run_15", "run_15_notify", "run_15_llm_9").

% === Episode 15 (14 discovered segments) ===
job("run_16").
job_status("run_16", "completed").
call("run_16", "run_16_step_0", "prompt_user", "completed").
call("run_16", "run_16_step_1", "llm_generate", "completed").
depends("run_16", "run_16_step_1", "run_16_step_0").
call("run_16", "run_16_llm_2", "llm_generate", "completed").
depends("run_16", "run_16_llm_2", "run_16_step_1").
call("run_16", "run_16_step_2", "move_x_pos", "completed").
depends("run_16", "run_16_step_2", "run_16_llm_2").
spawned_by("run_16", "run_16_step_2", "run_16_llm_2").
param("run_16", "run_16_step_2", "pos_distance", "0.723").
param("run_16", "run_16_step_2", "delta_gripper", "-0.01").
param("run_16", "run_16_step_2", "efficiency", "1.00").
param("run_16", "run_16_step_2", "mean_speed", "0.2413").
param("run_16", "run_16_step_2", "n_frames", "4").
param("run_16", "run_16_step_2", "start_gripper", "4.0").
param("run_16", "run_16_step_2", "end_gripper", "4.0").
call("run_16", "run_16_llm_3", "llm_generate", "completed").
depends("run_16", "run_16_llm_3", "run_16_step_2").
call("run_16", "run_16_step_3", "adjust_position_2", "completed").
depends("run_16", "run_16_step_3", "run_16_llm_3").
spawned_by("run_16", "run_16_step_3", "run_16_llm_3").
param("run_16", "run_16_step_3", "pos_distance", "0.403").
param("run_16", "run_16_step_3", "delta_gripper", "0.00").
param("run_16", "run_16_step_3", "efficiency", "0.67").
param("run_16", "run_16_step_3", "mean_speed", "0.2008").
param("run_16", "run_16_step_3", "n_frames", "4").
param("run_16", "run_16_step_3", "start_gripper", "4.0").
param("run_16", "run_16_step_3", "end_gripper", "4.0").
call("run_16", "run_16_llm_4", "llm_generate", "completed").
depends("run_16", "run_16_llm_4", "run_16_step_3").
call("run_16", "run_16_step_4", "move_x_pos", "completed").
depends("run_16", "run_16_step_4", "run_16_llm_4").
spawned_by("run_16", "run_16_step_4", "run_16_llm_4").
param("run_16", "run_16_step_4", "pos_distance", "0.600").
param("run_16", "run_16_step_4", "delta_gripper", "0.00").
param("run_16", "run_16_step_4", "efficiency", "0.76").
param("run_16", "run_16_step_4", "mean_speed", "0.1965").
param("run_16", "run_16_step_4", "n_frames", "5").
param("run_16", "run_16_step_4", "start_gripper", "4.0").
param("run_16", "run_16_step_4", "end_gripper", "4.0").
call("run_16", "run_16_llm_5", "llm_generate", "completed").
depends("run_16", "run_16_llm_5", "run_16_step_4").
call("run_16", "run_16_step_5", "move_y_neg_3", "completed").
depends("run_16", "run_16_step_5", "run_16_llm_5").
spawned_by("run_16", "run_16_step_5", "run_16_llm_5").
param("run_16", "run_16_step_5", "pos_distance", "0.445").
param("run_16", "run_16_step_5", "delta_gripper", "0.00").
param("run_16", "run_16_step_5", "efficiency", "0.50").
param("run_16", "run_16_step_5", "mean_speed", "0.2991").
param("run_16", "run_16_step_5", "n_frames", "4").
param("run_16", "run_16_step_5", "start_gripper", "4.0").
param("run_16", "run_16_step_5", "end_gripper", "4.0").
call("run_16", "run_16_llm_6", "llm_generate", "completed").
depends("run_16", "run_16_llm_6", "run_16_step_5").
call("run_16", "run_16_step_6", "adjust_position_8", "completed").
depends("run_16", "run_16_step_6", "run_16_llm_6").
spawned_by("run_16", "run_16_step_6", "run_16_llm_6").
param("run_16", "run_16_step_6", "pos_distance", "0.124").
param("run_16", "run_16_step_6", "delta_gripper", "0.00").
param("run_16", "run_16_step_6", "efficiency", "0.15").
param("run_16", "run_16_step_6", "mean_speed", "0.2137").
param("run_16", "run_16_step_6", "n_frames", "5").
param("run_16", "run_16_step_6", "start_gripper", "4.0").
param("run_16", "run_16_step_6", "end_gripper", "4.0").
call("run_16", "run_16_llm_7", "llm_generate", "completed").
depends("run_16", "run_16_llm_7", "run_16_step_6").
call("run_16", "run_16_step_7", "adjust_position_2", "completed").
depends("run_16", "run_16_step_7", "run_16_llm_7").
spawned_by("run_16", "run_16_step_7", "run_16_llm_7").
param("run_16", "run_16_step_7", "pos_distance", "0.189").
param("run_16", "run_16_step_7", "delta_gripper", "0.00").
param("run_16", "run_16_step_7", "efficiency", "0.41").
param("run_16", "run_16_step_7", "mean_speed", "0.1551").
param("run_16", "run_16_step_7", "n_frames", "4").
param("run_16", "run_16_step_7", "start_gripper", "4.0").
param("run_16", "run_16_step_7", "end_gripper", "4.0").
call("run_16", "run_16_llm_8", "llm_generate", "completed").
depends("run_16", "run_16_llm_8", "run_16_step_7").
call("run_16", "run_16_step_8", "adjust_position_8", "completed").
depends("run_16", "run_16_step_8", "run_16_llm_8").
spawned_by("run_16", "run_16_step_8", "run_16_llm_8").
param("run_16", "run_16_step_8", "pos_distance", "0.161").
param("run_16", "run_16_step_8", "delta_gripper", "0.00").
param("run_16", "run_16_step_8", "efficiency", "0.24").
param("run_16", "run_16_step_8", "mean_speed", "0.2256").
param("run_16", "run_16_step_8", "n_frames", "4").
param("run_16", "run_16_step_8", "start_gripper", "4.0").
param("run_16", "run_16_step_8", "end_gripper", "4.0").
call("run_16", "run_16_llm_9", "llm_generate", "completed").
depends("run_16", "run_16_llm_9", "run_16_step_8").
call("run_16", "run_16_step_9", "adjust_position_8", "completed").
depends("run_16", "run_16_step_9", "run_16_llm_9").
spawned_by("run_16", "run_16_step_9", "run_16_llm_9").
param("run_16", "run_16_step_9", "pos_distance", "0.052").
param("run_16", "run_16_step_9", "delta_gripper", "0.00").
param("run_16", "run_16_step_9", "efficiency", "0.07").
param("run_16", "run_16_step_9", "mean_speed", "0.1565").
param("run_16", "run_16_step_9", "n_frames", "6").
param("run_16", "run_16_step_9", "start_gripper", "4.0").
param("run_16", "run_16_step_9", "end_gripper", "4.0").
call("run_16", "run_16_llm_10", "llm_generate", "completed").
depends("run_16", "run_16_llm_10", "run_16_step_9").
call("run_16", "run_16_step_10", "adjust_position_2", "completed").
depends("run_16", "run_16_step_10", "run_16_llm_10").
spawned_by("run_16", "run_16_step_10", "run_16_llm_10").
param("run_16", "run_16_step_10", "pos_distance", "0.289").
param("run_16", "run_16_step_10", "delta_gripper", "0.00").
param("run_16", "run_16_step_10", "efficiency", "0.46").
param("run_16", "run_16_step_10", "mean_speed", "0.2098").
param("run_16", "run_16_step_10", "n_frames", "4").
param("run_16", "run_16_step_10", "start_gripper", "4.0").
param("run_16", "run_16_step_10", "end_gripper", "4.0").
call("run_16", "run_16_llm_11", "llm_generate", "completed").
depends("run_16", "run_16_llm_11", "run_16_step_10").
call("run_16", "run_16_step_11", "move_x_pos", "completed").
depends("run_16", "run_16_step_11", "run_16_llm_11").
spawned_by("run_16", "run_16_step_11", "run_16_llm_11").
param("run_16", "run_16_step_11", "pos_distance", "0.445").
param("run_16", "run_16_step_11", "delta_gripper", "0.00").
param("run_16", "run_16_step_11", "efficiency", "1.00").
param("run_16", "run_16_step_11", "mean_speed", "0.1487").
param("run_16", "run_16_step_11", "n_frames", "4").
param("run_16", "run_16_step_11", "start_gripper", "4.0").
param("run_16", "run_16_step_11", "end_gripper", "4.0").
call("run_16", "run_16_llm_12", "llm_generate", "completed").
depends("run_16", "run_16_llm_12", "run_16_step_11").
call("run_16", "run_16_step_12", "move_y_neg_3", "completed").
depends("run_16", "run_16_step_12", "run_16_llm_12").
spawned_by("run_16", "run_16_step_12", "run_16_llm_12").
param("run_16", "run_16_step_12", "pos_distance", "0.572").
param("run_16", "run_16_step_12", "delta_gripper", "0.00").
param("run_16", "run_16_step_12", "efficiency", "0.61").
param("run_16", "run_16_step_12", "mean_speed", "0.3112").
param("run_16", "run_16_step_12", "n_frames", "4").
param("run_16", "run_16_step_12", "start_gripper", "4.0").
param("run_16", "run_16_step_12", "end_gripper", "4.0").
call("run_16", "run_16_llm_13", "llm_generate", "completed").
depends("run_16", "run_16_llm_13", "run_16_step_12").
call("run_16", "run_16_step_13", "move_x_pos", "completed").
depends("run_16", "run_16_step_13", "run_16_llm_13").
spawned_by("run_16", "run_16_step_13", "run_16_llm_13").
param("run_16", "run_16_step_13", "pos_distance", "0.341").
param("run_16", "run_16_step_13", "delta_gripper", "-0.01").
param("run_16", "run_16_step_13", "efficiency", "0.74").
param("run_16", "run_16_step_13", "mean_speed", "0.1542").
param("run_16", "run_16_step_13", "n_frames", "4").
param("run_16", "run_16_step_13", "start_gripper", "4.0").
param("run_16", "run_16_step_13", "end_gripper", "4.0").
call("run_16", "run_16_llm_14", "llm_generate", "completed").
depends("run_16", "run_16_llm_14", "run_16_step_13").
call("run_16", "run_16_step_14", "adjust_position_8", "completed").
depends("run_16", "run_16_step_14", "run_16_llm_14").
spawned_by("run_16", "run_16_step_14", "run_16_llm_14").
param("run_16", "run_16_step_14", "pos_distance", "0.126").
param("run_16", "run_16_step_14", "delta_gripper", "0.00").
param("run_16", "run_16_step_14", "efficiency", "0.13").
param("run_16", "run_16_step_14", "mean_speed", "0.2008").
param("run_16", "run_16_step_14", "n_frames", "6").
param("run_16", "run_16_step_14", "start_gripper", "4.0").
param("run_16", "run_16_step_14", "end_gripper", "4.0").
call("run_16", "run_16_llm_15", "llm_generate", "completed").
depends("run_16", "run_16_llm_15", "run_16_step_14").
call("run_16", "run_16_step_15", "adjust_position_2", "completed").
depends("run_16", "run_16_step_15", "run_16_llm_15").
spawned_by("run_16", "run_16_step_15", "run_16_llm_15").
param("run_16", "run_16_step_15", "pos_distance", "0.303").
param("run_16", "run_16_step_15", "delta_gripper", "0.00").
param("run_16", "run_16_step_15", "efficiency", "0.33").
param("run_16", "run_16_step_15", "mean_speed", "0.2330").
param("run_16", "run_16_step_15", "n_frames", "5").
param("run_16", "run_16_step_15", "start_gripper", "4.0").
param("run_16", "run_16_step_15", "end_gripper", "4.0").
call("run_16", "run_16_llm_16", "llm_generate", "completed").
depends("run_16", "run_16_llm_16", "run_16_step_15").
call("run_16", "run_16_notify", "notify_user", "completed").
depends("run_16", "run_16_notify", "run_16_llm_16").
spawned_by("run_16", "run_16_notify", "run_16_llm_16").

% === Episode 16 (12 discovered segments) ===
job("run_17").
job_status("run_17", "completed").
call("run_17", "run_17_step_0", "prompt_user", "completed").
call("run_17", "run_17_step_1", "llm_generate", "completed").
depends("run_17", "run_17_step_1", "run_17_step_0").
call("run_17", "run_17_llm_2", "llm_generate", "completed").
depends("run_17", "run_17_llm_2", "run_17_step_1").
call("run_17", "run_17_step_2", "move_y_neg", "completed").
depends("run_17", "run_17_step_2", "run_17_llm_2").
spawned_by("run_17", "run_17_step_2", "run_17_llm_2").
param("run_17", "run_17_step_2", "pos_distance", "0.908").
param("run_17", "run_17_step_2", "delta_gripper", "1.67").
param("run_17", "run_17_step_2", "efficiency", "1.00").
param("run_17", "run_17_step_2", "mean_speed", "0.3028").
param("run_17", "run_17_step_2", "n_frames", "4").
param("run_17", "run_17_step_2", "start_gripper", "6.0").
param("run_17", "run_17_step_2", "end_gripper", "7.7").
call("run_17", "run_17_llm_3", "llm_generate", "completed").
depends("run_17", "run_17_llm_3", "run_17_step_2").
call("run_17", "run_17_step_3", "move_y_neg_2", "completed").
depends("run_17", "run_17_step_3", "run_17_llm_3").
spawned_by("run_17", "run_17_step_3", "run_17_llm_3").
param("run_17", "run_17_step_3", "pos_distance", "2.079").
param("run_17", "run_17_step_3", "delta_gripper", "0.25").
param("run_17", "run_17_step_3", "efficiency", "0.97").
param("run_17", "run_17_step_3", "mean_speed", "0.3557").
param("run_17", "run_17_step_3", "n_frames", "7").
param("run_17", "run_17_step_3", "start_gripper", "7.7").
param("run_17", "run_17_step_3", "end_gripper", "8.0").
call("run_17", "run_17_llm_4", "llm_generate", "completed").
depends("run_17", "run_17_llm_4", "run_17_step_3").
call("run_17", "run_17_step_4", "close_gripper", "completed").
depends("run_17", "run_17_step_4", "run_17_llm_4").
spawned_by("run_17", "run_17_step_4", "run_17_llm_4").
param("run_17", "run_17_step_4", "pos_distance", "0.228").
param("run_17", "run_17_step_4", "delta_gripper", "-3.07").
param("run_17", "run_17_step_4", "efficiency", "0.26").
param("run_17", "run_17_step_4", "mean_speed", "0.2188").
param("run_17", "run_17_step_4", "n_frames", "5").
param("run_17", "run_17_step_4", "start_gripper", "8.0").
param("run_17", "run_17_step_4", "end_gripper", "4.9").
call("run_17", "run_17_llm_5", "llm_generate", "completed").
depends("run_17", "run_17_llm_5", "run_17_step_4").
call("run_17", "run_17_step_5", "move_x_pos", "completed").
depends("run_17", "run_17_step_5", "run_17_llm_5").
spawned_by("run_17", "run_17_step_5", "run_17_llm_5").
param("run_17", "run_17_step_5", "pos_distance", "0.347").
param("run_17", "run_17_step_5", "delta_gripper", "-0.55").
param("run_17", "run_17_step_5", "efficiency", "0.84").
param("run_17", "run_17_step_5", "mean_speed", "0.1383").
param("run_17", "run_17_step_5", "n_frames", "4").
param("run_17", "run_17_step_5", "start_gripper", "4.9").
param("run_17", "run_17_step_5", "end_gripper", "4.3").
call("run_17", "run_17_llm_6", "llm_generate", "completed").
depends("run_17", "run_17_llm_6", "run_17_step_5").
call("run_17", "run_17_step_6", "move_y_pos", "completed").
depends("run_17", "run_17_step_6", "run_17_llm_6").
spawned_by("run_17", "run_17_step_6", "run_17_llm_6").
param("run_17", "run_17_step_6", "pos_distance", "0.811").
param("run_17", "run_17_step_6", "delta_gripper", "-0.61").
param("run_17", "run_17_step_6", "efficiency", "0.96").
param("run_17", "run_17_step_6", "mean_speed", "0.2114").
param("run_17", "run_17_step_6", "n_frames", "5").
param("run_17", "run_17_step_6", "start_gripper", "4.3").
param("run_17", "run_17_step_6", "end_gripper", "3.7").
call("run_17", "run_17_llm_7", "llm_generate", "completed").
depends("run_17", "run_17_llm_7", "run_17_step_6").
call("run_17", "run_17_step_7", "move_y_pos", "completed").
depends("run_17", "run_17_step_7", "run_17_llm_7").
spawned_by("run_17", "run_17_step_7", "run_17_llm_7").
param("run_17", "run_17_step_7", "pos_distance", "0.623").
param("run_17", "run_17_step_7", "delta_gripper", "-0.36").
param("run_17", "run_17_step_7", "efficiency", "0.72").
param("run_17", "run_17_step_7", "mean_speed", "0.2882").
param("run_17", "run_17_step_7", "n_frames", "4").
param("run_17", "run_17_step_7", "start_gripper", "3.7").
param("run_17", "run_17_step_7", "end_gripper", "3.4").
call("run_17", "run_17_llm_8", "llm_generate", "completed").
depends("run_17", "run_17_llm_8", "run_17_step_7").
call("run_17", "run_17_step_8", "move_y_pos", "completed").
depends("run_17", "run_17_step_8", "run_17_llm_8").
spawned_by("run_17", "run_17_step_8", "run_17_llm_8").
param("run_17", "run_17_step_8", "pos_distance", "0.900").
param("run_17", "run_17_step_8", "delta_gripper", "0.79").
param("run_17", "run_17_step_8", "efficiency", "0.97").
param("run_17", "run_17_step_8", "mean_speed", "0.3105").
param("run_17", "run_17_step_8", "n_frames", "4").
param("run_17", "run_17_step_8", "start_gripper", "3.4").
param("run_17", "run_17_step_8", "end_gripper", "4.2").
call("run_17", "run_17_llm_9", "llm_generate", "completed").
depends("run_17", "run_17_llm_9", "run_17_step_8").
call("run_17", "run_17_step_9", "adjust_position_2", "completed").
depends("run_17", "run_17_step_9", "run_17_llm_9").
spawned_by("run_17", "run_17_step_9", "run_17_llm_9").
param("run_17", "run_17_step_9", "pos_distance", "0.369").
param("run_17", "run_17_step_9", "delta_gripper", "2.09").
param("run_17", "run_17_step_9", "efficiency", "0.95").
param("run_17", "run_17_step_9", "mean_speed", "0.1289").
param("run_17", "run_17_step_9", "n_frames", "4").
param("run_17", "run_17_step_9", "start_gripper", "4.2").
param("run_17", "run_17_step_9", "end_gripper", "6.2").
call("run_17", "run_17_llm_10", "llm_generate", "completed").
depends("run_17", "run_17_llm_10", "run_17_step_9").
call("run_17", "run_17_step_10", "adjust_position_2", "completed").
depends("run_17", "run_17_step_10", "run_17_llm_10").
spawned_by("run_17", "run_17_step_10", "run_17_llm_10").
param("run_17", "run_17_step_10", "pos_distance", "0.498").
param("run_17", "run_17_step_10", "delta_gripper", "0.76").
param("run_17", "run_17_step_10", "efficiency", "0.52").
param("run_17", "run_17_step_10", "mean_speed", "0.1903").
param("run_17", "run_17_step_10", "n_frames", "6").
param("run_17", "run_17_step_10", "start_gripper", "6.2").
param("run_17", "run_17_step_10", "end_gripper", "7.0").
call("run_17", "run_17_llm_11", "llm_generate", "completed").
depends("run_17", "run_17_llm_11", "run_17_step_10").
call("run_17", "run_17_step_11", "move_y_neg_3", "completed").
depends("run_17", "run_17_step_11", "run_17_llm_11").
spawned_by("run_17", "run_17_step_11", "run_17_llm_11").
param("run_17", "run_17_step_11", "pos_distance", "0.727").
param("run_17", "run_17_step_11", "delta_gripper", "-0.29").
param("run_17", "run_17_step_11", "efficiency", "0.64").
param("run_17", "run_17_step_11", "mean_speed", "0.2273").
param("run_17", "run_17_step_11", "n_frames", "6").
param("run_17", "run_17_step_11", "start_gripper", "7.0").
param("run_17", "run_17_step_11", "end_gripper", "6.7").
call("run_17", "run_17_llm_12", "llm_generate", "completed").
depends("run_17", "run_17_llm_12", "run_17_step_11").
call("run_17", "run_17_step_12", "adjust_position", "completed").
depends("run_17", "run_17_step_12", "run_17_llm_12").
spawned_by("run_17", "run_17_step_12", "run_17_llm_12").
param("run_17", "run_17_step_12", "pos_distance", "0.154").
param("run_17", "run_17_step_12", "delta_gripper", "-2.20").
param("run_17", "run_17_step_12", "efficiency", "0.95").
param("run_17", "run_17_step_12", "mean_speed", "0.0541").
param("run_17", "run_17_step_12", "n_frames", "4").
param("run_17", "run_17_step_12", "start_gripper", "6.7").
param("run_17", "run_17_step_12", "end_gripper", "4.5").
call("run_17", "run_17_llm_13", "llm_generate", "completed").
depends("run_17", "run_17_llm_13", "run_17_step_12").
call("run_17", "run_17_step_13", "move_x_neg", "completed").
depends("run_17", "run_17_step_13", "run_17_llm_13").
spawned_by("run_17", "run_17_step_13", "run_17_llm_13").
param("run_17", "run_17_step_13", "pos_distance", "1.777").
param("run_17", "run_17_step_13", "delta_gripper", "-1.15").
param("run_17", "run_17_step_13", "efficiency", "0.89").
param("run_17", "run_17_step_13", "mean_speed", "0.2863").
param("run_17", "run_17_step_13", "n_frames", "8").
param("run_17", "run_17_step_13", "start_gripper", "4.5").
param("run_17", "run_17_step_13", "end_gripper", "3.4").
call("run_17", "run_17_llm_14", "llm_generate", "completed").
depends("run_17", "run_17_llm_14", "run_17_step_13").
call("run_17", "run_17_notify", "notify_user", "completed").
depends("run_17", "run_17_notify", "run_17_llm_14").
spawned_by("run_17", "run_17_notify", "run_17_llm_14").

% === Episode 17 (13 discovered segments) ===
job("run_18").
job_status("run_18", "completed").
call("run_18", "run_18_step_0", "prompt_user", "completed").
call("run_18", "run_18_step_1", "llm_generate", "completed").
depends("run_18", "run_18_step_1", "run_18_step_0").
call("run_18", "run_18_llm_2", "llm_generate", "completed").
depends("run_18", "run_18_llm_2", "run_18_step_1").
call("run_18", "run_18_step_2", "adjust_position_2", "completed").
depends("run_18", "run_18_step_2", "run_18_llm_2").
spawned_by("run_18", "run_18_step_2", "run_18_llm_2").
param("run_18", "run_18_step_2", "pos_distance", "0.517").
param("run_18", "run_18_step_2", "delta_gripper", "-1.59").
param("run_18", "run_18_step_2", "efficiency", "0.64").
param("run_18", "run_18_step_2", "mean_speed", "0.1341").
param("run_18", "run_18_step_2", "n_frames", "7").
param("run_18", "run_18_step_2", "start_gripper", "4.0").
param("run_18", "run_18_step_2", "end_gripper", "2.4").
call("run_18", "run_18_llm_3", "llm_generate", "completed").
depends("run_18", "run_18_llm_3", "run_18_step_2").
call("run_18", "run_18_step_3", "adjust_position", "completed").
depends("run_18", "run_18_step_3", "run_18_llm_3").
spawned_by("run_18", "run_18_step_3", "run_18_llm_3").
param("run_18", "run_18_step_3", "pos_distance", "0.261").
param("run_18", "run_18_step_3", "delta_gripper", "-0.81").
param("run_18", "run_18_step_3", "efficiency", "0.69").
param("run_18", "run_18_step_3", "mean_speed", "0.1263").
param("run_18", "run_18_step_3", "n_frames", "4").
param("run_18", "run_18_step_3", "start_gripper", "2.4").
param("run_18", "run_18_step_3", "end_gripper", "1.6").
call("run_18", "run_18_llm_4", "llm_generate", "completed").
depends("run_18", "run_18_llm_4", "run_18_step_3").
call("run_18", "run_18_step_4", "adjust_position_2", "completed").
depends("run_18", "run_18_step_4", "run_18_llm_4").
spawned_by("run_18", "run_18_step_4", "run_18_llm_4").
param("run_18", "run_18_step_4", "pos_distance", "0.363").
param("run_18", "run_18_step_4", "delta_gripper", "-0.45").
param("run_18", "run_18_step_4", "efficiency", "0.51").
param("run_18", "run_18_step_4", "mean_speed", "0.1793").
param("run_18", "run_18_step_4", "n_frames", "5").
param("run_18", "run_18_step_4", "start_gripper", "1.6").
param("run_18", "run_18_step_4", "end_gripper", "1.1").
call("run_18", "run_18_llm_5", "llm_generate", "completed").
depends("run_18", "run_18_llm_5", "run_18_step_4").
call("run_18", "run_18_step_5", "move_x_pos", "completed").
depends("run_18", "run_18_step_5", "run_18_llm_5").
spawned_by("run_18", "run_18_step_5", "run_18_llm_5").
param("run_18", "run_18_step_5", "pos_distance", "0.362").
param("run_18", "run_18_step_5", "delta_gripper", "-0.40").
param("run_18", "run_18_step_5", "efficiency", "0.98").
param("run_18", "run_18_step_5", "mean_speed", "0.1236").
param("run_18", "run_18_step_5", "n_frames", "4").
param("run_18", "run_18_step_5", "start_gripper", "1.1").
param("run_18", "run_18_step_5", "end_gripper", "0.8").
call("run_18", "run_18_llm_6", "llm_generate", "completed").
depends("run_18", "run_18_llm_6", "run_18_step_5").
call("run_18", "run_18_step_6", "adjust_position", "completed").
depends("run_18", "run_18_step_6", "run_18_llm_6").
spawned_by("run_18", "run_18_step_6", "run_18_llm_6").
param("run_18", "run_18_step_6", "pos_distance", "0.470").
param("run_18", "run_18_step_6", "delta_gripper", "-0.11").
param("run_18", "run_18_step_6", "efficiency", "0.80").
param("run_18", "run_18_step_6", "mean_speed", "0.1967").
param("run_18", "run_18_step_6", "n_frames", "4").
param("run_18", "run_18_step_6", "start_gripper", "0.8").
param("run_18", "run_18_step_6", "end_gripper", "0.6").
call("run_18", "run_18_llm_7", "llm_generate", "completed").
depends("run_18", "run_18_llm_7", "run_18_step_6").
call("run_18", "run_18_step_7", "adjust_position_2", "completed").
depends("run_18", "run_18_step_7", "run_18_llm_7").
spawned_by("run_18", "run_18_step_7", "run_18_llm_7").
param("run_18", "run_18_step_7", "pos_distance", "0.360").
param("run_18", "run_18_step_7", "delta_gripper", "0.96").
param("run_18", "run_18_step_7", "efficiency", "0.43").
param("run_18", "run_18_step_7", "mean_speed", "0.1389").
param("run_18", "run_18_step_7", "n_frames", "7").
param("run_18", "run_18_step_7", "start_gripper", "0.6").
param("run_18", "run_18_step_7", "end_gripper", "1.6").
call("run_18", "run_18_llm_8", "llm_generate", "completed").
depends("run_18", "run_18_llm_8", "run_18_step_7").
call("run_18", "run_18_step_8", "adjust_position_2", "completed").
depends("run_18", "run_18_step_8", "run_18_llm_8").
spawned_by("run_18", "run_18_step_8", "run_18_llm_8").
param("run_18", "run_18_step_8", "pos_distance", "0.390").
param("run_18", "run_18_step_8", "delta_gripper", "0.64").
param("run_18", "run_18_step_8", "efficiency", "0.53").
param("run_18", "run_18_step_8", "mean_speed", "0.1854").
param("run_18", "run_18_step_8", "n_frames", "5").
param("run_18", "run_18_step_8", "start_gripper", "1.6").
param("run_18", "run_18_step_8", "end_gripper", "2.2").
call("run_18", "run_18_llm_9", "llm_generate", "completed").
depends("run_18", "run_18_llm_9", "run_18_step_8").
call("run_18", "run_18_step_9", "adjust_position_2", "completed").
depends("run_18", "run_18_step_9", "run_18_llm_9").
spawned_by("run_18", "run_18_step_9", "run_18_llm_9").
param("run_18", "run_18_step_9", "pos_distance", "0.268").
param("run_18", "run_18_step_9", "delta_gripper", "0.62").
param("run_18", "run_18_step_9", "efficiency", "0.48").
param("run_18", "run_18_step_9", "mean_speed", "0.1843").
param("run_18", "run_18_step_9", "n_frames", "4").
param("run_18", "run_18_step_9", "start_gripper", "2.2").
param("run_18", "run_18_step_9", "end_gripper", "2.9").
call("run_18", "run_18_llm_10", "llm_generate", "completed").
depends("run_18", "run_18_llm_10", "run_18_step_9").
call("run_18", "run_18_step_10", "adjust_position_2", "completed").
depends("run_18", "run_18_step_10", "run_18_llm_10").
spawned_by("run_18", "run_18_step_10", "run_18_llm_10").
param("run_18", "run_18_step_10", "pos_distance", "0.194").
param("run_18", "run_18_step_10", "delta_gripper", "-0.64").
param("run_18", "run_18_step_10", "efficiency", "0.59").
param("run_18", "run_18_step_10", "mean_speed", "0.1102").
param("run_18", "run_18_step_10", "n_frames", "4").
param("run_18", "run_18_step_10", "start_gripper", "2.9").
param("run_18", "run_18_step_10", "end_gripper", "2.2").
call("run_18", "run_18_llm_11", "llm_generate", "completed").
depends("run_18", "run_18_llm_11", "run_18_step_10").
call("run_18", "run_18_step_11", "adjust_position_2", "completed").
depends("run_18", "run_18_step_11", "run_18_llm_11").
spawned_by("run_18", "run_18_step_11", "run_18_llm_11").
param("run_18", "run_18_step_11", "pos_distance", "0.189").
param("run_18", "run_18_step_11", "delta_gripper", "0.60").
param("run_18", "run_18_step_11", "efficiency", "0.41").
param("run_18", "run_18_step_11", "mean_speed", "0.1523").
param("run_18", "run_18_step_11", "n_frames", "4").
param("run_18", "run_18_step_11", "start_gripper", "2.2").
param("run_18", "run_18_step_11", "end_gripper", "2.8").
call("run_18", "run_18_llm_12", "llm_generate", "completed").
depends("run_18", "run_18_llm_12", "run_18_step_11").
call("run_18", "run_18_step_12", "adjust_position", "completed").
depends("run_18", "run_18_step_12", "run_18_llm_12").
spawned_by("run_18", "run_18_step_12", "run_18_llm_12").
param("run_18", "run_18_step_12", "pos_distance", "0.286").
param("run_18", "run_18_step_12", "delta_gripper", "-1.86").
param("run_18", "run_18_step_12", "efficiency", "0.79").
param("run_18", "run_18_step_12", "mean_speed", "0.1209").
param("run_18", "run_18_step_12", "n_frames", "4").
param("run_18", "run_18_step_12", "start_gripper", "2.8").
param("run_18", "run_18_step_12", "end_gripper", "1.0").
call("run_18", "run_18_llm_13", "llm_generate", "completed").
depends("run_18", "run_18_llm_13", "run_18_step_12").
call("run_18", "run_18_step_13", "close_gripper", "completed").
depends("run_18", "run_18_step_13", "run_18_llm_13").
spawned_by("run_18", "run_18_step_13", "run_18_llm_13").
param("run_18", "run_18_step_13", "pos_distance", "0.044").
param("run_18", "run_18_step_13", "delta_gripper", "-0.92").
param("run_18", "run_18_step_13", "efficiency", "0.16").
param("run_18", "run_18_step_13", "mean_speed", "0.0915").
param("run_18", "run_18_step_13", "n_frames", "4").
param("run_18", "run_18_step_13", "start_gripper", "1.0").
param("run_18", "run_18_step_13", "end_gripper", "0.0").
call("run_18", "run_18_llm_14", "llm_generate", "completed").
depends("run_18", "run_18_llm_14", "run_18_step_13").
call("run_18", "run_18_step_14", "adjust_position_2", "completed").
depends("run_18", "run_18_step_14", "run_18_llm_14").
spawned_by("run_18", "run_18_step_14", "run_18_llm_14").
param("run_18", "run_18_step_14", "pos_distance", "0.337").
param("run_18", "run_18_step_14", "delta_gripper", "0.70").
param("run_18", "run_18_step_14", "efficiency", "0.66").
param("run_18", "run_18_step_14", "mean_speed", "0.1023").
param("run_18", "run_18_step_14", "n_frames", "6").
param("run_18", "run_18_step_14", "start_gripper", "0.0").
param("run_18", "run_18_step_14", "end_gripper", "0.7").
call("run_18", "run_18_llm_15", "llm_generate", "completed").
depends("run_18", "run_18_llm_15", "run_18_step_14").
call("run_18", "run_18_notify", "notify_user", "completed").
depends("run_18", "run_18_notify", "run_18_llm_15").
spawned_by("run_18", "run_18_notify", "run_18_llm_15").

% === Episode 18 (10 discovered segments) ===
job("run_19").
job_status("run_19", "completed").
call("run_19", "run_19_step_0", "prompt_user", "completed").
call("run_19", "run_19_step_1", "llm_generate", "completed").
depends("run_19", "run_19_step_1", "run_19_step_0").
call("run_19", "run_19_llm_2", "llm_generate", "completed").
depends("run_19", "run_19_llm_2", "run_19_step_1").
call("run_19", "run_19_step_2", "move_x_pos", "completed").
depends("run_19", "run_19_step_2", "run_19_llm_2").
spawned_by("run_19", "run_19_step_2", "run_19_llm_2").
param("run_19", "run_19_step_2", "pos_distance", "0.987").
param("run_19", "run_19_step_2", "delta_gripper", "0.00").
param("run_19", "run_19_step_2", "efficiency", "0.96").
param("run_19", "run_19_step_2", "mean_speed", "0.1285").
param("run_19", "run_19_step_2", "n_frames", "9").
param("run_19", "run_19_step_2", "start_gripper", "4.0").
param("run_19", "run_19_step_2", "end_gripper", "4.0").
call("run_19", "run_19_llm_3", "llm_generate", "completed").
depends("run_19", "run_19_llm_3", "run_19_step_2").
call("run_19", "run_19_step_3", "adjust_position", "completed").
depends("run_19", "run_19_step_3", "run_19_llm_3").
spawned_by("run_19", "run_19_step_3", "run_19_llm_3").
param("run_19", "run_19_step_3", "pos_distance", "0.287").
param("run_19", "run_19_step_3", "delta_gripper", "-1.09").
param("run_19", "run_19_step_3", "efficiency", "1.00").
param("run_19", "run_19_step_3", "mean_speed", "0.0960").
param("run_19", "run_19_step_3", "n_frames", "4").
param("run_19", "run_19_step_3", "start_gripper", "4.0").
param("run_19", "run_19_step_3", "end_gripper", "2.9").
call("run_19", "run_19_llm_4", "llm_generate", "completed").
depends("run_19", "run_19_llm_4", "run_19_step_3").
call("run_19", "run_19_step_4", "close_gripper", "completed").
depends("run_19", "run_19_step_4", "run_19_llm_4").
spawned_by("run_19", "run_19_step_4", "run_19_llm_4").
param("run_19", "run_19_step_4", "pos_distance", "0.013").
param("run_19", "run_19_step_4", "delta_gripper", "-2.83").
param("run_19", "run_19_step_4", "efficiency", "0.19").
param("run_19", "run_19_step_4", "mean_speed", "0.0229").
param("run_19", "run_19_step_4", "n_frames", "4").
param("run_19", "run_19_step_4", "start_gripper", "2.9").
param("run_19", "run_19_step_4", "end_gripper", "0.1").
call("run_19", "run_19_llm_5", "llm_generate", "completed").
depends("run_19", "run_19_llm_5", "run_19_step_4").
call("run_19", "run_19_step_5", "move_z_pos_2", "completed").
depends("run_19", "run_19_step_5", "run_19_llm_5").
spawned_by("run_19", "run_19_step_5", "run_19_llm_5").
param("run_19", "run_19_step_5", "pos_distance", "0.891").
param("run_19", "run_19_step_5", "delta_gripper", "-0.02").
param("run_19", "run_19_step_5", "efficiency", "0.96").
param("run_19", "run_19_step_5", "mean_speed", "0.1544").
param("run_19", "run_19_step_5", "n_frames", "7").
param("run_19", "run_19_step_5", "start_gripper", "0.1").
param("run_19", "run_19_step_5", "end_gripper", "0.0").
call("run_19", "run_19_llm_6", "llm_generate", "completed").
depends("run_19", "run_19_llm_6", "run_19_step_5").
call("run_19", "run_19_step_6", "move_x_pos_4", "completed").
depends("run_19", "run_19_step_6", "run_19_llm_6").
spawned_by("run_19", "run_19_step_6", "run_19_llm_6").
param("run_19", "run_19_step_6", "pos_distance", "1.522").
param("run_19", "run_19_step_6", "delta_gripper", "-0.01").
param("run_19", "run_19_step_6", "efficiency", "1.00").
param("run_19", "run_19_step_6", "mean_speed", "0.3051").
param("run_19", "run_19_step_6", "n_frames", "6").
param("run_19", "run_19_step_6", "start_gripper", "0.0").
param("run_19", "run_19_step_6", "end_gripper", "0.0").
call("run_19", "run_19_llm_7", "llm_generate", "completed").
depends("run_19", "run_19_llm_7", "run_19_step_6").
call("run_19", "run_19_step_7", "adjust_position_2", "completed").
depends("run_19", "run_19_step_7", "run_19_llm_7").
spawned_by("run_19", "run_19_step_7", "run_19_llm_7").
param("run_19", "run_19_step_7", "pos_distance", "0.444").
param("run_19", "run_19_step_7", "delta_gripper", "0.05").
param("run_19", "run_19_step_7", "efficiency", "0.77").
param("run_19", "run_19_step_7", "mean_speed", "0.0482").
param("run_19", "run_19_step_7", "n_frames", "13").
param("run_19", "run_19_step_7", "start_gripper", "0.0").
param("run_19", "run_19_step_7", "end_gripper", "0.1").
call("run_19", "run_19_llm_8", "llm_generate", "completed").
depends("run_19", "run_19_llm_8", "run_19_step_7").
call("run_19", "run_19_step_8", "adjust_position_2", "completed").
depends("run_19", "run_19_step_8", "run_19_llm_8").
spawned_by("run_19", "run_19_step_8", "run_19_llm_8").
param("run_19", "run_19_step_8", "pos_distance", "0.050").
param("run_19", "run_19_step_8", "delta_gripper", "1.06").
param("run_19", "run_19_step_8", "efficiency", "0.85").
param("run_19", "run_19_step_8", "mean_speed", "0.0198").
param("run_19", "run_19_step_8", "n_frames", "4").
param("run_19", "run_19_step_8", "start_gripper", "0.1").
param("run_19", "run_19_step_8", "end_gripper", "1.1").
call("run_19", "run_19_llm_9", "llm_generate", "completed").
depends("run_19", "run_19_llm_9", "run_19_step_8").
call("run_19", "run_19_step_9", "adjust_position_5", "completed").
depends("run_19", "run_19_step_9", "run_19_llm_9").
spawned_by("run_19", "run_19_step_9", "run_19_llm_9").
param("run_19", "run_19_step_9", "pos_distance", "0.031").
param("run_19", "run_19_step_9", "delta_gripper", "1.92").
param("run_19", "run_19_step_9", "efficiency", "0.98").
param("run_19", "run_19_step_9", "mean_speed", "0.0107").
param("run_19", "run_19_step_9", "n_frames", "4").
param("run_19", "run_19_step_9", "start_gripper", "1.1").
param("run_19", "run_19_step_9", "end_gripper", "3.1").
call("run_19", "run_19_llm_10", "llm_generate", "completed").
depends("run_19", "run_19_llm_10", "run_19_step_9").
call("run_19", "run_19_step_10", "adjust_position_2", "completed").
depends("run_19", "run_19_step_10", "run_19_llm_10").
spawned_by("run_19", "run_19_step_10", "run_19_llm_10").
param("run_19", "run_19_step_10", "pos_distance", "0.208").
param("run_19", "run_19_step_10", "delta_gripper", "0.90").
param("run_19", "run_19_step_10", "efficiency", "1.00").
param("run_19", "run_19_step_10", "mean_speed", "0.0694").
param("run_19", "run_19_step_10", "n_frames", "4").
param("run_19", "run_19_step_10", "start_gripper", "3.1").
param("run_19", "run_19_step_10", "end_gripper", "4.0").
call("run_19", "run_19_llm_11", "llm_generate", "completed").
depends("run_19", "run_19_llm_11", "run_19_step_10").
call("run_19", "run_19_step_11", "adjust_position", "completed").
depends("run_19", "run_19_step_11", "run_19_llm_11").
spawned_by("run_19", "run_19_step_11", "run_19_llm_11").
param("run_19", "run_19_step_11", "pos_distance", "0.234").
param("run_19", "run_19_step_11", "delta_gripper", "0.00").
param("run_19", "run_19_step_11", "efficiency", "1.00").
param("run_19", "run_19_step_11", "mean_speed", "0.0781").
param("run_19", "run_19_step_11", "n_frames", "4").
param("run_19", "run_19_step_11", "start_gripper", "4.0").
param("run_19", "run_19_step_11", "end_gripper", "4.0").
call("run_19", "run_19_llm_12", "llm_generate", "completed").
depends("run_19", "run_19_llm_12", "run_19_step_11").
call("run_19", "run_19_notify", "notify_user", "completed").
depends("run_19", "run_19_notify", "run_19_llm_12").
spawned_by("run_19", "run_19_notify", "run_19_llm_12").

% === Episode 19 (6 discovered segments) ===
job("run_20").
job_status("run_20", "completed").
call("run_20", "run_20_step_0", "prompt_user", "completed").
call("run_20", "run_20_step_1", "llm_generate", "completed").
depends("run_20", "run_20_step_1", "run_20_step_0").
call("run_20", "run_20_llm_2", "llm_generate", "completed").
depends("run_20", "run_20_llm_2", "run_20_step_1").
call("run_20", "run_20_step_2", "move_y_neg", "completed").
depends("run_20", "run_20_step_2", "run_20_llm_2").
spawned_by("run_20", "run_20_step_2", "run_20_llm_2").
param("run_20", "run_20_step_2", "pos_distance", "0.517").
param("run_20", "run_20_step_2", "delta_gripper", "1.91").
param("run_20", "run_20_step_2", "efficiency", "0.97").
param("run_20", "run_20_step_2", "mean_speed", "0.1338").
param("run_20", "run_20_step_2", "n_frames", "5").
param("run_20", "run_20_step_2", "start_gripper", "6.0").
param("run_20", "run_20_step_2", "end_gripper", "7.9").
call("run_20", "run_20_llm_3", "llm_generate", "completed").
depends("run_20", "run_20_llm_3", "run_20_step_2").
call("run_20", "run_20_step_3", "move_x_pos", "completed").
depends("run_20", "run_20_step_3", "run_20_llm_3").
spawned_by("run_20", "run_20_step_3", "run_20_llm_3").
param("run_20", "run_20_step_3", "pos_distance", "0.823").
param("run_20", "run_20_step_3", "delta_gripper", "0.01").
param("run_20", "run_20_step_3", "efficiency", "0.97").
param("run_20", "run_20_step_3", "mean_speed", "0.2842").
param("run_20", "run_20_step_3", "n_frames", "4").
param("run_20", "run_20_step_3", "start_gripper", "7.9").
param("run_20", "run_20_step_3", "end_gripper", "8.0").
call("run_20", "run_20_llm_4", "llm_generate", "completed").
depends("run_20", "run_20_llm_4", "run_20_step_3").
call("run_20", "run_20_step_4", "move_x_pos_4", "completed").
depends("run_20", "run_20_step_4", "run_20_llm_4").
spawned_by("run_20", "run_20_step_4", "run_20_llm_4").
param("run_20", "run_20_step_4", "pos_distance", "1.644").
param("run_20", "run_20_step_4", "delta_gripper", "0.00").
param("run_20", "run_20_step_4", "efficiency", "0.95").
param("run_20", "run_20_step_4", "mean_speed", "0.2465").
param("run_20", "run_20_step_4", "n_frames", "8").
param("run_20", "run_20_step_4", "start_gripper", "8.0").
param("run_20", "run_20_step_4", "end_gripper", "8.0").
call("run_20", "run_20_llm_5", "llm_generate", "completed").
depends("run_20", "run_20_llm_5", "run_20_step_4").
call("run_20", "run_20_step_5", "close_gripper_3", "completed").
depends("run_20", "run_20_step_5", "run_20_llm_5").
spawned_by("run_20", "run_20_step_5", "run_20_llm_5").
param("run_20", "run_20_step_5", "pos_distance", "0.072").
param("run_20", "run_20_step_5", "delta_gripper", "-2.83").
param("run_20", "run_20_step_5", "efficiency", "0.69").
param("run_20", "run_20_step_5", "mean_speed", "0.0348").
param("run_20", "run_20_step_5", "n_frames", "4").
param("run_20", "run_20_step_5", "start_gripper", "8.0").
param("run_20", "run_20_step_5", "end_gripper", "5.1").
call("run_20", "run_20_llm_6", "llm_generate", "completed").
depends("run_20", "run_20_llm_6", "run_20_step_5").
call("run_20", "run_20_step_6", "adjust_position", "completed").
depends("run_20", "run_20_step_6", "run_20_llm_6").
spawned_by("run_20", "run_20_step_6", "run_20_llm_6").
param("run_20", "run_20_step_6", "pos_distance", "0.120").
param("run_20", "run_20_step_6", "delta_gripper", "-1.75").
param("run_20", "run_20_step_6", "efficiency", "0.82").
param("run_20", "run_20_step_6", "mean_speed", "0.0488").
param("run_20", "run_20_step_6", "n_frames", "4").
param("run_20", "run_20_step_6", "start_gripper", "5.1").
param("run_20", "run_20_step_6", "end_gripper", "3.4").
call("run_20", "run_20_llm_7", "llm_generate", "completed").
depends("run_20", "run_20_llm_7", "run_20_step_6").
call("run_20", "run_20_step_7", "adjust_position_4", "completed").
depends("run_20", "run_20_step_7", "run_20_llm_7").
spawned_by("run_20", "run_20_step_7", "run_20_llm_7").
param("run_20", "run_20_step_7", "pos_distance", "1.248").
param("run_20", "run_20_step_7", "delta_gripper", "-0.01").
param("run_20", "run_20_step_7", "efficiency", "0.87").
param("run_20", "run_20_step_7", "mean_speed", "0.0493").
param("run_20", "run_20_step_7", "n_frames", "30").
param("run_20", "run_20_step_7", "start_gripper", "3.4").
param("run_20", "run_20_step_7", "end_gripper", "3.4").
call("run_20", "run_20_llm_8", "llm_generate", "completed").
depends("run_20", "run_20_llm_8", "run_20_step_7").
call("run_20", "run_20_notify", "notify_user", "completed").
depends("run_20", "run_20_notify", "run_20_llm_8").
spawned_by("run_20", "run_20_notify", "run_20_llm_8").

% === Episode 20 (8 discovered segments) ===
job("run_21").
job_status("run_21", "completed").
call("run_21", "run_21_step_0", "prompt_user", "completed").
call("run_21", "run_21_step_1", "llm_generate", "completed").
depends("run_21", "run_21_step_1", "run_21_step_0").
call("run_21", "run_21_llm_2", "llm_generate", "completed").
depends("run_21", "run_21_llm_2", "run_21_step_1").
call("run_21", "run_21_step_2", "move_y_neg", "completed").
depends("run_21", "run_21_step_2", "run_21_llm_2").
spawned_by("run_21", "run_21_step_2", "run_21_llm_2").
param("run_21", "run_21_step_2", "pos_distance", "0.464").
param("run_21", "run_21_step_2", "delta_gripper", "1.70").
param("run_21", "run_21_step_2", "efficiency", "1.00").
param("run_21", "run_21_step_2", "mean_speed", "0.1549").
param("run_21", "run_21_step_2", "n_frames", "4").
param("run_21", "run_21_step_2", "start_gripper", "6.0").
param("run_21", "run_21_step_2", "end_gripper", "7.7").
call("run_21", "run_21_llm_3", "llm_generate", "completed").
depends("run_21", "run_21_llm_3", "run_21_step_2").
call("run_21", "run_21_step_3", "adjust_position_11", "completed").
depends("run_21", "run_21_step_3", "run_21_llm_3").
spawned_by("run_21", "run_21_step_3", "run_21_llm_3").
param("run_21", "run_21_step_3", "pos_distance", "1.016").
param("run_21", "run_21_step_3", "delta_gripper", "-0.16").
param("run_21", "run_21_step_3", "efficiency", "0.96").
param("run_21", "run_21_step_3", "mean_speed", "0.2122").
param("run_21", "run_21_step_3", "n_frames", "6").
param("run_21", "run_21_step_3", "start_gripper", "7.7").
param("run_21", "run_21_step_3", "end_gripper", "7.6").
call("run_21", "run_21_llm_4", "llm_generate", "completed").
depends("run_21", "run_21_llm_4", "run_21_step_3").
call("run_21", "run_21_step_4", "adjust_position", "completed").
depends("run_21", "run_21_step_4", "run_21_llm_4").
spawned_by("run_21", "run_21_step_4", "run_21_llm_4").
param("run_21", "run_21_step_4", "pos_distance", "0.297").
param("run_21", "run_21_step_4", "delta_gripper", "0.37").
param("run_21", "run_21_step_4", "efficiency", "0.98").
param("run_21", "run_21_step_4", "mean_speed", "0.1012").
param("run_21", "run_21_step_4", "n_frames", "4").
param("run_21", "run_21_step_4", "start_gripper", "7.6").
param("run_21", "run_21_step_4", "end_gripper", "7.9").
call("run_21", "run_21_llm_5", "llm_generate", "completed").
depends("run_21", "run_21_llm_5", "run_21_step_4").
call("run_21", "run_21_step_5", "move_y_neg_3", "completed").
depends("run_21", "run_21_step_5", "run_21_llm_5").
spawned_by("run_21", "run_21_step_5", "run_21_llm_5").
param("run_21", "run_21_step_5", "pos_distance", "0.830").
param("run_21", "run_21_step_5", "delta_gripper", "0.02").
param("run_21", "run_21_step_5", "efficiency", "0.69").
param("run_21", "run_21_step_5", "mean_speed", "0.1095").
param("run_21", "run_21_step_5", "n_frames", "12").
param("run_21", "run_21_step_5", "start_gripper", "7.9").
param("run_21", "run_21_step_5", "end_gripper", "8.0").
call("run_21", "run_21_llm_6", "llm_generate", "completed").
depends("run_21", "run_21_llm_6", "run_21_step_5").
call("run_21", "run_21_step_6", "adjust_position", "completed").
depends("run_21", "run_21_step_6", "run_21_llm_6").
spawned_by("run_21", "run_21_step_6", "run_21_llm_6").
param("run_21", "run_21_step_6", "pos_distance", "0.141").
param("run_21", "run_21_step_6", "delta_gripper", "0.01").
param("run_21", "run_21_step_6", "efficiency", "0.95").
param("run_21", "run_21_step_6", "mean_speed", "0.0495").
param("run_21", "run_21_step_6", "n_frames", "4").
param("run_21", "run_21_step_6", "start_gripper", "8.0").
param("run_21", "run_21_step_6", "end_gripper", "8.0").
call("run_21", "run_21_llm_7", "llm_generate", "completed").
depends("run_21", "run_21_llm_7", "run_21_step_6").
call("run_21", "run_21_step_7", "adjust_position", "completed").
depends("run_21", "run_21_step_7", "run_21_llm_7").
spawned_by("run_21", "run_21_step_7", "run_21_llm_7").
param("run_21", "run_21_step_7", "pos_distance", "0.151").
param("run_21", "run_21_step_7", "delta_gripper", "0.00").
param("run_21", "run_21_step_7", "efficiency", "1.00").
param("run_21", "run_21_step_7", "mean_speed", "0.0505").
param("run_21", "run_21_step_7", "n_frames", "4").
param("run_21", "run_21_step_7", "start_gripper", "8.0").
param("run_21", "run_21_step_7", "end_gripper", "8.0").
call("run_21", "run_21_llm_8", "llm_generate", "completed").
depends("run_21", "run_21_llm_8", "run_21_step_7").
call("run_21", "run_21_step_8", "adjust_position", "completed").
depends("run_21", "run_21_step_8", "run_21_llm_8").
spawned_by("run_21", "run_21_step_8", "run_21_llm_8").
param("run_21", "run_21_step_8", "pos_distance", "0.164").
param("run_21", "run_21_step_8", "delta_gripper", "0.00").
param("run_21", "run_21_step_8", "efficiency", "1.00").
param("run_21", "run_21_step_8", "mean_speed", "0.0550").
param("run_21", "run_21_step_8", "n_frames", "4").
param("run_21", "run_21_step_8", "start_gripper", "8.0").
param("run_21", "run_21_step_8", "end_gripper", "8.0").
call("run_21", "run_21_llm_9", "llm_generate", "completed").
depends("run_21", "run_21_llm_9", "run_21_step_8").
call("run_21", "run_21_step_9", "adjust_position_2", "completed").
depends("run_21", "run_21_step_9", "run_21_llm_9").
spawned_by("run_21", "run_21_step_9", "run_21_llm_9").
param("run_21", "run_21_step_9", "pos_distance", "0.175").
param("run_21", "run_21_step_9", "delta_gripper", "0.01").
param("run_21", "run_21_step_9", "efficiency", "0.77").
param("run_21", "run_21_step_9", "mean_speed", "0.0126").
param("run_21", "run_21_step_9", "n_frames", "19").
param("run_21", "run_21_step_9", "start_gripper", "8.0").
param("run_21", "run_21_step_9", "end_gripper", "8.0").
call("run_21", "run_21_llm_10", "llm_generate", "completed").
depends("run_21", "run_21_llm_10", "run_21_step_9").
call("run_21", "run_21_notify", "notify_user", "completed").
depends("run_21", "run_21_notify", "run_21_llm_10").
spawned_by("run_21", "run_21_notify", "run_21_llm_10").

% === Episode 21 (5 discovered segments) ===
job("run_22").
job_status("run_22", "completed").
call("run_22", "run_22_step_0", "prompt_user", "completed").
call("run_22", "run_22_step_1", "llm_generate", "completed").
depends("run_22", "run_22_step_1", "run_22_step_0").
call("run_22", "run_22_llm_2", "llm_generate", "completed").
depends("run_22", "run_22_llm_2", "run_22_step_1").
call("run_22", "run_22_step_2", "move_x_pos", "completed").
depends("run_22", "run_22_step_2", "run_22_llm_2").
spawned_by("run_22", "run_22_step_2", "run_22_llm_2").
param("run_22", "run_22_step_2", "pos_distance", "0.715").
param("run_22", "run_22_step_2", "delta_gripper", "1.70").
param("run_22", "run_22_step_2", "efficiency", "1.00").
param("run_22", "run_22_step_2", "mean_speed", "0.2386").
param("run_22", "run_22_step_2", "n_frames", "4").
param("run_22", "run_22_step_2", "start_gripper", "6.0").
param("run_22", "run_22_step_2", "end_gripper", "7.7").
call("run_22", "run_22_llm_3", "llm_generate", "completed").
depends("run_22", "run_22_llm_3", "run_22_step_2").
call("run_22", "run_22_step_3", "move_y_neg", "completed").
depends("run_22", "run_22_step_3", "run_22_llm_3").
spawned_by("run_22", "run_22_step_3", "run_22_llm_3").
param("run_22", "run_22_step_3", "pos_distance", "1.064").
param("run_22", "run_22_step_3", "delta_gripper", "0.23").
param("run_22", "run_22_step_3", "efficiency", "0.99").
param("run_22", "run_22_step_3", "mean_speed", "0.3585").
param("run_22", "run_22_step_3", "n_frames", "4").
param("run_22", "run_22_step_3", "start_gripper", "7.7").
param("run_22", "run_22_step_3", "end_gripper", "8.0").
call("run_22", "run_22_llm_4", "llm_generate", "completed").
depends("run_22", "run_22_llm_4", "run_22_step_3").
call("run_22", "run_22_step_4", "close_gripper_4", "completed").
depends("run_22", "run_22_step_4", "run_22_llm_4").
spawned_by("run_22", "run_22_step_4", "run_22_llm_4").
param("run_22", "run_22_step_4", "pos_distance", "0.465").
param("run_22", "run_22_step_4", "delta_gripper", "-3.83").
param("run_22", "run_22_step_4", "efficiency", "0.95").
param("run_22", "run_22_step_4", "mean_speed", "0.1637").
param("run_22", "run_22_step_4", "n_frames", "4").
param("run_22", "run_22_step_4", "start_gripper", "8.0").
param("run_22", "run_22_step_4", "end_gripper", "4.1").
call("run_22", "run_22_llm_5", "llm_generate", "completed").
depends("run_22", "run_22_llm_5", "run_22_step_4").
call("run_22", "run_22_step_5", "adjust_position", "completed").
depends("run_22", "run_22_step_5", "run_22_llm_5").
spawned_by("run_22", "run_22_step_5", "run_22_llm_5").
param("run_22", "run_22_step_5", "pos_distance", "0.285").
param("run_22", "run_22_step_5", "delta_gripper", "-0.81").
param("run_22", "run_22_step_5", "efficiency", "0.98").
param("run_22", "run_22_step_5", "mean_speed", "0.0967").
param("run_22", "run_22_step_5", "n_frames", "4").
param("run_22", "run_22_step_5", "start_gripper", "4.1").
param("run_22", "run_22_step_5", "end_gripper", "3.3").
call("run_22", "run_22_llm_6", "llm_generate", "completed").
depends("run_22", "run_22_llm_6", "run_22_step_5").
call("run_22", "run_22_step_6", "adjust_position_12", "completed").
depends("run_22", "run_22_step_6", "run_22_llm_6").
spawned_by("run_22", "run_22_step_6", "run_22_llm_6").
param("run_22", "run_22_step_6", "pos_distance", "2.020").
param("run_22", "run_22_step_6", "delta_gripper", "0.00").
param("run_22", "run_22_step_6", "efficiency", "0.48").
param("run_22", "run_22_step_6", "mean_speed", "0.1133").
param("run_22", "run_22_step_6", "n_frames", "38").
param("run_22", "run_22_step_6", "start_gripper", "3.3").
param("run_22", "run_22_step_6", "end_gripper", "3.3").
call("run_22", "run_22_llm_7", "llm_generate", "completed").
depends("run_22", "run_22_llm_7", "run_22_step_6").
call("run_22", "run_22_notify", "notify_user", "completed").
depends("run_22", "run_22_notify", "run_22_llm_7").
spawned_by("run_22", "run_22_notify", "run_22_llm_7").

% === Episode 22 (5 discovered segments) ===
job("run_23").
job_status("run_23", "completed").
call("run_23", "run_23_step_0", "prompt_user", "completed").
call("run_23", "run_23_step_1", "llm_generate", "completed").
depends("run_23", "run_23_step_1", "run_23_step_0").
call("run_23", "run_23_llm_2", "llm_generate", "completed").
depends("run_23", "run_23_llm_2", "run_23_step_1").
call("run_23", "run_23_step_2", "adjust_position_5", "completed").
depends("run_23", "run_23_step_2", "run_23_llm_2").
spawned_by("run_23", "run_23_step_2", "run_23_llm_2").
param("run_23", "run_23_step_2", "pos_distance", "0.341").
param("run_23", "run_23_step_2", "delta_gripper", "1.75").
param("run_23", "run_23_step_2", "efficiency", "0.99").
param("run_23", "run_23_step_2", "mean_speed", "0.1148").
param("run_23", "run_23_step_2", "n_frames", "4").
param("run_23", "run_23_step_2", "start_gripper", "6.0").
param("run_23", "run_23_step_2", "end_gripper", "7.8").
call("run_23", "run_23_llm_3", "llm_generate", "completed").
depends("run_23", "run_23_llm_3", "run_23_step_2").
call("run_23", "run_23_step_3", "move_y_neg", "completed").
depends("run_23", "run_23_step_3", "run_23_llm_3").
spawned_by("run_23", "run_23_step_3", "run_23_llm_3").
param("run_23", "run_23_step_3", "pos_distance", "1.015").
param("run_23", "run_23_step_3", "delta_gripper", "0.18").
param("run_23", "run_23_step_3", "efficiency", "1.00").
param("run_23", "run_23_step_3", "mean_speed", "0.3389").
param("run_23", "run_23_step_3", "n_frames", "4").
param("run_23", "run_23_step_3", "start_gripper", "7.8").
param("run_23", "run_23_step_3", "end_gripper", "8.0").
call("run_23", "run_23_llm_4", "llm_generate", "completed").
depends("run_23", "run_23_llm_4", "run_23_step_3").
call("run_23", "run_23_step_4", "adjust_position", "completed").
depends("run_23", "run_23_step_4", "run_23_llm_4").
spawned_by("run_23", "run_23_step_4", "run_23_llm_4").
param("run_23", "run_23_step_4", "pos_distance", "0.335").
param("run_23", "run_23_step_4", "delta_gripper", "-2.37").
param("run_23", "run_23_step_4", "efficiency", "0.80").
param("run_23", "run_23_step_4", "mean_speed", "0.1400").
param("run_23", "run_23_step_4", "n_frames", "4").
param("run_23", "run_23_step_4", "start_gripper", "8.0").
param("run_23", "run_23_step_4", "end_gripper", "5.6").
call("run_23", "run_23_llm_5", "llm_generate", "completed").
depends("run_23", "run_23_llm_5", "run_23_step_4").
call("run_23", "run_23_step_5", "close_gripper", "completed").
depends("run_23", "run_23_step_5", "run_23_llm_5").
spawned_by("run_23", "run_23_step_5", "run_23_llm_5").
param("run_23", "run_23_step_5", "pos_distance", "0.058").
param("run_23", "run_23_step_5", "delta_gripper", "-2.21").
param("run_23", "run_23_step_5", "efficiency", "0.54").
param("run_23", "run_23_step_5", "mean_speed", "0.0359").
param("run_23", "run_23_step_5", "n_frames", "4").
param("run_23", "run_23_step_5", "start_gripper", "5.6").
param("run_23", "run_23_step_5", "end_gripper", "3.4").
call("run_23", "run_23_llm_6", "llm_generate", "completed").
depends("run_23", "run_23_llm_6", "run_23_step_5").
call("run_23", "run_23_step_6", "adjust_position_4", "completed").
depends("run_23", "run_23_step_6", "run_23_llm_6").
spawned_by("run_23", "run_23_step_6", "run_23_llm_6").
param("run_23", "run_23_step_6", "pos_distance", "1.265").
param("run_23", "run_23_step_6", "delta_gripper", "0.00").
param("run_23", "run_23_step_6", "efficiency", "0.78").
param("run_23", "run_23_step_6", "mean_speed", "0.0438").
param("run_23", "run_23_step_6", "n_frames", "38").
param("run_23", "run_23_step_6", "start_gripper", "3.4").
param("run_23", "run_23_step_6", "end_gripper", "3.4").
call("run_23", "run_23_llm_7", "llm_generate", "completed").
depends("run_23", "run_23_llm_7", "run_23_step_6").
call("run_23", "run_23_notify", "notify_user", "completed").
depends("run_23", "run_23_notify", "run_23_llm_7").
spawned_by("run_23", "run_23_notify", "run_23_llm_7").

% === Episode 23 (7 discovered segments) ===
job("run_24").
job_status("run_24", "completed").
call("run_24", "run_24_step_0", "prompt_user", "completed").
call("run_24", "run_24_step_1", "llm_generate", "completed").
depends("run_24", "run_24_step_1", "run_24_step_0").
call("run_24", "run_24_llm_2", "llm_generate", "completed").
depends("run_24", "run_24_llm_2", "run_24_step_1").
call("run_24", "run_24_step_2", "move_x_pos", "completed").
depends("run_24", "run_24_step_2", "run_24_llm_2").
spawned_by("run_24", "run_24_step_2", "run_24_llm_2").
param("run_24", "run_24_step_2", "pos_distance", "0.685").
param("run_24", "run_24_step_2", "delta_gripper", "1.57").
param("run_24", "run_24_step_2", "efficiency", "0.99").
param("run_24", "run_24_step_2", "mean_speed", "0.2309").
param("run_24", "run_24_step_2", "n_frames", "4").
param("run_24", "run_24_step_2", "start_gripper", "6.0").
param("run_24", "run_24_step_2", "end_gripper", "7.6").
call("run_24", "run_24_llm_3", "llm_generate", "completed").
depends("run_24", "run_24_llm_3", "run_24_step_2").
call("run_24", "run_24_step_3", "move_y_neg", "completed").
depends("run_24", "run_24_step_3", "run_24_llm_3").
spawned_by("run_24", "run_24_step_3", "run_24_llm_3").
param("run_24", "run_24_step_3", "pos_distance", "0.957").
param("run_24", "run_24_step_3", "delta_gripper", "0.35").
param("run_24", "run_24_step_3", "efficiency", "0.97").
param("run_24", "run_24_step_3", "mean_speed", "0.3304").
param("run_24", "run_24_step_3", "n_frames", "4").
param("run_24", "run_24_step_3", "start_gripper", "7.6").
param("run_24", "run_24_step_3", "end_gripper", "8.0").
call("run_24", "run_24_llm_4", "llm_generate", "completed").
depends("run_24", "run_24_llm_4", "run_24_step_3").
call("run_24", "run_24_step_4", "move_y_neg_3", "completed").
depends("run_24", "run_24_step_4", "run_24_llm_4").
spawned_by("run_24", "run_24_step_4", "run_24_llm_4").
param("run_24", "run_24_step_4", "pos_distance", "0.729").
param("run_24", "run_24_step_4", "delta_gripper", "-0.15").
param("run_24", "run_24_step_4", "efficiency", "0.96").
param("run_24", "run_24_step_4", "mean_speed", "0.1520").
param("run_24", "run_24_step_4", "n_frames", "6").
param("run_24", "run_24_step_4", "start_gripper", "8.0").
param("run_24", "run_24_step_4", "end_gripper", "7.8").
call("run_24", "run_24_llm_5", "llm_generate", "completed").
depends("run_24", "run_24_llm_5", "run_24_step_4").
call("run_24", "run_24_step_5", "close_gripper", "completed").
depends("run_24", "run_24_step_5", "run_24_llm_5").
spawned_by("run_24", "run_24_step_5", "run_24_llm_5").
param("run_24", "run_24_step_5", "pos_distance", "0.244").
param("run_24", "run_24_step_5", "delta_gripper", "-1.69").
param("run_24", "run_24_step_5", "efficiency", "0.59").
param("run_24", "run_24_step_5", "mean_speed", "0.1391").
param("run_24", "run_24_step_5", "n_frames", "4").
param("run_24", "run_24_step_5", "start_gripper", "7.8").
param("run_24", "run_24_step_5", "end_gripper", "6.1").
call("run_24", "run_24_llm_6", "llm_generate", "completed").
depends("run_24", "run_24_llm_6", "run_24_step_5").
call("run_24", "run_24_step_6", "adjust_position", "completed").
depends("run_24", "run_24_step_6", "run_24_llm_6").
spawned_by("run_24", "run_24_step_6", "run_24_llm_6").
param("run_24", "run_24_step_6", "pos_distance", "0.339").
param("run_24", "run_24_step_6", "delta_gripper", "-2.86").
param("run_24", "run_24_step_6", "efficiency", "0.89").
param("run_24", "run_24_step_6", "mean_speed", "0.1267").
param("run_24", "run_24_step_6", "n_frames", "4").
param("run_24", "run_24_step_6", "start_gripper", "6.1").
param("run_24", "run_24_step_6", "end_gripper", "3.3").
call("run_24", "run_24_llm_7", "llm_generate", "completed").
depends("run_24", "run_24_llm_7", "run_24_step_6").
call("run_24", "run_24_step_7", "adjust_position", "completed").
depends("run_24", "run_24_step_7", "run_24_llm_7").
spawned_by("run_24", "run_24_step_7", "run_24_llm_7").
param("run_24", "run_24_step_7", "pos_distance", "0.248").
param("run_24", "run_24_step_7", "delta_gripper", "0.29").
param("run_24", "run_24_step_7", "efficiency", "0.88").
param("run_24", "run_24_step_7", "mean_speed", "0.0937").
param("run_24", "run_24_step_7", "n_frames", "4").
param("run_24", "run_24_step_7", "start_gripper", "3.3").
param("run_24", "run_24_step_7", "end_gripper", "3.5").
call("run_24", "run_24_llm_8", "llm_generate", "completed").
depends("run_24", "run_24_llm_8", "run_24_step_7").
call("run_24", "run_24_step_8", "move_z_pos_3", "completed").
depends("run_24", "run_24_step_8", "run_24_llm_8").
spawned_by("run_24", "run_24_step_8", "run_24_llm_8").
param("run_24", "run_24_step_8", "pos_distance", "1.819").
param("run_24", "run_24_step_8", "delta_gripper", "-0.18").
param("run_24", "run_24_step_8", "efficiency", "0.34").
param("run_24", "run_24_step_8", "mean_speed", "0.1869").
param("run_24", "run_24_step_8", "n_frames", "30").
param("run_24", "run_24_step_8", "start_gripper", "3.5").
param("run_24", "run_24_step_8", "end_gripper", "3.4").
call("run_24", "run_24_llm_9", "llm_generate", "completed").
depends("run_24", "run_24_llm_9", "run_24_step_8").
call("run_24", "run_24_notify", "notify_user", "completed").
depends("run_24", "run_24_notify", "run_24_llm_9").
spawned_by("run_24", "run_24_notify", "run_24_llm_9").

% === Episode 24 (11 discovered segments) ===
job("run_25").
job_status("run_25", "completed").
call("run_25", "run_25_step_0", "prompt_user", "completed").
call("run_25", "run_25_step_1", "llm_generate", "completed").
depends("run_25", "run_25_step_1", "run_25_step_0").
call("run_25", "run_25_llm_2", "llm_generate", "completed").
depends("run_25", "run_25_llm_2", "run_25_step_1").
call("run_25", "run_25_step_2", "open_gripper", "completed").
depends("run_25", "run_25_step_2", "run_25_llm_2").
spawned_by("run_25", "run_25_step_2", "run_25_llm_2").
param("run_25", "run_25_step_2", "pos_distance", "1.008").
param("run_25", "run_25_step_2", "delta_gripper", "1.92").
param("run_25", "run_25_step_2", "efficiency", "0.99").
param("run_25", "run_25_step_2", "mean_speed", "0.2541").
param("run_25", "run_25_step_2", "n_frames", "5").
param("run_25", "run_25_step_2", "start_gripper", "6.0").
param("run_25", "run_25_step_2", "end_gripper", "8.0").
call("run_25", "run_25_llm_3", "llm_generate", "completed").
depends("run_25", "run_25_llm_3", "run_25_step_2").
call("run_25", "run_25_step_3", "move_x_pos_4", "completed").
depends("run_25", "run_25_step_3", "run_25_llm_3").
spawned_by("run_25", "run_25_step_3", "run_25_llm_3").
param("run_25", "run_25_step_3", "pos_distance", "1.253").
param("run_25", "run_25_step_3", "delta_gripper", "0.00").
param("run_25", "run_25_step_3", "efficiency", "0.97").
param("run_25", "run_25_step_3", "mean_speed", "0.3221").
param("run_25", "run_25_step_3", "n_frames", "5").
param("run_25", "run_25_step_3", "start_gripper", "8.0").
param("run_25", "run_25_step_3", "end_gripper", "8.0").
call("run_25", "run_25_llm_4", "llm_generate", "completed").
depends("run_25", "run_25_llm_4", "run_25_step_3").
call("run_25", "run_25_step_4", "move_y_neg_4", "completed").
depends("run_25", "run_25_step_4", "run_25_llm_4").
spawned_by("run_25", "run_25_step_4", "run_25_llm_4").
param("run_25", "run_25_step_4", "pos_distance", "0.426").
param("run_25", "run_25_step_4", "delta_gripper", "-2.25").
param("run_25", "run_25_step_4", "efficiency", "0.67").
param("run_25", "run_25_step_4", "mean_speed", "0.2135").
param("run_25", "run_25_step_4", "n_frames", "4").
param("run_25", "run_25_step_4", "start_gripper", "8.0").
param("run_25", "run_25_step_4", "end_gripper", "5.7").
call("run_25", "run_25_llm_5", "llm_generate", "completed").
depends("run_25", "run_25_llm_5", "run_25_step_4").
call("run_25", "run_25_step_5", "close_gripper", "completed").
depends("run_25", "run_25_step_5", "run_25_llm_5").
spawned_by("run_25", "run_25_step_5", "run_25_llm_5").
param("run_25", "run_25_step_5", "pos_distance", "0.315").
param("run_25", "run_25_step_5", "delta_gripper", "-2.28").
param("run_25", "run_25_step_5", "efficiency", "0.48").
param("run_25", "run_25_step_5", "mean_speed", "0.2175").
param("run_25", "run_25_step_5", "n_frames", "4").
param("run_25", "run_25_step_5", "start_gripper", "5.7").
param("run_25", "run_25_step_5", "end_gripper", "3.4").
call("run_25", "run_25_llm_6", "llm_generate", "completed").
depends("run_25", "run_25_llm_6", "run_25_step_5").
call("run_25", "run_25_step_6", "move_y_neg", "completed").
depends("run_25", "run_25_step_6", "run_25_llm_6").
spawned_by("run_25", "run_25_step_6", "run_25_llm_6").
param("run_25", "run_25_step_6", "pos_distance", "0.802").
param("run_25", "run_25_step_6", "delta_gripper", "1.53").
param("run_25", "run_25_step_6", "efficiency", "0.84").
param("run_25", "run_25_step_6", "mean_speed", "0.2383").
param("run_25", "run_25_step_6", "n_frames", "5").
param("run_25", "run_25_step_6", "start_gripper", "3.4").
param("run_25", "run_25_step_6", "end_gripper", "5.0").
call("run_25", "run_25_llm_7", "llm_generate", "completed").
depends("run_25", "run_25_llm_7", "run_25_step_6").
call("run_25", "run_25_step_7", "adjust_position", "completed").
depends("run_25", "run_25_step_7", "run_25_llm_7").
spawned_by("run_25", "run_25_step_7", "run_25_llm_7").
param("run_25", "run_25_step_7", "pos_distance", "0.415").
param("run_25", "run_25_step_7", "delta_gripper", "-1.64").
param("run_25", "run_25_step_7", "efficiency", "0.94").
param("run_25", "run_25_step_7", "mean_speed", "0.1476").
param("run_25", "run_25_step_7", "n_frames", "4").
param("run_25", "run_25_step_7", "start_gripper", "5.0").
param("run_25", "run_25_step_7", "end_gripper", "3.3").
call("run_25", "run_25_llm_8", "llm_generate", "completed").
depends("run_25", "run_25_llm_8", "run_25_step_7").
call("run_25", "run_25_step_8", "move_z_pos_2", "completed").
depends("run_25", "run_25_step_8", "run_25_llm_8").
spawned_by("run_25", "run_25_step_8", "run_25_llm_8").
param("run_25", "run_25_step_8", "pos_distance", "1.121").
param("run_25", "run_25_step_8", "delta_gripper", "0.00").
param("run_25", "run_25_step_8", "efficiency", "0.74").
param("run_25", "run_25_step_8", "mean_speed", "0.2174").
param("run_25", "run_25_step_8", "n_frames", "8").
param("run_25", "run_25_step_8", "start_gripper", "3.3").
param("run_25", "run_25_step_8", "end_gripper", "3.3").
call("run_25", "run_25_llm_9", "llm_generate", "completed").
depends("run_25", "run_25_llm_9", "run_25_step_8").
call("run_25", "run_25_step_9", "adjust_position", "completed").
depends("run_25", "run_25_step_9", "run_25_llm_9").
spawned_by("run_25", "run_25_step_9", "run_25_llm_9").
param("run_25", "run_25_step_9", "pos_distance", "0.563").
param("run_25", "run_25_step_9", "delta_gripper", "0.00").
param("run_25", "run_25_step_9", "efficiency", "0.96").
param("run_25", "run_25_step_9", "mean_speed", "0.1960").
param("run_25", "run_25_step_9", "n_frames", "4").
param("run_25", "run_25_step_9", "start_gripper", "3.3").
param("run_25", "run_25_step_9", "end_gripper", "3.3").
call("run_25", "run_25_llm_10", "llm_generate", "completed").
depends("run_25", "run_25_llm_10", "run_25_step_9").
call("run_25", "run_25_step_10", "adjust_position_2", "completed").
depends("run_25", "run_25_step_10", "run_25_llm_10").
spawned_by("run_25", "run_25_step_10", "run_25_llm_10").
param("run_25", "run_25_step_10", "pos_distance", "0.140").
param("run_25", "run_25_step_10", "delta_gripper", "0.00").
param("run_25", "run_25_step_10", "efficiency", "0.55").
param("run_25", "run_25_step_10", "mean_speed", "0.0845").
param("run_25", "run_25_step_10", "n_frames", "4").
param("run_25", "run_25_step_10", "start_gripper", "3.3").
param("run_25", "run_25_step_10", "end_gripper", "3.3").
call("run_25", "run_25_llm_11", "llm_generate", "completed").
depends("run_25", "run_25_llm_11", "run_25_step_10").
call("run_25", "run_25_step_11", "move_y_pos", "completed").
depends("run_25", "run_25_step_11", "run_25_llm_11").
spawned_by("run_25", "run_25_step_11", "run_25_llm_11").
param("run_25", "run_25_step_11", "pos_distance", "0.599").
param("run_25", "run_25_step_11", "delta_gripper", "0.00").
param("run_25", "run_25_step_11", "efficiency", "1.00").
param("run_25", "run_25_step_11", "mean_speed", "0.2003").
param("run_25", "run_25_step_11", "n_frames", "4").
param("run_25", "run_25_step_11", "start_gripper", "3.3").
param("run_25", "run_25_step_11", "end_gripper", "3.3").
call("run_25", "run_25_llm_12", "llm_generate", "completed").
depends("run_25", "run_25_llm_12", "run_25_step_11").
call("run_25", "run_25_step_12", "adjust_position_6", "completed").
depends("run_25", "run_25_step_12", "run_25_llm_12").
spawned_by("run_25", "run_25_step_12", "run_25_llm_12").
param("run_25", "run_25_step_12", "pos_distance", "0.907").
param("run_25", "run_25_step_12", "delta_gripper", "0.00").
param("run_25", "run_25_step_12", "efficiency", "0.76").
param("run_25", "run_25_step_12", "mean_speed", "0.0990").
param("run_25", "run_25_step_12", "n_frames", "13").
param("run_25", "run_25_step_12", "start_gripper", "3.3").
param("run_25", "run_25_step_12", "end_gripper", "3.3").
call("run_25", "run_25_llm_13", "llm_generate", "completed").
depends("run_25", "run_25_llm_13", "run_25_step_12").
call("run_25", "run_25_notify", "notify_user", "completed").
depends("run_25", "run_25_notify", "run_25_llm_13").
spawned_by("run_25", "run_25_notify", "run_25_llm_13").

% === Episode 25 (8 discovered segments) ===
job("run_26").
job_status("run_26", "completed").
call("run_26", "run_26_step_0", "prompt_user", "completed").
call("run_26", "run_26_step_1", "llm_generate", "completed").
depends("run_26", "run_26_step_1", "run_26_step_0").
call("run_26", "run_26_llm_2", "llm_generate", "completed").
depends("run_26", "run_26_llm_2", "run_26_step_1").
call("run_26", "run_26_step_2", "move_y_neg", "completed").
depends("run_26", "run_26_step_2", "run_26_llm_2").
spawned_by("run_26", "run_26_step_2", "run_26_llm_2").
param("run_26", "run_26_step_2", "pos_distance", "0.861").
param("run_26", "run_26_step_2", "delta_gripper", "1.63").
param("run_26", "run_26_step_2", "efficiency", "1.00").
param("run_26", "run_26_step_2", "mean_speed", "0.2869").
param("run_26", "run_26_step_2", "n_frames", "4").
param("run_26", "run_26_step_2", "start_gripper", "6.0").
param("run_26", "run_26_step_2", "end_gripper", "7.7").
call("run_26", "run_26_llm_3", "llm_generate", "completed").
depends("run_26", "run_26_llm_3", "run_26_step_2").
call("run_26", "run_26_step_3", "move_y_neg_5", "completed").
depends("run_26", "run_26_step_3", "run_26_llm_3").
spawned_by("run_26", "run_26_step_3", "run_26_llm_3").
param("run_26", "run_26_step_3", "pos_distance", "1.824").
param("run_26", "run_26_step_3", "delta_gripper", "0.30").
param("run_26", "run_26_step_3", "efficiency", "0.96").
param("run_26", "run_26_step_3", "mean_speed", "0.3162").
param("run_26", "run_26_step_3", "n_frames", "7").
param("run_26", "run_26_step_3", "start_gripper", "7.7").
param("run_26", "run_26_step_3", "end_gripper", "8.0").
call("run_26", "run_26_llm_4", "llm_generate", "completed").
depends("run_26", "run_26_llm_4", "run_26_step_3").
call("run_26", "run_26_step_4", "close_gripper", "completed").
depends("run_26", "run_26_step_4", "run_26_llm_4").
spawned_by("run_26", "run_26_step_4", "run_26_llm_4").
param("run_26", "run_26_step_4", "pos_distance", "0.173").
param("run_26", "run_26_step_4", "delta_gripper", "-2.09").
param("run_26", "run_26_step_4", "efficiency", "0.51").
param("run_26", "run_26_step_4", "mean_speed", "0.1135").
param("run_26", "run_26_step_4", "n_frames", "4").
param("run_26", "run_26_step_4", "start_gripper", "8.0").
param("run_26", "run_26_step_4", "end_gripper", "5.9").
call("run_26", "run_26_llm_5", "llm_generate", "completed").
depends("run_26", "run_26_llm_5", "run_26_step_4").
call("run_26", "run_26_step_5", "adjust_position_2", "completed").
depends("run_26", "run_26_step_5", "run_26_llm_5").
spawned_by("run_26", "run_26_step_5", "run_26_llm_5").
param("run_26", "run_26_step_5", "pos_distance", "0.159").
param("run_26", "run_26_step_5", "delta_gripper", "1.40").
param("run_26", "run_26_step_5", "efficiency", "0.63").
param("run_26", "run_26_step_5", "mean_speed", "0.0841").
param("run_26", "run_26_step_5", "n_frames", "4").
param("run_26", "run_26_step_5", "start_gripper", "5.9").
param("run_26", "run_26_step_5", "end_gripper", "7.3").
call("run_26", "run_26_llm_6", "llm_generate", "completed").
depends("run_26", "run_26_llm_6", "run_26_step_5").
call("run_26", "run_26_step_6", "adjust_position_2", "completed").
depends("run_26", "run_26_step_6", "run_26_llm_6").
spawned_by("run_26", "run_26_step_6", "run_26_llm_6").
param("run_26", "run_26_step_6", "pos_distance", "0.299").
param("run_26", "run_26_step_6", "delta_gripper", "0.37").
param("run_26", "run_26_step_6", "efficiency", "0.67").
param("run_26", "run_26_step_6", "mean_speed", "0.1491").
param("run_26", "run_26_step_6", "n_frames", "4").
param("run_26", "run_26_step_6", "start_gripper", "7.3").
param("run_26", "run_26_step_6", "end_gripper", "7.6").
call("run_26", "run_26_llm_7", "llm_generate", "completed").
depends("run_26", "run_26_llm_7", "run_26_step_6").
call("run_26", "run_26_step_7", "adjust_position", "completed").
depends("run_26", "run_26_step_7", "run_26_llm_7").
spawned_by("run_26", "run_26_step_7", "run_26_llm_7").
param("run_26", "run_26_step_7", "pos_distance", "0.239").
param("run_26", "run_26_step_7", "delta_gripper", "-2.09").
param("run_26", "run_26_step_7", "efficiency", "0.65").
param("run_26", "run_26_step_7", "mean_speed", "0.1229").
param("run_26", "run_26_step_7", "n_frames", "4").
param("run_26", "run_26_step_7", "start_gripper", "7.6").
param("run_26", "run_26_step_7", "end_gripper", "5.6").
call("run_26", "run_26_llm_8", "llm_generate", "completed").
depends("run_26", "run_26_llm_8", "run_26_step_7").
call("run_26", "run_26_step_8", "adjust_position", "completed").
depends("run_26", "run_26_step_8", "run_26_llm_8").
spawned_by("run_26", "run_26_step_8", "run_26_llm_8").
param("run_26", "run_26_step_8", "pos_distance", "0.432").
param("run_26", "run_26_step_8", "delta_gripper", "-2.23").
param("run_26", "run_26_step_8", "efficiency", "0.88").
param("run_26", "run_26_step_8", "mean_speed", "0.1644").
param("run_26", "run_26_step_8", "n_frames", "4").
param("run_26", "run_26_step_8", "start_gripper", "5.6").
param("run_26", "run_26_step_8", "end_gripper", "3.3").
call("run_26", "run_26_llm_9", "llm_generate", "completed").
depends("run_26", "run_26_llm_9", "run_26_step_8").
call("run_26", "run_26_step_9", "adjust_position_13", "completed").
depends("run_26", "run_26_step_9", "run_26_llm_9").
spawned_by("run_26", "run_26_step_9", "run_26_llm_9").
param("run_26", "run_26_step_9", "pos_distance", "1.335").
param("run_26", "run_26_step_9", "delta_gripper", "0.00").
param("run_26", "run_26_step_9", "efficiency", "0.60").
param("run_26", "run_26_step_9", "mean_speed", "0.0890").
param("run_26", "run_26_step_9", "n_frames", "26").
param("run_26", "run_26_step_9", "start_gripper", "3.3").
param("run_26", "run_26_step_9", "end_gripper", "3.3").
call("run_26", "run_26_llm_10", "llm_generate", "completed").
depends("run_26", "run_26_llm_10", "run_26_step_9").
call("run_26", "run_26_notify", "notify_user", "completed").
depends("run_26", "run_26_notify", "run_26_llm_10").
spawned_by("run_26", "run_26_notify", "run_26_llm_10").

% === Episode 26 (6 discovered segments) ===
job("run_27").
job_status("run_27", "completed").
call("run_27", "run_27_step_0", "prompt_user", "completed").
call("run_27", "run_27_step_1", "llm_generate", "completed").
depends("run_27", "run_27_step_1", "run_27_step_0").
call("run_27", "run_27_llm_2", "llm_generate", "completed").
depends("run_27", "run_27_llm_2", "run_27_step_1").
call("run_27", "run_27_step_2", "adjust_position_2", "completed").
depends("run_27", "run_27_step_2", "run_27_llm_2").
spawned_by("run_27", "run_27_step_2", "run_27_llm_2").
param("run_27", "run_27_step_2", "pos_distance", "0.340").
param("run_27", "run_27_step_2", "delta_gripper", "1.92").
param("run_27", "run_27_step_2", "efficiency", "0.97").
param("run_27", "run_27_step_2", "mean_speed", "0.0877").
param("run_27", "run_27_step_2", "n_frames", "5").
param("run_27", "run_27_step_2", "start_gripper", "6.0").
param("run_27", "run_27_step_2", "end_gripper", "7.9").
call("run_27", "run_27_llm_3", "llm_generate", "completed").
depends("run_27", "run_27_llm_3", "run_27_step_2").
call("run_27", "run_27_step_3", "move_x_pos", "completed").
depends("run_27", "run_27_step_3", "run_27_llm_3").
spawned_by("run_27", "run_27_step_3", "run_27_llm_3").
param("run_27", "run_27_step_3", "pos_distance", "0.724").
param("run_27", "run_27_step_3", "delta_gripper", "0.01").
param("run_27", "run_27_step_3", "efficiency", "0.99").
param("run_27", "run_27_step_3", "mean_speed", "0.2437").
param("run_27", "run_27_step_3", "n_frames", "4").
param("run_27", "run_27_step_3", "start_gripper", "7.9").
param("run_27", "run_27_step_3", "end_gripper", "8.0").
call("run_27", "run_27_llm_4", "llm_generate", "completed").
depends("run_27", "run_27_llm_4", "run_27_step_3").
call("run_27", "run_27_step_4", "move_x_pos", "completed").
depends("run_27", "run_27_step_4", "run_27_llm_4").
spawned_by("run_27", "run_27_step_4", "run_27_llm_4").
param("run_27", "run_27_step_4", "pos_distance", "0.776").
param("run_27", "run_27_step_4", "delta_gripper", "0.00").
param("run_27", "run_27_step_4", "efficiency", "0.99").
param("run_27", "run_27_step_4", "mean_speed", "0.2601").
param("run_27", "run_27_step_4", "n_frames", "4").
param("run_27", "run_27_step_4", "start_gripper", "8.0").
param("run_27", "run_27_step_4", "end_gripper", "8.0").
call("run_27", "run_27_llm_5", "llm_generate", "completed").
depends("run_27", "run_27_llm_5", "run_27_step_4").
call("run_27", "run_27_step_5", "move_x_pos", "completed").
depends("run_27", "run_27_step_5", "run_27_llm_5").
spawned_by("run_27", "run_27_step_5", "run_27_llm_5").
param("run_27", "run_27_step_5", "pos_distance", "0.544").
param("run_27", "run_27_step_5", "delta_gripper", "0.00").
param("run_27", "run_27_step_5", "efficiency", "0.95").
param("run_27", "run_27_step_5", "mean_speed", "0.1906").
param("run_27", "run_27_step_5", "n_frames", "4").
param("run_27", "run_27_step_5", "start_gripper", "8.0").
param("run_27", "run_27_step_5", "end_gripper", "8.0").
call("run_27", "run_27_llm_6", "llm_generate", "completed").
depends("run_27", "run_27_llm_6", "run_27_step_5").
call("run_27", "run_27_step_6", "move_x_pos", "completed").
depends("run_27", "run_27_step_6", "run_27_llm_6").
spawned_by("run_27", "run_27_step_6", "run_27_llm_6").
param("run_27", "run_27_step_6", "pos_distance", "0.326").
param("run_27", "run_27_step_6", "delta_gripper", "0.00").
param("run_27", "run_27_step_6", "efficiency", "0.93").
param("run_27", "run_27_step_6", "mean_speed", "0.1167").
param("run_27", "run_27_step_6", "n_frames", "4").
param("run_27", "run_27_step_6", "start_gripper", "8.0").
param("run_27", "run_27_step_6", "end_gripper", "8.0").
call("run_27", "run_27_llm_7", "llm_generate", "completed").
depends("run_27", "run_27_llm_7", "run_27_step_6").
call("run_27", "run_27_step_7", "adjust_position_2", "completed").
depends("run_27", "run_27_step_7", "run_27_llm_7").
spawned_by("run_27", "run_27_step_7", "run_27_llm_7").
param("run_27", "run_27_step_7", "pos_distance", "0.641").
param("run_27", "run_27_step_7", "delta_gripper", "0.02").
param("run_27", "run_27_step_7", "efficiency", "0.59").
param("run_27", "run_27_step_7", "mean_speed", "0.0329").
param("run_27", "run_27_step_7", "n_frames", "34").
param("run_27", "run_27_step_7", "start_gripper", "8.0").
param("run_27", "run_27_step_7", "end_gripper", "8.0").
call("run_27", "run_27_llm_8", "llm_generate", "completed").
depends("run_27", "run_27_llm_8", "run_27_step_7").
call("run_27", "run_27_notify", "notify_user", "completed").
depends("run_27", "run_27_notify", "run_27_llm_8").
spawned_by("run_27", "run_27_notify", "run_27_llm_8").

% === Episode 27 (5 discovered segments) ===
job("run_28").
job_status("run_28", "completed").
call("run_28", "run_28_step_0", "prompt_user", "completed").
call("run_28", "run_28_step_1", "llm_generate", "completed").
depends("run_28", "run_28_step_1", "run_28_step_0").
call("run_28", "run_28_llm_2", "llm_generate", "completed").
depends("run_28", "run_28_llm_2", "run_28_step_1").
call("run_28", "run_28_step_2", "adjust_position_2", "completed").
depends("run_28", "run_28_step_2", "run_28_llm_2").
spawned_by("run_28", "run_28_step_2", "run_28_llm_2").
param("run_28", "run_28_step_2", "pos_distance", "0.106").
param("run_28", "run_28_step_2", "delta_gripper", "1.48").
param("run_28", "run_28_step_2", "efficiency", "0.96").
param("run_28", "run_28_step_2", "mean_speed", "0.0369").
param("run_28", "run_28_step_2", "n_frames", "4").
param("run_28", "run_28_step_2", "start_gripper", "6.0").
param("run_28", "run_28_step_2", "end_gripper", "7.5").
call("run_28", "run_28_llm_3", "llm_generate", "completed").
depends("run_28", "run_28_llm_3", "run_28_step_2").
call("run_28", "run_28_step_3", "move_y_neg_5", "completed").
depends("run_28", "run_28_step_3", "run_28_llm_3").
spawned_by("run_28", "run_28_step_3", "run_28_llm_3").
param("run_28", "run_28_step_3", "pos_distance", "1.730").
param("run_28", "run_28_step_3", "delta_gripper", "0.45").
param("run_28", "run_28_step_3", "efficiency", "0.97").
param("run_28", "run_28_step_3", "mean_speed", "0.2221").
param("run_28", "run_28_step_3", "n_frames", "9").
param("run_28", "run_28_step_3", "start_gripper", "7.5").
param("run_28", "run_28_step_3", "end_gripper", "8.0").
call("run_28", "run_28_llm_4", "llm_generate", "completed").
depends("run_28", "run_28_llm_4", "run_28_step_3").
call("run_28", "run_28_step_4", "adjust_position", "completed").
depends("run_28", "run_28_step_4", "run_28_llm_4").
spawned_by("run_28", "run_28_step_4", "run_28_llm_4").
param("run_28", "run_28_step_4", "pos_distance", "0.323").
param("run_28", "run_28_step_4", "delta_gripper", "-0.80").
param("run_28", "run_28_step_4", "efficiency", "0.86").
param("run_28", "run_28_step_4", "mean_speed", "0.1252").
param("run_28", "run_28_step_4", "n_frames", "4").
param("run_28", "run_28_step_4", "start_gripper", "8.0").
param("run_28", "run_28_step_4", "end_gripper", "7.2").
call("run_28", "run_28_llm_5", "llm_generate", "completed").
depends("run_28", "run_28_llm_5", "run_28_step_4").
call("run_28", "run_28_step_5", "adjust_position", "completed").
depends("run_28", "run_28_step_5", "run_28_llm_5").
spawned_by("run_28", "run_28_step_5", "run_28_llm_5").
param("run_28", "run_28_step_5", "pos_distance", "0.076").
param("run_28", "run_28_step_5", "delta_gripper", "-3.41").
param("run_28", "run_28_step_5", "efficiency", "0.86").
param("run_28", "run_28_step_5", "mean_speed", "0.0296").
param("run_28", "run_28_step_5", "n_frames", "4").
param("run_28", "run_28_step_5", "start_gripper", "7.2").
param("run_28", "run_28_step_5", "end_gripper", "3.8").
call("run_28", "run_28_llm_6", "llm_generate", "completed").
depends("run_28", "run_28_llm_6", "run_28_step_5").
call("run_28", "run_28_step_6", "adjust_position_4", "completed").
depends("run_28", "run_28_step_6", "run_28_llm_6").
spawned_by("run_28", "run_28_step_6", "run_28_llm_6").
param("run_28", "run_28_step_6", "pos_distance", "1.359").
param("run_28", "run_28_step_6", "delta_gripper", "-0.35").
param("run_28", "run_28_step_6", "efficiency", "0.88").
param("run_28", "run_28_step_6", "mean_speed", "0.0483").
param("run_28", "run_28_step_6", "n_frames", "33").
param("run_28", "run_28_step_6", "start_gripper", "3.8").
param("run_28", "run_28_step_6", "end_gripper", "3.4").
call("run_28", "run_28_llm_7", "llm_generate", "completed").
depends("run_28", "run_28_llm_7", "run_28_step_6").
call("run_28", "run_28_notify", "notify_user", "completed").
depends("run_28", "run_28_notify", "run_28_llm_7").
spawned_by("run_28", "run_28_notify", "run_28_llm_7").

% === Episode 28 (4 discovered segments) ===
job("run_29").
job_status("run_29", "completed").
call("run_29", "run_29_step_0", "prompt_user", "completed").
call("run_29", "run_29_step_1", "llm_generate", "completed").
depends("run_29", "run_29_step_1", "run_29_step_0").
call("run_29", "run_29_llm_2", "llm_generate", "completed").
depends("run_29", "run_29_llm_2", "run_29_step_1").
call("run_29", "run_29_step_2", "move_y_neg", "completed").
depends("run_29", "run_29_step_2", "run_29_llm_2").
spawned_by("run_29", "run_29_step_2", "run_29_llm_2").
param("run_29", "run_29_step_2", "pos_distance", "1.292").
param("run_29", "run_29_step_2", "delta_gripper", "1.91").
param("run_29", "run_29_step_2", "efficiency", "1.00").
param("run_29", "run_29_step_2", "mean_speed", "0.3243").
param("run_29", "run_29_step_2", "n_frames", "5").
param("run_29", "run_29_step_2", "start_gripper", "6.0").
param("run_29", "run_29_step_2", "end_gripper", "7.9").
call("run_29", "run_29_llm_3", "llm_generate", "completed").
depends("run_29", "run_29_llm_3", "run_29_step_2").
call("run_29", "run_29_step_3", "adjust_position", "completed").
depends("run_29", "run_29_step_3", "run_29_llm_3").
spawned_by("run_29", "run_29_step_3", "run_29_llm_3").
param("run_29", "run_29_step_3", "pos_distance", "0.547").
param("run_29", "run_29_step_3", "delta_gripper", "-2.40").
param("run_29", "run_29_step_3", "efficiency", "0.96").
param("run_29", "run_29_step_3", "mean_speed", "0.1905").
param("run_29", "run_29_step_3", "n_frames", "4").
param("run_29", "run_29_step_3", "start_gripper", "7.9").
param("run_29", "run_29_step_3", "end_gripper", "5.6").
call("run_29", "run_29_llm_4", "llm_generate", "completed").
depends("run_29", "run_29_llm_4", "run_29_step_3").
call("run_29", "run_29_step_4", "adjust_position", "completed").
depends("run_29", "run_29_step_4", "run_29_llm_4").
spawned_by("run_29", "run_29_step_4", "run_29_llm_4").
param("run_29", "run_29_step_4", "pos_distance", "0.237").
param("run_29", "run_29_step_4", "delta_gripper", "-2.23").
param("run_29", "run_29_step_4", "efficiency", "0.98").
param("run_29", "run_29_step_4", "mean_speed", "0.0805").
param("run_29", "run_29_step_4", "n_frames", "4").
param("run_29", "run_29_step_4", "start_gripper", "5.6").
param("run_29", "run_29_step_4", "end_gripper", "3.3").
call("run_29", "run_29_llm_5", "llm_generate", "completed").
depends("run_29", "run_29_llm_5", "run_29_step_4").
call("run_29", "run_29_step_5", "adjust_position_14", "completed").
depends("run_29", "run_29_step_5", "run_29_llm_5").
spawned_by("run_29", "run_29_step_5", "run_29_llm_5").
param("run_29", "run_29_step_5", "pos_distance", "1.895").
param("run_29", "run_29_step_5", "delta_gripper", "-0.01").
param("run_29", "run_29_step_5", "efficiency", "0.66").
param("run_29", "run_29_step_5", "mean_speed", "0.0740").
param("run_29", "run_29_step_5", "n_frames", "40").
param("run_29", "run_29_step_5", "start_gripper", "3.3").
param("run_29", "run_29_step_5", "end_gripper", "3.3").
call("run_29", "run_29_llm_6", "llm_generate", "completed").
depends("run_29", "run_29_llm_6", "run_29_step_5").
call("run_29", "run_29_notify", "notify_user", "completed").
depends("run_29", "run_29_notify", "run_29_llm_6").
spawned_by("run_29", "run_29_notify", "run_29_llm_6").

% === Episode 29 (4 discovered segments) ===
job("run_30").
job_status("run_30", "completed").
call("run_30", "run_30_step_0", "prompt_user", "completed").
call("run_30", "run_30_step_1", "llm_generate", "completed").
depends("run_30", "run_30_step_1", "run_30_step_0").
call("run_30", "run_30_llm_2", "llm_generate", "completed").
depends("run_30", "run_30_llm_2", "run_30_step_1").
call("run_30", "run_30_step_2", "move_x_pos", "completed").
depends("run_30", "run_30_step_2", "run_30_llm_2").
spawned_by("run_30", "run_30_step_2", "run_30_llm_2").
param("run_30", "run_30_step_2", "pos_distance", "0.352").
param("run_30", "run_30_step_2", "delta_gripper", "1.50").
param("run_30", "run_30_step_2", "efficiency", "0.99").
param("run_30", "run_30_step_2", "mean_speed", "0.1189").
param("run_30", "run_30_step_2", "n_frames", "4").
param("run_30", "run_30_step_2", "start_gripper", "6.1").
param("run_30", "run_30_step_2", "end_gripper", "7.6").
call("run_30", "run_30_llm_3", "llm_generate", "completed").
depends("run_30", "run_30_llm_3", "run_30_step_2").
call("run_30", "run_30_step_3", "move_y_neg_5", "completed").
depends("run_30", "run_30_step_3", "run_30_llm_3").
spawned_by("run_30", "run_30_step_3", "run_30_llm_3").
param("run_30", "run_30_step_3", "pos_distance", "1.736").
param("run_30", "run_30_step_3", "delta_gripper", "0.41").
param("run_30", "run_30_step_3", "efficiency", "0.99").
param("run_30", "run_30_step_3", "mean_speed", "0.2502").
param("run_30", "run_30_step_3", "n_frames", "8").
param("run_30", "run_30_step_3", "start_gripper", "7.6").
param("run_30", "run_30_step_3", "end_gripper", "8.0").
call("run_30", "run_30_llm_4", "llm_generate", "completed").
depends("run_30", "run_30_llm_4", "run_30_step_3").
call("run_30", "run_30_step_4", "move_x_pos_4", "completed").
depends("run_30", "run_30_step_4", "run_30_llm_4").
spawned_by("run_30", "run_30_step_4", "run_30_llm_4").
param("run_30", "run_30_step_4", "pos_distance", "1.192").
param("run_30", "run_30_step_4", "delta_gripper", "0.00").
param("run_30", "run_30_step_4", "efficiency", "1.00").
param("run_30", "run_30_step_4", "mean_speed", "0.3982").
param("run_30", "run_30_step_4", "n_frames", "4").
param("run_30", "run_30_step_4", "start_gripper", "8.0").
param("run_30", "run_30_step_4", "end_gripper", "8.0").
call("run_30", "run_30_llm_5", "llm_generate", "completed").
depends("run_30", "run_30_llm_5", "run_30_step_4").
call("run_30", "run_30_step_5", "adjust_position_15", "completed").
depends("run_30", "run_30_step_5", "run_30_llm_5").
spawned_by("run_30", "run_30_step_5", "run_30_llm_5").
param("run_30", "run_30_step_5", "pos_distance", "0.448").
param("run_30", "run_30_step_5", "delta_gripper", "0.02").
param("run_30", "run_30_step_5", "efficiency", "0.31").
param("run_30", "run_30_step_5", "mean_speed", "0.0396").
param("run_30", "run_30_step_5", "n_frames", "37").
param("run_30", "run_30_step_5", "start_gripper", "8.0").
param("run_30", "run_30_step_5", "end_gripper", "8.0").
call("run_30", "run_30_llm_6", "llm_generate", "completed").
depends("run_30", "run_30_llm_6", "run_30_step_5").
call("run_30", "run_30_notify", "notify_user", "completed").
depends("run_30", "run_30_notify", "run_30_llm_6").
spawned_by("run_30", "run_30_notify", "run_30_llm_6").

% === Episode 30 (6 discovered segments) ===
job("run_31").
job_status("run_31", "completed").
call("run_31", "run_31_step_0", "prompt_user", "completed").
call("run_31", "run_31_step_1", "llm_generate", "completed").
depends("run_31", "run_31_step_1", "run_31_step_0").
call("run_31", "run_31_llm_2", "llm_generate", "completed").
depends("run_31", "run_31_llm_2", "run_31_step_1").
call("run_31", "run_31_step_2", "adjust_position_2", "completed").
depends("run_31", "run_31_step_2", "run_31_llm_2").
spawned_by("run_31", "run_31_step_2", "run_31_llm_2").
param("run_31", "run_31_step_2", "pos_distance", "0.441").
param("run_31", "run_31_step_2", "delta_gripper", "1.91").
param("run_31", "run_31_step_2", "efficiency", "0.97").
param("run_31", "run_31_step_2", "mean_speed", "0.1134").
param("run_31", "run_31_step_2", "n_frames", "5").
param("run_31", "run_31_step_2", "start_gripper", "6.0").
param("run_31", "run_31_step_2", "end_gripper", "7.9").
call("run_31", "run_31_llm_3", "llm_generate", "completed").
depends("run_31", "run_31_llm_3", "run_31_step_2").
call("run_31", "run_31_step_3", "move_x_pos", "completed").
depends("run_31", "run_31_step_3", "run_31_llm_3").
spawned_by("run_31", "run_31_step_3", "run_31_llm_3").
param("run_31", "run_31_step_3", "pos_distance", "1.080").
param("run_31", "run_31_step_3", "delta_gripper", "0.01").
param("run_31", "run_31_step_3", "efficiency", "0.97").
param("run_31", "run_31_step_3", "mean_speed", "0.2771").
param("run_31", "run_31_step_3", "n_frames", "5").
param("run_31", "run_31_step_3", "start_gripper", "7.9").
param("run_31", "run_31_step_3", "end_gripper", "8.0").
call("run_31", "run_31_llm_4", "llm_generate", "completed").
depends("run_31", "run_31_llm_4", "run_31_step_3").
call("run_31", "run_31_step_4", "move_x_pos", "completed").
depends("run_31", "run_31_step_4", "run_31_llm_4").
spawned_by("run_31", "run_31_step_4", "run_31_llm_4").
param("run_31", "run_31_step_4", "pos_distance", "0.686").
param("run_31", "run_31_step_4", "delta_gripper", "0.00").
param("run_31", "run_31_step_4", "efficiency", "0.98").
param("run_31", "run_31_step_4", "mean_speed", "0.2339").
param("run_31", "run_31_step_4", "n_frames", "4").
param("run_31", "run_31_step_4", "start_gripper", "8.0").
param("run_31", "run_31_step_4", "end_gripper", "8.0").
call("run_31", "run_31_llm_5", "llm_generate", "completed").
depends("run_31", "run_31_llm_5", "run_31_step_4").
call("run_31", "run_31_step_5", "move_x_pos", "completed").
depends("run_31", "run_31_step_5", "run_31_llm_5").
spawned_by("run_31", "run_31_step_5", "run_31_llm_5").
param("run_31", "run_31_step_5", "pos_distance", "0.376").
param("run_31", "run_31_step_5", "delta_gripper", "0.00").
param("run_31", "run_31_step_5", "efficiency", "0.95").
param("run_31", "run_31_step_5", "mean_speed", "0.1317").
param("run_31", "run_31_step_5", "n_frames", "4").
param("run_31", "run_31_step_5", "start_gripper", "8.0").
param("run_31", "run_31_step_5", "end_gripper", "8.0").
call("run_31", "run_31_llm_6", "llm_generate", "completed").
depends("run_31", "run_31_llm_6", "run_31_step_5").
call("run_31", "run_31_step_6", "move_x_pos", "completed").
depends("run_31", "run_31_step_6", "run_31_llm_6").
spawned_by("run_31", "run_31_step_6", "run_31_llm_6").
param("run_31", "run_31_step_6", "pos_distance", "0.381").
param("run_31", "run_31_step_6", "delta_gripper", "0.00").
param("run_31", "run_31_step_6", "efficiency", "0.97").
param("run_31", "run_31_step_6", "mean_speed", "0.1310").
param("run_31", "run_31_step_6", "n_frames", "4").
param("run_31", "run_31_step_6", "start_gripper", "8.0").
param("run_31", "run_31_step_6", "end_gripper", "8.0").
call("run_31", "run_31_llm_7", "llm_generate", "completed").
depends("run_31", "run_31_llm_7", "run_31_step_6").
call("run_31", "run_31_step_7", "adjust_position_2", "completed").
depends("run_31", "run_31_step_7", "run_31_llm_7").
spawned_by("run_31", "run_31_step_7", "run_31_llm_7").
param("run_31", "run_31_step_7", "pos_distance", "0.267").
param("run_31", "run_31_step_7", "delta_gripper", "0.03").
param("run_31", "run_31_step_7", "efficiency", "0.41").
param("run_31", "run_31_step_7", "mean_speed", "0.0202").
param("run_31", "run_31_step_7", "n_frames", "33").
param("run_31", "run_31_step_7", "start_gripper", "8.0").
param("run_31", "run_31_step_7", "end_gripper", "8.0").
call("run_31", "run_31_llm_8", "llm_generate", "completed").
depends("run_31", "run_31_llm_8", "run_31_step_7").
call("run_31", "run_31_notify", "notify_user", "completed").
depends("run_31", "run_31_notify", "run_31_llm_8").
spawned_by("run_31", "run_31_notify", "run_31_llm_8").

% === Episode 31 (11 discovered segments) ===
job("run_32").
job_status("run_32", "completed").
call("run_32", "run_32_step_0", "prompt_user", "completed").
call("run_32", "run_32_step_1", "llm_generate", "completed").
depends("run_32", "run_32_step_1", "run_32_step_0").
call("run_32", "run_32_llm_2", "llm_generate", "completed").
depends("run_32", "run_32_llm_2", "run_32_step_1").
call("run_32", "run_32_step_2", "move_y_neg", "completed").
depends("run_32", "run_32_step_2", "run_32_llm_2").
spawned_by("run_32", "run_32_step_2", "run_32_llm_2").
param("run_32", "run_32_step_2", "pos_distance", "1.197").
param("run_32", "run_32_step_2", "delta_gripper", "1.92").
param("run_32", "run_32_step_2", "efficiency", "0.98").
param("run_32", "run_32_step_2", "mean_speed", "0.2441").
param("run_32", "run_32_step_2", "n_frames", "6").
param("run_32", "run_32_step_2", "start_gripper", "6.0").
param("run_32", "run_32_step_2", "end_gripper", "8.0").
call("run_32", "run_32_llm_3", "llm_generate", "completed").
depends("run_32", "run_32_llm_3", "run_32_step_2").
call("run_32", "run_32_step_3", "move_x_pos", "completed").
depends("run_32", "run_32_step_3", "run_32_llm_3").
spawned_by("run_32", "run_32_step_3", "run_32_llm_3").
param("run_32", "run_32_step_3", "pos_distance", "0.784").
param("run_32", "run_32_step_3", "delta_gripper", "0.00").
param("run_32", "run_32_step_3", "efficiency", "0.98").
param("run_32", "run_32_step_3", "mean_speed", "0.2668").
param("run_32", "run_32_step_3", "n_frames", "4").
param("run_32", "run_32_step_3", "start_gripper", "8.0").
param("run_32", "run_32_step_3", "end_gripper", "8.0").
call("run_32", "run_32_llm_4", "llm_generate", "completed").
depends("run_32", "run_32_llm_4", "run_32_step_3").
call("run_32", "run_32_step_4", "move_y_neg_5", "completed").
depends("run_32", "run_32_step_4", "run_32_llm_4").
spawned_by("run_32", "run_32_step_4", "run_32_llm_4").
param("run_32", "run_32_step_4", "pos_distance", "1.327").
param("run_32", "run_32_step_4", "delta_gripper", "0.00").
param("run_32", "run_32_step_4", "efficiency", "0.64").
param("run_32", "run_32_step_4", "mean_speed", "0.2589").
param("run_32", "run_32_step_4", "n_frames", "9").
param("run_32", "run_32_step_4", "start_gripper", "8.0").
param("run_32", "run_32_step_4", "end_gripper", "8.0").
call("run_32", "run_32_llm_5", "llm_generate", "completed").
depends("run_32", "run_32_llm_5", "run_32_step_4").
call("run_32", "run_32_step_5", "adjust_position", "completed").
depends("run_32", "run_32_step_5", "run_32_llm_5").
spawned_by("run_32", "run_32_step_5", "run_32_llm_5").
param("run_32", "run_32_step_5", "pos_distance", "0.554").
param("run_32", "run_32_step_5", "delta_gripper", "-2.37").
param("run_32", "run_32_step_5", "efficiency", "0.62").
param("run_32", "run_32_step_5", "mean_speed", "0.1798").
param("run_32", "run_32_step_5", "n_frames", "6").
param("run_32", "run_32_step_5", "start_gripper", "8.0").
param("run_32", "run_32_step_5", "end_gripper", "5.6").
call("run_32", "run_32_llm_6", "llm_generate", "completed").
depends("run_32", "run_32_llm_6", "run_32_step_5").
call("run_32", "run_32_step_6", "adjust_position_2", "completed").
depends("run_32", "run_32_step_6", "run_32_llm_6").
spawned_by("run_32", "run_32_step_6", "run_32_llm_6").
param("run_32", "run_32_step_6", "pos_distance", "0.317").
param("run_32", "run_32_step_6", "delta_gripper", "0.64").
param("run_32", "run_32_step_6", "efficiency", "0.74").
param("run_32", "run_32_step_6", "mean_speed", "0.1420").
param("run_32", "run_32_step_6", "n_frames", "4").
param("run_32", "run_32_step_6", "start_gripper", "5.6").
param("run_32", "run_32_step_6", "end_gripper", "6.2").
call("run_32", "run_32_llm_7", "llm_generate", "completed").
depends("run_32", "run_32_llm_7", "run_32_step_6").
call("run_32", "run_32_step_7", "close_gripper", "completed").
depends("run_32", "run_32_step_7", "run_32_llm_7").
spawned_by("run_32", "run_32_step_7", "run_32_llm_7").
param("run_32", "run_32_step_7", "pos_distance", "0.387").
param("run_32", "run_32_step_7", "delta_gripper", "-2.23").
param("run_32", "run_32_step_7", "efficiency", "0.27").
param("run_32", "run_32_step_7", "mean_speed", "0.1610").
param("run_32", "run_32_step_7", "n_frames", "10").
param("run_32", "run_32_step_7", "start_gripper", "6.2").
param("run_32", "run_32_step_7", "end_gripper", "4.0").
call("run_32", "run_32_llm_8", "llm_generate", "completed").
depends("run_32", "run_32_llm_8", "run_32_step_7").
call("run_32", "run_32_step_8", "adjust_position", "completed").
depends("run_32", "run_32_step_8", "run_32_llm_8").
spawned_by("run_32", "run_32_step_8", "run_32_llm_8").
param("run_32", "run_32_step_8", "pos_distance", "0.480").
param("run_32", "run_32_step_8", "delta_gripper", "-0.90").
param("run_32", "run_32_step_8", "efficiency", "0.67").
param("run_32", "run_32_step_8", "mean_speed", "0.2375").
param("run_32", "run_32_step_8", "n_frames", "4").
param("run_32", "run_32_step_8", "start_gripper", "4.0").
param("run_32", "run_32_step_8", "end_gripper", "3.1").
call("run_32", "run_32_llm_9", "llm_generate", "completed").
depends("run_32", "run_32_llm_9", "run_32_step_8").
call("run_32", "run_32_step_9", "close_gripper", "completed").
depends("run_32", "run_32_step_9", "run_32_llm_9").
spawned_by("run_32", "run_32_step_9", "run_32_llm_9").
param("run_32", "run_32_step_9", "pos_distance", "0.317").
param("run_32", "run_32_step_9", "delta_gripper", "-1.49").
param("run_32", "run_32_step_9", "efficiency", "0.48").
param("run_32", "run_32_step_9", "mean_speed", "0.2201").
param("run_32", "run_32_step_9", "n_frames", "4").
param("run_32", "run_32_step_9", "start_gripper", "3.1").
param("run_32", "run_32_step_9", "end_gripper", "1.6").
call("run_32", "run_32_llm_10", "llm_generate", "completed").
depends("run_32", "run_32_llm_10", "run_32_step_9").
call("run_32", "run_32_step_10", "move_x_neg_2", "completed").
depends("run_32", "run_32_step_10", "run_32_llm_10").
spawned_by("run_32", "run_32_step_10", "run_32_llm_10").
param("run_32", "run_32_step_10", "pos_distance", "0.816").
param("run_32", "run_32_step_10", "delta_gripper", "0.82").
param("run_32", "run_32_step_10", "efficiency", "0.96").
param("run_32", "run_32_step_10", "mean_speed", "0.2835").
param("run_32", "run_32_step_10", "n_frames", "4").
param("run_32", "run_32_step_10", "start_gripper", "1.6").
param("run_32", "run_32_step_10", "end_gripper", "2.4").
call("run_32", "run_32_llm_11", "llm_generate", "completed").
depends("run_32", "run_32_llm_11", "run_32_step_10").
call("run_32", "run_32_step_11", "adjust_position_2", "completed").
depends("run_32", "run_32_step_11", "run_32_llm_11").
spawned_by("run_32", "run_32_step_11", "run_32_llm_11").
param("run_32", "run_32_step_11", "pos_distance", "0.189").
param("run_32", "run_32_step_11", "delta_gripper", "0.16").
param("run_32", "run_32_step_11", "efficiency", "0.72").
param("run_32", "run_32_step_11", "mean_speed", "0.0876").
param("run_32", "run_32_step_11", "n_frames", "4").
param("run_32", "run_32_step_11", "start_gripper", "2.4").
param("run_32", "run_32_step_11", "end_gripper", "2.6").
call("run_32", "run_32_llm_12", "llm_generate", "completed").
depends("run_32", "run_32_llm_12", "run_32_step_11").
call("run_32", "run_32_step_12", "adjust_position", "completed").
depends("run_32", "run_32_step_12", "run_32_llm_12").
spawned_by("run_32", "run_32_step_12", "run_32_llm_12").
param("run_32", "run_32_step_12", "pos_distance", "0.249").
param("run_32", "run_32_step_12", "delta_gripper", "-0.92").
param("run_32", "run_32_step_12", "efficiency", "0.75").
param("run_32", "run_32_step_12", "mean_speed", "0.0833").
param("run_32", "run_32_step_12", "n_frames", "5").
param("run_32", "run_32_step_12", "start_gripper", "2.6").
param("run_32", "run_32_step_12", "end_gripper", "1.7").
call("run_32", "run_32_llm_13", "llm_generate", "completed").
depends("run_32", "run_32_llm_13", "run_32_step_12").
call("run_32", "run_32_notify", "notify_user", "completed").
depends("run_32", "run_32_notify", "run_32_llm_13").
spawned_by("run_32", "run_32_notify", "run_32_llm_13").

% === Episode 32 (7 discovered segments) ===
job("run_33").
job_status("run_33", "completed").
call("run_33", "run_33_step_0", "prompt_user", "completed").
call("run_33", "run_33_step_1", "llm_generate", "completed").
depends("run_33", "run_33_step_1", "run_33_step_0").
call("run_33", "run_33_llm_2", "llm_generate", "completed").
depends("run_33", "run_33_llm_2", "run_33_step_1").
call("run_33", "run_33_step_2", "move_x_pos", "completed").
depends("run_33", "run_33_step_2", "run_33_llm_2").
spawned_by("run_33", "run_33_step_2", "run_33_llm_2").
param("run_33", "run_33_step_2", "pos_distance", "0.696").
param("run_33", "run_33_step_2", "delta_gripper", "1.53").
param("run_33", "run_33_step_2", "efficiency", "1.00").
param("run_33", "run_33_step_2", "mean_speed", "0.2330").
param("run_33", "run_33_step_2", "n_frames", "4").
param("run_33", "run_33_step_2", "start_gripper", "6.0").
param("run_33", "run_33_step_2", "end_gripper", "7.6").
call("run_33", "run_33_llm_3", "llm_generate", "completed").
depends("run_33", "run_33_llm_3", "run_33_step_2").
call("run_33", "run_33_step_3", "move_x_pos_4", "completed").
depends("run_33", "run_33_step_3", "run_33_llm_3").
spawned_by("run_33", "run_33_step_3", "run_33_llm_3").
param("run_33", "run_33_step_3", "pos_distance", "1.795").
param("run_33", "run_33_step_3", "delta_gripper", "0.40").
param("run_33", "run_33_step_3", "efficiency", "0.99").
param("run_33", "run_33_step_3", "mean_speed", "0.2591").
param("run_33", "run_33_step_3", "n_frames", "8").
param("run_33", "run_33_step_3", "start_gripper", "7.6").
param("run_33", "run_33_step_3", "end_gripper", "8.0").
call("run_33", "run_33_llm_4", "llm_generate", "completed").
depends("run_33", "run_33_llm_4", "run_33_step_3").
call("run_33", "run_33_step_4", "move_y_neg_6", "completed").
depends("run_33", "run_33_step_4", "run_33_llm_4").
spawned_by("run_33", "run_33_step_4", "run_33_llm_4").
param("run_33", "run_33_step_4", "pos_distance", "1.028").
param("run_33", "run_33_step_4", "delta_gripper", "-1.26").
param("run_33", "run_33_step_4", "efficiency", "0.99").
param("run_33", "run_33_step_4", "mean_speed", "0.3443").
param("run_33", "run_33_step_4", "n_frames", "4").
param("run_33", "run_33_step_4", "start_gripper", "8.0").
param("run_33", "run_33_step_4", "end_gripper", "6.7").
call("run_33", "run_33_llm_5", "llm_generate", "completed").
depends("run_33", "run_33_llm_5", "run_33_step_4").
call("run_33", "run_33_step_5", "adjust_position_8", "completed").
depends("run_33", "run_33_step_5", "run_33_llm_5").
spawned_by("run_33", "run_33_step_5", "run_33_llm_5").
param("run_33", "run_33_step_5", "pos_distance", "0.186").
param("run_33", "run_33_step_5", "delta_gripper", "-0.79").
param("run_33", "run_33_step_5", "efficiency", "0.29").
param("run_33", "run_33_step_5", "mean_speed", "0.2174").
param("run_33", "run_33_step_5", "n_frames", "4").
param("run_33", "run_33_step_5", "start_gripper", "6.7").
param("run_33", "run_33_step_5", "end_gripper", "5.9").
call("run_33", "run_33_llm_6", "llm_generate", "completed").
depends("run_33", "run_33_llm_6", "run_33_step_5").
call("run_33", "run_33_step_6", "close_gripper", "completed").
depends("run_33", "run_33_step_6", "run_33_llm_6").
spawned_by("run_33", "run_33_step_6", "run_33_llm_6").
param("run_33", "run_33_step_6", "pos_distance", "0.261").
param("run_33", "run_33_step_6", "delta_gripper", "-2.07").
param("run_33", "run_33_step_6", "efficiency", "0.54").
param("run_33", "run_33_step_6", "mean_speed", "0.1616").
param("run_33", "run_33_step_6", "n_frames", "4").
param("run_33", "run_33_step_6", "start_gripper", "5.9").
param("run_33", "run_33_step_6", "end_gripper", "3.8").
call("run_33", "run_33_llm_7", "llm_generate", "completed").
depends("run_33", "run_33_llm_7", "run_33_step_6").
call("run_33", "run_33_step_7", "move_z_pos_4", "completed").
depends("run_33", "run_33_step_7", "run_33_llm_7").
spawned_by("run_33", "run_33_step_7", "run_33_llm_7").
param("run_33", "run_33_step_7", "pos_distance", "1.475").
param("run_33", "run_33_step_7", "delta_gripper", "-0.44").
param("run_33", "run_33_step_7", "efficiency", "0.92").
param("run_33", "run_33_step_7", "mean_speed", "0.2286").
param("run_33", "run_33_step_7", "n_frames", "8").
param("run_33", "run_33_step_7", "start_gripper", "3.8").
param("run_33", "run_33_step_7", "end_gripper", "3.4").
call("run_33", "run_33_llm_8", "llm_generate", "completed").
depends("run_33", "run_33_llm_8", "run_33_step_7").
call("run_33", "run_33_step_8", "adjust_position_8", "completed").
depends("run_33", "run_33_step_8", "run_33_llm_8").
spawned_by("run_33", "run_33_step_8", "run_33_llm_8").
param("run_33", "run_33_step_8", "pos_distance", "0.232").
param("run_33", "run_33_step_8", "delta_gripper", "0.00").
param("run_33", "run_33_step_8", "efficiency", "0.09").
param("run_33", "run_33_step_8", "mean_speed", "0.1158").
param("run_33", "run_33_step_8", "n_frames", "24").
param("run_33", "run_33_step_8", "start_gripper", "3.4").
param("run_33", "run_33_step_8", "end_gripper", "3.4").
call("run_33", "run_33_llm_9", "llm_generate", "completed").
depends("run_33", "run_33_llm_9", "run_33_step_8").
call("run_33", "run_33_notify", "notify_user", "completed").
depends("run_33", "run_33_notify", "run_33_llm_9").
spawned_by("run_33", "run_33_notify", "run_33_llm_9").

% === Episode 33 (8 discovered segments) ===
job("run_34").
job_status("run_34", "completed").
call("run_34", "run_34_step_0", "prompt_user", "completed").
call("run_34", "run_34_step_1", "llm_generate", "completed").
depends("run_34", "run_34_step_1", "run_34_step_0").
call("run_34", "run_34_llm_2", "llm_generate", "completed").
depends("run_34", "run_34_llm_2", "run_34_step_1").
call("run_34", "run_34_step_2", "adjust_position", "completed").
depends("run_34", "run_34_step_2", "run_34_llm_2").
spawned_by("run_34", "run_34_step_2", "run_34_llm_2").
param("run_34", "run_34_step_2", "pos_distance", "0.287").
param("run_34", "run_34_step_2", "delta_gripper", "-0.16").
param("run_34", "run_34_step_2", "efficiency", "0.96").
param("run_34", "run_34_step_2", "mean_speed", "0.0995").
param("run_34", "run_34_step_2", "n_frames", "4").
param("run_34", "run_34_step_2", "start_gripper", "4.0").
param("run_34", "run_34_step_2", "end_gripper", "3.8").
call("run_34", "run_34_llm_3", "llm_generate", "completed").
depends("run_34", "run_34_llm_3", "run_34_step_2").
call("run_34", "run_34_step_3", "adjust_position", "completed").
depends("run_34", "run_34_step_3", "run_34_llm_3").
spawned_by("run_34", "run_34_step_3", "run_34_llm_3").
param("run_34", "run_34_step_3", "pos_distance", "0.317").
param("run_34", "run_34_step_3", "delta_gripper", "-0.71").
param("run_34", "run_34_step_3", "efficiency", "0.94").
param("run_34", "run_34_step_3", "mean_speed", "0.1126").
param("run_34", "run_34_step_3", "n_frames", "4").
param("run_34", "run_34_step_3", "start_gripper", "3.8").
param("run_34", "run_34_step_3", "end_gripper", "3.1").
call("run_34", "run_34_llm_4", "llm_generate", "completed").
depends("run_34", "run_34_llm_4", "run_34_step_3").
call("run_34", "run_34_step_4", "adjust_position", "completed").
depends("run_34", "run_34_step_4", "run_34_llm_4").
spawned_by("run_34", "run_34_step_4", "run_34_llm_4").
param("run_34", "run_34_step_4", "pos_distance", "0.485").
param("run_34", "run_34_step_4", "delta_gripper", "-1.39").
param("run_34", "run_34_step_4", "efficiency", "0.92").
param("run_34", "run_34_step_4", "mean_speed", "0.1055").
param("run_34", "run_34_step_4", "n_frames", "6").
param("run_34", "run_34_step_4", "start_gripper", "3.1").
param("run_34", "run_34_step_4", "end_gripper", "1.7").
call("run_34", "run_34_llm_5", "llm_generate", "completed").
depends("run_34", "run_34_llm_5", "run_34_step_4").
call("run_34", "run_34_step_5", "adjust_position", "completed").
depends("run_34", "run_34_step_5", "run_34_llm_5").
spawned_by("run_34", "run_34_step_5", "run_34_llm_5").
param("run_34", "run_34_step_5", "pos_distance", "0.218").
param("run_34", "run_34_step_5", "delta_gripper", "-0.93").
param("run_34", "run_34_step_5", "efficiency", "0.78").
param("run_34", "run_34_step_5", "mean_speed", "0.0696").
param("run_34", "run_34_step_5", "n_frames", "5").
param("run_34", "run_34_step_5", "start_gripper", "1.7").
param("run_34", "run_34_step_5", "end_gripper", "0.8").
call("run_34", "run_34_llm_6", "llm_generate", "completed").
depends("run_34", "run_34_llm_6", "run_34_step_5").
call("run_34", "run_34_step_6", "move_x_pos", "completed").
depends("run_34", "run_34_step_6", "run_34_llm_6").
spawned_by("run_34", "run_34_step_6", "run_34_llm_6").
param("run_34", "run_34_step_6", "pos_distance", "0.308").
param("run_34", "run_34_step_6", "delta_gripper", "1.07").
param("run_34", "run_34_step_6", "efficiency", "0.87").
param("run_34", "run_34_step_6", "mean_speed", "0.1173").
param("run_34", "run_34_step_6", "n_frames", "4").
param("run_34", "run_34_step_6", "start_gripper", "0.8").
param("run_34", "run_34_step_6", "end_gripper", "1.9").
call("run_34", "run_34_llm_7", "llm_generate", "completed").
depends("run_34", "run_34_llm_7", "run_34_step_6").
call("run_34", "run_34_step_7", "adjust_position", "completed").
depends("run_34", "run_34_step_7", "run_34_llm_7").
spawned_by("run_34", "run_34_step_7", "run_34_llm_7").
param("run_34", "run_34_step_7", "pos_distance", "0.474").
param("run_34", "run_34_step_7", "delta_gripper", "-0.70").
param("run_34", "run_34_step_7", "efficiency", "0.86").
param("run_34", "run_34_step_7", "mean_speed", "0.1835").
param("run_34", "run_34_step_7", "n_frames", "4").
param("run_34", "run_34_step_7", "start_gripper", "1.9").
param("run_34", "run_34_step_7", "end_gripper", "1.2").
call("run_34", "run_34_llm_8", "llm_generate", "completed").
depends("run_34", "run_34_llm_8", "run_34_step_7").
call("run_34", "run_34_step_8", "move_x_pos", "completed").
depends("run_34", "run_34_step_8", "run_34_llm_8").
spawned_by("run_34", "run_34_step_8", "run_34_llm_8").
param("run_34", "run_34_step_8", "pos_distance", "0.321").
param("run_34", "run_34_step_8", "delta_gripper", "-0.11").
param("run_34", "run_34_step_8", "efficiency", "0.69").
param("run_34", "run_34_step_8", "mean_speed", "0.1540").
param("run_34", "run_34_step_8", "n_frames", "4").
param("run_34", "run_34_step_8", "start_gripper", "1.2").
param("run_34", "run_34_step_8", "end_gripper", "1.1").
call("run_34", "run_34_llm_9", "llm_generate", "completed").
depends("run_34", "run_34_llm_9", "run_34_step_8").
call("run_34", "run_34_step_9", "adjust_position_16", "completed").
depends("run_34", "run_34_step_9", "run_34_llm_9").
spawned_by("run_34", "run_34_step_9", "run_34_llm_9").
param("run_34", "run_34_step_9", "pos_distance", "0.988").
param("run_34", "run_34_step_9", "delta_gripper", "-1.03").
param("run_34", "run_34_step_9", "efficiency", "0.26").
param("run_34", "run_34_step_9", "mean_speed", "0.1492").
param("run_34", "run_34_step_9", "n_frames", "26").
param("run_34", "run_34_step_9", "start_gripper", "1.1").
param("run_34", "run_34_step_9", "end_gripper", "0.0").
call("run_34", "run_34_llm_10", "llm_generate", "completed").
depends("run_34", "run_34_llm_10", "run_34_step_9").
call("run_34", "run_34_notify", "notify_user", "completed").
depends("run_34", "run_34_notify", "run_34_llm_10").
spawned_by("run_34", "run_34_notify", "run_34_llm_10").

% === Episode 34 (5 discovered segments) ===
job("run_35").
job_status("run_35", "completed").
call("run_35", "run_35_step_0", "prompt_user", "completed").
call("run_35", "run_35_step_1", "llm_generate", "completed").
depends("run_35", "run_35_step_1", "run_35_step_0").
call("run_35", "run_35_llm_2", "llm_generate", "completed").
depends("run_35", "run_35_llm_2", "run_35_step_1").
call("run_35", "run_35_step_2", "move_y_neg", "completed").
depends("run_35", "run_35_step_2", "run_35_llm_2").
spawned_by("run_35", "run_35_step_2", "run_35_llm_2").
param("run_35", "run_35_step_2", "pos_distance", "0.852").
param("run_35", "run_35_step_2", "delta_gripper", "1.55").
param("run_35", "run_35_step_2", "efficiency", "1.00").
param("run_35", "run_35_step_2", "mean_speed", "0.2847").
param("run_35", "run_35_step_2", "n_frames", "4").
param("run_35", "run_35_step_2", "start_gripper", "6.0").
param("run_35", "run_35_step_2", "end_gripper", "7.6").
call("run_35", "run_35_llm_3", "llm_generate", "completed").
depends("run_35", "run_35_llm_3", "run_35_step_2").
call("run_35", "run_35_step_3", "move_y_neg_2", "completed").
depends("run_35", "run_35_step_3", "run_35_llm_3").
spawned_by("run_35", "run_35_step_3", "run_35_llm_3").
param("run_35", "run_35_step_3", "pos_distance", "2.284").
param("run_35", "run_35_step_3", "delta_gripper", "0.37").
param("run_35", "run_35_step_3", "efficiency", "0.93").
param("run_35", "run_35_step_3", "mean_speed", "0.3056").
param("run_35", "run_35_step_3", "n_frames", "9").
param("run_35", "run_35_step_3", "start_gripper", "7.6").
param("run_35", "run_35_step_3", "end_gripper", "8.0").
call("run_35", "run_35_llm_4", "llm_generate", "completed").
depends("run_35", "run_35_llm_4", "run_35_step_3").
call("run_35", "run_35_step_4", "move_y_neg_3", "completed").
depends("run_35", "run_35_step_4", "run_35_llm_4").
spawned_by("run_35", "run_35_step_4", "run_35_llm_4").
param("run_35", "run_35_step_4", "pos_distance", "0.707").
param("run_35", "run_35_step_4", "delta_gripper", "-1.18").
param("run_35", "run_35_step_4", "efficiency", "0.92").
param("run_35", "run_35_step_4", "mean_speed", "0.2575").
param("run_35", "run_35_step_4", "n_frames", "4").
param("run_35", "run_35_step_4", "start_gripper", "8.0").
param("run_35", "run_35_step_4", "end_gripper", "6.8").
call("run_35", "run_35_llm_5", "llm_generate", "completed").
depends("run_35", "run_35_llm_5", "run_35_step_4").
call("run_35", "run_35_step_5", "adjust_position", "completed").
depends("run_35", "run_35_step_5", "run_35_llm_5").
spawned_by("run_35", "run_35_step_5", "run_35_llm_5").
param("run_35", "run_35_step_5", "pos_distance", "0.286").
param("run_35", "run_35_step_5", "delta_gripper", "-3.36").
param("run_35", "run_35_step_5", "efficiency", "0.92").
param("run_35", "run_35_step_5", "mean_speed", "0.1032").
param("run_35", "run_35_step_5", "n_frames", "4").
param("run_35", "run_35_step_5", "start_gripper", "6.8").
param("run_35", "run_35_step_5", "end_gripper", "3.4").
call("run_35", "run_35_llm_6", "llm_generate", "completed").
depends("run_35", "run_35_llm_6", "run_35_step_5").
call("run_35", "run_35_step_6", "move_z_pos_5", "completed").
depends("run_35", "run_35_step_6", "run_35_llm_6").
spawned_by("run_35", "run_35_step_6", "run_35_llm_6").
param("run_35", "run_35_step_6", "pos_distance", "2.403").
param("run_35", "run_35_step_6", "delta_gripper", "-0.03").
param("run_35", "run_35_step_6", "efficiency", "0.45").
param("run_35", "run_35_step_6", "mean_speed", "0.1675").
param("run_35", "run_35_step_6", "n_frames", "33").
param("run_35", "run_35_step_6", "start_gripper", "3.4").
param("run_35", "run_35_step_6", "end_gripper", "3.4").
call("run_35", "run_35_llm_7", "llm_generate", "completed").
depends("run_35", "run_35_llm_7", "run_35_step_6").
call("run_35", "run_35_notify", "notify_user", "completed").
depends("run_35", "run_35_notify", "run_35_llm_7").
spawned_by("run_35", "run_35_notify", "run_35_llm_7").

% === Episode 35 (3 discovered segments) ===
job("run_36").
job_status("run_36", "completed").
call("run_36", "run_36_step_0", "prompt_user", "completed").
call("run_36", "run_36_step_1", "llm_generate", "completed").
depends("run_36", "run_36_step_1", "run_36_step_0").
call("run_36", "run_36_llm_2", "llm_generate", "completed").
depends("run_36", "run_36_llm_2", "run_36_step_1").
call("run_36", "run_36_step_2", "adjust_position_5", "completed").
depends("run_36", "run_36_step_2", "run_36_llm_2").
spawned_by("run_36", "run_36_step_2", "run_36_llm_2").
param("run_36", "run_36_step_2", "pos_distance", "0.015").
param("run_36", "run_36_step_2", "delta_gripper", "1.71").
param("run_36", "run_36_step_2", "efficiency", "1.00").
param("run_36", "run_36_step_2", "mean_speed", "0.0050").
param("run_36", "run_36_step_2", "n_frames", "4").
param("run_36", "run_36_step_2", "start_gripper", "6.0").
param("run_36", "run_36_step_2", "end_gripper", "7.7").
call("run_36", "run_36_llm_3", "llm_generate", "completed").
depends("run_36", "run_36_llm_3", "run_36_step_2").
call("run_36", "run_36_step_3", "adjust_position_17", "completed").
depends("run_36", "run_36_step_3", "run_36_llm_3").
spawned_by("run_36", "run_36_step_3", "run_36_llm_3").
param("run_36", "run_36_step_3", "pos_distance", "0.873").
param("run_36", "run_36_step_3", "delta_gripper", "0.24").
param("run_36", "run_36_step_3", "efficiency", "0.82").
param("run_36", "run_36_step_3", "mean_speed", "0.0313").
param("run_36", "run_36_step_3", "n_frames", "35").
param("run_36", "run_36_step_3", "start_gripper", "7.7").
param("run_36", "run_36_step_3", "end_gripper", "8.0").
call("run_36", "run_36_llm_4", "llm_generate", "completed").
depends("run_36", "run_36_llm_4", "run_36_step_3").
call("run_36", "run_36_step_4", "adjust_position_7", "completed").
depends("run_36", "run_36_step_4", "run_36_llm_4").
spawned_by("run_36", "run_36_step_4", "run_36_llm_4").
param("run_36", "run_36_step_4", "pos_distance", "1.576").
param("run_36", "run_36_step_4", "delta_gripper", "0.00").
param("run_36", "run_36_step_4", "efficiency", "0.89").
param("run_36", "run_36_step_4", "mean_speed", "0.1477").
param("run_36", "run_36_step_4", "n_frames", "13").
param("run_36", "run_36_step_4", "start_gripper", "8.0").
param("run_36", "run_36_step_4", "end_gripper", "8.0").
call("run_36", "run_36_llm_5", "llm_generate", "completed").
depends("run_36", "run_36_llm_5", "run_36_step_4").
call("run_36", "run_36_notify", "notify_user", "completed").
depends("run_36", "run_36_notify", "run_36_llm_5").
spawned_by("run_36", "run_36_notify", "run_36_llm_5").

% === Episode 36 (9 discovered segments) ===
job("run_37").
job_status("run_37", "completed").
call("run_37", "run_37_step_0", "prompt_user", "completed").
call("run_37", "run_37_step_1", "llm_generate", "completed").
depends("run_37", "run_37_step_1", "run_37_step_0").
call("run_37", "run_37_llm_2", "llm_generate", "completed").
depends("run_37", "run_37_llm_2", "run_37_step_1").
call("run_37", "run_37_step_2", "adjust_position_18", "completed").
depends("run_37", "run_37_step_2", "run_37_llm_2").
spawned_by("run_37", "run_37_step_2", "run_37_llm_2").
param("run_37", "run_37_step_2", "pos_distance", "1.349").
param("run_37", "run_37_step_2", "delta_gripper", "-0.01").
param("run_37", "run_37_step_2", "efficiency", "0.87").
param("run_37", "run_37_step_2", "mean_speed", "0.1297").
param("run_37", "run_37_step_2", "n_frames", "13").
param("run_37", "run_37_step_2", "start_gripper", "4.0").
param("run_37", "run_37_step_2", "end_gripper", "4.0").
call("run_37", "run_37_llm_3", "llm_generate", "completed").
depends("run_37", "run_37_llm_3", "run_37_step_2").
call("run_37", "run_37_step_3", "adjust_position", "completed").
depends("run_37", "run_37_step_3", "run_37_llm_3").
spawned_by("run_37", "run_37_step_3", "run_37_llm_3").
param("run_37", "run_37_step_3", "pos_distance", "0.276").
param("run_37", "run_37_step_3", "delta_gripper", "-1.60").
param("run_37", "run_37_step_3", "efficiency", "0.91").
param("run_37", "run_37_step_3", "mean_speed", "0.1007").
param("run_37", "run_37_step_3", "n_frames", "4").
param("run_37", "run_37_step_3", "start_gripper", "4.0").
param("run_37", "run_37_step_3", "end_gripper", "2.4").
call("run_37", "run_37_llm_4", "llm_generate", "completed").
depends("run_37", "run_37_llm_4", "run_37_step_3").
call("run_37", "run_37_step_4", "adjust_position", "completed").
depends("run_37", "run_37_step_4", "run_37_llm_4").
spawned_by("run_37", "run_37_step_4", "run_37_llm_4").
param("run_37", "run_37_step_4", "pos_distance", "0.290").
param("run_37", "run_37_step_4", "delta_gripper", "-2.15").
param("run_37", "run_37_step_4", "efficiency", "0.99").
param("run_37", "run_37_step_4", "mean_speed", "0.0973").
param("run_37", "run_37_step_4", "n_frames", "4").
param("run_37", "run_37_step_4", "start_gripper", "2.4").
param("run_37", "run_37_step_4", "end_gripper", "0.3").
call("run_37", "run_37_llm_5", "llm_generate", "completed").
depends("run_37", "run_37_llm_5", "run_37_step_4").
call("run_37", "run_37_step_5", "move_x_pos_2", "completed").
depends("run_37", "run_37_step_5", "run_37_llm_5").
spawned_by("run_37", "run_37_step_5", "run_37_llm_5").
param("run_37", "run_37_step_5", "pos_distance", "1.955").
param("run_37", "run_37_step_5", "delta_gripper", "-0.01").
param("run_37", "run_37_step_5", "efficiency", "0.95").
param("run_37", "run_37_step_5", "mean_speed", "0.2570").
param("run_37", "run_37_step_5", "n_frames", "9").
param("run_37", "run_37_step_5", "start_gripper", "0.3").
param("run_37", "run_37_step_5", "end_gripper", "0.2").
call("run_37", "run_37_llm_6", "llm_generate", "completed").
depends("run_37", "run_37_llm_6", "run_37_step_5").
call("run_37", "run_37_step_6", "adjust_position", "completed").
depends("run_37", "run_37_step_6", "run_37_llm_6").
spawned_by("run_37", "run_37_step_6", "run_37_llm_6").
param("run_37", "run_37_step_6", "pos_distance", "0.139").
param("run_37", "run_37_step_6", "delta_gripper", "0.00").
param("run_37", "run_37_step_6", "efficiency", "0.82").
param("run_37", "run_37_step_6", "mean_speed", "0.0422").
param("run_37", "run_37_step_6", "n_frames", "5").
param("run_37", "run_37_step_6", "start_gripper", "0.2").
param("run_37", "run_37_step_6", "end_gripper", "0.2").
call("run_37", "run_37_llm_7", "llm_generate", "completed").
depends("run_37", "run_37_llm_7", "run_37_step_6").
call("run_37", "run_37_step_7", "adjust_position_2", "completed").
depends("run_37", "run_37_step_7", "run_37_llm_7").
spawned_by("run_37", "run_37_step_7", "run_37_llm_7").
param("run_37", "run_37_step_7", "pos_distance", "0.015").
param("run_37", "run_37_step_7", "delta_gripper", "0.69").
param("run_37", "run_37_step_7", "efficiency", "0.65").
param("run_37", "run_37_step_7", "mean_speed", "0.0079").
param("run_37", "run_37_step_7", "n_frames", "4").
param("run_37", "run_37_step_7", "start_gripper", "0.2").
param("run_37", "run_37_step_7", "end_gripper", "0.9").
call("run_37", "run_37_llm_8", "llm_generate", "completed").
depends("run_37", "run_37_llm_8", "run_37_step_7").
call("run_37", "run_37_step_8", "adjust_position_5", "completed").
depends("run_37", "run_37_step_8", "run_37_llm_8").
spawned_by("run_37", "run_37_step_8", "run_37_llm_8").
param("run_37", "run_37_step_8", "pos_distance", "0.056").
param("run_37", "run_37_step_8", "delta_gripper", "2.25").
param("run_37", "run_37_step_8", "efficiency", "0.93").
param("run_37", "run_37_step_8", "mean_speed", "0.0200").
param("run_37", "run_37_step_8", "n_frames", "4").
param("run_37", "run_37_step_8", "start_gripper", "0.9").
param("run_37", "run_37_step_8", "end_gripper", "3.2").
call("run_37", "run_37_llm_9", "llm_generate", "completed").
depends("run_37", "run_37_llm_9", "run_37_step_8").
call("run_37", "run_37_step_9", "adjust_position_2", "completed").
depends("run_37", "run_37_step_9", "run_37_llm_9").
spawned_by("run_37", "run_37_step_9", "run_37_llm_9").
param("run_37", "run_37_step_9", "pos_distance", "0.185").
param("run_37", "run_37_step_9", "delta_gripper", "0.77").
param("run_37", "run_37_step_9", "efficiency", "1.00").
param("run_37", "run_37_step_9", "mean_speed", "0.0618").
param("run_37", "run_37_step_9", "n_frames", "4").
param("run_37", "run_37_step_9", "start_gripper", "3.2").
param("run_37", "run_37_step_9", "end_gripper", "4.0").
call("run_37", "run_37_llm_10", "llm_generate", "completed").
depends("run_37", "run_37_llm_10", "run_37_step_9").
call("run_37", "run_37_step_10", "adjust_position", "completed").
depends("run_37", "run_37_step_10", "run_37_llm_10").
spawned_by("run_37", "run_37_step_10", "run_37_llm_10").
param("run_37", "run_37_step_10", "pos_distance", "0.822").
param("run_37", "run_37_step_10", "delta_gripper", "0.01").
param("run_37", "run_37_step_10", "efficiency", "0.99").
param("run_37", "run_37_step_10", "mean_speed", "0.0832").
param("run_37", "run_37_step_10", "n_frames", "11").
param("run_37", "run_37_step_10", "start_gripper", "4.0").
param("run_37", "run_37_step_10", "end_gripper", "4.0").
call("run_37", "run_37_llm_11", "llm_generate", "completed").
depends("run_37", "run_37_llm_11", "run_37_step_10").
call("run_37", "run_37_notify", "notify_user", "completed").
depends("run_37", "run_37_notify", "run_37_llm_11").
spawned_by("run_37", "run_37_notify", "run_37_llm_11").

% === Episode 37 (4 discovered segments) ===
job("run_38").
job_status("run_38", "completed").
call("run_38", "run_38_step_0", "prompt_user", "completed").
call("run_38", "run_38_step_1", "llm_generate", "completed").
depends("run_38", "run_38_step_1", "run_38_step_0").
call("run_38", "run_38_llm_2", "llm_generate", "completed").
depends("run_38", "run_38_llm_2", "run_38_step_1").
call("run_38", "run_38_step_2", "move_y_neg_2", "completed").
depends("run_38", "run_38_step_2", "run_38_llm_2").
spawned_by("run_38", "run_38_step_2", "run_38_llm_2").
param("run_38", "run_38_step_2", "pos_distance", "3.134").
param("run_38", "run_38_step_2", "delta_gripper", "1.93").
param("run_38", "run_38_step_2", "efficiency", "0.96").
param("run_38", "run_38_step_2", "mean_speed", "0.2333").
param("run_38", "run_38_step_2", "n_frames", "15").
param("run_38", "run_38_step_2", "start_gripper", "6.0").
param("run_38", "run_38_step_2", "end_gripper", "8.0").
call("run_38", "run_38_llm_3", "llm_generate", "completed").
depends("run_38", "run_38_llm_3", "run_38_step_2").
call("run_38", "run_38_step_3", "adjust_position", "completed").
depends("run_38", "run_38_step_3", "run_38_llm_3").
spawned_by("run_38", "run_38_step_3", "run_38_llm_3").
param("run_38", "run_38_step_3", "pos_distance", "0.188").
param("run_38", "run_38_step_3", "delta_gripper", "-0.34").
param("run_38", "run_38_step_3", "efficiency", "0.96").
param("run_38", "run_38_step_3", "mean_speed", "0.0656").
param("run_38", "run_38_step_3", "n_frames", "4").
param("run_38", "run_38_step_3", "start_gripper", "8.0").
param("run_38", "run_38_step_3", "end_gripper", "7.6").
call("run_38", "run_38_llm_4", "llm_generate", "completed").
depends("run_38", "run_38_llm_4", "run_38_step_3").
call("run_38", "run_38_step_4", "close_gripper_3", "completed").
depends("run_38", "run_38_step_4", "run_38_llm_4").
spawned_by("run_38", "run_38_step_4", "run_38_llm_4").
param("run_38", "run_38_step_4", "pos_distance", "0.078").
param("run_38", "run_38_step_4", "delta_gripper", "-3.71").
param("run_38", "run_38_step_4", "efficiency", "0.74").
param("run_38", "run_38_step_4", "mean_speed", "0.0353").
param("run_38", "run_38_step_4", "n_frames", "4").
param("run_38", "run_38_step_4", "start_gripper", "7.6").
param("run_38", "run_38_step_4", "end_gripper", "3.9").
call("run_38", "run_38_llm_5", "llm_generate", "completed").
depends("run_38", "run_38_llm_5", "run_38_step_4").
call("run_38", "run_38_step_5", "adjust_position_4", "completed").
depends("run_38", "run_38_step_5", "run_38_llm_5").
spawned_by("run_38", "run_38_step_5", "run_38_llm_5").
param("run_38", "run_38_step_5", "pos_distance", "1.571").
param("run_38", "run_38_step_5", "delta_gripper", "-0.55").
param("run_38", "run_38_step_5", "efficiency", "0.83").
param("run_38", "run_38_step_5", "mean_speed", "0.0651").
param("run_38", "run_38_step_5", "n_frames", "30").
param("run_38", "run_38_step_5", "start_gripper", "3.9").
param("run_38", "run_38_step_5", "end_gripper", "3.4").
call("run_38", "run_38_llm_6", "llm_generate", "completed").
depends("run_38", "run_38_llm_6", "run_38_step_5").
call("run_38", "run_38_notify", "notify_user", "completed").
depends("run_38", "run_38_notify", "run_38_llm_6").
spawned_by("run_38", "run_38_notify", "run_38_llm_6").

% === Episode 38 (8 discovered segments) ===
job("run_39").
job_status("run_39", "completed").
call("run_39", "run_39_step_0", "prompt_user", "completed").
call("run_39", "run_39_step_1", "llm_generate", "completed").
depends("run_39", "run_39_step_1", "run_39_step_0").
call("run_39", "run_39_llm_2", "llm_generate", "completed").
depends("run_39", "run_39_llm_2", "run_39_step_1").
call("run_39", "run_39_step_2", "move_x_pos", "completed").
depends("run_39", "run_39_step_2", "run_39_llm_2").
spawned_by("run_39", "run_39_step_2", "run_39_llm_2").
param("run_39", "run_39_step_2", "pos_distance", "0.966").
param("run_39", "run_39_step_2", "delta_gripper", "0.01").
param("run_39", "run_39_step_2", "efficiency", "0.74").
param("run_39", "run_39_step_2", "mean_speed", "0.1624").
param("run_39", "run_39_step_2", "n_frames", "9").
param("run_39", "run_39_step_2", "start_gripper", "4.0").
param("run_39", "run_39_step_2", "end_gripper", "4.0").
call("run_39", "run_39_llm_3", "llm_generate", "completed").
depends("run_39", "run_39_llm_3", "run_39_step_2").
call("run_39", "run_39_step_3", "adjust_position", "completed").
depends("run_39", "run_39_step_3", "run_39_llm_3").
spawned_by("run_39", "run_39_step_3", "run_39_llm_3").
param("run_39", "run_39_step_3", "pos_distance", "0.270").
param("run_39", "run_39_step_3", "delta_gripper", "-0.97").
param("run_39", "run_39_step_3", "efficiency", "0.89").
param("run_39", "run_39_step_3", "mean_speed", "0.1017").
param("run_39", "run_39_step_3", "n_frames", "4").
param("run_39", "run_39_step_3", "start_gripper", "4.0").
param("run_39", "run_39_step_3", "end_gripper", "3.0").
call("run_39", "run_39_llm_4", "llm_generate", "completed").
depends("run_39", "run_39_llm_4", "run_39_step_3").
call("run_39", "run_39_step_4", "adjust_position", "completed").
depends("run_39", "run_39_step_4", "run_39_llm_4").
spawned_by("run_39", "run_39_step_4", "run_39_llm_4").
param("run_39", "run_39_step_4", "pos_distance", "0.106").
param("run_39", "run_39_step_4", "delta_gripper", "-2.80").
param("run_39", "run_39_step_4", "efficiency", "0.85").
param("run_39", "run_39_step_4", "mean_speed", "0.0417").
param("run_39", "run_39_step_4", "n_frames", "4").
param("run_39", "run_39_step_4", "start_gripper", "3.0").
param("run_39", "run_39_step_4", "end_gripper", "0.2").
call("run_39", "run_39_llm_5", "llm_generate", "completed").
depends("run_39", "run_39_llm_5", "run_39_step_4").
call("run_39", "run_39_step_5", "move_x_pos_2", "completed").
depends("run_39", "run_39_step_5", "run_39_llm_5").
spawned_by("run_39", "run_39_step_5", "run_39_llm_5").
param("run_39", "run_39_step_5", "pos_distance", "2.276").
param("run_39", "run_39_step_5", "delta_gripper", "-0.02").
param("run_39", "run_39_step_5", "efficiency", "0.94").
param("run_39", "run_39_step_5", "mean_speed", "0.3026").
param("run_39", "run_39_step_5", "n_frames", "9").
param("run_39", "run_39_step_5", "start_gripper", "0.2").
param("run_39", "run_39_step_5", "end_gripper", "0.2").
call("run_39", "run_39_llm_6", "llm_generate", "completed").
depends("run_39", "run_39_llm_6", "run_39_step_5").
call("run_39", "run_39_step_6", "move_x_pos", "completed").
depends("run_39", "run_39_step_6", "run_39_llm_6").
spawned_by("run_39", "run_39_step_6", "run_39_llm_6").
param("run_39", "run_39_step_6", "pos_distance", "0.497").
param("run_39", "run_39_step_6", "delta_gripper", "0.02").
param("run_39", "run_39_step_6", "efficiency", "0.99").
param("run_39", "run_39_step_6", "mean_speed", "0.1680").
param("run_39", "run_39_step_6", "n_frames", "4").
param("run_39", "run_39_step_6", "start_gripper", "0.2").
param("run_39", "run_39_step_6", "end_gripper", "0.2").
call("run_39", "run_39_llm_7", "llm_generate", "completed").
depends("run_39", "run_39_llm_7", "run_39_step_6").
call("run_39", "run_39_step_7", "adjust_position_2", "completed").
depends("run_39", "run_39_step_7", "run_39_llm_7").
spawned_by("run_39", "run_39_step_7", "run_39_llm_7").
param("run_39", "run_39_step_7", "pos_distance", "0.076").
param("run_39", "run_39_step_7", "delta_gripper", "2.22").
param("run_39", "run_39_step_7", "efficiency", "0.78").
param("run_39", "run_39_step_7", "mean_speed", "0.0327").
param("run_39", "run_39_step_7", "n_frames", "4").
param("run_39", "run_39_step_7", "start_gripper", "0.2").
param("run_39", "run_39_step_7", "end_gripper", "2.4").
call("run_39", "run_39_llm_8", "llm_generate", "completed").
depends("run_39", "run_39_llm_8", "run_39_step_7").
call("run_39", "run_39_step_8", "adjust_position_5", "completed").
depends("run_39", "run_39_step_8", "run_39_llm_8").
spawned_by("run_39", "run_39_step_8", "run_39_llm_8").
param("run_39", "run_39_step_8", "pos_distance", "0.058").
param("run_39", "run_39_step_8", "delta_gripper", "1.51").
param("run_39", "run_39_step_8", "efficiency", "1.00").
param("run_39", "run_39_step_8", "mean_speed", "0.0196").
param("run_39", "run_39_step_8", "n_frames", "4").
param("run_39", "run_39_step_8", "start_gripper", "2.4").
param("run_39", "run_39_step_8", "end_gripper", "4.0").
call("run_39", "run_39_llm_9", "llm_generate", "completed").
depends("run_39", "run_39_llm_9", "run_39_step_8").
call("run_39", "run_39_step_9", "adjust_position", "completed").
depends("run_39", "run_39_step_9", "run_39_llm_9").
spawned_by("run_39", "run_39_step_9", "run_39_llm_9").
param("run_39", "run_39_step_9", "pos_distance", "0.580").
param("run_39", "run_39_step_9", "delta_gripper", "0.02").
param("run_39", "run_39_step_9", "efficiency", "0.99").
param("run_39", "run_39_step_9", "mean_speed", "0.0324").
param("run_39", "run_39_step_9", "n_frames", "19").
param("run_39", "run_39_step_9", "start_gripper", "4.0").
param("run_39", "run_39_step_9", "end_gripper", "4.0").
call("run_39", "run_39_llm_10", "llm_generate", "completed").
depends("run_39", "run_39_llm_10", "run_39_step_9").
call("run_39", "run_39_notify", "notify_user", "completed").
depends("run_39", "run_39_notify", "run_39_llm_10").
spawned_by("run_39", "run_39_notify", "run_39_llm_10").

% === Episode 39 (6 discovered segments) ===
job("run_40").
job_status("run_40", "completed").
call("run_40", "run_40_step_0", "prompt_user", "completed").
call("run_40", "run_40_step_1", "llm_generate", "completed").
depends("run_40", "run_40_step_1", "run_40_step_0").
call("run_40", "run_40_llm_2", "llm_generate", "completed").
depends("run_40", "run_40_llm_2", "run_40_step_1").
call("run_40", "run_40_step_2", "move_y_neg", "completed").
depends("run_40", "run_40_step_2", "run_40_llm_2").
spawned_by("run_40", "run_40_step_2", "run_40_llm_2").
param("run_40", "run_40_step_2", "pos_distance", "0.523").
param("run_40", "run_40_step_2", "delta_gripper", "1.92").
param("run_40", "run_40_step_2", "efficiency", "0.98").
param("run_40", "run_40_step_2", "mean_speed", "0.1334").
param("run_40", "run_40_step_2", "n_frames", "5").
param("run_40", "run_40_step_2", "start_gripper", "6.0").
param("run_40", "run_40_step_2", "end_gripper", "7.9").
call("run_40", "run_40_llm_3", "llm_generate", "completed").
depends("run_40", "run_40_llm_3", "run_40_step_2").
call("run_40", "run_40_step_3", "move_x_pos_4", "completed").
depends("run_40", "run_40_step_3", "run_40_llm_3").
spawned_by("run_40", "run_40_step_3", "run_40_llm_3").
param("run_40", "run_40_step_3", "pos_distance", "1.310").
param("run_40", "run_40_step_3", "delta_gripper", "0.01").
param("run_40", "run_40_step_3", "efficiency", "0.99").
param("run_40", "run_40_step_3", "mean_speed", "0.3319").
param("run_40", "run_40_step_3", "n_frames", "5").
param("run_40", "run_40_step_3", "start_gripper", "7.9").
param("run_40", "run_40_step_3", "end_gripper", "8.0").
call("run_40", "run_40_llm_4", "llm_generate", "completed").
depends("run_40", "run_40_llm_4", "run_40_step_3").
call("run_40", "run_40_step_4", "move_x_pos", "completed").
depends("run_40", "run_40_step_4", "run_40_llm_4").
spawned_by("run_40", "run_40_step_4", "run_40_llm_4").
param("run_40", "run_40_step_4", "pos_distance", "0.767").
param("run_40", "run_40_step_4", "delta_gripper", "0.00").
param("run_40", "run_40_step_4", "efficiency", "0.99").
param("run_40", "run_40_step_4", "mean_speed", "0.2583").
param("run_40", "run_40_step_4", "n_frames", "4").
param("run_40", "run_40_step_4", "start_gripper", "8.0").
param("run_40", "run_40_step_4", "end_gripper", "8.0").
call("run_40", "run_40_llm_5", "llm_generate", "completed").
depends("run_40", "run_40_llm_5", "run_40_step_4").
call("run_40", "run_40_step_5", "close_gripper_4", "completed").
depends("run_40", "run_40_step_5", "run_40_llm_5").
spawned_by("run_40", "run_40_step_5", "run_40_llm_5").
param("run_40", "run_40_step_5", "pos_distance", "0.450").
param("run_40", "run_40_step_5", "delta_gripper", "-3.79").
param("run_40", "run_40_step_5", "efficiency", "0.99").
param("run_40", "run_40_step_5", "mean_speed", "0.1508").
param("run_40", "run_40_step_5", "n_frames", "4").
param("run_40", "run_40_step_5", "start_gripper", "8.0").
param("run_40", "run_40_step_5", "end_gripper", "4.2").
call("run_40", "run_40_llm_6", "llm_generate", "completed").
depends("run_40", "run_40_llm_6", "run_40_step_5").
call("run_40", "run_40_step_6", "adjust_position", "completed").
depends("run_40", "run_40_step_6", "run_40_llm_6").
spawned_by("run_40", "run_40_step_6", "run_40_llm_6").
param("run_40", "run_40_step_6", "pos_distance", "0.352").
param("run_40", "run_40_step_6", "delta_gripper", "-0.68").
param("run_40", "run_40_step_6", "efficiency", "0.93").
param("run_40", "run_40_step_6", "mean_speed", "0.1257").
param("run_40", "run_40_step_6", "n_frames", "4").
param("run_40", "run_40_step_6", "start_gripper", "4.2").
param("run_40", "run_40_step_6", "end_gripper", "3.5").
call("run_40", "run_40_llm_7", "llm_generate", "completed").
depends("run_40", "run_40_llm_7", "run_40_step_6").
call("run_40", "run_40_step_7", "adjust_position_4", "completed").
depends("run_40", "run_40_step_7", "run_40_llm_7").
spawned_by("run_40", "run_40_step_7", "run_40_llm_7").
param("run_40", "run_40_step_7", "pos_distance", "1.278").
param("run_40", "run_40_step_7", "delta_gripper", "-0.18").
param("run_40", "run_40_step_7", "efficiency", "0.78").
param("run_40", "run_40_step_7", "mean_speed", "0.0510").
param("run_40", "run_40_step_7", "n_frames", "33").
param("run_40", "run_40_step_7", "start_gripper", "3.5").
param("run_40", "run_40_step_7", "end_gripper", "3.3").
call("run_40", "run_40_llm_8", "llm_generate", "completed").
depends("run_40", "run_40_llm_8", "run_40_step_7").
call("run_40", "run_40_notify", "notify_user", "completed").
depends("run_40", "run_40_notify", "run_40_llm_8").
spawned_by("run_40", "run_40_notify", "run_40_llm_8").

% === Episode 40 (6 discovered segments) ===
job("run_41").
job_status("run_41", "completed").
call("run_41", "run_41_step_0", "prompt_user", "completed").
call("run_41", "run_41_step_1", "llm_generate", "completed").
depends("run_41", "run_41_step_1", "run_41_step_0").
call("run_41", "run_41_llm_2", "llm_generate", "completed").
depends("run_41", "run_41_llm_2", "run_41_step_1").
call("run_41", "run_41_step_2", "move_x_pos", "completed").
depends("run_41", "run_41_step_2", "run_41_llm_2").
spawned_by("run_41", "run_41_step_2", "run_41_llm_2").
param("run_41", "run_41_step_2", "pos_distance", "0.386").
param("run_41", "run_41_step_2", "delta_gripper", "1.51").
param("run_41", "run_41_step_2", "efficiency", "1.00").
param("run_41", "run_41_step_2", "mean_speed", "0.1290").
param("run_41", "run_41_step_2", "n_frames", "4").
param("run_41", "run_41_step_2", "start_gripper", "6.0").
param("run_41", "run_41_step_2", "end_gripper", "7.6").
call("run_41", "run_41_llm_3", "llm_generate", "completed").
depends("run_41", "run_41_llm_3", "run_41_step_2").
call("run_41", "run_41_step_3", "move_y_neg_2", "completed").
depends("run_41", "run_41_step_3", "run_41_llm_3").
spawned_by("run_41", "run_41_step_3", "run_41_llm_3").
param("run_41", "run_41_step_3", "pos_distance", "2.285").
param("run_41", "run_41_step_3", "delta_gripper", "0.41").
param("run_41", "run_41_step_3", "efficiency", "0.93").
param("run_41", "run_41_step_3", "mean_speed", "0.2462").
param("run_41", "run_41_step_3", "n_frames", "11").
param("run_41", "run_41_step_3", "start_gripper", "7.6").
param("run_41", "run_41_step_3", "end_gripper", "8.0").
call("run_41", "run_41_llm_4", "llm_generate", "completed").
depends("run_41", "run_41_llm_4", "run_41_step_3").
call("run_41", "run_41_step_4", "adjust_position", "completed").
depends("run_41", "run_41_step_4", "run_41_llm_4").
spawned_by("run_41", "run_41_step_4", "run_41_llm_4").
param("run_41", "run_41_step_4", "pos_distance", "0.531").
param("run_41", "run_41_step_4", "delta_gripper", "-1.01").
param("run_41", "run_41_step_4", "efficiency", "1.00").
param("run_41", "run_41_step_4", "mean_speed", "0.1773").
param("run_41", "run_41_step_4", "n_frames", "4").
param("run_41", "run_41_step_4", "start_gripper", "8.0").
param("run_41", "run_41_step_4", "end_gripper", "6.9").
call("run_41", "run_41_llm_5", "llm_generate", "completed").
depends("run_41", "run_41_llm_5", "run_41_step_4").
call("run_41", "run_41_step_5", "adjust_position", "completed").
depends("run_41", "run_41_step_5", "run_41_llm_5").
spawned_by("run_41", "run_41_step_5", "run_41_llm_5").
param("run_41", "run_41_step_5", "pos_distance", "0.081").
param("run_41", "run_41_step_5", "delta_gripper", "-3.63").
param("run_41", "run_41_step_5", "efficiency", "0.88").
param("run_41", "run_41_step_5", "mean_speed", "0.0307").
param("run_41", "run_41_step_5", "n_frames", "4").
param("run_41", "run_41_step_5", "start_gripper", "6.9").
param("run_41", "run_41_step_5", "end_gripper", "3.3").
call("run_41", "run_41_llm_6", "llm_generate", "completed").
depends("run_41", "run_41_llm_6", "run_41_step_5").
call("run_41", "run_41_step_6", "adjust_position_4", "completed").
depends("run_41", "run_41_step_6", "run_41_llm_6").
spawned_by("run_41", "run_41_step_6", "run_41_llm_6").
param("run_41", "run_41_step_6", "pos_distance", "1.702").
param("run_41", "run_41_step_6", "delta_gripper", "-0.04").
param("run_41", "run_41_step_6", "efficiency", "0.93").
param("run_41", "run_41_step_6", "mean_speed", "0.0704").
param("run_41", "run_41_step_6", "n_frames", "27").
param("run_41", "run_41_step_6", "start_gripper", "3.3").
param("run_41", "run_41_step_6", "end_gripper", "3.3").
call("run_41", "run_41_llm_7", "llm_generate", "completed").
depends("run_41", "run_41_llm_7", "run_41_step_6").
call("run_41", "run_41_step_7", "adjust_position", "completed").
depends("run_41", "run_41_step_7", "run_41_llm_7").
spawned_by("run_41", "run_41_step_7", "run_41_llm_7").
param("run_41", "run_41_step_7", "pos_distance", "0.207").
param("run_41", "run_41_step_7", "delta_gripper", "-0.14").
param("run_41", "run_41_step_7", "efficiency", "0.85").
param("run_41", "run_41_step_7", "mean_speed", "0.0608").
param("run_41", "run_41_step_7", "n_frames", "5").
param("run_41", "run_41_step_7", "start_gripper", "3.3").
param("run_41", "run_41_step_7", "end_gripper", "3.1").
call("run_41", "run_41_llm_8", "llm_generate", "completed").
depends("run_41", "run_41_llm_8", "run_41_step_7").
call("run_41", "run_41_notify", "notify_user", "completed").
depends("run_41", "run_41_notify", "run_41_llm_8").
spawned_by("run_41", "run_41_notify", "run_41_llm_8").

% === Episode 41 (7 discovered segments) ===
job("run_42").
job_status("run_42", "completed").
call("run_42", "run_42_step_0", "prompt_user", "completed").
call("run_42", "run_42_step_1", "llm_generate", "completed").
depends("run_42", "run_42_step_1", "run_42_step_0").
call("run_42", "run_42_llm_2", "llm_generate", "completed").
depends("run_42", "run_42_llm_2", "run_42_step_1").
call("run_42", "run_42_step_2", "adjust_position_2", "completed").
depends("run_42", "run_42_step_2", "run_42_llm_2").
spawned_by("run_42", "run_42_step_2", "run_42_llm_2").
param("run_42", "run_42_step_2", "pos_distance", "0.179").
param("run_42", "run_42_step_2", "delta_gripper", "1.72").
param("run_42", "run_42_step_2", "efficiency", "0.98").
param("run_42", "run_42_step_2", "mean_speed", "0.0611").
param("run_42", "run_42_step_2", "n_frames", "4").
param("run_42", "run_42_step_2", "start_gripper", "6.0").
param("run_42", "run_42_step_2", "end_gripper", "7.8").
call("run_42", "run_42_llm_3", "llm_generate", "completed").
depends("run_42", "run_42_llm_3", "run_42_step_2").
call("run_42", "run_42_step_3", "move_x_pos_4", "completed").
depends("run_42", "run_42_step_3", "run_42_llm_3").
spawned_by("run_42", "run_42_step_3", "run_42_llm_3").
param("run_42", "run_42_step_3", "pos_distance", "1.152").
param("run_42", "run_42_step_3", "delta_gripper", "0.20").
param("run_42", "run_42_step_3", "efficiency", "0.91").
param("run_42", "run_42_step_3", "mean_speed", "0.2119").
param("run_42", "run_42_step_3", "n_frames", "7").
param("run_42", "run_42_step_3", "start_gripper", "7.8").
param("run_42", "run_42_step_3", "end_gripper", "8.0").
call("run_42", "run_42_llm_4", "llm_generate", "completed").
depends("run_42", "run_42_llm_4", "run_42_step_3").
call("run_42", "run_42_step_4", "adjust_position", "completed").
depends("run_42", "run_42_step_4", "run_42_llm_4").
spawned_by("run_42", "run_42_step_4", "run_42_llm_4").
param("run_42", "run_42_step_4", "pos_distance", "0.187").
param("run_42", "run_42_step_4", "delta_gripper", "-0.95").
param("run_42", "run_42_step_4", "efficiency", "0.92").
param("run_42", "run_42_step_4", "mean_speed", "0.0680").
param("run_42", "run_42_step_4", "n_frames", "4").
param("run_42", "run_42_step_4", "start_gripper", "8.0").
param("run_42", "run_42_step_4", "end_gripper", "7.0").
call("run_42", "run_42_llm_5", "llm_generate", "completed").
depends("run_42", "run_42_llm_5", "run_42_step_4").
call("run_42", "run_42_step_5", "adjust_position", "completed").
depends("run_42", "run_42_step_5", "run_42_llm_5").
spawned_by("run_42", "run_42_step_5", "run_42_llm_5").
param("run_42", "run_42_step_5", "pos_distance", "0.107").
param("run_42", "run_42_step_5", "delta_gripper", "-3.47").
param("run_42", "run_42_step_5", "efficiency", "0.92").
param("run_42", "run_42_step_5", "mean_speed", "0.0389").
param("run_42", "run_42_step_5", "n_frames", "4").
param("run_42", "run_42_step_5", "start_gripper", "7.0").
param("run_42", "run_42_step_5", "end_gripper", "3.5").
call("run_42", "run_42_llm_6", "llm_generate", "completed").
depends("run_42", "run_42_llm_6", "run_42_step_5").
call("run_42", "run_42_step_6", "adjust_position", "completed").
depends("run_42", "run_42_step_6", "run_42_llm_6").
spawned_by("run_42", "run_42_step_6", "run_42_llm_6").
param("run_42", "run_42_step_6", "pos_distance", "0.285").
param("run_42", "run_42_step_6", "delta_gripper", "-0.39").
param("run_42", "run_42_step_6", "efficiency", "0.99").
param("run_42", "run_42_step_6", "mean_speed", "0.0964").
param("run_42", "run_42_step_6", "n_frames", "4").
param("run_42", "run_42_step_6", "start_gripper", "3.5").
param("run_42", "run_42_step_6", "end_gripper", "3.2").
call("run_42", "run_42_llm_7", "llm_generate", "completed").
depends("run_42", "run_42_llm_7", "run_42_step_6").
call("run_42", "run_42_step_7", "adjust_position", "completed").
depends("run_42", "run_42_step_7", "run_42_llm_7").
spawned_by("run_42", "run_42_step_7", "run_42_llm_7").
param("run_42", "run_42_step_7", "pos_distance", "0.434").
param("run_42", "run_42_step_7", "delta_gripper", "-3.07").
param("run_42", "run_42_step_7", "efficiency", "0.94").
param("run_42", "run_42_step_7", "mean_speed", "0.1546").
param("run_42", "run_42_step_7", "n_frames", "4").
param("run_42", "run_42_step_7", "start_gripper", "3.2").
param("run_42", "run_42_step_7", "end_gripper", "0.1").
call("run_42", "run_42_llm_8", "llm_generate", "completed").
depends("run_42", "run_42_llm_8", "run_42_step_7").
call("run_42", "run_42_step_8", "adjust_position_15", "completed").
depends("run_42", "run_42_step_8", "run_42_llm_8").
spawned_by("run_42", "run_42_step_8", "run_42_llm_8").
param("run_42", "run_42_step_8", "pos_distance", "0.493").
param("run_42", "run_42_step_8", "delta_gripper", "-0.06").
param("run_42", "run_42_step_8", "efficiency", "0.50").
param("run_42", "run_42_step_8", "mean_speed", "0.0354").
param("run_42", "run_42_step_8", "n_frames", "29").
param("run_42", "run_42_step_8", "start_gripper", "0.1").
param("run_42", "run_42_step_8", "end_gripper", "0.0").
call("run_42", "run_42_llm_9", "llm_generate", "completed").
depends("run_42", "run_42_llm_9", "run_42_step_8").
call("run_42", "run_42_notify", "notify_user", "completed").
depends("run_42", "run_42_notify", "run_42_llm_9").
spawned_by("run_42", "run_42_notify", "run_42_llm_9").

% === Episode 42 (10 discovered segments) ===
job("run_43").
job_status("run_43", "completed").
call("run_43", "run_43_step_0", "prompt_user", "completed").
call("run_43", "run_43_step_1", "llm_generate", "completed").
depends("run_43", "run_43_step_1", "run_43_step_0").
call("run_43", "run_43_llm_2", "llm_generate", "completed").
depends("run_43", "run_43_llm_2", "run_43_step_1").
call("run_43", "run_43_step_2", "move_y_neg", "completed").
depends("run_43", "run_43_step_2", "run_43_llm_2").
spawned_by("run_43", "run_43_step_2", "run_43_llm_2").
param("run_43", "run_43_step_2", "pos_distance", "0.949").
param("run_43", "run_43_step_2", "delta_gripper", "1.79").
param("run_43", "run_43_step_2", "efficiency", "0.98").
param("run_43", "run_43_step_2", "mean_speed", "0.2425").
param("run_43", "run_43_step_2", "n_frames", "5").
param("run_43", "run_43_step_2", "start_gripper", "6.0").
param("run_43", "run_43_step_2", "end_gripper", "7.8").
call("run_43", "run_43_llm_3", "llm_generate", "completed").
depends("run_43", "run_43_llm_3", "run_43_step_2").
call("run_43", "run_43_step_3", "move_x_pos", "completed").
depends("run_43", "run_43_step_3", "run_43_llm_3").
spawned_by("run_43", "run_43_step_3", "run_43_llm_3").
param("run_43", "run_43_step_3", "pos_distance", "0.771").
param("run_43", "run_43_step_3", "delta_gripper", "0.13").
param("run_43", "run_43_step_3", "efficiency", "0.99").
param("run_43", "run_43_step_3", "mean_speed", "0.2601").
param("run_43", "run_43_step_3", "n_frames", "4").
param("run_43", "run_43_step_3", "start_gripper", "7.8").
param("run_43", "run_43_step_3", "end_gripper", "8.0").
call("run_43", "run_43_llm_4", "llm_generate", "completed").
depends("run_43", "run_43_llm_4", "run_43_step_3").
call("run_43", "run_43_step_4", "move_x_pos", "completed").
depends("run_43", "run_43_step_4", "run_43_llm_4").
spawned_by("run_43", "run_43_step_4", "run_43_llm_4").
param("run_43", "run_43_step_4", "pos_distance", "0.847").
param("run_43", "run_43_step_4", "delta_gripper", "0.00").
param("run_43", "run_43_step_4", "efficiency", "0.90").
param("run_43", "run_43_step_4", "mean_speed", "0.3144").
param("run_43", "run_43_step_4", "n_frames", "4").
param("run_43", "run_43_step_4", "start_gripper", "8.0").
param("run_43", "run_43_step_4", "end_gripper", "8.0").
call("run_43", "run_43_llm_5", "llm_generate", "completed").
depends("run_43", "run_43_llm_5", "run_43_step_4").
call("run_43", "run_43_step_5", "move_x_pos", "completed").
depends("run_43", "run_43_step_5", "run_43_llm_5").
spawned_by("run_43", "run_43_step_5", "run_43_llm_5").
param("run_43", "run_43_step_5", "pos_distance", "0.656").
param("run_43", "run_43_step_5", "delta_gripper", "0.00").
param("run_43", "run_43_step_5", "efficiency", "0.97").
param("run_43", "run_43_step_5", "mean_speed", "0.2247").
param("run_43", "run_43_step_5", "n_frames", "4").
param("run_43", "run_43_step_5", "start_gripper", "8.0").
param("run_43", "run_43_step_5", "end_gripper", "8.0").
call("run_43", "run_43_llm_6", "llm_generate", "completed").
depends("run_43", "run_43_llm_6", "run_43_step_5").
call("run_43", "run_43_step_6", "adjust_position", "completed").
depends("run_43", "run_43_step_6", "run_43_llm_6").
spawned_by("run_43", "run_43_step_6", "run_43_llm_6").
param("run_43", "run_43_step_6", "pos_distance", "0.216").
param("run_43", "run_43_step_6", "delta_gripper", "0.00").
param("run_43", "run_43_step_6", "efficiency", "0.94").
param("run_43", "run_43_step_6", "mean_speed", "0.0770").
param("run_43", "run_43_step_6", "n_frames", "4").
param("run_43", "run_43_step_6", "start_gripper", "8.0").
param("run_43", "run_43_step_6", "end_gripper", "8.0").
call("run_43", "run_43_llm_7", "llm_generate", "completed").
depends("run_43", "run_43_llm_7", "run_43_step_6").
call("run_43", "run_43_step_7", "adjust_position_2", "completed").
depends("run_43", "run_43_step_7", "run_43_llm_7").
spawned_by("run_43", "run_43_step_7", "run_43_llm_7").
param("run_43", "run_43_step_7", "pos_distance", "0.305").
param("run_43", "run_43_step_7", "delta_gripper", "-0.62").
param("run_43", "run_43_step_7", "efficiency", "0.50").
param("run_43", "run_43_step_7", "mean_speed", "0.0872").
param("run_43", "run_43_step_7", "n_frames", "8").
param("run_43", "run_43_step_7", "start_gripper", "8.0").
param("run_43", "run_43_step_7", "end_gripper", "7.3").
call("run_43", "run_43_llm_8", "llm_generate", "completed").
depends("run_43", "run_43_llm_8", "run_43_step_7").
call("run_43", "run_43_step_8", "adjust_position", "completed").
depends("run_43", "run_43_step_8", "run_43_llm_8").
spawned_by("run_43", "run_43_step_8", "run_43_llm_8").
param("run_43", "run_43_step_8", "pos_distance", "0.440").
param("run_43", "run_43_step_8", "delta_gripper", "-0.99").
param("run_43", "run_43_step_8", "efficiency", "0.96").
param("run_43", "run_43_step_8", "mean_speed", "0.0418").
param("run_43", "run_43_step_8", "n_frames", "12").
param("run_43", "run_43_step_8", "start_gripper", "7.3").
param("run_43", "run_43_step_8", "end_gripper", "6.3").
call("run_43", "run_43_llm_9", "llm_generate", "completed").
depends("run_43", "run_43_llm_9", "run_43_step_8").
call("run_43", "run_43_step_9", "adjust_position", "completed").
depends("run_43", "run_43_step_9", "run_43_llm_9").
spawned_by("run_43", "run_43_step_9", "run_43_llm_9").
param("run_43", "run_43_step_9", "pos_distance", "0.068").
param("run_43", "run_43_step_9", "delta_gripper", "-0.90").
param("run_43", "run_43_step_9", "efficiency", "0.68").
param("run_43", "run_43_step_9", "mean_speed", "0.0332").
param("run_43", "run_43_step_9", "n_frames", "4").
param("run_43", "run_43_step_9", "start_gripper", "6.3").
param("run_43", "run_43_step_9", "end_gripper", "5.4").
call("run_43", "run_43_llm_10", "llm_generate", "completed").
depends("run_43", "run_43_llm_10", "run_43_step_9").
call("run_43", "run_43_step_10", "adjust_position", "completed").
depends("run_43", "run_43_step_10", "run_43_llm_10").
spawned_by("run_43", "run_43_step_10", "run_43_llm_10").
param("run_43", "run_43_step_10", "pos_distance", "0.615").
param("run_43", "run_43_step_10", "delta_gripper", "-1.26").
param("run_43", "run_43_step_10", "efficiency", "0.72").
param("run_43", "run_43_step_10", "mean_speed", "0.1073").
param("run_43", "run_43_step_10", "n_frames", "9").
param("run_43", "run_43_step_10", "start_gripper", "5.4").
param("run_43", "run_43_step_10", "end_gripper", "4.2").
call("run_43", "run_43_llm_11", "llm_generate", "completed").
depends("run_43", "run_43_llm_11", "run_43_step_10").
call("run_43", "run_43_step_11", "adjust_position", "completed").
depends("run_43", "run_43_step_11", "run_43_llm_11").
spawned_by("run_43", "run_43_step_11", "run_43_llm_11").
param("run_43", "run_43_step_11", "pos_distance", "0.351").
param("run_43", "run_43_step_11", "delta_gripper", "-1.30").
param("run_43", "run_43_step_11", "efficiency", "0.86").
param("run_43", "run_43_step_11", "mean_speed", "0.1025").
param("run_43", "run_43_step_11", "n_frames", "5").
param("run_43", "run_43_step_11", "start_gripper", "4.2").
param("run_43", "run_43_step_11", "end_gripper", "2.9").
call("run_43", "run_43_llm_12", "llm_generate", "completed").
depends("run_43", "run_43_llm_12", "run_43_step_11").
call("run_43", "run_43_notify", "notify_user", "completed").
depends("run_43", "run_43_notify", "run_43_llm_12").
spawned_by("run_43", "run_43_notify", "run_43_llm_12").

% === Episode 43 (6 discovered segments) ===
job("run_44").
job_status("run_44", "completed").
call("run_44", "run_44_step_0", "prompt_user", "completed").
call("run_44", "run_44_step_1", "llm_generate", "completed").
depends("run_44", "run_44_step_1", "run_44_step_0").
call("run_44", "run_44_llm_2", "llm_generate", "completed").
depends("run_44", "run_44_llm_2", "run_44_step_1").
call("run_44", "run_44_step_2", "move_x_pos", "completed").
depends("run_44", "run_44_step_2", "run_44_llm_2").
spawned_by("run_44", "run_44_step_2", "run_44_llm_2").
param("run_44", "run_44_step_2", "pos_distance", "0.771").
param("run_44", "run_44_step_2", "delta_gripper", "1.64").
param("run_44", "run_44_step_2", "efficiency", "1.00").
param("run_44", "run_44_step_2", "mean_speed", "0.2577").
param("run_44", "run_44_step_2", "n_frames", "4").
param("run_44", "run_44_step_2", "start_gripper", "6.0").
param("run_44", "run_44_step_2", "end_gripper", "7.7").
call("run_44", "run_44_llm_3", "llm_generate", "completed").
depends("run_44", "run_44_llm_3", "run_44_step_2").
call("run_44", "run_44_step_3", "move_x_pos", "completed").
depends("run_44", "run_44_step_3", "run_44_llm_3").
spawned_by("run_44", "run_44_step_3", "run_44_llm_3").
param("run_44", "run_44_step_3", "pos_distance", "0.704").
param("run_44", "run_44_step_3", "delta_gripper", "0.28").
param("run_44", "run_44_step_3", "efficiency", "1.00").
param("run_44", "run_44_step_3", "mean_speed", "0.2353").
param("run_44", "run_44_step_3", "n_frames", "4").
param("run_44", "run_44_step_3", "start_gripper", "7.7").
param("run_44", "run_44_step_3", "end_gripper", "8.0").
call("run_44", "run_44_llm_4", "llm_generate", "completed").
depends("run_44", "run_44_llm_4", "run_44_step_3").
call("run_44", "run_44_step_4", "close_gripper", "completed").
depends("run_44", "run_44_step_4", "run_44_llm_4").
spawned_by("run_44", "run_44_step_4", "run_44_llm_4").
param("run_44", "run_44_step_4", "pos_distance", "0.059").
param("run_44", "run_44_step_4", "delta_gripper", "-3.76").
param("run_44", "run_44_step_4", "efficiency", "0.43").
param("run_44", "run_44_step_4", "mean_speed", "0.0456").
param("run_44", "run_44_step_4", "n_frames", "4").
param("run_44", "run_44_step_4", "start_gripper", "8.0").
param("run_44", "run_44_step_4", "end_gripper", "4.2").
call("run_44", "run_44_llm_5", "llm_generate", "completed").
depends("run_44", "run_44_llm_5", "run_44_step_4").
call("run_44", "run_44_step_5", "adjust_position_2", "completed").
depends("run_44", "run_44_step_5", "run_44_llm_5").
spawned_by("run_44", "run_44_step_5", "run_44_llm_5").
param("run_44", "run_44_step_5", "pos_distance", "0.336").
param("run_44", "run_44_step_5", "delta_gripper", "0.29").
param("run_44", "run_44_step_5", "efficiency", "0.72").
param("run_44", "run_44_step_5", "mean_speed", "0.1555").
param("run_44", "run_44_step_5", "n_frames", "4").
param("run_44", "run_44_step_5", "start_gripper", "4.2").
param("run_44", "run_44_step_5", "end_gripper", "4.5").
call("run_44", "run_44_llm_6", "llm_generate", "completed").
depends("run_44", "run_44_llm_6", "run_44_step_5").
call("run_44", "run_44_step_6", "adjust_position", "completed").
depends("run_44", "run_44_step_6", "run_44_llm_6").
spawned_by("run_44", "run_44_step_6", "run_44_llm_6").
param("run_44", "run_44_step_6", "pos_distance", "0.355").
param("run_44", "run_44_step_6", "delta_gripper", "-1.10").
param("run_44", "run_44_step_6", "efficiency", "0.78").
param("run_44", "run_44_step_6", "mean_speed", "0.1510").
param("run_44", "run_44_step_6", "n_frames", "4").
param("run_44", "run_44_step_6", "start_gripper", "4.5").
param("run_44", "run_44_step_6", "end_gripper", "3.4").
call("run_44", "run_44_llm_7", "llm_generate", "completed").
depends("run_44", "run_44_llm_7", "run_44_step_6").
call("run_44", "run_44_step_7", "adjust_position_4", "completed").
depends("run_44", "run_44_step_7", "run_44_llm_7").
spawned_by("run_44", "run_44_step_7", "run_44_llm_7").
param("run_44", "run_44_step_7", "pos_distance", "1.604").
param("run_44", "run_44_step_7", "delta_gripper", "-0.03").
param("run_44", "run_44_step_7", "efficiency", "0.69").
param("run_44", "run_44_step_7", "mean_speed", "0.0686").
param("run_44", "run_44_step_7", "n_frames", "35").
param("run_44", "run_44_step_7", "start_gripper", "3.4").
param("run_44", "run_44_step_7", "end_gripper", "3.4").
call("run_44", "run_44_llm_8", "llm_generate", "completed").
depends("run_44", "run_44_llm_8", "run_44_step_7").
call("run_44", "run_44_notify", "notify_user", "completed").
depends("run_44", "run_44_notify", "run_44_llm_8").
spawned_by("run_44", "run_44_notify", "run_44_llm_8").

% === Episode 44 (5 discovered segments) ===
job("run_45").
job_status("run_45", "completed").
call("run_45", "run_45_step_0", "prompt_user", "completed").
call("run_45", "run_45_step_1", "llm_generate", "completed").
depends("run_45", "run_45_step_1", "run_45_step_0").
call("run_45", "run_45_llm_2", "llm_generate", "completed").
depends("run_45", "run_45_llm_2", "run_45_step_1").
call("run_45", "run_45_step_2", "adjust_position_5", "completed").
depends("run_45", "run_45_step_2", "run_45_llm_2").
spawned_by("run_45", "run_45_step_2", "run_45_llm_2").
param("run_45", "run_45_step_2", "pos_distance", "0.074").
param("run_45", "run_45_step_2", "delta_gripper", "1.67").
param("run_45", "run_45_step_2", "efficiency", "0.99").
param("run_45", "run_45_step_2", "mean_speed", "0.0252").
param("run_45", "run_45_step_2", "n_frames", "4").
param("run_45", "run_45_step_2", "start_gripper", "6.0").
param("run_45", "run_45_step_2", "end_gripper", "7.7").
call("run_45", "run_45_llm_3", "llm_generate", "completed").
depends("run_45", "run_45_llm_3", "run_45_step_2").
call("run_45", "run_45_step_3", "adjust_position_10", "completed").
depends("run_45", "run_45_step_3", "run_45_llm_3").
spawned_by("run_45", "run_45_step_3", "run_45_llm_3").
param("run_45", "run_45_step_3", "pos_distance", "2.415").
param("run_45", "run_45_step_3", "delta_gripper", "0.28").
param("run_45", "run_45_step_3", "efficiency", "0.91").
param("run_45", "run_45_step_3", "mean_speed", "0.0754").
param("run_45", "run_45_step_3", "n_frames", "36").
param("run_45", "run_45_step_3", "start_gripper", "7.7").
param("run_45", "run_45_step_3", "end_gripper", "8.0").
call("run_45", "run_45_llm_4", "llm_generate", "completed").
depends("run_45", "run_45_llm_4", "run_45_step_3").
call("run_45", "run_45_step_4", "adjust_position", "completed").
depends("run_45", "run_45_step_4", "run_45_llm_4").
spawned_by("run_45", "run_45_step_4", "run_45_llm_4").
param("run_45", "run_45_step_4", "pos_distance", "0.250").
param("run_45", "run_45_step_4", "delta_gripper", "-0.70").
param("run_45", "run_45_step_4", "efficiency", "0.94").
param("run_45", "run_45_step_4", "mean_speed", "0.0885").
param("run_45", "run_45_step_4", "n_frames", "4").
param("run_45", "run_45_step_4", "start_gripper", "8.0").
param("run_45", "run_45_step_4", "end_gripper", "7.3").
call("run_45", "run_45_llm_5", "llm_generate", "completed").
depends("run_45", "run_45_llm_5", "run_45_step_4").
call("run_45", "run_45_step_5", "adjust_position_5", "completed").
depends("run_45", "run_45_step_5", "run_45_llm_5").
spawned_by("run_45", "run_45_step_5", "run_45_llm_5").
param("run_45", "run_45_step_5", "pos_distance", "0.490").
param("run_45", "run_45_step_5", "delta_gripper", "0.68").
param("run_45", "run_45_step_5", "efficiency", "0.99").
param("run_45", "run_45_step_5", "mean_speed", "0.1235").
param("run_45", "run_45_step_5", "n_frames", "5").
param("run_45", "run_45_step_5", "start_gripper", "7.3").
param("run_45", "run_45_step_5", "end_gripper", "8.0").
call("run_45", "run_45_llm_6", "llm_generate", "completed").
depends("run_45", "run_45_llm_6", "run_45_step_5").
call("run_45", "run_45_step_6", "adjust_position", "completed").
depends("run_45", "run_45_step_6", "run_45_llm_6").
spawned_by("run_45", "run_45_step_6", "run_45_llm_6").
param("run_45", "run_45_step_6", "pos_distance", "0.480").
param("run_45", "run_45_step_6", "delta_gripper", "-2.58").
param("run_45", "run_45_step_6", "efficiency", "0.96").
param("run_45", "run_45_step_6", "mean_speed", "0.1253").
param("run_45", "run_45_step_6", "n_frames", "5").
param("run_45", "run_45_step_6", "start_gripper", "8.0").
param("run_45", "run_45_step_6", "end_gripper", "5.4").
call("run_45", "run_45_llm_7", "llm_generate", "completed").
depends("run_45", "run_45_llm_7", "run_45_step_6").
call("run_45", "run_45_notify", "notify_user", "completed").
depends("run_45", "run_45_notify", "run_45_llm_7").
spawned_by("run_45", "run_45_notify", "run_45_llm_7").

% === Episode 45 (14 discovered segments) ===
job("run_46").
job_status("run_46", "completed").
call("run_46", "run_46_step_0", "prompt_user", "completed").
call("run_46", "run_46_step_1", "llm_generate", "completed").
depends("run_46", "run_46_step_1", "run_46_step_0").
call("run_46", "run_46_llm_2", "llm_generate", "completed").
depends("run_46", "run_46_llm_2", "run_46_step_1").
call("run_46", "run_46_step_2", "close_gripper_4", "completed").
depends("run_46", "run_46_step_2", "run_46_llm_2").
spawned_by("run_46", "run_46_step_2", "run_46_llm_2").
param("run_46", "run_46_step_2", "pos_distance", "0.847").
param("run_46", "run_46_step_2", "delta_gripper", "-3.47").
param("run_46", "run_46_step_2", "efficiency", "0.99").
param("run_46", "run_46_step_2", "mean_speed", "0.2856").
param("run_46", "run_46_step_2", "n_frames", "4").
param("run_46", "run_46_step_2", "start_gripper", "6.0").
param("run_46", "run_46_step_2", "end_gripper", "2.6").
call("run_46", "run_46_llm_3", "llm_generate", "completed").
depends("run_46", "run_46_llm_3", "run_46_step_2").
call("run_46", "run_46_step_3", "move_y_pos", "completed").
depends("run_46", "run_46_step_3", "run_46_llm_3").
spawned_by("run_46", "run_46_step_3", "run_46_llm_3").
param("run_46", "run_46_step_3", "pos_distance", "0.668").
param("run_46", "run_46_step_3", "delta_gripper", "0.17").
param("run_46", "run_46_step_3", "efficiency", "0.75").
param("run_46", "run_46_step_3", "mean_speed", "0.2985").
param("run_46", "run_46_step_3", "n_frames", "4").
param("run_46", "run_46_step_3", "start_gripper", "2.6").
param("run_46", "run_46_step_3", "end_gripper", "2.7").
call("run_46", "run_46_llm_4", "llm_generate", "completed").
depends("run_46", "run_46_llm_4", "run_46_step_3").
call("run_46", "run_46_step_4", "move_x_pos", "completed").
depends("run_46", "run_46_step_4", "run_46_llm_4").
spawned_by("run_46", "run_46_step_4", "run_46_llm_4").
param("run_46", "run_46_step_4", "pos_distance", "0.766").
param("run_46", "run_46_step_4", "delta_gripper", "-0.22").
param("run_46", "run_46_step_4", "efficiency", "1.00").
param("run_46", "run_46_step_4", "mean_speed", "0.2559").
param("run_46", "run_46_step_4", "n_frames", "4").
param("run_46", "run_46_step_4", "start_gripper", "2.7").
param("run_46", "run_46_step_4", "end_gripper", "2.5").
call("run_46", "run_46_llm_5", "llm_generate", "completed").
depends("run_46", "run_46_llm_5", "run_46_step_4").
call("run_46", "run_46_step_5", "move_y_neg_3", "completed").
depends("run_46", "run_46_step_5", "run_46_llm_5").
spawned_by("run_46", "run_46_step_5", "run_46_llm_5").
param("run_46", "run_46_step_5", "pos_distance", "0.507").
param("run_46", "run_46_step_5", "delta_gripper", "0.25").
param("run_46", "run_46_step_5", "efficiency", "0.60").
param("run_46", "run_46_step_5", "mean_speed", "0.2819").
param("run_46", "run_46_step_5", "n_frames", "4").
param("run_46", "run_46_step_5", "start_gripper", "2.5").
param("run_46", "run_46_step_5", "end_gripper", "2.8").
call("run_46", "run_46_llm_6", "llm_generate", "completed").
depends("run_46", "run_46_llm_6", "run_46_step_5").
call("run_46", "run_46_step_6", "move_z_pos_6", "completed").
depends("run_46", "run_46_step_6", "run_46_llm_6").
spawned_by("run_46", "run_46_step_6", "run_46_llm_6").
param("run_46", "run_46_step_6", "pos_distance", "0.579").
param("run_46", "run_46_step_6", "delta_gripper", "0.12").
param("run_46", "run_46_step_6", "efficiency", "0.63").
param("run_46", "run_46_step_6", "mean_speed", "0.3053").
param("run_46", "run_46_step_6", "n_frames", "4").
param("run_46", "run_46_step_6", "start_gripper", "2.8").
param("run_46", "run_46_step_6", "end_gripper", "2.9").
call("run_46", "run_46_llm_7", "llm_generate", "completed").
depends("run_46", "run_46_llm_7", "run_46_step_6").
call("run_46", "run_46_step_7", "adjust_position_2", "completed").
depends("run_46", "run_46_step_7", "run_46_llm_7").
spawned_by("run_46", "run_46_step_7", "run_46_llm_7").
param("run_46", "run_46_step_7", "pos_distance", "0.402").
param("run_46", "run_46_step_7", "delta_gripper", "1.56").
param("run_46", "run_46_step_7", "efficiency", "0.49").
param("run_46", "run_46_step_7", "mean_speed", "0.2745").
param("run_46", "run_46_step_7", "n_frames", "4").
param("run_46", "run_46_step_7", "start_gripper", "2.9").
param("run_46", "run_46_step_7", "end_gripper", "4.4").
call("run_46", "run_46_llm_8", "llm_generate", "completed").
depends("run_46", "run_46_llm_8", "run_46_step_7").
call("run_46", "run_46_step_8", "move_y_neg_4", "completed").
depends("run_46", "run_46_step_8", "run_46_llm_8").
spawned_by("run_46", "run_46_step_8", "run_46_llm_8").
param("run_46", "run_46_step_8", "pos_distance", "0.617").
param("run_46", "run_46_step_8", "delta_gripper", "-1.18").
param("run_46", "run_46_step_8", "efficiency", "0.55").
param("run_46", "run_46_step_8", "mean_speed", "0.3717").
param("run_46", "run_46_step_8", "n_frames", "4").
param("run_46", "run_46_step_8", "start_gripper", "4.4").
param("run_46", "run_46_step_8", "end_gripper", "3.3").
call("run_46", "run_46_llm_9", "llm_generate", "completed").
depends("run_46", "run_46_llm_9", "run_46_step_8").
call("run_46", "run_46_step_9", "adjust_position_2", "completed").
depends("run_46", "run_46_step_9", "run_46_llm_9").
spawned_by("run_46", "run_46_step_9", "run_46_llm_9").
param("run_46", "run_46_step_9", "pos_distance", "0.589").
param("run_46", "run_46_step_9", "delta_gripper", "0.06").
param("run_46", "run_46_step_9", "efficiency", "0.68").
param("run_46", "run_46_step_9", "mean_speed", "0.2867").
param("run_46", "run_46_step_9", "n_frames", "4").
param("run_46", "run_46_step_9", "start_gripper", "3.3").
param("run_46", "run_46_step_9", "end_gripper", "3.3").
call("run_46", "run_46_llm_10", "llm_generate", "completed").
depends("run_46", "run_46_llm_10", "run_46_step_9").
call("run_46", "run_46_step_10", "adjust_position_2", "completed").
depends("run_46", "run_46_step_10", "run_46_llm_10").
spawned_by("run_46", "run_46_step_10", "run_46_llm_10").
param("run_46", "run_46_step_10", "pos_distance", "0.517").
param("run_46", "run_46_step_10", "delta_gripper", "0.22").
param("run_46", "run_46_step_10", "efficiency", "0.85").
param("run_46", "run_46_step_10", "mean_speed", "0.2030").
param("run_46", "run_46_step_10", "n_frames", "4").
param("run_46", "run_46_step_10", "start_gripper", "3.3").
param("run_46", "run_46_step_10", "end_gripper", "3.5").
call("run_46", "run_46_llm_11", "llm_generate", "completed").
depends("run_46", "run_46_llm_11", "run_46_step_10").
call("run_46", "run_46_step_11", "move_y_neg_3", "completed").
depends("run_46", "run_46_step_11", "run_46_llm_11").
spawned_by("run_46", "run_46_step_11", "run_46_llm_11").
param("run_46", "run_46_step_11", "pos_distance", "0.480").
param("run_46", "run_46_step_11", "delta_gripper", "-0.03").
param("run_46", "run_46_step_11", "efficiency", "0.60").
param("run_46", "run_46_step_11", "mean_speed", "0.2672").
param("run_46", "run_46_step_11", "n_frames", "4").
param("run_46", "run_46_step_11", "start_gripper", "3.5").
param("run_46", "run_46_step_11", "end_gripper", "3.5").
call("run_46", "run_46_llm_12", "llm_generate", "completed").
depends("run_46", "run_46_llm_12", "run_46_step_11").
call("run_46", "run_46_step_12", "open_gripper_2", "completed").
depends("run_46", "run_46_step_12", "run_46_llm_12").
spawned_by("run_46", "run_46_step_12", "run_46_llm_12").
param("run_46", "run_46_step_12", "pos_distance", "1.309").
param("run_46", "run_46_step_12", "delta_gripper", "3.46").
param("run_46", "run_46_step_12", "efficiency", "0.71").
param("run_46", "run_46_step_12", "mean_speed", "0.2306").
param("run_46", "run_46_step_12", "n_frames", "9").
param("run_46", "run_46_step_12", "start_gripper", "3.5").
param("run_46", "run_46_step_12", "end_gripper", "7.0").
call("run_46", "run_46_llm_13", "llm_generate", "completed").
depends("run_46", "run_46_llm_13", "run_46_step_12").
call("run_46", "run_46_step_13", "move_y_neg_3", "completed").
depends("run_46", "run_46_step_13", "run_46_llm_13").
spawned_by("run_46", "run_46_step_13", "run_46_llm_13").
param("run_46", "run_46_step_13", "pos_distance", "0.611").
param("run_46", "run_46_step_13", "delta_gripper", "-1.80").
param("run_46", "run_46_step_13", "efficiency", "0.55").
param("run_46", "run_46_step_13", "mean_speed", "0.2781").
param("run_46", "run_46_step_13", "n_frames", "5").
param("run_46", "run_46_step_13", "start_gripper", "7.0").
param("run_46", "run_46_step_13", "end_gripper", "5.2").
call("run_46", "run_46_llm_14", "llm_generate", "completed").
depends("run_46", "run_46_llm_14", "run_46_step_13").
call("run_46", "run_46_step_14", "adjust_position_2", "completed").
depends("run_46", "run_46_step_14", "run_46_llm_14").
spawned_by("run_46", "run_46_step_14", "run_46_llm_14").
param("run_46", "run_46_step_14", "pos_distance", "0.211").
param("run_46", "run_46_step_14", "delta_gripper", "0.98").
param("run_46", "run_46_step_14", "efficiency", "0.52").
param("run_46", "run_46_step_14", "mean_speed", "0.1347").
param("run_46", "run_46_step_14", "n_frames", "4").
param("run_46", "run_46_step_14", "start_gripper", "5.2").
param("run_46", "run_46_step_14", "end_gripper", "6.2").
call("run_46", "run_46_llm_15", "llm_generate", "completed").
depends("run_46", "run_46_llm_15", "run_46_step_14").
call("run_46", "run_46_step_15", "adjust_position_2", "completed").
depends("run_46", "run_46_step_15", "run_46_llm_15").
spawned_by("run_46", "run_46_step_15", "run_46_llm_15").
param("run_46", "run_46_step_15", "pos_distance", "0.101").
param("run_46", "run_46_step_15", "delta_gripper", "1.69").
param("run_46", "run_46_step_15", "efficiency", "0.24").
param("run_46", "run_46_step_15", "mean_speed", "0.1068").
param("run_46", "run_46_step_15", "n_frames", "5").
param("run_46", "run_46_step_15", "start_gripper", "6.2").
param("run_46", "run_46_step_15", "end_gripper", "7.8").
call("run_46", "run_46_llm_16", "llm_generate", "completed").
depends("run_46", "run_46_llm_16", "run_46_step_15").
call("run_46", "run_46_notify", "notify_user", "completed").
depends("run_46", "run_46_notify", "run_46_llm_16").
spawned_by("run_46", "run_46_notify", "run_46_llm_16").

% === Episode 46 (9 discovered segments) ===
job("run_47").
job_status("run_47", "completed").
call("run_47", "run_47_step_0", "prompt_user", "completed").
call("run_47", "run_47_step_1", "llm_generate", "completed").
depends("run_47", "run_47_step_1", "run_47_step_0").
call("run_47", "run_47_llm_2", "llm_generate", "completed").
depends("run_47", "run_47_llm_2", "run_47_step_1").
call("run_47", "run_47_step_2", "move_x_pos", "completed").
depends("run_47", "run_47_step_2", "run_47_llm_2").
spawned_by("run_47", "run_47_step_2", "run_47_llm_2").
param("run_47", "run_47_step_2", "pos_distance", "0.928").
param("run_47", "run_47_step_2", "delta_gripper", "-0.01").
param("run_47", "run_47_step_2", "efficiency", "0.92").
param("run_47", "run_47_step_2", "mean_speed", "0.1266").
param("run_47", "run_47_step_2", "n_frames", "9").
param("run_47", "run_47_step_2", "start_gripper", "4.0").
param("run_47", "run_47_step_2", "end_gripper", "4.0").
call("run_47", "run_47_llm_3", "llm_generate", "completed").
depends("run_47", "run_47_llm_3", "run_47_step_2").
call("run_47", "run_47_step_3", "adjust_position", "completed").
depends("run_47", "run_47_step_3", "run_47_llm_3").
spawned_by("run_47", "run_47_step_3", "run_47_llm_3").
param("run_47", "run_47_step_3", "pos_distance", "0.265").
param("run_47", "run_47_step_3", "delta_gripper", "-1.04").
param("run_47", "run_47_step_3", "efficiency", "1.00").
param("run_47", "run_47_step_3", "mean_speed", "0.0886").
param("run_47", "run_47_step_3", "n_frames", "4").
param("run_47", "run_47_step_3", "start_gripper", "4.0").
param("run_47", "run_47_step_3", "end_gripper", "3.0").
call("run_47", "run_47_llm_4", "llm_generate", "completed").
depends("run_47", "run_47_llm_4", "run_47_step_3").
call("run_47", "run_47_step_4", "close_gripper", "completed").
depends("run_47", "run_47_step_4", "run_47_llm_4").
spawned_by("run_47", "run_47_step_4", "run_47_llm_4").
param("run_47", "run_47_step_4", "pos_distance", "0.033").
param("run_47", "run_47_step_4", "delta_gripper", "-2.92").
param("run_47", "run_47_step_4", "efficiency", "0.51").
param("run_47", "run_47_step_4", "mean_speed", "0.0214").
param("run_47", "run_47_step_4", "n_frames", "4").
param("run_47", "run_47_step_4", "start_gripper", "3.0").
param("run_47", "run_47_step_4", "end_gripper", "0.1").
call("run_47", "run_47_llm_5", "llm_generate", "completed").
depends("run_47", "run_47_llm_5", "run_47_step_4").
call("run_47", "run_47_step_5", "adjust_position_17", "completed").
depends("run_47", "run_47_step_5", "run_47_llm_5").
spawned_by("run_47", "run_47_step_5", "run_47_llm_5").
param("run_47", "run_47_step_5", "pos_distance", "1.028").
param("run_47", "run_47_step_5", "delta_gripper", "-0.03").
param("run_47", "run_47_step_5", "efficiency", "0.95").
param("run_47", "run_47_step_5", "mean_speed", "0.1543").
param("run_47", "run_47_step_5", "n_frames", "8").
param("run_47", "run_47_step_5", "start_gripper", "0.1").
param("run_47", "run_47_step_5", "end_gripper", "0.0").
call("run_47", "run_47_llm_6", "llm_generate", "completed").
depends("run_47", "run_47_llm_6", "run_47_step_5").
call("run_47", "run_47_step_6", "move_x_pos_4", "completed").
depends("run_47", "run_47_step_6", "run_47_llm_6").
spawned_by("run_47", "run_47_step_6", "run_47_llm_6").
param("run_47", "run_47_step_6", "pos_distance", "1.277").
param("run_47", "run_47_step_6", "delta_gripper", "-0.01").
param("run_47", "run_47_step_6", "efficiency", "1.00").
param("run_47", "run_47_step_6", "mean_speed", "0.3195").
param("run_47", "run_47_step_6", "n_frames", "5").
param("run_47", "run_47_step_6", "start_gripper", "0.0").
param("run_47", "run_47_step_6", "end_gripper", "0.0").
call("run_47", "run_47_llm_7", "llm_generate", "completed").
depends("run_47", "run_47_llm_7", "run_47_step_6").
call("run_47", "run_47_step_7", "adjust_position_2", "completed").
depends("run_47", "run_47_step_7", "run_47_llm_7").
spawned_by("run_47", "run_47_step_7", "run_47_llm_7").
param("run_47", "run_47_step_7", "pos_distance", "0.599").
param("run_47", "run_47_step_7", "delta_gripper", "0.02").
param("run_47", "run_47_step_7", "efficiency", "0.77").
param("run_47", "run_47_step_7", "mean_speed", "0.0518").
param("run_47", "run_47_step_7", "n_frames", "16").
param("run_47", "run_47_step_7", "start_gripper", "0.0").
param("run_47", "run_47_step_7", "end_gripper", "0.1").
call("run_47", "run_47_llm_8", "llm_generate", "completed").
depends("run_47", "run_47_llm_8", "run_47_step_7").
call("run_47", "run_47_step_8", "adjust_position_2", "completed").
depends("run_47", "run_47_step_8", "run_47_llm_8").
spawned_by("run_47", "run_47_step_8", "run_47_llm_8").
param("run_47", "run_47_step_8", "pos_distance", "0.031").
param("run_47", "run_47_step_8", "delta_gripper", "0.68").
param("run_47", "run_47_step_8", "efficiency", "0.77").
param("run_47", "run_47_step_8", "mean_speed", "0.0135").
param("run_47", "run_47_step_8", "n_frames", "4").
param("run_47", "run_47_step_8", "start_gripper", "0.1").
param("run_47", "run_47_step_8", "end_gripper", "0.7").
call("run_47", "run_47_llm_9", "llm_generate", "completed").
depends("run_47", "run_47_llm_9", "run_47_step_8").
call("run_47", "run_47_step_9", "adjust_position_5", "completed").
depends("run_47", "run_47_step_9", "run_47_llm_9").
spawned_by("run_47", "run_47_step_9", "run_47_llm_9").
param("run_47", "run_47_step_9", "pos_distance", "0.064").
param("run_47", "run_47_step_9", "delta_gripper", "1.94").
param("run_47", "run_47_step_9", "efficiency", "0.99").
param("run_47", "run_47_step_9", "mean_speed", "0.0217").
param("run_47", "run_47_step_9", "n_frames", "4").
param("run_47", "run_47_step_9", "start_gripper", "0.7").
param("run_47", "run_47_step_9", "end_gripper", "2.7").
call("run_47", "run_47_llm_10", "llm_generate", "completed").
depends("run_47", "run_47_llm_10", "run_47_step_9").
call("run_47", "run_47_step_10", "adjust_position_5", "completed").
depends("run_47", "run_47_step_10", "run_47_llm_10").
spawned_by("run_47", "run_47_step_10", "run_47_llm_10").
param("run_47", "run_47_step_10", "pos_distance", "0.195").
param("run_47", "run_47_step_10", "delta_gripper", "1.29").
param("run_47", "run_47_step_10", "efficiency", "1.00").
param("run_47", "run_47_step_10", "mean_speed", "0.0649").
param("run_47", "run_47_step_10", "n_frames", "4").
param("run_47", "run_47_step_10", "start_gripper", "2.7").
param("run_47", "run_47_step_10", "end_gripper", "4.0").
call("run_47", "run_47_llm_11", "llm_generate", "completed").
depends("run_47", "run_47_llm_11", "run_47_step_10").
call("run_47", "run_47_notify", "notify_user", "completed").
depends("run_47", "run_47_notify", "run_47_llm_11").
spawned_by("run_47", "run_47_notify", "run_47_llm_11").

% === Episode 47 (6 discovered segments) ===
job("run_48").
job_status("run_48", "completed").
call("run_48", "run_48_step_0", "prompt_user", "completed").
call("run_48", "run_48_step_1", "llm_generate", "completed").
depends("run_48", "run_48_step_1", "run_48_step_0").
call("run_48", "run_48_llm_2", "llm_generate", "completed").
depends("run_48", "run_48_llm_2", "run_48_step_1").
call("run_48", "run_48_step_2", "adjust_position_2", "completed").
depends("run_48", "run_48_step_2", "run_48_llm_2").
spawned_by("run_48", "run_48_step_2", "run_48_llm_2").
param("run_48", "run_48_step_2", "pos_distance", "0.195").
param("run_48", "run_48_step_2", "delta_gripper", "1.65").
param("run_48", "run_48_step_2", "efficiency", "0.98").
param("run_48", "run_48_step_2", "mean_speed", "0.0662").
param("run_48", "run_48_step_2", "n_frames", "4").
param("run_48", "run_48_step_2", "start_gripper", "6.0").
param("run_48", "run_48_step_2", "end_gripper", "7.7").
call("run_48", "run_48_llm_3", "llm_generate", "completed").
depends("run_48", "run_48_llm_3", "run_48_step_2").
call("run_48", "run_48_step_3", "move_y_neg_2", "completed").
depends("run_48", "run_48_step_3", "run_48_llm_3").
spawned_by("run_48", "run_48_step_3", "run_48_llm_3").
param("run_48", "run_48_step_3", "pos_distance", "2.764").
param("run_48", "run_48_step_3", "delta_gripper", "0.29").
param("run_48", "run_48_step_3", "efficiency", "0.89").
param("run_48", "run_48_step_3", "mean_speed", "0.1722").
param("run_48", "run_48_step_3", "n_frames", "19").
param("run_48", "run_48_step_3", "start_gripper", "7.7").
param("run_48", "run_48_step_3", "end_gripper", "8.0").
call("run_48", "run_48_llm_4", "llm_generate", "completed").
depends("run_48", "run_48_llm_4", "run_48_step_3").
call("run_48", "run_48_step_4", "adjust_position", "completed").
depends("run_48", "run_48_step_4", "run_48_llm_4").
spawned_by("run_48", "run_48_step_4", "run_48_llm_4").
param("run_48", "run_48_step_4", "pos_distance", "0.083").
param("run_48", "run_48_step_4", "delta_gripper", "-1.91").
param("run_48", "run_48_step_4", "efficiency", "0.78").
param("run_48", "run_48_step_4", "mean_speed", "0.0356").
param("run_48", "run_48_step_4", "n_frames", "4").
param("run_48", "run_48_step_4", "start_gripper", "8.0").
param("run_48", "run_48_step_4", "end_gripper", "6.1").
call("run_48", "run_48_llm_5", "llm_generate", "completed").
depends("run_48", "run_48_llm_5", "run_48_step_4").
call("run_48", "run_48_step_5", "close_gripper", "completed").
depends("run_48", "run_48_step_5", "run_48_llm_5").
spawned_by("run_48", "run_48_step_5", "run_48_llm_5").
param("run_48", "run_48_step_5", "pos_distance", "0.031").
param("run_48", "run_48_step_5", "delta_gripper", "-3.97").
param("run_48", "run_48_step_5", "efficiency", "0.45").
param("run_48", "run_48_step_5", "mean_speed", "0.0173").
param("run_48", "run_48_step_5", "n_frames", "5").
param("run_48", "run_48_step_5", "start_gripper", "6.1").
param("run_48", "run_48_step_5", "end_gripper", "2.1").
call("run_48", "run_48_llm_6", "llm_generate", "completed").
depends("run_48", "run_48_llm_6", "run_48_step_5").
call("run_48", "run_48_step_6", "adjust_position", "completed").
depends("run_48", "run_48_step_6", "run_48_llm_6").
spawned_by("run_48", "run_48_step_6", "run_48_llm_6").
param("run_48", "run_48_step_6", "pos_distance", "0.221").
param("run_48", "run_48_step_6", "delta_gripper", "-2.05").
param("run_48", "run_48_step_6", "efficiency", "0.87").
param("run_48", "run_48_step_6", "mean_speed", "0.0848").
param("run_48", "run_48_step_6", "n_frames", "4").
param("run_48", "run_48_step_6", "start_gripper", "2.1").
param("run_48", "run_48_step_6", "end_gripper", "0.0").
call("run_48", "run_48_llm_7", "llm_generate", "completed").
depends("run_48", "run_48_llm_7", "run_48_step_6").
call("run_48", "run_48_step_7", "adjust_position_11", "completed").
depends("run_48", "run_48_step_7", "run_48_llm_7").
spawned_by("run_48", "run_48_step_7", "run_48_llm_7").
param("run_48", "run_48_step_7", "pos_distance", "0.846").
param("run_48", "run_48_step_7", "delta_gripper", "-0.01").
param("run_48", "run_48_step_7", "efficiency", "0.94").
param("run_48", "run_48_step_7", "mean_speed", "0.0501").
param("run_48", "run_48_step_7", "n_frames", "19").
param("run_48", "run_48_step_7", "start_gripper", "0.0").
param("run_48", "run_48_step_7", "end_gripper", "0.0").
call("run_48", "run_48_llm_8", "llm_generate", "completed").
depends("run_48", "run_48_llm_8", "run_48_step_7").
call("run_48", "run_48_notify", "notify_user", "completed").
depends("run_48", "run_48_notify", "run_48_llm_8").
spawned_by("run_48", "run_48_notify", "run_48_llm_8").

% === Episode 48 (6 discovered segments) ===
job("run_49").
job_status("run_49", "completed").
call("run_49", "run_49_step_0", "prompt_user", "completed").
call("run_49", "run_49_step_1", "llm_generate", "completed").
depends("run_49", "run_49_step_1", "run_49_step_0").
call("run_49", "run_49_llm_2", "llm_generate", "completed").
depends("run_49", "run_49_llm_2", "run_49_step_1").
call("run_49", "run_49_step_2", "move_y_neg", "completed").
depends("run_49", "run_49_step_2", "run_49_llm_2").
spawned_by("run_49", "run_49_step_2", "run_49_llm_2").
param("run_49", "run_49_step_2", "pos_distance", "0.943").
param("run_49", "run_49_step_2", "delta_gripper", "1.67").
param("run_49", "run_49_step_2", "efficiency", "1.00").
param("run_49", "run_49_step_2", "mean_speed", "0.3147").
param("run_49", "run_49_step_2", "n_frames", "4").
param("run_49", "run_49_step_2", "start_gripper", "6.0").
param("run_49", "run_49_step_2", "end_gripper", "7.7").
call("run_49", "run_49_llm_3", "llm_generate", "completed").
depends("run_49", "run_49_llm_3", "run_49_step_2").
call("run_49", "run_49_step_3", "move_y_neg", "completed").
depends("run_49", "run_49_step_3", "run_49_llm_3").
spawned_by("run_49", "run_49_step_3", "run_49_llm_3").
param("run_49", "run_49_step_3", "pos_distance", "1.109").
param("run_49", "run_49_step_3", "delta_gripper", "0.25").
param("run_49", "run_49_step_3", "efficiency", "1.00").
param("run_49", "run_49_step_3", "mean_speed", "0.3696").
param("run_49", "run_49_step_3", "n_frames", "4").
param("run_49", "run_49_step_3", "start_gripper", "7.7").
param("run_49", "run_49_step_3", "end_gripper", "8.0").
call("run_49", "run_49_llm_4", "llm_generate", "completed").
depends("run_49", "run_49_llm_4", "run_49_step_3").
call("run_49", "run_49_step_4", "move_x_pos_4", "completed").
depends("run_49", "run_49_step_4", "run_49_llm_4").
spawned_by("run_49", "run_49_step_4", "run_49_llm_4").
param("run_49", "run_49_step_4", "pos_distance", "1.076").
param("run_49", "run_49_step_4", "delta_gripper", "-1.30").
param("run_49", "run_49_step_4", "efficiency", "0.97").
param("run_49", "run_49_step_4", "mean_speed", "0.3713").
param("run_49", "run_49_step_4", "n_frames", "4").
param("run_49", "run_49_step_4", "start_gripper", "8.0").
param("run_49", "run_49_step_4", "end_gripper", "6.7").
call("run_49", "run_49_llm_5", "llm_generate", "completed").
depends("run_49", "run_49_llm_5", "run_49_step_4").
call("run_49", "run_49_step_5", "close_gripper", "completed").
depends("run_49", "run_49_step_5", "run_49_llm_5").
spawned_by("run_49", "run_49_step_5", "run_49_llm_5").
param("run_49", "run_49_step_5", "pos_distance", "0.203").
param("run_49", "run_49_step_5", "delta_gripper", "-2.95").
param("run_49", "run_49_step_5", "efficiency", "0.62").
param("run_49", "run_49_step_5", "mean_speed", "0.1085").
param("run_49", "run_49_step_5", "n_frames", "4").
param("run_49", "run_49_step_5", "start_gripper", "6.7").
param("run_49", "run_49_step_5", "end_gripper", "3.7").
call("run_49", "run_49_llm_6", "llm_generate", "completed").
depends("run_49", "run_49_llm_6", "run_49_step_5").
call("run_49", "run_49_step_6", "move_z_pos_2", "completed").
depends("run_49", "run_49_step_6", "run_49_llm_6").
spawned_by("run_49", "run_49_step_6", "run_49_llm_6").
param("run_49", "run_49_step_6", "pos_distance", "1.340").
param("run_49", "run_49_step_6", "delta_gripper", "-0.02").
param("run_49", "run_49_step_6", "efficiency", "0.98").
param("run_49", "run_49_step_6", "mean_speed", "0.2743").
param("run_49", "run_49_step_6", "n_frames", "6").
param("run_49", "run_49_step_6", "start_gripper", "3.7").
param("run_49", "run_49_step_6", "end_gripper", "3.7").
call("run_49", "run_49_llm_7", "llm_generate", "completed").
depends("run_49", "run_49_llm_7", "run_49_step_6").
call("run_49", "run_49_step_7", "adjust_position_12", "completed").
depends("run_49", "run_49_step_7", "run_49_llm_7").
spawned_by("run_49", "run_49_step_7", "run_49_llm_7").
param("run_49", "run_49_step_7", "pos_distance", "1.886").
param("run_49", "run_49_step_7", "delta_gripper", "-0.01").
param("run_49", "run_49_step_7", "efficiency", "0.67").
param("run_49", "run_49_step_7", "mean_speed", "0.0885").
param("run_49", "run_49_step_7", "n_frames", "33").
param("run_49", "run_49_step_7", "start_gripper", "3.7").
param("run_49", "run_49_step_7", "end_gripper", "3.7").
call("run_49", "run_49_llm_8", "llm_generate", "completed").
depends("run_49", "run_49_llm_8", "run_49_step_7").
call("run_49", "run_49_notify", "notify_user", "completed").
depends("run_49", "run_49_notify", "run_49_llm_8").
spawned_by("run_49", "run_49_notify", "run_49_llm_8").

% === Episode 49 (8 discovered segments) ===
job("run_50").
job_status("run_50", "completed").
call("run_50", "run_50_step_0", "prompt_user", "completed").
call("run_50", "run_50_step_1", "llm_generate", "completed").
depends("run_50", "run_50_step_1", "run_50_step_0").
call("run_50", "run_50_llm_2", "llm_generate", "completed").
depends("run_50", "run_50_llm_2", "run_50_step_1").
call("run_50", "run_50_step_2", "adjust_position_2", "completed").
depends("run_50", "run_50_step_2", "run_50_llm_2").
spawned_by("run_50", "run_50_step_2", "run_50_llm_2").
param("run_50", "run_50_step_2", "pos_distance", "0.372").
param("run_50", "run_50_step_2", "delta_gripper", "1.91").
param("run_50", "run_50_step_2", "efficiency", "0.99").
param("run_50", "run_50_step_2", "mean_speed", "0.0937").
param("run_50", "run_50_step_2", "n_frames", "5").
param("run_50", "run_50_step_2", "start_gripper", "6.0").
param("run_50", "run_50_step_2", "end_gripper", "7.9").
call("run_50", "run_50_llm_3", "llm_generate", "completed").
depends("run_50", "run_50_llm_3", "run_50_step_2").
call("run_50", "run_50_step_3", "move_x_pos", "completed").
depends("run_50", "run_50_step_3", "run_50_llm_3").
spawned_by("run_50", "run_50_step_3", "run_50_llm_3").
param("run_50", "run_50_step_3", "pos_distance", "0.712").
param("run_50", "run_50_step_3", "delta_gripper", "0.01").
param("run_50", "run_50_step_3", "efficiency", "1.00").
param("run_50", "run_50_step_3", "mean_speed", "0.2380").
param("run_50", "run_50_step_3", "n_frames", "4").
param("run_50", "run_50_step_3", "start_gripper", "7.9").
param("run_50", "run_50_step_3", "end_gripper", "8.0").
call("run_50", "run_50_llm_4", "llm_generate", "completed").
depends("run_50", "run_50_llm_4", "run_50_step_3").
call("run_50", "run_50_step_4", "move_x_pos", "completed").
depends("run_50", "run_50_step_4", "run_50_llm_4").
spawned_by("run_50", "run_50_step_4", "run_50_llm_4").
param("run_50", "run_50_step_4", "pos_distance", "0.707").
param("run_50", "run_50_step_4", "delta_gripper", "0.00").
param("run_50", "run_50_step_4", "efficiency", "0.99").
param("run_50", "run_50_step_4", "mean_speed", "0.2377").
param("run_50", "run_50_step_4", "n_frames", "4").
param("run_50", "run_50_step_4", "start_gripper", "8.0").
param("run_50", "run_50_step_4", "end_gripper", "8.0").
call("run_50", "run_50_llm_5", "llm_generate", "completed").
depends("run_50", "run_50_llm_5", "run_50_step_4").
call("run_50", "run_50_step_5", "move_x_pos", "completed").
depends("run_50", "run_50_step_5", "run_50_llm_5").
spawned_by("run_50", "run_50_step_5", "run_50_llm_5").
param("run_50", "run_50_step_5", "pos_distance", "0.648").
param("run_50", "run_50_step_5", "delta_gripper", "0.00").
param("run_50", "run_50_step_5", "efficiency", "0.96").
param("run_50", "run_50_step_5", "mean_speed", "0.2242").
param("run_50", "run_50_step_5", "n_frames", "4").
param("run_50", "run_50_step_5", "start_gripper", "8.0").
param("run_50", "run_50_step_5", "end_gripper", "8.0").
call("run_50", "run_50_llm_6", "llm_generate", "completed").
depends("run_50", "run_50_llm_6", "run_50_step_5").
call("run_50", "run_50_step_6", "move_x_pos", "completed").
depends("run_50", "run_50_step_6", "run_50_llm_6").
spawned_by("run_50", "run_50_step_6", "run_50_llm_6").
param("run_50", "run_50_step_6", "pos_distance", "0.632").
param("run_50", "run_50_step_6", "delta_gripper", "0.00").
param("run_50", "run_50_step_6", "efficiency", "1.00").
param("run_50", "run_50_step_6", "mean_speed", "0.2112").
param("run_50", "run_50_step_6", "n_frames", "4").
param("run_50", "run_50_step_6", "start_gripper", "8.0").
param("run_50", "run_50_step_6", "end_gripper", "8.0").
call("run_50", "run_50_llm_7", "llm_generate", "completed").
depends("run_50", "run_50_llm_7", "run_50_step_6").
call("run_50", "run_50_step_7", "move_x_pos", "completed").
depends("run_50", "run_50_step_7", "run_50_llm_7").
spawned_by("run_50", "run_50_step_7", "run_50_llm_7").
param("run_50", "run_50_step_7", "pos_distance", "0.407").
param("run_50", "run_50_step_7", "delta_gripper", "-1.04").
param("run_50", "run_50_step_7", "efficiency", "0.96").
param("run_50", "run_50_step_7", "mean_speed", "0.1409").
param("run_50", "run_50_step_7", "n_frames", "4").
param("run_50", "run_50_step_7", "start_gripper", "8.0").
param("run_50", "run_50_step_7", "end_gripper", "6.9").
call("run_50", "run_50_llm_8", "llm_generate", "completed").
depends("run_50", "run_50_llm_8", "run_50_step_7").
call("run_50", "run_50_step_8", "adjust_position", "completed").
depends("run_50", "run_50_step_8", "run_50_llm_8").
spawned_by("run_50", "run_50_step_8", "run_50_llm_8").
param("run_50", "run_50_step_8", "pos_distance", "0.090").
param("run_50", "run_50_step_8", "delta_gripper", "-3.54").
param("run_50", "run_50_step_8", "efficiency", "0.95").
param("run_50", "run_50_step_8", "mean_speed", "0.0316").
param("run_50", "run_50_step_8", "n_frames", "4").
param("run_50", "run_50_step_8", "start_gripper", "6.9").
param("run_50", "run_50_step_8", "end_gripper", "3.4").
call("run_50", "run_50_llm_9", "llm_generate", "completed").
depends("run_50", "run_50_llm_9", "run_50_step_8").
call("run_50", "run_50_step_9", "adjust_position_4", "completed").
depends("run_50", "run_50_step_9", "run_50_llm_9").
spawned_by("run_50", "run_50_step_9", "run_50_llm_9").
param("run_50", "run_50_step_9", "pos_distance", "1.436").
param("run_50", "run_50_step_9", "delta_gripper", "-0.08").
param("run_50", "run_50_step_9", "efficiency", "0.86").
param("run_50", "run_50_step_9", "mean_speed", "0.0617").
param("run_50", "run_50_step_9", "n_frames", "28").
param("run_50", "run_50_step_9", "start_gripper", "3.4").
param("run_50", "run_50_step_9", "end_gripper", "3.3").
call("run_50", "run_50_llm_10", "llm_generate", "completed").
depends("run_50", "run_50_llm_10", "run_50_step_9").
call("run_50", "run_50_notify", "notify_user", "completed").
depends("run_50", "run_50_notify", "run_50_llm_10").
spawned_by("run_50", "run_50_notify", "run_50_llm_10").
