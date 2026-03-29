% === Job: Travel Updates Last 6 Hours  — waiting ===
job("run_1").
job_status("run_1", "waiting").
job_title("run_1", "Travel Updates Last 6 Hours").
call("run_1", "run_1_step_1", "prompt_user", "completed").
call("run_1", "run_1_step_2", "llm_generate", "completed").
depends("run_1", "run_1_step_2", "run_1_step_1").
call("run_1", "run_1_step_3", "gmail_search", "failed").
depends("run_1", "run_1_step_3", "run_1_step_2").
spawned_by("run_1", "run_1_step_3", "run_1_step_2").
param("run_1", "run_1_step_3", "query", "after:2026/03/28").
param("run_1", "run_1_step_3", "account", "acct_1").
param("run_1", "run_1_step_3", "max_results", "20").
call("run_1", "run_1_step_4", "gmail_search", "completed").
depends("run_1", "run_1_step_4", "run_1_step_2").
spawned_by("run_1", "run_1_step_4", "run_1_step_2").
param("run_1", "run_1_step_4", "query", "after:2026/03/28").
param("run_1", "run_1_step_4", "account", "acct_2").
param("run_1", "run_1_step_4", "max_results", "20").
call("run_1", "run_1_step_5", "llm_generate", "completed").
depends("run_1", "run_1_step_5", "run_1_step_3").
depends("run_1", "run_1_step_5", "run_1_step_4").
call("run_1", "run_1_step_6", "sheets_read", "completed").
depends("run_1", "run_1_step_6", "run_1_step_5").
spawned_by("run_1", "run_1_step_6", "run_1_step_5").
param("run_1", "run_1_step_6", "spreadsheet", "spreadsheet_1").
param("run_1", "run_1_step_6", "account", "acct_3").
call("run_1", "run_1_step_7", "gmail_search", "completed").
depends("run_1", "run_1_step_7", "run_1_step_5").
spawned_by("run_1", "run_1_step_7", "run_1_step_5").
param("run_1", "run_1_step_7", "query", "after:2026/03/28 (flight OR hotel OR booking OR reservation OR airbnb OR airline OR confirmation OR travel OR train OR ferry)").
param("run_1", "run_1_step_7", "account", "acct_4").
param("run_1", "run_1_step_7", "max_results", "20").
call("run_1", "run_1_step_8", "gmail_search", "completed").
depends("run_1", "run_1_step_8", "run_1_step_5").
spawned_by("run_1", "run_1_step_8", "run_1_step_5").
param("run_1", "run_1_step_8", "query", "after:2026/03/28 (flight OR hotel OR booking OR reservation OR airbnb OR airline OR confirmation OR travel OR train OR ferry)").
param("run_1", "run_1_step_8", "account", "acct_2").
param("run_1", "run_1_step_8", "max_results", "20").
call("run_1", "run_1_step_9", "llm_generate", "completed").
depends("run_1", "run_1_step_9", "run_1_step_7").
depends("run_1", "run_1_step_9", "run_1_step_8").
depends("run_1", "run_1_step_9", "run_1_step_6").
call("run_1", "run_1_step_10", "notify_user", "completed").
depends("run_1", "run_1_step_10", "run_1_step_9").
spawned_by("run_1", "run_1_step_10", "run_1_step_9").

% === Job: Travel Updates Last 6 Hours  — failed ===
job("run_2").
job_status("run_2", "failed").
job_title("run_2", "Travel Updates Last 6 Hours").
call("run_2", "run_2_step_1", "prompt_user", "completed").
call("run_2", "run_2_step_2", "llm_generate", "completed").
depends("run_2", "run_2_step_2", "run_2_step_1").
call("run_2", "run_2_step_3", "gmail_search", "failed").
depends("run_2", "run_2_step_3", "run_2_step_2").
spawned_by("run_2", "run_2_step_3", "run_2_step_2").
param("run_2", "run_2_step_3", "query", "travel flight hotel booking confirmation after:2026/03/28").
param("run_2", "run_2_step_3", "account", "acct_1").
param("run_2", "run_2_step_3", "max_results", "20").
call("run_2", "run_2_step_4", "gmail_search", "completed").
depends("run_2", "run_2_step_4", "run_2_step_2").
spawned_by("run_2", "run_2_step_4", "run_2_step_2").
param("run_2", "run_2_step_4", "query", "travel flight hotel booking confirmation after:2026/03/28").
param("run_2", "run_2_step_4", "account", "acct_2").
param("run_2", "run_2_step_4", "max_results", "20").
call("run_2", "run_2_step_5", "sheets_get_spreadsheet", "completed").
depends("run_2", "run_2_step_5", "run_2_step_2").
spawned_by("run_2", "run_2_step_5", "run_2_step_2").
param("run_2", "run_2_step_5", "spreadsheet", "spreadsheet_2").
param("run_2", "run_2_step_5", "account", "acct_3").
call("run_2", "run_2_step_6", "sheets_get_spreadsheet", "completed").
depends("run_2", "run_2_step_6", "run_2_step_2").
spawned_by("run_2", "run_2_step_6", "run_2_step_2").
param("run_2", "run_2_step_6", "spreadsheet", "spreadsheet_1").
param("run_2", "run_2_step_6", "account", "acct_3").
call("run_2", "run_2_step_7", "llm_generate", "completed").
depends("run_2", "run_2_step_7", "run_2_step_3").
depends("run_2", "run_2_step_7", "run_2_step_4").
depends("run_2", "run_2_step_7", "run_2_step_6").
depends("run_2", "run_2_step_7", "run_2_step_5").
call("run_2", "run_2_step_8", "sheets_read", "completed").
depends("run_2", "run_2_step_8", "run_2_step_7").
spawned_by("run_2", "run_2_step_8", "run_2_step_7").
param("run_2", "run_2_step_8", "spreadsheet", "spreadsheet_1").
param("run_2", "run_2_step_8", "account", "acct_3").
param("run_2", "run_2_step_8", "sheet_name", "Flights").
call("run_2", "run_2_step_9", "sheets_read", "completed").
depends("run_2", "run_2_step_9", "run_2_step_7").
spawned_by("run_2", "run_2_step_9", "run_2_step_7").
param("run_2", "run_2_step_9", "spreadsheet", "spreadsheet_2").
param("run_2", "run_2_step_9", "account", "acct_3").
param("run_2", "run_2_step_9", "sheet_name", "🏨 Hotels").
call("run_2", "run_2_step_10", "gmail_list_threads", "completed").
depends("run_2", "run_2_step_10", "run_2_step_7").
spawned_by("run_2", "run_2_step_10", "run_2_step_7").
param("run_2", "run_2_step_10", "account", "acct_4").
call("run_2", "run_2_step_11", "gmail_list_threads", "completed").
depends("run_2", "run_2_step_11", "run_2_step_7").
spawned_by("run_2", "run_2_step_11", "run_2_step_7").
param("run_2", "run_2_step_11", "account", "acct_5").
call("run_2", "run_2_step_12", "llm_generate", "completed").
depends("run_2", "run_2_step_12", "run_2_step_10").
depends("run_2", "run_2_step_12", "run_2_step_11").
depends("run_2", "run_2_step_12", "run_2_step_8").
depends("run_2", "run_2_step_12", "run_2_step_9").
call("run_2", "run_2_step_13", "gmail_list_threads", "completed").
depends("run_2", "run_2_step_13", "run_2_step_12").
spawned_by("run_2", "run_2_step_13", "run_2_step_12").
param("run_2", "run_2_step_13", "account", "acct_5").
call("run_2", "run_2_step_14", "gmail_list_threads", "completed").
depends("run_2", "run_2_step_14", "run_2_step_12").
spawned_by("run_2", "run_2_step_14", "run_2_step_12").
param("run_2", "run_2_step_14", "account", "acct_4").
call("run_2", "run_2_step_15", "gmail_list_threads", "completed").
depends("run_2", "run_2_step_15", "run_2_step_12").
spawned_by("run_2", "run_2_step_15", "run_2_step_12").
param("run_2", "run_2_step_15", "account", "acct_2").
call("run_2", "run_2_step_16", "llm_generate", "completed").
depends("run_2", "run_2_step_16", "run_2_step_14").
depends("run_2", "run_2_step_16", "run_2_step_13").
depends("run_2", "run_2_step_16", "run_2_step_15").
call("run_2", "run_2_step_17", "gmail_get_thread", "completed").
depends("run_2", "run_2_step_17", "run_2_step_16").
spawned_by("run_2", "run_2_step_17", "run_2_step_16").
param("run_2", "run_2_step_17", "thread_id", "thread_1").
param("run_2", "run_2_step_17", "account", "acct_5").
call("run_2", "run_2_step_18", "gmail_get_thread", "completed").
depends("run_2", "run_2_step_18", "run_2_step_16").
spawned_by("run_2", "run_2_step_18", "run_2_step_16").
param("run_2", "run_2_step_18", "thread_id", "thread_2").
param("run_2", "run_2_step_18", "account", "acct_4").
call("run_2", "run_2_step_19", "gmail_get_thread", "completed").
depends("run_2", "run_2_step_19", "run_2_step_16").
spawned_by("run_2", "run_2_step_19", "run_2_step_16").
param("run_2", "run_2_step_19", "thread_id", "thread_3").
param("run_2", "run_2_step_19", "account", "acct_4").
call("run_2", "run_2_step_20", "llm_generate", "completed").
depends("run_2", "run_2_step_20", "run_2_step_18").
depends("run_2", "run_2_step_20", "run_2_step_19").
depends("run_2", "run_2_step_20", "run_2_step_17").
call("run_2", "run_2_step_21", "notify_user", "completed").
depends("run_2", "run_2_step_21", "run_2_step_20").
spawned_by("run_2", "run_2_step_21", "run_2_step_20").

% === Job: Travel Updates Last 6 Hours  — failed ===
job("run_3").
job_status("run_3", "failed").
job_title("run_3", "Travel Updates Last 6 Hours").
call("run_3", "run_3_step_1", "prompt_user", "completed").
call("run_3", "run_3_step_2", "llm_generate", "completed").
depends("run_3", "run_3_step_2", "run_3_step_1").
call("run_3", "run_3_step_3", "gmail_search", "failed").
depends("run_3", "run_3_step_3", "run_3_step_2").
spawned_by("run_3", "run_3_step_3", "run_3_step_2").
param("run_3", "run_3_step_3", "query", "travel flight hotel booking confirmation after:2026/03/28").
param("run_3", "run_3_step_3", "account", "acct_1").
param("run_3", "run_3_step_3", "max_results", "20").
call("run_3", "run_3_step_4", "sheets_get_spreadsheet", "completed").
depends("run_3", "run_3_step_4", "run_3_step_2").
spawned_by("run_3", "run_3_step_4", "run_3_step_2").
param("run_3", "run_3_step_4", "spreadsheet", "spreadsheet_2").
param("run_3", "run_3_step_4", "account", "acct_3").
call("run_3", "run_3_step_5", "sheets_get_spreadsheet", "completed").
depends("run_3", "run_3_step_5", "run_3_step_2").
spawned_by("run_3", "run_3_step_5", "run_3_step_2").
param("run_3", "run_3_step_5", "spreadsheet", "spreadsheet_1").
param("run_3", "run_3_step_5", "account", "acct_3").
call("run_3", "run_3_step_6", "gmail_search", "completed").
depends("run_3", "run_3_step_6", "run_3_step_2").
spawned_by("run_3", "run_3_step_6", "run_3_step_2").
param("run_3", "run_3_step_6", "query", "travel flight hotel booking confirmation after:2026/03/28").
param("run_3", "run_3_step_6", "account", "acct_2").
param("run_3", "run_3_step_6", "max_results", "20").
call("run_3", "run_3_step_7", "llm_generate", "completed").
depends("run_3", "run_3_step_7", "run_3_step_3").
depends("run_3", "run_3_step_7", "run_3_step_6").
depends("run_3", "run_3_step_7", "run_3_step_5").
depends("run_3", "run_3_step_7", "run_3_step_4").
call("run_3", "run_3_step_8", "gmail_list_threads", "completed").
depends("run_3", "run_3_step_8", "run_3_step_7").
spawned_by("run_3", "run_3_step_8", "run_3_step_7").
param("run_3", "run_3_step_8", "account", "acct_4").
call("run_3", "run_3_step_9", "sheets_read", "completed").
depends("run_3", "run_3_step_9", "run_3_step_7").
spawned_by("run_3", "run_3_step_9", "run_3_step_7").
param("run_3", "run_3_step_9", "spreadsheet", "spreadsheet_2").
param("run_3", "run_3_step_9", "account", "acct_3").
param("run_3", "run_3_step_9", "sheet_name", "🏨 Hotels").
call("run_3", "run_3_step_10", "sheets_read", "completed").
depends("run_3", "run_3_step_10", "run_3_step_7").
spawned_by("run_3", "run_3_step_10", "run_3_step_7").
param("run_3", "run_3_step_10", "spreadsheet", "spreadsheet_1").
param("run_3", "run_3_step_10", "account", "acct_3").
param("run_3", "run_3_step_10", "sheet_name", "Flights").
call("run_3", "run_3_step_11", "gmail_list_threads", "completed").
depends("run_3", "run_3_step_11", "run_3_step_7").
spawned_by("run_3", "run_3_step_11", "run_3_step_7").
param("run_3", "run_3_step_11", "account", "acct_5").
call("run_3", "run_3_step_12", "llm_generate", "completed").
depends("run_3", "run_3_step_12", "run_3_step_8").
depends("run_3", "run_3_step_12", "run_3_step_11").
depends("run_3", "run_3_step_12", "run_3_step_10").
depends("run_3", "run_3_step_12", "run_3_step_9").
call("run_3", "run_3_step_13", "gmail_list_threads", "completed").
depends("run_3", "run_3_step_13", "run_3_step_12").
spawned_by("run_3", "run_3_step_13", "run_3_step_12").
param("run_3", "run_3_step_13", "account", "acct_2").
call("run_3", "run_3_step_14", "gmail_list_threads", "completed").
depends("run_3", "run_3_step_14", "run_3_step_12").
spawned_by("run_3", "run_3_step_14", "run_3_step_12").
param("run_3", "run_3_step_14", "account", "acct_4").
call("run_3", "run_3_step_15", "gmail_list_threads", "completed").
depends("run_3", "run_3_step_15", "run_3_step_12").
spawned_by("run_3", "run_3_step_15", "run_3_step_12").
param("run_3", "run_3_step_15", "account", "acct_5").
call("run_3", "run_3_step_16", "llm_generate", "completed").
depends("run_3", "run_3_step_16", "run_3_step_14").
depends("run_3", "run_3_step_16", "run_3_step_15").
depends("run_3", "run_3_step_16", "run_3_step_13").
call("run_3", "run_3_step_17", "notify_user", "completed").
depends("run_3", "run_3_step_17", "run_3_step_16").
spawned_by("run_3", "run_3_step_17", "run_3_step_16").

% === Job: Travel Updates Last 6 Hours  — failed ===
job("run_4").
job_status("run_4", "failed").
job_title("run_4", "Travel Updates Last 6 Hours").
call("run_4", "run_4_step_1", "prompt_user", "completed").
call("run_4", "run_4_step_2", "llm_generate", "completed").
depends("run_4", "run_4_step_2", "run_4_step_1").
call("run_4", "run_4_step_3", "sheets_read", "completed").
depends("run_4", "run_4_step_3", "run_4_step_2").
spawned_by("run_4", "run_4_step_3", "run_4_step_2").
param("run_4", "run_4_step_3", "spreadsheet", "spreadsheet_2").
param("run_4", "run_4_step_3", "account", "acct_3").
call("run_4", "run_4_step_4", "sheets_read", "completed").
depends("run_4", "run_4_step_4", "run_4_step_2").
spawned_by("run_4", "run_4_step_4", "run_4_step_2").
param("run_4", "run_4_step_4", "spreadsheet", "spreadsheet_1").
param("run_4", "run_4_step_4", "account", "acct_3").
call("run_4", "run_4_step_5", "gmail_search", "completed").
depends("run_4", "run_4_step_5", "run_4_step_2").
spawned_by("run_4", "run_4_step_5", "run_4_step_2").
param("run_4", "run_4_step_5", "query", "travel flight hotel booking after:2026/03/28").
param("run_4", "run_4_step_5", "account", "acct_2").
param("run_4", "run_4_step_5", "max_results", "20").
call("run_4", "run_4_step_6", "gmail_search", "failed").
depends("run_4", "run_4_step_6", "run_4_step_2").
spawned_by("run_4", "run_4_step_6", "run_4_step_2").
param("run_4", "run_4_step_6", "query", "travel flight hotel booking after:2026/03/28").
param("run_4", "run_4_step_6", "account", "acct_1").
param("run_4", "run_4_step_6", "max_results", "20").
call("run_4", "run_4_step_7", "llm_generate", "completed").
depends("run_4", "run_4_step_7", "run_4_step_6").
depends("run_4", "run_4_step_7", "run_4_step_5").
depends("run_4", "run_4_step_7", "run_4_step_4").
depends("run_4", "run_4_step_7", "run_4_step_3").
call("run_4", "run_4_step_8", "gmail_search", "completed").
depends("run_4", "run_4_step_8", "run_4_step_7").
spawned_by("run_4", "run_4_step_8", "run_4_step_7").
param("run_4", "run_4_step_8", "query", "travel flight hotel booking confirmation after:2026/03/28").
param("run_4", "run_4_step_8", "account", "acct_4").
param("run_4", "run_4_step_8", "max_results", "20").
call("run_4", "run_4_step_9", "gmail_search", "completed").
depends("run_4", "run_4_step_9", "run_4_step_7").
spawned_by("run_4", "run_4_step_9", "run_4_step_7").
param("run_4", "run_4_step_9", "query", "travel flight hotel booking confirmation after:2026/03/28").
param("run_4", "run_4_step_9", "account", "acct_5").
param("run_4", "run_4_step_9", "max_results", "20").
call("run_4", "run_4_step_10", "llm_generate", "completed").
depends("run_4", "run_4_step_10", "run_4_step_8").
depends("run_4", "run_4_step_10", "run_4_step_9").
call("run_4", "run_4_step_11", "gmail_list_threads", "completed").
depends("run_4", "run_4_step_11", "run_4_step_10").
spawned_by("run_4", "run_4_step_11", "run_4_step_10").
param("run_4", "run_4_step_11", "account", "acct_5").
call("run_4", "run_4_step_12", "gmail_list_threads", "completed").
depends("run_4", "run_4_step_12", "run_4_step_10").
spawned_by("run_4", "run_4_step_12", "run_4_step_10").
param("run_4", "run_4_step_12", "account", "acct_2").
call("run_4", "run_4_step_13", "gmail_list_threads", "completed").
depends("run_4", "run_4_step_13", "run_4_step_10").
spawned_by("run_4", "run_4_step_13", "run_4_step_10").
param("run_4", "run_4_step_13", "account", "acct_4").
call("run_4", "run_4_step_14", "llm_generate", "completed").
depends("run_4", "run_4_step_14", "run_4_step_11").
depends("run_4", "run_4_step_14", "run_4_step_13").
depends("run_4", "run_4_step_14", "run_4_step_12").
call("run_4", "run_4_step_15", "notify_user", "completed").
depends("run_4", "run_4_step_15", "run_4_step_14").
spawned_by("run_4", "run_4_step_15", "run_4_step_14").

% === Job: Travel Updates Last 6 Hours  — failed ===
job("run_5").
job_status("run_5", "failed").
job_title("run_5", "Travel Updates Last 6 Hours").
call("run_5", "run_5_step_1", "prompt_user", "completed").
call("run_5", "run_5_step_2", "llm_generate", "completed").
depends("run_5", "run_5_step_2", "run_5_step_1").
call("run_5", "run_5_step_3", "sheets_read", "completed").
depends("run_5", "run_5_step_3", "run_5_step_2").
spawned_by("run_5", "run_5_step_3", "run_5_step_2").
param("run_5", "run_5_step_3", "spreadsheet", "spreadsheet_1").
param("run_5", "run_5_step_3", "account", "acct_3").
call("run_5", "run_5_step_4", "gmail_search", "completed").
depends("run_5", "run_5_step_4", "run_5_step_2").
spawned_by("run_5", "run_5_step_4", "run_5_step_2").
param("run_5", "run_5_step_4", "query", "travel flight hotel booking confirmation").
param("run_5", "run_5_step_4", "account", "acct_2").
param("run_5", "run_5_step_4", "max_results", "20").
call("run_5", "run_5_step_5", "gmail_search", "failed").
depends("run_5", "run_5_step_5", "run_5_step_2").
spawned_by("run_5", "run_5_step_5", "run_5_step_2").
param("run_5", "run_5_step_5", "query", "travel flight hotel booking confirmation").
param("run_5", "run_5_step_5", "account", "acct_1").
param("run_5", "run_5_step_5", "max_results", "20").
call("run_5", "run_5_step_6", "llm_generate", "completed").
depends("run_5", "run_5_step_6", "run_5_step_5").
depends("run_5", "run_5_step_6", "run_5_step_4").
depends("run_5", "run_5_step_6", "run_5_step_3").
call("run_5", "run_5_step_7", "gmail_search", "completed").
depends("run_5", "run_5_step_7", "run_5_step_6").
spawned_by("run_5", "run_5_step_7", "run_5_step_6").
param("run_5", "run_5_step_7", "query", "after:2026/03/27 subject:(flight OR hotel OR booking OR confirmation OR reservation OR itinerary OR airbnb OR travel)").
param("run_5", "run_5_step_7", "account", "acct_2").
param("run_5", "run_5_step_7", "max_results", "20").
call("run_5", "run_5_step_8", "gmail_search", "completed").
depends("run_5", "run_5_step_8", "run_5_step_6").
spawned_by("run_5", "run_5_step_8", "run_5_step_6").
param("run_5", "run_5_step_8", "query", "after:2026/03/27 subject:(flight OR hotel OR booking OR confirmation OR reservation OR itinerary OR airbnb OR travel)").
param("run_5", "run_5_step_8", "account", "acct_4").
param("run_5", "run_5_step_8", "max_results", "20").
call("run_5", "run_5_step_9", "llm_generate", "completed").
depends("run_5", "run_5_step_9", "run_5_step_7").
depends("run_5", "run_5_step_9", "run_5_step_8").
call("run_5", "run_5_step_10", "gmail_search", "completed").
depends("run_5", "run_5_step_10", "run_5_step_9").
spawned_by("run_5", "run_5_step_10", "run_5_step_9").
param("run_5", "run_5_step_10", "query", "after:2026/03/28").
param("run_5", "run_5_step_10", "account", "acct_2").
param("run_5", "run_5_step_10", "max_results", "20").
call("run_5", "run_5_step_11", "gmail_search", "completed").
depends("run_5", "run_5_step_11", "run_5_step_9").
spawned_by("run_5", "run_5_step_11", "run_5_step_9").
param("run_5", "run_5_step_11", "query", "after:2026/03/28").
param("run_5", "run_5_step_11", "account", "acct_4").
param("run_5", "run_5_step_11", "max_results", "20").
call("run_5", "run_5_step_12", "llm_generate", "completed").
depends("run_5", "run_5_step_12", "run_5_step_10").
depends("run_5", "run_5_step_12", "run_5_step_11").
call("run_5", "run_5_step_13", "notify_user", "completed").
depends("run_5", "run_5_step_13", "run_5_step_12").
spawned_by("run_5", "run_5_step_13", "run_5_step_12").

% === Job: Travel Updates Last 6 Hours  — failed ===
job("run_6").
job_status("run_6", "failed").
job_title("run_6", "Travel Updates Last 6 Hours").
call("run_6", "run_6_step_1", "prompt_user", "completed").
call("run_6", "run_6_step_2", "llm_generate", "completed").
depends("run_6", "run_6_step_2", "run_6_step_1").
call("run_6", "run_6_step_3", "gmail_search", "completed").
depends("run_6", "run_6_step_3", "run_6_step_2").
spawned_by("run_6", "run_6_step_3", "run_6_step_2").
param("run_6", "run_6_step_3", "query", "after:2026/03/27 travel flight hotel booking confirmation").
param("run_6", "run_6_step_3", "account", "acct_2").
param("run_6", "run_6_step_3", "max_results", "20").
call("run_6", "run_6_step_4", "gmail_search", "failed").
depends("run_6", "run_6_step_4", "run_6_step_2").
spawned_by("run_6", "run_6_step_4", "run_6_step_2").
param("run_6", "run_6_step_4", "query", "after:2026/03/27 travel flight hotel booking confirmation").
param("run_6", "run_6_step_4", "account", "acct_1").
param("run_6", "run_6_step_4", "max_results", "20").
call("run_6", "run_6_step_5", "llm_generate", "completed").
depends("run_6", "run_6_step_5", "run_6_step_4").
depends("run_6", "run_6_step_5", "run_6_step_3").
call("run_6", "run_6_step_6", "gmail_search", "completed").
depends("run_6", "run_6_step_6", "run_6_step_5").
spawned_by("run_6", "run_6_step_6", "run_6_step_5").
param("run_6", "run_6_step_6", "query", "travel flight hotel booking confirmation").
param("run_6", "run_6_step_6", "account", "acct_4").
param("run_6", "run_6_step_6", "max_results", "20").
call("run_6", "run_6_step_7", "gmail_search", "completed").
depends("run_6", "run_6_step_7", "run_6_step_5").
spawned_by("run_6", "run_6_step_7", "run_6_step_5").
param("run_6", "run_6_step_7", "query", "travel flight hotel booking confirmation").
param("run_6", "run_6_step_7", "account", "acct_5").
param("run_6", "run_6_step_7", "max_results", "20").
call("run_6", "run_6_step_8", "llm_generate", "completed").
depends("run_6", "run_6_step_8", "run_6_step_6").
depends("run_6", "run_6_step_8", "run_6_step_7").
call("run_6", "run_6_step_9", "gmail_search", "completed").
depends("run_6", "run_6_step_9", "run_6_step_8").
spawned_by("run_6", "run_6_step_9", "run_6_step_8").
param("run_6", "run_6_step_9", "query", "after:2026/03/27").
param("run_6", "run_6_step_9", "account", "acct_4").
param("run_6", "run_6_step_9", "max_results", "30").
call("run_6", "run_6_step_10", "gmail_search", "completed").
depends("run_6", "run_6_step_10", "run_6_step_8").
spawned_by("run_6", "run_6_step_10", "run_6_step_8").
param("run_6", "run_6_step_10", "query", "after:2026/03/27").
param("run_6", "run_6_step_10", "account", "acct_2").
param("run_6", "run_6_step_10", "max_results", "30").
call("run_6", "run_6_step_11", "gmail_search", "completed").
depends("run_6", "run_6_step_11", "run_6_step_8").
spawned_by("run_6", "run_6_step_11", "run_6_step_8").
param("run_6", "run_6_step_11", "query", "after:2026/03/27").
param("run_6", "run_6_step_11", "account", "acct_5").
param("run_6", "run_6_step_11", "max_results", "30").
call("run_6", "run_6_step_12", "llm_generate", "completed").
depends("run_6", "run_6_step_12", "run_6_step_9").
depends("run_6", "run_6_step_12", "run_6_step_11").
depends("run_6", "run_6_step_12", "run_6_step_10").
call("run_6", "run_6_step_13", "notify_user", "completed").
depends("run_6", "run_6_step_13", "run_6_step_12").
spawned_by("run_6", "run_6_step_13", "run_6_step_12").

% === Job: Travel Updates Last 6 Hours  — failed ===
job("run_7").
job_status("run_7", "failed").
job_title("run_7", "Travel Updates Last 6 Hours").
call("run_7", "run_7_step_1", "prompt_user", "completed").
call("run_7", "run_7_step_2", "llm_generate", "completed").
depends("run_7", "run_7_step_2", "run_7_step_1").
call("run_7", "run_7_step_3", "sheets_read", "completed").
depends("run_7", "run_7_step_3", "run_7_step_2").
spawned_by("run_7", "run_7_step_3", "run_7_step_2").
param("run_7", "run_7_step_3", "spreadsheet", "spreadsheet_1").
param("run_7", "run_7_step_3", "account", "acct_3").
call("run_7", "run_7_step_4", "gmail_search", "failed").
depends("run_7", "run_7_step_4", "run_7_step_2").
spawned_by("run_7", "run_7_step_4", "run_7_step_2").
param("run_7", "run_7_step_4", "query", "after:2026/03/27 travel flight hotel booking confirmation").
param("run_7", "run_7_step_4", "account", "acct_1").
param("run_7", "run_7_step_4", "max_results", "20").
call("run_7", "run_7_step_5", "llm_generate", "completed").
depends("run_7", "run_7_step_5", "run_7_step_4").
depends("run_7", "run_7_step_5", "run_7_step_3").
call("run_7", "run_7_step_6", "gmail_search", "completed").
depends("run_7", "run_7_step_6", "run_7_step_5").
spawned_by("run_7", "run_7_step_6", "run_7_step_5").
param("run_7", "run_7_step_6", "query", "travel flight hotel booking confirmation").
param("run_7", "run_7_step_6", "account", "acct_4").
param("run_7", "run_7_step_6", "max_results", "20").
call("run_7", "run_7_step_7", "gmail_search", "completed").
depends("run_7", "run_7_step_7", "run_7_step_5").
spawned_by("run_7", "run_7_step_7", "run_7_step_5").
param("run_7", "run_7_step_7", "query", "travel flight hotel booking confirmation").
param("run_7", "run_7_step_7", "account", "acct_5").
param("run_7", "run_7_step_7", "max_results", "20").
call("run_7", "run_7_step_8", "gmail_search", "completed").
depends("run_7", "run_7_step_8", "run_7_step_5").
spawned_by("run_7", "run_7_step_8", "run_7_step_5").
param("run_7", "run_7_step_8", "query", "travel flight hotel booking confirmation").
param("run_7", "run_7_step_8", "account", "acct_2").
param("run_7", "run_7_step_8", "max_results", "20").
call("run_7", "run_7_step_9", "llm_generate", "completed").
depends("run_7", "run_7_step_9", "run_7_step_7").
depends("run_7", "run_7_step_9", "run_7_step_8").
depends("run_7", "run_7_step_9", "run_7_step_6").
call("run_7", "run_7_step_10", "gmail_search", "completed").
depends("run_7", "run_7_step_10", "run_7_step_9").
spawned_by("run_7", "run_7_step_10", "run_7_step_9").
param("run_7", "run_7_step_10", "query", "after:2026/03/27").
param("run_7", "run_7_step_10", "account", "acct_4").
param("run_7", "run_7_step_10", "max_results", "30").
call("run_7", "run_7_step_11", "gmail_search", "completed").
depends("run_7", "run_7_step_11", "run_7_step_9").
spawned_by("run_7", "run_7_step_11", "run_7_step_9").
param("run_7", "run_7_step_11", "query", "after:2026/03/27").
param("run_7", "run_7_step_11", "account", "acct_5").
param("run_7", "run_7_step_11", "max_results", "30").
call("run_7", "run_7_step_12", "llm_generate", "completed").
depends("run_7", "run_7_step_12", "run_7_step_11").
depends("run_7", "run_7_step_12", "run_7_step_10").
call("run_7", "run_7_step_13", "notify_user", "completed").
depends("run_7", "run_7_step_13", "run_7_step_12").
spawned_by("run_7", "run_7_step_13", "run_7_step_12").

% === Job: Travel Updates Last 6 Hours  — failed ===
job("run_8").
job_status("run_8", "failed").
job_title("run_8", "Travel Updates Last 6 Hours").
call("run_8", "run_8_step_1", "prompt_user", "completed").
call("run_8", "run_8_step_2", "llm_generate", "completed").
depends("run_8", "run_8_step_2", "run_8_step_1").
call("run_8", "run_8_step_3", "gmail_search", "completed").
depends("run_8", "run_8_step_3", "run_8_step_2").
spawned_by("run_8", "run_8_step_3", "run_8_step_2").
param("run_8", "run_8_step_3", "query", "after:2026/03/27 travel flight hotel booking confirmation").
param("run_8", "run_8_step_3", "account", "acct_2").
param("run_8", "run_8_step_3", "max_results", "20").
call("run_8", "run_8_step_4", "gmail_search", "completed").
depends("run_8", "run_8_step_4", "run_8_step_2").
spawned_by("run_8", "run_8_step_4", "run_8_step_2").
param("run_8", "run_8_step_4", "query", "after:2026/03/27 travel flight hotel booking confirmation").
param("run_8", "run_8_step_4", "account", "acct_4").
param("run_8", "run_8_step_4", "max_results", "20").
call("run_8", "run_8_step_5", "gmail_search", "failed").
depends("run_8", "run_8_step_5", "run_8_step_2").
spawned_by("run_8", "run_8_step_5", "run_8_step_2").
param("run_8", "run_8_step_5", "query", "after:2026/03/27 travel flight hotel booking confirmation").
param("run_8", "run_8_step_5", "account", "acct_1").
param("run_8", "run_8_step_5", "max_results", "20").
call("run_8", "run_8_step_6", "llm_generate", "completed").
depends("run_8", "run_8_step_6", "run_8_step_5").
depends("run_8", "run_8_step_6", "run_8_step_3").
depends("run_8", "run_8_step_6", "run_8_step_4").
call("run_8", "run_8_step_7", "gmail_list_threads", "completed").
depends("run_8", "run_8_step_7", "run_8_step_6").
spawned_by("run_8", "run_8_step_7", "run_8_step_6").
param("run_8", "run_8_step_7", "account", "acct_2").
param("run_8", "run_8_step_7", "labels", "INBOX").
call("run_8", "run_8_step_8", "gmail_list_threads", "completed").
depends("run_8", "run_8_step_8", "run_8_step_6").
spawned_by("run_8", "run_8_step_8", "run_8_step_6").
param("run_8", "run_8_step_8", "account", "acct_4").
param("run_8", "run_8_step_8", "labels", "INBOX").
call("run_8", "run_8_step_9", "gmail_list_threads", "completed").
depends("run_8", "run_8_step_9", "run_8_step_6").
spawned_by("run_8", "run_8_step_9", "run_8_step_6").
param("run_8", "run_8_step_9", "account", "acct_5").
param("run_8", "run_8_step_9", "labels", "INBOX").
call("run_8", "run_8_step_10", "llm_generate", "completed").
depends("run_8", "run_8_step_10", "run_8_step_7").
depends("run_8", "run_8_step_10", "run_8_step_8").
depends("run_8", "run_8_step_10", "run_8_step_9").
call("run_8", "run_8_step_11", "gmail_get_thread", "completed").
depends("run_8", "run_8_step_11", "run_8_step_10").
spawned_by("run_8", "run_8_step_11", "run_8_step_10").
param("run_8", "run_8_step_11", "thread_id", "thread_4").
param("run_8", "run_8_step_11", "account", "acct_5").
call("run_8", "run_8_step_12", "gmail_get_thread", "completed").
depends("run_8", "run_8_step_12", "run_8_step_10").
spawned_by("run_8", "run_8_step_12", "run_8_step_10").
param("run_8", "run_8_step_12", "thread_id", "thread_5").
param("run_8", "run_8_step_12", "account", "acct_5").
call("run_8", "run_8_step_13", "gmail_get_thread", "completed").
depends("run_8", "run_8_step_13", "run_8_step_10").
spawned_by("run_8", "run_8_step_13", "run_8_step_10").
param("run_8", "run_8_step_13", "thread_id", "thread_6").
param("run_8", "run_8_step_13", "account", "acct_5").
call("run_8", "run_8_step_14", "gmail_get_thread", "completed").
depends("run_8", "run_8_step_14", "run_8_step_10").
spawned_by("run_8", "run_8_step_14", "run_8_step_10").
param("run_8", "run_8_step_14", "thread_id", "thread_7").
param("run_8", "run_8_step_14", "account", "acct_5").
call("run_8", "run_8_step_15", "gmail_get_thread", "completed").
depends("run_8", "run_8_step_15", "run_8_step_10").
spawned_by("run_8", "run_8_step_15", "run_8_step_10").
param("run_8", "run_8_step_15", "thread_id", "thread_8").
param("run_8", "run_8_step_15", "account", "acct_5").
call("run_8", "run_8_step_16", "llm_generate", "completed").
depends("run_8", "run_8_step_16", "run_8_step_13").
depends("run_8", "run_8_step_16", "run_8_step_14").
depends("run_8", "run_8_step_16", "run_8_step_15").
depends("run_8", "run_8_step_16", "run_8_step_12").
depends("run_8", "run_8_step_16", "run_8_step_11").
call("run_8", "run_8_step_17", "sheets_read", "completed").
depends("run_8", "run_8_step_17", "run_8_step_16").
spawned_by("run_8", "run_8_step_17", "run_8_step_16").
param("run_8", "run_8_step_17", "spreadsheet", "spreadsheet_1").
param("run_8", "run_8_step_17", "account", "acct_3").
call("run_8", "run_8_step_18", "gmail_get_attachment", "completed").
depends("run_8", "run_8_step_18", "run_8_step_16").
spawned_by("run_8", "run_8_step_18", "run_8_step_16").
param("run_8", "run_8_step_18", "account", "acct_5").
call("run_8", "run_8_step_19", "sheets_read", "completed").
depends("run_8", "run_8_step_19", "run_8_step_16").
spawned_by("run_8", "run_8_step_19", "run_8_step_16").
param("run_8", "run_8_step_19", "spreadsheet", "spreadsheet_2").
param("run_8", "run_8_step_19", "account", "acct_3").
call("run_8", "run_8_step_20", "llm_generate", "completed").
depends("run_8", "run_8_step_20", "run_8_step_18").
depends("run_8", "run_8_step_20", "run_8_step_17").
depends("run_8", "run_8_step_20", "run_8_step_19").
call("run_8", "run_8_step_21", "notify_user", "completed").
depends("run_8", "run_8_step_21", "run_8_step_20").
spawned_by("run_8", "run_8_step_21", "run_8_step_20").

% === Job: Travel Updates Last 6 Hours  — failed ===
job("run_9").
job_status("run_9", "failed").
job_title("run_9", "Travel Updates Last 6 Hours").
call("run_9", "run_9_step_1", "prompt_user", "completed").
call("run_9", "run_9_step_2", "llm_generate", "completed").
depends("run_9", "run_9_step_2", "run_9_step_1").
call("run_9", "run_9_step_3", "gmail_search", "completed").
depends("run_9", "run_9_step_3", "run_9_step_2").
spawned_by("run_9", "run_9_step_3", "run_9_step_2").
param("run_9", "run_9_step_3", "query", "after:2026/03/27 travel flight hotel booking confirmation").
param("run_9", "run_9_step_3", "account", "acct_2").
param("run_9", "run_9_step_3", "max_results", "20").
call("run_9", "run_9_step_4", "gmail_search", "failed").
depends("run_9", "run_9_step_4", "run_9_step_2").
spawned_by("run_9", "run_9_step_4", "run_9_step_2").
param("run_9", "run_9_step_4", "query", "after:2026/03/27 travel flight hotel booking confirmation").
param("run_9", "run_9_step_4", "account", "acct_1").
param("run_9", "run_9_step_4", "max_results", "20").
call("run_9", "run_9_step_5", "llm_generate", "completed").
depends("run_9", "run_9_step_5", "run_9_step_4").
depends("run_9", "run_9_step_5", "run_9_step_3").
call("run_9", "run_9_step_6", "sheets_read", "completed").
depends("run_9", "run_9_step_6", "run_9_step_5").
spawned_by("run_9", "run_9_step_6", "run_9_step_5").
param("run_9", "run_9_step_6", "spreadsheet", "spreadsheet_1").
param("run_9", "run_9_step_6", "account", "acct_3").
call("run_9", "run_9_step_7", "gmail_search", "completed").
depends("run_9", "run_9_step_7", "run_9_step_5").
spawned_by("run_9", "run_9_step_7", "run_9_step_5").
param("run_9", "run_9_step_7", "query", "travel flight hotel booking confirmation").
param("run_9", "run_9_step_7", "account", "acct_5").
param("run_9", "run_9_step_7", "max_results", "20").
call("run_9", "run_9_step_8", "gmail_search", "completed").
depends("run_9", "run_9_step_8", "run_9_step_5").
spawned_by("run_9", "run_9_step_8", "run_9_step_5").
param("run_9", "run_9_step_8", "query", "travel flight hotel booking confirmation").
param("run_9", "run_9_step_8", "account", "acct_4").
param("run_9", "run_9_step_8", "max_results", "20").
call("run_9", "run_9_step_9", "llm_generate", "completed").
depends("run_9", "run_9_step_9", "run_9_step_8").
depends("run_9", "run_9_step_9", "run_9_step_7").
depends("run_9", "run_9_step_9", "run_9_step_6").
call("run_9", "run_9_step_10", "gmail_list_threads", "completed").
depends("run_9", "run_9_step_10", "run_9_step_9").
spawned_by("run_9", "run_9_step_10", "run_9_step_9").
param("run_9", "run_9_step_10", "account", "acct_2").
param("run_9", "run_9_step_10", "labels", "INBOX").
call("run_9", "run_9_step_11", "gmail_list_threads", "completed").
depends("run_9", "run_9_step_11", "run_9_step_9").
spawned_by("run_9", "run_9_step_11", "run_9_step_9").
param("run_9", "run_9_step_11", "account", "acct_5").
param("run_9", "run_9_step_11", "labels", "INBOX").
call("run_9", "run_9_step_12", "gmail_list_threads", "completed").
depends("run_9", "run_9_step_12", "run_9_step_9").
spawned_by("run_9", "run_9_step_12", "run_9_step_9").
param("run_9", "run_9_step_12", "account", "acct_4").
param("run_9", "run_9_step_12", "labels", "INBOX").
call("run_9", "run_9_step_13", "llm_generate", "completed").
depends("run_9", "run_9_step_13", "run_9_step_11").
depends("run_9", "run_9_step_13", "run_9_step_12").
depends("run_9", "run_9_step_13", "run_9_step_10").
call("run_9", "run_9_step_14", "gmail_search", "completed").
depends("run_9", "run_9_step_14", "run_9_step_13").
spawned_by("run_9", "run_9_step_14", "run_9_step_13").
param("run_9", "run_9_step_14", "query", "after:2026/03/27 (flight OR hotel OR booking OR airbnb OR confirmation OR itinerary OR check-in)").
param("run_9", "run_9_step_14", "account", "acct_2").
param("run_9", "run_9_step_14", "max_results", "20").
call("run_9", "run_9_step_15", "gmail_search", "completed").
depends("run_9", "run_9_step_15", "run_9_step_13").
spawned_by("run_9", "run_9_step_15", "run_9_step_13").
param("run_9", "run_9_step_15", "query", "after:2026/03/27 (flight OR hotel OR booking OR airbnb OR confirmation OR itinerary OR check-in)").
param("run_9", "run_9_step_15", "account", "acct_4").
param("run_9", "run_9_step_15", "max_results", "20").
call("run_9", "run_9_step_16", "gmail_search", "completed").
depends("run_9", "run_9_step_16", "run_9_step_13").
spawned_by("run_9", "run_9_step_16", "run_9_step_13").
param("run_9", "run_9_step_16", "query", "after:2026/03/27 (flight OR hotel OR booking OR airbnb OR confirmation OR itinerary OR check-in OR checkin OR visa)").
param("run_9", "run_9_step_16", "account", "acct_5").
param("run_9", "run_9_step_16", "max_results", "20").
call("run_9", "run_9_step_17", "gmail_get_thread", "completed").
depends("run_9", "run_9_step_17", "run_9_step_13").
spawned_by("run_9", "run_9_step_17", "run_9_step_13").
param("run_9", "run_9_step_17", "thread_id", "thread_5").
param("run_9", "run_9_step_17", "account", "acct_5").
call("run_9", "run_9_step_18", "llm_generate", "completed").
depends("run_9", "run_9_step_18", "run_9_step_17").
depends("run_9", "run_9_step_18", "run_9_step_16").
depends("run_9", "run_9_step_18", "run_9_step_15").
depends("run_9", "run_9_step_18", "run_9_step_14").
call("run_9", "run_9_step_19", "notify_user", "completed").
depends("run_9", "run_9_step_19", "run_9_step_18").
spawned_by("run_9", "run_9_step_19", "run_9_step_18").

% === Job: Travel Updates Last 6 Hours  — failed ===
job("run_10").
job_status("run_10", "failed").
job_title("run_10", "Travel Updates Last 6 Hours").
call("run_10", "run_10_step_1", "prompt_user", "completed").
call("run_10", "run_10_step_2", "llm_generate", "completed").
depends("run_10", "run_10_step_2", "run_10_step_1").
call("run_10", "run_10_step_3", "sheets_get_spreadsheet", "completed").
depends("run_10", "run_10_step_3", "run_10_step_2").
spawned_by("run_10", "run_10_step_3", "run_10_step_2").
param("run_10", "run_10_step_3", "spreadsheet", "spreadsheet_2").
param("run_10", "run_10_step_3", "account", "acct_6").
call("run_10", "run_10_step_4", "gmail_search", "failed").
depends("run_10", "run_10_step_4", "run_10_step_2").
spawned_by("run_10", "run_10_step_4", "run_10_step_2").
param("run_10", "run_10_step_4", "query", "after:2026/03/27").
param("run_10", "run_10_step_4", "account", "acct_1").
param("run_10", "run_10_step_4", "max_results", "30").
call("run_10", "run_10_step_5", "sheets_get_spreadsheet", "completed").
depends("run_10", "run_10_step_5", "run_10_step_2").
spawned_by("run_10", "run_10_step_5", "run_10_step_2").
param("run_10", "run_10_step_5", "spreadsheet", "spreadsheet_1").
param("run_10", "run_10_step_5", "account", "acct_6").
call("run_10", "run_10_step_6", "gmail_search", "completed").
depends("run_10", "run_10_step_6", "run_10_step_2").
spawned_by("run_10", "run_10_step_6", "run_10_step_2").
param("run_10", "run_10_step_6", "query", "after:2026/03/27").
param("run_10", "run_10_step_6", "account", "acct_2").
param("run_10", "run_10_step_6", "max_results", "30").
call("run_10", "run_10_step_7", "llm_generate", "completed").
depends("run_10", "run_10_step_7", "run_10_step_4").
depends("run_10", "run_10_step_7", "run_10_step_6").
depends("run_10", "run_10_step_7", "run_10_step_5").
depends("run_10", "run_10_step_7", "run_10_step_3").
call("run_10", "run_10_step_8", "sheets_read", "completed").
depends("run_10", "run_10_step_8", "run_10_step_7").
spawned_by("run_10", "run_10_step_8", "run_10_step_7").
param("run_10", "run_10_step_8", "spreadsheet", "spreadsheet_1").
param("run_10", "run_10_step_8", "account", "acct_6").
param("run_10", "run_10_step_8", "sheet_name", "Flights").
call("run_10", "run_10_step_9", "sheets_read", "completed").
depends("run_10", "run_10_step_9", "run_10_step_7").
spawned_by("run_10", "run_10_step_9", "run_10_step_7").
param("run_10", "run_10_step_9", "spreadsheet", "spreadsheet_2").
param("run_10", "run_10_step_9", "account", "acct_6").
param("run_10", "run_10_step_9", "sheet_name", "🏨 Hotels").
call("run_10", "run_10_step_10", "gmail_search", "completed").
depends("run_10", "run_10_step_10", "run_10_step_7").
spawned_by("run_10", "run_10_step_10", "run_10_step_7").
param("run_10", "run_10_step_10", "query", "after:2026/03/27").
param("run_10", "run_10_step_10", "account", "acct_4").
param("run_10", "run_10_step_10", "max_results", "30").
call("run_10", "run_10_step_11", "llm_generate", "completed").
depends("run_10", "run_10_step_11", "run_10_step_10").
depends("run_10", "run_10_step_11", "run_10_step_8").
depends("run_10", "run_10_step_11", "run_10_step_9").
call("run_10", "run_10_step_12", "notify_user", "completed").
depends("run_10", "run_10_step_12", "run_10_step_11").
spawned_by("run_10", "run_10_step_12", "run_10_step_11").

% === Job: Travel Updates Last 6 Hours  — failed ===
job("run_11").
job_status("run_11", "failed").
job_title("run_11", "Travel Updates Last 6 Hours").
call("run_11", "run_11_step_1", "prompt_user", "completed").
call("run_11", "run_11_step_2", "llm_generate", "completed").
depends("run_11", "run_11_step_2", "run_11_step_1").
call("run_11", "run_11_step_3", "gmail_search", "completed").
depends("run_11", "run_11_step_3", "run_11_step_2").
spawned_by("run_11", "run_11_step_3", "run_11_step_2").
param("run_11", "run_11_step_3", "query", "after:2026/03/26 travel flight hotel booking reservation").
param("run_11", "run_11_step_3", "account", "acct_2").
param("run_11", "run_11_step_3", "max_results", "20").
call("run_11", "run_11_step_4", "gmail_search", "failed").
depends("run_11", "run_11_step_4", "run_11_step_2").
spawned_by("run_11", "run_11_step_4", "run_11_step_2").
param("run_11", "run_11_step_4", "query", "after:2026/03/26 travel flight hotel booking reservation").
param("run_11", "run_11_step_4", "account", "acct_1").
param("run_11", "run_11_step_4", "max_results", "20").
call("run_11", "run_11_step_5", "llm_generate", "completed").
depends("run_11", "run_11_step_5", "run_11_step_4").
depends("run_11", "run_11_step_5", "run_11_step_3").
call("run_11", "run_11_step_6", "gmail_search", "completed").
depends("run_11", "run_11_step_6", "run_11_step_5").
spawned_by("run_11", "run_11_step_6", "run_11_step_5").
param("run_11", "run_11_step_6", "query", "after:2026/03/26 travel flight hotel booking reservation").
param("run_11", "run_11_step_6", "account", "acct_4").
param("run_11", "run_11_step_6", "max_results", "20").
call("run_11", "run_11_step_7", "gmail_search", "completed").
depends("run_11", "run_11_step_7", "run_11_step_5").
spawned_by("run_11", "run_11_step_7", "run_11_step_5").
param("run_11", "run_11_step_7", "query", "after:2026/03/26 travel flight hotel booking reservation").
param("run_11", "run_11_step_7", "account", "acct_5").
param("run_11", "run_11_step_7", "max_results", "20").
call("run_11", "run_11_step_8", "llm_generate", "completed").
depends("run_11", "run_11_step_8", "run_11_step_6").
depends("run_11", "run_11_step_8", "run_11_step_7").
call("run_11", "run_11_step_9", "gmail_list_threads", "completed").
depends("run_11", "run_11_step_9", "run_11_step_8").
spawned_by("run_11", "run_11_step_9", "run_11_step_8").
param("run_11", "run_11_step_9", "account", "acct_2").
call("run_11", "run_11_step_10", "gmail_list_threads", "completed").
depends("run_11", "run_11_step_10", "run_11_step_8").
spawned_by("run_11", "run_11_step_10", "run_11_step_8").
param("run_11", "run_11_step_10", "account", "acct_4").
call("run_11", "run_11_step_11", "gmail_list_threads", "completed").
depends("run_11", "run_11_step_11", "run_11_step_8").
spawned_by("run_11", "run_11_step_11", "run_11_step_8").
param("run_11", "run_11_step_11", "account", "acct_5").
call("run_11", "run_11_step_12", "llm_generate", "completed").
depends("run_11", "run_11_step_12", "run_11_step_10").
depends("run_11", "run_11_step_12", "run_11_step_11").
depends("run_11", "run_11_step_12", "run_11_step_9").
call("run_11", "run_11_step_13", "notify_user", "completed").
depends("run_11", "run_11_step_13", "run_11_step_12").
spawned_by("run_11", "run_11_step_13", "run_11_step_12").

% === Job: Travel Updates Last 6 Hours  — failed ===
job("run_12").
job_status("run_12", "failed").
job_title("run_12", "Travel Updates Last 6 Hours").
call("run_12", "run_12_step_1", "prompt_user", "completed").
call("run_12", "run_12_step_2", "llm_generate", "completed").
depends("run_12", "run_12_step_2", "run_12_step_1").
call("run_12", "run_12_step_3", "gmail_search", "completed").
depends("run_12", "run_12_step_3", "run_12_step_2").
spawned_by("run_12", "run_12_step_3", "run_12_step_2").
param("run_12", "run_12_step_3", "query", "after:2026/03/26").
param("run_12", "run_12_step_3", "account", "acct_2").
param("run_12", "run_12_step_3", "max_results", "30").
call("run_12", "run_12_step_4", "gmail_search", "failed").
depends("run_12", "run_12_step_4", "run_12_step_2").
spawned_by("run_12", "run_12_step_4", "run_12_step_2").
param("run_12", "run_12_step_4", "query", "after:2026/03/26").
param("run_12", "run_12_step_4", "account", "acct_1").
param("run_12", "run_12_step_4", "max_results", "30").
call("run_12", "run_12_step_5", "llm_generate", "completed").
depends("run_12", "run_12_step_5", "run_12_step_4").
depends("run_12", "run_12_step_5", "run_12_step_3").
call("run_12", "run_12_step_6", "gmail_search", "completed").
depends("run_12", "run_12_step_6", "run_12_step_5").
spawned_by("run_12", "run_12_step_6", "run_12_step_5").
param("run_12", "run_12_step_6", "query", "after:2026/03/26").
param("run_12", "run_12_step_6", "account", "acct_4").
param("run_12", "run_12_step_6", "max_results", "30").
call("run_12", "run_12_step_7", "gmail_search", "completed").
depends("run_12", "run_12_step_7", "run_12_step_5").
spawned_by("run_12", "run_12_step_7", "run_12_step_5").
param("run_12", "run_12_step_7", "query", "(flight OR hotel OR booking OR airbnb OR airline OR reservation OR visa OR travel OR itinerary) after:2026/03/26").
param("run_12", "run_12_step_7", "account", "acct_2").
param("run_12", "run_12_step_7", "max_results", "20").
call("run_12", "run_12_step_8", "llm_generate", "completed").
depends("run_12", "run_12_step_8", "run_12_step_6").
depends("run_12", "run_12_step_8", "run_12_step_7").
call("run_12", "run_12_step_9", "notify_user", "completed").
depends("run_12", "run_12_step_9", "run_12_step_8").
spawned_by("run_12", "run_12_step_9", "run_12_step_8").

% === Job: Travel Updates Last 6 Hours  — failed ===
job("run_13").
job_status("run_13", "failed").
job_title("run_13", "Travel Updates Last 6 Hours").
call("run_13", "run_13_step_1", "prompt_user", "completed").
call("run_13", "run_13_step_2", "llm_generate", "completed").
depends("run_13", "run_13_step_2", "run_13_step_1").
call("run_13", "run_13_step_3", "gmail_search", "completed").
depends("run_13", "run_13_step_3", "run_13_step_2").
spawned_by("run_13", "run_13_step_3", "run_13_step_2").
param("run_13", "run_13_step_3", "query", "after:2026/03/26 travel flight hotel booking confirmation").
param("run_13", "run_13_step_3", "account", "acct_2").
param("run_13", "run_13_step_3", "max_results", "20").
call("run_13", "run_13_step_4", "gmail_search", "failed").
depends("run_13", "run_13_step_4", "run_13_step_2").
spawned_by("run_13", "run_13_step_4", "run_13_step_2").
param("run_13", "run_13_step_4", "query", "after:2026/03/26 travel flight hotel booking confirmation").
param("run_13", "run_13_step_4", "account", "acct_1").
param("run_13", "run_13_step_4", "max_results", "20").
call("run_13", "run_13_step_5", "llm_generate", "completed").
depends("run_13", "run_13_step_5", "run_13_step_4").
depends("run_13", "run_13_step_5", "run_13_step_3").
call("run_13", "run_13_step_6", "gmail_search", "completed").
depends("run_13", "run_13_step_6", "run_13_step_5").
spawned_by("run_13", "run_13_step_6", "run_13_step_5").
param("run_13", "run_13_step_6", "query", "travel flight hotel booking confirmation").
param("run_13", "run_13_step_6", "account", "acct_5").
param("run_13", "run_13_step_6", "max_results", "20").
call("run_13", "run_13_step_7", "gmail_search", "completed").
depends("run_13", "run_13_step_7", "run_13_step_5").
spawned_by("run_13", "run_13_step_7", "run_13_step_5").
param("run_13", "run_13_step_7", "query", "travel flight hotel booking confirmation").
param("run_13", "run_13_step_7", "account", "acct_4").
param("run_13", "run_13_step_7", "max_results", "20").
call("run_13", "run_13_step_8", "sheets_read", "completed").
depends("run_13", "run_13_step_8", "run_13_step_5").
spawned_by("run_13", "run_13_step_8", "run_13_step_5").
param("run_13", "run_13_step_8", "spreadsheet", "spreadsheet_1").
param("run_13", "run_13_step_8", "account", "acct_3").
call("run_13", "run_13_step_9", "llm_generate", "completed").
depends("run_13", "run_13_step_9", "run_13_step_7").
depends("run_13", "run_13_step_9", "run_13_step_6").
depends("run_13", "run_13_step_9", "run_13_step_8").
call("run_13", "run_13_step_10", "gmail_search", "completed").
depends("run_13", "run_13_step_10", "run_13_step_9").
spawned_by("run_13", "run_13_step_10", "run_13_step_9").
param("run_13", "run_13_step_10", "query", "after:2026/03/26 subject:(flight OR hotel OR booking OR confirmation OR itinerary OR check-in OR travel)").
param("run_13", "run_13_step_10", "account", "acct_5").
param("run_13", "run_13_step_10", "max_results", "20").
call("run_13", "run_13_step_11", "gmail_search", "completed").
depends("run_13", "run_13_step_11", "run_13_step_9").
spawned_by("run_13", "run_13_step_11", "run_13_step_9").
param("run_13", "run_13_step_11", "query", "after:2026/03/26").
param("run_13", "run_13_step_11", "account", "acct_4").
param("run_13", "run_13_step_11", "max_results", "20").
call("run_13", "run_13_step_12", "llm_generate", "completed").
depends("run_13", "run_13_step_12", "run_13_step_10").
depends("run_13", "run_13_step_12", "run_13_step_11").
call("run_13", "run_13_step_13", "sheets_get_spreadsheet", "completed").
depends("run_13", "run_13_step_13", "run_13_step_12").
spawned_by("run_13", "run_13_step_13", "run_13_step_12").
param("run_13", "run_13_step_13", "spreadsheet", "spreadsheet_2").
param("run_13", "run_13_step_13", "account", "acct_3").
call("run_13", "run_13_step_14", "gmail_get_thread", "completed").
depends("run_13", "run_13_step_14", "run_13_step_12").
spawned_by("run_13", "run_13_step_14", "run_13_step_12").
param("run_13", "run_13_step_14", "thread_id", "thread_7").
param("run_13", "run_13_step_14", "account", "acct_5").
call("run_13", "run_13_step_15", "gmail_get_thread", "completed").
depends("run_13", "run_13_step_15", "run_13_step_12").
spawned_by("run_13", "run_13_step_15", "run_13_step_12").
param("run_13", "run_13_step_15", "thread_id", "thread_8").
param("run_13", "run_13_step_15", "account", "acct_5").
call("run_13", "run_13_step_16", "llm_generate", "completed").
depends("run_13", "run_13_step_16", "run_13_step_14").
depends("run_13", "run_13_step_16", "run_13_step_15").
depends("run_13", "run_13_step_16", "run_13_step_13").
call("run_13", "run_13_step_17", "sheets_read", "completed").
depends("run_13", "run_13_step_17", "run_13_step_16").
spawned_by("run_13", "run_13_step_17", "run_13_step_16").
param("run_13", "run_13_step_17", "spreadsheet", "spreadsheet_2").
param("run_13", "run_13_step_17", "account", "acct_3").
param("run_13", "run_13_step_17", "sheet_name", "🏨 Hotels").
call("run_13", "run_13_step_18", "llm_look_at", "completed").
depends("run_13", "run_13_step_18", "run_13_step_16").
spawned_by("run_13", "run_13_step_18", "run_13_step_16").
param("run_13", "run_13_step_18", "type", "attachment_inspection").
call("run_13", "run_13_step_19", "llm_generate", "completed").
depends("run_13", "run_13_step_19", "run_13_step_18").
depends("run_13", "run_13_step_19", "run_13_step_17").
call("run_13", "run_13_step_20", "sheets_update_values", "completed").
depends("run_13", "run_13_step_20", "run_13_step_19").
spawned_by("run_13", "run_13_step_20", "run_13_step_19").
param("run_13", "run_13_step_20", "spreadsheet", "spreadsheet_2").
param("run_13", "run_13_step_20", "account", "acct_3").
param("run_13", "run_13_step_20", "range", "🏨 Hotels!H2:J2").
param("run_13", "run_13_step_20", "target_sheet", "🏨 Hotels").
param("run_13", "run_13_step_20", "target_col_start", "H").
param("run_13", "run_13_step_20", "target_col_end", "J").
param("run_13", "run_13_step_20", "target_row_start", "2").
param("run_13", "run_13_step_20", "target_row_end", "2").
param("run_13", "run_13_step_20", "target_col_count", "3").
param("run_13", "run_13_step_20", "target_row_count", "1").
param("run_13", "run_13_step_20", "value_type", "detail_fill").
param("run_13", "run_13_step_20", "value_row_count", "1").
param("run_13", "run_13_step_20", "value_col_count", "3").
call("run_13", "run_13_step_21", "llm_generate", "completed").
depends("run_13", "run_13_step_21", "run_13_step_20").
call("run_13", "run_13_step_22", "notify_user", "completed").
depends("run_13", "run_13_step_22", "run_13_step_21").
spawned_by("run_13", "run_13_step_22", "run_13_step_21").

% === Job: Travel Updates Last 6 Hours  — failed ===
job("run_14").
job_status("run_14", "failed").
job_title("run_14", "Travel Updates Last 6 Hours").
call("run_14", "run_14_step_1", "prompt_user", "completed").
call("run_14", "run_14_step_2", "llm_generate", "completed").
depends("run_14", "run_14_step_2", "run_14_step_1").
call("run_14", "run_14_step_3", "gmail_search", "completed").
depends("run_14", "run_14_step_3", "run_14_step_2").
spawned_by("run_14", "run_14_step_3", "run_14_step_2").
param("run_14", "run_14_step_3", "query", "travel flight hotel booking confirmation").
param("run_14", "run_14_step_3", "account", "acct_2").
param("run_14", "run_14_step_3", "max_results", "20").
call("run_14", "run_14_step_4", "gmail_search", "failed").
depends("run_14", "run_14_step_4", "run_14_step_2").
spawned_by("run_14", "run_14_step_4", "run_14_step_2").
param("run_14", "run_14_step_4", "query", "travel flight hotel booking confirmation").
param("run_14", "run_14_step_4", "account", "acct_1").
param("run_14", "run_14_step_4", "max_results", "20").
call("run_14", "run_14_step_5", "llm_generate", "completed").
depends("run_14", "run_14_step_5", "run_14_step_4").
depends("run_14", "run_14_step_5", "run_14_step_3").
call("run_14", "run_14_step_6", "gmail_search", "completed").
depends("run_14", "run_14_step_6", "run_14_step_5").
spawned_by("run_14", "run_14_step_6", "run_14_step_5").
param("run_14", "run_14_step_6", "query", "after:2026/03/26 flight OR hotel OR booking OR reservation OR airbnb OR confirmation").
param("run_14", "run_14_step_6", "account", "acct_2").
param("run_14", "run_14_step_6", "max_results", "20").
call("run_14", "run_14_step_7", "gmail_search", "completed").
depends("run_14", "run_14_step_7", "run_14_step_5").
spawned_by("run_14", "run_14_step_7", "run_14_step_5").
param("run_14", "run_14_step_7", "query", "after:2026/03/26 flight OR hotel OR booking OR reservation OR airbnb OR confirmation").
param("run_14", "run_14_step_7", "account", "acct_4").
param("run_14", "run_14_step_7", "max_results", "20").
call("run_14", "run_14_step_8", "llm_generate", "completed").
depends("run_14", "run_14_step_8", "run_14_step_6").
depends("run_14", "run_14_step_8", "run_14_step_7").
call("run_14", "run_14_step_9", "gmail_search", "completed").
depends("run_14", "run_14_step_9", "run_14_step_8").
spawned_by("run_14", "run_14_step_9", "run_14_step_8").
param("run_14", "run_14_step_9", "query", "after:2026/03/26").
param("run_14", "run_14_step_9", "account", "acct_4").
param("run_14", "run_14_step_9", "max_results", "30").
call("run_14", "run_14_step_10", "gmail_search", "completed").
depends("run_14", "run_14_step_10", "run_14_step_8").
spawned_by("run_14", "run_14_step_10", "run_14_step_8").
param("run_14", "run_14_step_10", "query", "after:2026/03/26").
param("run_14", "run_14_step_10", "account", "acct_2").
param("run_14", "run_14_step_10", "max_results", "30").
call("run_14", "run_14_step_11", "llm_generate", "completed").
depends("run_14", "run_14_step_11", "run_14_step_10").
depends("run_14", "run_14_step_11", "run_14_step_9").
call("run_14", "run_14_step_12", "notify_user", "completed").
depends("run_14", "run_14_step_12", "run_14_step_11").
spawned_by("run_14", "run_14_step_12", "run_14_step_11").

% === Job: Travel Updates Last 6 Hours  — failed ===
job("run_15").
job_status("run_15", "failed").
job_title("run_15", "Travel Updates Last 6 Hours").
call("run_15", "run_15_step_1", "prompt_user", "completed").
call("run_15", "run_15_step_2", "llm_generate", "completed").
depends("run_15", "run_15_step_2", "run_15_step_1").
call("run_15", "run_15_step_3", "gmail_search", "completed").
depends("run_15", "run_15_step_3", "run_15_step_2").
spawned_by("run_15", "run_15_step_3", "run_15_step_2").
param("run_15", "run_15_step_3", "query", "after:2026/03/26 travel flight hotel booking confirmation").
param("run_15", "run_15_step_3", "account", "acct_2").
param("run_15", "run_15_step_3", "max_results", "20").
call("run_15", "run_15_step_4", "gmail_search", "failed").
depends("run_15", "run_15_step_4", "run_15_step_2").
spawned_by("run_15", "run_15_step_4", "run_15_step_2").
param("run_15", "run_15_step_4", "query", "after:2026/03/26 travel flight hotel booking confirmation").
param("run_15", "run_15_step_4", "account", "acct_1").
param("run_15", "run_15_step_4", "max_results", "20").
call("run_15", "run_15_step_5", "llm_generate", "completed").
depends("run_15", "run_15_step_5", "run_15_step_4").
depends("run_15", "run_15_step_5", "run_15_step_3").
call("run_15", "run_15_step_6", "gmail_search", "completed").
depends("run_15", "run_15_step_6", "run_15_step_5").
spawned_by("run_15", "run_15_step_6", "run_15_step_5").
param("run_15", "run_15_step_6", "query", "travel flight hotel booking confirmation").
param("run_15", "run_15_step_6", "account", "acct_5").
param("run_15", "run_15_step_6", "max_results", "10").
call("run_15", "run_15_step_7", "gmail_search", "completed").
depends("run_15", "run_15_step_7", "run_15_step_5").
spawned_by("run_15", "run_15_step_7", "run_15_step_5").
param("run_15", "run_15_step_7", "query", "travel flight hotel booking confirmation").
param("run_15", "run_15_step_7", "account", "acct_4").
param("run_15", "run_15_step_7", "max_results", "10").
call("run_15", "run_15_step_8", "llm_generate", "completed").
depends("run_15", "run_15_step_8", "run_15_step_7").
depends("run_15", "run_15_step_8", "run_15_step_6").
call("run_15", "run_15_step_9", "gmail_list_threads", "completed").
depends("run_15", "run_15_step_9", "run_15_step_8").
spawned_by("run_15", "run_15_step_9", "run_15_step_8").
param("run_15", "run_15_step_9", "account", "acct_2").
param("run_15", "run_15_step_9", "labels", "INBOX").
call("run_15", "run_15_step_10", "gmail_list_threads", "completed").
depends("run_15", "run_15_step_10", "run_15_step_8").
spawned_by("run_15", "run_15_step_10", "run_15_step_8").
param("run_15", "run_15_step_10", "account", "acct_4").
param("run_15", "run_15_step_10", "labels", "INBOX").
call("run_15", "run_15_step_11", "gmail_list_threads", "completed").
depends("run_15", "run_15_step_11", "run_15_step_8").
spawned_by("run_15", "run_15_step_11", "run_15_step_8").
param("run_15", "run_15_step_11", "account", "acct_5").
param("run_15", "run_15_step_11", "labels", "INBOX").
call("run_15", "run_15_step_12", "llm_generate", "completed").
depends("run_15", "run_15_step_12", "run_15_step_11").
depends("run_15", "run_15_step_12", "run_15_step_10").
depends("run_15", "run_15_step_12", "run_15_step_9").
call("run_15", "run_15_step_13", "gmail_get_thread", "completed").
depends("run_15", "run_15_step_13", "run_15_step_12").
spawned_by("run_15", "run_15_step_13", "run_15_step_12").
param("run_15", "run_15_step_13", "thread_id", "thread_6").
param("run_15", "run_15_step_13", "account", "acct_5").
call("run_15", "run_15_step_14", "gmail_get_thread", "completed").
depends("run_15", "run_15_step_14", "run_15_step_12").
spawned_by("run_15", "run_15_step_14", "run_15_step_12").
param("run_15", "run_15_step_14", "thread_id", "thread_9").
param("run_15", "run_15_step_14", "account", "acct_5").
call("run_15", "run_15_step_15", "gmail_get_thread", "completed").
depends("run_15", "run_15_step_15", "run_15_step_12").
spawned_by("run_15", "run_15_step_15", "run_15_step_12").
param("run_15", "run_15_step_15", "thread_id", "thread_8").
param("run_15", "run_15_step_15", "account", "acct_5").
call("run_15", "run_15_step_16", "llm_generate", "completed").
depends("run_15", "run_15_step_16", "run_15_step_13").
depends("run_15", "run_15_step_16", "run_15_step_15").
depends("run_15", "run_15_step_16", "run_15_step_14").
call("run_15", "run_15_step_17", "sheets_read", "completed").
depends("run_15", "run_15_step_17", "run_15_step_16").
spawned_by("run_15", "run_15_step_17", "run_15_step_16").
param("run_15", "run_15_step_17", "spreadsheet", "spreadsheet_1").
param("run_15", "run_15_step_17", "account", "acct_3").
call("run_15", "run_15_step_18", "llm_generate", "completed").
depends("run_15", "run_15_step_18", "run_15_step_17").
call("run_15", "run_15_step_19", "sheets_read", "completed").
depends("run_15", "run_15_step_19", "run_15_step_18").
spawned_by("run_15", "run_15_step_19", "run_15_step_18").
param("run_15", "run_15_step_19", "spreadsheet", "spreadsheet_2").
param("run_15", "run_15_step_19", "account", "acct_3").
call("run_15", "run_15_step_20", "llm_generate", "completed").
depends("run_15", "run_15_step_20", "run_15_step_19").
call("run_15", "run_15_step_21", "sheets_update_values", "completed").
depends("run_15", "run_15_step_21", "run_15_step_20").
spawned_by("run_15", "run_15_step_21", "run_15_step_20").
param("run_15", "run_15_step_21", "spreadsheet", "spreadsheet_2").
param("run_15", "run_15_step_21", "account", "acct_3").
param("run_15", "run_15_step_21", "range", "🏨 Hotels!L2:L3").
param("run_15", "run_15_step_21", "target_sheet", "🏨 Hotels").
param("run_15", "run_15_step_21", "target_col_start", "L").
param("run_15", "run_15_step_21", "target_col_end", "L").
param("run_15", "run_15_step_21", "target_row_start", "2").
param("run_15", "run_15_step_21", "target_row_end", "3").
param("run_15", "run_15_step_21", "target_col_count", "1").
param("run_15", "run_15_step_21", "target_row_count", "2").
param("run_15", "run_15_step_21", "value_type", "status_flip").
param("run_15", "run_15_step_21", "value_constant", "Completed ✅").
param("run_15", "run_15_step_21", "value_row_count", "2").
param("run_15", "run_15_step_21", "value_col_count", "1").
call("run_15", "run_15_step_22", "sheets_update_values", "completed").
depends("run_15", "run_15_step_22", "run_15_step_20").
spawned_by("run_15", "run_15_step_22", "run_15_step_20").
param("run_15", "run_15_step_22", "spreadsheet", "spreadsheet_2").
param("run_15", "run_15_step_22", "account", "acct_3").
param("run_15", "run_15_step_22", "range", "🏨 Hotels!O2").
param("run_15", "run_15_step_22", "target_sheet", "🏨 Hotels").
param("run_15", "run_15_step_22", "target_col_start", "O").
param("run_15", "run_15_step_22", "target_col_end", "O").
param("run_15", "run_15_step_22", "target_row_start", "2").
param("run_15", "run_15_step_22", "target_row_end", "2").
param("run_15", "run_15_step_22", "target_col_count", "1").
param("run_15", "run_15_step_22", "target_row_count", "1").
param("run_15", "run_15_step_22", "value_type", "constant_fill").
param("run_15", "run_15_step_22", "value_constant", "Room 301. WiFi: Cardinal / 1HUNDRED").
param("run_15", "run_15_step_22", "value_row_count", "1").
param("run_15", "run_15_step_22", "value_col_count", "1").
call("run_15", "run_15_step_23", "llm_generate", "completed").
depends("run_15", "run_15_step_23", "run_15_step_21").
depends("run_15", "run_15_step_23", "run_15_step_22").
call("run_15", "run_15_step_24", "notify_user", "completed").
depends("run_15", "run_15_step_24", "run_15_step_23").
spawned_by("run_15", "run_15_step_24", "run_15_step_23").

% === Job: Travel Updates Last 6 Hours  — failed ===
job("run_16").
job_status("run_16", "failed").
job_title("run_16", "Travel Updates Last 6 Hours").
call("run_16", "run_16_step_1", "prompt_user", "completed").
call("run_16", "run_16_step_2", "llm_generate", "completed").
depends("run_16", "run_16_step_2", "run_16_step_1").
call("run_16", "run_16_step_3", "gmail_search", "completed").
depends("run_16", "run_16_step_3", "run_16_step_2").
spawned_by("run_16", "run_16_step_3", "run_16_step_2").
param("run_16", "run_16_step_3", "query", "after:2026/03/25 travel flight hotel booking confirmation").
param("run_16", "run_16_step_3", "account", "acct_2").
param("run_16", "run_16_step_3", "max_results", "20").
call("run_16", "run_16_step_4", "sheets_get_spreadsheet", "completed").
depends("run_16", "run_16_step_4", "run_16_step_2").
spawned_by("run_16", "run_16_step_4", "run_16_step_2").
param("run_16", "run_16_step_4", "spreadsheet", "spreadsheet_1").
param("run_16", "run_16_step_4", "account", "acct_3").
call("run_16", "run_16_step_5", "gmail_search", "failed").
depends("run_16", "run_16_step_5", "run_16_step_2").
spawned_by("run_16", "run_16_step_5", "run_16_step_2").
param("run_16", "run_16_step_5", "query", "after:2026/03/25 travel flight hotel booking confirmation").
param("run_16", "run_16_step_5", "account", "acct_1").
param("run_16", "run_16_step_5", "max_results", "20").
call("run_16", "run_16_step_6", "llm_generate", "completed").
depends("run_16", "run_16_step_6", "run_16_step_5").
depends("run_16", "run_16_step_6", "run_16_step_3").
depends("run_16", "run_16_step_6", "run_16_step_4").
call("run_16", "run_16_step_7", "gmail_search", "completed").
depends("run_16", "run_16_step_7", "run_16_step_6").
spawned_by("run_16", "run_16_step_7", "run_16_step_6").
param("run_16", "run_16_step_7", "query", "travel flight hotel booking confirmation").
param("run_16", "run_16_step_7", "account", "acct_4").
param("run_16", "run_16_step_7", "max_results", "20").
call("run_16", "run_16_step_8", "sheets_read", "completed").
depends("run_16", "run_16_step_8", "run_16_step_6").
spawned_by("run_16", "run_16_step_8", "run_16_step_6").
param("run_16", "run_16_step_8", "spreadsheet", "spreadsheet_1").
param("run_16", "run_16_step_8", "account", "acct_3").
param("run_16", "run_16_step_8", "sheet_name", "Flights").
call("run_16", "run_16_step_9", "sheets_read", "completed").
depends("run_16", "run_16_step_9", "run_16_step_6").
spawned_by("run_16", "run_16_step_9", "run_16_step_6").
param("run_16", "run_16_step_9", "spreadsheet", "spreadsheet_2").
param("run_16", "run_16_step_9", "account", "acct_3").
call("run_16", "run_16_step_10", "sheets_read", "completed").
depends("run_16", "run_16_step_10", "run_16_step_6").
spawned_by("run_16", "run_16_step_10", "run_16_step_6").
param("run_16", "run_16_step_10", "spreadsheet", "spreadsheet_3").
param("run_16", "run_16_step_10", "account", "acct_3").
call("run_16", "run_16_step_11", "gmail_search", "completed").
depends("run_16", "run_16_step_11", "run_16_step_6").
spawned_by("run_16", "run_16_step_11", "run_16_step_6").
param("run_16", "run_16_step_11", "query", "travel flight hotel booking confirmation").
param("run_16", "run_16_step_11", "account", "acct_5").
param("run_16", "run_16_step_11", "max_results", "20").
call("run_16", "run_16_step_12", "llm_generate", "completed").
depends("run_16", "run_16_step_12", "run_16_step_7").
depends("run_16", "run_16_step_12", "run_16_step_11").
depends("run_16", "run_16_step_12", "run_16_step_8").
depends("run_16", "run_16_step_12", "run_16_step_9").
depends("run_16", "run_16_step_12", "run_16_step_10").
call("run_16", "run_16_step_13", "notify_user", "completed").
depends("run_16", "run_16_step_13", "run_16_step_12").
spawned_by("run_16", "run_16_step_13", "run_16_step_12").

% === Job: Travel Updates Last 6 Hours  — failed ===
job("run_17").
job_status("run_17", "failed").
job_title("run_17", "Travel Updates Last 6 Hours").
call("run_17", "run_17_step_1", "prompt_user", "completed").
call("run_17", "run_17_step_2", "llm_generate", "completed").
depends("run_17", "run_17_step_2", "run_17_step_1").
call("run_17", "run_17_step_3", "gmail_search", "failed").
depends("run_17", "run_17_step_3", "run_17_step_2").
spawned_by("run_17", "run_17_step_3", "run_17_step_2").
param("run_17", "run_17_step_3", "query", "after:2026/03/25 travel flight hotel booking confirmation").
param("run_17", "run_17_step_3", "account", "acct_1").
param("run_17", "run_17_step_3", "max_results", "20").
call("run_17", "run_17_step_4", "sheets_read", "completed").
depends("run_17", "run_17_step_4", "run_17_step_2").
spawned_by("run_17", "run_17_step_4", "run_17_step_2").
param("run_17", "run_17_step_4", "spreadsheet", "spreadsheet_1").
param("run_17", "run_17_step_4", "account", "acct_3").
call("run_17", "run_17_step_5", "llm_generate", "completed").
depends("run_17", "run_17_step_5", "run_17_step_3").
depends("run_17", "run_17_step_5", "run_17_step_4").
call("run_17", "run_17_step_6", "gmail_search", "completed").
depends("run_17", "run_17_step_6", "run_17_step_5").
spawned_by("run_17", "run_17_step_6", "run_17_step_5").
param("run_17", "run_17_step_6", "query", "travel flight hotel booking confirmation after:2026/03/25").
param("run_17", "run_17_step_6", "account", "acct_2").
param("run_17", "run_17_step_6", "max_results", "20").
call("run_17", "run_17_step_7", "gmail_search", "completed").
depends("run_17", "run_17_step_7", "run_17_step_5").
spawned_by("run_17", "run_17_step_7", "run_17_step_5").
param("run_17", "run_17_step_7", "query", "travel flight hotel booking confirmation after:2026/03/25").
param("run_17", "run_17_step_7", "account", "acct_5").
param("run_17", "run_17_step_7", "max_results", "20").
call("run_17", "run_17_step_8", "gmail_search", "completed").
depends("run_17", "run_17_step_8", "run_17_step_5").
spawned_by("run_17", "run_17_step_8", "run_17_step_5").
param("run_17", "run_17_step_8", "query", "travel flight hotel booking confirmation after:2026/03/25").
param("run_17", "run_17_step_8", "account", "acct_4").
param("run_17", "run_17_step_8", "max_results", "20").
call("run_17", "run_17_step_9", "llm_generate", "completed").
depends("run_17", "run_17_step_9", "run_17_step_8").
depends("run_17", "run_17_step_9", "run_17_step_7").
depends("run_17", "run_17_step_9", "run_17_step_6").
call("run_17", "run_17_step_10", "notify_user", "completed").
depends("run_17", "run_17_step_10", "run_17_step_9").
spawned_by("run_17", "run_17_step_10", "run_17_step_9").

% === Job: Travel Updates Last 6 Hours  — failed ===
job("run_18").
job_status("run_18", "failed").
job_title("run_18", "Travel Updates Last 6 Hours").
call("run_18", "run_18_step_1", "prompt_user", "completed").
call("run_18", "run_18_step_2", "llm_generate", "completed").
depends("run_18", "run_18_step_2", "run_18_step_1").
call("run_18", "run_18_step_3", "gmail_search", "failed").
depends("run_18", "run_18_step_3", "run_18_step_2").
spawned_by("run_18", "run_18_step_3", "run_18_step_2").
param("run_18", "run_18_step_3", "query", "travel flight hotel booking confirmation reservation after:2026/03/25").
param("run_18", "run_18_step_3", "account", "acct_1").
param("run_18", "run_18_step_3", "max_results", "20").
call("run_18", "run_18_step_4", "gmail_search", "completed").
depends("run_18", "run_18_step_4", "run_18_step_2").
spawned_by("run_18", "run_18_step_4", "run_18_step_2").
param("run_18", "run_18_step_4", "query", "travel flight hotel booking confirmation reservation after:2026/03/25").
param("run_18", "run_18_step_4", "account", "acct_2").
param("run_18", "run_18_step_4", "max_results", "20").
call("run_18", "run_18_step_5", "gmail_search", "completed").
depends("run_18", "run_18_step_5", "run_18_step_2").
spawned_by("run_18", "run_18_step_5", "run_18_step_2").
param("run_18", "run_18_step_5", "query", "travel flight hotel booking confirmation reservation after:2026/03/25").
param("run_18", "run_18_step_5", "account", "acct_4").
param("run_18", "run_18_step_5", "max_results", "20").
call("run_18", "run_18_step_6", "llm_generate", "completed").
depends("run_18", "run_18_step_6", "run_18_step_3").
depends("run_18", "run_18_step_6", "run_18_step_4").
depends("run_18", "run_18_step_6", "run_18_step_5").
call("run_18", "run_18_step_7", "gmail_list_threads", "completed").
depends("run_18", "run_18_step_7", "run_18_step_6").
spawned_by("run_18", "run_18_step_7", "run_18_step_6").
param("run_18", "run_18_step_7", "account", "acct_4").
call("run_18", "run_18_step_8", "gmail_list_threads", "completed").
depends("run_18", "run_18_step_8", "run_18_step_6").
spawned_by("run_18", "run_18_step_8", "run_18_step_6").
param("run_18", "run_18_step_8", "account", "acct_2").
call("run_18", "run_18_step_9", "gmail_search", "completed").
depends("run_18", "run_18_step_9", "run_18_step_6").
spawned_by("run_18", "run_18_step_9", "run_18_step_6").
param("run_18", "run_18_step_9", "query", "travel flight hotel booking confirmation reservation after:2026/03/25").
param("run_18", "run_18_step_9", "account", "acct_5").
param("run_18", "run_18_step_9", "max_results", "20").
call("run_18", "run_18_step_10", "llm_generate", "completed").
depends("run_18", "run_18_step_10", "run_18_step_9").
depends("run_18", "run_18_step_10", "run_18_step_8").
depends("run_18", "run_18_step_10", "run_18_step_7").
call("run_18", "run_18_step_11", "notify_user", "completed").
depends("run_18", "run_18_step_11", "run_18_step_10").
spawned_by("run_18", "run_18_step_11", "run_18_step_10").

% === Job: Travel Updates Last 6 Hours  — failed ===
job("run_19").
job_status("run_19", "failed").
job_title("run_19", "Travel Updates Last 6 Hours").
call("run_19", "run_19_step_1", "prompt_user", "completed").
call("run_19", "run_19_step_2", "llm_generate", "completed").
depends("run_19", "run_19_step_2", "run_19_step_1").
call("run_19", "run_19_step_3", "daslab_trigger_schedule", "completed").
depends("run_19", "run_19_step_3", "run_19_step_2").
spawned_by("run_19", "run_19_step_3", "run_19_step_2").
param("run_19", "run_19_step_3", "triggered_job", "run_20").
call("run_19", "run_19_step_4", "llm_generate", "completed").
depends("run_19", "run_19_step_4", "run_19_step_3").
call("run_19", "run_19_step_5", "notify_user", "completed").
depends("run_19", "run_19_step_5", "run_19_step_4").
spawned_by("run_19", "run_19_step_5", "run_19_step_4").

% === Job: Travel Updates Last 6 Hours  — failed ===
job("run_21").
job_status("run_21", "failed").
job_title("run_21", "Travel Updates Last 6 Hours").
call("run_21", "run_21_step_1", "prompt_user", "completed").
call("run_21", "run_21_step_2", "llm_generate", "completed").
depends("run_21", "run_21_step_2", "run_21_step_1").
call("run_21", "run_21_step_3", "daslab_trigger_schedule", "completed").
depends("run_21", "run_21_step_3", "run_21_step_2").
spawned_by("run_21", "run_21_step_3", "run_21_step_2").
param("run_21", "run_21_step_3", "triggered_job", "run_20").
call("run_21", "run_21_step_4", "llm_generate", "completed").
depends("run_21", "run_21_step_4", "run_21_step_3").
call("run_21", "run_21_step_5", "notify_user", "completed").
depends("run_21", "run_21_step_5", "run_21_step_4").
spawned_by("run_21", "run_21_step_5", "run_21_step_4").

% === Job: Travel Updates Last 6 Hours  — failed ===
job("run_22").
job_status("run_22", "failed").
job_title("run_22", "Travel Updates Last 6 Hours").
call("run_22", "run_22_step_1", "prompt_user", "completed").
call("run_22", "run_22_step_2", "llm_generate", "completed").
depends("run_22", "run_22_step_2", "run_22_step_1").
call("run_22", "run_22_step_3", "daslab_trigger_schedule", "completed").
depends("run_22", "run_22_step_3", "run_22_step_2").
spawned_by("run_22", "run_22_step_3", "run_22_step_2").
param("run_22", "run_22_step_3", "triggered_job", "run_20").
call("run_22", "run_22_step_4", "llm_generate", "completed").
depends("run_22", "run_22_step_4", "run_22_step_3").
call("run_22", "run_22_step_5", "notify_user", "completed").
depends("run_22", "run_22_step_5", "run_22_step_4").
spawned_by("run_22", "run_22_step_5", "run_22_step_4").

% === Job: Travel Updates Last 6 Hours  — failed ===
job("run_23").
job_status("run_23", "failed").
job_title("run_23", "Travel Updates Last 6 Hours").
call("run_23", "run_23_step_1", "prompt_user", "completed").
call("run_23", "run_23_step_2", "llm_generate", "completed").
depends("run_23", "run_23_step_2", "run_23_step_1").
call("run_23", "run_23_step_3", "daslab_trigger_schedule", "completed").
depends("run_23", "run_23_step_3", "run_23_step_2").
spawned_by("run_23", "run_23_step_3", "run_23_step_2").
param("run_23", "run_23_step_3", "triggered_job", "run_20").
call("run_23", "run_23_step_4", "llm_generate", "completed").
depends("run_23", "run_23_step_4", "run_23_step_3").
call("run_23", "run_23_step_5", "notify_user", "completed").
depends("run_23", "run_23_step_5", "run_23_step_4").
spawned_by("run_23", "run_23_step_5", "run_23_step_4").

% === Job: Travel Updates Last 6 Hours  — failed ===
job("run_24").
job_status("run_24", "failed").
job_title("run_24", "Travel Updates Last 6 Hours").
call("run_24", "run_24_step_1", "prompt_user", "completed").
call("run_24", "run_24_step_2", "llm_generate", "completed").
depends("run_24", "run_24_step_2", "run_24_step_1").
call("run_24", "run_24_step_3", "daslab_trigger_schedule", "completed").
depends("run_24", "run_24_step_3", "run_24_step_2").
spawned_by("run_24", "run_24_step_3", "run_24_step_2").
param("run_24", "run_24_step_3", "triggered_job", "run_20").
call("run_24", "run_24_step_4", "llm_generate", "completed").
depends("run_24", "run_24_step_4", "run_24_step_3").
call("run_24", "run_24_step_5", "notify_user", "completed").
depends("run_24", "run_24_step_5", "run_24_step_4").
spawned_by("run_24", "run_24_step_5", "run_24_step_4").

% === Job: Travel Updates Last 6 Hours  — failed ===
job("run_25").
job_status("run_25", "failed").
job_title("run_25", "Travel Updates Last 6 Hours").
call("run_25", "run_25_step_1", "prompt_user", "completed").
call("run_25", "run_25_step_2", "llm_generate", "completed").
depends("run_25", "run_25_step_2", "run_25_step_1").
call("run_25", "run_25_step_3", "daslab_trigger_schedule", "completed").
depends("run_25", "run_25_step_3", "run_25_step_2").
spawned_by("run_25", "run_25_step_3", "run_25_step_2").
param("run_25", "run_25_step_3", "triggered_job", "run_20").
call("run_25", "run_25_step_4", "llm_generate", "completed").
depends("run_25", "run_25_step_4", "run_25_step_3").
call("run_25", "run_25_step_5", "daslab_list_jobs", "completed").
depends("run_25", "run_25_step_5", "run_25_step_4").
spawned_by("run_25", "run_25_step_5", "run_25_step_4").
call("run_25", "run_25_step_6", "llm_generate", "completed").
depends("run_25", "run_25_step_6", "run_25_step_5").
call("run_25", "run_25_step_7", "daslab_list_jobs", "completed").
depends("run_25", "run_25_step_7", "run_25_step_6").
spawned_by("run_25", "run_25_step_7", "run_25_step_6").
call("run_25", "run_25_step_8", "llm_generate", "completed").
depends("run_25", "run_25_step_8", "run_25_step_7").
call("run_25", "run_25_step_9", "gmail_search", "completed").
depends("run_25", "run_25_step_9", "run_25_step_8").
spawned_by("run_25", "run_25_step_9", "run_25_step_8").
param("run_25", "run_25_step_9", "query", "after:2026/03/25 subject:(flight OR hotel OR booking OR reservation OR travel OR airline OR airport)").
param("run_25", "run_25_step_9", "account", "acct_2").
param("run_25", "run_25_step_9", "max_results", "20").
call("run_25", "run_25_step_10", "gmail_search", "failed").
depends("run_25", "run_25_step_10", "run_25_step_8").
spawned_by("run_25", "run_25_step_10", "run_25_step_8").
param("run_25", "run_25_step_10", "query", "after:2026/03/25 subject:(flight OR hotel OR booking OR reservation OR travel OR airline OR airport)").
param("run_25", "run_25_step_10", "account", "acct_1").
param("run_25", "run_25_step_10", "max_results", "20").
call("run_25", "run_25_step_11", "gmail_search", "completed").
depends("run_25", "run_25_step_11", "run_25_step_8").
spawned_by("run_25", "run_25_step_11", "run_25_step_8").
param("run_25", "run_25_step_11", "query", "after:2026/03/25 subject:(flight OR hotel OR booking OR reservation OR travel OR airline OR airport)").
param("run_25", "run_25_step_11", "account", "acct_4").
param("run_25", "run_25_step_11", "max_results", "20").
call("run_25", "run_25_step_12", "llm_generate", "completed").
depends("run_25", "run_25_step_12", "run_25_step_10").
depends("run_25", "run_25_step_12", "run_25_step_9").
depends("run_25", "run_25_step_12", "run_25_step_11").
call("run_25", "run_25_step_13", "notify_user", "completed").
depends("run_25", "run_25_step_13", "run_25_step_12").
spawned_by("run_25", "run_25_step_13", "run_25_step_12").

% === Job: Travel Updates Last 6 Hours  — failed ===
job("run_26").
job_status("run_26", "failed").
job_title("run_26", "Travel Updates Last 6 Hours").
call("run_26", "run_26_step_1", "prompt_user", "completed").
call("run_26", "run_26_step_2", "llm_generate", "completed").
depends("run_26", "run_26_step_2", "run_26_step_1").
call("run_26", "run_26_step_3", "daslab_trigger_schedule", "completed").
depends("run_26", "run_26_step_3", "run_26_step_2").
spawned_by("run_26", "run_26_step_3", "run_26_step_2").
param("run_26", "run_26_step_3", "triggered_job", "run_20").
call("run_26", "run_26_step_4", "llm_generate", "completed").
depends("run_26", "run_26_step_4", "run_26_step_3").
call("run_26", "run_26_step_5", "daslab_list_jobs", "completed").
depends("run_26", "run_26_step_5", "run_26_step_4").
spawned_by("run_26", "run_26_step_5", "run_26_step_4").
call("run_26", "run_26_step_6", "llm_generate", "completed").
depends("run_26", "run_26_step_6", "run_26_step_5").
call("run_26", "run_26_step_7", "daslab_list_jobs", "completed").
depends("run_26", "run_26_step_7", "run_26_step_6").
spawned_by("run_26", "run_26_step_7", "run_26_step_6").
call("run_26", "run_26_step_8", "llm_generate", "completed").
depends("run_26", "run_26_step_8", "run_26_step_7").
call("run_26", "run_26_step_9", "gmail_search", "completed").
depends("run_26", "run_26_step_9", "run_26_step_8").
spawned_by("run_26", "run_26_step_9", "run_26_step_8").
param("run_26", "run_26_step_9", "query", "after:2026/03/25").
param("run_26", "run_26_step_9", "account", "acct_2").
param("run_26", "run_26_step_9", "max_results", "20").
call("run_26", "run_26_step_10", "gmail_search", "failed").
depends("run_26", "run_26_step_10", "run_26_step_8").
spawned_by("run_26", "run_26_step_10", "run_26_step_8").
param("run_26", "run_26_step_10", "query", "after:2026/03/25").
param("run_26", "run_26_step_10", "account", "acct_1").
param("run_26", "run_26_step_10", "max_results", "20").
call("run_26", "run_26_step_11", "llm_generate", "completed").
depends("run_26", "run_26_step_11", "run_26_step_10").
depends("run_26", "run_26_step_11", "run_26_step_9").
call("run_26", "run_26_step_12", "gmail_search", "completed").
depends("run_26", "run_26_step_12", "run_26_step_11").
spawned_by("run_26", "run_26_step_12", "run_26_step_11").
param("run_26", "run_26_step_12", "query", "after:2026/03/25").
param("run_26", "run_26_step_12", "account", "acct_4").
param("run_26", "run_26_step_12", "max_results", "20").
call("run_26", "run_26_step_13", "gmail_search", "completed").
depends("run_26", "run_26_step_13", "run_26_step_11").
spawned_by("run_26", "run_26_step_13", "run_26_step_11").
param("run_26", "run_26_step_13", "query", "after:2026/03/25 (flight OR hotel OR booking OR airbnb OR airline OR reservation OR confirmation OR itinerary OR visa)").
param("run_26", "run_26_step_13", "account", "acct_2").
param("run_26", "run_26_step_13", "max_results", "20").
call("run_26", "run_26_step_14", "llm_generate", "completed").
depends("run_26", "run_26_step_14", "run_26_step_13").
depends("run_26", "run_26_step_14", "run_26_step_12").
call("run_26", "run_26_step_15", "notify_user", "completed").
depends("run_26", "run_26_step_15", "run_26_step_14").
spawned_by("run_26", "run_26_step_15", "run_26_step_14").

% === Job: Travel Updates Last 6 Hours  — failed ===
job("run_27").
job_status("run_27", "failed").
job_title("run_27", "Travel Updates Last 6 Hours").
call("run_27", "run_27_step_1", "prompt_user", "completed").
call("run_27", "run_27_step_2", "llm_generate", "completed").
depends("run_27", "run_27_step_2", "run_27_step_1").
call("run_27", "run_27_step_3", "daslab_trigger_schedule", "completed").
depends("run_27", "run_27_step_3", "run_27_step_2").
spawned_by("run_27", "run_27_step_3", "run_27_step_2").
param("run_27", "run_27_step_3", "triggered_job", "run_20").
call("run_27", "run_27_step_4", "llm_generate", "completed").
depends("run_27", "run_27_step_4", "run_27_step_3").
call("run_27", "run_27_step_5", "daslab_list_jobs", "completed").
depends("run_27", "run_27_step_5", "run_27_step_4").
spawned_by("run_27", "run_27_step_5", "run_27_step_4").
call("run_27", "run_27_step_6", "llm_generate", "completed").
depends("run_27", "run_27_step_6", "run_27_step_5").
call("run_27", "run_27_step_7", "daslab_list_jobs", "completed").
depends("run_27", "run_27_step_7", "run_27_step_6").
spawned_by("run_27", "run_27_step_7", "run_27_step_6").
call("run_27", "run_27_step_8", "llm_generate", "completed").
depends("run_27", "run_27_step_8", "run_27_step_7").
call("run_27", "run_27_step_9", "gmail_search", "completed").
depends("run_27", "run_27_step_9", "run_27_step_8").
spawned_by("run_27", "run_27_step_9", "run_27_step_8").
param("run_27", "run_27_step_9", "query", "after:2026/03/25 travel flight hotel booking confirmation").
param("run_27", "run_27_step_9", "account", "acct_2").
param("run_27", "run_27_step_9", "max_results", "20").
call("run_27", "run_27_step_10", "gmail_search", "failed").
depends("run_27", "run_27_step_10", "run_27_step_8").
spawned_by("run_27", "run_27_step_10", "run_27_step_8").
param("run_27", "run_27_step_10", "query", "after:2026/03/25 travel flight hotel booking confirmation").
param("run_27", "run_27_step_10", "account", "acct_1").
param("run_27", "run_27_step_10", "max_results", "20").
call("run_27", "run_27_step_11", "llm_generate", "completed").
depends("run_27", "run_27_step_11", "run_27_step_10").
depends("run_27", "run_27_step_11", "run_27_step_9").
call("run_27", "run_27_step_12", "gmail_search", "completed").
depends("run_27", "run_27_step_12", "run_27_step_11").
spawned_by("run_27", "run_27_step_12", "run_27_step_11").
param("run_27", "run_27_step_12", "query", "after:2026/03/25 travel flight hotel booking confirmation").
param("run_27", "run_27_step_12", "account", "acct_4").
param("run_27", "run_27_step_12", "max_results", "20").
call("run_27", "run_27_step_13", "gmail_search", "completed").
depends("run_27", "run_27_step_13", "run_27_step_11").
spawned_by("run_27", "run_27_step_13", "run_27_step_11").
param("run_27", "run_27_step_13", "query", "after:2026/03/25 travel flight hotel booking confirmation").
param("run_27", "run_27_step_13", "account", "acct_5").
param("run_27", "run_27_step_13", "max_results", "20").
call("run_27", "run_27_step_14", "llm_generate", "completed").
depends("run_27", "run_27_step_14", "run_27_step_12").
depends("run_27", "run_27_step_14", "run_27_step_13").
call("run_27", "run_27_step_15", "gmail_list_threads", "completed").
depends("run_27", "run_27_step_15", "run_27_step_14").
spawned_by("run_27", "run_27_step_15", "run_27_step_14").
param("run_27", "run_27_step_15", "account", "acct_4").
param("run_27", "run_27_step_15", "labels", "INBOX").
call("run_27", "run_27_step_16", "gmail_list_threads", "completed").
depends("run_27", "run_27_step_16", "run_27_step_14").
spawned_by("run_27", "run_27_step_16", "run_27_step_14").
param("run_27", "run_27_step_16", "account", "acct_2").
param("run_27", "run_27_step_16", "labels", "INBOX").
call("run_27", "run_27_step_17", "gmail_list_threads", "completed").
depends("run_27", "run_27_step_17", "run_27_step_14").
spawned_by("run_27", "run_27_step_17", "run_27_step_14").
param("run_27", "run_27_step_17", "account", "acct_5").
param("run_27", "run_27_step_17", "labels", "INBOX").
call("run_27", "run_27_step_18", "llm_generate", "completed").
depends("run_27", "run_27_step_18", "run_27_step_15").
depends("run_27", "run_27_step_18", "run_27_step_17").
depends("run_27", "run_27_step_18", "run_27_step_16").
call("run_27", "run_27_step_19", "gmail_get_thread", "completed").
depends("run_27", "run_27_step_19", "run_27_step_18").
spawned_by("run_27", "run_27_step_19", "run_27_step_18").
param("run_27", "run_27_step_19", "thread_id", "thread_10").
param("run_27", "run_27_step_19", "account", "acct_5").
call("run_27", "run_27_step_20", "gmail_get_thread", "completed").
depends("run_27", "run_27_step_20", "run_27_step_18").
spawned_by("run_27", "run_27_step_20", "run_27_step_18").
param("run_27", "run_27_step_20", "thread_id", "thread_11").
param("run_27", "run_27_step_20", "account", "acct_5").
call("run_27", "run_27_step_21", "llm_generate", "completed").
depends("run_27", "run_27_step_21", "run_27_step_19").
depends("run_27", "run_27_step_21", "run_27_step_20").
call("run_27", "run_27_step_22", "llm_look_at", "completed").
depends("run_27", "run_27_step_22", "run_27_step_21").
spawned_by("run_27", "run_27_step_22", "run_27_step_21").
param("run_27", "run_27_step_22", "type", "attachment_inspection").
call("run_27", "run_27_step_23", "sheets_read", "completed").
depends("run_27", "run_27_step_23", "run_27_step_21").
spawned_by("run_27", "run_27_step_23", "run_27_step_21").
param("run_27", "run_27_step_23", "spreadsheet", "spreadsheet_1").
param("run_27", "run_27_step_23", "account", "acct_3").
call("run_27", "run_27_step_24", "llm_generate", "completed").
depends("run_27", "run_27_step_24", "run_27_step_22").
depends("run_27", "run_27_step_24", "run_27_step_23").
call("run_27", "run_27_step_25", "sheets_get_values", "completed").
depends("run_27", "run_27_step_25", "run_27_step_24").
spawned_by("run_27", "run_27_step_25", "run_27_step_24").
param("run_27", "run_27_step_25", "spreadsheet", "spreadsheet_1").
param("run_27", "run_27_step_25", "account", "acct_3").
param("run_27", "run_27_step_25", "range", "Flights!A1:O6").
call("run_27", "run_27_step_26", "llm_generate", "completed").
depends("run_27", "run_27_step_26", "run_27_step_25").
call("run_27", "run_27_step_27", "sheets_update_values", "completed").
depends("run_27", "run_27_step_27", "run_27_step_26").
spawned_by("run_27", "run_27_step_27", "run_27_step_26").
param("run_27", "run_27_step_27", "spreadsheet", "spreadsheet_1").
param("run_27", "run_27_step_27", "account", "acct_3").
param("run_27", "run_27_step_27", "range", "Flights!L3:L5").
param("run_27", "run_27_step_27", "target_sheet", "Flights").
param("run_27", "run_27_step_27", "target_col_start", "L").
param("run_27", "run_27_step_27", "target_col_end", "L").
param("run_27", "run_27_step_27", "target_row_start", "3").
param("run_27", "run_27_step_27", "target_row_end", "5").
param("run_27", "run_27_step_27", "target_col_count", "1").
param("run_27", "run_27_step_27", "target_row_count", "3").
param("run_27", "run_27_step_27", "value_type", "status_flip").
param("run_27", "run_27_step_27", "value_constant", "Completed ✅").
param("run_27", "run_27_step_27", "value_row_count", "3").
param("run_27", "run_27_step_27", "value_col_count", "1").
call("run_27", "run_27_step_28", "llm_generate", "completed").
depends("run_27", "run_27_step_28", "run_27_step_27").
call("run_27", "run_27_step_29", "notify_user", "completed").
depends("run_27", "run_27_step_29", "run_27_step_28").
spawned_by("run_27", "run_27_step_29", "run_27_step_28").

% === Job: Travel Updates Last 6 Hours  — completed ===
job("run_20").
job_status("run_20", "completed").
job_title("run_20", "Travel Updates Last 6 Hours").
call("run_20", "run_20_step_1", "prompt_user", "completed").

