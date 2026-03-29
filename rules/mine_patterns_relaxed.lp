% ============================================================================
% mine_patterns_relaxed.lp — Relaxed compilation (25% threshold)
%
% Same as mine_patterns.lp but with lower core-tool threshold.
% Use when many runs fail/don't complete all steps.
%
% Core tool threshold: 25% of runs (vs 50% in mine_patterns.lp)
% ============================================================================

% --- Tool classification ---
glue_tool("llm_generate").
glue_tool("notify_user").
glue_tool("prompt_user").

actionable_call(J, C, T) :- call(J, C, T, "completed"), not glue_tool(T).
actionable_call(J, C, T) :- call(J, C, T, "failed"), not glue_tool(T).
tool_in_job(J, T) :- actionable_call(J, C, T).
known_tool(T) :- tool_in_job(_, T).
job_count(N) :- N = #count { J : job(J) }.
tool_job_count(T, N) :- known_tool(T), N = #count { J : tool_in_job(J, T) }, N > 0.

% RELAXED: core if in >=25% of runs
core_tool(T) :- tool_job_count(T, N), job_count(Total), N * 4 >= Total.

% --- Parallel detection ---
parallel_pair(T1, T2) :- actionable_call(J, C1, T1),
                          actionable_call(J, C2, T2),
                          spawned_by(J, C1, P),
                          spawned_by(J, C2, P),
                          T1 < T2,
                          core_tool(T1), core_tool(T2).

% --- Dependency ordering ---
depends_transitive(J, A, B) :- depends(J, A, B).
depends_transitive(J, A, C) :- depends(J, A, B), depends_transitive(J, B, C).

precedes_in_job(J, T1, T2) :- actionable_call(J, C1, T1),
                                actionable_call(J, C2, T2),
                                spawned_by(J, C1, P1),
                                spawned_by(J, C2, P2),
                                P1 != P2, T1 != T2,
                                core_tool(T1), core_tool(T2),
                                depends_transitive(J, P2, P1).

precedes_in_job(J, T1, T2) :- actionable_call(J, C1, T1),
                                actionable_call(J, C2, T2),
                                spawned_by(J, C2, P2),
                                depends(J, P2, C1),
                                T1 != T2,
                                core_tool(T1), core_tool(T2).

consistent_order(T1, T2) :- precedes_in_job(_, T1, T2),
                             not precedes_in_job(_, T2, T1).

conflicting_order(T1, T2) :- precedes_in_job(_, T1, T2),
                              precedes_in_job(_, T2, T1),
                              T1 < T2.

order_evidence(T1, T2, N) :- conflicting_order(T1, T2),
                              N = #count { J : precedes_in_job(J, T1, T2) }.
order_evidence(T2, T1, N) :- conflicting_order(T1, T2),
                              N = #count { J : precedes_in_job(J, T2, T1) }.

{ chosen_order(T1, T2) ; chosen_order(T2, T1) } = 1 :- conflicting_order(T1, T2).

order_cost(T1, T2, N) :- chosen_order(T1, T2), conflicting_order(T1a, T2a),
                          T1 = T1a, T2 = T2a,
                          order_evidence(T2, T1, N).
order_cost(T1, T2, N) :- chosen_order(T1, T2), conflicting_order(T2a, T1a),
                          T1 = T1a, T2 = T2a,
                          order_evidence(T2, T1, N).

% --- Conditional execution ---
conditional_tool(B, A) :- known_tool(A), known_tool(B), A != B,
                           tool_job_count(A, NA), tool_job_count(B, NB),
                           NB < NA, NB > 0, NB * 10 >= NA,
                           not b_without_a(B, A).
b_without_a(B, A) :- tool_in_job(J, B), not tool_in_job(J, A),
                      known_tool(A), known_tool(B).
conditional_rate(B, A, NB, NA) :- conditional_tool(B, A),
                                   tool_job_count(A, NA), tool_job_count(B, NB).

% --- Co-occurrence and fusion ---
co_occurs(A, B) :- known_tool(A), known_tool(B), A < B,
                    tool_job_count(A, N), tool_job_count(B, N),
                    not split_occurrence(A, B).
split_occurrence(A, B) :- tool_in_job(J, A), not tool_in_job(J, B),
                           known_tool(A), known_tool(B), A < B.
split_occurrence(A, B) :- tool_in_job(J, B), not tool_in_job(J, A),
                           known_tool(A), known_tool(B), A < B.

fusion_candidate(A, B) :- consistent_order(A, B),
                           not split_occurrence(A, B),
                           core_tool(A), core_tool(B),
                           not has_intermediate(A, B).
has_intermediate(A, B) :- consistent_order(A, X), consistent_order(X, B),
                           core_tool(X), A != X, X != B.

% --- Mutually exclusive ---
mutually_exclusive(A, B) :- known_tool(A), known_tool(B), A < B,
                             tool_in_job(_, A), tool_in_job(_, B),
                             not both_in_any_job(A, B).
both_in_any_job(A, B) :- tool_in_job(J, A), tool_in_job(J, B), A < B.

% --- Parameters ---
known_param(T, K, V) :- actionable_call(_, C, T), param(_, C, K, V), core_tool(T).
param_value_count(T, K, V, N) :- known_param(T, K, V),
                                  N = #count { J : actionable_call(J, C, T), param(J, C, K, V) },
                                  N > 0.
stable_param(T, K, V) :- param_value_count(T, K, V, N), tool_job_count(T, TotalForTool),
                          N * 2 >= TotalForTool.
variable_param(T, K) :- known_param(T, K, _), core_tool(T), not has_stable_value(T, K).
has_stable_value(T, K) :- stable_param(T, K, _).

% --- Phase detection ---
has_core_predecessor(T) :- consistent_order(T2, T), core_tool(T2).
has_core_predecessor(T) :- chosen_order(T2, T), core_tool(T2).
phase_zero(T) :- core_tool(T), not has_core_predecessor(T).

% --- Optimization ---
#minimize { N@2,T1,T2 : order_cost(T1, T2, N) }.
#minimize { 1@1,T : has_core_predecessor(T), core_tool(T) }.

% --- Output ---
#show core_tool/1.
#show consistent_order/2.
#show conflicting_order/2.
#show chosen_order/2.
#show parallel_pair/2.
#show stable_param/3.
#show variable_param/2.
#show phase_zero/1.
#show tool_job_count/2.
#show job_count/1.
#show conditional_tool/2.
#show conditional_rate/4.
#show co_occurs/2.
#show fusion_candidate/2.
#show mutually_exclusive/2.
