% ============================================================================
% mine_patterns.lp — Compile observed workflows into deterministic programs
%
% Completely generic — works on any trace data with no workflow-specific
% knowledge. All patterns are discovered from the data alone.
%
% Given execution trace facts (job/call/depends/spawned_by/param), discovers:
%   1. Core tools (appear consistently across runs)
%   2. Parallel groups and dependency orderings
%   3. Conditional execution patterns (branching)
%   4. Co-occurrence groups and fusion candidates
%   5. Stable parameters
%   6. Optimal phase assignment under parallelism constraints
%
% Core tool threshold: 50% of runs (use mine_patterns_relaxed.lp for 25%)
% ============================================================================

% --- Tool classification ---

% Internal/orchestration tools that aren't part of the actual workflow.
% These are the only configuration needed — everything else is discovered.
glue_tool("llm_generate").
glue_tool("notify_user").
glue_tool("prompt_user").

actionable_call(J, C, T) :- call(J, C, T, "completed"), not glue_tool(T).
actionable_call(J, C, T) :- call(J, C, T, "failed"), not glue_tool(T).

% A tool type appears in a run if at least one call of that type exists
tool_in_job(J, T) :- actionable_call(J, C, T).

% Enumerate tools that appear in any run
known_tool(T) :- tool_in_job(_, T).

% Count how many runs exist
job_count(N) :- N = #count { J : job(J) }.

% Count how many runs each tool appears in
tool_job_count(T, N) :- known_tool(T), N = #count { J : tool_in_job(J, T) }, N > 0.

% A tool is "core" if it appears in at least half the runs
core_tool(T) :- tool_job_count(T, N), job_count(Total), N * 2 >= Total.

% --- Parallel detection ---

% Two tools are parallelizable if they share the same parent call in any run
parallel_pair(T1, T2) :- actionable_call(J, C1, T1),
                          actionable_call(J, C2, T2),
                          spawned_by(J, C1, P),
                          spawned_by(J, C2, P),
                          T1 < T2,
                          core_tool(T1), core_tool(T2).

% --- Dependency ordering ---

% Transitive dependency
depends_transitive(J, A, B) :- depends(J, A, B).
depends_transitive(J, A, C) :- depends(J, A, B), depends_transitive(J, B, C).

% T1 precedes T2 in a run: T2's parent depends transitively on T1's parent
precedes_in_job(J, T1, T2) :- actionable_call(J, C1, T1),
                                actionable_call(J, C2, T2),
                                spawned_by(J, C1, P1),
                                spawned_by(J, C2, P2),
                                P1 != P2, T1 != T2,
                                core_tool(T1), core_tool(T2),
                                depends_transitive(J, P2, P1).

% Also: T2's parent directly waits for a call of T1
precedes_in_job(J, T1, T2) :- actionable_call(J, C1, T1),
                                actionable_call(J, C2, T2),
                                spawned_by(J, C2, P2),
                                depends(J, P2, C1),
                                T1 != T2,
                                core_tool(T1), core_tool(T2).

% Consistent ordering: T1 before T2 in at least one run, never T2 before T1
consistent_order(T1, T2) :- precedes_in_job(_, T1, T2),
                             not precedes_in_job(_, T2, T1).

% Conflicting ordering: evidence for both directions
conflicting_order(T1, T2) :- precedes_in_job(_, T1, T2),
                              precedes_in_job(_, T2, T1),
                              T1 < T2.

% Count evidence for each direction of conflicting orderings
order_evidence(T1, T2, N) :- conflicting_order(T1, T2),
                              N = #count { J : precedes_in_job(J, T1, T2) }.
order_evidence(T2, T1, N) :- conflicting_order(T1, T2),
                              N = #count { J : precedes_in_job(J, T2, T1) }.

% For conflicting orders, choose the majority direction
% This is a genuine choice point — the solver explores alternatives
{ chosen_order(T1, T2) ; chosen_order(T2, T1) } = 1 :- conflicting_order(T1, T2).

% Prefer the ordering with more evidence (optimization below)
order_cost(T1, T2, N) :- chosen_order(T1, T2), conflicting_order(T1a, T2a),
                          T1 = T1a, T2 = T2a,
                          order_evidence(T2, T1, N).
order_cost(T1, T2, N) :- chosen_order(T1, T2), conflicting_order(T2a, T1a),
                          T1 = T1a, T2 = T2a,
                          order_evidence(T2, T1, N).

% --- Conditional execution (branching) ---

% Tool B is conditional on tool A: B only appears in a subset of runs that have A
% This detects "if search found results, then read threads" patterns
conditional_tool(B, A) :- known_tool(A), known_tool(B), A != B,
                           tool_job_count(A, NA), tool_job_count(B, NB),
                           NB < NA,
                           NB > 0,
                           % B appears in at least 10% of runs with A
                           NB * 10 >= NA,
                           % Every run with B also has A
                           not b_without_a(B, A).

b_without_a(B, A) :- tool_in_job(J, B), not tool_in_job(J, A),
                      known_tool(A), known_tool(B).

% Conditional tool rate: what fraction of A-runs also have B
conditional_rate(B, A, NB, NA) :- conditional_tool(B, A),
                                   tool_job_count(A, NA),
                                   tool_job_count(B, NB).

% --- Co-occurrence and fusion ---

% Two tools always appear together (in the same runs)
co_occurs(A, B) :- known_tool(A), known_tool(B), A < B,
                    tool_job_count(A, N), tool_job_count(B, N),
                    not split_occurrence(A, B).

split_occurrence(A, B) :- tool_in_job(J, A), not tool_in_job(J, B),
                           known_tool(A), known_tool(B), A < B.
split_occurrence(A, B) :- tool_in_job(J, B), not tool_in_job(J, A),
                           known_tool(A), known_tool(B), A < B.

% Fusion candidate: A always immediately precedes B, they co-occur,
% and B never appears without A preceding it
fusion_candidate(A, B) :- consistent_order(A, B),
                           not split_occurrence(A, B),
                           core_tool(A), core_tool(B),
                           % No other core tool between them
                           not has_intermediate(A, B).

has_intermediate(A, B) :- consistent_order(A, X), consistent_order(X, B),
                           core_tool(X), A != X, X != B.

% --- Mutually exclusive tools (branch alternatives) ---

% Two tools never appear in the same run — they are alternatives
mutually_exclusive(A, B) :- known_tool(A), known_tool(B), A < B,
                             tool_in_job(_, A), tool_in_job(_, B),
                             not both_in_any_job(A, B).

both_in_any_job(A, B) :- tool_in_job(J, A), tool_in_job(J, B), A < B.

% --- Parameter extraction ---

% Enumerate known (tool, key, value) triples
known_param(T, K, V) :- actionable_call(_, C, T), param(_, C, K, V), core_tool(T).

% Count occurrences
param_value_count(T, K, V, N) :- known_param(T, K, V),
                                  N = #count { J : actionable_call(J, C, T), param(J, C, K, V) },
                                  N > 0.

% A parameter value is "stable" if it appears in most runs that use this tool
stable_param(T, K, V) :- param_value_count(T, K, V, N), tool_job_count(T, TotalForTool),
                          N * 2 >= TotalForTool.

% A parameter key is "variable" if no single value dominates
variable_param(T, K) :- known_param(T, K, _), core_tool(T),
                         not has_stable_value(T, K).
has_stable_value(T, K) :- stable_param(T, K, _).

% --- Optimal phase assignment ---

% Phase detection: tools with no core predecessors go in phase 0
has_core_predecessor(T) :- consistent_order(T2, T), core_tool(T2).
has_core_predecessor(T) :- chosen_order(T2, T), core_tool(T2).
phase_zero(T) :- core_tool(T), not has_core_predecessor(T).

% --- Optimization ---

% Minimize: violations of observed orderings (for conflicting cases)
#minimize { N@2,T1,T2 : order_cost(T1, T2, N) }.

% Minimize: number of phases (prefer parallelism)
% (lower priority than respecting evidence)
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
