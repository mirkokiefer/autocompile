# Trace Format Specification

autocompile accepts execution traces in two formats: JSON and ASP (Logic Programming facts).

## JSON Format

```json
{
  "runs": [
    {
      "id": "run_1",
      "status": "completed",
      "steps": [
        {
          "id": "step_1",
          "tool": "gmail_search",
          "params": {
            "query": "flight confirmation",
            "account": "acct_1",
            "max_results": "20"
          },
          "output": {
            "thread_ids": ["t_1", "t_2"]
          },
          "depends_on": [],
          "spawned_by": null,
          "status": "completed"
        },
        {
          "id": "step_2",
          "tool": "gmail_get_thread",
          "params": {
            "thread_id": "t_1",
            "account": "acct_1"
          },
          "depends_on": ["step_1"],
          "spawned_by": "step_0",
          "status": "completed"
        }
      ]
    }
  ]
}
```

### Fields

**Run-level:**

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | string | yes | Unique identifier for this execution run |
| `status` | string | no | Run outcome: "completed", "failed", etc. |
| `steps` | array | yes | Ordered list of tool calls in this run |

**Step-level:**

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | string | yes | Unique identifier within the run |
| `tool` | string | yes | Name of the tool/function that was called |
| `params` | object | no | Input parameters passed to the tool |
| `output` | object | no | Output/return value from the tool (used for model benchmarking) |
| `depends_on` | array | no | Step IDs that must complete before this step runs |
| `spawned_by` | string | no | ID of the step that initiated this step (e.g., an LLM call that decided to invoke this tool) |
| `status` | string | no | Step outcome: "completed", "failed", etc. |

### Notes

- All parameter values should be strings. autocompile handles type inference internally.
- The `output` field is optional but enables model compilation (benchmarking cheaper models against recorded outputs).
- `depends_on` captures explicit data dependencies. `spawned_by` captures the orchestration relationship (which decision point triggered this step).
- Steps with `status: "failed"` are included in pattern mining -- a tool that consistently fails is still a pattern.

## ASP Format (Logic Programming Facts)

For direct use with the Clingo compilation strategy:

```prolog
% Run declaration
job("run_1").
job_status("run_1", "completed").

% Step declarations: job_id, step_id, tool_name, status
call("run_1", "step_1", "gmail_search", "completed").
call("run_1", "step_2", "gmail_get_thread", "completed").

% Dependencies: job_id, step_id, dependency_step_id
depends("run_1", "step_2", "step_1").

% Orchestration: job_id, step_id, parent_step_id
spawned_by("run_1", "step_1", "step_0").
spawned_by("run_1", "step_2", "step_0").

% Parameters: job_id, step_id, key, value
param("run_1", "step_1", "query", "flight confirmation").
param("run_1", "step_1", "account", "acct_1").
param("run_1", "step_1", "max_results", "20").
```

### Predicates

| Predicate | Arity | Description |
|---|---|---|
| `job/1` | 1 | Declares a run |
| `job_status/2` | 2 | Run status |
| `call/4` | 4 | A tool call: (run_id, step_id, tool, status) |
| `depends/3` | 3 | Explicit dependency: (run_id, step_id, dep_step_id) |
| `spawned_by/3` | 3 | Orchestration parent: (run_id, step_id, parent_step_id) |
| `param/4` | 4 | Parameter value: (run_id, step_id, key, value) |

### Glue tools

Most agent frameworks have internal tools (LLM calls, notification handlers, etc.) that aren't part of the actual workflow logic. These are filtered by the ASP rules via `glue_tool` declarations. The default rules filter:

```prolog
glue_tool("llm_generate").
glue_tool("notify_user").
glue_tool("prompt_user").
```

Add your framework's internal tools to the rules file.

## Converting between formats

autocompile can consume either format. The ASP format is used directly by the Clingo strategy. JSON traces are converted to ASP internally when needed.
