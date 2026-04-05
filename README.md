# autocompile

AI agents spend most of their compute re-deriving decisions that were already answered by the last hundred runs. autocompile watches processes run and discovers their structure from data. What's invariant becomes compiled code. What varies becomes a parameter. What conflicts gets resolved by optimization. The LLM isn't eliminated -- it's relocated to exactly the decisions that require judgment.

The output is a program that separates the known from the unknown -- a map of where intelligence is actually needed, with empirical accuracy metrics for every compiled step.

## The pipeline

```
1. Observe    Collect execution traces from repeated workflow runs
2. Compile    Mine patterns using Answer Set Programming (Clingo)
3. Benchmark  Validate compiled program against held-out traces
4. Codegen    Emit an executable job spec for your runtime
```

## What the compiler discovers

Given execution traces, autocompile automatically identifies:

- **Core tools** -- which tools appear consistently across runs
- **Parallel groups** -- which tools always run concurrently
- **Dependency ordering** -- which tools must precede others
- **Conflicting orderings** -- when evidence disagrees, the solver picks the optimal direction
- **Conditional execution** -- "tool B only runs when tool A produced results"
- **Fusion candidates** -- sequential steps that can be merged into one operation
- **Mutually exclusive tools** -- branch alternatives that never co-occur
- **Stable vs variable parameters** -- which inputs are constant vs runtime-dependent

All patterns are discovered from data alone. The ASP rules are completely generic -- no workflow-specific knowledge needed.

## Why ASP (Answer Set Programming)

The simple patterns (frequency counting, parameter stability) don't need a logic solver. But real workflows have **conflicting evidence** -- tool A precedes B in 60% of runs, but B precedes A in 40%. autocompile uses Clingo's choice rules to model these conflicts and optimization to resolve them:

```prolog
% When orderings conflict, choose one direction
{ chosen_order(A, B) ; chosen_order(B, A) } = 1 :- conflicting_order(A, B).

% Minimize violations against observed evidence
#minimize { N@2,T1,T2 : order_cost(T1, T2, N) }.
```

The solver explores all consistent combinations and returns the optimal compilation. As rules grow more complex (resource constraints, branching logic, cross-workflow optimization), the ASP program grows linearly while a hand-coded solver would grow combinatorially.

## Examples

### Travel updates (177 runs)

27 real production traces + 150 synthetic traces modeled on observed patterns. The compiler discovers a 5-phase workflow with conditional branching:

```
Phase 0:  gmail_search x3 accounts (parallel)
Phase 1:  sheets_read x2 spreadsheets
            ↳ conditional on gmail_search (95% of runs)
Phase 2:  gmail_list_threads x3 accounts
            ↳ conditional on gmail_search (53% of runs)
Phase 3:  gmail_get_thread x3 accounts
            ↳ conditional on sheets_read (34% of runs)
Phase 4:  sheets_update_values
            ↳ conditional on gmail_get_thread (94% of runs)
```

4 conflicting orderings resolved by optimization. 2 fusion candidates identified (`gmail_search + gmail_get_thread`). Benchmark against held-out traces: **88% parameter accuracy, 96% of runs matched**.

### Order updates (118 runs)

118 real production traces. 6 conflicting orderings resolved. 4-phase compiled DAG. Benchmark: **73% parameter accuracy, 96% of runs matched**.

### A note on data

The travel example includes synthetic traces, clearly labeled in the `.lp` files. The order example is 100% real production data.

## Quick start

```bash
pip install clingo
git clone https://github.com/mirkokiefer/autocompile
cd autocompile

# 1. Compile: mine patterns from traces
python src/compile.py \
  --traces examples/travel-updates/traces.lp \
  --rules rules/mine_patterns_relaxed.lp \
  --output result.json

# 2. Benchmark: validate against traces
python src/benchmark.py \
  --compiled result.json \
  --holdout examples/travel-updates/traces.json

# 3. Codegen: emit executable pseudocode
python src/codegen.py --compiled result.json --target pseudo

# 4. Codegen: emit runnable job spec
python src/codegen.py --compiled result.json --target daslab --output job.json
```

## Compilation operations

autocompile applies compiler optimizations to observed workflows:

- **Constant folding** -- Steps that always produce the same output are replaced with the cached result
- **Strength reduction** -- Expensive steps are downgraded to cheaper equivalents where benchmarks confirm equivalence
- **Inlining / fusion** -- Sequential steps with deterministic data flow are fused into a single operation
- **Parallelization** -- Independent steps are scheduled concurrently
- **Dead code elimination** -- Steps whose outputs are never used downstream are removed
- **Branch compilation** -- Conditional execution patterns are inferred from co-occurrence data

## Model compilation

For steps that remain as `llm_invoke`, autocompile can test whether a cheaper model produces equivalent results. Using the inputs and outputs from existing traces as ground truth:

```
extract_booking_details:
  claude-sonnet-4-5   25/25 correct (baseline)
  qwen-3.5-35b        24/25 correct (96%)    downgrade candidate
  regex extraction     18/25 correct (72%)    not ready
```

*Status: WIP. The benchmarking framework supports this but model comparison is not yet implemented.*

## Datalog backends

The `datalog/` directory ports the monotonic fragment (~70% of the ASP rules) to three Datalog engines. All produce identical results to Clingo:

```
Engine         Solve time    Notes
─────────────────────────────────────────────
Clingo (ASP)      52ms       Full pipeline (choice rules + optimization)
Soufflé          175ms       Compiled Datalog (includes subprocess overhead)
egglog            39ms       Datalog + equality saturation
Python            15s        Reference implementation (pure Python, no deps)
```

The remaining ~30% (conflicting order resolution via choice rules and `#minimize`) stays in Clingo. See `datalog/` for details.

## Trace format

autocompile takes execution traces as JSON or ASP facts:

```json
{
  "runs": [
    {
      "id": "run_1",
      "steps": [
        {"id": "step_1", "tool": "gmail_search", "params": {"query": "flights"}, "status": "completed"},
        {"id": "step_2", "tool": "sheets_read", "depends_on": ["step_1"], "status": "completed"}
      ]
    }
  ]
}
```

See [spec/trace-format.md](spec/trace-format.md) for the full specification.

## Project structure

```
autocompile/
├── src/
│   ├── compile.py            # Trace → compiled workflow (ASP strategy)
│   ├── benchmark.py          # Validate compiled workflow against traces
│   └── codegen.py            # Compiled workflow → executable program
├── rules/
│   ├── mine_patterns.lp      # ASP rules (50% threshold)
│   └── mine_patterns_relaxed.lp  # ASP rules (25% threshold)
├── datalog/
│   ├── mine_patterns.dl      # Soufflé port of ASP rules
│   ├── souffle_compile.py    # Soufflé runner
│   ├── egglog_compile.py     # egglog runner
│   ├── engine.py             # Pure Python Datalog evaluator
│   └── compile.py            # Python engine runner
├── examples/
│   ├── travel-updates/       # 177 runs (27 real + 150 synthetic)
│   └── order-updates/        # 118 runs (100% real)
├── experiments/              # Cross-domain prototypes (robotics, lab, edge compute)
└── spec/
    └── trace-format.md       # Trace format specification
```

## Prior art

- **Compiler optimization** -- The operations are textbook. We apply them to workflows instead of instruction streams.
- **Trace-based JIT** (V8, LuaJIT) -- Observe runtime behavior to decide what to optimize. autocompile infers the program itself from traces.
- **Process mining** (Celonis) -- Discovers workflow models from event logs. autocompile compiles the discovered workflow into an executable program.
- **Program synthesis** -- Generates programs from input/output examples. autocompile applies this at the workflow step level.
- **Autonomous research** (autoresearch) -- LLM mutates code, keeps improvements. The LLM generates variations. autocompile discovers structure from observed variations instead.

## Status

Early-stage. The core pipeline works on agent workflow traces today. The same ASP rules are domain-generic -- they work on any process that produces sequential action traces. The `experiments/` directory has prototypes for robotics (LeRobot), autonomous lab protocols, and edge compute pipelines.

## License

MIT
