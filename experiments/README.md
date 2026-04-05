# Experiments

Exploratory work extending autocompile beyond agent workflows. These are prototypes and research directions — not validated to the same level as the core pipeline.

## Domains (synthetic traces)

The robotics, lab synthesis, and edge compute examples use **generated traces** from scripts that encode the expected structure. They demonstrate that the ASP rules are domain-generic, but the "discoveries" are circular — the generator bakes in the answer.

- `examples/robotics-pick-place/` — 20 synthetic pick-and-place demos
- `examples/lab-synthesis/` — 25 synthetic Suzuki coupling reactions
- `examples/edge-compute/` — 25 synthetic anomaly detection pipeline runs

See `DOMAINS.md` for cross-domain results.

## LeRobot (real robot data)

Real trajectories from HuggingFace LeRobot datasets, converted to ASP facts.

- `examples/lerobot-pick-place/` — 50 episodes from `lerobot/ucsd_pick_and_place_dataset`
- `examples/lerobot-discovered/` — Primitives discovered via changepoint detection
- `examples/single-demo/` — Single demo + LLM world model → compiled program

## Source

- `src/segment_discover.py` — Changepoint-based primitive discovery from trajectories
- `src/single_demo_compile.py` — Compile from one demo using LLM-generated variations
- `src/scene_synth.py` — Analysis-by-synthesis (VLM + USD + Blender)
- `src/scene_loop.py` — Dehydrate → Clingo search → Hydrate loop
- `src/scene_layout.py` — Clingo-powered spatial layout solver
- `src/world_model.py` — LLM world model extraction
- `src/demo_diff.py` — Rewrite rule discovery from demo quality diffing
- `src/trajectory_rewrite.py` — Trajectory optimization via rewrite rules
- `src/llm_baseline.py` — Compare autocompile vs pure LLM compilation
- `src/lerobot_to_asp.py` — LeRobot dataset → ASP fact converter
- `src/bench.py` — Unified benchmark harness
- `src/rerun_viewer.py` — Rerun visualization for trajectory inspection
