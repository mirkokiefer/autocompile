# Autocompile: Cross-Domain Results

One pipeline, zero code changes, four domains. The same 208 lines of ASP rules
discover meaningful optimization patterns from agent workflows, robot demonstrations,
autonomous lab experiments, and distributed edge compute traces.

## Results Summary

| Domain | Traces | Core Tools | Compiled | Parallel | Fusion | Exclusive | Conditional |
|--------|--------|-----------|----------|----------|--------|-----------|-------------|
| Agent workflows | 177 | 5/11 (45%) | ✓ | gmail_search ∥ sheets_read | gmail + threads | — | threads (53%) |
| **Robotics** | 20 | 7/9 (78%) | ✓ | sense_object ∥ sense_target | move_arm + close_gripper | — | check_grasp (40%) |
| **Lab synthesis** | 25 | 9/10 (90%) | ✓ | prepare_catalyst ∥ weigh_substrate | cool + extract | crystallize ⊕ column | purify_column (52%) |
| **Edge compute** | 25 | 5/7 (71%) | ✓ | 3× read_sensor | preprocess + inference | recalibrate ⊕ alert | send_alert (36%) |

## Robotics: Pick-and-Place

20 teleoperated demonstrations → compiled manipulation program.

```
Phase 0:  sense_object(sensor=rgbd) || sense_target(sensor=rgbd)
Phase 1:  move_arm(approach_height=0.15, speed=0.5)
Phase 2:  close_gripper(force=0.8)
Phase 3:  move_to_target(carry_height=0.25, speed=0.3)
Phase 4:  open_gripper(release_speed=0.2)
Phase 5:  retract_arm(retract_height=0.35)
```

**Discovered primitives (fusion candidates):**
- `move_arm + close_gripper` → "grasp" primitive
- `open_gripper + retract_arm` → "release and retreat" primitive

**Compiled constants:** approach_height, gripper_force, carry_height, retract_height, speeds
**Runtime variables:** object position (target_x, target_y), target zone, grip width
**Conditional:** check_grasp (40% of demos), adjust_grip (10%, conditional on check_grasp)

## Lab Synthesis: Suzuki Cross-Coupling

25 automated experiments → compiled reaction protocol.

```
Phase 0:  prepare_catalyst(Pd_dppf_Cl2, 3 mol%) || weigh_substrate(phenylboronic_acid)
Phase 1:  charge_reactor(DMF/water, K2CO3, N2 atmosphere)
Phase 2:  heat_reaction(80°C, 500 rpm)
Phase 3:  monitor_reaction(TLC, 50 µL)
Phase 4:  cool_reaction(25°C, air)
Phase 5:  extract_product(EtOAc, 3 washes, Na2SO4)
Phase 6:  purify_column(silica, hex/EtOAc 10→40%)  [52% of runs]
Phase 7:  analyze_sample(HPLC, C18, UV 254nm)
```

**Branch alternatives discovered:** `crystallize XOR purify_column` — the system found
that these two purification methods never co-occur, without any chemistry knowledge.

**Compiled constants:** catalyst, temperature, solvent, base, atmosphere, stir speed,
extraction conditions, column conditions, analytical method.
**Runtime variables:** substrate (SMILES), amount (mmol), reaction time.

## Edge Compute: Anomaly Detection Pipeline

25 pipeline runs across ESP32 / Raspberry Pi / Jetson / Mac Studio.

```
Phase 0:  read_sensor × 3 (esp32_1: temp, esp32_2: vibration, esp32_3: pressure)
Phase 1:  preprocess_data(device=pi_1, normalize, window=256)
Phase 2:  run_inference(device=jetson_1, model=anomaly_v3, fp16)
Phase 3:  aggregate_results(device=mac_studio_1, json)
Phase 4:  store_results(device=mac_studio_1, timescaledb, 90 days)
```

**Device affinity discovered:**
- `run_inference` → always `jetson_1` (stable param)
- `aggregate_results` + `store_results` → always `mac_studio_1`
- `preprocess_data` → mostly `pi_1` (stable, but sometimes `mac_studio_1`)

**Fusion = co-location:** `preprocess_data + run_inference` — push preprocessing
to the same device as inference to eliminate data transfer.

**Conditional:** `send_alert` (36% of runs), `recalibrate_sensor` (16%).
**Mutually exclusive:** `recalibrate_sensor XOR send_alert` — never both in same run.

## What This Demonstrates

The same ASP rules discover domain-appropriate optimizations without modification:

| ASP pattern | Agent workflows | Robotics | Lab synthesis | Edge compute |
|-------------|----------------|----------|---------------|--------------|
| `core_tool` | Core API calls | Manipulation primitives | Essential reaction steps | Pipeline stages |
| `parallel_pair` | Multi-account search | Dual perception | Parallel reagent prep | Multi-sensor reads |
| `fusion_candidate` | Merge API calls | Discover motor primitives | One-pot reactions | Co-locate compute |
| `mutually_exclusive` | — | — | Purification methods | Alert vs recalibrate |
| `stable_param` | API config | Physical constants | Reaction conditions | Device affinity |
| `variable_param` | Search queries | Object positions | Substrate identity | Batch ID, threshold |
| `conditional_tool` | Follow-up reads | Grasp verification | Optional purification | Anomaly alerting |

The patterns mean different things in each domain, but the discovery mechanism is identical.
This is a universal trace compiler.
