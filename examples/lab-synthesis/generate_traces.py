#!/usr/bin/env python3
"""
Generate realistic autonomous lab synthesis traces.

Simulates 25 runs of Suzuki cross-coupling reactions in a self-driving lab.
Based on real reaction protocols from the Open Reaction Database.

Reaction structure:
  Phase 0: prepare_catalyst || weigh_substrate  (parallel prep)
  Phase 1: charge_reactor                       (combine reagents)
  Phase 2: heat_reaction                        (run reaction)
  Phase 3: monitor_reaction                     (in-situ monitoring)
  Phase 4: cool_reaction                        (quench)
  Phase 5: extract_product                      (workup)
  Phase 6: purify_column | crystallize          (mutually exclusive purification)
  Phase 7: analyze_sample                       (characterization)

Expected discoveries by autocompile:
  - core_tool: prepare_catalyst, weigh_substrate, charge_reactor, heat_reaction,
               monitor_reaction, cool_reaction, extract_product, analyze_sample
  - parallel_pair: (prepare_catalyst, weigh_substrate)
  - fusion_candidate: (cool_reaction, extract_product)
  - mutually_exclusive: (purify_column, crystallize)
  - stable_param: catalyst=Pd_dppf_Cl2, solvent=DMF_water, temperature=80, ...
  - variable_param: substrate_smiles, substrate_mmol, reaction_time_min
  - conditional_tool: purify_column (~56%), crystallize (~28%)
"""

import random

random.seed(42)

NUM_RUNS = 25

# Substrates vary per run (different aryl halides)
SUBSTRATES = [
    "c1ccc(Br)cc1", "c1ccc(I)cc1", "c1cc(Br)ccc1C",
    "c1ccc(Br)c(F)c1", "c1cc(I)cc(OC)c1", "c1ccc(Br)cc1F",
    "c1ccc(I)cc1OC", "c1cc(Br)cc(C)c1", "c1ccc(Br)c(C)c1",
    "c1cc(I)ccc1N", "c1ccc(Br)cc1O", "c1cc(Br)cc(F)c1",
    "c1ccc(I)c(OC)c1", "c1cc(Br)ccc1CF3", "c1ccc(Br)cc1Cl",
    "c1cc(I)cc(C)c1", "c1ccc(Br)c(N)c1", "c1cc(Br)ccc1OC",
    "c1ccc(I)cc1C", "c1cc(Br)cc(Cl)c1", "c1ccc(Br)cc1",
    "c1cc(I)ccc1F", "c1ccc(Br)c(OC)c1", "c1cc(Br)ccc1N",
    "c1ccc(I)cc1Cl",
]

# Substrate amounts vary
MMOLS = ["0.50", "0.75", "1.00", "0.60", "0.80", "1.20", "0.90", "0.70"]

# Reaction times vary based on substrate reactivity
REACTION_TIMES = ["60", "90", "120", "45", "80", "100", "75", "110", "95", "55"]


def generate_run(run_idx):
    """Generate one synthesis experiment trace."""
    run_id = f"run_{run_idx}"
    step_num = 0
    lines = []

    substrate = SUBSTRATES[run_idx - 1]
    mmol = MMOLS[(run_idx - 1) % len(MMOLS)]
    rxn_time = REACTION_TIMES[(run_idx - 1) % len(REACTION_TIMES)]

    # Purification strategy: 14 runs use column, 7 use crystallize, 4 skip (direct to analysis)
    if run_idx <= 14:
        purify_method = "column"
    elif run_idx <= 21:
        purify_method = "crystallize"
    else:
        purify_method = None  # crude goes directly to analysis

    # Run 8 fails at monitoring (bad reaction), run 17 fails at extraction
    run_status = "completed"
    fail_at = None
    if run_idx == 8:
        run_status = "failed"
        fail_at = "monitor"
    if run_idx == 17:
        run_status = "failed"
        fail_at = "extract"

    def sid():
        nonlocal step_num
        step_num += 1
        return f"{run_id}_step_{step_num}"

    def call(step_id, tool, status="completed"):
        lines.append(f'call("{run_id}", "{step_id}", "{tool}", "{status}").')

    def dep(step_id, dep_id):
        lines.append(f'depends("{run_id}", "{step_id}", "{dep_id}").')

    def spawn(step_id, parent_id):
        lines.append(f'spawned_by("{run_id}", "{step_id}", "{parent_id}").')

    def param(step_id, key, value):
        lines.append(f'param("{run_id}", "{step_id}", "{key}", "{value}").')

    lines.append(f'job("{run_id}").')
    lines.append(f'job_status("{run_id}", "{run_status}").')

    # Step 1: prompt_user (experiment request)
    s1 = sid()
    call(s1, "prompt_user")

    # Step 2: llm_generate (plan experiment)
    s2 = sid()
    call(s2, "llm_generate")
    dep(s2, s1)

    # Step 3: prepare_catalyst (weigh and dissolve catalyst)
    s3 = sid()
    call(s3, "prepare_catalyst")
    dep(s3, s2)
    spawn(s3, s2)
    param(s3, "catalyst", "Pd_dppf_Cl2")
    param(s3, "loading_mol_pct", "3")
    param(s3, "solvent", "DMF")

    # Step 4: weigh_substrate (parallel with catalyst prep)
    s4 = sid()
    call(s4, "weigh_substrate")
    dep(s4, s2)
    spawn(s4, s2)
    param(s4, "substrate_smiles", substrate)
    param(s4, "amount_mmol", mmol)
    param(s4, "coupling_partner", "phenylboronic_acid")
    param(s4, "coupling_mmol", "1.50")

    # Step 5: llm_generate (reagents ready)
    s5 = sid()
    call(s5, "llm_generate")
    dep(s5, s3)
    dep(s5, s4)

    # Step 6: charge_reactor (combine everything)
    s6 = sid()
    call(s6, "charge_reactor")
    dep(s6, s5)
    spawn(s6, s5)
    param(s6, "solvent", "DMF_water")
    param(s6, "solvent_volume_mL", "5.0")
    param(s6, "base", "K2CO3")
    param(s6, "base_equiv", "2.0")
    param(s6, "atmosphere", "N2")

    # Step 7: llm_generate
    s7 = sid()
    call(s7, "llm_generate")
    dep(s7, s6)

    # Step 8: heat_reaction
    s8 = sid()
    call(s8, "heat_reaction")
    dep(s8, s7)
    spawn(s8, s7)
    param(s8, "temperature_C", "80")
    param(s8, "stir_rpm", "500")
    param(s8, "time_min", rxn_time)

    # Step 9: llm_generate
    s9 = sid()
    call(s9, "llm_generate")
    dep(s9, s8)

    # Step 10: monitor_reaction
    s10 = sid()
    status = "failed" if fail_at == "monitor" else "completed"
    call(s10, "monitor_reaction", status)
    dep(s10, s9)
    spawn(s10, s9)
    param(s10, "method", "TLC")
    param(s10, "sample_volume_uL", "50")

    if fail_at == "monitor":
        s_llm = sid()
        call(s_llm, "llm_generate")
        dep(s_llm, s10)
        s_notify = sid()
        call(s_notify, "notify_user")
        dep(s_notify, s_llm)
        spawn(s_notify, s_llm)
        return lines

    # Step 11: llm_generate
    s11 = sid()
    call(s11, "llm_generate")
    dep(s11, s10)

    # Step 12: cool_reaction
    s12 = sid()
    call(s12, "cool_reaction")
    dep(s12, s11)
    spawn(s12, s11)
    param(s12, "target_temp_C", "25")
    param(s12, "method", "air_cool")

    # Step 13: llm_generate
    s13 = sid()
    call(s13, "llm_generate")
    dep(s13, s12)

    # Step 14: extract_product
    s14 = sid()
    status = "failed" if fail_at == "extract" else "completed"
    call(s14, "extract_product", status)
    dep(s14, s13)
    spawn(s14, s13)
    param(s14, "extraction_solvent", "ethyl_acetate")
    param(s14, "washes", "3")
    param(s14, "dry_agent", "Na2SO4")

    if fail_at == "extract":
        s_llm = sid()
        call(s_llm, "llm_generate")
        dep(s_llm, s14)
        s_notify = sid()
        call(s_notify, "notify_user")
        dep(s_notify, s_llm)
        spawn(s_notify, s_llm)
        return lines

    if purify_method:
        # Step 15: llm_generate
        s15 = sid()
        call(s15, "llm_generate")
        dep(s15, s14)

        if purify_method == "column":
            s_purify = sid()
            call(s_purify, "purify_column")
            dep(s_purify, s15)
            spawn(s_purify, s15)
            param(s_purify, "stationary_phase", "silica_gel")
            param(s_purify, "eluent", "hexane_ethyl_acetate")
            param(s_purify, "gradient", "10_to_40_pct")
        else:
            s_purify = sid()
            call(s_purify, "crystallize")
            dep(s_purify, s15)
            spawn(s_purify, s15)
            param(s_purify, "solvent", "ethanol_water")
            param(s_purify, "temperature_C", "4")
            param(s_purify, "time_hours", "12")

        s_prev = s_purify
    else:
        s_prev = s14

    # Step: llm_generate
    s_llm_a = sid()
    call(s_llm_a, "llm_generate")
    dep(s_llm_a, s_prev)

    # Step: analyze_sample
    s_analyze = sid()
    call(s_analyze, "analyze_sample")
    dep(s_analyze, s_llm_a)
    spawn(s_analyze, s_llm_a)
    param(s_analyze, "method", "HPLC")
    param(s_analyze, "column", "C18")
    param(s_analyze, "detector", "UV_254nm")

    # Step: notify_user
    s_llm_end = sid()
    call(s_llm_end, "llm_generate")
    dep(s_llm_end, s_analyze)
    s_notify = sid()
    call(s_notify, "notify_user")
    dep(s_notify, s_llm_end)
    spawn(s_notify, s_llm_end)

    return lines


def main():
    output = []
    output.append("% === Autonomous Lab: Suzuki Cross-Coupling Synthesis ===")
    output.append(f"% {NUM_RUNS} experiment runs in a self-driving chemistry lab")
    output.append("% Reaction: Aryl halide + Phenylboronic acid -> Biaryl (Pd-catalyzed)")
    output.append("% Generated by generate_traces.py")
    output.append("")

    for i in range(1, NUM_RUNS + 1):
        desc = "column purification"
        if i > 14 and i <= 21:
            desc = "crystallization"
        elif i > 21:
            desc = "crude analysis (no purification)"
        if i == 8:
            desc = "reaction failed at monitoring"
        if i == 17:
            desc = "extraction failed"
        output.append(f"% === Experiment {i}: {desc} ===")
        output.extend(generate_run(i))
        output.append("")

    print("\n".join(output))


if __name__ == "__main__":
    main()
