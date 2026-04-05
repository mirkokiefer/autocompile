#!/usr/bin/env python3
"""
mimicel_to_asp.py — Convert MIMICEL event logs to autocompile ASP traces.

MIMICEL is an event log extracted from MIMIC-IV Emergency Department data
(PhysioNet, credentialed access). Each ED stay is a trace of clinical events.

Input: mimicel.csv (from PhysioNet) or synthetic sample
Output: ASP facts for autocompile (call/depends/param)

Usage:
    # Convert real MIMICEL data (filter to sepsis cases)
    python src/mimicel_to_asp.py \
        --input data-private/mimicel.csv \
        --filter-icd "A41,R65" \
        --max-cases 500 \
        --output examples/medical-ed/traces.lp

    # Convert all cases (no filter)
    python src/mimicel_to_asp.py --input mimicel.csv --output traces.lp

    # Generate synthetic sample for testing
    python src/mimicel_to_asp.py --synthetic 200 --output examples/medical-ed/traces.lp
"""

import argparse
import csv
import io
import random
import sys
from collections import defaultdict
from pathlib import Path


# Activity names in MIMICEL
ACTIVITIES = [
    "ED_registration",       # "Enter the ED" → renamed for clarity as tool name
    "triage",                # "Triage in the ED"
    "vitals_check",          # "Take routine vital signs"
    "med_reconciliation",    # "Medicine reconciliation"
    "med_dispensation",      # "Medicine dispensation"
    "ED_discharge",          # "Discharge from the ED"
]

# Map from MIMICEL activity names to our tool names
ACTIVITY_MAP = {
    "Enter the ED": "ED_registration",
    "Triage in the ED": "triage",
    "Take routine vital signs": "vitals_check",
    "Medicine reconciliation": "med_reconciliation",
    "Medicine dispensation": "med_dispensation",
    "Discharge from the ED": "ED_discharge",
}

# Columns that become parameters for each activity type
PARAM_COLUMNS = {
    "ED_registration": ["gender", "race", "arrival_transport"],
    "triage": ["acuity", "chiefcomplaint", "temperature", "heartrate",
               "resprate", "o2sat", "sbp", "dbp", "pain"],
    "vitals_check": ["temperature", "heartrate", "resprate", "o2sat",
                     "sbp", "dbp", "pain", "rhythm"],
    "med_reconciliation": ["name", "gsn", "ndc", "etcdescription"],
    "med_dispensation": ["name", "gsn"],
    "ED_discharge": ["disposition"],
}


def load_mimicel(path: Path, filter_icd: list[str] | None = None,
                 max_cases: int | None = None) -> dict[str, list[dict]]:
    """Load MIMICEL CSV, group by stay_id, optionally filter by ICD codes."""
    cases = defaultdict(list)
    icd_stays = set()

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # If filtering by ICD, first pass to find matching stay_ids
    if filter_icd:
        prefixes = [code.strip().upper() for code in filter_icd]
        for row in rows:
            icd = (row.get("icd_code") or "").strip().upper()
            if any(icd.startswith(p) for p in prefixes):
                icd_stays.add(row["stay_id"])
        print(f"ICD filter: {len(icd_stays)} stays match codes {prefixes}")

    # Group events by stay_id
    for row in rows:
        stay_id = row["stay_id"]
        if filter_icd and stay_id not in icd_stays:
            continue
        cases[stay_id].append(row)

    # Sort events within each case by timestamp
    for stay_id in cases:
        cases[stay_id].sort(key=lambda r: r.get("timestamps", ""))

    # Limit number of cases
    if max_cases and len(cases) > max_cases:
        keys = list(cases.keys())[:max_cases]
        cases = {k: cases[k] for k in keys}

    return dict(cases)


def case_to_asp(stay_id: str, events: list[dict], run_id: str) -> list[str]:
    """Convert one ED stay to ASP facts."""
    facts = []
    facts.append(f'job("{run_id}").')

    # Case-level attributes from first event
    first = events[0]
    if first.get("disposition"):
        facts.append(f'job_status("{run_id}", "{_clean(first["disposition"])}").')

    prev_step_id = None
    for i, event in enumerate(events):
        step_id = f"{run_id}_step_{i+1}"
        activity_raw = event.get("activity", "unknown")
        tool = ACTIVITY_MAP.get(activity_raw, _clean(activity_raw))

        facts.append(f'call("{run_id}", "{step_id}", "{tool}", "completed").')

        # Dependencies: each event depends on the previous
        if prev_step_id:
            facts.append(f'depends("{run_id}", "{step_id}", "{prev_step_id}").')

        # First event spawns subsequent events (simple model)
        if i == 0:
            pass  # root event
        elif i == 1:
            # Triage spawned by registration
            facts.append(f'spawned_by("{run_id}", "{step_id}", "{run_id}_step_1").')
        else:
            # Subsequent events spawned by triage (step 2) if it exists,
            # otherwise by registration (step 1)
            parent = f"{run_id}_step_2" if len(events) > 1 else f"{run_id}_step_1"
            facts.append(f'spawned_by("{run_id}", "{step_id}", "{parent}").')

        # Parameters: non-null columns relevant to this activity
        param_cols = PARAM_COLUMNS.get(tool, [])
        for col in param_cols:
            val = event.get(col, "")
            if val and val.strip() and val.strip().lower() != "nan":
                val_clean = _clean(val.strip())
                if val_clean:
                    facts.append(f'param("{run_id}", "{step_id}", "{col}", "{val_clean}").')

        prev_step_id = step_id

    return facts


def _clean(s: str) -> str:
    """Clean string for ASP: escape quotes, limit length."""
    s = s.replace('"', '\\"').replace("\n", " ").replace("\r", "")
    if len(s) > 100:
        s = s[:97] + "..."
    return s


def generate_synthetic(n_cases: int) -> dict[str, list[dict]]:
    """Generate synthetic MIMICEL-like cases for testing.

    Based on real ED visit patterns:
    - Every visit: registration → triage → discharge
    - Most visits: 2-6 vital sign checks
    - Some visits: medication reconciliation and/or dispensation
    - Acuity 1-5 (1=most severe, 5=least)
    - Higher acuity → more vitals checks, more meds, longer stay
    """
    cases = {}

    chief_complaints = [
        "chest pain", "abdominal pain", "shortness of breath",
        "headache", "fever", "back pain", "nausea/vomiting",
        "altered mental status", "laceration", "fall",
        "dizziness", "cough", "weakness", "syncope",
        "allergic reaction", "seizure", "overdose",
    ]

    dispositions = [
        "home", "admitted", "transfer", "AMA", "expired",
    ]
    disposition_weights = [0.55, 0.30, 0.05, 0.08, 0.02]

    medications = [
        "acetaminophen", "ibuprofen", "morphine", "ondansetron",
        "ketorolac", "normal_saline", "ceftriaxone", "vancomycin",
        "metoprolol", "aspirin", "heparin", "lorazepam",
        "diphenhydramine", "famotidine", "piperacillin_tazobactam",
        "epinephrine", "norepinephrine", "amoxicillin",
    ]

    med_classes = {
        "acetaminophen": "analgesic", "ibuprofen": "NSAID",
        "morphine": "opioid", "ondansetron": "antiemetic",
        "ketorolac": "NSAID", "normal_saline": "IV_fluid",
        "ceftriaxone": "antibiotic", "vancomycin": "antibiotic",
        "metoprolol": "beta_blocker", "aspirin": "antiplatelet",
        "heparin": "anticoagulant", "lorazepam": "benzodiazepine",
        "diphenhydramine": "antihistamine", "famotidine": "antacid",
        "piperacillin_tazobactam": "antibiotic",
        "epinephrine": "vasopressor", "norepinephrine": "vasopressor",
        "amoxicillin": "antibiotic",
    }

    arrival_modes = ["ambulance", "walk-in", "police", "helicopter"]
    arrival_weights = [0.35, 0.55, 0.08, 0.02]

    genders = ["M", "F"]
    races = ["WHITE", "BLACK", "HISPANIC", "ASIAN", "OTHER"]

    for i in range(n_cases):
        stay_id = str(10000 + i)
        events = []
        base_time = f"2150-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
        hour = random.randint(0, 23)
        minute = random.randint(0, 59)

        acuity = random.choices([1, 2, 3, 4, 5], weights=[0.05, 0.20, 0.40, 0.25, 0.10])[0]
        complaint = random.choice(chief_complaints)
        gender = random.choice(genders)
        race = random.choice(races)
        arrival = random.choices(arrival_modes, weights=arrival_weights)[0]
        disposition = random.choices(dispositions, weights=disposition_weights)[0]

        # Higher acuity = more likely admitted
        if acuity <= 2:
            disposition = random.choices(
                ["admitted", "home", "transfer"],
                weights=[0.65, 0.25, 0.10]
            )[0]

        # Base vital signs (vary by acuity)
        base_hr = random.gauss(85 + (5 - acuity) * 8, 15)
        base_sbp = random.gauss(130 - (5 - acuity) * 5, 20)
        base_dbp = random.gauss(80 - (5 - acuity) * 3, 12)
        base_temp = random.gauss(37.0 + (5 - acuity) * 0.2, 0.5)
        base_rr = random.gauss(18 + (5 - acuity) * 2, 4)
        base_o2 = random.gauss(97 - (5 - acuity) * 1.5, 2)

        def make_timestamp(h, m):
            return f"{base_time} {h:02d}:{m:02d}:00"

        # 1. Registration
        events.append({
            "stay_id": stay_id, "subject_id": str(20000 + i),
            "timestamps": make_timestamp(hour, minute),
            "activity": "Enter the ED",
            "gender": gender, "race": race,
            "arrival_transport": arrival,
            "disposition": disposition,
            "acuity": "", "chiefcomplaint": "",
            "temperature": "", "heartrate": "", "resprate": "",
            "o2sat": "", "sbp": "", "dbp": "", "pain": "",
            "rhythm": "", "name": "", "gsn": "", "ndc": "",
            "etcdescription": "",
        })

        # 2. Triage (1-5 min after registration)
        minute += random.randint(1, 5)
        pain = random.randint(0, 10) if random.random() > 0.2 else ""
        events.append({
            "stay_id": stay_id, "subject_id": str(20000 + i),
            "timestamps": make_timestamp(hour, minute % 60),
            "activity": "Triage in the ED",
            "gender": "", "race": "", "arrival_transport": "",
            "disposition": "",
            "acuity": str(acuity), "chiefcomplaint": complaint,
            "temperature": f"{base_temp:.1f}",
            "heartrate": f"{base_hr:.0f}",
            "resprate": f"{base_rr:.0f}",
            "o2sat": f"{max(80, base_o2):.0f}",
            "sbp": f"{base_sbp:.0f}",
            "dbp": f"{base_dbp:.0f}",
            "pain": str(pain) if pain != "" else "",
            "rhythm": "", "name": "", "gsn": "", "ndc": "",
            "etcdescription": "",
        })

        # 3. Vital sign checks (2-8, more for sicker patients)
        n_vitals = max(1, int(random.gauss(2 + (5 - acuity) * 1.2, 1.5)))
        for v in range(n_vitals):
            minute += random.randint(15, 60)
            hr_drift = random.gauss(0, 5)
            events.append({
                "stay_id": stay_id, "subject_id": str(20000 + i),
                "timestamps": make_timestamp((hour + minute // 60) % 24, minute % 60),
                "activity": "Take routine vital signs",
                "gender": "", "race": "", "arrival_transport": "",
                "disposition": "", "acuity": "", "chiefcomplaint": "",
                "temperature": f"{base_temp + random.gauss(0, 0.3):.1f}",
                "heartrate": f"{base_hr + hr_drift:.0f}",
                "resprate": f"{base_rr + random.gauss(0, 2):.0f}",
                "o2sat": f"{max(80, base_o2 + random.gauss(0, 1)):.0f}",
                "sbp": f"{base_sbp + random.gauss(0, 8):.0f}",
                "dbp": f"{base_dbp + random.gauss(0, 5):.0f}",
                "pain": "",
                "rhythm": random.choice(["normal_sinus", "normal_sinus",
                                         "sinus_tachycardia", "atrial_fibrillation"]),
                "name": "", "gsn": "", "ndc": "", "etcdescription": "",
            })

        # 4. Medication reconciliation (60% of visits)
        if random.random() < 0.60:
            n_meds_recon = random.randint(1, 4)
            for _ in range(n_meds_recon):
                minute += random.randint(5, 20)
                med = random.choice(medications)
                events.append({
                    "stay_id": stay_id, "subject_id": str(20000 + i),
                    "timestamps": make_timestamp((hour + minute // 60) % 24, minute % 60),
                    "activity": "Medicine reconciliation",
                    "gender": "", "race": "", "arrival_transport": "",
                    "disposition": "", "acuity": "", "chiefcomplaint": "",
                    "temperature": "", "heartrate": "", "resprate": "",
                    "o2sat": "", "sbp": "", "dbp": "", "pain": "",
                    "rhythm": "",
                    "name": med, "gsn": f"gsn_{med[:4]}",
                    "ndc": f"ndc_{random.randint(1000,9999)}",
                    "etcdescription": med_classes.get(med, "other"),
                })

        # 5. Medication dispensation (70% of visits, higher for sicker)
        dispense_prob = 0.50 + (5 - acuity) * 0.10
        if random.random() < dispense_prob:
            # Sicker patients get more meds
            n_meds = max(1, int(random.gauss(1 + (5 - acuity) * 0.8, 1)))
            # Choose meds based on complaint/acuity
            if complaint == "chest pain":
                likely_meds = ["aspirin", "morphine", "heparin", "metoprolol", "normal_saline"]
            elif complaint in ("fever", "cough"):
                likely_meds = ["acetaminophen", "ceftriaxone", "normal_saline"]
            elif complaint == "abdominal pain":
                likely_meds = ["ondansetron", "morphine", "ketorolac", "normal_saline"]
            elif complaint == "shortness of breath":
                likely_meds = ["normal_saline", "ceftriaxone", "epinephrine"]
            elif complaint == "seizure":
                likely_meds = ["lorazepam", "normal_saline"]
            elif complaint == "allergic reaction":
                likely_meds = ["epinephrine", "diphenhydramine", "famotidine"]
            elif complaint == "overdose":
                likely_meds = ["normal_saline", "acetaminophen"]
            else:
                likely_meds = ["acetaminophen", "ibuprofen", "ondansetron", "normal_saline"]

            chosen = random.sample(likely_meds, min(n_meds, len(likely_meds)))
            for med in chosen:
                minute += random.randint(5, 30)
                events.append({
                    "stay_id": stay_id, "subject_id": str(20000 + i),
                    "timestamps": make_timestamp((hour + minute // 60) % 24, minute % 60),
                    "activity": "Medicine dispensation",
                    "gender": "", "race": "", "arrival_transport": "",
                    "disposition": "", "acuity": "", "chiefcomplaint": "",
                    "temperature": "", "heartrate": "", "resprate": "",
                    "o2sat": "", "sbp": "", "dbp": "", "pain": "",
                    "rhythm": "",
                    "name": med, "gsn": f"gsn_{med[:4]}", "ndc": "",
                    "etcdescription": "",
                })

        # 6. Discharge (30 min - 8h after last event)
        minute += random.randint(30, 480 if acuity <= 2 else 120)
        events.append({
            "stay_id": stay_id, "subject_id": str(20000 + i),
            "timestamps": make_timestamp((hour + minute // 60) % 24, minute % 60),
            "activity": "Discharge from the ED",
            "gender": "", "race": "", "arrival_transport": "",
            "disposition": disposition,
            "acuity": "", "chiefcomplaint": "",
            "temperature": "", "heartrate": "", "resprate": "",
            "o2sat": "", "sbp": "", "dbp": "", "pain": "",
            "rhythm": "", "name": "", "gsn": "", "ndc": "",
            "etcdescription": "",
        })

        cases[stay_id] = events

    return cases


def write_asp(cases: dict[str, list[dict]], output: Path):
    """Write all cases as ASP facts."""
    lines = []
    lines.append(f"% === ED Treatment Pathway Traces ({len(cases)} cases) ===")
    lines.append(f"% Converted from MIMICEL (MIMIC-IV Emergency Department event logs)")
    lines.append("")

    for i, (stay_id, events) in enumerate(cases.items()):
        run_id = f"run_{i+1}"
        # Get case metadata
        first = events[0]
        complaint = ""
        acuity = ""
        for e in events:
            if e.get("chiefcomplaint"):
                complaint = e["chiefcomplaint"]
            if e.get("acuity"):
                acuity = e["acuity"]

        disposition = first.get("disposition", "unknown")
        lines.append(f"% === Case {run_id}: {complaint} (acuity {acuity}) — {disposition} ===")

        asp_facts = case_to_asp(stay_id, events, run_id)
        lines.extend(asp_facts)
        lines.append("")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))
    print(f"Written {len(cases)} cases to {output}")

    # Stats
    total_events = sum(len(e) for e in cases.values())
    activity_counts = defaultdict(int)
    for events in cases.values():
        for e in events:
            activity_counts[ACTIVITY_MAP.get(e["activity"], e["activity"])] += 1

    print(f"Total events: {total_events}")
    print(f"Events per case: {total_events / len(cases):.1f} mean")
    print("Activity distribution:")
    for act, count in sorted(activity_counts.items(), key=lambda x: -x[1]):
        print(f"  {act}: {count} ({count * 100 // total_events}%)")


def main():
    parser = argparse.ArgumentParser(description="Convert MIMICEL to autocompile traces")
    parser.add_argument("--input", help="Path to mimicel.csv")
    parser.add_argument("--synthetic", type=int, help="Generate N synthetic cases instead")
    parser.add_argument("--filter-icd", help="Comma-separated ICD code prefixes to filter")
    parser.add_argument("--max-cases", type=int, help="Maximum number of cases")
    parser.add_argument("--output", default="examples/medical-ed/traces.lp")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    output = Path(args.output) if Path(args.output).is_absolute() else project_root / args.output

    if args.synthetic:
        print(f"Generating {args.synthetic} synthetic ED cases...")
        random.seed(42)
        cases = generate_synthetic(args.synthetic)
    elif args.input:
        input_path = Path(args.input) if Path(args.input).is_absolute() else project_root / args.input
        if not input_path.exists():
            sys.exit(f"ERROR: {input_path} not found")
        filter_icd = args.filter_icd.split(",") if args.filter_icd else None
        cases = load_mimicel(input_path, filter_icd, args.max_cases)
    else:
        sys.exit("ERROR: Provide --input or --synthetic")

    write_asp(cases, output)


if __name__ == "__main__":
    main()
