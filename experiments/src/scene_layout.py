#!/usr/bin/env python3
"""
scene_layout.py — Clingo-powered scene layout from spatial constraints.

Instead of LLM iterating on object placement (slow, expensive), we:
  1. VLM identifies objects + spatial relationships (one call)
  2. Convert relationships to ASP constraints
  3. Clingo solves for positions (milliseconds)
  4. Generate USD from solved positions
  5. Render once, VLM verifies once

This replaces 5+ LLM adjustment iterations with 1 Clingo solve.

Usage:
    uv run python src/scene_layout.py --constraints constraints.json --output scene.usda
"""

import json
import sys
from pathlib import Path

try:
    import clingo
except ImportError:
    sys.exit("ERROR: clingo required")


# ============================================================================
# ASP Scene Layout Solver
# ============================================================================

LAYOUT_RULES = """
% Scene layout solver — place objects satisfying spatial constraints
% Positions are in centimeters (integer), converted to meters for USD output

% Grid resolution: 5cm steps (coarse but fast)
#const grid_min = -10.
#const grid_max = 10.
#const grid_z_min = 0.
#const grid_z_max = 24.

% Each object gets exactly one position
{ pos_x(O, X) : X = grid_min..grid_max } = 1 :- object(O).
{ pos_y(O, Y) : Y = grid_min..grid_max } = 1 :- object(O).
{ pos_z(O, Z) : Z = grid_z_min..grid_z_max } = 1 :- object(O).

% --- Spatial constraints ---

% left_of(A, B): A.x < B.x
:- left_of(A, B), pos_x(A, XA), pos_x(B, XB), XA >= XB.

% right_of(A, B): A.x > B.x
:- right_of(A, B), pos_x(A, XA), pos_x(B, XB), XA <= XB.

% above(A, B): A.z > B.z
:- above(A, B), pos_z(A, ZA), pos_z(B, ZB), ZA <= ZB.

% on_surface(A, B): A.z = B.z + B.height, A within B's x/y bounds
:- on_surface(A, B), pos_z(A, ZA), pos_z(B, ZB), height(B, HB), ZA != ZB + HB.
:- on_surface(A, B), pos_x(A, XA), pos_x(B, XB), width(B, WB), |XA - XB| > WB / 2.
:- on_surface(A, B), pos_y(A, YA), pos_y(B, YB), depth(B, DB), |YA - YB| > DB / 2.

% behind(A, B): A.y < B.y (farther from camera)
:- behind(A, B), pos_y(A, YA), pos_y(B, YB), YA >= YB.

% in_front(A, B): A.y > B.y
:- in_front(A, B), pos_y(A, YA), pos_y(B, YB), YA <= YB.

% center_x(A): A near x=0
:- center_x(A), pos_x(A, X), |X| > 10.

% flush_back(A, B): A.y ≈ B.y (same depth)
:- flush_back(A, B), pos_y(A, YA), pos_y(B, YB), |YA - YB| > 5.

% Non-overlapping omitted for speed — rely on spatial constraints instead

% --- Optimization: prefer compact layouts ---
#minimize { |X|@1,O : pos_x(O, X), object(O) }.
#minimize { |Y|@1,O : pos_y(O, Y), object(O) }.

% --- Output ---
#show pos_x/2.
#show pos_y/2.
#show pos_z/2.
"""


def solve_layout(objects, constraints):
    """
    Solve scene layout using Clingo.

    objects: dict of {name: {width, depth, height, color}}
    constraints: list of {type: "left_of"|"on_surface"|..., args: [A, B]}

    Returns: dict of {name: {x, y, z}} in meters
    """
    facts = []

    # Object declarations with sizes (in cm)
    for name, props in objects.items():
        facts.append(f'object("{name}").')
        facts.append(f'width("{name}", {int(props["width"] * 20)}).')
        facts.append(f'depth("{name}", {int(props["depth"] * 20)}).')
        facts.append(f'height("{name}", {int(props["height"] * 20)}).')

    # Constraints
    for c in constraints:
        ctype = c["type"]
        args = c["args"]
        if len(args) == 1:
            facts.append(f'{ctype}("{args[0]}").')
        elif len(args) == 2:
            facts.append(f'{ctype}("{args[0]}", "{args[1]}").')

    # Combine facts + rules
    program = "\n".join(facts) + "\n" + LAYOUT_RULES

    # Solve
    ctl = clingo.Control(["--opt-mode=optN", "1", "--rand-freq=0.1"])
    ctl.add("base", [], program)
    ctl.ground([("base", [])])

    positions = {}

    def on_model(model):
        positions.clear()
        for atom in model.symbols(shown=True):
            name = atom.arguments[0].string
            val = atom.arguments[1].number
            if name not in positions:
                positions[name] = {}
            if atom.name == "pos_x":
                positions[name]["x"] = val / 20.0  # grid units to meters
            elif atom.name == "pos_y":
                positions[name]["y"] = val / 20.0
            elif atom.name == "pos_z":
                positions[name]["z"] = val / 20.0

    handle = ctl.solve(on_model=on_model)

    if not handle.satisfiable:
        print("UNSATISFIABLE — constraints are contradictory")
        return None

    return positions


def positions_to_usd(objects, positions):
    """Generate USD from solved positions."""
    lines = ['#usda 1.0', '(', '    metersPerUnit = 1.0', '    upAxis = "Z"', ')']

    shape_map = {"cylinder": "Cylinder", "cube": "Cube", "sphere": "Sphere"}

    for name, props in objects.items():
        pos = positions.get(name, {"x": 0, "y": 0, "z": 0})
        shape = shape_map.get(props.get("shape", "cube"), "Cube")
        w, d, h = props["width"] / 2, props["depth"] / 2, props["height"] / 2
        r, g, b = props.get("color", [0.5, 0.5, 0.5])

        # Center of object
        cx = pos["x"]
        cy = pos["y"]
        cz = pos["z"] + h  # pos_z is bottom, USD translate is center

        clean_name = name.replace(" ", "_").replace("-", "_")
        lines.append(f'def {shape} "{clean_name}"')
        lines.append("{")
        lines.append(f'    double3 xformOp:translate = ({cx:.3f}, {cy:.3f}, {cz:.3f})')
        lines.append(f'    double3 xformOp:scale = ({w:.3f}, {d:.3f}, {h:.3f})')
        lines.append(f'    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]')
        lines.append(f'    color3f[] primvars:displayColor = [({r:.2f}, {g:.2f}, {b:.2f})]')
        lines.append("}")

    return "\n".join(lines)


def demo_kitchen_scene():
    """Demo: solve the toy kitchen layout from VLM-derived constraints."""

    # Objects with sizes (meters) and colors — from VLM description
    objects = {
        "wall": {"width": 1.0, "depth": 0.04, "height": 0.8, "color": [0.1, 0.3, 0.3], "shape": "cube"},
        "cabinet": {"width": 0.8, "depth": 0.5, "height": 0.4, "color": [0.92, 0.92, 0.9], "shape": "cube"},
        "countertop": {"width": 0.85, "depth": 0.55, "height": 0.02, "color": [0.75, 0.6, 0.38], "shape": "cube"},
        "bowl": {"width": 0.15, "depth": 0.15, "height": 0.08, "color": [0.6, 0.8, 0.8], "shape": "cylinder"},
        "sink": {"width": 0.2, "depth": 0.18, "height": 0.06, "color": [0.25, 0.25, 0.28], "shape": "cube"},
        "faucet": {"width": 0.03, "depth": 0.03, "height": 0.12, "color": [0.6, 0.6, 0.65], "shape": "cylinder"},
        "paper_towel": {"width": 0.1, "depth": 0.1, "height": 0.2, "color": [0.9, 0.9, 0.92], "shape": "cube"},
        "shelf_opening": {"width": 0.2, "depth": 0.2, "height": 0.2, "color": [0.1, 0.08, 0.06], "shape": "cube"},
    }

    # Spatial constraints — what the VLM told us about the scene
    constraints = [
        # Vertical stacking
        {"type": "on_surface", "args": ["countertop", "cabinet"]},
        {"type": "on_surface", "args": ["bowl", "countertop"]},
        {"type": "on_surface", "args": ["sink", "countertop"]},
        {"type": "on_surface", "args": ["faucet", "countertop"]},

        # Horizontal layout
        {"type": "left_of", "args": ["bowl", "sink"]},
        {"type": "left_of", "args": ["shelf_opening", "sink"]},
        {"type": "right_of", "args": ["faucet", "bowl"]},

        # Wall behind everything
        {"type": "behind", "args": ["wall", "cabinet"]},
        {"type": "above", "args": ["paper_towel", "countertop"]},
        {"type": "left_of", "args": ["paper_towel", "bowl"]},

        # Centering
        {"type": "center_x", "args": ["cabinet"]},
        {"type": "center_x", "args": ["countertop"]},
    ]

    print("Solving kitchen layout with Clingo...")
    print(f"  {len(objects)} objects, {len(constraints)} constraints")

    import time
    start = time.time()
    positions = solve_layout(objects, constraints)
    elapsed = time.time() - start

    if positions:
        print(f"  Solved in {elapsed:.3f}s")
        print(f"\n  Positions (meters):")
        for name, pos in sorted(positions.items()):
            print(f"    {name:20s} x={pos['x']:6.2f}  y={pos['y']:6.2f}  z={pos['z']:6.2f}")

        # Generate USD
        usd = positions_to_usd(objects, positions)

        output_path = Path(__file__).parent.parent / "examples" / "lerobot-discovered" / "video" / "ep_0" / "synthesis" / "scene_clingo.usda"
        output_path.write_text(usd)
        print(f"\n  USD written to {output_path}")

        return usd, positions
    else:
        print("  Failed to solve layout")
        return None, None


if __name__ == "__main__":
    demo_kitchen_scene()
