#!/usr/bin/env python3
"""
scene_synth.py — Analysis by Synthesis: build calibrated world models from video.

Pipeline:
  1. VLM looks at video frame → describes objects and layout
  2. LLM generates a USD scene (.usda) from the description
  3. Blender renders the USD scene from estimated camera angle
  4. VLM compares render to real frame → gives adjustment feedback
  5. LLM adjusts the USD scene based on feedback
  6. Iterate until VLM says "close enough"
  7. Output: calibrated USD world model

Usage:
    uv run python src/scene_synth.py \
        --frame examples/lerobot-discovered/video/ep_0/frame_001.png \
        --iterations 3

Requires: Blender installed at /Applications/Blender.app, OPENROUTER_API_KEY in .env
"""

import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("ERROR: requests required")

# Load .env
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())

BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"


# ============================================================================
# VLM Calls
# ============================================================================

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def vlm_call(messages, model="anthropic/claude-sonnet-4", max_tokens=2000):
    """Call VLM via OpenRouter."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("OPENROUTER_API_KEY not set")

    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": max_tokens,
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def vlm_describe_scene(frame_path):
    """Step 1: VLM describes what it sees."""
    return vlm_call([{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{encode_image(frame_path)}"}},
            {"type": "text", "text": """Describe this scene for 3D reconstruction. Be precise about:

1. Camera angle (looking down? from the side? approximate elevation and azimuth)
2. Each object: what it is, approximate position (left/center/right, near/far), approximate size relative to the scene, color/material
3. The surface/background
4. Lighting direction

Format as a structured list. Use approximate metric dimensions if you can (e.g., "bowl ~15cm diameter"). Position everything relative to scene center."""}
        ]
    }])


def vlm_generate_usd(scene_description):
    """Step 2: LLM generates USD scene from description."""
    return vlm_call([{
        "role": "user",
        "content": f"""Generate a USD ASCII (.usda) file that represents this scene:

{scene_description}

CRITICAL CONVENTIONS (Blender USD import):
- Coordinate system: Z-up, Y-forward. A table 0.9m tall has translate = (0, 0, 0.45) with scale on Z.
- Cube primitive is 2m per side by default. scale=(0.4, 0.3, 0.01) gives a 0.8m x 0.6m x 0.02m slab.
- Cylinder default is 2m tall, 1m radius. scale=(0.08, 0.08, 0.03) gives radius=0.08m, height=0.06m.
- Use xformOp:translate for position, xformOp:scale for size. Always include xformOpOrder.
- Colors: use color3f[] primvars:displayColor = [(R, G, B)] with values 0-1.
- DO NOT include Camera or Light definitions — those are added by the renderer.
- The full scene should fit in a 1m x 1m x 1.2m bounding box, centered near origin.

WORKING EXAMPLE — a table with a bowl on it:
```
#usda 1.0
def Xform "World"
{{
    def Cube "Tabletop"
    {{
        double3 xformOp:translate = (0, 0, 0.45)
        double3 xformOp:scale = (0.4, 0.3, 0.01)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        color3f[] primvars:displayColor = [(0.7, 0.5, 0.3)]
    }}
    def Cylinder "Bowl"
    {{
        double3 xformOp:translate = (-0.1, 0, 0.48)
        double3 xformOp:scale = (0.06, 0.06, 0.02)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        color3f[] primvars:displayColor = [(0.9, 0.9, 0.95)]
    }}
}}
```

Now generate the FULL scene matching the description above. Use distinct colors for each object (not all white/gray). Make the wall dark teal (0.1, 0.3, 0.3), countertop warm wood (0.7, 0.55, 0.35), cabinet white (0.92, 0.92, 0.9), sink dark (0.25, 0.25, 0.25), bowl light blue (0.7, 0.85, 0.85).

Output ONLY the .usda content. No markdown. Start with #usda 1.0"""
    }], max_tokens=3000)


def vlm_compare(real_frame_path, render_path):
    """Step 4: VLM compares render to real frame."""
    return vlm_call([{
        "role": "user",
        "content": [
            {"type": "text", "text": "Image 1 is a REAL photo. Image 2 is a 3D RENDER attempting to match it."},
            {"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{encode_image(real_frame_path)}"}},
            {"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{encode_image(render_path)}"}},
            {"type": "text", "text": """Compare these two images. For each object in the real photo:
1. Is it present in the render? If not, what's missing?
2. Is its position roughly correct? If not, which direction should it move?
3. Is its size roughly correct? If not, bigger or smaller?
4. Is the camera angle similar? If not, how should it change?

Rate overall match: 1-10 (1=completely wrong, 10=very close).

Then provide SPECIFIC adjustment instructions as a JSON object:
```json
{
  "match_score": N,
  "adjustments": [
    {"object": "name", "action": "move/resize/add/remove", "details": "specifics"}
  ],
  "camera_adjustment": "description or null"
}
```"""}
        ]
    }])


def vlm_adjust_usd(current_usd, feedback):
    """Step 5: LLM adjusts USD based on VLM feedback."""
    return vlm_call([{
        "role": "user",
        "content": f"""Here is a USD scene file:

{current_usd}

A visual comparison with the real scene produced this feedback:

{feedback}

Adjust the USD file to address the feedback. Move objects, resize them, adjust the camera, add missing objects, or remove incorrect ones.

Output ONLY the complete adjusted .usda file content. No markdown fences. Start with #usda 1.0"""
    }], max_tokens=3000)


# ============================================================================
# Blender Rendering
# ============================================================================

BLENDER_RENDER_SCRIPT = '''
import bpy
import sys
import os
import math

usd_path = sys.argv[-2]
output_path = sys.argv[-1]

# Clear scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# Import USD
try:
    bpy.ops.wm.usd_import(filepath=usd_path)
    print(f"Imported USD: {usd_path}")
except Exception as e:
    print(f"USD import failed: {e}")
    try:
        bpy.ops.import_scene.usd(filepath=usd_path)
    except Exception as e2:
        print(f"Both USD imports failed: {e2}")
        sys.exit(1)

# Note: metersPerUnit=1.0 is set in USD preprocessing, so no scale fix needed

# Convert displayColor to actual materials
# Try multiple methods since USD import handles colors differently per version
for obj in bpy.data.objects:
    if obj.type != 'MESH':
        continue

    mesh = obj.data
    r, g, b = 0.5, 0.5, 0.5
    found_color = False

    # Method 1: color_attributes (vertex colors)
    for attr in mesh.color_attributes:
        if len(attr.data) > 0:
            c = attr.data[0].color
            r, g, b = c[0], c[1], c[2]
            found_color = True
            break

    # Method 2: check existing materials for displayColor
    if not found_color and obj.data.materials:
        for mat in obj.data.materials:
            if mat and mat.use_nodes:
                for node in mat.node_tree.nodes:
                    if node.type == 'BSDF_PRINCIPLED':
                        bc = node.inputs["Base Color"].default_value
                        if bc[0] != 0.8 or bc[1] != 0.8:  # not default gray
                            r, g, b = bc[0], bc[1], bc[2]
                            found_color = True
                            break

    # Method 3: check custom properties
    if not found_color:
        for key in obj.keys():
            if 'color' in key.lower() or 'display' in key.lower():
                val = obj[key]
                if hasattr(val, '__len__') and len(val) >= 3:
                    r, g, b = val[0], val[1], val[2]
                    found_color = True
                    break

    # Create/replace material with the color
    mat_name = f"mat_{obj.name}"
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (r, g, b, 1)
        bsdf.inputs["Roughness"].default_value = 0.6
        bsdf.inputs["Specular IOR Level"].default_value = 0.3

    obj.data.materials.clear()
    obj.data.materials.append(mat)
    print(f"  Material for {obj.name}: ({r:.2f}, {g:.2f}, {b:.2f}) found={found_color}")

# Auto-frame: compute scene bounds and point camera at center
from mathutils import Vector

all_coords = []
for obj in bpy.data.objects:
    if obj.type == 'MESH':
        for corner in obj.bound_box:
            world_corner = obj.matrix_world @ Vector(corner)
            all_coords.append(world_corner)

# Remove any USD camera (often has wrong coordinates)
for obj in list(bpy.data.objects):
    if obj.type == 'CAMERA':
        bpy.data.objects.remove(obj)

if all_coords:
    xs = [c.x for c in all_coords]
    ys = [c.y for c in all_coords]
    zs = [c.z for c in all_coords]
    center = Vector(((min(xs)+max(xs))/2, (min(ys)+max(ys))/2, (min(zs)+max(zs))/2))
    size = max(max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs))

    # Find the "front" of the scene: the max Y face (where cabinet doors/details are)
    front_y = max(ys)
    cam_dist = size * 1.5
    # Camera in front (max Y + distance), slightly above center, slightly to the right
    cam_loc = (center.x + size * 0.1, front_y + cam_dist * 0.7, center.z + cam_dist * 0.3)
    # Look at the center of the scene, biased slightly upward
    look_at = Vector((center.x, center.y, center.z + size * 0.1))
else:
    cam_loc = (0, 1.5, 0.7)
    look_at = Vector((0, 0, 0.3))

bpy.ops.object.camera_add(location=cam_loc)
cam = bpy.context.object
direction = look_at - Vector(cam_loc)
cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
cam.data.lens = 32
bpy.context.scene.camera = cam

# Ensure there is a light — add one if USD light was not imported
has_light = any(obj.type == 'LIGHT' for obj in bpy.data.objects)
if not has_light:
    bpy.ops.object.light_add(type='SUN', location=(0, -1, 3))
    bpy.context.object.data.energy = 3
    bpy.context.object.rotation_euler = (math.radians(50), 0, 0)

# Also add fill light for softer shadows
bpy.ops.object.light_add(type='AREA', location=(0, 1, 2))
bpy.context.object.data.energy = 50
bpy.context.object.data.size = 2

# Render settings
bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'
bpy.context.scene.render.resolution_x = 224
bpy.context.scene.render.resolution_y = 224
bpy.context.scene.render.film_transparent = False
bpy.context.scene.render.filepath = output_path

# World background
world = bpy.data.worlds.get('World')
if world is None:
    world = bpy.data.worlds.new('World')
bpy.context.scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes.get('Background')
if bg:
    bg.inputs[0].default_value = (0.15, 0.15, 0.15, 1)

bpy.ops.render.render(write_still=True)
print(f"Rendered to {output_path}")
'''


def render_usd(usd_content, output_path):
    """Render a USD scene using Blender."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.usda', delete=False) as f:
        f.write(usd_content)
        usd_path = f.name

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(BLENDER_RENDER_SCRIPT)
        script_path = f.name

    try:
        result = subprocess.run(
            [BLENDER, "--background", "--python", script_path, "--", usd_path, output_path],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            print(f"  Blender stderr: {result.stderr[-500:]}")
            return False

        if os.path.exists(output_path):
            return True
        else:
            print(f"  Render file not created")
            return False
    except subprocess.TimeoutExpired:
        print(f"  Blender timed out")
        return False
    finally:
        os.unlink(usd_path)
        os.unlink(script_path)


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analysis by synthesis: VLM + USD + Blender")
    parser.add_argument("--frame", required=True, help="Path to real video frame")
    parser.add_argument("--iterations", type=int, default=3, help="Refinement iterations")
    parser.add_argument("--output-dir", default=None, help="Directory for outputs")
    parser.add_argument("--model", default="anthropic/claude-sonnet-4")
    args = parser.parse_args()

    frame_path = Path(args.frame).resolve()
    if not frame_path.exists():
        sys.exit(f"Frame not found: {frame_path}")

    output_dir = Path(args.output_dir) if args.output_dir else frame_path.parent / "synthesis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"ANALYSIS BY SYNTHESIS")
    print(f"{'='*60}")
    print(f"Frame: {frame_path}")
    print(f"Model: {args.model}")
    print(f"Iterations: {args.iterations}")
    print(f"Output: {output_dir}")

    # Step 1: VLM describes the scene
    print(f"\n--- Step 1: VLM describes scene ---")
    description = vlm_describe_scene(str(frame_path))
    print(description)
    (output_dir / "description.txt").write_text(description)

    # Step 2: Generate initial USD
    print(f"\n--- Step 2: Generate USD scene ---")
    usd_content = vlm_generate_usd(description)

    # Clean up — sometimes LLM wraps in markdown
    if "```" in usd_content:
        lines = usd_content.split("\n")
        clean_lines = []
        in_block = False
        for line in lines:
            if line.strip().startswith("```"):
                in_block = not in_block
                continue
            if in_block or line.startswith("#usda") or (clean_lines and clean_lines[0].startswith("#usda")):
                clean_lines.append(line)
        usd_content = "\n".join(clean_lines)

    if not usd_content.startswith("#usda"):
        idx = usd_content.find("#usda")
        if idx >= 0:
            usd_content = usd_content[idx:]
        else:
            print("  WARNING: LLM didn't produce valid USD. Attempting anyway.")

    # Ensure metersPerUnit and upAxis are set (prevents Blender cm-to-m scaling)
    if "metersPerUnit" not in usd_content:
        usd_content = usd_content.replace(
            "#usda 1.0",
            '#usda 1.0\n(\n    metersPerUnit = 1.0\n    upAxis = "Z"\n)',
            1
        )

    # Strip the World Xform wrapper if present (causes nested scale issues)
    if 'def Xform "World"' in usd_content:
        lines = usd_content.split("\n")
        out = []
        skip_world_open = False
        depth = 0
        for line in lines:
            s = line.strip()
            if s == 'def Xform "World"':
                skip_world_open = True
                continue
            if skip_world_open and s == "{":
                skip_world_open = False
                depth = 1
                continue
            if depth > 0:
                if "{" in s:
                    depth += s.count("{")
                if "}" in s:
                    depth -= s.count("}")
                if depth <= 0:
                    continue
                # Remove one level of indentation
                out.append(line[4:] if line.startswith("    ") else line)
            else:
                out.append(line)
        usd_content = "\n".join(out)

    usd_path = output_dir / "scene_v0.usda"
    usd_path.write_text(usd_content)
    print(f"  Written to {usd_path}")
    print(f"  First 10 lines:")
    for line in usd_content.split("\n")[:10]:
        print(f"    {line}")

    # Step 3: Render
    print(f"\n--- Step 3: Render initial scene ---")
    render_path = str(output_dir / "render_v0.png")
    success = render_usd(usd_content, render_path)
    if not success:
        print("  Initial render failed. Saving USD for manual inspection.")
        print(f"  USD: {usd_path}")
        print(f"  Try: /Applications/Blender.app/Contents/MacOS/Blender --python ... to debug")
        return

    print(f"  Rendered to {render_path}")

    # Iteration loop
    for iteration in range(args.iterations):
        print(f"\n--- Iteration {iteration + 1}/{args.iterations}: Compare and adjust ---")

        # Step 4: VLM compares
        feedback = vlm_compare(str(frame_path), render_path)
        print(f"  Feedback:")
        for line in feedback.split("\n"):
            print(f"    {line}")
        (output_dir / f"feedback_v{iteration}.txt").write_text(feedback)

        # Check if match is good enough
        try:
            # Try to extract match score
            import re
            score_match = re.search(r'"match_score"\s*:\s*(\d+)', feedback)
            if score_match:
                score = int(score_match.group(1))
                print(f"\n  Match score: {score}/10")
                if score >= 8:
                    print(f"  Good enough! Stopping iteration.")
                    break
        except:
            pass

        # Step 5: Adjust USD
        print(f"\n  Adjusting USD based on feedback...")
        usd_content = vlm_adjust_usd(usd_content, feedback)

        # Clean markdown
        if "```" in usd_content:
            lines = usd_content.split("\n")
            clean_lines = []
            in_block = False
            for line in lines:
                if line.strip().startswith("```"):
                    in_block = not in_block
                    continue
                clean_lines.append(line)
            usd_content = "\n".join(clean_lines)
        idx = usd_content.find("#usda")
        if idx >= 0:
            usd_content = usd_content[idx:]

        usd_path = output_dir / f"scene_v{iteration + 1}.usda"
        usd_path.write_text(usd_content)

        # Re-render
        render_path = str(output_dir / f"render_v{iteration + 1}.png")
        success = render_usd(usd_content, render_path)
        if not success:
            print(f"  Render failed at iteration {iteration + 1}")
            break

        print(f"  Rendered v{iteration + 1} to {render_path}")

    # Final output
    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"{'='*60}")
    print(f"USD scenes: {output_dir}/scene_v*.usda")
    print(f"Renders: {output_dir}/render_v*.png")
    print(f"Feedback: {output_dir}/feedback_v*.txt")
    print(f"\nFinal USD world model: {usd_path}")
    print(f"View in Blender: /Applications/Blender.app/Contents/MacOS/Blender {usd_path}")


if __name__ == "__main__":
    main()
