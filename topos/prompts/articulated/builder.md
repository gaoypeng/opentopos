Write `src/build.py` — the Blender entry point. This is mechanical glue: import each part's builder, run it, validate the world bbox against the design contract.

1. Use Read to load `src/design.json`. Use Glob to confirm `src/parts/*.py` exist.

2. Write `src/build.py` with this structure (adapt the `BUILDERS` dict to the actual parts in `design.json`):

```
import bpy
import json, sys
from pathlib import Path
from mathutils import Vector

HERE = Path(__file__).parent.resolve()
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

bpy.ops.wm.read_factory_settings(use_empty=True)

DESIGN = json.loads((HERE / "design.json").read_text())

# Import each part's builder. There is one Python file per part under parts/,
# each exposing a build_<lowercase-name>() that returns the bpy object.
from parts.frame import build_frame
from parts.drawer import build_drawer
from parts.handle import build_handle
# ... add one import per part in design.json

BUILDERS = {
    "Frame": build_frame,
    "Drawer": build_drawer,
    "Handle": build_handle,
    # ... add one entry per part. The key MUST match design.json parts[i].name exactly.
}

def _attach_fallback_material(obj, spec):
    """If no material was assigned by _apply_texture (no PNG), attach a flat
    Principled BSDF from spec['color_rgba'] so the GLB ships with at
    least a baseColorFactor (otherwise viewers render default gray)."""
    if obj.data.materials:
        return
    fallback_mat = bpy.data.materials.new(name=f"{obj.name}_default")
    fallback_mat.use_nodes = True
    bsdf = fallback_mat.node_tree.nodes.get("Principled BSDF")
    rgba = tuple(spec.get("color_rgba") or (0.7, 0.7, 0.7, 1.0))
    if bsdf:
        bsdf.inputs["Base Color"].default_value = rgba
        bsdf.inputs["Roughness"].default_value = 0.6
    obj.data.materials.append(fallback_mat)
    print(f"[MATERIAL_FALLBACK] {obj.name}: no explicit material; attached flat BSDF {rgba}")


def _apply_texture(obj, builder_fn):
    """Framework-owned texture pass (parts contain NO texture code). If a
    generated material image exists at ``src/textures/<lower>.png`` — written by
    the ``generate_texture_image`` tool from design.json's ``texture.prompt`` —
    UV-unwrap the object (Smart UV Project) and bind the PNG as the Principled
    Base Color. Returns True if an image was bound; False when there's no PNG, so
    the caller applies the flat ``color_rgba`` fallback.

    ``<lower>`` is derived from the builder's ``__name__`` (``build_<lower>``) —
    the same stem the texture tool used (``_camel_to_snake(part_name)``) — so the
    two sides can't drift even for multi-word parts (``DrawerTop`` →
    ``drawer_top``). Non-fatal: log + fall through to flat on any failure."""
    builder_name = getattr(builder_fn, "__name__", "")
    lower_name = builder_name[len("build_"):] if builder_name.startswith("build_") else obj.name.lower()
    png = HERE / "textures" / f"{lower_name}.png"
    if not png.is_file():
        return False
    try:
        # 1) UV unwrap — required before binding an image or sampling is undefined.
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.smart_project(island_margin=0.02)   # default angle_limit (version-safe)
        bpy.ops.object.mode_set(mode="OBJECT")
        # 2) Material: ImageTexture -> Principled Base Color.
        mat = bpy.data.materials.new(name=f"{obj.name}_tex")
        mat.use_nodes = True
        nt = mat.node_tree
        nt.nodes.clear()
        timg = nt.nodes.new("ShaderNodeTexImage")
        timg.image = bpy.data.images.load(str(png), check_existing=True)
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Roughness"].default_value = 0.6
        nt.links.new(timg.outputs["Color"], bsdf.inputs["Base Color"])
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
        obj.data.materials.clear()
        obj.data.materials.append(mat)
        print(f"[TEXTURE] {obj.name}: bound {png.name}")
        return True
    except Exception as e:
        print(f"[TEXTURE_WARN] {obj.name}: image bind failed ({e}); using flat color")
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except Exception:
            pass
        return False


# Build each part. For parts with an `instances` list, build once and
# duplicate per-instance with the given rotation/translation. For
# single-instance parts, just call the builder once.
for spec in DESIGN["parts"]:
    builder = BUILDERS[spec["name"]]
    instances = spec.get("instances")
    if instances:
        # Template + instances: call builder once for the canonical shape,
        # then copy + transform per entry. Each instance becomes its own
        # scene object named <PascalName>_<i> (matches part_judge ids).
        canonical = builder()
        canonical.name = f"{spec['name']}_template"   # temp name; we'll delete or rename below
        for i, inst in enumerate(instances):
            obj = canonical.copy()
            obj.data = canonical.data.copy()
            bpy.context.collection.objects.link(obj)
            obj.name = f"{spec['name']}_{i}"
            # Apply per-instance transform (rotation_euler + translation, in order)
            rpy = inst.get("rotation_euler") or [0.0, 0.0, 0.0]
            txyz = inst.get("translation") or [0.0, 0.0, 0.0]
            obj.rotation_euler = tuple(rpy)
            obj.location = tuple(
                (obj.location[a] if hasattr(obj.location, "__getitem__") else 0.0) + txyz[a]
                for a in range(3)
            )
            _apply_texture(obj, builder)
            _attach_fallback_material(obj, spec)   # flat color_rgba if no image was bound
        # Remove the now-unused template object
        bpy.data.objects.remove(canonical, do_unlink=True)
        bpy.context.view_layer.update()
        print(f"[INSTANCES] {spec['name']}: built {len(instances)} instance(s)")
    else:
        # Single instance — original flow.
        obj = builder()
        obj.name = spec["name"]
        # Place canonical-mode parts after construction. Default ("baked") is a no-op.
        if spec.get("place_method", "baked") == "canonical":
            obj.location = tuple(spec["world_xyz"])
            if spec.get("world_rpy"):
                obj.rotation_euler = tuple(spec["world_rpy"])
            bpy.context.view_layer.update()
        _apply_texture(obj, builder)
        _attach_fallback_material(obj, spec)   # flat color_rgba if no image was bound

# Validate every part's world bbox against the design contract. For instance
# parts, validate ONE instance (the first) — extents are per-instance, not
# per-cluster; rotation can shift the world bbox shape, so report per-instance
# extents and don't fail on the rotated-bbox-changes case. Prints OK / WARN
# per part with mm-precision error. Does NOT raise — downstream tools should
# still render so the judge can evaluate.
TOL = 0.005


def _world_bbox(obj):
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    xs, ys, zs = zip(*[(v.x, v.y, v.z) for v in corners])
    bmin = Vector((min(xs), min(ys), min(zs)))
    bmax = Vector((max(xs), max(ys), max(zs)))
    return (bmax + bmin) * 0.5, bmax - bmin


print("=== bbox contract validation ===")
for spec in DESIGN["parts"]:
    name = spec["name"]
    instances = spec.get("instances")
    if instances:
        # For instance parts: validate the first instance (canonical pose) +
        # report instance count. Per-instance bbox check uses world_xyz /
        # world_extents from spec (which describe ONE canonical instance).
        first_id = f"{name}_0"
        if first_id not in bpy.data.objects:
            print(f"[MISSING] {name}: instance {first_id!r} not in scene")
            continue
        obj = bpy.data.objects[first_id]
        center, extents = _world_bbox(obj)
        exp_e = Vector(spec["world_extents"])
        err_e = (extents - exp_e).length
        # Center check is loose for instance parts — instance 0's rotation
        # may shift its center off spec["world_xyz"]. Only check extents.
        tag = "OK" if err_e < TOL * 3 else "WARN"  # 15mm tolerance for rotated bbox
        print(f"[{tag}] {name} (×{len(instances)} instances): instance_0 extents=({extents.x:.3f},{extents.y:.3f},{extents.z:.3f}) err_extents={err_e*1000:.1f}mm")
        continue
    # Single-instance part — original check.
    if name not in bpy.data.objects:
        print(f"[MISSING] {name}: builder did not produce an object")
        continue
    obj = bpy.data.objects[name]
    center, extents = _world_bbox(obj)
    exp_c = Vector(spec["world_xyz"])
    exp_e = Vector(spec["world_extents"])
    err_c = (center - exp_c).length
    err_e = (extents - exp_e).length
    tag = "OK" if (err_c < TOL and err_e < TOL) else "WARN"
    print(f"[{tag}] {name}: center=({center.x:+.3f},{center.y:+.3f},{center.z:+.3f}) extents=({extents.x:.3f},{extents.y:.3f},{extents.z:.3f}) err_center={err_c*1000:.1f}mm err_extents={err_e*1000:.1f}mm")
```

3. The `BUILDERS` dict must list every part by name in `design.json`. List every part — none can be skipped.

4. Validation prints WARN/OK but **must NOT raise** — let downstream render/export/judge proceed so the judge gets visual feedback.

5. **If validation prints any `[WARN]` tags, investigate and fix them before finishing.** WARN tags indicate geometry contract failures (parts out of position, wrong size, collisions, floating attachments). Each WARN describes the specific error and often suggests the fix (e.g. "shift Pelvis +6.0mm"). Run `build.py` again after fixing to verify all parts show `[OK]`. Ignoring WARNs leads to visible defects the judge penalizes.

   **Exception — `[COLLISION_WARN]` on non-furniture topologies.** The collision check uses AABB (bounding-box) overlap, which is an *upper bound*: two parts can have overlapping boxes while their actual meshes never touch. For `object_class` in `lattice-frame` / `articulated-mechanism` / `bladed-assembly` this is the COMMON case, not a defect — a fork's box surrounds the wheel's box, frame tubes cross, blades share a hub. Do **NOT** burn build iterations chasing such WARNs: treat a `[COLLISION_WARN]` on these classes as advisory and act **only** if the reported overlap is large (a clear fraction of the smaller part) or the renders visibly show interpenetration. For `object_class: static-shell` (furniture/boxy bodies), keep treating every `[COLLISION_WARN]` as a real defect to fix.

6. The script will be invoked as `blender --background --python src/build.py` with cwd = workspace root. Paths inside the script resolve relative to `Path(__file__).parent`.

7. **Geometry contracts beyond bbox.** The bbox contract above only validates each part's outer AABB. It silently passes several common geometry failures the judge then complains about (e.g. "frame looks solid, not hollow"; "drawer doesn't fill the cavity"; "handle is floating in front of the drawer"). If any of the following apply to this project, read the `topos_geometry_contracts` skill and append its drop-in validation blocks after the bbox loop:

   - Any part in `design.json` has a `"cavity"` field → add the **fill-ratio** check.
   - Any part has a `"cavity"` whose face is coincident with an outer face → add the **cavity-opening** check.
   - `design.json` has 2+ parts → add the **inter-part collision** check.
   - Any part has `"cavity"` AND is the parent of a joint → add the **cavity-fit** check.
   - Any joint has `"type": "fixed"` → add the **fixed-joint attachment** check (catches floating sub-parts like handles that don't actually touch their parent).

   Each block is ~20-30 lines, prints its own `[OK]`/`[WARN]` lines, and does not raise.

Use Read + Glob to inspect, then Write to create `src/build.py`.
