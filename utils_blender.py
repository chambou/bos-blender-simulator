import bpy
import bmesh
import numpy
import mathutils
import json

config_path = "config_stereo.json"
with open(config_path, "r") as f:
    config = json.load(f)

def create_hollow_cylinder(
    inner_radius=0.8,
    outer_radius=1.0,
    height=2.0,
    vertices=64
):
    if outer_radius <= inner_radius:
        raise ValueError("outer_radius must be > inner_radius")

    # OUTER CYLINDER
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=vertices,
        radius=outer_radius,
        depth=height,
        location=(0, 0, height / 2)
    )
    outer = bpy.context.object

    # INNER CYLINDER (slightly taller to avoid boolean artifacts)
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=vertices,
        radius=inner_radius,
        depth=height + 0.01,
        location=(0, 0, height / 2)
    )
    inner = bpy.context.object

    # BOOLEAN DIFFERENCE
    mod = outer.modifiers.new(name="Hollow", type='BOOLEAN')
    mod.operation = 'DIFFERENCE'
    mod.object = inner

    bpy.context.view_layer.objects.active = outer
    bpy.ops.object.modifier_apply(modifier="Hollow")

    # delete inner cylinder
    bpy.data.objects.remove(inner, do_unlink=True)

    return outer

def look_at(obj, target):
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()


def create_plane(location=(0,0,0), direction=(0,0,1), width=2, height=2):

    direction = mathutils.Vector(direction).normalized()

    # Choose a stable "up" reference
    up_ref = mathutils.Vector((0, 1, 0))

    # If direction is parallel to up_ref, change reference
    if abs(direction.dot(up_ref)) > 0.99:
        up_ref = mathutils.Vector((1, 0, 0))

    # Build orthonormal basis
    right = up_ref.cross(direction).normalized()
    up = direction.cross(right).normalized()

    # Create plane mesh
    bpy.ops.mesh.primitive_plane_add(location=(0,0,0))
    obj = bpy.context.object

    # Scale plane in local space
    obj.scale = (height / 2, width / 2, 1)
    bpy.ops.object.transform_apply(scale=True)

    # Build rotation matrix (columns = axes)
    mat = mathutils.Matrix((
        right,
        up,
        direction
    )).transposed()

    obj.matrix_world = mat.to_4x4()
    obj.location = location

    return obj

def add_image_material(obj, image_path):
    
    mat = bpy.data.materials.new(name="ImgMat")
    mat.use_nodes = True

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    tex = nodes.new(type='ShaderNodeTexImage')
    emission = nodes.new(type='ShaderNodeEmission')
    out = nodes.new(type='ShaderNodeOutputMaterial')

    # Load image
    tex.image = bpy.data.images.load(image_path)

    # Optional: better color fidelity
    tex.interpolation = 'Linear'
    tex.extension = 'CLIP'

    # Connect nodes
    links.new(tex.outputs["Color"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], out.inputs["Surface"])

    # Strength = brightness of your "screen"
    emission.inputs["Strength"].default_value = 1.1

    obj.data.materials.append(mat)

def create_turbu_screen(vector_middle_camera, pos, width, height, displacement_path, refractive_index, thickness):

    # =========================
    # PARAMETERS
    # =========================
    SUBDIV = 200
    # STRENGTH = 0.05
    STRENGTH = 1                # for real world matching

    # =========================
    # CREATE GRID
    # =========================
    mesh = bpy.data.meshes.new("TurbuScreenMesh")
    obj = bpy.data.objects.new("TurbuScreen", mesh)
    bpy.context.collection.objects.link(obj)

    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=SUBDIV, y_segments=SUBDIV, size=1.0)
    # bmesh.ops.reverse_faces(bm, faces=bm.faces[:])
    bm.to_mesh(mesh)
    bm.free()

    obj.scale = (width / 2, height / 2, 1.0)
    obj.location = mathutils.Vector(pos.tolist())

    direction = mathutils.Vector(vector_middle_camera.tolist()).normalized()
    obj.rotation_mode = 'QUATERNION'
    obj.rotation_quaternion = direction.to_track_quat('Z', 'Y')

    obj.visible_shadow = False

    # =========================
    # SUBDIV MODIFIER (CRITICAL)
    # =========================
    sub = obj.modifiers.new("subd", 'SUBSURF')
    sub.levels = 0
    sub.render_levels = 0

    # =========================
    # DISPLACEMENT MODIFIER (KEY CHANGE)
    # =========================
    tex = bpy.data.textures.new("disp_tex", type='IMAGE')
    tex.image = bpy.data.images.load(displacement_path)
    tex.image.colorspace_settings.name = 'Non-Color'
    tex.use_interpolation = True
    tex.filter_size = 1   # 🔥 your smoothing

    disp_mod = obj.modifiers.new("disp", 'DISPLACE')
    disp_mod.texture = tex
    disp_mod.strength = STRENGTH
    disp_mod.mid_level = 0.0


    # =========================
    # SOLIDIFY
    # =========================
    if config["distortions"]["solidify"] == 1:
        solidify = obj.modifiers.new("solidify", "SOLIDIFY")
        solidify.thickness = thickness
        solidify.offset = 0.0       # distributes thickness evenly around the surface
        solidify.use_even_offset = True
        solidify.solidify_mode = 'EXTRUDE'
        solidify.nonmanifold_thickness_mode = 'FIXED'
        solidify.nonmanifold_boundary_mode = 'NONE'
        # solidify.nonmanifold_thickness_mode = 'FIXED'
    else:
        pass

    # # =========================
    # # FIX NORMAL ORIENTATION
    # # =========================
    # bpy.context.view_layer.objects.active = obj
    # obj.select_set(True)
    # bpy.ops.object.mode_set(mode='EDIT')
    # bpy.ops.mesh.select_all(action='SELECT')
    # bpy.ops.mesh.flip_normals()
    # bpy.ops.object.mode_set(mode='OBJECT')

    # =========================
    # MATERIAL (ONLY VISUALIZATION)
    # =========================


    mat = bpy.data.materials.new(name="TurbuPhaseScreen")
    mat.use_nodes = True
    mat.blend_method = 'OPAQUE' # was 'BLEND'

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    bsdf = nodes.new(type="ShaderNodeBsdfRefraction")
    bsdf.inputs["IOR"].default_value = refractive_index
    bsdf.inputs["Roughness"].default_value = 0.0  # sharp, undiffused bend

    output = nodes.new(type="ShaderNodeOutputMaterial")

    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    obj.data.materials.append(mat)
    # for mod in obj.modifiers:
    #     print(mod.name, mod.type, "show_viewport:", mod.show_viewport, "show_in_editmode:", mod.show_in_editmode)

    return obj
