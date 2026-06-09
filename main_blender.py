import numpy as np
import json
import bpy
import mathutils
from pathlib import Path
import importlib
import sys

# --------------------------------------------------
# Project root (portable Blender-safe)
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

# --------------------------------------------------
# Add local module path (no absolute path)
# --------------------------------------------------
sys.path.append(str(PROJECT_ROOT))

import utils_blender
importlib.reload(utils_blender)
from utils_blender import *

# --------------------------------------------------
# Load config (portable)
# --------------------------------------------------
config_path = PROJECT_ROOT / "config.json"

with open(config_path, "r") as f:
    config = json.load(f)

#############
# WORLD SETUP
#############

world = bpy.context.scene.world
if world is None:
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world

world.use_nodes = True
nodes = world.node_tree.nodes
links = world.node_tree.links

nodes.clear()
bg = nodes.new(type="ShaderNodeBackground")
out = nodes.new(type="ShaderNodeOutputWorld")

# white sky
bg.inputs["Color"].default_value = (1, 1, 1, 1)
bg.inputs["Strength"].default_value = config["environment"]["ambiant_light_strengh"]
links.new(bg.outputs["Background"], out.inputs["Surface"])
    
    
# Param
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.device = 'GPU'
scene.unit_settings.system = 'METRIC'
scene.unit_settings.scale_length = 1.0

# Clean
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)


angle_separation = config["BOS"]["cameras_spacing"] * np.atan(2*config["camera"]["sensor_size"]/ config["camera"]["focal_length"])
size_screen_width = config["BOS"]["distance_camera_screen"] * config["camera"]["sensor_size"]/ config["camera"]["focal_length"]
size_screen_height = config["BOS"]["distance_camera_screen"] * config["camera"]["sensor_size"]/ config["camera"]["focal_length"] * config["camera"]["resolution_y"] / config["camera"]["resolution_x"]

#######################################
# Compute cameras and screens positions
#######################################

screen_positions = []
camera_positions = []
for k in range(0,config["BOS"]["cameras_number"]):
    x_screen = config["BOS"]["distance_camera_screen"] / 2 * np.cos(k * angle_separation)
    y_screen = config["BOS"]["distance_camera_screen"] / 2 * np.sin(k * angle_separation)
    z_screen =  size_screen_height / 2 + config["BOS"]["height"]

    x_cam = config["BOS"]["distance_camera_screen"] / 2 * np.cos(k * angle_separation + np.pi)
    y_cam = config["BOS"]["distance_camera_screen"] / 2 * np.sin(k * angle_separation + np.pi)
    z_cam =  size_screen_height / 2 + config["BOS"]["height"]

    screen_positions.append(np.array([x_screen, y_screen, z_screen]))
    camera_positions.append(np.array([x_cam, y_cam, z_cam]))

print(camera_positions)

#######################################
# Create cameras and screens 
#######################################

cams = []
for i, pos in enumerate(camera_positions):
    bpy.ops.object.camera_add(location=pos)
    cam = bpy.context.object
    cam.data.lens = config["camera"]["focal_length"]
    cam.data.sensor_width = config["camera"]["sensor_size"]
    cam.data.sensor_fit = 'HORIZONTAL'
    cam.data.display_size = 0.2
    look_at(cam, mathutils.Vector(screen_positions[i].tolist()))
    cam.name = f"Cam_{i}"
    cams.append(cam)
    
    
    # Plane
    plane = create_plane(
        location=mathutils.Vector((screen_positions[i]).tolist()),
        direction=mathutils.Vector((screen_positions[i]-pos).tolist()),  # plane faces diagonally
        width=size_screen_width,
        height=size_screen_height
    )
    
    
    # UV unwrap
    bpy.context.view_layer.objects.active = plane
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project()
    bpy.ops.object.mode_set(mode='OBJECT')

    # Add image
    add_image_material(
        plane,
        config["BOS"]["screen_pattern_file"]
    )



#######################################
# Create distortion
#######################################

if config["distortions"]["turbulence"] == 1:
    
    for k, dist in enumerate(config["distortions"]["turbulence_distance"]):

        vector_middle_camera = (screen_positions[len(camera_positions)//2] - camera_positions[len(camera_positions)//2])
        pos = camera_positions[len(camera_positions)//2] + dist * vector_middle_camera
        width= dist * size_screen_height #config["BOS"]["distance_camera_screen"]
        height= size_screen_height + 2 * config["BOS"]["height"]

        create_turbu_screen(vector_middle_camera, pos, width, height,config["distortions"]["turbulence_path"][k] )


#######################################
# Add object
#######################################

if config["distortions"]["object"] == 1:

    dist = config["distortions"]["object_distance"]
    vector_middle_camera = (screen_positions[len(camera_positions)//2] - camera_positions[len(camera_positions)//2])
    pos = camera_positions[len(camera_positions)//2] + dist * vector_middle_camera

    with bpy.data.libraries.load(config["distortions"]["object_path"], link=False) as (data_from, data_to):
        data_to.objects = data_from.objects

    # Link appended object to scene
    for obj in data_to.objects:
        if obj is not None:
            bpy.context.collection.objects.link(obj)

            obj.name = "ObjectScreen"
            
            # Position (X, Y, Z)
            obj.location = mathutils.Vector(pos.tolist())

            # Taille (scale)
            s = config["distortions"]["object_scale"]
            obj.scale = (s, s, s)

if config["environment"]["telescope"] == 1:

    bpy.ops.wm.obj_import(
        filepath = PROJECT_ROOT /"data/VLT.obj",

        # ORIENTATION
        forward_axis='Z',   # typical for OBJ
        up_axis='Y',

        # SCALE
        global_scale=0.0003
    )
    

if config["environment"]["dome_walls"] == 1:
    create_hollow_cylinder(
        inner_radius=config["BOS"]["distance_camera_screen"] / 2 * 1.2,
        outer_radius=config["BOS"]["distance_camera_screen"] / 2 * 1.3,
        height=size_screen_height / 2 + config["BOS"]["height"]
        )
        
        
###############
# Render 
##############

if config["render"]["activate_render"] == 1:

    format_map = {
    "tiff": "TIFF",
    "png": "PNG",
    "jpg": "JPEG",
    "jpeg": "JPEG",
    "exr": "OPEN_EXR"
    }

    fmt = config["render"]["format"].lower()
    scene.render.image_settings.file_format = format_map.get(fmt, "TIFF")
    scene.render.image_settings.color_depth = config["camera"]["bit_depth"]  # optionnel

    # FIRST: without phase screen
    for k, cam in enumerate(cams):
        if config["render"]["active_cameras"] == "all" or k in config["render"]["active_cameras"]:
            scene.camera = cam
            scene.render.filepath = config["render"]["render_path"]+f"render_ref_{k}"
            
            prefix_turbu = "TurbuScreen"
            prefix_obj = "ObjectScreen"

            objs = [obj for obj in bpy.data.objects if (obj.name.startswith(prefix_turbu) or obj.name.startswith(prefix_obj))]
            for obj in objs:
                obj.hide_render = True
                
            bpy.context.view_layer.update()
            bpy.ops.render.render(write_still=True, use_viewport=False)
            print("Reference frame - Camera done:", cam.name, flush=True)
    
    # THEN: add phase screen
    for k, cam in enumerate(cams):
        if config["render"]["active_cameras"] == "all" or k in config["render"]["active_cameras"]:
            scene.camera = cam
            scene.render.filepath = config["render"]["render_path"]+f"render_{k}"
            
            prefix_turbu = "TurbuScreen"
            prefix_obj = "ObjectScreen"

            objs = [obj for obj in bpy.data.objects if (obj.name.startswith(prefix_turbu) or obj.name.startswith(prefix_obj))]
            for obj in objs:
                obj.hide_render = False
                
            bpy.context.view_layer.update()
            bpy.ops.render.render(write_still=True, use_viewport=False)
            print("Camera done:", cam.name, flush=True)