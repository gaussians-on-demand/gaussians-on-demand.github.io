#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
from argparse import ArgumentParser


caps_dict = {
    'bicycle': 6091454,
    'bonsai': 1258387,
    'counter': 1205418,
    'flowers': 3597987,
    'garden': 5871850,
    'kitchen': 1827448,
    'room': 1539301,
    'stump': 4832587,
    'treehill': 3740475,
    'truck': 2582356,
    'train': 1087721,
    'drjohnson': 3298379,
    'playroom': 2323837
}

# Scene groups
mipnerf360_outdoor_scenes = ["bicycle", "flowers", "garden", "stump", "treehill"]
mipnerf360_indoor_scenes = ["room", "counter", "kitchen", "bonsai"]
tanks_and_temples_scenes = ["truck", "train"]
deep_blending_scenes = ["drjohnson", "playroom"]

# Argument parser
parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="./output/baseline")
parser.add_argument("--basedir", default="")
args, _ = parser.parse_known_args()

# Combine all scenes
all_scenes = mipnerf360_outdoor_scenes + mipnerf360_indoor_scenes + tanks_and_temples_scenes + deep_blending_scenes

if not args.skip_training or not args.skip_rendering:
    parser.add_argument('--mipnerf360', "-m360", type=str, default='/scratch/nerf/dataset/nerf_real_360')
    parser.add_argument("--tanksandtemples", "-tat", type=str, default='/scratch/nerf/dataset/tandt')
    parser.add_argument("--deepblending", "-db", type=str, default='/scratch/nerf/dataset/db')
    args = parser.parse_args()

# Helper function to construct the source path
def get_source_path(scene, group):
    if group == "outdoor" or group == "indoor":
        return os.path.join(args.mipnerf360, scene)
    elif group == "tanks_and_temples":
        return os.path.join(args.tanksandtemples, scene)
    elif group == "deep_blending":
        return os.path.join(args.deepblending, scene)
    return ""

# Training phase
if not args.skip_training:
    common_args = " --quiet --eval --test_iterations -1"
    scene_groups = [
        (mipnerf360_outdoor_scenes, "outdoor", "images_4"),
        (mipnerf360_indoor_scenes, "indoor", "images_2"),
        (tanks_and_temples_scenes, "tanks_and_temples", None),
        (deep_blending_scenes, "deep_blending", None),
    ]
    
    with open('commands.txt', 'a') as f:
        for scenes, group, images_arg in scene_groups:
            for scene in scenes:
                source = get_source_path(scene, group)
                images_option = f" -i {images_arg}" if images_arg else ""
                cap_max = caps_dict.get(scene, "unknown")
                command = (
                    f"python3 {args.basedir}train.py -s {source}{images_option} -m {args.output_path}/{scene} "
                    f"{common_args} --cap_max {cap_max}"
                )
                print(command, file=f)
            #os.system(command)

# Rendering phase
# if not args.skip_rendering:
#     common_args = " --quiet --eval --skip_train"
#     all_sources = [get_source_path(scene, "outdoor") if scene in mipnerf360_outdoor_scenes
#                    else get_source_path(scene, "indoor") if scene in mipnerf360_indoor_scenes
#                    else get_source_path(scene, "tanks_and_temples") if scene in tanks_and_temples_scenes
#                    else get_source_path(scene, "deep_blending")
#                    for scene in all_scenes]

#     for scene, source in zip(all_scenes, all_sources):
#         command = (
#             f"python3 {args.basedir}render.py --iteration 30000 -s {source} -m {args.output_path}/{scene} "
#             f"{common_args}"
#         )
#         os.system(command)

# # Metrics phase
# if not args.skip_metrics:
#     scenes_string = " ".join([f"\"{args.output_path}/{scene}\"" for scene in all_scenes])
#     command = f"python3 {args.basedir}metrics.py -m {scenes_string}"
#     os.system(command)
