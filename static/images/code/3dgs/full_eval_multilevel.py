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

mipnerf360_outdoor_scenes = ["bicycle", "flowers", "garden", "stump", "treehill"]
mipnerf360_indoor_scenes = ["room", "counter", "kitchen", "bonsai"]
tanks_and_temples_scenes = ["truck", "train"]
deep_blending_scenes = ["drjohnson", "playroom"]

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="./output/multilevel")
parser.add_argument("--G", type=int, default=5) #irrelevant if weighted_sampling==False
parser.add_argument("--weighted_sampling", action='store_true', default=False)
parser.add_argument("--num_levels", type=int, default=8)
parser.add_argument("--min", type=float, default=100_000)
parser.add_argument("--max", type=float, default=0.75)
parser.add_argument('--iterative', action='store_true', default=True)
parser.add_argument('--random', action='store_true', default=False)
parser.add_argument('--pretrained_dir', type=str, help='start pretrained DIR', default='./output/baseline')

args, _ = parser.parse_known_args()

all_scenes = mipnerf360_outdoor_scenes + mipnerf360_indoor_scenes + tanks_and_temples_scenes + deep_blending_scenes

if not args.skip_training or not args.skip_rendering:
    parser.add_argument('--mipnerf360', "-m360", type=str, default='/scratch/nerf/dataset/nerf_real_360')
    parser.add_argument("--tanksandtemples", "-tat", type=str, default='/scratch/nerf/dataset/tandt')
    parser.add_argument("--deepblending", "-db", type=str, default='/scratch/nerf/dataset/db')
    args = parser.parse_args()

def generate_command(scene, source_base, images_flag=None):
    source = f"{source_base}/{scene}"
    output_dir = f"{args.output_path}/{scene}/G={args.G}_L={args.num_levels}_min={args.min}_max={args.max}_ws={args.weighted_sampling}"
    pretrained_dir = f"{args.pretrained_dir}/{scene}"
    images_flag = f"-i {images_flag}" if images_flag else ""
    command = (
        f"python3 scalable_train.py -s {source} {images_flag} -m {output_dir} "
        f"--eval {common_args} {scalable_args} --pretrained_dir {pretrained_dir}"
    )
    return command

if not args.skip_training:
    common_args = "--eval"
    scalable_args = f"--G {args.G} --num_levels {args.num_levels} --min {args.min} --max {args.max}"
    if args.iterative:
        scalable_args += " --iterative"
    if args.random:
        scalable_args += " --random"
    if args.weighted_sampling:
        scalable_args += " --weighted_sampling"

    scenes_config = [
        (mipnerf360_outdoor_scenes, args.mipnerf360, "images_4"),
        (mipnerf360_indoor_scenes, args.mipnerf360, "images_2"),
        (tanks_and_temples_scenes, args.tanksandtemples, None),
        (deep_blending_scenes, args.deepblending, None),
    ]

    with open('commands.txt', 'a') as f:
        for scenes, source_base, images_flag in scenes_config:
            for scene in scenes:
                command = generate_command(scene, source_base, images_flag)
                print(command, file=f)  
