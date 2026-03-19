from argparse import ArgumentParser
import sys
import os
import subprocess


#select your data_dir
data_dir='/scratch/nerf/dataset'
exp_dir='./output/baseline'


mipnerf360_indoor_scenes = ["room", "counter", "kitchen", "bonsai"]
mipnerf360_outdoor_scenes = ["bicycle", "flowers", "garden", "stump", "treehill"]
tanks_and_temples_scenes = ["truck", "train"]
deep_blending_scenes = ["drjohnson", "playroom"]


command = 'python3 train.py --eval --source_path {data_dir}/{dataset_name}/{scene} --lod {lod} --voxel_size {vsize} --appearance_dim {appearance_dim} --ratio {ratio} ' + \
        '-m {exp_dir}/{dataset_name}/{scene}'
        

with open('commands.txt', 'a') as f:

    for scene in mipnerf360_indoor_scenes:
        
        formatted = command.format(dataset_name='nerf_real_360', data_dir=data_dir, exp_dir=exp_dir, scene=scene, lod=0, vsize=0.001, appearance_dim=0, ratio=1)
        #os.system(formatted)
        print(formatted, file=f)

    for scene in mipnerf360_outdoor_scenes:
            
        formatted = command.format(dataset_name='nerf_real_360', data_dir=data_dir, exp_dir=exp_dir, scene=scene, lod=0, vsize=0.001, appearance_dim=0, ratio=1)
        #os.system(formatted)
        print(formatted, file=f)
        
    for scene in tanks_and_temples_scenes:
            
        formatted = command.format(dataset_name='tandt', data_dir=data_dir, exp_dir=exp_dir, scene=scene, lod=0, vsize=0.01, appearance_dim=0, ratio=1)
        #os.system(formatted)
        print(formatted, file=f)
        
    for scene in deep_blending_scenes:
            
        formatted = command.format(dataset_name='db', data_dir=data_dir, exp_dir=exp_dir, scene=scene, lod=0, vsize=0.005, appearance_dim=0, ratio=1)
        #os.system(formatted)
        print(formatted, file=f)

