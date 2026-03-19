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
import json
from lpips import lpips
import copy
import os
import time
import numpy as np
import torch
from PIL import Image
from random import randint
from train import prepare_output_and_logger, training_report
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui, render_r
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import wandb 
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import torch.nn as nn
import math
import random 


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
           

def geometric_progression(tmin, tmax, L):
    b = np.exp((np.log(tmax) - np.log(tmin)) / (L - 1))
    print(b)
    arr = np.array([i for i in range(L)])
    return ((b ** arr) * tmin).astype(np.int32)
    
            
def write(string):
    with open(os.path.join(os.getcwd(), 'fps.csv'), 'a') as f:
        print(string, file=f)

    
def accumulate_gradients(scene, opt, gaussians, pipe, background, mask, render_fun):
    print('Accumulating gradients...')
    for viewpoint_cam in scene.getTrainCameras():

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render_fun(viewpoint_cam, gaussians, pipe, bg, mask=mask)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward(retain_graph=True)
    print('Done')
    
    
def fine_tuning(scene, opt, gaussians, pipe, background, mask, progress_bar, first_iter):
    
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    for iteration in range(first_iter, first_iter + opt.fine_tune_iters+1):        

        gaussians.update_learning_rate_orig(iteration)

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

    
        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, mask=mask)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        loss.backward()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "num_gaussians": gaussians._xyz.shape[0]})
                progress_bar.update(10)
            if iteration == opt.iterations+1:
                progress_bar.close()

            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none = True)

    


def evaluation(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, masks, load_iter=None):
  
    prepare_output_and_logger(dataset)
    
    
    gaussians = GaussianModel(dataset.sh_degree, variable_sh_bands=args.reduced)
    scene = Scene(dataset, gaussians)



    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


    render_fun =  render_r if args.reduced else render 
    
    if args.reduced:
        files = os.listdir(os.path.join(args.model_path, 'point_cloud', 'iteration_60000'))
        model_paths = [os.path.join(args.model_path, 'point_cloud', 'iteration_60000', f) for f in files if '_quantised_half' in f]
        
        for i, path in enumerate(sorted(model_paths)):
            gaussians.load_ply(path, quantised=True, half_float=True)
            test(None, 'evaluation', None, None, l1_loss, None, testing_iterations, scene, render_fun, (pipe, background), 
                    dump_images=False, eval_lpips=False, mask=None, level=args.num_levels-i-1)
    else:
        ply_path = os.path.join(args.model_path, 'point_cloud', 'iteration_60000', 'point_cloud.ply')
        gaussians.load_ply(ply_path)
        masks = torch.load(os.path.join(args.model_path, 'masks.pt'))
        #masks.reverse()
        for i, mask in enumerate(masks):
            test(None, 'evaluation', None, None, l1_loss, None, testing_iterations, scene, render_fun, (pipe, background), 
                    dump_images=False, eval_lpips=False, mask=mask, level=args.num_levels-i-1)
        
        # disjointed_masks = [masks[0]]
        # c_mask = masks[0]
        # for i in range(1, len(masks)):
        #     d_mask = np.logical_xor(c_mask, masks[i])
        #     c_mask = np.logical_or(c_mask, masks[i])
        #     disjointed_masks.append(d_mask)
            
        # for i, d_mask in enumerate(disjointed_masks):
        #     test(None, 'evaluation', None, None, l1_loss, None, testing_iterations, scene, render_fun, (pipe, background), 
        #             dump_images=True, eval_lpips=False, mask=d_mask, level=f'partial_{i}')

@torch.no_grad()
def test(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, moe=None, 
         gate=None, distance=None, dump_images=False, eval_lpips=False, mask=None, level=None, size=None):
    torch.cuda.empty_cache()
    validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                            {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})
    
    
    if level is not None:
        os.makedirs(os.path.join(args.model_path, 'renders', f'L{level}'), exist_ok=True)
    
    gaussians = scene.gaussians
    gaussians_cpy = copy.deepcopy(gaussians)
    scene.gaussians = gaussians_cpy

    with torch.no_grad():
        if mask is not None:
            scene.gaussians._xyz = scene.gaussians._xyz[mask]
            scene.gaussians._opacity = scene.gaussians._opacity[mask]
            scene.gaussians._rotation = scene.gaussians._rotation[mask]
            scene.gaussians._scaling = scene.gaussians._scaling[mask]
            scene.gaussians._features_dc = scene.gaussians._features_dc[mask]
            scene.gaussians._features_rest = scene.gaussians._features_rest[mask]

    for config in validation_configs:
        if config['cameras'] and len(config['cameras']) > 0:
            l1_test = 0.0
            psnr_test = 0.0
            ssim_test = 0.0
            lpips_test = 0.0
            fps_test = []
            for idx, viewpoint in enumerate(config['cameras']):
 
                pkg = renderFunc(viewpoint, scene.gaussians, measure_fps=True, *renderArgs)
                fps = pkg['fps']
                fps_test.append(fps)
                image = torch.clamp(pkg["render"], 0.0, 1.0)
                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                if dump_images and config['name'] == 'test':
                    np_img = (np.transpose(image.cpu().numpy(), (1, 2, 0)) * 255).astype(np.uint8)
                    img = Image.fromarray(np_img)
                    img.save(os.path.join(args.model_path, 'renders', f'L{level}', f'{idx}.png'))

                l1_test += l1_loss(image, gt_image).mean().double()
                psnr_test += psnr(image, gt_image).mean().double()
                ssim_test += ssim(image, gt_image).mean().double()
          
                    
            psnr_test /= len(config['cameras'])
            ssim_test /= len(config['cameras'])
            lpips_test /= len(config['cameras'])
            l1_test /= len(config['cameras'])   
            
            fps_test = np.mean(fps_test)
            
            num_gaussians = mask.sum().item() if mask is not None else scene.gaussians._xyz.shape[0]
            if config['name'] == 'test':
                
                write(f'{args.scene}, {fps_test}')   
                #wandb.log({'psnr': psnr_test, 'ssim': ssim_test, 'lpips':lpips, 'num_gaussians': num_gaussians})    
                #print("{},{}".format(args.scene, fps_test))
                print(f'{args.scene}, {fps_test}, {psnr_test}')
    
    scene.gaussians = gaussians
    torch.cuda.empty_cache()


if __name__ == "__main__":
    
    #os.environ["WANDB_API_KEY"] = "3d66ee3e1eda4ba9070df043338d82ef9e0919e8"
    #wandb.login()

    #set_seed(8072020)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1000, 2000, 3000, 5000, 7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[60_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str)
    parser.add_argument("--dump_images", type=bool, default=False)
    parser.add_argument("--random", action='store_true', default=False)
    parser.add_argument("--iterative", action='store_true', default=False)
    parser.add_argument("--avoid_sample", action='store_true', default=False)
    parser.add_argument("--fine_tune", action='store_true', default=False)
    
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--num_levels", type=int, default=8)
    parser.add_argument("--min", type=int, default=100_000)
    parser.add_argument("--max", type=float, default=0.75)
    parser.add_argument("--weighted_sampling", action='store_true', default=False)
    parser.add_argument("--eval_only", action='store_true', default=False)
    parser.add_argument("--eval_ssim", action='store_true', default=False)
    parser.add_argument("--eval_lpips", action='store_true', default=False)
    parser.add_argument("--G", type=int, default=5)
    parser.add_argument("--reduced", action='store_true', default=False)
    parser.add_argument("--qa", action='store_true', default=False)
    parser.add_argument("--test_only", action='store_true', default=False)
    
    args = parser.parse_args(sys.argv[1:])
    args.eval = True
    
    args.cull_SH = 15_000

    args.test_iterations = [1, 2000, 5000, 7000, 10000, 15000, 20000, 25000, 30000]

    print("Optimizing " + args.model_path)
    print(args.source_path)
    print(args.start_checkpoint)
    
    print('cuda available: ')
    print(torch.cuda.is_available())
    
    print('current gpu: ', torch.cuda.current_device())

    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    #args.start_checkpoint = '/scratch/gaussian-splatting/gaussian-splatting/LightGaussian/gaussian-splatting/output/nerf_real_360/bonsai/_0.02/chkpnt30000.pth'
    scenes = { 
                #'nerf_real_360': ['bicycle', 'bonsai', 'counter', 'flowers', 'garden', 'kitchen', 'room', 'stump', 'treehill'],
                #'nerf_real_360': ['bonsai']
                #'tandt' : ['train', 'truck'],
                'db': ['drjohnson', 'playroom']
    }
    
    for dataset_name in scenes:
        for scene in scenes[dataset_name]:

            #args.model_path = f'/scratch/gaussian-splatting/gaussian-splatting/LightGaussian/gaussian-splatting/multi_level_models/{args.scene}_reduce={args.reduced}_{args.lambda_sh_sparsity}_low'
            args.scene = scene
            args.dataset_name = dataset_name
            if args.scene in ["bicycle", "flowers", "garden", "stump", "treehill"]:
                    args.images = 'images_4'
                    
            elif args.scene in ["room", "counter", "kitchen", "bonsai"]:
                args.images = 'images_2'
            else:
                args.images = 'images'
            
            #args.position_lr_max_steps = 60_000
            masks = []
            args.source_path = f'/scratch/nerf/dataset/{dataset_name}/{args.scene}'
            
            args.model_path = f'/scratch/gaussian-splatting/gaussian-splatting/LightGaussian/gaussian-splatting/multi_level_models/{args.dataset_name}/{args.scene}/G=4_L=8_min=100000_max=0.5_ws=False_reduced=False'
            #args.model_path = f'/scratch/gaussian-splatting/gaussian-splatting/LightGaussian/gaussian-splatting/multi_level_models/{args.dataset_name}/{args.scene}/G=4_L=8_min=100000_max=0.75_ws=False_reduced=Truestd=0.02_sh_sparsity=0.1_cdist=6'
            #args.model_path = f'/scratch/gaussian-splatting/gaussian-splatting/LightGaussian/gaussian-splatting/multi_level_models/{args.scene}/G=5_L=8_min=100000_max=0.75_ws=False'
            
            args.reduced = 'reduced=True' in args.model_path
            evaluation(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, masks=masks)
            write('\n\n')