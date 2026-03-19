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
    with open(os.path.join(args.model_path, 'results.csv'), 'a') as f:
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
    
    
def fine_tuning(scene, opt, gaussians, pipe, background, mask, first_iter):
    
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    
    progress_bar = tqdm(range(first_iter, first_iter + opt.fine_tune_iters+1), desc="Training progress")
    

    for iteration in range(first_iter, first_iter + opt.fine_tune_iters+1):        

        gaussians.update_learning_rate(iteration)

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

    
        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render_r(viewpoint_cam, gaussians, pipe, bg, mask=mask)
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


    with torch.no_grad():
        scene_cpy = copy.deepcopy(scene)
        scene_cpy.gaussians.produce_clusters(store_dict_path=scene.model_path) 

        q_path = scene_cpy.save(60_000, quantise=True, mask=mask, level=first_iter)
        hq_path = scene_cpy.save(60_000, quantise=True, half_float=True, mask=mask, level=first_iter)


        scene_cpy.gaussians.load_ply(q_path, quantised=True)
        size = os.path.getsize(q_path) / 2**20
        print(size)
        test(None, f'quantized_final', None, None, l1_loss, None, None, scene_cpy, render_r, (pipe, background), 
                    dump_images=False, eval_lpips=True, mask=None, level=first_iter, size=size)


        scene_cpy.gaussians.load_ply(hq_path, quantised=True, half_float=True)
        size = os.path.getsize(hq_path) / 2**20
        print(size)
        test(None, f'half_quantized_final', None, None, l1_loss, None, None, scene_cpy, render_r, (pipe, background), 
                    dump_images=False, eval_lpips=True, level=first_iter, size=size)
        write('')
        return scene
        
        
    
def prune_and_fine_tune(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, fine_tune=False):
    first_iter = 30_000
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    
    print('scena costruita...ora loading')
    
    n_start = gaussians._xyz.shape[0] // 4
    # if checkpoint:
    #     (model_params, first_iter) = torch.load(checkpoint)
    #     gaussians.restore(model_params, opt)
    
    gaussians.load_ply(ply_path)
    gaussians.training_setup(opt)

    print('tutto caricato')
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    print('progress_bar')
    #progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    print('progress_bar ok')
    masks = []
    iteration = 1
    
    k = 0
    keep = True
    num_gaussians = gaussians._xyz.shape[0]
    
    print(args.min, args.num_levels)
    k_s = list(np.flip((num_gaussians - geometric_progression(args.min, num_gaussians*args.max, args.num_levels))))
    
    print(k_s)
    # k_s = [num_gaussians - num_gaussians//(2**i) for i in range(1, 5) ]
    mask = None
    render_fun = render_r 
    test(None, iteration, None, None, l1_loss, None, testing_iterations, scene, render_fun, (pipe, background), dump_images=False, eval_lpips=False, mask=None)
    
    with torch.no_grad():
        gaussians.cull_sh_bands(scene.getTrainCameras(), threshold=args.cdist_threshold*np.sqrt(3)/255, std_threshold=args.std_threshold)
    
    scene = fine_tuning(scene, opt, gaussians, pipe, background, None, first_iter)
    first_iter += opt.fine_tune_iters
    
    print('inizio costruzione maschere')
    for i in range(args.num_levels):
        if keep:
            accumulate_gradients(scene, opt, gaussians, pipe, background, mask, render_fun)
            
        num_gaussians = num_gaussians if mask is None else mask.sum()
        #k += int(num_gaussians * gamma)
        k = k_s[i] 
        # if i > 0:
        #     k -= k_s[0]
        mask = gaussians.gradient_prune(k, args.random)

        keep = args.iterative
        masks.append(mask)
        test(None, iteration, None, None, l1_loss, None, testing_iterations, scene, render_fun, (pipe, background), dump_images=False, eval_lpips=False, mask=mask)
        if args.iterative:
            gaussians.optimizer.zero_grad(set_to_none=True)

        scene = fine_tuning(scene, opt, gaussians, pipe, background, mask, first_iter)
        first_iter += opt.fine_tune_iters
        #test(None, iteration, None, None, l1_loss, None, testing_iterations, scene, render, (pipe, background), dump_images=False, eval_lpips=False, mask=mask)

    
    # for mask in masks:
    #     test(None, iteration, None, None, l1_loss, None, testing_iterations, scene, render_fun, (pipe, background), dump_images=False, eval_lpips=False, mask=mask)
    
    if not args.iterative:
        gaussians.optimizer.zero_grad(set_to_none=True)
    
    torch.save(masks, os.path.join(args.model_path, 'masks.pt'))
    
    return scene, masks


def fine_tune(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, scene, masks, load_iter=None):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    # gaussians = GaussianModel(dataset.sh_degree)
    # scene = Scene(dataset, gaussians)
    
    
    # if checkpoint:
    #     (model_params, first_iter) = torch.load(checkpoint)
    #     gaussians.restore(model_params, opt)
    #gaussians.load_ply(ply_path)
    gaussians = scene.gaussians
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, first_iter+opt.iterations), desc="Training progress")
    
    testing_iterations = list(map(lambda x: x + first_iter, testing_iterations))
    
    eval_lpips = False
    dump_images = False
    
    if args.weighted_sampling:
        progr = np.flip(geometric_progression(1, args.G, len(masks)))
        probabilities = torch.tensor(progr / progr.sum(), dtype=torch.float32)
        weights = progr
    else:
        probabilities = torch.tensor([1 for i in range(len(masks))], dtype=torch.float32)
    
    indices = torch.multinomial(probabilities, opt.iterations, replacement=True)


    render_fun = render_r if args.reduced else render


    first_iter += 1
    for iteration in range(first_iter, first_iter+opt.iterations + 1):        
        
        m_index = indices[iteration-first_iter-1]
        mask = masks[m_index]
        
        iter_start.record()
        
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        if args.avoid_sample:
            loss = 0.0
            for i, mask in enumerate(masks):
                render_pkg = render_fun(viewpoint_cam, gaussians, pipe, bg, mask=mask, lambda_sh_sparsity=args.lambda_sh_sparsity)
                image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                gt_image = viewpoint_cam.original_image.cuda()
                Ll1 = l1_loss(image, gt_image)
                loss = weights[i] * (loss + (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)))
        else:
            render_pkg = render_fun(viewpoint_cam, gaussians, pipe, bg, mask=mask)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            
            if args.reduced:
                loss = loss + args.lambda_sh_sparsity * gaussians._features_rest[mask][visibility_filter].abs().mean()
        
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "num_gaussians": gaussians._xyz.shape[0]})
                progress_bar.update(10)
            if iteration == opt.iterations+first_iter:
                progress_bar.close()

            if iteration in testing_iterations:
                
                for i, mask in enumerate(masks):
                    gaussians = scene.gaussians
                    gaussians_cpy = copy.deepcopy(gaussians)
                    scene.gaussians = gaussians_cpy
                        
                    scene.gaussians._xyz = scene.gaussians._xyz[mask]
                    scene.gaussians._opacity = scene.gaussians._opacity[mask]
                    scene.gaussians._rotation = scene.gaussians._rotation[mask]
                    scene.gaussians._scaling = scene.gaussians._scaling[mask]
                    scene.gaussians._features_dc = scene.gaussians._features_dc[mask]
                    scene.gaussians._features_rest = scene.gaussians._features_rest[mask]
                    scene.gaussians._degrees = scene.gaussians._degrees[mask]
                    
                    if iteration == testing_iterations[-1] and not args.test_only:
                        eval_lpips = True 
                        dump_images = True
                    
                    test(None, iteration, None, None, l1_loss, None, testing_iterations, scene, render_fun, (pipe, background), 
                         dump_images=dump_images, eval_lpips=eval_lpips, mask=None, level=args.num_levels-i-1)
                    
                    scene.gaussians = gaussians
            
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none = True)
            
            if iteration in [args.cull_SH] and args.reduced:
                gaussians.cull_sh_bands(scene.getTrainCameras(), threshold=args.cdist_threshold*np.sqrt(3)/255, std_threshold=args.std_threshold)


            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                #torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + "_L={L}.pth")
            
            if args.test_only:
                return
    
    if args.reduced:
        quantized_paths = []
        half_quantized_path = []
        with torch.no_grad():
            scene.gaussians.produce_clusters(store_dict_path=scene.model_path) 
            for i, mask in enumerate(masks):
                q_path = scene.save(60_000, quantise=True, mask=mask, level=i)
                hq_path = scene.save(60_000, quantise=True, half_float=True, mask=mask, level=i)
                quantized_paths.append(q_path)
                half_quantized_path.append(hq_path)
        
            for q_path in quantized_paths:
                gaussians.load_ply(q_path, quantised=True)
                size = os.path.getsize(q_path) / 2**20
                print(size)
                test(None, 'quantized_final', None, None, l1_loss, None, testing_iterations, scene, render_fun, (pipe, background), 
                            dump_images=dump_images, eval_lpips=eval_lpips, mask=None, level=args.num_levels-i-1, size=size)

            for hq_path in half_quantized_path:
                gaussians.load_ply(hq_path, quantised=True, half_float=True)
                size = os.path.getsize(hq_path) / 2**20
                print(size)
                test(None, 'half_quantized_final', None, None, l1_loss, None, testing_iterations, scene, render_fun, (pipe, background), 
                            dump_images=dump_images, eval_lpips=eval_lpips, mask=None, level=args.num_levels-i-1, size=size)

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
            ms_per_render = []
            for idx, viewpoint in enumerate(config['cameras']):
                start = time.time()
                image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                ms = time.time() - start
                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                ms_per_render.append(ms)
                if dump_images and config['name'] == 'test':
                    np_img = (np.transpose(image.cpu().numpy(), (1, 2, 0)) * 255).astype(np.uint8)
                    img = Image.fromarray(np_img)
                    img.save(os.path.join(args.model_path, 'renders', f'L{level}', f'{idx}.png'))

                l1_test += l1_loss(image, gt_image).mean().double()
                psnr_test += psnr(image, gt_image).mean().double()
                ssim_test += ssim(image, gt_image).mean().double()
                if eval_lpips:
                    lpips_test += lpips_loss(image, gt_image, normalize=True).item()
                    
            psnr_test /= len(config['cameras'])
            ssim_test /= len(config['cameras'])
            lpips_test /= len(config['cameras'])
            l1_test /= len(config['cameras'])   
            fps = 1.0 / np.array(ms_per_render)[1:].mean()
            
            num_gaussians = mask.sum().item() if mask is not None else scene.gaussians._xyz.shape[0]
            if config['name'] == 'test':
                
                write(f'{iteration}, {psnr_test}, {ssim_test}, {lpips_test}, {num_gaussians}, {fps}, {size}')   
                #wandb.log({'psnr': psnr_test, 'ssim': ssim_test, 'lpips':lpips, 'num_gaussians': num_gaussians})    
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}, FPS {}, #W {}".format(iteration, config['name'], l1_test, psnr_test, fps, num_gaussians))
    
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
    lpips_loss = lpips.LPIPS(net='vgg').to('cuda')
    
    #args.start_checkpoint = '/scratch/gaussian-splatting/gaussian-splatting/LightGaussian/3dgs-mcmc/output/nerf_real_360/bonsai/_0.02/chkpnt30000.pth'
    args.source_path = f'/scratch/nerf/dataset/{args.dataset_name}/{args.scene}'
    
    #args.model_path = f'/scratch/gaussian-splatting/gaussian-splatting/LightGaussian/3dgs-mcmc/multi_level_models/{args.scene}_reduce={args.reduced}_{args.lambda_sh_sparsity}_low'
    
    if args.scene in ["bicycle", "flowers", "garden", "stump", "treehill"]:
            args.images = 'images_4'
            
    elif args.scene in ["room", "counter", "kitchen", "bonsai"]:
        args.images = 'images_2'
    
    #args.position_lr_max_steps = 60_000
    ply_path = f'/scratch/gaussian-splatting/gaussian-splatting/LightGaussian/3dgs-mcmc/output/{args.scene}/point_cloud/iteration_30000/point_cloud.ply'
    #ply_path = f'/scratch/gaussian-splatting/gaussian-splatting/LightGaussian/3dgs-mcmc/output/{args.scene}/sparsify=True/point_cloud/iteration_30000/point_cloud.ply'
    
    
    scene, masks = prune_and_fine_tune(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.fine_tune)
    #fine_tune(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, scene, masks)

    # All done
    print("\nTraining complete.")

    wandb.finish()
    
    
