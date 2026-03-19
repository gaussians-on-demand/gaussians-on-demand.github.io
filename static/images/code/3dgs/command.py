import uuid


num_levels = 8
G = 4
max = 0.33
min = 100_000
weighted_sampling = False
reduced = True

std_threshold = 0.02
lambda_sh_sparsity =  0.1
cdist_threshold  = 6


scenes = { 
          
            #   'nerf_real_360': ['bonsai', 'bicycle', 'counter', 'flowers', 'garden', 'kitchen', 'room', 'stump', 'treehill'],
            
            #'tandt' : ['truck', 'train'],
        
            'db': ['drjohnson', 'playroom'] 
            }

gpu_counter = 0

with open('commands.txt', 'a') as f:
    print('\n', file=f)
    remove = []
                    
    for dataset_name in scenes:
        for scene in scenes[dataset_name]:
            scalable_args = f' --G {G} --num_levels {num_levels} --min {min} --max {max} --iterative'
            if weighted_sampling:
                scalable_args += ' --weighted_sampling'
            if reduced:
                scalable_args += f' --reduced --std_threshold {std_threshold} --cdist_threshold {cdist_threshold} --lambda_sh_sparsity {lambda_sh_sparsity} '
                
            # if scene in ["bicycle", "flowers", "garden", "stump", "treehill"]:
            #     scalable_args += ' --images images_4'
            # elif scene in ["room", "counter", "kitchen", "bonsai"]:
            #     scalable_args += ' --images images_2'
            
            source_path = f' --source_path /scratch/nerf/dataset/{dataset_name}/{scene}'
            model_path = f' --model_path /scratch/gaussian-splatting/gaussian-splatting/LightGaussian/gaussian-splatting' + \
                        f"/multi_level_models/{dataset_name}/{scene}/G={G}_L={num_levels}_min={min}_max={max}_ws={weighted_sampling}_reduced={reduced}"
            name = f"--name multiscale_3dgs_{scene}_{G}_{num_levels}_{weighted_sampling}_{reduced}"
            
            if reduced:
                model_path += f'std={std_threshold}_sh_sparsity={lambda_sh_sparsity}_cdist={cdist_threshold}'
            
            #start_checkpoint = f" --start_checkpoint /scratch/gaussian-splatting/gaussian-splatting/LightGaussian/gaussian-splatting/output/{scene}/chkpnt30000.pth"
           
            command =   f'submit {name} '  + \
                        "eidos-service.di.unito.it/disario/gaussian_splatting:no_it " + \
                        "/scratch/gaussian-splatting/gaussian-splatting/LightGaussian/gaussian-splatting/scalable_train2.py " + \
                        f"--eval --scene {scene} --dataset_name {dataset_name}" + source_path + model_path + scalable_args + ' &'
                        
                        

                
            print(command, file=f)
            remove.append(f'docker service rm --name multiscale_3dgs_{scene}_{G}_{num_levels}')
            #remove.append(f'docker service rm --name multiscale_3dgs_{scene}_baseline')

    
    # print('\n', file=f)
    # for r in remove:
    #     print(r, file=f)
        
        
        


    # command =   f'submit --name multiscale_3dgs_{scene}_baseline'  + \
    #             "eidos-service.di.unito.it/disario/gaussian_splatting:no_it " + \
    #             "/scratch/gaussian-splatting/gaussian-splatting/LightGaussian/gaussian-splatting/train.py " + \
    #             f"--eval --scene {scene} --dataset_name {dataset_name} & "
        
    # command =   f'submit --name multiscale_3dgs_{scene}_baseline '  + \
    #     "eidos-service.di.unito.it/disario/gaussian_splatting:no_it " + \
    #     "/scratch/gaussian-splatting/gaussian-splatting/LightGaussian/gaussian-splatting/train.py " + \
    #     f"--eval --scene {scene} --dataset_name {dataset_name} & "