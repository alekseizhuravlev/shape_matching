def process_data(data_x, base_path):
    # clear the scene
    scene = trimesh.Scene()

    cmap = trimesh.visual.color.interpolate(data_x['verts'][0][:, 1], 'jet')

    # add the first mesh
    mesh1 = trimesh.Trimesh(vertices=data_x['verts'][0].cpu().numpy(), faces=data_x['faces'][0].cpu().numpy())
    mesh1.visual.vertex_colors = cmap[:len(mesh1.vertices)].clip(0, 255)
    scene.add_geometry(mesh1)

    png = scene.save_image(resolution=(500, 250), visible=False)
    # f"{data_x['name'][0]}-{data_y['name'][0]}.png", 
    
    with open(f"{base_path}/{data_x['name'][0]}.png", "wb") as f:
        f.write(png)
    
    # destroy the scene
    del(scene)   


if __name__ == '__main__':
    
    ###############################################
    # Import the libraries
    ###############################################

    import sys
    sys.path.append('/home/s94zalek/shape_matching')

    from datasets_code import build_dataloader, build_dataset
    from utils.options import parse_options
    from train import create_train_val_dataloader
    import torch
    import numpy as np

    import os
    os.chdir('/home/s94zalek/shape_matching')
    # os.environ['DISPLAY'] = ':1'
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

    import trimesh
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import pyvirtualdisplay


    # Creates a virtual display
    disp = pyvirtualdisplay.Display(visible=0, size=(500, 250))
    disp.start()

    
    ###############################################
    # Load the datasets
    ###############################################
    
    # faust, scape, shrec16_cuts, topkids, smal
    config_names = ['faust', 'scape', 'shrec16_cuts']
    
    for config_name in config_names:
    
        root_path = '/home/s94zalek/shape_matching'
        opt = parse_options(root_path, is_train=False, use_argparse=False,
                            opt_path = f'options/train/{config_name}.yaml')

        opt['root_path'] = root_path

        for d in ['train_dataset', 'test_dataset']:
            if d in opt['datasets']:
                # we don't need the distance
                # but we need the correspondence
                opt['datasets'][d]['return_dist'] = False
                opt['datasets'][d]['return_corr'] = True

        # create train and test dataloaders
        result = create_train_val_dataloader(opt)
        train_loader, train_sampler, test_loader, total_epochs, total_iters = result


        unique_names = set()
        save_path = f'/home/s94zalek/shape_matching/{opt["datasets"]["train_dataset"]["data_root"]}/mesh_images'


        # run the function on the train loader
        # if config_name not in ['topkids']:
        #     os.makedirs(f'{save_path}/train', exist_ok=True)
        #     pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        #     for i, data in pbar:
        #         # data_x = data['first']
        #         data_x = data['second']
                
        #         if data_x['name'][0] in unique_names:
        #             continue
        #         else:
        #             unique_names.add(data_x['name'][0])
                
        #         pbar.set_description(f'Train {config_name} {data_x["name"][0]}')
                
        #         process_data(data_x, base_path=f'{save_path}/train')
                
                
        # run the function on the test loader
        
        unique_names = set()
        os.makedirs(f'{save_path}/test', exist_ok=True)
        
        pbar = tqdm(enumerate(test_loader))
        for i, data in pbar:
            # data_x = data['first']
            data_x = data['second']
            
            if data_x['name'][0] in unique_names:
                continue
            else:
                unique_names.add(data_x['name'][0])
            
            pbar.set_description(f'Test {config_name} {data_x["name"][0]}')
            
            process_data(data_x, base_path=f'{save_path}/test')          
            
     
    ###############################################
    # Close the virtual display
    ###############################################
            
    disp.stop()
        




    
