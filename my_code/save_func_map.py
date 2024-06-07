



def index_with_P(tensor_to_index, P):
    # assert tensor_to_index.shape[0] == P.shape[0], f'tensor_to_index.shape {tensor_to_index.shape} != P.shape {P.shape}'
    
    indexed_tensor = tensor_to_index[P]
    
    # replace values where P = -1 with zeros
    if type(indexed_tensor) == torch.Tensor:
        backend = torch 
    elif type(indexed_tensor) == np.ndarray:
        backend = np
    else:
        raise ValueError(f'Unknown type of indexed_tensor {type(indexed_tensor)}')
    
    indexed_tensor[P == -1] = backend.zeros_like(indexed_tensor[P == -1])
    return indexed_tensor
    


def plot_meshes_with_corr_before(data_x, data_y, Pyx, base_path):
    # clear the scene
    scene = trimesh.Scene()

    cmap = trimesh.visual.color.interpolate(data_x['verts'][0][:, 1], 'jet')

    # add the first mesh
    mesh1 = trimesh.Trimesh(vertices=data_x['verts'][0].cpu().numpy(), faces=data_x['faces'][0].cpu().numpy())
    mesh1.visual.vertex_colors = cmap[:len(mesh1.vertices)].clip(0, 255)
    scene.add_geometry(mesh1)

    mesh2 = trimesh.Trimesh(vertices=data_y['verts'][0].cpu().numpy() + np.array([1, 0, 0]), faces=data_y['faces'][0].cpu().numpy())
    cmap2 = index_with_P(cmap, Pyx)[:len(mesh2.vertices)]
    mesh2.visual.vertex_colors = cmap2
    scene.add_geometry(mesh2)

    png = scene.save_image(resolution=(500, 250), visible=False)
    # f"{data_x['name'][0]}-{data_y['name'][0]}.png", 
    
    with open(f"{base_path}/{data_x['name'][0]}-{data_y['name'][0]}.png", "wb") as f:
        f.write(png)
    
    # destroy the scene
    del(scene)
    
    
def plot_meshes_with_corr_after(data_x, data_y, Pyx_after, base_path):

    scene = trimesh.Scene()

    cmap = trimesh.visual.color.interpolate(data_x['verts'][0][:, 1], 'jet')

    # add the first mesh
    mesh1 = trimesh.Trimesh(vertices=data_x['verts'][0].cpu().numpy(), faces=data_x['faces'][0].cpu().numpy())
    mesh1.visual.vertex_colors = cmap[:len(mesh1.vertices)].clip(0, 255)
    scene.add_geometry(mesh1)


    mesh2 = trimesh.Trimesh(vertices=data_y['verts'][0].cpu().numpy() + np.array([1, 0, 0]), faces=data_y['faces'][0].cpu().numpy())

    cmap2 = Pyx_after @ (cmap.astype(np.float32) / 255)
    cmap2 = (torch.abs(cmap2).numpy() * 255).clip(0, 255).astype(np.uint8)
    cmap2[:, 3] = 255

    # cmap2 = index_with_P(cmap, Pyx)[:len(mesh2.vertices)]
    # print(cmap2.shape, len(mesh2.vertices))

    mesh2.visual.vertex_colors = cmap2[:len(mesh2.vertices)]
    scene.add_geometry(mesh2)

    png = scene.save_image(resolution=(500, 250), visible=False)
    # f"{data_x['name'][0]}-{data_y['name'][0]}.png", 
    
    with open(f"{base_path}/{data_x['name'][0]}-{data_y['name'][0]}.png", "wb") as f:
        f.write(png)
    
    # destroy the scene
    del(scene)


def process_data(data_x, data_y, base_path):
    
    ##################################################
    # Calculate the p2p maps
    ##################################################
    
    # get p2p correspondences
    Pxy = -torch.ones(data_x['verts'].shape[1], dtype=torch.int64)
    Pxy[data_x['corr']] = data_y['corr']

    Pyx = -torch.ones(data_y['verts'].shape[1], dtype=torch.int64)
    Pyx[data_y['corr']] = data_x['corr']
    
    # plot the meshes after applying correspondences in both directions
    p2p_before_path = f"{base_path}/p2p_before_images"
    os.makedirs(p2p_before_path, exist_ok=True)
    
    plot_meshes_with_corr_before(data_x, data_y, Pyx, base_path=p2p_before_path)
    plot_meshes_with_corr_before(data_y, data_x, Pxy, base_path=p2p_before_path)
    
    
    ##################################################
    # Calculate the functional maps
    ##################################################
    
    # get the eigenvectors
    phi_x = data_x['evecs'][0]
    phi_x_T = data_x['evecs_trans'][0]
    phi_y = data_y['evecs'][0]
    phi_y_T = data_y['evecs_trans'][0]

    # calculate the functional maps
    # Cxy = phi_y_T @ index_with_P(phi_x, Pyx)
    # Cyx = phi_x_T @ index_with_P(phi_y, Pxy)
    
    # Cxy = (torch.pinverse(phi_y[data_y['corr']]) @ phi_x[data_x['corr']])[0]
    # Cyx = (torch.pinverse(phi_x[data_x['corr']]) @ phi_y[data_y['corr']])[0]
    
    Cxy = torch.linalg.lstsq(phi_y[data_y['corr']], phi_x[data_x['corr']]).solution[0]
    Cyx = torch.linalg.lstsq(phi_x[data_x['corr']], phi_y[data_y['corr']]).solution[0]

    
    # Cxy = (phi_y_T.transpose(0, 1)[data_y['corr']].transpose(1, 2) @ phi_x[data_x['corr']])[0]
    # Cyx = (phi_x_T.transpose(0, 1)[data_x['corr']].transpose(1, 2) @ phi_y[data_y['corr']])[0]


    
    # save the functional maps
    fmap_path = f"{base_path}/fmap"
    os.makedirs(fmap_path, exist_ok=True)
    
    # save as txt
    np.savetxt(f"{fmap_path}/{data_x['name'][0]}-{data_y['name'][0]}.txt", Cxy.cpu().numpy())
    np.savetxt(f"{fmap_path}/{data_y['name'][0]}-{data_x['name'][0]}.txt", Cyx.cpu().numpy())
        
    
    ##################################################
    # Plot the functional maps
    ##################################################
    
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    # Cxy
    Cxy_plot = ax[0].imshow(Cxy.cpu().numpy(), cmap='bwr', vmin=-1, vmax=1)
    ax[0].axis('off')
    ax[0].set_title('Cxy (proper indexing)')
    cbar = plt.colorbar(Cxy_plot)
    cbar.set_label('Cxy')

    # Cyx
    Cyx_plot = ax[1].imshow(Cyx.cpu().numpy(), cmap='bwr', vmin=-1, vmax=1)
    ax[1].axis('off')
    ax[1].set_title('Cyx (proper indexing)')
    cbar = plt.colorbar(Cyx_plot)
    cbar.set_label('Cyx')

    # get the path to save the images
    fmap_images_path = f"{base_path}/fmap_images"
    os.makedirs(fmap_images_path, exist_ok=True)
    
    plt.savefig(f"{fmap_images_path}/{data_x['name'][0]}-{data_y['name'][0]}_func_map.png")
    
    # close the plot
    plt.close()

    ##################################################
    # Plot the p2p maps after applying the functional maps
    ##################################################

    # get the p2p maps back
    Pyx_after = phi_y @ Cxy @ phi_x_T
    Pxy_after = phi_x @ Cyx @ phi_y_T
    
    p2p_after_path = f"{base_path}/p2p_after_images"
    os.makedirs(p2p_after_path, exist_ok=True)

    plot_meshes_with_corr_after(data_x, data_y, Pyx_after, base_path=p2p_after_path)
    plot_meshes_with_corr_after(data_y, data_x, Pxy_after, base_path=p2p_after_path)
    



if __name__ == '__main__':
    
    ###############################################
    # Import the libraries
    ###############################################

    import sys
    import os
    curr_dir = os.getcwd()
    if 's94zalek_hpc' in curr_dir:
        sys.path.append('/home/s94zalek_hpc/shape_matching')
    else:
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
        save_path = f'/home/s94zalek/shape_matching/func_maps/{config_name}'

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

        max_iter = 15

        # run the function on the train loader
        
        if config_name not in ['topkids']:
            os.makedirs(f'{save_path}/train', exist_ok=True)
            pbar = tqdm(enumerate(train_loader), total=max_iter)
            for i, data in pbar:
                data_x = data['first']
                data_y = data['second']
                pbar.set_description(f'Train {config_name} {data_x["name"][0]}-{data_y["name"][0]}')
                
                process_data(data_x, data_y, base_path=f'{save_path}/train')
                
                if i >= max_iter:
                    break
            
            
        # run the function on the test loader
        os.makedirs(f'{save_path}/test', exist_ok=True)
        pbar = tqdm(enumerate(test_loader), total=max_iter)
        for i, data in pbar:
            data_x = data['first']
            data_y = data['second']
            pbar.set_description(f'Test {config_name} {data_x["name"][0]}-{data_y["name"][0]}')
            
            process_data(data_x, data_y, base_path=f'{save_path}/test')
            
            if i >= max_iter:
                break
            
     
    ###############################################
    # Close the virtual display
    ###############################################
            
    disp.stop()
        




    
