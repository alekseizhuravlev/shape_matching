import pyvirtualdisplay
import trimesh
import my_code.diffusion_training_sign_corr.data_loading as data_loading
import yaml
import json
from tqdm import tqdm
import torch
import numpy as np
import trimesh.scene
import trimesh.scene.lighting
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import PIL.Image


def interpolate_colors(values, cmap, dtype=np.uint8):
    # make input always float
    values = np.asanyarray(values, dtype=np.float64).ravel()
    # scale values to 0.0 - 1.0 and get colors
    colors = cmap((values - values.min()) / values.ptp())
    # convert to 0-255 RGBA
    rgba = trimesh.visual.color.to_rgba(colors, dtype=dtype)
    
    return rgba


def get_colored_meshes(verts_x, faces_x, verts_y, faces_y, p2p, dataset_name, axes_color_gradient=[0, 1],
                 base_cmap='jet'):
    
    # assert axes_color_gradient is a list or tuple
    assert isinstance(axes_color_gradient, (list, tuple)), "axes_color_gradient must be a list or tuple"
    assert verts_y.shape[0] == len(p2p), f"verts_y {verts_y.shape} and p2p {p2p.shape} must have the same length"
    
    
    if 'DT4D' in dataset_name:
        verts_x_cloned = verts_x.clone()
        
        verts_x[:, 0] = verts_x_cloned[:, 0]
        verts_x[:, 1] = verts_x_cloned[:, 2]
        verts_x[:, 2] = -verts_x_cloned[:, 1]
        
        verts_y_cloned = verts_y.clone()
        
        verts_y[:, 0] = verts_y_cloned[:, 0]
        verts_y[:, 1] = verts_y_cloned[:, 2]
        verts_y[:, 2] = -verts_y_cloned[:, 1]
    
    ##################################################
    # color gradient
    ##################################################
    
    coords_x_norm = torch.zeros_like(verts_x)
    for i in range(3):
        coords_x_norm[:, i] = (verts_x[:, i] - verts_x[:, i].min()) / (verts_x[:, i].max() - verts_x[:, i].min())

    coords_interpolated = torch.zeros(verts_x.shape[0])
    for i in axes_color_gradient:
        coords_interpolated += coords_x_norm[:, i]
        
    if type(base_cmap) == str:
        cmap = trimesh.visual.color.interpolate(coords_interpolated, base_cmap)
    else:
        cmap = interpolate_colors(coords_interpolated, base_cmap)
        
    cmap2 = cmap[p2p].clip(0, 255)

    ##################################################
    # add the meshes
    ################################################

    # 1
    mesh1 = trimesh.Trimesh(vertices=verts_x, faces=faces_x, validate=True)
    mesh1.visual.vertex_colors = cmap[:len(mesh1.vertices)].clip(0, 255)
    
    # mesh1.apply_transform(mesh1.principal_inertia_transform)
            
    #         # rotate by 90 degrees along the x-axis and 90 degrees along the z-axis
    # mesh1.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0], [0, 0, 0]))
    # mesh1.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0], [0, 0, 0]))
    
    
           
    # 2
    mesh2 = trimesh.Trimesh(vertices=verts_y, faces=faces_y, validate=True)
    mesh2.visual.vertex_colors = cmap2[:len(mesh2.vertices)]
    
    # mesh1.vertices -= np.mean(mesh1.vertices, axis=0)
    # mesh2.vertices -= np.mean(mesh2.vertices, axis=0)
    
    # if 'DT4D' in dataset_name:
    #     mesh1.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2, [1, 0, 0], [0, 0, 0]))
    #     mesh2.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2, [1, 0, 0], [0, 0, 0]))
        # mesh2.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0], [0, 0, 0]))
    
    
    # mesh1.visual.vertex_colors = np.ones_like(mesh1.visual.vertex_colors) * 250
    # mesh2.visual.vertex_colors = np.ones_like(mesh2.visual.vertex_colors) * 250
    
    
    mesh1.apply_transform(trimesh.transformations.rotation_matrix(np.pi/8, [0, 1, 0], [0, 0, 0]))
    mesh2.apply_transform(trimesh.transformations.rotation_matrix(np.pi/8, [0, 1, 0], [0, 0, 0]))
    
    return mesh1, mesh2
    
    # mesh2.apply_transform(trimesh.transformations.rotation_matrix(np.pi/8, [1, 0, 0], [0, 0, 0]))
    
    # trimesh.smoothing.filter_taubin(mesh1, iterations=3)
    # trimesh.smoothing.filter_taubin(mesh2, iterations=3)
    
    # scene.add_geometry(mesh1)
    # scene.add_geometry(mesh2)
    
    # scene.add_geometry(trimesh.creation.axis(origin_size=0.05))

    # return scene
    
    
def get_cmap():
    SAMPLES = 100
    ice = px.colors.sample_colorscale(
        
        # px.colors.cyclical.Edge,
        # px.colors.sequential.Jet,
        # px.colors.diverging.Picnic,
        
        
        # px.colors.cyclical.IceFire,
        
        px.colors.cyclical.HSV,
        # px.colors.sequential.Blackbody,
        # px.colors.sequential.Viridis,
        SAMPLES)
    rgb = [px.colors.unconvert_from_RGB_255(px.colors.unlabel_rgb(c)) for c in ice]

    cmap = mcolors.ListedColormap(rgb, name='Ice', N=SAMPLES)

    return cmap


def render_mesh(scene, mesh):
    
    scene.geometry.clear()
    scene.add_geometry(mesh)
    
    scene.set_camera()
    
    proportion = (mesh.vertices[:, 0].max() - mesh.vertices[:, 0].min()) / (mesh.vertices[:, 1].max() - mesh.vertices[:, 1].min())
    # proportion=1
        
    png = scene.save_image(resolution=(int(proportion*1080), 1080), visible=True)

    return png


if __name__ == '__main__':

    # Creates a virtual display
    # disp = 
    # disp.start()

    with pyvirtualdisplay.Display(visible=False, size=(1920, 1080)) as disp:
        
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
        os.environ['DISPLAY'] = ':0.0'

        scene = trimesh.Scene()


        # dataset_name = 'FAUST_r_pair'
        dataset_name = 'SCAPE_r_pair'
        # dataset_name = 'SHREC19_r_pair'
        # dataset_name = 'DT4D_inter_pair'
        # dataset_name = 'DT4D_intra_pair'

        single_dataset, pair_dataset = data_loading.get_val_dataset(
            dataset_name, 'test', 128, preload=False, return_evecs=True, centering='bbox'
        )
        
        
        if dataset_name == 'SHREC19_r_pair':
            file_name = '/lustre/mlnvme/data/s94zalek_hpc-shape_matching/ddpm_checkpoints/single_64_1-2ev_64-128-128_remeshed_fixed/eval/epoch_99/SHREC19_r_pair-test/no_smoothing/2024-11-04_22-27-59/pairwise_results.json'
        elif dataset_name == 'DT4D_intra_pair':
            file_name = '/lustre/mlnvme/data/s94zalek_hpc-shape_matching/ddpm_checkpoints/single_template_remeshed/eval/checkpoint_99.pt/DT4D_intra_pair-test/no_smoothing/2024-11-10_21-20-05/pairwise_results.json'
        elif dataset_name == 'DT4D_inter_pair':
            file_name = '/lustre/mlnvme/data/s94zalek_hpc-shape_matching/ddpm_checkpoints/single_template_remeshed/eval/checkpoint_99.pt/DT4D_inter_pair-test/no_smoothing/2024-11-10_21-20-05/pairwise_results.json'
        elif dataset_name == 'FAUST_r_pair':
            file_name = '/lustre/mlnvme/data/s94zalek_hpc-shape_matching/ddpm_checkpoints/single_64_1-2ev_64-128-128_remeshed_fixed/eval/epoch_99/FAUST_r_pair-test/no_smoothing/2024-11-04_22-27-59/pairwise_results.json'
        elif dataset_name == 'SCAPE_r_pair':
            file_name = '/lustre/mlnvme/data/s94zalek_hpc-shape_matching/ddpm_checkpoints/single_64_1-2ev_64-128-128_remeshed_fixed/eval/epoch_99/SCAPE_r_pair-test/no_smoothing/2024-11-04_22-27-59/pairwise_results.json'
                
        with open(file_name, 'r') as f:
            p2p_saved = json.load(f)
        
        
        geo_err_list = torch.tensor([p2p_saved[i]['geo_err_median_pairzo'] for i in range(len(p2p_saved))])
        idxs_geo_err = torch.argsort(geo_err_list, descending=True)


        base_path = f'/lustre/mlnvme/data/s94zalek_hpc-shape_matching/figures/p2p/{dataset_name}'
        
        # if os.path.exists(base_path):
        #     os.system(f'rm -r {base_path}')
        
        os.makedirs(f"{base_path}/single", exist_ok=True)
        os.makedirs(f"{base_path}/combined", exist_ok=True)

        cmap = get_cmap()

        random_order = torch.randperm(len(idxs_geo_err))[:400]
        
        # random_order = [29]
        
        for k in tqdm(random_order):
            
            indx = k

        # for k in tqdm(range(len(idxs_geo_err))):

            # indx = idxs_geo_err[k]
            
            data_i = pair_dataset[indx]
            p2p_i = p2p_saved[indx]
            p2p_pairzo = torch.tensor(p2p_i['p2p_median_pairzo'])

            # mesh_1 = trimesh.Trimesh(data_i['first']['verts'], data_i['first']['faces'])
            # mesh_2 = trimesh.Trimesh(data_i['second']['verts'], data_i['second']['faces'])



            scene.geometry.clear()

            mesh1, mesh2 = get_colored_meshes( 
                data_i['first']['verts'], data_i['first']['faces'],
                data_i['second']['verts'], data_i['second']['faces'],
                p2p_pairzo,
                axes_color_gradient=[0, 1],
                base_cmap=cmap,
                dataset_name=dataset_name
            )
            
            png1 = render_mesh(scene, mesh1)
            png2 = render_mesh(scene, mesh2)
            
            

            # proportion = 1.2 * (data_i['second']['verts'][:, 0].max() - data_i['second']['verts'][:, 0].min()) / (data_i['second']['verts'][:, 1].max() - data_i['second']['verts'][:, 1].min())
            
            # png = scene.save_image(resolution=(int(proportion*1080), 1080), visible=True)


            # get number of files in the directory
            # files = os.listdir(base_path)
            # num_files = len(files)
            
            with open(f"{base_path}/single/{k:04d}_0.png", "wb") as f:
                f.write(png1)
                
            with open(f"{base_path}/single/{k:04d}_1.png", "wb") as f:
                f.write(png2)
                
            
            # open pngs again and combine them
            
            with PIL.Image.open(f"{base_path}/single/{k:04d}_0.png") as png1:
                
                
                with PIL.Image.open(f"{base_path}/single/{k:04d}_1.png") as png2:
            
                    png_combined = PIL.Image.new('RGB', (png1.width + png2.width, png1.height))
                    png_combined.paste(png1, (0, 0))
                    png_combined.paste(png2, (png1.width, 0))
                    
                    # save combined image
                    
                    png_combined.save(f"{base_path}/combined/{k:04d}_combined_{geo_err_list[indx].item():.1f}.png")
            
            
                