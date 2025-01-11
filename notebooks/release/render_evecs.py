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



        dataset_name = 'FAUST_r_pair'

        single_dataset, pair_dataset = data_loading.get_val_dataset(
            dataset_name, 'test', 128, preload=False, return_evecs=True, centering='bbox'
        )


        base_path = f'/lustre/mlnvme/data/s94zalek_hpc-shape_matching/figures/evecs'
        os.makedirs(f"{base_path}/single", exist_ok=True)
        os.makedirs(f"{base_path}/combined", exist_ok=True)
        
        # random_order = torch.randperm(len(idxs_geo_err))[:400]
        
        random_order = range(0, 32, 2)
        
        for k in tqdm(random_order):
            
            indx = k

        # for k in tqdm(range(len(idxs_geo_err))):

            # indx = idxs_geo_err[k]
            
            data = single_dataset[0]
            mesh = trimesh.Trimesh(data['verts'], data['faces'])
            
            # print(data['evecs'].shape)

            cmap = trimesh.visual.color.interpolate(data['evecs'][:, k], 'bwr')
            mesh.visual.vertex_colors = cmap[:mesh.vertices.shape[0]]

            png1 = render_mesh(scene, mesh)
            
            cmap = trimesh.visual.color.interpolate(-data['evecs'][:, k], 'bwr')
            mesh.visual.vertex_colors = cmap[:mesh.vertices.shape[0]]
            
            png2 = render_mesh(scene, mesh)
            
            

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
                    
                    png_combined.save(f"{base_path}/combined/{k:04d}_combined.png")
            
            
                