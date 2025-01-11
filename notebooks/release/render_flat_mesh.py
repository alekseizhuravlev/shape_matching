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


if __name__ == '__main__':

    # Creates a virtual display
    # disp = 
    # disp.start()

    with pyvirtualdisplay.Display(visible=False, size=(1920, 1080)) as disp:
        
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
        os.environ['DISPLAY'] = ':0.0'

        scene = trimesh.Scene()


        mesh_list = [
            '/home/s94zalek_hpc/shape_matching/data/SURREAL_full/template/original/template.off',
            # 'data/SHREC19_r/off/30.off',
            #  'data/DT4D_r/off/zlorp/DancingRunningMan259.off',
            # 'data/DT4D_r/off/zlorp/DancingRunningMan265.off',
            # 'data/DT4D_r/off/zlorp/DancingRunningMan272.off',
            # 'data/DT4D_r/off/zlorp/DancingRunningMan279.off',
            # 'data/DT4D_r/off/zlorp/DancingRunningMan285.off',
            # 'data/DT4D_r/off/zlorp/DancingRunningMan292.off',
            # 'data/DT4D_r/off/zlorp/DancingRunningMan298.off',
            # 'data/DT4D_r/off/zlorp/DancingRunningMan305.off',
            # 'data/DT4D_r/off/zlorp/DancingRunningMan312.off',
            # 'data/DT4D_r/off/zlorp/DancingRunningMan325.off'
        ]


        for file_name in tqdm(mesh_list):
            
            scene.geometry.clear()

            mesh = trimesh.load(file_name)
            
            mesh.apply_transform(mesh.principal_inertia_transform)
            
            # rotate by 90 degrees along the x-axis and 90 degrees along the z-axis
            mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0], [0, 0, 0]))
            mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0], [0, 0, 0]))
            
            # mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2, [1, 0, 0], [0, 0, 0]))
            
            # mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/8, [0, 1, 0], [0, 0, 0]))
            
            # mesh.visual.vertex_colors = np.ones_like(mesh.visual.vertex_colors) * 250
            
            verts_x = torch.tensor(mesh.vertices)
            
            coords_x_norm = torch.zeros_like(verts_x)
            for i in range(3):
                coords_x_norm[:, i] = (verts_x[:, i] - verts_x[:, i].min()) / (verts_x[:, i].max() - verts_x[:, i].min())

            coords_interpolated = torch.zeros(verts_x.shape[0])
            for i in [0, 1]:
                coords_interpolated += coords_x_norm[:, i]
                
            
            # mesh.visual.vertex_colors = cmap = trimesh.visual.color.interpolate(coords_interpolated, 'jet')
            
            
            from notebooks.release.render_mesh_images import get_cmap, interpolate_colors
            
            cmap = get_cmap()
            mesh.visual.vertex_colors = interpolate_colors(coords_interpolated, cmap)[:len(mesh.vertices)]
            
            
            
            scene.add_geometry(mesh)
            scene.set_camera()
            
            proportion = (mesh.vertices[:, 0].max() - mesh.vertices[:, 0].min()) / (mesh.vertices[:, 1].max() - mesh.vertices[:, 1].min())
            
            png = scene.save_image(resolution=(int(proportion*1080), 1080), visible=True)


            base_path = '/lustre/mlnvme/data/s94zalek_hpc-shape_matching/figures/flat'

            # get number of files in the directory
            files = os.listdir(base_path)
            num_files = len(files)
            
            with open(f"{base_path}/flat_{num_files+2:02d}.png", "wb") as f:
                f.write(png)
            