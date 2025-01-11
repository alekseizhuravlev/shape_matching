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
import networks.diffusion_network as diffusion_network
import yaml
import my_code.sign_canonicalization.training as sign_training


import PIL.Image


def render_mesh(scene, mesh):
    
    scene.geometry.clear()
    scene.add_geometry(mesh)
    
    scene.set_camera()
    
    proportion = (mesh.vertices[:, 0].max() - mesh.vertices[:, 0].min()) / (mesh.vertices[:, 1].max() - mesh.vertices[:, 1].min())
    # proportion=1
        
    png = scene.save_image(resolution=(int(proportion*1080), 1080), visible=True)

    return png


# def render_double(scene, mesh1, mesh2):
    
#     scene.geometry.clear()
    
#     scene.add_geometry(mesh1)
#     scene.add_geometry(mesh2)

#     scene.set_camera(angles=(-0.5, 0, 0), distance=(1.7), center=(0.5, 0, 0), resolution=None, fov=None)

#         proportion = (mesh.vertices[:, 0].max() - mesh.vertices[:, 0].min()) / (mesh.vertices[:, 1].max() - mesh.vertices[:, 1].min())
#     # proportion=1
        
#     png = scene.save_image(resolution=(int(proportion*1080), 1080), visible=True)

#     return png
    


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
        
        # exp_name = 'signNet_32_FAUST_orig'

        # exp_name = 'signNet_remeshed_mass_6b_1ev_10_0.2_0.8'
        
        exp_name = 'signNet_64_remeshed_mass_6b_1-2ev_10_0.2_0.8'

        exp_dir = f'/home/s94zalek_hpc/shape_matching/my_code/experiments/sign_net/{exp_name}'

        with open(f'{exp_dir}/config.yaml', 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            
        start_dim = config['start_dim']
        feature_dim = config['feature_dim']
        evecs_per_support = config['evecs_per_support']


        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        net = diffusion_network.DiffusionNet(
            **config['net_params']
            ).to(device)
        
        net.load_state_dict(torch.load(f'{exp_dir}/50000.pth'))
        
        
        
        
        
        
        
        
        


        base_path = f'/lustre/mlnvme/data/s94zalek_hpc-shape_matching/figures/corr_vecs_evecs/{exp_name}'
        os.makedirs(f"{base_path}/single", exist_ok=True)
        os.makedirs(f"{base_path}/combined", exist_ok=True)
        
        # random_order = torch.randperm(len(idxs_geo_err))[:400]
        
        
        
        data = single_dataset[10]
              
    
        ##############################################
        # Set the variables
        ##############################################

        # data = double_shape['second']
        verts = data['verts'].unsqueeze(0).to(device)
        faces = data['faces'].unsqueeze(0).to(device)    

        evecs_orig = data['evecs'].unsqueeze(0)[:, :, config['start_dim']:config['start_dim']+config['feature_dim']].to(device)
        
        if 'with_mass' in config and config['with_mass']:
            mass_mat = torch.diag_embed(
                data['mass'].unsqueeze(0)
                ).to(device)
        else:
            mass_mat = None

        # predict the sign change
        with torch.no_grad():
            sign_pred_0, supp_vec_0, prod_0 = sign_training.predict_sign_change(
                net, verts, faces, evecs_orig, 
                mass_mat=mass_mat, input_type=net.input_type,
                evecs_per_support=config['evecs_per_support'],
                mass=data['mass'].unsqueeze(0), L=data['L'].unsqueeze(0),
                evals=data['evals'].unsqueeze(0), evecs=data['evecs'].unsqueeze(0),
                gradX=data['gradX'].unsqueeze(0), gradY=data['gradY'].unsqueeze(0)
                )
            
        if 'with_mass' in config and config["with_mass"]:

            print('Using mass')

            supp_vec_norm = torch.nn.functional.normalize(
                supp_vec_0[0].transpose(0, 1) \
                    @ mass_mat[0],
                p=2, dim=1)
            
            evecs_cond = supp_vec_norm @ evecs_orig[0]
            supp_vec_norm = supp_vec_norm.transpose(0, 1).unsqueeze(0)

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        random_order = range(0, 64, 1)
        
        for k in tqdm(random_order):
            
            indx = k

            
            # mesh = trimesh.Trimesh(data['verts'], data['faces'])
            
            # cmap = trimesh.visual.color.interpolate(data['evecs'][:, k], 'bwr')
            # mesh.visual.vertex_colors = cmap[:mesh.vertices.shape[0]]

            # png1 = render_mesh(scene, mesh)
            
            # cmap = trimesh.visual.color.interpolate(-data['evecs'][:, k], 'bwr')
            # mesh.visual.vertex_colors = cmap[:mesh.vertices.shape[0]]
            
            # png2 = render_mesh(scene, mesh)
            
            
            evec_id = k

            # supp_vec = supp_vec_0[0, :, evec_id].cpu()
            supp_vec = supp_vec_norm[0, :, evec_id].cpu()

            # supp_vec is a vector in [-1, 1]
            # make that the minimum negative value and maximum positive value have the same absolute value
            # but the zero value is still zero
            max_abs = torch.max(torch.abs(supp_vec))

            idx_min = torch.argmin(supp_vec)
            idx_max = torch.argmax(supp_vec)

            supp_vec[idx_min] = -max_abs
            supp_vec[idx_max] = max_abs


            mesh1 = trimesh.Trimesh(verts[0].cpu().numpy(), faces[0].cpu().numpy())
            cmap1 = trimesh.visual.color.interpolate(supp_vec, 'bwr')

            # smooth the colors
            # cmap1 = (cmap1.astype(np.int32) + np.roll(cmap1.astype(np.int32), 1) + np.roll(cmap1.astype(np.int32), -1)) / 3
            # cmap1 = cmap1.clip(0, 255).astype(np.uint8)

            cmap1_faces = trimesh.visual.color.vertex_to_face_color(cmap1, mesh1.faces)
            mesh1.visual.face_colors = cmap1_faces.clip(0, 255).astype(np.uint8)
            # mesh1.visual.vertex_colors = cmap1[:len(mesh1.vertices)].clip(0, 255).astype(np.uint8)

            mesh2 = trimesh.Trimesh(verts[0].cpu().numpy() + np.array([1, 0, 0]), faces[0].cpu().numpy())
            cmap2 = trimesh.visual.color.interpolate(evecs_orig[0, :, evec_id].cpu().numpy(), 'bwr')
            # mesh2.visual.vertex_colors = cmap2[:len(mesh2.vertices)].clip(0, 255).astype(np.uint8)

            cmap2_faces = trimesh.visual.color.vertex_to_face_color(cmap2, mesh2.faces)
            mesh2.visual.face_colors = cmap2_faces.clip(0, 255).astype(np.uint8)

            
            png1 = render_mesh(scene, mesh1)
            png2 = render_mesh(scene, mesh2)
            

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
                    
                    png_combined.save(f"{base_path}/combined/{k:04d}_combined_{sign_pred_0[0,k]:.2f}.png")
            
            
                