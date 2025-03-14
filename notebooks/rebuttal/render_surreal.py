import torch
import trimesh
import numpy as np
import pyvirtualdisplay
from tqdm import tqdm
import os


if __name__ == '__main__':
    data_surreal = torch.load('/lustre/mlnvme/data/s94zalek_hpc-shape_matching/mmap_datas_surreal_train.pth', mmap=True, weights_only=True)


    template = trimesh.load('/home/s94zalek_hpc/shape_matching/data/SURREAL_full/template/original/template.off',
                            process=False, validate=False)
    scene = trimesh.Scene()


    with pyvirtualdisplay.Display(visible=False, size=(1920, 1080)) as disp:

        for idx in tqdm(range(20, 220000, 10000)):

            scene.geometry.clear()

            # idx = 20

            mesh_i = trimesh.Trimesh(vertices=data_surreal[idx].cpu().numpy(), faces=template.faces)
            mesh_i.visual.vertex_colors = np.array([240, 240, 240, 255], dtype=np.uint8)


            scene.add_geometry(mesh_i)


            scene.set_camera()
        

            proportion = (mesh_i.vertices[:, 0].max() - mesh_i.vertices[:, 0].min()) / (mesh_i.vertices[:, 1].max() - mesh_i.vertices[:, 1].min())

            png = scene.save_image(resolution=(int(proportion*1080), 1080), visible=True)

            base_dir = "/home/s94zalek_hpc/shape_matching/notebooks/rebuttal/surreal_meshes"
            os.makedirs(base_dir, exist_ok=True)

            with open(f"{base_dir}/{idx}.png", "wb") as f:
                f.write(png)