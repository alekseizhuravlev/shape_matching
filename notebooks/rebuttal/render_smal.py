import os

import numpy as np
import pyvirtualdisplay
import torch
import trimesh
from tqdm import tqdm

if __name__ == "__main__":
    data_surreal = torch.load(
        "/lustre/mlnvme/data/s94zalek_hpc-shape_matching/SMAL_shapes_train_nocat.pt",
        mmap=True,
        weights_only=True,
    )

    template = trimesh.load(
        "/home/s94zalek_hpc/shape_matching/data/SMAL_templates/original/template.off",
        process=False,
        validate=False,
    )
    scene = trimesh.Scene()

    base_dir = "/home/s94zalek_hpc/shape_matching/notebooks/rebuttal/smal_meshes"

    # remove the directory if it exists
    if os.path.exists(base_dir):
        os.system(f"rm -r {base_dir}")

    os.makedirs(base_dir, exist_ok=True)

    with pyvirtualdisplay.Display(visible=False, size=(1920, 1080)) as disp:
        for idx in tqdm(range(20, 200000, 2500)):
            scene.geometry.clear()

            verts = data_surreal[idx]

            verts_cloned = verts.clone()

            verts[:, 0] = verts_cloned[:, 2]
            verts[:, 1] = -verts_cloned[:, 1]
            verts[:, 2] = verts_cloned[:, 0]

            mesh_i = trimesh.Trimesh(vertices=verts, faces=template.faces)
            mesh_i.visual.vertex_colors = np.array([240, 240, 240, 255], dtype=np.uint8)

            mesh_i.apply_transform(mesh_i.principal_inertia_transform)

            mesh_i.apply_transform(
                trimesh.transformations.rotation_matrix(np.pi / 4, [0, 1, 0], [0, 0, 0])
            )

            scene.add_geometry(mesh_i)

            scene.set_camera()

            proportion = (mesh_i.vertices[:, 0].max() - mesh_i.vertices[:, 0].min()) / (
                mesh_i.vertices[:, 1].max() - mesh_i.vertices[:, 1].min()
            )

            png = scene.save_image(
                resolution=(int(proportion * 1080), 1080), visible=True
            )

            with open(f"{base_dir}/{idx}.png", "wb") as f:
                f.write(png)
