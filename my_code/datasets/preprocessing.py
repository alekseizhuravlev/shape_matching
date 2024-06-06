import numpy as np
import trimesh
import torch

def center(vertices):
    """
    Center the input mesh using its bounding box
    """
    bbox = torch.tensor([[torch.max(vertices[:,0]), torch.max(vertices[:,1]), torch.max(vertices[:,2])],
                     [torch.min(vertices[:,0]), torch.min(vertices[:,1]), torch.min(vertices[:,2])]])

    translation = (bbox[0] + bbox[1]) / 2
    translated_vertices = vertices - translation
    
    return translated_vertices, translation


def scale(input_verts, input_faces,
          ref_verts, ref_faces):
    """
    Scales the input mesh to match the volume of the reference mesh
    """
    
    input_mesh = trimesh.Trimesh(vertices=input_verts, faces=input_faces, process=False)
    ref_mesh = trimesh.Trimesh(vertices=ref_verts, faces=ref_faces, process=False)
    
    area = np.power(ref_mesh.volume / input_mesh.volume, 1.0/3)
    scaled_verts = input_verts * torch.tensor(area)
    
    return scaled_verts, area