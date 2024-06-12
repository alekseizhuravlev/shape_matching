import numpy as np
import trimesh
import torch
import os

from utils.geometry_util import laplacian_decomposition, get_operators

def center(vertices):
    """
    Center the input mesh using its bounding box
    """
    raise RuntimeError("Use center_mean")

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
    
    raise RuntimeError("Use normalize_face_area")
    
    input_mesh = trimesh.Trimesh(vertices=input_verts, faces=input_faces, process=False)
    ref_mesh = trimesh.Trimesh(vertices=ref_verts, faces=ref_faces, process=False)
    
    area = np.power(ref_mesh.volume / input_mesh.volume, 1.0/3)
    scaled_verts = input_verts * torch.tensor(area)
    
    return scaled_verts, area


def center_mean(verts):
    """
    Center the vertices by subtracting the mean
    """
    verts -= torch.mean(verts, axis=0)
    return verts


def center_bbox(vertices):
    """
    Center the input mesh using its bounding box
    """
    bbox = torch.tensor([[torch.max(vertices[:,0]), torch.max(vertices[:,1]), torch.max(vertices[:,2])],
                     [torch.min(vertices[:,0]), torch.min(vertices[:,1]), torch.min(vertices[:,2])]])

    translation = (bbox[0] + bbox[1]) / 2
    translated_vertices = vertices - translation
    
    return translated_vertices


def normalize_face_area(verts, faces):
    """
    Calculate the square root of the area through laplacian decomposition
    Normalize the vertices by it
    """
    verts = np.array(verts)
    faces = np.array(faces)
    
    old_sqrt_area = laplacian_decomposition(verts=verts, faces=faces, k=1)[-1]
    verts /= old_sqrt_area
    
    return torch.tensor(verts)


def get_spectral_ops(item, num_evecs, cache_dir=None):
    if cache_dir is not None and not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    _, mass, L, evals, evecs, _, _ = get_operators(item['verts'], item.get('faces'),
                                                   k=num_evecs,
                                                   cache_dir=cache_dir)
    evals = evals.unsqueeze(0)
    evecs_trans = evecs.T * mass[None]
    item['evecs'] = evecs[:, :num_evecs]
    item['evecs_trans'] = evecs_trans[:num_evecs]
    item['evals'] = evals[:num_evecs]
    item['mass'] = mass
    item['L'] = L.to_dense()

    return item
