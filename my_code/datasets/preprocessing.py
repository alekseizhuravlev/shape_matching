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


def inertia_transform(verts, faces):
    
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    
    mesh.apply_transform(mesh.principal_inertia_transform)
    
    return torch.tensor(mesh.vertices, dtype=torch.float32)
    


def get_spectral_ops(item, num_evecs, cache_dir=None):
    if cache_dir is not None and not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    # _, mass, L, evals, evecs, _, _ = get_operators(item['verts'], item.get('faces'),
    #                                                k=num_evecs,
    #                                                cache_dir=cache_dir)
    # evals = evals.unsqueeze(0)
    # evecs_trans = evecs.T * mass[None]
    # item['evecs'] = evecs[:, :num_evecs]
    # item['evecs_trans'] = evecs_trans[:num_evecs]
    # item['evals'] = evals[:num_evecs]
    # item['mass'] = mass
    # item['L'] = L.to_dense()
    
    _, mass, L, evals, evecs, gradX, gradY = get_operators(item['verts'], item.get('faces'),
                                                   k=num_evecs,
                                                   cache_dir=cache_dir)
    # evals = evals.unsqueeze(0)
    evecs_trans = evecs.T * mass[None]
    item['evecs'] = evecs[:, :num_evecs]
    item['evecs_trans'] = evecs_trans[:num_evecs]
    item['evals'] = evals[:num_evecs]
    item['mass'] = mass
    # item['L'] = L.to_dense()
    # item['gradX'] = gradX.to_dense()
    # item['gradY'] = gradY.to_dense()
    item['L'] = L
    item['gradX'] = gradX
    item['gradY'] = gradY

    return item


def canonicalize_fmap(canon_type, data_payload):
    
    # initial functional map
    C_gt_xy = data_payload['second']['C_gt_xy'].clone()
    assert len(C_gt_xy.shape) == 3, f'Expected 3D tensor, got {C_gt_xy.shape}'
    
    # initial eigenvectors and correspondences
    evecs_first = data_payload['first']['evecs'].clone()
    evecs_second = data_payload['second']['evecs'].clone()

    corr_first = data_payload['first']['corr']
    corr_second = data_payload['second']['corr']


    ###########################################################
    # Set the signs
    ###########################################################

    # option 1: take the sign of the sum of the row
    sum_per_row = torch.sum(C_gt_xy, dim=2).transpose(0, 1)
    signs_sum_per_row = torch.sign(sum_per_row)
    
    # option 2: take the sign of the max element of the row
    arg_max_in_row = torch.argmax(torch.abs(C_gt_xy), dim=2)
    sign_max_in_row = torch.sign(
        C_gt_xy[torch.arange(C_gt_xy.shape[0]), torch.arange(C_gt_xy.shape[1]), arg_max_in_row]
    ).transpose(0, 1)
    
    # option 3: for rows that have abs sum < 0.2, use the max
    abs_sum = torch.abs(sum_per_row).flatten()
    mask = abs_sum < 0.2

    signs_both = torch.ones_like(signs_sum_per_row)
    signs_both[mask] = signs_both[mask] * sign_max_in_row[mask]
    signs_both[~mask] = signs_both[~mask] * signs_sum_per_row[~mask]

       
    # update the eigenvectors
    if canon_type == 'sum':
        evecs_second = evecs_second * signs_sum_per_row[:, 0]
        
    elif canon_type == 'max':
        evecs_second = evecs_second * sign_max_in_row[:, 0]
        
    elif canon_type == 'both':
        evecs_second = evecs_second * signs_both[:, 0]
        
        # abs_sum = torch.abs(sum_per_row)
        # mask = abs_sum < 0.2
        # evecs_second[mask] = evecs_second[mask] * sign_max_in_row[mask, 0]
        # evecs_second[~mask] = evecs_second[~mask] * signs_sum_per_row[~mask, 0]
        
    else:
        raise ValueError(f'Unknown canonicalization type {canon_type}')

    # canonicalize the functional map
    C_gt_xy_norm = torch.linalg.lstsq(
        evecs_second[corr_second],
        evecs_first[corr_first]
        ).solution
    
    return C_gt_xy_norm.unsqueeze(0), evecs_second
    
    # # save the old values
    # data_payload['second']['C_gt_xy_uncan'] = data_payload['second']['C_gt_xy'].detach().clone()
    # data_payload['second']['evecs_uncan'] = data_payload['second']['evecs'].detach().clone()
    
    # # update the data payload
    # data_payload['second']['C_gt_xy'] = C_gt_xy_norm.unsqueeze(0)
    # data_payload['second']['evecs'] = evecs_second 
    
    # return data_payload
