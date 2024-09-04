import pymeshlab
import torch
import trimesh
import numpy as np
import utils.fmap_util as fmap_util

ms = pymeshlab.MeshSet()


def augmentation_pipeline(verts_orig, faces_orig, augmentations):
    
    # randomly choose the remeshing type
    remesh_type = np.random.choice(['isotropic', 'anisotropic'], p=[1-augmentations["remesh"]["anisotropic"]["probability"], augmentations["remesh"]["anisotropic"]["probability"]])
    
    if remesh_type == 'isotropic':
        
        # isotropic remeshing
        simplify_strength = np.random.uniform(augmentations["remesh"]["isotropic"]["simplify_strength_min"], augmentations["remesh"]["isotropic"]["simplify_strength_max"])
        verts, faces = remesh_simplify_iso(
            verts_orig,
            faces_orig,
            n_remesh_iters=augmentations["remesh"]["isotropic"]["n_remesh_iters"],
            remesh_targetlen=augmentations["remesh"]["isotropic"]["remesh_targetlen"],
            simplify_strength=simplify_strength,
        )
    else:
        
        # anisotropic remeshing
        fraction_to_simplify = np.random.uniform(augmentations["remesh"]["anisotropic"]["fraction_to_simplify_min"], augmentations["remesh"]["anisotropic"]["fraction_to_simplify_max"])
        simplify_strength = np.random.uniform(augmentations["remesh"]["anisotropic"]["simplify_strength_min"], augmentations["remesh"]["anisotropic"]["simplify_strength_max"])
        
        verts, faces = remesh_simplify_anis(
            verts_orig,
            faces_orig,
            n_remesh_iters=augmentations["remesh"]["anisotropic"]["n_remesh_iters"],
            fraction_to_simplify=fraction_to_simplify,
            simplify_strength=simplify_strength,
            weighted_by=augmentations["remesh"]["anisotropic"]["weighted_by"]
        )
        
    # correspondence by a nearest neighbor search
    corr = fmap_util.nn_query(
        verts,
        verts_orig, 
        )
    
    return verts, faces, corr


def remesh_simplify_iso(
    verts,
    faces,
    n_remesh_iters,
    remesh_targetlen,
    simplify_strength,
):
    mesh = pymeshlab.Mesh(verts, faces)
    ms.add_mesh(mesh)

    if n_remesh_iters > 0:
        ms.meshing_isotropic_explicit_remeshing(
            iterations=n_remesh_iters,
            targetlen=pymeshlab.PercentageValue(remesh_targetlen),
        )
        
    if 0 < simplify_strength < 1:
        ms.meshing_decimation_quadric_edge_collapse(
            targetperc=simplify_strength,
        )
        
    v_qec = torch.tensor(
        ms.current_mesh().vertex_matrix(), dtype=torch.float32
    )
    f_qec = torch.tensor(
        ms.current_mesh().face_matrix(), dtype=torch.int32
    )
    
    ms.clear()
    
    return v_qec, f_qec


def remesh_simplify_anis(
    verts,
    faces,
    n_remesh_iters,
    fraction_to_simplify,
    simplify_strength,
    weighted_by
    ):
    
    assert weighted_by in ['area', 'face_count']   
    
    mesh = pymeshlab.Mesh(verts, faces)
    ms.add_mesh(mesh)
    
    # isotropic remeshing
    if n_remesh_iters > 0:
        ms.meshing_isotropic_explicit_remeshing(
            iterations=n_remesh_iters,
        ) 
        
    # mesh after remeshing   
    v_r = ms.current_mesh().vertex_matrix()
    f_r = ms.current_mesh().face_matrix()
    
    if weighted_by == 'area':
        # face area
        mesh_r = trimesh.Trimesh(v_r, f_r)
        area_faces = mesh_r.area_faces
        total_area_faces = area_faces.sum()

        # choose a random face, with probability proportional to its area
        rand_idx = np.random.choice(len(area_faces), p=area_faces / total_area_faces)
        
    elif weighted_by == 'face_count':
        # choose a random face
        rand_idx = np.random.randint(0, len(f_r))

    # select the face
    ms.set_selection_none()
    ms.compute_selection_by_condition_per_face(
        condselect= f'(fi == {rand_idx})'
    )
    
    # select the simplification area by dilatation
    for dil_iter in range(100):
        
        # stopping criterion
        if weighted_by == 'area':
            selected_area = sum(area_faces[ms.current_mesh().face_selection_array()])
            if selected_area >= total_area_faces * fraction_to_simplify:
                # print('dil_iter', dil_iter)
                break
            
        elif weighted_by == 'face_count':
            selected_faces = sum(ms.current_mesh().face_selection_array())
            if selected_faces >= len(f_r) * fraction_to_simplify:
                # print('dil_iter', dil_iter)
                break
        ms.apply_selection_dilatation()
        

    selected_faces = ms.current_mesh().face_selection_array()

    # simplify the mesh
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=int(sum(ms.current_mesh().face_selection_array()) * simplify_strength),
        selected=True
    )

    # get the vertices and faces
    v_qec = torch.tensor(
        ms.current_mesh().vertex_matrix(), dtype=torch.float32
    )
    f_qec = torch.tensor(
        ms.current_mesh().face_matrix(), dtype=torch.int32
    )
    
    ms.clear()
    
    return v_qec, f_qec
    # return v_qec, f_qec, v_r, f_r, selected_faces
    

