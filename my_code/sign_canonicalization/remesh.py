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


def augmentation_pipeline_partial(verts_orig, faces_orig, augmentations):
    
    
    # randomly choose the remeshing type
    remesh_type = np.random.choice(
        ['isotropic', 'anisotropic'],
        p=[1-augmentations["remesh"]["anisotropic"]["probability"],
           augmentations["remesh"]["anisotropic"]["probability"]])
    
    
    if remesh_type == 'isotropic':
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
        
        
    apply_partiality = np.random.choice(
        ['none', 'partial'],
        p=[1-augmentations["remesh"]["partial"]["probability"],
           augmentations["remesh"]["partial"]["probability"]])
    
        
    if apply_partiality == 'partial':
        
        fraction_to_keep = np.random.uniform(augmentations["remesh"]["partial"]["fraction_to_keep_min"], augmentations["remesh"]["partial"]["fraction_to_keep_max"])
        n_seed_samples = np.random.choice(augmentations["remesh"]["partial"]["n_seed_samples"])
        
        remove_selection = n_seed_samples != 1
        if remove_selection:
            fraction_to_select = 1 - fraction_to_keep
        else:
            fraction_to_select = fraction_to_keep

        verts, faces = remesh_partial(
            # verts_orig,
            # faces_orig,
            # n_remesh_iters=augmentations["remesh"]["partial"]["n_remesh_iters"],
            
            verts,
            faces,
            fraction_to_select=fraction_to_select,
            n_seed_samples=n_seed_samples,
            weighted_by=augmentations["remesh"]["partial"]["weighted_by"],
            remove_selection=remove_selection
        )
        
    # !!! correspondences are going in the opposite direction
    corr = fmap_util.nn_query(
        verts_orig,
        verts,
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
    

def remesh_partial(
    verts,
    faces,
    # n_remesh_iters,
    fraction_to_select,
    n_seed_samples,
    weighted_by,
    remove_selection: bool,
    ):
    
    assert weighted_by in ['area', 'face_count']

    mesh = pymeshlab.Mesh(verts, faces)
    ms.add_mesh(mesh)
    
    # isotropic remeshing
    # if n_remesh_iters > 0:
    #     ms.meshing_isotropic_explicit_remeshing(
    #         iterations=n_remesh_iters,
    #     ) 
        
    # mesh after remeshing   
    # v_r = ms.current_mesh().vertex_matrix()
    # f_r = ms.current_mesh().face_matrix()
    
    v_r = verts
    f_r = faces
    
    if weighted_by == 'area':
        # face area
        mesh_r = trimesh.Trimesh(v_r, f_r, process=False)
        area_faces = mesh_r.area_faces
        total_area_faces = area_faces.sum()

        # choose a random face, with probability proportional to its area
        rand_idxs = np.random.choice(len(area_faces), size=n_seed_samples,
                                    p=area_faces / total_area_faces)
        
    elif weighted_by == 'face_count':
        # choose a random face
        rand_idxs = np.random.randint(0, len(f_r), size=n_seed_samples)

    # select the face
    ms.set_selection_none()
    
    # make a query string to select all faces with rand_idxs
    query_str = ''
    for i, rand_idx in enumerate(rand_idxs):
        query_str += f'(fi == {rand_idx})'
        if i < len(rand_idxs) - 1:
            query_str += ' || '
    
    ms.compute_selection_by_condition_per_face(
        condselect= query_str
    )
    
    # select the simplification area by dilatation
    for dil_iter in range(100):
        
        # stopping criterion
        if weighted_by == 'area':
            selected_area = sum(area_faces[ms.current_mesh().face_selection_array()])
            if selected_area >= total_area_faces * fraction_to_select:
                # print('dil_iter', dil_iter)
                break
            
        elif weighted_by == 'face_count':
            selected_faces = sum(ms.current_mesh().face_selection_array())
            if selected_faces >= len(f_r) * fraction_to_select:
                # print('dil_iter', dil_iter)
                break
        ms.apply_selection_dilatation()
        
    selected_faces = ms.current_mesh().face_selection_array()


    if remove_selection:
    
        # remove the selected faces
        ms.meshing_remove_selected_vertices_and_faces()
        
        ms.generate_splitting_by_connected_components()
        
        # get the number of vertices in each connected component
        n_vertices_list = []
        for i in range(ms.mesh_number()):
            mesh_i = ms.mesh(i)
            n_vertices_list.append(mesh_i.vertex_matrix().shape[0])
               
        # sort the connected components by the number of vertices, ascending
        idx_max_vertices = np.argsort(n_vertices_list)  
        
        # get the vertices and faces of the largest connected component
        # 2nd from the end, last one is the full mesh        
        mesh_partial = ms.mesh(idx_max_vertices[-2])
            
    else:
        ms.generate_from_selected_faces()
        
        # n_vertices_list = []
        # for i in range(ms.mesh_number()):
        #     mesh_i = ms.mesh(i)
        #     n_vertices_list.append(mesh_i.vertex_matrix().shape[0])
            
        # print('n_vertices_list:', n_vertices_list)
            
        # idx_max_vertices = np.argsort(n_vertices_list)
        
        # mesh_partial = ms.mesh(idx_max_vertices[0])
        
        # mesh_partial = ms.mesh(0)
        
        mesh_partial = ms.current_mesh()
        
    v_qec = torch.tensor(
        mesh_partial.vertex_matrix(), dtype=torch.float32
    )
    f_qec = torch.tensor(
        mesh_partial.face_matrix(), dtype=torch.int32
    )

    # check that there are no disconnected components    
    mesh_result = trimesh.Trimesh(v_qec, f_qec, process=False)
    connected_components = len(trimesh.graph.connected_components(
        mesh_result.edges
    ))
    assert connected_components == 1, f'More than one connected component: {connected_components}'

    # print('Connected components:', connected_components)
    # if connected_components > 1:
    #     print('!!!!!!!! More than one connected component')
    
    ms.clear()
    
    return v_qec, f_qec