import pymeshlab
import torch

ms = pymeshlab.MeshSet()

def remesh_simplify(
    verts,
    faces,
    n_remesh_iters=10,
    simplify_percent=0.75,
):
    mesh = pymeshlab.Mesh(verts, faces)
    ms.add_mesh(mesh)

    if n_remesh_iters > 0:
        ms.meshing_isotropic_explicit_remeshing(
            iterations=n_remesh_iters,
        )
        
    if 0 < simplify_percent < 1:
        ms.meshing_decimation_quadric_edge_collapse(
            targetperc=simplify_percent,
        )
        
    v_qec = torch.tensor(
        ms.current_mesh().vertex_matrix(), dtype=torch.float32
    )
    f_qec = torch.tensor(
        ms.current_mesh().face_matrix(), dtype=torch.int32
    )
    
    ms.clear()
    
    return v_qec, f_qec
