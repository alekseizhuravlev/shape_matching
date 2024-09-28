import torch

def get_median_p2p_map(p2p_maps, dist_x):
    
    assert len(p2p_maps.shape) == 2, "p2p_maps should be [n, dist_x.shape[0]]"
    
    # print("p2p_maps.shape", p2p_maps.shape)
    
    median_p2p_map = torch.zeros(p2p_maps.shape[1], dtype=torch.int64)
    confidence_scores = torch.zeros(p2p_maps.shape[1])
    
    for i in range(p2p_maps.shape[1]):
    
        vertex_indices = p2p_maps[:, i]
        
        geo_dists_points = dist_x[vertex_indices][:, vertex_indices]

        # find index of minimum geo_dists_points.sum(axis=1)
        idx_median = vertex_indices[
            torch.argmin(geo_dists_points.sum(axis=1))
        ]
        
        median_p2p_map[i] = idx_median
        
        # confidence score
        dists_to_median = dist_x[vertex_indices][:, vertex_indices][torch.argmin(geo_dists_points.sum(axis=1))]
        
        confidence_scores[i] = torch.sqrt(
            torch.mean(dists_to_median.square())
            )
        
    return median_p2p_map, confidence_scores
    
    
def dirichlet_energy(p2p_12, X_2, W_1):
    """
    p2p_12: point-to-point map from mesh 1 to mesh 2
    X_2: vertices of mesh 2
    W_1: Laplacian of mesh 1
    """
 
    assert len(p2p_12.shape) == 1
    assert len(X_2.shape) == 2
    assert len(W_1.shape) == 2
    
    mapped_verts = X_2[p2p_12]
    
    return torch.trace(mapped_verts.transpose(0, 1) @ W_1 @ mapped_verts)
    