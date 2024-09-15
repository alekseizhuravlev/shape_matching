import torch

def get_median_p2p_map(p2p_maps, dist_x):
    
    assert len(p2p_maps.shape) == 2, "p2p_maps should be [n, dist_x.shape[0]]"
    
    # print("p2p_maps.shape", p2p_maps.shape)
    
    median_p2p_map = torch.zeros(p2p_maps.shape[1], dtype=torch.int64)
    
    # print("median_p2p_map.shape", median_p2p_map.shape)
    
    for i in range(p2p_maps.shape[1]):
    
        vertex_indices = p2p_maps[:, i]
        
        
    
        geo_dists_points = dist_x[vertex_indices][:, vertex_indices]
        # [300, 300]
        
        # if i == 0:
        #     print("vertex_indices.shape", vertex_indices.shape)
        #     print("geo_dists_points.shape", geo_dists_points.shape)
        
        # find index of minimum geo_dists_points.sum(axis=1)
        idx_median = vertex_indices[
            torch.argmin(geo_dists_points.sum(axis=1))
        ]
        
        median_p2p_map[i] = idx_median
        
    return median_p2p_map
    
    