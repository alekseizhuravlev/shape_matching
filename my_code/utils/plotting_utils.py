import numpy as np
import trimesh
import torch

def plot_Cxy(figure, axis, Cxy_plt, title, min_dim, max_dim, show_grid, show_colorbar):
    
    axis_plot = axis.imshow(Cxy_plt[min_dim:max_dim, min_dim:max_dim], cmap='bwr', vmin=-1, vmax=1)
    
    if show_colorbar:
        figure.colorbar(axis_plot, ax=axis)
    
    axis.set_title(f'{title}: {min_dim}-{max_dim}')

    axis.set_xticks(np.arange(-0.5, max_dim - min_dim, 1.0))
    axis.set_yticks(np.arange(-0.5, max_dim - min_dim, 1.0)) 
    
    if show_grid:
        axis.grid(which='both')    
    
    axis.set_xticklabels([])
    axis.set_yticklabels([])
    
    
def plot_p2p_map(scene, verts_x, faces_x, verts_y, faces_y, p2p, axes_color_gradient=[0, 1],
                 base_cmap='jet'):
    
    # assert axes_color_gradient is a list or tuple
    assert isinstance(axes_color_gradient, (list, tuple)), "axes_color_gradient must be a list or tuple"
    assert verts_y.shape[0] == len(p2p), f"verts_y {verts_y.shape} and p2p {p2p.shape} must have the same length"
    
    
    # normalize verts_x[:, 0] between 0 and 1
    # coords_x_norm = (verts_x[:, 0] - verts_x[:, 0].min()) / (verts_x[:, 0].max() - verts_x[:, 0].min())
    # coords_y_norm = (verts_x[:, 1] - verts_x[:, 1].min()) / (verts_x[:, 1].max() - verts_x[:, 1].min())
    # coords_z_norm = (verts_x[:, 2] - verts_x[:, 2].min()) / (verts_x[:, 2].max() - verts_x[:, 2].min())

    coords_x_norm = torch.zeros_like(verts_x)
    for i in range(3):
        coords_x_norm[:, i] = (verts_x[:, i] - verts_x[:, i].min()) / (verts_x[:, i].max() - verts_x[:, i].min())

    coords_interpolated = torch.zeros(verts_x.shape[0])
    for i in axes_color_gradient:
        coords_interpolated += coords_x_norm[:, i]
        
    # first colormap = interpolated y-axis values
    cmap = trimesh.visual.color.interpolate(coords_interpolated, base_cmap)
    
    # cmap = trimesh.visual.color.interpolate(verts_x[:, 0] + verts_x[:, 1], 'jet')
    
    # second colormap = first colormap values mapped to second mesh
    cmap2 = cmap[p2p].clip(0, 255)

    # add the first mesh
    mesh1 = trimesh.Trimesh(vertices=verts_x, faces=faces_x, validate=True)
    mesh1.visual.vertex_colors = cmap[:len(mesh1.vertices)].clip(0, 255)
    scene.add_geometry(mesh1)
    
    
    
    # add the second mesh
    mesh2 = trimesh.Trimesh(vertices=verts_y + np.array([1, 0, 0]), faces=faces_y, validate=True)
    mesh2.visual.vertex_colors = cmap2[:len(mesh2.vertices)]
    scene.add_geometry(mesh2)
    
    scene.add_geometry(trimesh.creation.axis(origin_size=0.05))

    return scene