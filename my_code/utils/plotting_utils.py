import numpy as np
import trimesh


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
    
    
def plot_p2p_map(scene, verts_x, faces_x, verts_y, faces_y, p2p):

    # first colormap = interpolated y-axis values
    cmap = trimesh.visual.color.interpolate(verts_x[:, 1], 'jet')
    
    # second colormap = first colormap values mapped to second mesh
    cmap2 = cmap[p2p].clip(0, 255)

    # add the first mesh
    mesh1 = trimesh.Trimesh(vertices=verts_x, faces=faces_x)
    mesh1.visual.vertex_colors = cmap[:len(mesh1.vertices)].clip(0, 255)
    scene.add_geometry(mesh1)
    
    # add the second mesh
    mesh2 = trimesh.Trimesh(vertices=verts_y + np.array([1, 0, 0]), faces=faces_y)
    mesh2.visual.vertex_colors = cmap2[:len(mesh2.vertices)]
    scene.add_geometry(mesh2)

    return scene