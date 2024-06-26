import numpy as np

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