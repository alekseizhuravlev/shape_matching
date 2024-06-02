import numpy as np
import matplotlib.pyplot as plt
# from utils.registry import METRIC_REGISTRY

counter = 0

# @METRIC_REGISTRY.register()
def calculate_geodesic_error(dist_x, corr_x, corr_y, p2p, return_mean=True):
    """
    Calculate the geodesic error between predicted correspondence and gt correspondence

    Args:
        dist_x (np.ndarray): Geodesic distance matrix of shape x. shape [Vx, Vx]
        corr_x (np.ndarray): Ground truth correspondences of shape x. shape [V]
        corr_y (np.ndarray): Ground truth correspondences of shape y. shape [V]
        p2p (np.ndarray): Point-to-point map (shape y -> shape x). shape [Vy]
        return_mean (bool, optional): Average the geodesic error. Default True.
    Returns:
        avg_geodesic_error (np.ndarray): Average geodesic error.
    """
    
    
    # def perm2mat(perm, dim0):
    #     import torch
        
    #     n = len(perm)
    #     mat = torch.zeros(dim0, n)
    #     mat[perm, torch.arange(n)] = 1
    #     return mat
    
    # print('Calculating geodesic error')
    # print('dist_x.shape:', dist_x.shape)
    # print('corr_x.shape:', corr_x.shape)
    # print('corr_y.shape:', corr_y.shape)
    # print('p2p.shape:', p2p.shape)
    
    # print('p2p[corr_y].shape:', p2p[corr_y].shape)
    
    # print('corr_x:', corr_x[:10])
    # print('corr_y:', corr_y[:10])
    # print('p2p:', p2p[:10])
    # print('p2p[corr_y]:', p2p[corr_y][:10])
    
    
    # P_yGT = perm2mat(corr_y, dim0=p2p.shape[0]).to('cuda')
    # P_xGT = perm2mat(corr_x, dim0=5000).to('cuda')
    
    # p2p_mat = perm2mat(p2p, dim0=5000).to('cuda')
    
    # rhs = p2p_mat @ P_yGT
    
    # rhs_vec = np.argmax(rhs.cpu(), axis=0)
    # # print('rhs_vec:', rhs_vec[:10])
    
    # assert np.allclose(rhs_vec.numpy(), p2p[corr_y])
    
    # global counter
    # counter += 1
    
    # if counter > 1:
    #     exit(0)
    
    
    ind21 = np.stack([corr_x, p2p[corr_y]], axis=-1)
    
    ind21 = np.ravel_multi_index(ind21.T, dims=[dist_x.shape[0], dist_x.shape[0]])
    geo_err = np.take(dist_x, ind21)
    if return_mean:
        return geo_err.mean()
    else:
        return geo_err


# @METRIC_REGISTRY.register()
def plot_pck(geo_err, threshold=0.10, steps=40, show_figure=False):
    """
    plot pck curve and compute auc.
    Args:
        geo_err (np.ndarray): geodesic error list.
        threshold (float, optional): threshold upper bound. Default 0.15.
        steps (int, optional): number of steps between [0, threshold]. Default 30.
    Returns:
        auc (float): area under curve.
        fig (matplotlib.pyplot.figure): pck curve.
        pcks (np.ndarray): pcks.
    """
    assert threshold > 0 and steps > 0
    geo_err = np.ravel(geo_err)
    thresholds = np.linspace(0., threshold, steps)
    pcks = []
    for i in range(thresholds.shape[0]):
        thres = thresholds[i]
        pck = np.mean((geo_err <= thres).astype(float))
        pcks.append(pck)
    pcks = np.array(pcks)
    # compute auc
    auc = np.trapz(pcks, np.linspace(0., 1., steps))

    # display figure
    if show_figure:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(thresholds, pcks, 'r-')
        ax.set_xlim(0., threshold)
        return auc, fig, pcks
    else:
        return auc, pcks, thresholds
