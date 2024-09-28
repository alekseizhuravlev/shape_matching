import utils.fmap_util as fmap_util

def zoomout(FM_12, evects1, evects2, nit, step, A2=None):

    k = FM_12.shape[0]
    for _ in range(nit):
        
        p2p_21 = fmap_util.fmap2pointmap(FM_12, evects1[:, :k], evects2[:, :k])
        
        k = k + step
        
        # FM_12 = fmap_util.pointmap2fmap(p2p_21, evects1[:, :k], evects2[:, :k])
        
        if A2 is not None:
            if A2.ndim == 1:
                FM_12 = evects2[:, :k].T @ (A2[:, None] * evects1[p2p_21, :k])
            else:
                FM_12 = evects2[:, :k].T @ (A2 @ evects1[p2p_21, :k])
        else:
            FM_12 = fmap_util.pointmap2fmap(p2p_21, evects2[:, :k], evects1[:, :k])
        
    return FM_12

    