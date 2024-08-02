from my_code.datasets.surreal_dataset_3dc import TemplateSurrealDataset3DC
import utils.geometry_util as geometry_util
import scipy.sparse
import zipfile
import numpy as np
from tqdm import tqdm
import os


def zip_write_operators(verts, faces, k, cache_zip, search_path, normals=None):

    assert verts.dim() == 2, 'Please call get_all_operators() for a batch of vertices'
    
    verts_np = geometry_util.torch2np(verts)
    faces_np = geometry_util.torch2np(faces) if faces is not None else None

    if np.isnan(verts_np).any():
        raise ValueError('detect NaN vertices.')

    # recompute
    frames, mass, L, evals, evecs, gradX, gradY = geometry_util.compute_operators(verts, faces, k, normals)

    dtype_np = np.float32

    # save
    frames_np = geometry_util.torch2np(frames).astype(dtype_np)
    mass_np = geometry_util.torch2np(mass).astype(dtype_np)
    evals_np = geometry_util.torch2np(evals).astype(dtype_np)
    evecs_np = geometry_util.torch2np(evecs).astype(dtype_np)
    L_np = geometry_util.sparse_torch_to_np(L).astype(dtype_np)
    gradX_np = geometry_util.sparse_torch_to_np(gradX).astype(dtype_np)
    gradY_np = geometry_util.sparse_torch_to_np(gradY).astype(dtype_np)

    # save to zip
    with cache_zip.open(search_path, 'w') as f:
        np.savez(
            f,
            verts=verts_np,
            faces=faces_np,
            k_eig=k,
            frames=frames_np,
            mass=mass_np,
            evals=evals_np,
            evecs=evecs_np,
            L_data=L_np.data,
            L_indices=L_np.indices,
            L_indptr=L_np.indptr,
            L_shape=L_np.shape,
            gradX_data=gradX_np.data,
            gradX_indices=gradX_np.indices,
            gradX_indptr=gradX_np.indptr,
            gradX_shape=gradX_np.shape,
            gradY_data=gradY_np.data,
            gradY_indices=gradY_np.indices,
            gradY_indptr=gradY_np.indptr,
            gradY_shape=gradY_np.shape,
        )
    return


if __name__ == '__main__':
    
    idx_start = 0
    idx_end = 10000

    base_folder = '/lustre/mlnvme/data/s94zalek_hpc-shape_matching/SURREAL'
    os.makedirs(base_folder, exist_ok=True)

    # create the dataset
    dataset = TemplateSurrealDataset3DC(
        # shape_path=f'/home/s94zalek_hpc/3D-CODED/data/datas_surreal_train.pth',
        shape_path='/lustre/mlnvme/data/s94zalek_hpc-shape_matching/mmap_datas_surreal_train.pth',
        num_evecs=128,
        use_cuda=False,
        cache_lb_dir=None,
        return_evecs=False
    )    
    
    
    with zipfile.ZipFile(f'{base_folder}/{idx_start:06d}_{idx_end:06d}.zip',
                            'w', compression=zipfile.ZIP_STORED) as zip_file:
        for i in tqdm(range(idx_start, idx_end)):
            
            data_i = dataset[i]
            verts = data_i['second']['verts']
            faces = data_i['second']['faces']
            
            # print(f'Processing {i:06d}')
            zip_write_operators(verts=verts, faces=faces, k=128,
                                cache_zip=zip_file, search_path=f'{i:06d}.npz')
            
    
    