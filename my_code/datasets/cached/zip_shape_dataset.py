import utils.geometry_util as geometry_util
import scipy.sparse
import zipfile
import numpy as np
from tqdm import tqdm
import os
import torch


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


def read_sp_mat(npzfile, prefix):
    data = npzfile[prefix + '_data']
    indices = npzfile[prefix + '_indices']
    indptr = npzfile[prefix + '_indptr']
    shape = npzfile[prefix + '_shape']
    mat = scipy.sparse.csc_matrix((data, indices, indptr), shape=shape)
    return mat


def zip_read_operators(cache_zip, search_path, k):

    with cache_zip.open(search_path) as f:
        with np.load(f, allow_pickle=True) as npzfile:

            cache_verts = npzfile['verts']
            cache_faces = npzfile['faces']
            cache_k = npzfile['k_eig'].item()
            
            if cache_k < k:
                raise ValueError(f'cache_k={cache_k} is less than k={k}.')

            # this entry matches. return it.
            frames = npzfile['frames']
            mass = npzfile['mass']
            L = read_sp_mat(npzfile, 'L')
            evals = npzfile['evals'][:k]
            evecs = npzfile['evecs'][:, :k]
            gradX = read_sp_mat(npzfile, 'gradX')
            gradY = read_sp_mat(npzfile, 'gradY')

    device = torch.device('cpu')
    dtype = torch.float32

    cache_verts = torch.from_numpy(cache_verts).to(device=device, dtype=dtype)
    cache_faces = torch.from_numpy(cache_faces).to(device=device, dtype=torch.int64)
    frames = torch.from_numpy(frames).to(device=device, dtype=dtype)
    mass = torch.from_numpy(mass).to(device=device, dtype=dtype)
    L = geometry_util.sparse_np_to_torch(L).to(device=device, dtype=dtype)
    evals = torch.from_numpy(evals).to(device=device, dtype=dtype)
    evecs = torch.from_numpy(evecs).to(device=device, dtype=dtype)
    gradX = geometry_util.sparse_np_to_torch(gradX).to(device=device, dtype=dtype)
    gradY = geometry_util.sparse_np_to_torch(gradY).to(device=device, dtype=dtype)


    return frames, cache_verts, cache_faces, mass, L, evals, evecs, gradX, gradY


class ZipCollection:
    '''
    Context manager for opening and closing multiple zip files
    '''
    
    def __init__(self, zip_files_path_list):
        self.zip_files_path_list = zip_files_path_list
        self.zip_files = []
        
    def __enter__(self):
        print(f'Opening {len(self.zip_files_path_list)} zip files...')
        
        for zip_file_path in self.zip_files_path_list:
            self.zip_files.append(zipfile.ZipFile(zip_file_path, 'r'))
        return self.zip_files
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        print(f'Closing {len(self.zip_files)} zip files...')
        
        for zip_file in self.zip_files:
            zip_file.close()
        return
    
    def open(self):
        return self.__enter__()
    
    def close(self):
        return self.__exit__(None, None, None)


class ZipFileDataset(torch.utils.data.Dataset):
    def __init__(self, zip_files, k):

        self.k = k        
        self.zip_files = zip_files

        # get all the namelists and their lengths
        self.zip_namelists = []
        self.zip_namelists_len = []
        for zip_file in self.zip_files:
            zip_file_namelist = zip_file.namelist()
            self.zip_namelists.append(zip_file_namelist)
            self.zip_namelists_len.append(len(zip_file_namelist))
        
    def __len__(self):
        return sum(self.zip_namelists_len)
    
    def __getitem__(self, idx):
        
        # find the source zip file
        zip_file_idx = 0
        while idx >= self.zip_namelists_len[zip_file_idx]:
            idx -= self.zip_namelists_len[zip_file_idx]
            zip_file_idx += 1
            
        # get the filename
        file_name = self.zip_namelists[zip_file_idx][idx]
        
        # read the operators
        frames, verts, faces, mass, L, evals, evecs, gradX, gradY = zip_read_operators(
            self.zip_files[zip_file_idx], file_name, self.k)
        
        # construct the output payload
        item = {
            'verts': verts,
            'faces': faces,
            'evals': evals[:self.k].unsqueeze(0),
            'evecs': evecs[:, :self.k],
            'evecs_trans': evecs[:, :self.k].T * mass[None],
            'mass': mass,
            'L': L,
            'gradX': gradX,
            'gradY': gradY,
            'corr': torch.arange(len(verts)),
        }
        return item



if __name__ == '__main__':
    
    from my_code.datasets.surreal_dataset_3dc import TemplateSurrealDataset3DC
    
    idx_start = 10000
    idx_end = 20000

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
            
    
    