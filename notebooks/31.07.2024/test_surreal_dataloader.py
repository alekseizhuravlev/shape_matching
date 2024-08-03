from my_code.datasets.surreal_dataset_3dc import TemplateSurrealDataset3DC
import torch
from tqdm import tqdm
import time

if __name__ == '__main__':

    dataset_3dc = TemplateSurrealDataset3DC(
        # shape_path=f'/home/s94zalek_hpc/3D-CODED/data/datas_surreal_train.pth',
        shape_path='/lustre/mlnvme/data/s94zalek_hpc-shape_matching/mmap_datas_surreal_train.pth',
        num_evecs=32,
        use_cuda=False,
        cache_lb_dir=None,
        return_evecs=True,
        mmap=True
    )    

    dataloader_3dc = torch.utils.data.DataLoader(dataset_3dc, batch_size=1, shuffle=False,
                                                num_workers=16, persistent_workers=True)
    
    with tqdm(range(len(dataset_3dc))) as iterator:
        for i, data in enumerate(dataloader_3dc):
            iterator.update(1)
            time.sleep(0.5)