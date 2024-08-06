import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networks.diffusion_network as diffusion_network
import os
import utils.geometry_util as geometry_util
import utils.shape_util as shape_util
from tqdm import tqdm
import my_code.datasets.shape_dataset as shape_dataset



def predict_sign_change(net, verts, faces, evecs_flip, evecs_cond, input_type, **kwargs):
    
    # normalize the evecs
    evecs_flip = torch.nn.functional.normalize(evecs_flip, p=2, dim=1)
    
    if evecs_cond is not None:
        evecs_cond = torch.nn.functional.normalize(evecs_cond, p=2, dim=1)
        evecs_input = torch.cat([evecs_flip, evecs_cond], dim=-1)
        
    elif input_type == 'wks':
        evecs_input = None
    elif input_type == 'evecs':
        evecs_input = evecs_flip
    else:
        raise ValueError(f'Unknown input type {input_type}')
        
    # process the flipped evecs
    support_vector_flip = net(
        verts=verts,
        faces=faces,
        feats=evecs_input,
        **kwargs
    ) # [1 x 6890 x 1]

    # normalize the support vector
    support_vector_norm = torch.nn.functional.normalize(support_vector_flip, p=2, dim=1)
    
    if support_vector_norm.shape[-1] != evecs_flip.shape[-1]:
        # copy each element to match the number of evecs
        assert evecs_flip.shape[-1] % support_vector_norm.shape[-1] == 0
        
        repeat_factor = evecs_flip.shape[-1] // support_vector_norm.shape[-1]
        
        support_vector_norm_repeated = torch.repeat_interleave(
            support_vector_norm, repeat_factor, dim=-1)
    else:
        support_vector_norm_repeated = support_vector_norm
        
    
    # multiply the support vector by the flipped evecs [1 x 6890 x 4].T @ [1 x 6890 x 4]
    product_with_support = support_vector_norm_repeated.transpose(1, 2) @ evecs_flip

    if product_with_support.shape[1] == product_with_support.shape[2]:
        # take only diagonal elements
        sign_flip_predicted = torch.diagonal(product_with_support, dim1=1, dim2=2)
        
    # get the sign of the support vector
    # sign_flip_predicted = product_with_support
 
    return sign_flip_predicted, support_vector_norm_repeated, product_with_support


def load_cached_shapes(save_folder, lapl_type):

    # prepare the folders
    mesh_folder = f'{save_folder}/off'
    diff_folder = f'{save_folder}/diffusion'


    # get all meshes in the folder
    mesh_files = sorted([f for f in os.listdir(mesh_folder) if f.endswith('.off')])

    shapes_list = []
    for file in tqdm(mesh_files):

        verts, faces = shape_util.read_shape(os.path.join(mesh_folder, file))
        verts = torch.tensor(verts, dtype=torch.float32)
        faces = torch.tensor(faces, dtype=torch.int32)

        if lapl_type == 'pcl':
            _, _, _, _, evecs, _, _ = geometry_util.get_operators(verts, None,
                                                    k=128,
                                                    cache_dir=diff_folder)
        else:
            _, _, _, _, evecs, _, _ = geometry_util.get_operators(verts, faces,
                                                    k=128,
                                                    cache_dir=diff_folder)
        shapes_list.append({
            'verts': verts,
            'faces': faces,
            'evecs': evecs,
        })
        
    return shapes_list, diff_folder


if __name__ == '__main__':
    
    start_dim = 0

    feature_dim = 32
    evecs_per_support = 4
    n_block = 6
    
    input_type = 'wks'
    lapl_type = 'mesh'

    train_folder = 'SURREAL_train_rot_180_180_180_normal_True_noise_0.0_-0.05_0.05_lapl_mesh_scale_0.9_1.1'
    test_folder = 'FAUST_r'
    
    chkpt_name = f'sign_double_start_{start_dim}_feat_{feature_dim}_{n_block}block_factor{evecs_per_support}_dataset_{train_folder}_{input_type}'

    
    
    experiment_dir = f'/home/s94zalek_hpc/shape_matching/my_code/experiments/{chkpt_name}'
    os.makedirs(experiment_dir)
    
    

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = diffusion_network.DiffusionNet(
        in_channels=feature_dim,
        out_channels=feature_dim // evecs_per_support,
        cache_dir=None,
        input_type='wks',
        k_eig=128,
        n_block=n_block,
        ).to(device)

    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=1, end_factor=0.1, 
        total_iters=50000)
    
    
    train_shapes, train_diff_folder = load_cached_shapes(
        f'/home/s94zalek_hpc/shape_matching/data_sign_training/train/{train_folder}',
        lapl_type=lapl_type
    )
    # test_shapes, test_diff_folder = load_cached_shapes(
    #     f'/home/s94zalek_hpc/shape_matching/data_sign_training/test/{test_folder}',
    #     lapl_type=lapl_type
    # )
    
    # train_diff_folder = f'/home/s94zalek_hpc/shape_matching/data_sign_training/train/{train_folder}/diffusion'
    # train_dataset = shape_dataset.SingleShapeDataset(
    #     data_root = f'/home/s94zalek_hpc/shape_matching/data_sign_training/train/{train_folder}',
    #     centering = 'bbox',
    #     num_evecs=128,
    #     lb_cache_dir=train_diff_folder,
    # ) 
    
    # train_shapes = []
    # for i in tqdm(range(len(train_dataset))):
    #     data_i = train_dataset[i]
    #     train_shapes.append({
    #         'verts': data_i['verts'],
    #         'faces': data_i['faces'],
    #         'evecs': data_i['evecs'],
    #         })
    
    test_diff_folder = f'/home/s94zalek_hpc/shape_matching/data_sign_training/test/{test_folder}/diffusion'
    test_dataset = shape_dataset.SingleShapeDataset(
        data_root = f'/home/s94zalek_hpc/shape_matching/data_sign_training/test/{test_folder}',
        centering = 'bbox',
        num_evecs=128,
        lb_cache_dir=test_diff_folder,
    )     
    
    test_shapes = []
    for i in tqdm(range(len(test_dataset))):
        data_i = test_dataset[i]
        test_shapes.append({
            'verts': data_i['verts'],
            'faces': data_i['faces'],
            'evecs': data_i['evecs'],
            })
    
    
    loss_fn = torch.nn.MSELoss()
    losses = torch.tensor([])
    train_iterator = tqdm(range(40000))
       
    net.cache_dir = train_diff_folder      
            
    curr_iter = 0
    for epoch in range(len(train_iterator) // len(train_shapes)):
        
        # train_shapes_shuffled = train_shapes.copy()
        np.random.shuffle(train_shapes)
        
        
        for curr_idx in range(len(train_shapes)):     

            ##############################################
            # Select a shape
            ##############################################
            # curr_idx = np.random.randint(0, len(train_shapes))
        
            train_shape = train_shapes[curr_idx]

            verts = train_shape['verts'].unsqueeze(0).to(device)
            
            if lapl_type == 'pcl':
                faces = None
            else:
                faces = train_shape['faces'].unsqueeze(0).to(device)    

            evecs_orig = train_shape['evecs'][:, start_dim:start_dim+feature_dim].unsqueeze(0).to(device)


            ##############################################
            # Set the signs on shape 0
            ##############################################

            # create a random combilation of +1 and -1, length = feature_dim
            sign_gt_0 = torch.randint(0, 2, (feature_dim,)).float().to(device)
            
            sign_gt_0[sign_gt_0 == 0] = -1
            sign_gt_0 = sign_gt_0.float().unsqueeze(0)

            # multiply evecs [6890 x 16] by sign_flip [16]
            evecs_flip_0 = evecs_orig * sign_gt_0
            
            # predict the sign change
            sign_pred_0 = predict_sign_change(net, verts, faces, evecs_flip_0, 
                                                    evecs_cond=None, input_type=input_type)[0]
            
            ##############################################
            # Set the signs on shape 1
            ##############################################
            
            # create a random combilation of +1 and -1, length = feature_dim
            sign_gt_1 = torch.randint(0, 2, (feature_dim,)).float().to(device)
            
            sign_gt_1[sign_gt_1 == 0] = -1
            sign_gt_1 = sign_gt_1.float().unsqueeze(0)
            
            # multiply evecs [6890 x 16] by sign_flip [16]
            evecs_flip_1 = evecs_orig * sign_gt_1
            
            # predict the sign change
            sign_pred_1 = predict_sign_change(net, verts, faces, evecs_flip_1, 
                                                    evecs_cond=None, input_type=input_type)[0]
            
            ##############################################
            # Calculate the loss
            ##############################################
            
            # calculate the ground truth sign difference
            sign_diff_gt = sign_gt_1 * sign_gt_0
            
            # calculate the sign difference between predicted evecs
            sign_diff_pred = sign_pred_1 * sign_pred_0
            
            # calculate the loss
            loss = loss_fn(
                sign_diff_pred.reshape(sign_diff_pred.shape[0], -1),
                sign_diff_gt.reshape(sign_diff_gt.shape[0], -1)
                )

            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()
            
            losses = torch.cat([losses, torch.tensor([loss.item()])])
            
            # print mean of last 10 losses
            train_iterator.set_description(f'loss={torch.mean(losses[-10:]):.3f}')
            
            # plot the losses every 1000 iterations
            if curr_iter > 0 and curr_iter % (len(train_iterator) // 10) == 0:
                pd.Series(losses.numpy()).rolling(10).mean().plot()
                plt.yscale('log')
                # plt.show()
                
                plt.savefig(f'{experiment_dir}/losses_{curr_iter}.png')
                plt.close()
                
                torch.save(
                    net.state_dict(),
                    f'{experiment_dir}/{curr_iter}.pth'
                    )
                
            curr_iter += 1
            train_iterator.update(1)
            
            
    # save model checkpoint
    torch.save(
        net.state_dict(),
        f'{experiment_dir}/{curr_iter}.pth')
    
    
    ###################################################
    # Test
    ###################################################

    tqdm._instances.clear()

    shapes_to_test = test_shapes
    net.cache_dir = test_diff_folder
           
                
    test_iterator = tqdm(range(1001))
    incorrect_signs_list = torch.tensor([])
    curr_iter = 0

    for epoch in range(len(test_iterator) // len(shapes_to_test)):        
        for curr_idx in range(len(shapes_to_test)):     


            ##############################################
            # Select a shape
            ##############################################
            
            test_shape = shapes_to_test[curr_idx]    
            
            verts = test_shape['verts'].unsqueeze(0).to(device)
            
            if lapl_type == 'pcl':
                faces = None
            else:
                faces = test_shape['faces'].unsqueeze(0).to(device)    
            
            evecs_orig = test_shape['evecs'][:, start_dim:start_dim+feature_dim].unsqueeze(0).to(device)

            ##############################################
            # Set the signs on shape 0
            ##############################################

            # create a random combilation of +1 and -1, length = feature_dim
            sign_gt_0 = torch.randint(0, 2, (feature_dim,)).float().to(device)
            
            sign_gt_0[sign_gt_0 == 0] = -1
            sign_gt_0 = sign_gt_0.float().unsqueeze(0)

            # multiply evecs [6890 x 16] by sign_flip [16]
            evecs_flip_0 = evecs_orig * sign_gt_0
            
            # predict the sign change
            with torch.no_grad():
                sign_pred_0, supp_vec_0, _ = predict_sign_change(net, verts, faces, evecs_flip_0, 
                                                    evecs_cond=None, input_type=input_type)
            
            ##############################################
            # Set the signs on shape 1
            ##############################################
            
            # create a random combilation of +1 and -1, length = feature_dim
            sign_gt_1 = torch.randint(0, 2, (feature_dim,)).float().to(device)
            
            sign_gt_1[sign_gt_1 == 0] = -1
            sign_gt_1 = sign_gt_1.float().unsqueeze(0)
            
            # multiply evecs [6890 x 16] by sign_flip [16]
            evecs_flip_1 = evecs_orig * sign_gt_1
            
            # predict the sign change
            with torch.no_grad():
                sign_pred_1, supp_vec_1, _ = predict_sign_change(net, verts, faces, evecs_flip_1, 
                                                    evecs_cond=None, input_type=input_type)
            
            ##############################################
            # Calculate the loss
            ##############################################
            
            # calculate the ground truth sign difference
            sign_diff_gt = sign_gt_1 * sign_gt_0
            
            # calculate the sign difference between predicted evecs
            sign_diff_pred = sign_pred_1 * sign_pred_0
            
            sign_correct = sign_diff_pred.sign() * sign_diff_gt.sign() 
            
            
            # count the number of incorrect signs
            count_incorrect_signs = (sign_correct < 0).int().sum()
                
            # incorrect_signs_list.append(count_incorrect_signs)
            incorrect_signs_list = torch.cat([incorrect_signs_list, torch.tensor([count_incorrect_signs])])
            
            
            test_iterator.set_description(f'Mean incorrect signs {incorrect_signs_list.float().mean():.2f} / {feature_dim}')
            test_iterator.update(1)
            # if count_incorrect_signs > 7:
            #     raise ValueError('Too many incorrect signs')
        
    output_str = f"Results for {len(incorrect_signs_list)} test shapes\n"
    output_str += f"Incorrect signs per shape: {incorrect_signs_list.float().mean():.2f} / {feature_dim}\n"
    output_str += f"Max incorrect signs {incorrect_signs_list.max()}"
    
    print(output_str)
    with open(f'{experiment_dir}/results.txt', 'w') as f:
        f.write(output_str)

    # print()
    # print('Shape idx', curr_idx)
    # # print('GT', sign_diff_gt)
    # # print('PRED', sign_diff_pred)
    # # print('Correct', sign_correct)
    # print(f'Incorrect signs {torch.sum(sign_correct != 1)} / {feature_dim}')
    # print(incorrect_signs_list)


    # plt.plot(support_vector_norm.squeeze().detach().cpu().numpy(), '.', alpha=0.1)
    # plt.ylim(-0.1, 0.1)
    # # plt.yscale('log')
    # plt.show()
