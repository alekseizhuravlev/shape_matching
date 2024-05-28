import sys

sys.path.append('/home/s94zalek/shape_matching/my_code/datasets')

import numpy as np
# import smplx
import torch

from tqdm import tqdm


from smpl_webuser.serialization import load_model
import os


def generate_surreal(m, pose, beta):
    """
    This function generation 1 human using a random pose and shape estimation from surreal
    """
    ## Assign gaussian pose
    m.pose[:] = pose
    m.betas[:] = beta
    # m.pose[0:3]=0
        
    point_set = m.r.astype(np.float32)
    
    #normalize
    centroid = np.expand_dims(np.mean(point_set[:,0:3], axis = 0), 0) #Useless because dataset has been normalised already
    point_set[:,0:3] = point_set[:,0:3] - centroid

    return point_set, m.f


def get_random(database, poses, betas):
    beta_id = np.random.randint(np.shape(betas)[0]-1)
    beta = betas[beta_id]
    pose_id = np.random.randint(len(poses)-1)
    pose_ = database[poses[pose_id]]
    pose_id = np.random.randint(np.shape(pose_)[0])
    pose = pose_[pose_id]
    return pose, beta


def generate_shapes(n_body_types_male, n_body_types_female, n_poses_straight, n_poses_bent):
    
    database = np.load("/home/s94zalek/shape_matching/data/SURREAL_full/smpl_data.npz")
    
    # fix numpy random seed
    np.random.seed(120)
    
    # generate random male betas
    random_beta_indices_male = np.random.choice(len(database['maleshapes']), n_body_types_male, replace=False)
    random_betas_male = database['maleshapes'][random_beta_indices_male]
    m_male = load_model("/home/s94zalek/shape_matching/data/SURREAL_full/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl")
    
    # generate random female betas
    random_beta_indices_female = np.random.choice(len(database['femaleshapes']), n_body_types_female, replace=False)
    random_betas_female = database['femaleshapes'][random_beta_indices_female]
    m_female = load_model("/home/s94zalek/shape_matching/data/SURREAL_full/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl")
    
    
    # get pose sequences
    pose_sequences = [i for i in database.keys() if "pose" in i]
    
    # gather all poses from sequences into a single list
    pose_database = []
    for i in range(len(pose_sequences)):
        # length of a pose sequence i
        pose_length = len(database[pose_sequences[i]])
        
        # add all the poses in the sequence to the database
        # i is the sequence index, j is the pose index in the sequence
        pose_database += [(i, j) for j in range(pose_length)]
        

    # sanity check
    num_poses = 0
    for i in pose_sequences:
        num_poses = num_poses + np.shape(database[i])[0]
    assert len(pose_database) == num_poses, f"Number of poses in the database is {len(pose_database)} but should be {num_poses}"    
    
        
    # generate random poses without replacement
    random_pose_indices = np.random.choice(len(pose_database), n_poses_straight + n_poses_bent, replace=False)
    random_poses = []
    
    # generate straight poses
    for i in random_pose_indices[:n_poses_straight]:
        # get the pose sequence using the 1st index
        pose_sequence = pose_sequences[pose_database[i][0]]
        
        # get the pose in the sequence using the 2nd index
        pose = database[pose_sequence][pose_database[i][1]]
        pose[0:3]=0 
        
        random_poses.append(pose)
        
    # generate bent poses
    for i in random_pose_indices[n_poses_straight:]:
        # get the pose sequence using the 1st index
        pose_sequence = pose_sequences[pose_database[i][0]]
        
        # get the pose in the sequence using the 2nd index
        pose = database[pose_sequence][pose_database[i][1]]
        
        # bend the pose
        a = np.random.randn(12)
        pose[1] = 0
        pose[2] = 0
        pose[3] = -1.0 + 0.1*a[0]
        pose[4] = 0 + 0.1*a[1]
        pose[5] = 0 + 0.1*a[2]
        pose[6] = -1.0 + 0.1*a[0]
        pose[7] = 0 + 0.1*a[3]
        pose[8] = 0 + 0.1*a[4]
        pose[9] = 0.9 + 0.1*a[6]
        pose[0] = - (-0.8 + 0.1*a[0] )
        pose[18] = 0.2 + 0.1*a[7]
        pose[43] = 1.5 + 0.1*a[8]
        pose[40] = -1.5 + 0.1*a[9]
        pose[44] = -0.15 
        pose[41] = 0.15
        pose[48:54] = 0 
        
        random_poses.append(pose)
        

    # generate the data
    output = {
        'verts': [],
        'faces': [],
        'poses': [],
        'betas': []
    }
    
    for beta in tqdm(random_betas_male, desc="Generating male shapes"):
        for pose in random_poses:
            verts, faces = generate_surreal(m_male, pose, beta)
            
            output['verts'].append(verts)
            output['faces'].append(faces)
            output['poses'].append(pose)
            output['betas'].append(beta)
            
    for beta in tqdm(random_betas_female, desc="Generating female shapes"):
        for pose in random_poses:
            verts, faces = generate_surreal(m_female, pose, beta)
            
            output['verts'].append(verts)
            output['faces'].append(faces)
            output['poses'].append(pose)
            output['betas'].append(beta)
        
    # stack the arrays
    output['verts'] = torch.tensor(np.stack(output['verts']).astype(np.float32))
    output['faces'] = torch.tensor(np.stack(output['faces']).astype(np.int32))
    output['poses'] = torch.tensor(np.stack(output['poses']).astype(np.float32))
    output['betas'] = torch.tensor(np.stack(output['betas']).astype(np.float32))
        
    return output   


if __name__ == '__main__':

    print(generate_shapes(n_body_types=5, n_poses=5, male=True, bent=False))