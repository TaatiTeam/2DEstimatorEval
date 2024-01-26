# Adapted from https://github.com/facebookresearch/VideoPose3D/blob/main/data/prepare_data_h36m.py

import argparse
import os
import numpy as np
import cdflib
from glob import glob

import sys
sys.path.append('../')
from data.h36m_dataset import Human36mDataset
from common.camera import world_to_camera, project_to_2d, image_coordinates
from common.utils import wrap

subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cdf-dir', required=True, type=str, metavar='PATH')
    parser.add_argument('--output-filename-3d', default='data_3d_h36m', type=str)
    parser.add_argument('--output-filename-2d', default='data_2d_h36m_gt', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
        
    print('Converting original Human3.6M dataset from', args.cdf_dir, '(CDF files)')
    output = {}
        
    for subject in subjects:
        output[subject] = {}
        file_list = glob(args.cdf_dir + '/' + subject + '/MyPoseFeatures/D3_Positions/*.cdf')
        assert len(file_list) == 30, "Expected 30 files for subject " + subject + ", got " + str(len(file_list))
        for f in file_list:
            action = os.path.splitext(os.path.basename(f))[0]
            
            if subject == 'S11' and action == 'Directions':
                continue # Discard corrupted video
                
            # Use consistent naming convention
            canonical_name = action.replace('TakingPhoto', 'Photo') \
                                    .replace('WalkingDog', 'WalkDog')
            
            hf = cdflib.CDF(f)
            positions = hf['Pose'].reshape(-1, 32, 3)
            positions /= 1000 # Meters instead of millimeters
            output[subject][canonical_name] = positions.astype('float32')
    
    print('Saving...')
    np.savez_compressed(args.output_filename_3d, positions_3d=output)
    
    print('Done.')
            
    # Create 2D pose file
    print('')
    print('Computing ground-truth 2D poses...')
    dataset = Human36mDataset(f"{args.output_filename_3d}.npz")
    output_2d_poses = {}
    for subject in dataset.subjects():
        output_2d_poses[subject] = {}
        for action in dataset[subject].keys():
            anim = dataset[subject][action]
            
            positions_2d = []
            for cam in anim['cameras']:
                pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                pos_2d = wrap(project_to_2d, pos_3d, cam['intrinsic'], unsqueeze=True)
                pos_2d_pixel_space = image_coordinates(pos_2d, w=cam['res_w'], h=cam['res_h'])
                positions_2d.append(pos_2d_pixel_space.astype('float32'))
            output_2d_poses[subject][action] = positions_2d
            
    print('Saving...')
    metadata = {
        'num_joints': dataset.skeleton().num_joints(),
        'keypoints_symmetry': [dataset.skeleton().joints_left(), dataset.skeleton().joints_right()]
    }
    np.savez_compressed(args.output_filename_2d, positions_2d=output_2d_poses, metadata=metadata)
    
    print('Done.')