import numpy as np
from common.arguments import parse_args
import torch
import matplotlib.pyplot as plt

import torch.nn as nn
import os

from einops import rearrange
from copy import deepcopy

from common.camera import *
from common.model_cross import *

from common.loss import *
from common.utils import *
from common.data_utils import *
from common.model_utils import *
from common.h36m_dataset import Human36mDataset
from common.torch_dataset import H36MTorchDataset
from model.motionagformer import MotionAGFormer

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

connections = [
    (10, 9),
    (9, 8),
    (8, 11),
    (8, 14),
    (14, 15),
    (15, 16),
    (11, 12),
    (12, 13),
    (8, 7),
    (7, 0),
    (0, 4),
    (0, 1),
    (1, 2),
    (2, 3),
    (4, 5),
    (5, 6)
]
L, R = 'b', 'r'
joint_colors = [L, R, R, R, L, L, L, L, L, L, L, L, L, L, R, R, R]
connection_colors = [L, L, L, R, R, R, L, L, L, L, L, R, R, R, L, L]


def compare(datasets, models, seq_number, frame=100):
    title_mapper = {
        'vitpose': 'ViTPose',
        'pct': 'PCT',
        'cpn_ft_h36m_dbb': 'CPN',
        'moganet': 'MogaNet',
        'transpose': 'TransPose',
        'merge_average': 'Average Merging',
        'merge_manual': 'WTA Merging',
        'merge_weighted_average': 'Weighted Average Merging',
        'concatenate': 'Concatenate Merging'
    }
    orders = ['transpose', 'moganet', 'pct', 'vitpose', 'cpn_ft_h36m_dbb', 'merge_average', 'merge_weighted_average',
              'merge_manual', 'concatenate']
    with torch.no_grad():
        outs = {}
        gts = {}
        for key in orders:
            models[key].eval()
            
            x, gt = datasets[key][seq_number]
            gt = gt.numpy()
            x = x.unsqueeze(0).cuda()

            x_flipped = flip_data(x)
            pred_1 = models[key](x)
            pred_flipped = models[key](x_flipped)
            pred_2 = flip_data(pred_flipped)
            pred = (pred_1 + pred_2) / 2
            pred = pred.cpu().numpy()

            outs[key] = pred[0, frame]
            gts[key] = gt[frame]

    min_values, max_values = None, None
    for key in orders:
        gt, out = gts[key], outs[key]
        min_gt = np.min(gt, axis=0)
        min_out = np.min(out, axis=0)
        max_gt = np.max(gt, axis=0)
        max_out = np.max(out, axis=0)
        if min_values is None:
            min_values = np.min([min_gt, min_out], axis=0)
        else:
            min_values = np.min([min_gt, min_out, min_values], axis=0)
        if max_values is None:
            max_values = np.min([max_gt, max_out], axis=0)
        else:
            max_values = np.min([max_gt, max_out, max_values], axis=0)
    
    axis_size = max_values - min_values
    aspect_ratio = np.array(axis_size / np.min(axis_size), dtype=int)
    
    fig = plt.figure(figsize=(15, 8))
    axes = [fig.add_subplot(251, projection='3d'), fig.add_subplot(252, projection='3d'),
            fig.add_subplot(253, projection='3d'), fig.add_subplot(254, projection='3d'),
            fig.add_subplot(255, projection='3d'), fig.add_subplot(256, projection='3d'),
            fig.add_subplot(257, projection='3d'), fig.add_subplot(258, projection='3d'),
            fig.add_subplot(259, projection='3d')]
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    for idx, key in enumerate(orders):
        ax = axes[idx]
        ax.set_xlim3d([min_values[0], max_values[0]])
        ax.set_ylim3d([min_values[1], max_values[1]])
        ax.set_zlim3d([min_values[2], max_values[2]])
        ax.set_box_aspect(aspect_ratio)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.view_init(120, 9, -90)

        out, gt = outs[key], gts[key]

        x, y, z = out[:, 0], out[:, 1], out[:, 2]

        for j, connection in enumerate(connections):
            start = out[connection[0], :]
            end = out[connection[1], :]
            xs = [start[0], end[0]]
            ys = [start[1], end[1]]
            zs = [start[2], end[2]]
            ax.plot(xs, ys, zs, c=connection_colors[j])

            start_gt = gt[connection[0], :]
            end_gt = gt[connection[1], :]
            xs = [start_gt[0], end_gt[0]]
            ys = [start_gt[1], end_gt[1]]
            zs = [start_gt[2], end_gt[2]]
            ax.plot(xs, ys, zs, color='black', linewidth=3, alpha=0.2)

            ax.scatter(x, y, z, c=joint_colors)
            ax.set_title(title_mapper[key])

    plt.savefig(f'comparison_seq{seq_number}_frame{frame}', dpi=300)

def main():
    args = parse_args()

    # Data 3D
    dataset_path = f'data/data_3d_{args.dataset}.npz'
    dataset_3d = Human36mDataset(dataset_path)
    preprocess_3d_data(dataset_3d)
    joints_left, joints_right = list(dataset_3d.skeleton().joints_left()), list(dataset_3d.skeleton().joints_right())

    data_2d = {
        'vitpose': None,
        'pct': None,
        'cpn_ft_h36m_dbb': None,
        'moganet': None,
        'transpose': None,
        'merge_average': None,
        'merge_manual': None,
        'merge_weighted_average': None,
        'concatenate': None
    }
    # Data 2D
    for key in data_2d:
        keypoint_names = [key] if key != 'concatenate' else ['vitpose', 'pct', 'moganet']
        keypoints_2d = None
        for keypoint_name in keypoint_names:
            data_2d_path = f'data/data_2d_{args.dataset}_{keypoint_name}.npz'
            keypoints_2d_new, _, _, _ = load_2d_data(data_2d_path)
            verify_2d_3d_matching(keypoints_2d_new, dataset_3d)
            normalize_2d_data(keypoints_2d_new, dataset_3d)
            if keypoints_2d is None:
                keypoints_2d = keypoints_2d_new
            else:
                concatenate_2d_data(keypoints_2d, keypoints_2d_new)
        data_2d[key] = keypoints_2d

    receptive_field = 243
    dataset = {}
    for key in data_2d:
        dataset[key] = H36MTorchDataset('test', data_2d[key], dataset_3d, receptive_field, stride_ratio=1)


    models = {
        'vitpose': None,
        'pct': None,
        'cpn_ft_h36m_dbb': None,
        'moganet': None,
        'transpose': None,
        'merge_average': None,
        'merge_manual': None,
        'merge_weighted_average': None,
        'concatenate': None
    }
    for key in models:
        models[key] = MotionAGFormer(n_layers=16, dim_in=2 if key != 'concatenate' else 6,
                              dim_feat=128, dim_rep=512, n_frames=receptive_field, neighbour_num=2)
        models[key] = nn.DataParallel(models[key])
        models[key] = models[key].cuda()
        
    checkpoints = {
        'vitpose': 'MotionAGFormer-ViTPose.bin',
        'pct': 'MotionAGFormer-PCT.bin',
        'cpn_ft_h36m_dbb': 'MotionAGFormer-CPN.bin',
        'moganet': 'MotionAGFormer-MogaNet.bin',
        'transpose': 'MotionAGFormer-TransPose.bin',
        'merge_average': 'MotionAGFormer-merge-average.bin',
        'merge_manual': 'MotionAGFormer-merge-WTA.bin',
        'merge_weighted_average': 'MotionAGFormer-merge-weighted-average.bin',
        'concatenate': 'MotionAGFormer-merge-concatenate.bin'
    }

    for key in checkpoints:
        chk_filename = os.path.join(args.checkpoint, checkpoints[key])
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        models[key].load_state_dict(checkpoint['model_pos'], strict=True)
    compare(dataset, models, seq_number=400, frame=70)



if __name__ == "__main__":
    main()