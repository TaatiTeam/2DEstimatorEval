# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Modified by Qitao Zhao (qitaozhao@mail.sdu.edu.cn)

import numpy as np
import wandb

from common.arguments import parse_args
import torch

import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm

from common.camera import *
from model.poseformerv2 import *

from common.loss import *
from time import time
from common.utils import *
from common.h36m_dataset import Human36mDataset
from common.data_utils import *
from common.model_utils import count_number_of_parameters


def eval_data_prepare(receptive_field, inputs_2d, inputs_3d):
    inputs_2d_p = torch.squeeze(inputs_2d)
    inputs_3d_p = inputs_3d.permute(1,0,2,3)
    out_num = inputs_2d_p.shape[0] - receptive_field + 1
    eval_input_2d = torch.empty(out_num, receptive_field, inputs_2d_p.shape[1], inputs_2d_p.shape[2])
    for i in range(out_num):
        eval_input_2d[i,:,:,:] = inputs_2d_p[i:i+receptive_field, :, :]
    return eval_input_2d, inputs_3d_p


def train_one_epoch(model_pos, train_generator, optimizer, losses_3d_train):
    epoch_loss_3d_train = 0
    N = 0
    model_pos.train()

    for _, batch_3d, batch_2d in tqdm(train_generator.next_epoch()):
        inputs_3d = torch.from_numpy(batch_3d.astype('float32')) # [512, 1, 17, 3]
        inputs_2d = torch.from_numpy(batch_2d.astype('float32')) # [512, 3, 17, 2]

        if torch.cuda.is_available():
            inputs_3d = inputs_3d.cuda()
            inputs_2d = inputs_2d.cuda()
        inputs_3d[:, :, 0] = 0

        optimizer.zero_grad()

        # Predict 3D poses
        predicted_3d_pos = model_pos(inputs_2d)

        loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
        epoch_loss_3d_train += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()

        N += inputs_3d.shape[0] * inputs_3d.shape[1]

        loss_total = loss_3d_pos

        loss_total.backward()

        optimizer.step()
        del inputs_2d, inputs_3d, loss_3d_pos, predicted_3d_pos
        torch.cuda.empty_cache()

    losses_3d_train.append(epoch_loss_3d_train / N)
    torch.cuda.empty_cache()


def evaluate(model_pos, test_generator, kps_left, kps_right, receptive_field,
             joints_left, joints_right, losses_3d_valid):
    
    with torch.no_grad():
        model_pos.eval()

        epoch_loss_3d_valid = 0
        N = 0
        # Evaluate on test set
        for _, batch, batch_2d in tqdm(test_generator.next_epoch()):
            inputs_3d = torch.from_numpy(batch.astype('float32')) # [1, 2356, 17, 3]
            inputs_2d = torch.from_numpy(batch_2d.astype('float32')) # [1, 2358, 17, 2]

            ##### apply test-time-augmentation (following Videopose3d)
            inputs_2d_flip = inputs_2d.clone()
            inputs_2d_flip[:, :, :, 0] *= -1
            inputs_2d_flip[:, :, kps_left + kps_right, :] = inputs_2d_flip[:, :, kps_right + kps_left, :]

            ##### convert size
            inputs_2d, inputs_3d = eval_data_prepare(receptive_field, inputs_2d, inputs_3d) # [2356, 3, 17, 2] 
            inputs_2d_flip, _ = eval_data_prepare(receptive_field, inputs_2d_flip, inputs_3d)
    
            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()
                inputs_2d_flip = inputs_2d_flip.cuda()
                inputs_3d = inputs_3d.cuda()

            inputs_3d[:, :, 0] = 0

            predicted_3d_pos = model_pos(inputs_2d)
            predicted_3d_pos_flip = model_pos(inputs_2d_flip)
            predicted_3d_pos_flip[:, :, :, 0] *= -1
            predicted_3d_pos_flip[:, :, joints_left + joints_right] = predicted_3d_pos_flip[:, :,
                                                                        joints_right + joints_left]

            predicted_3d_pos = torch.mean(torch.cat((predicted_3d_pos, predicted_3d_pos_flip), dim=1), dim=1,
                                            keepdim=True)

            loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
            torch.cuda.empty_cache()

            epoch_loss_3d_valid += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
            N += inputs_3d.shape[0] * inputs_3d.shape[1]

            del inputs_2d, inputs_2d_flip, inputs_3d, loss_3d_pos, predicted_3d_pos, predicted_3d_pos_flip
            torch.cuda.empty_cache()

        losses_3d_valid.append(epoch_loss_3d_valid / N)


def main():
    args = parse_args()
    create_checkpoint_dir_if_not_exists(args.checkpoint)

    # Data 3D
    dataset_path = f'data/data_3d_h36m.npz'
    dataset_3d = Human36mDataset(dataset_path)
    preprocess_3d_data(dataset_3d)
    joints_left, joints_right = list(dataset_3d.skeleton().joints_left()), list(dataset_3d.skeleton().joints_right())

    # Data 2D
    keypoint_names = [args.keypoints] if args.keypoints != 'adaptive_merge' else ['vitpose', 'pct', 'moganet']
    keypoints_2d = None
    for keypoint_name in keypoint_names:
        data_2d_path = f'data/data_2d_{args.dataset}_{keypoint_name}.npz'
        keypoints_2d_new, kps_left, kps_right, num_joints = load_2d_data(data_2d_path)
        verify_2d_3d_matching(keypoints_2d_new, dataset_3d)
        normalize_2d_data(keypoints_2d_new, dataset_3d)
        if keypoints_2d is None:
            keypoints_2d = keypoints_2d_new
        else:
            concatenate_2d_data(keypoints_2d, keypoints_2d_new)

    subjects_train = 'S1,S5,S6,S7,S8'.split(',')
    subjects_test = 'S9,S11'.split(',')

    receptive_field = args.number_of_frames
    print(f'[INFO] Receptive field: {receptive_field} frames')
    pad = (receptive_field -1) // 2  # Padding on each side

    train_generator = init_train_generator(subjects_train, keypoints_2d, dataset_3d, pad,
                                           kps_left, kps_right, joints_left, joints_right, args)
    test_generator = init_test_generator(subjects_test, keypoints_2d, dataset_3d, pad,
                                         kps_left, kps_right, joints_left, joints_right)
    print(f'[INFO] Training on {train_generator.num_frames()} frames')
    print(f'[INFO] Testing on {test_generator.num_frames()} frames')

    model_pos = PoseTransformerV2(num_frame=receptive_field, num_joints=num_joints, in_chans=len(keypoint_names) * 2,
            num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=0, args=args)

    model_params = count_number_of_parameters(model_pos)
    print(f'[INFO] Trainable parameter count: {model_params}')

    if torch.cuda.is_available():
        model_pos = nn.DataParallel(model_pos)
        model_pos = model_pos.cuda()

    if args.evaluate:
        losses_3d_valid = []
        chk_filename = os.path.join(args.checkpoint, args.evaluate)
        print(f'[INFO] Loading checkpoint from {chk_filename}')
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        model_pos.load_state_dict(checkpoint['model_pos'], strict=False)
        evaluate(model_pos, test_generator, kps_left, kps_right, receptive_field,
                    joints_left, joints_right, losses_3d_valid)
        print(f'MPJPE on Validation data: {losses_3d_valid[-1] * 1000}')
    else:
        # Learning
        lr = args.learning_rate
        optimizer = optim.AdamW(model_pos.parameters(), lr=lr, weight_decay=0.1)
        lr_decay = args.lr_decay
        losses_3d_train = []
        losses_3d_valid = []

        start_epoch = 0
        min_loss = float('inf')

        if args.resume:
            chk_filename = os.path.join(args.checkpoint, args.resume)
            print(f'[INFO] Loading checkpoint from {chk_filename}')
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
            model_pos.load_state_dict(checkpoint['model_pos'], strict=False)

            start_epoch = checkpoint['epoch']
            if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
                train_generator.set_random_state(checkpoint['random_state'])
            else:
                print('[WARNING] This checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')

            lr = checkpoint['lr']
            min_loss = checkpoint['min_loss']
            wandb_id = args.wandb_id if args.wandb_id is not None else checkpoint['wandb_id']

            wandb.init(id=wandb_id,
                        project='2DEstimatorEvaluationt',
                        resume="must",
                        settings=wandb.Settings(start_method='fork'))
        else:
            wandb_id = wandb.util.generate_id()
            wandb.init(id=wandb_id,
                        name=args.wandb_name,
                        project='2DEstimatorEvaluationt',
                        settings=wandb.Settings(start_method='fork'))
            wandb.config.update(args)

        for epoch in range(start_epoch, args.epochs):
            train_one_epoch(model_pos, train_generator, optimizer, losses_3d_train)
            evaluate(model_pos, test_generator, kps_left, kps_right, receptive_field,
                    joints_left, joints_right, losses_3d_valid)
            print(f'[{epoch + 1}] lr {lr} 3d_train {losses_3d_train[-1] * 1000} 3d_valid {losses_3d_valid[-1] * 1000}')
            wandb.log({
                'lr': lr,
                'loss/train': losses_3d_train[-1] * 1000,
                'loss/valid': losses_3d_valid[-1] * 1000
            }, step=epoch + 1)

            # Decay learning rate exponentially
            lr *= lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay

            #### save best checkpoint
            best_chk_path = os.path.join(args.checkpoint, 'best_epoch.bin')
            if losses_3d_valid[-1] * 1000 < min_loss:
                min_loss = losses_3d_valid[-1] * 1000
                print("save best checkpoint", flush=True)
                torch.save({
                    'epoch': epoch + 1,
                    'lr': lr,
                    'random_state': train_generator.random_state(),
                    'optimizer': optimizer.state_dict(),
                    'model_pos': model_pos.state_dict(),
                    'min_loss': min_loss,
                    'wandb_id': wandb_id
                }, best_chk_path)
            ## save last checkpoint
            last_chk_path = os.path.join(args.checkpoint, 'last_epoch.bin')
            print('Saving checkpoint to', last_chk_path, flush=True)

            torch.save({
                'epoch': epoch + 1,
                'lr': lr,
                'random_state': train_generator.random_state(),
                'optimizer': optimizer.state_dict(),
                'model_pos': model_pos.state_dict(),
                'min_loss': min_loss,
                'wandb_id': wandb_id
            }, last_chk_path)
        
        artifact = wandb.Artifact(f'model', type='model')
        artifact.add_file(last_chk_path)
        artifact.add_file(best_chk_path)
        wandb.log_artifact(artifact)



if __name__ == "__main__":
    main()
     