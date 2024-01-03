import numpy as np
from tqdm import tqdm
from common.arguments import parse_args
import torch
import wandb

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys

from einops import rearrange
from copy import deepcopy

from common.camera import *
from common.model_cross import *

from common.loss import *
from common.utils import *
from common.data_utils import *
from common.model_utils import *
from common.h36m_dataset import Human36mDataset
from model.motionagformer import MotionAGFormer

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def eval_data_prepare(receptive_field, inputs_2d, inputs_3d):
    assert inputs_2d.shape[:-1] == inputs_3d.shape[:-1], "2d and 3d inputs shape must be same! "+str(inputs_2d.shape)+str(inputs_3d.shape)
    inputs_2d_p = torch.squeeze(inputs_2d)
    inputs_3d_p = torch.squeeze(inputs_3d)

    if inputs_2d_p.shape[0] / receptive_field > inputs_2d_p.shape[0] // receptive_field: 
        out_num = inputs_2d_p.shape[0] // receptive_field+1
    elif inputs_2d_p.shape[0] / receptive_field == inputs_2d_p.shape[0] // receptive_field:
        out_num = inputs_2d_p.shape[0] // receptive_field

    eval_input_2d = torch.empty(out_num, receptive_field, inputs_2d_p.shape[1], inputs_2d_p.shape[2])
    eval_input_3d = torch.empty(out_num, receptive_field, inputs_3d_p.shape[1], inputs_3d_p.shape[2])

    for i in range(out_num-1):
        eval_input_2d[i,:,:,:] = inputs_2d_p[i*receptive_field:i*receptive_field+receptive_field,:,:]
        eval_input_3d[i,:,:,:] = inputs_3d_p[i*receptive_field:i*receptive_field+receptive_field,:,:]
    if inputs_2d_p.shape[0] < receptive_field:
        from torch.nn import functional as F
        pad_right = receptive_field-inputs_2d_p.shape[0]
        inputs_2d_p = rearrange(inputs_2d_p, 'b f c -> f c b')
        inputs_2d_p = F.pad(inputs_2d_p, (0,pad_right), mode='replicate')
        # inputs_2d_p = np.pad(inputs_2d_p, ((0, receptive_field-inputs_2d_p.shape[0]), (0, 0), (0, 0)), 'edge')
        inputs_2d_p = rearrange(inputs_2d_p, 'f c b -> b f c')
    if inputs_3d_p.shape[0] < receptive_field:
        pad_right = receptive_field-inputs_3d_p.shape[0]
        inputs_3d_p = rearrange(inputs_3d_p, 'b f c -> f c b')
        inputs_3d_p = F.pad(inputs_3d_p, (0,pad_right), mode='replicate')
        inputs_3d_p = rearrange(inputs_3d_p, 'f c b -> b f c')
    eval_input_2d[-1,:,:,:] = inputs_2d_p[-receptive_field:,:,:]
    eval_input_3d[-1,:,:,:] = inputs_3d_p[-receptive_field:,:,:]

    return eval_input_2d, eval_input_3d


def evaluate(model_pos, test_generator, kps_left, kps_right, receptive_field, joints_left, joints_right):
    epoch_loss_3d_pos = 0
    epoch_loss_3d_pos_procrustes = 0
    epoch_loss_3d_pos_scale = 0
    epoch_loss_3d_vel = 0
    with torch.no_grad():
        model_eval = model_pos
        N = 0
        for _, batch, batch_2d in test_generator.next_epoch():
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            inputs_3d = torch.from_numpy(batch.astype('float32'))

            inputs_2d_flip = inputs_2d.clone()
            inputs_2d_flip [:, :, :, 0] *= -1
            inputs_2d_flip[:, :, kps_left + kps_right,:] = inputs_2d_flip[:, :, kps_right + kps_left,:]

            inputs_2d, inputs_3d = eval_data_prepare(receptive_field, inputs_2d, inputs_3d)
            inputs_2d_flip, _ = eval_data_prepare(receptive_field, inputs_2d_flip, inputs_3d)

            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()
                inputs_2d_flip = inputs_2d_flip.cuda()
                inputs_3d = inputs_3d.cuda()
                
            inputs_3d[:, :, 0] = 0
            
            predicted_3d_pos = model_eval(inputs_2d)
            predicted_3d_pos_flip = model_eval(inputs_2d_flip)
            predicted_3d_pos_flip[:, :, :, 0] *= -1
            predicted_3d_pos_flip[:, :, joints_left + joints_right] = predicted_3d_pos_flip[:, :,
                                                                      joints_right + joints_left]
            for i in range(predicted_3d_pos.shape[0]):
                predicted_3d_pos[i,:,:,:] = (predicted_3d_pos[i,:,:,:] + predicted_3d_pos_flip[i,:,:,:])/2

            error = mpjpe(predicted_3d_pos, inputs_3d)

            epoch_loss_3d_pos_scale += inputs_3d.shape[0]*inputs_3d.shape[1] * n_mpjpe(predicted_3d_pos, inputs_3d).item()

            epoch_loss_3d_pos += inputs_3d.shape[0]*inputs_3d.shape[1] * error.item()
            N += inputs_3d.shape[0] * inputs_3d.shape[1]

            inputs = inputs_3d.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
            predicted_3d_pos = predicted_3d_pos.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])

            epoch_loss_3d_pos_procrustes += inputs_3d.shape[0]*inputs_3d.shape[1] * p_mpjpe(predicted_3d_pos, inputs)

            # Compute velocity error
            epoch_loss_3d_vel += inputs_3d.shape[0]*inputs_3d.shape[1] * mean_velocity_error(predicted_3d_pos, inputs)

    e1 = (epoch_loss_3d_pos / N)*1000
    e2 = (epoch_loss_3d_pos_procrustes / N)*1000
    e3 = (epoch_loss_3d_pos_scale / N)*1000
    ev = (epoch_loss_3d_vel / N)*1000
    print('Protocol #1 Error (MPJPE):', e1, 'mm')
    print('Protocol #2 Error (P-MPJPE):', e2, 'mm')
    print('Protocol #3 Error (N-MPJPE):', e3, 'mm')
    print('Velocity Error (MPJVE):', ev, 'mm')
    print('----------')

    return e1, e2, e3, ev


def train_one_epoch(model_pos, train_generator, optimizer, losses_3d_train):
    epoch_loss_3d_train = 0
    N = 0
    model_pos.train()

    for _, batch_3d, batch_2d in tqdm(train_generator.next_epoch()):
        inputs_3d = torch.from_numpy(batch_3d.astype('float32')) 
        inputs_2d = torch.from_numpy(batch_2d.astype('float32'))

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


def main():
    args = parse_args()
    create_checkpoint_dir_if_not_exists(args.checkpoint)

    # Data 3D
    dataset_path = f'data/data_3d_{args.dataset}.npz'
    dataset_3d = Human36mDataset(dataset_path)
    preprocess_3d_data(dataset_3d)
    joints_left, joints_right = list(dataset_3d.skeleton().joints_left()), list(dataset_3d.skeleton().joints_right())

    # Data 2D
    data_2d_path = f'data/data_2d_{args.dataset}_{args.keypoints}.npz'
    keypoints_2d, kps_left, kps_right, num_joints = load_2d_data(data_2d_path)
    verify_2d_3d_matching(keypoints_2d, dataset_3d)
    normalize_2d_data(keypoints_2d, dataset_3d)

    subjects_train = args.subjects_train.split(',')
    subjects_test = args.subjects_test.split(',')

    action_filter = None if args.actions == '*' else args.actions.split(',')
    if action_filter is not None:
        print('[INFO] Selected actions:', action_filter)

    receptive_field = args.number_of_frames
    print(f'[INFO] Receptive field: {receptive_field} frames')
    pad = (receptive_field -1) // 2 # Padding on each side

    train_generator = init_train_generator(subjects_train, keypoints_2d, dataset_3d,
                                           pad, kps_left, kps_right, joints_left, joints_right,
                                           stride=args.downsample, action_filter=action_filter)
    test_generator = init_test_generator(subjects_test, keypoints_2d, dataset_3d, pad,
                                         kps_left, kps_right, joints_left, joints_right,
                                         stride=args.downsample, action_filter=action_filter)
    print(f'[INFO] Training on {train_generator.num_frames()} frames')
    print(f'[INFO] Testing on {test_generator.num_frames()} frames')

    model_pos = MotionAGFormer(n_layers=16, dim_in=2, dim_feat=128, dim_rep=512, n_frames=receptive_field,
                           neighbour_num=2)
    model_params = count_number_of_parameters(model_pos)
    print('[INFO] Trainable parameter count:', model_params/1000000, 'Million')

    if torch.cuda.is_available():
        model_pos = nn.DataParallel(model_pos)
        model_pos = model_pos.cuda()

    if args.evaluate:
        chk_filename = os.path.join(args.checkpoint, args.evaluate)
        print(f'[INFO] Loading checkpoint from {chk_filename}')
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        model_pos.load_state_dict(checkpoint['model_pos'], strict=False)
        evaluate(model_pos, test_generator, kps_left, kps_right, receptive_field,
                    joints_left, joints_right)
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
                        project='2DEstimatorEvaluation',
                        resume="must",
                        settings=wandb.Settings(start_method='fork'))
        else:
            wandb_id = wandb.util.generate_id()
            wandb.init(id=wandb_id,
                        name=args.wandb_name,
                        project='2DEstimatorEvaluation',
                        settings=wandb.Settings(start_method='fork'))
            wandb.config.update(args)
        
        for epoch in range(start_epoch, args.epochs):
            train_one_epoch(model_pos, train_generator, optimizer, losses_3d_train)
            e1, e2, e3, ev = evaluate(model_pos, test_generator, kps_left, kps_right,
                                      receptive_field, joints_left, joints_right, losses_3d_valid)
            print(f'[{epoch + 1}] lr {lr} 3d_train {losses_3d_train[-1] * 1000} 3d_valid {losses_3d_valid[-1] * 1000}')
            wandb.log({
                'lr': lr,
                'train/loss': losses_3d_train[-1] * 1000,
                'valid/e1': e1,
                'valid/e2': e2,
                'valid/e3': e3,
                'valid/ev': ev
            }, step=epoch + 1)

            # Decay learning rate exponentially
            lr *= lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay

            #### save best checkpoint
            best_chk_path = os.path.join(args.checkpoint, 'best_epoch.bin')
            if e1 < min_loss:
                min_loss = e1
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