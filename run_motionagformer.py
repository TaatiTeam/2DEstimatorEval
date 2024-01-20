import numpy as np
from tqdm import tqdm
from common.arguments import parse_args
import torch
import wandb

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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


def evaluate(model_pos, test_loader, e1):
    model_pos.eval()
    with torch.no_grad():
        for x, y in tqdm(test_loader):
            batch_size = x.shape[0]
            x, y = x.cuda(), y.cuda()
            x_flipped = flip_data(x)

            pred_1 = model_pos(x)
            pred_flipped = model_pos(x_flipped)
            pred_2 = flip_data(pred_flipped)
            pred = (pred_1 + pred_2) / 2

            pred[..., 0, :] = 0  # Position of sacrum to zero

            error = mpjpe(pred, y)
            e1.update(error.item() * 1000, batch_size)

    print('Protocol #1 Error (MPJPE):', e1.avg, 'mm')


def train_one_epoch(model_pos, train_loader, optimizer, l_mpjpe, l_scale, l_velocity, l_total, args):
    model_pos.train()
    for x, y in tqdm(train_loader):
        batch_size = x.shape[0]
        x, y = x.cuda(), y.cuda()

        pred = model_pos(x)

        optimizer.zero_grad()

        loss_3d_mpjpe = mpjpe(pred, y)
        loss_3d_scale = n_mpjpe(pred, y)
        loss_3d_velocity = loss_velocity(pred, y)
        loss_total = loss_3d_mpjpe + \
                     args.lambda_scale * loss_3d_scale + \
                     args.lambda_velocity * loss_3d_velocity
        loss_total.backward()
        optimizer.step()

        l_mpjpe.update(loss_3d_mpjpe.item() * 1000, batch_size)
        l_scale.update(loss_3d_scale.item() * 1000, batch_size)
        l_velocity.update(loss_3d_velocity.item() * 1000, batch_size)
        l_total.update(loss_total.item() * 1000, batch_size)
        

def main():
    args = parse_args()
    create_checkpoint_dir_if_not_exists(args.checkpoint)

    # Data 3D
    dataset_path = f'data/data_3d_{args.dataset}.npz'
    dataset_3d = Human36mDataset(dataset_path)
    preprocess_3d_data(dataset_3d)
    joints_left, joints_right = list(dataset_3d.skeleton().joints_left()), list(dataset_3d.skeleton().joints_right())

    # Data 2D
    keypoint_names = [args.keypoints] if args.keypoints != 'concatenate' else ['vitpose', 'pct', 'moganet']
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

    receptive_field = args.number_of_frames
    print(f'[INFO] Receptive field: {receptive_field} frames')

    train_dataset = H36MTorchDataset('train', keypoints_2d, dataset_3d, receptive_field, stride_ratio=3)
    test_dataset = H36MTorchDataset('test', keypoints_2d, dataset_3d, receptive_field, stride_ratio=1)
    print(f'[INFO] Training on {len(train_dataset)} sequences ({len(train_dataset) * receptive_field} frames)')
    print(f'[INFO] Testing on {len(test_dataset)} sequences ({len(test_dataset) * receptive_field} frames)')

    common_loader_params = {
        'batch_size': args.batch_size,
        'num_workers': 15,
        'pin_memory': True,
        'prefetch_factor': 5,
        'persistent_workers': True
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **common_loader_params)
    test_loader = DataLoader(test_dataset, shuffle=False, **common_loader_params)

    model_pos = MotionAGFormer(n_layers=16, dim_in=len(keypoint_names) * 2,
                              dim_feat=128, dim_rep=512, n_frames=receptive_field, neighbour_num=2,
                              use_adaptive_merging=args.adaptive_merging)
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
        e1 = AverageMeter()
        evaluate(model_pos, test_loader, e1)
    else:
        # Learning
        lr = args.learning_rate
        optimizer = optim.AdamW(model_pos.parameters(), lr=lr, weight_decay=0.01)
        lr_decay = args.lr_decay

        start_epoch = 0
        min_loss = float('inf')

        chk_filename = os.path.join(args.checkpoint, args.resume)
        if args.resume and os.path.exists(chk_filename):
            print(f'[INFO] Loading checkpoint from {chk_filename}')
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
            model_pos.load_state_dict(checkpoint['model_pos'], strict=False)

            start_epoch = checkpoint['epoch']
            if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                print('[WARNING] This checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')

            lr = checkpoint['lr']
            min_loss = checkpoint['min_loss']
            wandb_id = checkpoint['wandb_id']

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
            wandb_id = wandb.run.id
        
        for epoch in range(start_epoch, args.epochs):
            loss_mpjpe, loss_scale, loss_velocity, loss_train = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
            train_one_epoch(model_pos, train_loader, optimizer, loss_mpjpe, loss_scale, loss_velocity,
                            loss_train, args)
            e1 = AverageMeter()
            evaluate(model_pos, test_loader, e1)
            print(f'[{epoch + 1}] lr {lr} 3d_train {loss_train.avg} 3d_valid {e1.avg}')
            wandb.log({
                'lr': lr,
                'train/loss': loss_train.avg,
                'train/mpjpe': loss_mpjpe.avg,
                'train/scale': loss_scale.avg,
                'train/velocity': loss_velocity.avg,
                'valid/e1': e1.avg,
            }, step=epoch + 1)

            # Decay learning rate exponentially
            lr *= lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay

            #### save best checkpoint
            best_chk_path = os.path.join(args.checkpoint, 'best_epoch.bin')
            if e1.avg < min_loss:
                min_loss = e1.avg
                print("save best checkpoint", flush=True)
                torch.save({
                    'epoch': epoch + 1,
                    'lr': lr,
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