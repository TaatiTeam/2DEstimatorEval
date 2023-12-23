import os
import wandb
import torch
import random
import logging
from tqdm import tqdm
import torch.utils.data
import torch.optim as optim
from common.opt import opts
from common.utils import *
from common.load_data_hm36_tds import Fusion
from common.h36m_dataset import Human36mDataset
from model.motionagformer import MotionAGFormer


opt = opts().parse()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu


def train(opt, actions, train_loader, model, optimizer, epoch):
    return step('train', opt, actions, train_loader, model, optimizer, epoch)


def val(opt, actions, val_loader, model):
    with torch.no_grad():
        return step('test',  opt, actions, val_loader, model)


def step(split, opt, actions, dataLoader, model, optimizer=None, epoch=None):

    if split == 'train':
        model.train()
    else:
        model.eval()

    loss_all = {'loss': AccumLoss()}
    action_error_sum = define_error_list(actions)

    for data in tqdm(dataLoader, 0):
        gt_3D, input_2D, action = data
        [input_2D, gt_3D] = get_variable(split, [input_2D, gt_3D])

        if split =='train':
             output_3D = model(input_2D)
        else:
            input_2D, output_3D = input_augmentation(input_2D, model)

        out_target = gt_3D.clone()
        out_target[:, :, 0] = 0
        output_3D_single = output_3D[:, opt.pad].unsqueeze(1)

        if split == 'train':
            loss = mpjpe_cal(output_3D, out_target)

            N = input_2D.size(0)
            loss_all['loss'].update(loss.detach().cpu().numpy() * N, N)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        elif split == 'test':
            output_3D[:, :, 0, :] = 0
            action_error_sum = test_calculation(output_3D_single, out_target, action, action_error_sum, opt.dataset, subject)
            

    if split == 'train':
        return loss_all['loss'].avg
    elif split == 'test':
        p1, p2 = print_error(opt.dataset, action_error_sum, opt.train)
        return p1, p2


def input_augmentation(input_2D, model):
    joints_left = [4, 5, 6, 11, 12, 13]  
    joints_right = [1, 2, 3, 14, 15, 16]

    input_2D_non_flip = input_2D[:, 0]
    input_2D_flip = input_2D[:, 1]

    output_3D_non_flip = model(input_2D_non_flip)
    output_3D_flip = model(input_2D_flip)

    output_3D_flip[:, :, :, 0] *= -1

    output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :]

    output_3D = (output_3D_non_flip + output_3D_flip) / 2
    input_2D = input_2D_non_flip

    return input_2D, output_3D


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    opt.manualSeed = 42

    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    if opt.train:
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
                            filename=os.path.join(opt.checkpoint, 'train.log'), level=logging.INFO)
            
    root_path = opt.root_path
    dataset_path = os.path.join(root_path, f"data_3d_{opt.dataset}.npz")

    dataset = Human36mDataset(dataset_path, opt)
    actions = define_actions(opt.actions)

    if opt.train:
        train_data = Fusion(opt=opt, train=True, dataset=dataset, root_path=root_path, tds=opt.t_downsample)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batchSize//opt.stride,
                                                       shuffle=True, num_workers=int(opt.workers), pin_memory=True)

    test_data = Fusion(opt=opt, train=False,dataset=dataset, root_path =root_path, tds=opt.t_downsample)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batchSize//opt.stride,
                                                  shuffle=False, num_workers=int(opt.workers), pin_memory=True)

    opt.out_joints = dataset.skeleton().num_joints()

    model = MotionAGFormer(n_layers=16, dim_in=3, dim_feat=128, dim_rep=512, n_frames=8,
                           neighbour_num=2)

    if torch.cuda.is_available():
        model = nn.DataParallel(model)
        model = model.cuda()
    
    all_param = []
    lr = opt.lr
    for i_model in model:
        all_param += list(model[i_model].parameters())
    optimizer = optim.Adam(all_param, lr=opt.lr, amsgrad=True)

    start_epoch = 1
    if opt.resume:
        chk_filename = os.path.join(opt.checkpoint, opt.resume)
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['model'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        lr = checkpoint['lr']
        opt.previous_best_threshold = checkpoint['previous_best_threshold']
        wandb_id = checkpoint['wandb_id']

        wandb.init(id=wandb_id,
                    project='2DEstimatorEvaluation',
                    resume="must",
                    settings=wandb.Settings(start_method='fork'))
    else:
        wandb_id = wandb.util.generate_id()
        wandb.init(id=wandb_id,
                   name=opt.wandb_name,
                   project='2DEstimatorEvaluation',
                   settings=wandb.Settings(start_method='fork'))
        wandb.config.update(opt)

    for epoch in range(start_epoch, opt.nepoch):
        if opt.train: 
            loss = train(opt, actions, train_dataloader, model, optimizer, epoch)
        
        p1, p2 = val(opt, actions, test_dataloader, model)

        if opt.train and p1 < opt.previous_best_threshold:
            opt.previous_name = save_model(opt.previous_name, opt.checkpoint, epoch, p1, model, 'no_refine')

            if opt.refine:
                opt.previous_refine_name = save_model(opt.previous_refine_name, opt.checkpoint, epoch,
                                                      p1, model['refine'], 'refine')
            opt.previous_best_threshold = p1

        if not opt.train:
            print('p1: %.2f, p2: %.2f' % (p1, p2))
            break
        else:
            logging.info('epoch: %d, lr: %.7f, loss: %.4f, p1: %.2f, p2: %.2f' % (epoch, lr, loss, p1, p2))
            print('e: %d, lr: %.7f, loss: %.4f, p1: %.2f, p2: %.2f' % (epoch, lr, loss, p1, p2))

        if epoch % opt.large_decay_epoch == 0: 
            for param_group in optimizer.param_groups:
                param_group['lr'] *= opt.lr_decay_large
                lr *= opt.lr_decay_large
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= opt.lr_decay
                lr *= opt.lr_decay





