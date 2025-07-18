# -*- coding: utf-8 -*-

import datetime
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import sys
import argparse
import math

# from models import spiking_resnet_imagenet

import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda import amp

import torchvision.transforms as transforms
import torchvision.datasets as datasets
# from torchtoolbox.transform import Cutout  ### pip install torchtoolbox


import pickle

from models import resnet
from modules import neuron, surrogate
from modules.Bop import Bop, CustomScheduler

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig, seed_all
from data import datapool


from functions import train, evaluate
import data.utils as utils



seed_all(seed=2025, benchmark=False)



def parse_args():
    parser = argparse.ArgumentParser(description='Train on ImageNet')
    parser.add_argument('--gpu_id', default='0', type=str, help='gpu id')
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--model', type=str, default='birealnet18')

    # dataset has cifar10, cifar100, cifardvs and imagenet
    parser.add_argument('--dataset', default='imagenet', type=str)
    parser.add_argument('--optimizer', default='Bop', type=str, help='use which optimizer. SGD Adam or Bop')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay for SGD or Adam')
    parser.add_argument('--j', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8) workers')

    parser.add_argument('--lr_scheduler', default='CosALR', type=str, help='use which schedule. StepLR or CosALR')
    parser.add_argument('--step_size', default=100, type=float, help='step_size for StepLR')
    parser.add_argument('--step_gamma', default=0.1, type=float, help='gamma for StepLR')
    parser.add_argument('--T_max', default=300, type=int, help='T_max for CosineAnnealingLR')
    parser.add_argument('--print_freq', default=100, type=int, help='print freq in train and evaluate')

    parser.add_argument('--start_epoch', default=0, type=int, help='starting epoch')
    parser.add_argument('--resume', type=int, help='resume or not from the checkpoint path')
    parser.add_argument('--output_dir', type=str, help='root dir for saving logs and checkpoint', default='./logs')

    # ### Settings of the LIFSpike Neuron
    parser.add_argument('--T', default=6, type=int, help='simulating time-steps')
    parser.add_argument('--tau', default=2., type=float)    # The decay constant is lambda= 1.0-1.0/tau, lambda < 1.0


    parser.add_argument('--amp', action='store_true', help='automatic mixed precision training')  # default false
    parser.add_argument('--tb', action='store_true', help='using tensorboard')  # using tensorboard
    parser.add_argument('--autoaug', action='store_true', help='using auto augmentation')  # default false
    parser.add_argument('--cutout', action='store_true', help='using cutout')
    parser.add_argument('--drop_rate', type=float, default=0.0, help='dropout rate')

    parser.add_argument('--test_only', action='store_true')  # 'store_true' means by default, it is false.

    # parser.add_argument('--stochdepth_rate', type=float, default=0.0)
    # parser.add_argument('--cnf', type=str)
    # parser.add_argument('--T_train', default=None, type=int)

    parser.add_argument('--dts_cache', type=str, default='./dts_cache')
    parser.add_argument('--loss_lambda', type=float, default=0.05)
    parser.add_argument('--online_update', action='store_true',  help='use online update')   # default means false.


    # BSO args
    parser.add_argument('--threshold',default=1e-8,type=float, metavar='N')
    parser.add_argument('--beta1',default=0.999,type=float,metavar='N')
    parser.add_argument('--beta2',default=0.99999,type=float,metavar='N')


    parser.add_argument('--data_path', default='/data/dataset/ImageNet', type=str)
    parser.add_argument('--train_dir', default='train', type=str)
    parser.add_argument('--val_dir', default='val', type=str)
    parser.add_argument('--stochdepth_rate', type=float, default=0.0)

    parser.add_argument('--distributed', action='store_true', help='Enable distributed training')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--rank', default=-1, type=int)
    parser.add_argument('--world-size', default=-1, type=int)
    parser.add_argument('--gpu', default=None, type=int)
    parser.add_argument('--grad_with_rate', action='store_true', help='use gradient with rate')
    


    args = parser.parse_args()
    # args = parser.parse_args([])  # for jupyter notebook
    print(args)

    return args



def logger_plot_save(logger_train_test_acc, logger_train_test_loss, prefix_name, plot_dir):
    # snames = ['Acc1.', 'Acc5.']
    # snames = ['Loss', 'Batch Loss', 'CE Loss', 'MSE Loss', 'Loss_0', 'Acc_0']

    # ## Plot the loss and accuracy
    fig = logger_train_test_acc.plot(['Train Acc1.', 'Test Acc1.'])
    fig.savefig(plot_dir + f'/{prefix_name}_logger_acc1.pdf')
    fig = logger_train_test_acc.plot(['Train Acc5.', 'Test Acc5.'])
    fig.savefig(plot_dir + f'/{prefix_name}_logger_acc5.pdf')
    fig = logger_train_test_acc.plot(['LR'])
    fig.savefig(plot_dir + f'/{prefix_name}_logger_LR.pdf')

    # # ### fig = logger.plot()  # this cause problems, as the scales are different
    fig = logger_train_test_loss.plot(['Train Loss', 'Test Loss'])
    fig.savefig(plot_dir + f'/{prefix_name}_logger_loss.pdf')
    fig = logger_train_test_loss.plot(['Train Batch Loss', 'Test Batch Loss'])
    fig.savefig(plot_dir + f'/{prefix_name}_logger_batch_loss.pdf')
    fig = logger_train_test_loss.plot(['Train CE Loss', 'Test CE Loss'])
    fig.savefig(plot_dir + f'/{prefix_name}_logger_ce_loss.pdf')
    fig = logger_train_test_loss.plot(['Train MSE Loss', 'Test MSE Loss'])
    fig.savefig(plot_dir + f'/{prefix_name}_logger_mse_loss.pdf')

    fig = logger_train_test_loss.plot(['Train Loss_0', 'Test Loss_0'])
    fig.savefig(plot_dir + f'/{prefix_name}_logger_loss_0.pdf')
    fig = logger_train_test_loss.plot(['Train Acc_0', 'Test Acc_0'])
    fig.savefig(plot_dir + f'/{prefix_name}_logger_acc_0.pdf')

    # ## Save it to a file
    fname = plot_dir + f'/{prefix_name}_logger_acc.pkl'
    with open(fname, 'wb') as f:
        pickle.dump(logger_train_test_acc.numbers, f)
    fname = plot_dir + f'/{prefix_name}_logger_loss.pkl'
    with open(fname, 'wb') as f:
        pickle.dump(logger_train_test_loss.numbers, f)
    # ## and later you can load it
    # with open('test_logger.pkl', 'rb') as f:
    #     dt = pickle.load(f)



def main(args):
    
    args.distributed = True
    args.world_size = 4
    args.gpu_id = '0,1,2,3'
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    # Check if CUDA (GPU support) is available
    if torch.cuda.is_available():
        args.device = torch.device("cuda")
        print("CUDA (GPU) is available.")
    else:
        args.device = torch.device("cpu")
        print("CUDA (GPU) is not available. Using CPU.")
    device = args.device

    if args.distributed:
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            args.rank = int(os.environ["RANK"])
            args.world_size = int(os.environ['WORLD_SIZE'])
            args.gpu = int(os.environ['LOCAL_RANK'])
        else:
            print('Not using distributed mode')
            args.distributed = False
            return

        torch.cuda.set_device(args.gpu)
        args.dist_backend = 'nccl'
        print('| distributed init (rank {}): {}'.format(
            args.rank, args.dist_url), flush=True)
        torch.distributed.init_process_group(
            backend=args.dist_backend, 
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank)
        torch.distributed.barrier()


    print("Initial args: ", args)
    print("=== Initial args in main() === ")
    # Iterate through the attributes of the args object
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    
    # ImageNet data load
    num_classes = 1000
    traindir = os.path.join(args.data_path, args.train_dir)
    valdir = os.path.join(args.data_path, args.val_dir)
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
    
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
        
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True) if args.distributed else None
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size // args.world_size,
        shuffle=(train_sampler is None),
        num_workers=args.j,
        pin_memory=True,
        sampler=train_sampler,
        persistent_workers=True,
        drop_last=True
    )
        
    test_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size // args.world_size , shuffle=False,
        num_workers=args.j, pin_memory=True)

    model = resnet.__dict__[args.model](
        single_step_neuron=neuron.OnlineLIFNode,
        tau=args.tau,
        # surrogate_function=surrogate.Sigmoid(alpha=4.),
        # track_rate=True,
        c_in=3,
        num_classes=1000,
        # neuron_dropout=args.drop_rate,
        drop_rate=args.drop_rate,
        stochdepth_rate=args.stochdepth_rate,
        grad_with_rate=True
        # v_reset=None
    )

    print("===> Creating model")
    print(model)

    print('=== *** Total Parameters: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    # model.cuda()
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.cuda(args.gpu)
        model = DDP(model, device_ids=[args.gpu], find_unused_parameters=True)
    else:
        model = model.to(args.device)


    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "Bop":
        # parameters = split_weights(model)
        optimizer = Bop(model,lr=args.lr,threshold=args.threshold, beta1=1.0-args.beta1, beta2=1.0-args.beta2, weight_decay=args.weight_decay)
        beta1_sch = CustomScheduler(optimizer, param_name="beta1", decay_epochs=20, decay=0.1)
        thres_sch = CustomScheduler(optimizer, param_name="threshold", decay_epochs=20, decay=0.1)
    else:
        raise NotImplementedError(args.optimizer)


    if args.lr_scheduler == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.step_gamma)
    elif args.lr_scheduler == 'CosALR':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max)
    else:
        raise NotImplementedError(args.lr_scheduler)

    if args.amp:
        scaler = amp.GradScaler()
    else:
        scaler = None


    max_test_acc1 = 0.0
    max_test_acc5 = 0.0


    fname = f'_{args.dataset}_{args.model}_T_{args.T}_opt_{args.optimizer}' \
        f'_lr_{args.lr}_bs_{args.batch_size}_wd_{args.weight_decay}_epochs_{args.epochs}' \
        f'_autoaug_{args.autoaug}_coutout_{args.cutout}'
    output_dir = args.output_dir + fname

    if args.lr_scheduler == 'CosALR':
        output_dir += f'_CosALR_{args.T_max}'
    elif args.lr_scheduler == 'StepLR':
        output_dir += f'_StepLR_{args.step_size}_{args.step_gamma}'
    else:
        raise NotImplementedError(args.lr_scheduler)
    if args.amp:
        output_dir += '_amp'

    prefix_name = f'{args.dataset}_{args.model}_T_{args.T}_tau_{args.tau}'
    utils.mkdir(output_dir)
    print(output_dir)
    plot_dir = os.path.join('./output_dir', fname + 'plots')
    utils.mkdir(plot_dir)


    # ## Resume, optionally resume from a checkpoint
    if args.resume:
        print('==> Resuming from checkpoint...')
        resume_name = os.path.join(output_dir, f'{prefix_name}_checkpoint_latest.pth')
        assert os.path.isfile(resume_name), 'Error: no checkpoint directory found!'

        checkpoint = torch.load(resume_name, map_location='cpu')
        state_dict = checkpoint['model_state_dict']

        if args.distributed:
            try:
                model.load_state_dict(state_dict)
            except:
                new_state_dict = {}
                for k, v in state_dict.items():
                    name = k
                    if not k.startswith('module.'):
                        name = 'module.' + k
                    new_state_dict[name] = v
                model.load_state_dict(new_state_dict)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace('module.', '')
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)



        # model.load_state_dict(checkpoint['model_state_dict'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        max_test_acc1 = checkpoint['max_test_acc1']
        max_test_acc5 = checkpoint['max_test_acc5']

        # ## The logger with 'append' mode
        logger = utils.get_logger(output_dir + f'/{prefix_name}_training_log.log', file_mode='a')
        logger.parent = None
        logger.info(output_dir)
        logger.info(args)
        logger.info("Resume training")

        print('==> Creating/Resuming loggers...')
        logger_train_test_acc = Logger(output_dir + f'/{prefix_name}_logger_train_test_acc.txt', title='TrainTestAcc')
        logger_train_test_loss = Logger(output_dir + f'/{prefix_name}_logger_train_test_loss.txt', title='TrainTestLoss')
        
        # 设置logger格式
        snames = ['Acc1.', 'Acc5.']
        logger_train_test_acc.set_names(
            ['Epoch', 'LR'] + ['Train ' + i for i in snames] + ['Test ' + i for i in snames]
        )
        logger_train_test_acc.set_formats([
            '{0:d}', '{0:.6f}',
            '{0:.3f}', '{0:.3f}',
            '{0:.3f}', '{0:.3f}',
        ])

        snames = ['Loss', 'Batch Loss', 'CE Loss', 'MSE Loss', 'Loss_0', 'Acc_0']
        logger_train_test_loss.set_names(
            ['Epoch', 'LR'] + ['Train ' + i for i in snames] + ['Test ' + i for i in snames]
        )
        logger_train_test_loss.set_formats([
            '{0:d}', '{0:.6f}',
            '{0:.4f}', '{0:.4f}', '{0:.4f}', '{0:.4f}', '{0:.4f}', '{0:.3f}',
            '{0:.4f}', '{0:.4f}', '{0:.4f}', '{0:.4f}', '{0:.4f}', '{0:.3f}',
        ])

    # ## If there is not from resume
    else:
        # ## The logger with 'write' mode
        logger = utils.get_logger(output_dir + f'/{prefix_name}_training_log.log', file_mode='w')
        logger.parent = None
        logger.info(output_dir)
        logger.info(args)
        logger.info("Start training")

        print('No existing log file, have to create a new one.')
        print('==> Trainig from epoch=0...')

        # #### New added, using the logger
        print('==> Creating new logger_train and logger_test...')
        logger_train_test_acc = Logger(output_dir + f'/{prefix_name}_logger_train_test_acc.txt', title='TrainTestAcc')
        snames = ['Acc1.', 'Acc5.']
        logger_train_test_acc.set_names(
            ['Epoch', 'LR'] + ['Train ' + i for i in snames] + ['Test ' + i for i in snames]
            )
        logger_train_test_acc.set_formats([
            '{0:d}', '{0:.6f}',
            '{0:.3f}', '{0:.3f}',
            '{0:.3f}', '{0:.3f}',
            ])

        logger_train_test_loss = Logger(output_dir + f'/{prefix_name}_logger_train_test_loss.txt', title='TrainTestLoss')
        snames = ['Loss', 'Batch Loss', 'CE Loss', 'MSE Loss', 'Loss_0', 'Acc_0']
        logger_train_test_loss.set_names(
            ['Epoch', 'LR'] + ['Train ' + i for i in snames] + ['Test ' + i for i in snames]
            )
        logger_train_test_loss.set_formats([
            '{0:d}', '{0:.6f}',
            '{0:.4f}', '{0:.4f}', '{0:.4f}', '{0:.4f}', '{0:.4f}', '{0:.3f}',
            '{0:.4f}', '{0:.4f}', '{0:.4f}', '{0:.4f}', '{0:.4f}', '{0:.3f}',
            ])
        # # #### New added

    criterion_mse = nn.MSELoss(reduce=True)
    t_step = args.T

    if args.test_only:
        results = evaluate(test_loader, model, criterion_mse, num_classes, device, t_step, args)
        print(results)
        return

    if args.tb:  # using tensorboard
        purge_step_train = args.start_epoch
        purge_step_te = args.start_epoch
        train_tb_writer = SummaryWriter(output_dir + f'/{prefix_name}_train_logs', purge_step=purge_step_train)
        test_tb_writer = SummaryWriter(output_dir + f'/{prefix_name}_test_logs', purge_step=purge_step_te)
        print(f'purge_step_train={purge_step_train}, purge_step_te={purge_step_te}')

    with open(output_dir + f'/{prefix_name}_args.txt', 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))

    print('==> Start training')
    print('==> args that feed into training!')
    print(f'==> Trainig from epoch={args.start_epoch}...')
    print(args)

    start_time = time.time()

    writer = SummaryWriter(os.path.join(output_dir, f'{prefix_name}_logs.logs'), purge_step=args.start_epoch)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        save_max = False
        cur_lr = optimizer.param_groups[0]["lr"]

        results = train(train_loader, model, optimizer, criterion_mse, num_classes, device, epoch, t_step, args, scaler)

        if args.distributed:
            torch.distributed.barrier()


        train_acc1, train_acc5 = results[:2]
        train_loss, train_loss_batch = results[2:4]
        train_loss_ce, train_loss_mse = results[4:6]
        train_loss_, train_acc_ = results[6:8]


        logger.info(
            'Train Epoch:[{}/{}]\t train_acc1={:.3f}\t train_acc5={:.3f}\t '
            'train_loss={:.5f}\t train_loss_batch={:.5f}\t train_loss_ce={:.5f}\t'
            'train_loss_mse={:.5f}\t train_loss_={:.5f}\t train_acc_={:.5f}\t'
            .format(epoch, args.epochs, train_acc1, train_acc5,
                    train_loss, train_loss_batch, train_loss_ce,
                    train_loss_mse, train_loss_, train_acc_,
                    )
            )
        print(
            'Train Epoch:[{}/{}]\t train_acc1={:.3f}\t train_acc5={:.3f}\t '
            'train_loss={:.5f}\t train_loss_batch={:.5f}\t train_loss_ce={:.5f}\t'
            'train_loss_mse={:.5f}\t train_loss_={:.5f}\t train_acc_={:.5f}\t\n'
            .format(epoch, args.epochs, train_acc1, train_acc5,
                    train_loss, train_loss_batch, train_loss_ce,
                    train_loss_mse, train_loss_, train_acc_,
                    )
            )
        if args.tb:
            train_tb_writer.add_scalar('train_acc1', train_acc1, epoch)
            train_tb_writer.add_scalar('train_acc5', train_acc5, epoch)
            train_tb_writer.add_scalar('train_loss', train_loss, epoch)
            train_tb_writer.add_scalar('train_loss_batch', train_loss_batch, epoch)
            train_tb_writer.add_scalar('train_loss_ce', train_loss_ce, epoch)
            train_tb_writer.add_scalar('train_loss_mse', train_loss_mse, epoch)
            train_tb_writer.add_scalar('train_loss_', train_loss_, epoch)
            train_tb_writer.add_scalar('train_acc_', train_acc_, epoch)

        writer.add_scalar('train_loss', train_loss_, epoch)
        writer.add_scalar('train_acc', train_acc_, epoch)

        lr_scheduler.step()
        if args.optimizer == 'Bop':
            beta1_sch.step(epoch)
            thres_sch.step(epoch)
#########################################

        results = evaluate(test_loader, model, criterion_mse, num_classes, device, t_step, args)
        test_acc1, test_acc5 = results[:2]
        test_loss, test_loss_batch = results[2:4]
        test_loss_ce, test_loss_mse = results[4:6]
        test_loss_, test_acc_ = results[6:8]

        # ## Append logger file
        logger_train_test_acc.append(
            [epoch, cur_lr,
             train_acc1, train_acc5,
             test_acc1, test_acc5,
             ])

        logger_train_test_loss.append(
            [epoch, cur_lr,
             train_loss, train_loss_batch, train_loss_ce, train_loss_mse, train_loss_, train_acc_,
             test_loss, test_loss_batch, test_loss_ce, test_loss_mse, test_loss_, test_acc_,
             ])


        logger.info(
            'Test Epoch:[{}/{}]\t test_acc1={:.3f}\t test_acc5={:.3f}\t '
            'test_loss={:.5f}\t test_loss_batch={:.5f}\t test_loss_ce={:.5f}\t'
            'test_loss_mse={:.5f}\t test_loss_={:.5f}\t test_acc_={:.5f}\t'
            .format(epoch, args.epochs, test_acc1, test_acc5,
                    test_loss, test_loss_batch, test_loss_ce,
                    test_loss_mse, test_loss_, test_acc_,
                    )
            )
        print(
            'Test Epoch:[{}/{}]\t test_acc1={:.3f}\t test_acc5={:.3f}\t '
            'test_loss={:.5f}\t test_loss_batch={:.5f}\t test_loss_ce={:.5f}\t'
            'test_loss_mse={:.5f}\t test_loss_={:.5f}\t test_acc_={:.5f}\t\n'
            .format(epoch, args.epochs, test_acc1, test_acc5,
                    test_loss, test_loss_batch, test_loss_ce,
                    test_loss_mse, test_loss_, test_acc_,
                    )
            )

        if args.tb and test_tb_writer is not None:
            test_tb_writer.add_scalar('test_acc1', test_acc1, epoch)
            test_tb_writer.add_scalar('test_acc5', test_acc5, epoch)
            test_tb_writer.add_scalar('test_loss', test_loss, epoch)
            test_tb_writer.add_scalar('test_loss_batch', test_loss_batch, epoch)
            test_tb_writer.add_scalar('test_loss_ce', test_loss_ce, epoch)
            test_tb_writer.add_scalar('test_loss_mse', test_loss_mse, epoch)
            test_tb_writer.add_scalar('test_loss_', test_loss_, epoch)
            test_tb_writer.add_scalar('test_acc_', test_acc_, epoch)

        writer.add_scalar('test_loss', test_loss_, epoch)
        writer.add_scalar('test_acc', test_acc_, epoch)

        save_max = False
        if max_test_acc1 < test_acc1:
            max_test_acc1 = test_acc1
            max_test_acc5 = test_acc5
            save_max = True

        if output_dir:
            checkpoint = {
                'model_arch': args.model,
                'dataset': args.dataset,
                'model_state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'max_test_acc1': max_test_acc1,
                'max_test_acc5': max_test_acc5,
                'epoch': epoch,
                'args': args,
            }

            torch.save(checkpoint, os.path.join(output_dir, f'{prefix_name}_checkpoint_latest.pth'))

            save_flag = False
            # if epoch % 64 == 0 or epoch == args.epochs - 1:
            if epoch == args.epochs - 1:
                save_flag = True
            if save_flag:
                torch.save(checkpoint, os.path.join(output_dir, f'{prefix_name}_checkpoint_{epoch}.pth'))

            if save_max:
                torch.save(checkpoint, os.path.join(output_dir, f'{prefix_name}_checkpoint_max_test_acc1.pth'))

        print(args)
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(output_dir)

        total_time = time.time() - start_time
        print(f'epoch={epoch}, train_loss={train_loss}, train_acc1={train_acc1}, '
              f'test_loss={test_loss}, test_acc1={test_acc1}, '
              f'max_test_acc1={max_test_acc1}, total_time={total_time}, '
              f'escape_time={(datetime.datetime.now()+datetime.timedelta(seconds=total_time * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}'
              )

        print('Training time {}\t max_test_acc1: {} \t max_test_acc5: {} \n'
                      .format(total_time_str, max_test_acc1, max_test_acc5))

        logger.info('Training time {}\t max_test_acc1: {} \t max_test_acc5: {}'
                    .format(total_time_str, max_test_acc1, max_test_acc5))
        logger.info('\n')


    # ## Outside the for loop, Finish write information
    logger_train_test_acc.close()
    logger_train_test_loss.close()
    print('* Done training')


    # ## Print the best accuracy
    print('================================ Training finished! ================================')
    print(f'Best test acc:   {max_test_acc1} ')

    # ## Plot the loss and accuracy
    logger_plot_save(logger_train_test_acc, logger_train_test_loss, prefix_name, plot_dir)



if __name__ == '__main__':
    args = parse_args()

    print("Initial args: ", args)
    print("=== Initial args in __main__ === ")
    # Iterate through the attributes of the args object
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    main(args)