import datetime
import os
import time
import torch
import torch.utils.data
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import math
from torch.cuda import amp
import smodels, utils
from spikingjelly.clock_driven import functional
from spikingjelly.datasets import dvs128_gesture

_seed_ = 2020
import random
random.seed(2020)

torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.cuda.manual_seed_all(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import numpy as np
np.random.seed(_seed_)

def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, print_freq, scaler=None, T_train=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', utils.SmoothedValue(window_size=10, fmt='{value}'))

    header = 'Epoch: [{}]'.format(epoch)

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        image = image.float()  # [N, T, C, H, W]

        if T_train:
            sec_list = np.random.choice(image.shape[1], T_train, replace=False)
            sec_list.sort()
            image = image[:, sec_list]

        if scaler is not None:
            with amp.autocast():
                output = model(image)
                loss = criterion(output, target)
        else:
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            loss.backward()
            optimizer.step()

        functional.reset_net(model)

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        loss_s = loss.item()
        if math.isnan(loss_s):
            raise ValueError('loss is Nan')
        acc1_s = acc1.item()
        acc5_s = acc5.item()

        metric_logger.update(loss=loss_s, lr=optimizer.param_groups[0]["lr"])

        metric_logger.meters['acc1'].update(acc1_s, n=batch_size)
        metric_logger.meters['acc5'].update(acc5_s, n=batch_size)
        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    return metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg



def evaluate(model, criterion, data_loader, device, print_freq=100, header='Test:'):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            image = image.float()
            output = model(image)
            loss = criterion(output, target)
            functional.reset_net(model)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    loss, acc1, acc5 = metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg
    print(f' * Acc@1 = {acc1}, Acc@5 = {acc5}, loss = {loss}')
    return loss, acc1, acc5

def load_data(dataset_dir, distributed, T):
    # Data loading code
    print("Loading data")

    st = time.time()

    dataset_train = dvs128_gesture.DVS128Gesture(root=dataset_dir, train=True, data_type='frame', frames_number=T, split_by='number')
    dataset_test = dvs128_gesture.DVS128Gesture(root=dataset_dir, train=False, data_type='frame', frames_number=T,
                                                 split_by='number')


    print("Took", time.time() - st)

    print("Creating data loaders")
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset_train)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset_train, dataset_test, train_sampler, test_sampler

def main(args):

    max_test_acc1 = 0.
    test_acc5_at_max_test_acc1 = 0.


    train_tb_writer = None
    te_tb_writer = None

    utils.init_distributed_mode(args)
    print(args)

    output_dir = os.path.join(args.output_dir, f'{args.model}_b{args.batch_size}_T{args.T}')

    if args.T_train:
        output_dir += f'_Ttrain{args.T_train}'

    if args.weight_decay:
        output_dir += f'_wd{args.weight_decay}'

    output_dir += f'_steplr{args.lr_step_size}_{args.lr_gamma}'

    if args.adam:
        output_dir += '_adam'
    else:
        output_dir += '_sgd'

    if args.connect_f:
        output_dir += f'_cnf_{args.connect_f}'

    if not os.path.exists(output_dir):
        utils.mkdir(output_dir)

    output_dir = os.path.join(output_dir, f'lr{args.lr}')
    if not os.path.exists(output_dir):
        utils.mkdir(output_dir)



    device = torch.device(args.device)

    data_path = args.data_path

    dataset_train, dataset_test, train_sampler, test_sampler = load_data(data_path, args.distributed, args.T)
    print(f'dataset_train:{dataset_train.__len__()}, dataset_test:{dataset_test.__len__()}')

    data_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, pin_memory=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size,
        sampler=test_sampler, num_workers=args.workers, pin_memory=True)

    model = smodels.__dict__[args.model](args.connect_f)
    print("Creating model")

    model.to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = nn.CrossEntropyLoss()

    if args.adam:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.amp:
        scaler = amp.GradScaler()
    else:
        scaler = None

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        max_test_acc1 = checkpoint['max_test_acc1']
        test_acc5_at_max_test_acc1 = checkpoint['test_acc5_at_max_test_acc1']

    if args.test_only:

        evaluate(model, criterion, data_loader_test, device=device, header='Test:')

        return

    if args.tb and utils.is_main_process():
        purge_step_train = args.start_epoch
        purge_step_te = args.start_epoch
        train_tb_writer = SummaryWriter(output_dir + '_logs/train', purge_step=purge_step_train)
        te_tb_writer = SummaryWriter(output_dir + '_logs/te', purge_step=purge_step_te)
        with open(output_dir + '_logs/args.txt', 'w', encoding='utf-8') as args_txt:
            args_txt.write(str(args))

        print(f'purge_step_train={purge_step_train}, purge_step_te={purge_step_te}')

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        save_max = False
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_loss, train_acc1, train_acc5 = train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args.print_freq, scaler, args.T_train)
        if utils.is_main_process():
            train_tb_writer.add_scalar('train_loss', train_loss, epoch)
            train_tb_writer.add_scalar('train_acc1', train_acc1, epoch)
            train_tb_writer.add_scalar('train_acc5', train_acc5, epoch)
        lr_scheduler.step()

        test_loss, test_acc1, test_acc5 = evaluate(model, criterion, data_loader_test, device=device, header='Test:')
        if te_tb_writer is not None:
            if utils.is_main_process():

                te_tb_writer.add_scalar('test_loss', test_loss, epoch)
                te_tb_writer.add_scalar('test_acc1', test_acc1, epoch)
                te_tb_writer.add_scalar('test_acc5', test_acc5, epoch)


        if max_test_acc1 < test_acc1:
            max_test_acc1 = test_acc1
            test_acc5_at_max_test_acc1 = test_acc5
            save_max = True


        if output_dir:

            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
                'max_test_acc1': max_test_acc1,
                'test_acc5_at_max_test_acc1': test_acc5_at_max_test_acc1,
            }

            if save_max:
                utils.save_on_master(
                    checkpoint,
                    os.path.join(output_dir, 'checkpoint_max_test_acc1.pth'))
        print(args)
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))

        print('Training time {}'.format(total_time_str), 'max_test_acc1', max_test_acc1, 'test_acc5_at_max_test_acc1', test_acc5_at_max_test_acc1)
        print(output_dir)
    if output_dir:
        utils.save_on_master(
            checkpoint,
            os.path.join(output_dir, f'checkpoint_{epoch}.pth'))

    return max_test_acc1



def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')
    parser.add_argument('--model', help='model')

    parser.add_argument('--data-path', help='dataset')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=16, type=int)
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--lr', type=float, help='initial learning rate', default=0.1)
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='Momentum for SGD. Adam will not use momentum')
    parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                        metavar='W', help='weight decay (default: 0)',
                        dest='weight_decay')
    parser.add_argument('--lr-step-size', default=30, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=64, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )

    # Mixed precision training parameters
    parser.add_argument('--amp', action='store_true',
                        help='Use AMP training')


    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--tb', action='store_true',
                        help='Use TensorBoard to record logs')
    parser.add_argument('--T', default=16, type=int, help='simulation steps')
    parser.add_argument('--adam', action='store_true',
                        help='Use Adam')

    parser.add_argument('--connect_f', type=str, help='element-wise connect function')
    parser.add_argument('--T_train', type=int)

    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parse_args()
    main(args)

'''
/raid/wfang/datasets/DVS128Gesture

python train.py --tb --amp --output-dir ./logs --model PlainNet --device cuda:0 --lr-step-size 64 --epoch 192 --T_train 12 --T 16 --data-path /raid/wfang/datasets/DVS128Gesture

python train.py --tb --amp --output-dir ./logs --model SEWResNet --connect_f ADD --device cuda:0 --lr-step-size 64 --epoch 192 --T_train 12 --T 16 --data-path /raid/wfang/datasets/DVS128Gesture

'''
