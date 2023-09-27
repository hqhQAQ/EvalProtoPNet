import os
import re
import time
import torch
import shutil
import random
import logging
import datetime
import argparse
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from pathlib import Path

import model, train_and_test as tnt
import util.utils as utils
from util.utils import str2bool
from torch.utils.tensorboard import SummaryWriter
from util.preprocess import mean, std
from util.eval_interpretability import evaluate_consistency


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If you are using multi-GPU
    np.random.seed(seed)  # Numpy module
    random.seed(seed)  # Python random module
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_outlog(args):
    if args.eval: # Evaluation only
        logfile_dir = os.path.join(args.output_dir, "eval-logs")
    else: # Training
        logfile_dir = os.path.join(args.output_dir, "train-logs")
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    tb_dir = os.path.join(args.output_dir, "tf-logs")
    tb_log_dir = os.path.join(tb_dir, args.base_architecture+ "_" + args.data_set)
    os.makedirs(logfile_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)
    tb_writer = SummaryWriter(
        log_dir=os.path.join(
            tb_dir,
            args.base_architecture+ "_" + args.data_set
        ),
        flush_secs=1
    )
    logger = utils.get_logger(
        level=logging.INFO,
        mode="w",
        name=None,
        logger_fp=os.path.join(
            logfile_dir,
            args.base_architecture+ "_" + args.data_set + ".log"
        )
    )

    logger = logging.getLogger("main")
    return tb_writer, logger


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1028)
parser.add_argument('--output_dir', default='output_debug/test/')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--resume', default='', help='resume from checkpoint')  
# Data
parser.add_argument('--data_set', default='CUB2011', 
    choices=['CUB2011U', 'Car', 'Dogs', 'CUB2011'], type=str)
parser.add_argument('--data_path', type=str, default='datasets/cub200_cropped/')
parser.add_argument('--train_batch_size', default=80, type=int)
parser.add_argument('--test_batch_size', default=150, type=int)

# Model
parser.add_argument('--base_architecture', type=str, default='vgg16')
parser.add_argument('--input_size', default=224, type=int, help='images input size')
parser.add_argument('--save_ep_freq', default=400, type=int, help='save epoch frequency')
parser.add_argument('--prototype_shape', nargs='+', type=int, default=[2000, 64, 1, 1])
parser.add_argument('--prototype_activation_function', type=str, default='log')
parser.add_argument('--add_on_layers_type', type=str, default='regular')

# Loss
parser.add_argument('--use_ortho_loss', type=str2bool, default=True)
parser.add_argument('--ortho_coe', type=float, default=1e-4)
parser.add_argument('--consis_coe', type=float, default=0.30)
parser.add_argument('--consis_thresh', type=float, default=0.10)

# Optimizer & Scheduler
parser.add_argument('--opt', default='adam', type=str, metavar='OPTIMIZER')
parser.add_argument('--sched', default='step', type=str, metavar='SCHEDULER')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR')
parser.add_argument('--features_lr', type=float, default=1e-4)
parser.add_argument('--add_on_layers_lr', type=float, default=3e-3)
parser.add_argument('--prototype_vectors_lr', type=float, default=3e-3)
parser.add_argument('--activation_weight_lr', type=float, default=1e-6)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N')
parser.add_argument('--decay_epochs', type=int, default=5)
parser.add_argument('--decay_rate', type=float, default=0.1)

# Distributed training
parser.add_argument('--device', default='cuda', help='device to use for training / testing')
parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')

args = parser.parse_args()

__global_values__ = dict(it=0)
seed = args.seed + utils.get_rank()
set_seed(seed)

# Distributed Training
utils.init_distributed_mode(args)

tb_writer, logger = get_outlog(args)

device = torch.device(args.device)

# Setting Parameters
base_architecture = args.base_architecture
dataset_name = args.data_set

base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)
model_dir = args.output_dir

os.makedirs(model_dir, exist_ok=True)

shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'models', base_architecture_type + '_features.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'model.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'train_and_test.py'), dst=model_dir)

if dataset_name == 'CUB2011':
    args.nb_classes = 200
elif dataset_name == 'Car':
    args.nb_classes = 196
img_size = args.input_size

joint_optimizer_lrs = {'features': args.features_lr,
                    'add_on_layers': args.add_on_layers_lr,
                    'prototype_vectors': args.prototype_vectors_lr,
                    'activation_weight': args.activation_weight_lr}
warm_optimizer_lrs = {'add_on_layers': args.add_on_layers_lr,
                    'prototype_vectors': args.prototype_vectors_lr,
                    'activation_weight': args.activation_weight_lr}
coefs = {
    'crs_ent': 1,
    'orth': 1e-4,
    'clst': 0.8,
    'sep': -0.08,
    'consis': args.consis_coe,
}

normalize = transforms.Normalize(mean=mean,std=std)

# All datasets
train_dataset = datasets.ImageFolder(
    os.path.join(args.data_path, 'train_cropped_augmented'),
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ]))
test_dataset = datasets.ImageFolder(
    os.path.join(args.data_path, 'test_cropped'),
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ]))

if args.distributed:
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    if args.dist_eval:
        if len(test_dataset) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(
            test_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(test_dataset)
else:
    sampler_train = torch.utils.data.RandomSampler(train_dataset)
    sampler_val = torch.utils.data.SequentialSampler(test_dataset)

# train loader & test loader
train_loader = torch.utils.data.DataLoader(
    train_dataset, sampler=sampler_train,
    batch_size=args.train_batch_size,
    num_workers=4, 
    pin_memory=False)
test_loader = torch.utils.data.DataLoader(
    test_dataset, sampler=sampler_val,
    batch_size=args.test_batch_size,
    num_workers=4, 
    pin_memory=False)

# construct the model
ppnet = model.construct_OursNet(base_architecture=args.base_architecture,
                              pretrained=True, img_size=img_size,
                              prototype_shape=args.prototype_shape,
                              num_classes=args.nb_classes,
                              prototype_activation_function=args.prototype_activation_function,
                              add_on_layers_type=args.add_on_layers_type)
ppnet.to(device)
ppnet_without_ddp = ppnet
if args.distributed:
    ppnet = torch.nn.parallel.DistributedDataParallel(ppnet, device_ids=[args.gpu], find_unused_parameters=True)
    ppnet_without_ddp = ppnet.module
n_parameters = sum(p.numel() for p in ppnet.parameters() if p.requires_grad)
logger.info('number of params: {}'.format(n_parameters))

if args.resume:
    checkpoint = torch.load(args.resume, map_location='cpu')
    ppnet_without_ddp.load_state_dict(checkpoint['model'])
    # test_acc, _ = tnt.test(model=ppnet, epoch=args.epochs, dataloader=test_loader, coefs=coefs, args=args, tb_writer=tb_writer, iteration=__global_values__["it"])

# Define optimizer
joint_optimizer_specs = \
[{'params': ppnet_without_ddp.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3},
 {'params': ppnet_without_ddp.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
 {'params': ppnet_without_ddp.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
 {'params': ppnet_without_ddp.activation_weight, 'lr': joint_optimizer_lrs['activation_weight']},
]
joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=args.decay_epochs, gamma=args.decay_rate)

warm_optimizer_specs = \
[{'params': ppnet_without_ddp.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
 {'params': ppnet_without_ddp.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']},
 {'params': ppnet_without_ddp.activation_weight, 'lr': warm_optimizer_lrs['activation_weight']},
]
warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

max_accuracy, max_consis_score = 0.0, 0.0
output_dir = Path(args.output_dir)

# Train the model
logger.info(f"Start training for {args.epochs} epochs")
start_time = time.time()
for epoch in range(args.epochs):
    if epoch < args.warmup_epochs:
        tnt.warm_only(model=ppnet)
        _, train_results = tnt.train(model=ppnet, epoch=epoch, dataloader=train_loader, optimizer=warm_optimizer,
                    coefs=coefs, args=args, tb_writer=tb_writer, iteration=__global_values__["it"])
    else:
        tnt.joint(model=ppnet)
        joint_lr_scheduler.step()
        _, train_results = tnt.train(model=ppnet, epoch=epoch, dataloader=train_loader, optimizer=joint_optimizer,
                    coefs=coefs, args=args, tb_writer=tb_writer, iteration=__global_values__["it"])

    test_acc, losses = tnt.test(model=ppnet, epoch=epoch, dataloader=test_loader, coefs=coefs, args=args, tb_writer=tb_writer, iteration=__global_values__["it"])
    tb_writer.add_scalar("epoch/val_acc1", test_acc, epoch)
    tb_writer.add_scalar("epoch/val_loss", losses['cross_entropy'], epoch)

    consistency_score = evaluate_consistency(ppnet, args)

    if utils.get_rank() == 0:
        logger.info(f"Consistency score of the network on the {len(test_dataset)} test images: {consistency_score:.2f}%")
        logger.info(f"Accuracy of the network on the {len(test_dataset)} test images: {test_acc:.2f}%")
    
    if epoch == args.epochs - 1:
        checkpoint_paths = [output_dir / 'checkpoints/save_model.pth']
        for checkpoint_path in checkpoint_paths:
            utils.save_on_master({
                'model': ppnet_without_ddp.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)
    max_accuracy = max(max_accuracy, test_acc)
    max_consis_score = max(max_consis_score, consistency_score)

    if utils.get_rank() == 0:
        logger.info(f'Max consistency score: {max_consis_score:.2f}%')
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')

total_time = time.time() - start_time
total_time_str = str(datetime.timedelta(seconds=int(total_time)))
logger.info('Training time {}'.format(total_time_str))