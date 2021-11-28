import os
import math
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam, AdamW

from losses.mae_loss import MSELoss
from data.Datasets import build_dataset
from helpers.utils import setup_seed, Metric_rank
#from models.mae import VisionTransfromersTiny as MAE
from models.mae import MAEVisionTransformers as MAE

from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from train import train_mae, val_mae


parser = argparse.ArgumentParser()

# ----- data
parser.add_argument('--data-path', default='./data/', type=str, help='dataset path')
parser.add_argument('--data-set', default='CIFAR', choices=['CIFAR'], type=str, 
                                  help='Image Net dataset path')
parser.add_argument('--img_size', type=int, default=224)
parser.add_argument('--crop_size', type=int, default=224)
parser.add_argument('--color_jitter', type=float,  default=0.4)
parser.add_argument('--train-interpolation', type=str, default='bicubic',
                    help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
parser.add_argument('--val_size', type=float,  default=0.1)

# ---- optimizer
parser.add_argument('--optimizer_name', default="adamw", type=str)
parser.add_argument('--lr', default=1e-1, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--cosine', default=0, type=int)
parser.add_argument('--weight_decay', default=5e-2, type=float)

# --- vit 
parser.add_argument('--patch_size', default=16, type=int)
parser.add_argument('--finetune', default=0, type=int)

# ---- train
parser.add_argument('--warmup_epochs', default=5, type=int)
parser.add_argument('--num_epochs', default=90, type=int)
parser.add_argument('--step_display', default=1, type=int)
parser.add_argument('--use-gpu', action='store_true')

def main(args):

    # Precise device
    use_cuda = True if torch.cuda.is_available() and args.use_gpu else False
    print("On Cuda!" if use_cuda else "On CPU")
    
    # metric
    train_losses_metric = Metric_rank("train_losses")
    train_accuracy_metric = Metric_rank("train_accuracy")
    train_metric = {  
                      "losses": train_losses_metric,
                      "accuracy": train_accuracy_metric
                   }

    # model
    model = MAE( img_size = args.crop_size, patch_size = args.patch_size,  
                 encoder_dim = 192, encoder_depth = 12, encoder_heads = 3,
                 decoder_dim = 512, decoder_depth = 8, decoder_heads = 16, 
                 mask_ratio = 0.75
               )
    #print(f"=============== model architecture ===============")
    #print(model)
    
    if use_cuda:
        model.cuda()
    
    # Prepare Datasets
    train_loader, val_loader, nb_classes, len_train, len_val = build_dataset(args)
    

    # Choose an Optimizer
    if args.optimizer_name == "sgd":
        optimizer = SGD(model.parameters(), args.lr, weight_decay=args.weight_decay, 
                                                     momentum=0.9)
    elif args.optimizer_name == "adam":
        optimizer = Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    elif args.optimizer_name == "adamw":
        optimizer = AdamW(model.parameters(), args.lr, betas=(0.9, 0.95), 
                                                       weight_decay=args.weight_decay)
    else:
        raise NotImplementedError(f"{args.optimizer_name} optimizer have not been implemented!")
    

    # Mixup params
    mixup = 0.8          # mixup alpha,mixup enabled if > 0
    cutmix = 1.0         # cutmix alpha,cutmix enabled if > 0
    cutmix_minmax = None # cutmix min/max ratio
    mixup_prob = 1.0     # Probability of performing mixup or cutmix when either/both is enabled
    switch_prob = 0.5 # Probability of switching to cutmix when both mixup and cutmix enabled
    mixup_mode = 'batch' # How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
    smoothing = 0.1      # Label smoothing
    num_classes = 1000
    
    # mixup & cutmix
    mixup_fn = None
    if mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None:
        mixup_fn = Mixup( mixup_alpha=mixup, cutmix_alpha=cutmix, cutmix_minmax=cutmix_minmax,
                          prob=mixup_prob, switch_prob=switch_prob, mode=mixup_mode,
                          label_smoothing=smoothing, num_classes=num_classes)
        print("use the mixup function ")
    
    
    start_epoch = 1
    batch_iter = 0
    ngpus_per_node = torch.cuda.device_count()
    train_batch = math.ceil(len_train / (args.batch_size * ngpus_per_node))
    total_batch = train_batch * args.num_epochs

    val_batch = math.ceil(len_val / (args.batch_size * ngpus_per_node))
    val_total_batch = val_batch * args.num_epochs

    scaler = torch.cuda.amp.GradScaler()

    criterion = MSELoss()
    
    # training loop
    print()
    print("Start training...")
    for epoch in range(start_epoch, args.num_epochs + 1):
        batch_iter, scaler = train_mae(args, scaler, train_loader, mixup_fn, model, criterion, 
                                       optimizer, epoch, batch_iter, total_batch, train_batch, 
                                       train_metric, use_cuda)
        epoch_loss = val_mae(args, val_loader, model, epoch, val_total_batch, use_cuda)

if __name__ == "__main__":
    args = parser.parse_args()
    setup_seed()
    main(args)
