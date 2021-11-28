import time
import numpy as np
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast

from losses.mae_loss import MSELoss, build_mask
from helpers.scheduling import step_learning_rate, cos_learning_rate



# train function per batch
def train_mae(args, scaler, train_loader, mixup_fn, model, criterion, optimizer, 
              epoch, batch_iter, total_batch, train_batch, train_metric, use_cuda=False):

    model.train()

    loader_length = len(train_loader)
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        batch_start = time.time()

        if args.cosine:
            # cosine learning rate
            lr = cos_learning_rate(args, epoch, batch_iter, optimizer, train_batch)
        else:
            # step learning rate
            lr = step_learning_rate(args, epoch, batch_iter, optimizer, train_batch)
        
        if use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        # mixup or cutmix
        if mixup_fn is not None:
            inputs, targets = mixup_fn(inputs, targets)
            targets = targets.argmax(dim=1)  # translate the miuxp one hot to float

        with autocast():
            outputs, mask_index = model(inputs)
            mask = build_mask(mask_index, args.patch_size, args.crop_size)
            losses = criterion(outputs, inputs, mask)

        optimizer.zero_grad()

        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()


        # record the average momentum result
        train_metric["losses"].update(losses.data.item())

        batch_time = time.time() - batch_start

        batch_iter += 1

        if epoch % args.step_display == 0:
            print("[Training] Time: {} Epoch: [{}/{}] batch_idx: [{}/{}] batch_iter: [{}/{}] batch_losses: {:.4f} LearningRate: {:.9f} BatchTime: {:.4f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    epoch,
                    args.num_epochs,
                    batch_idx,
                    train_batch,
                    batch_iter,
                    total_batch,
                    losses.data.item(),
                    lr,
                    batch_time
                ))


    return batch_iter, scaler


# Validation phase
def val_mae(args, val_loader, model, epoch, val_total_batch, use_cuda=False):
    
    model.eval()
    epoch_losses, epoch_accuracy = 0.0, 0.0
    criterion = MSELoss() # nn.CrossEntropyLoss()
    
    batch_acc_list = []
    batch_loss_list = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):

            if use_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()

            batch_size = inputs.shape[0]

            with autocast():
                outputs, mask_index = model(inputs)
                mask = build_mask(mask_index, args.patch_size, args.crop_size)
                losses = criterion(outputs, inputs, mask)

            batch_loss_list.append(losses.data.item())
            
            if epoch % args.step_display == 0:
                print(f"Validation Epoch: [{epoch}/{args.num_epochs}] batch_idx: [{batch_idx}/{val_total_batch}] batch_losses: {losses.data.item()}")
            
    epoch_loss = np.mean(batch_loss_list)


    

    return epoch_loss


  
    
    
    
    
    
