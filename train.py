"""
This function is used to train the models, I mean, all models
Code inspired from: https://github.com/asyml/vision-transformer-pytorch
Because we did not get pretrained model, we plan to train all datasets from scratch!

Created by Kunhong Yu
Date: 2021/07/15
"""
import torch as t
from config import Config
from utils import get_data_loader, setup_device, accuracy, get_model, vis_results_simple
from tqdm import tqdm
import os

opt = Config()

def train_epoch(epoch, model, dataloader, cost, optimizer, batch_size, lr_scheduler, device):
    """Train model only one epoch
    Args :
        --epoch: current epoch
        --model: model instance
        --dataloader
        --cost
        --optimizer
        --batch_size
        --lr_scheduler
        --device
    return :
        --loss
        --acc1
        --acc5
    """
    # training loop
    epoch_loss = 0.
    epoch_acc1 = 0.
    epoch_acc5 = 0.
    count = 0
    for batch_idx, (batch_data, batch_target) in enumerate(dataloader):
        batch_data = batch_data.to(device)
        batch_target = batch_target.to(device)

        optimizer.zero_grad()
        batch_pred = model(batch_data)
        loss = cost(batch_pred, batch_target)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        if batch_idx % batch_size == 0:
            count += 1
            acc1, acc5 = accuracy(batch_pred, batch_target, topk = (1, 5))
            epoch_loss += loss.item()
            epoch_acc1 += acc1.item()
            epoch_acc5 += acc5.item()

            print("\t\033[32m Batch INFO\033[0m Train Epoch: {:03d} Batch: {:05d}/{:05d} Loss: {:.4f} Top@1: {:.2f}%, Top@5: {:.2f}%."
                  .format(epoch + 1, batch_idx, len(dataloader), loss.item(), acc1.item(), acc5.item()))

    return epoch_loss / count, epoch_acc1 / count, epoch_acc5 / count


def train(**kwargs):
    """Train models"""
    opt.parse(**kwargs)
    opt.print()

    # device
    device, device_ids = setup_device(opt.use_gpu, opt.n_gpu)

    # Step 0 Decide the structure of the model
    # Step 1 Load the data set
    dataloader, num_classes = get_data_loader(data_root = opt.data_root, transform = opt.train_transform,
                                              dataset = opt.dataset, mode = 'train', batch_size = opt.batch_size, num_workers = 1)

    # Step 2 Reshape the inputs
    # Step 3 Normalize the inputs
    # Step 4 Initialize parameters
    # Step 5 Forward propagation(Vectorizarion/Activation functions)
    model = get_model(opt.model_name, 3, opt.input_size, num_classes)
    model.to(device)
    if device_ids is not None and len(device_ids) > 1:
        model = t.nn.DataParallel(model, device_ids = device_ids)

    # Step 6 Compute cost
    cost = t.nn.CrossEntropyLoss().to(device)

    # Step 7 Backward propagation(Vectorization/Activation functions gradients)
    # optimizer = t.optim.SGD(params = filter(lambda x : x.requires_grad, model.parameters()),
    #                         lr = opt.init_lr,
    #                         weight_decay = opt.weight_decay,
    #                         momentum = 0.9,
    #                         nesterov = True)
    if opt.optimizer == 'adamw':
        optimizer = t.optim.AdamW(params = filter(lambda x : x.requires_grad, model.parameters()), lr = opt.init_lr, weight_decay = opt.weight_decay, amsgrad = False)
    elif opt.optimizer == 'adam':
        optimizer = t.optim.Adam(params = filter(lambda x: x.requires_grad, model.parameters()), lr = opt.init_lr,
                                 weight_decay = opt.weight_decay, amsgrad = False)
    elif opt.optimizer == 'momentum':
        optimizer = t.optim.SGD(params = filter(lambda x: x.requires_grad, model.parameters()), lr = opt.init_lr,
                                weight_decay = opt.weight_decay, momentum = 0.9)
    else:
        raise Exception('No other optimizers!')
    # lr_scheduler = t.optim.lr_scheduler.OneCycleLR(optimizer = optimizer,
    #                                                max_lr = opt.init_lr,
    #                                                pct_start = opt.warm_steps / opt.train_steps,
    #                                                total_steps = opt.train_steps)

    # Step 8 Update parameters
    lr_scheduler = t.optim.lr_scheduler.MultiStepLR(optimizer, gamma = opt.gamma, milestones = opt.milestones)

    losses = []
    accs1 = []
    accs5 = []
    for epoch in tqdm(range(opt.epochs)):
        print('Epoch : %d / %d.' % (epoch + 1, opt.epochs))
        epoch_loss, epoch_acc1, epoch_acc5 = train_epoch(epoch, model, dataloader, cost, optimizer, opt.batch_size, lr_scheduler, device)
        losses.append(epoch_loss)
        accs1.append(epoch_acc1)
        accs5.append(epoch_acc5)

        lr_scheduler.step()

    print('Training is done!')

    filename = os.path.join('./checkpoints', f'trained_model_{opt.dataset}_{opt.input_size}_{opt.model_name}.pth')
    t.save(model, filename)
    vis_results_simple(losses, accs1, accs5, opt.model_name, opt.dataset, opt.input_size)