"""
This function is used to evaluate model

Created by Kunhong Yu
Date: 2021/07/15
"""
import os
import torch as t
from config import Config
from utils import setup_device, get_data_loader, eval_main
from datetime import datetime

opt = Config()

def test(**kwargs):
    """Test models"""
    opt.parse(**kwargs)
    opt.print()

    # device
    device, device_ids = setup_device(opt.use_gpu, opt.n_gpu)

    # data
    dataloader, num_classes = get_data_loader(opt.data_root, opt.test_transform, dataset = opt.dataset, mode = 'test', batch_size = opt.batch_size, num_workers = 1)

    # model
    filename = os.path.join('./checkpoints', f'trained_model_{opt.dataset}_{opt.input_size}_{opt.model_name}.pth')
    model = t.load(filename)

    # send model to device
    model = model.to(device)
    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.2fM" % (total / 1e6))

    if len(device_ids) > 1:
        model = t.nn.DataParallel(model, device_ids = device_ids)

    # starting evaluation
    print("Starting evaluation")

    model.eval()
    with t.no_grad():
        _, acc1, acc5 = eval_main(model, dataloader, device)

    eval_string = "Evaluation of model {:s} on dataset {:s} with input size {}, Top@1: {:.4f}%, Top@5: {:.4f}%.".format(opt.model_name, opt.dataset, opt.input_size, acc1, acc5)
    print('\n\033[34m' + eval_string + '\033[0m')

    filename = os.path.join('./results/', 'saved_test_results.txt')
    string = '*' * 40 + str(datetime.now()) + '*' * 40 + '\n'
    string += "Model capacity : Total params: %.2fM" % (total / 1e6) + '\n'
    for k, _ in opt.__class__.__dict__.items():
        v = getattr(opt, k)
        if not k.startswith('__'):
            string += str(k) + ' --> ' + str(v) + '\n'

    string += '\n' + eval_string + '\n\n'

    with open(filename, 'a+') as f:
        f.write(string)
        f.flush()

    print('Testing is done!')