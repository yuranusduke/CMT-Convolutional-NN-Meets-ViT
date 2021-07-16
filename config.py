"""
This file contains configurations

Created by Kunhong Yu
Date: 2021/07/14
"""
import torch as t
import torchvision as tv

class Config(object):
    """
    Args :
        --data_root: default is './data/'
        --dataset: data set name, default is 'cifar10', also support 'cifar100' and 'imagenet'(which we don't experiment due to limited computation) and 'tiny_imagenet' and 'flowers', also 'cars'
        --input_size: default is 160, also support 192, 224, 256, 288

        --model_name: default is 'cmt_ti', also support 'cmt_xs', 'cmt_s', 'cmt_b'
        --use_gpu: True as default
        --n_gpu: default is 1

        --batch_size: default is 32
        --epochs: default is 160
        --optimizer: default is 'adamw', also support 'adam'/'momentum'
        --init_lr: default is 1e-5
        --gamma: learning rate decay rate, default is 0.2
        --milestones: we use steplr decay, default is [30, 60, 90, 120, 150]
        --weight_decay: default is 1e-5
    """

    ############
    #    Data  #
    ############
    data_root = './data/'
    dataset = 'cifar10'
    input_size = 160

    ############
    #   Model  #
    ############
    model_name = 'cmt_ti'
    use_gpu = True
    n_gpu = 1

    ############
    #   Train  #
    ############
    batch_size = 32
    epochs = 160
    optimizer = 'adamw'
    init_lr = 1e-5
    gamma = 0.2
    milestones = [30, 60, 90, 120, 150]
    weight_decay = 1e-5

    def parse(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                print(k + 'does not exist, will be added!')

            setattr(self, k, v)

        if getattr(self, 'dataset') == 'cifar10' or getattr(self, 'dataset') == 'cifar100':
            self.train_transform = tv.transforms.Compose([
                tv.transforms.RandomCrop(32, padding = 4),
                tv.transforms.RandomHorizontalFlip(),
                tv.transforms.RandomRotation(15),
                tv.transforms.Resize((getattr(self, 'input_size'), getattr(self, 'input_size'))),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(0.5, 0.5)
            ])

            self.test_transform = tv.transforms.Compose([
                tv.transforms.Resize((getattr(self, 'input_size'), getattr(self, 'input_size'))),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(0.5, 0.5)
            ])

        elif getattr(self, 'dataset').count('imagenet'):
            self.train_transform = tv.transforms.Compose([
                tv.transforms.RandomHorizontalFlip(),
                tv.transforms.Resize((getattr(self, 'input_size'), getattr(self, 'input_size'))),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

            self.test_transform = tv.transforms.Compose([
                tv.transforms.Resize((getattr(self, 'input_size'), getattr(self, 'input_size'))),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

        elif getattr(self, 'dataset').count('flower'):
            self.train_transform = tv.transforms.Compose([
                tv.transforms.Resize((getattr(self, 'input_size'), getattr(self, 'input_size'))),
                tv.transforms.RandomHorizontalFlip(),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

            self.test_transform = tv.transforms.Compose([
                tv.transforms.Resize((getattr(self, 'input_size'), getattr(self, 'input_size'))),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

        elif getattr(self, 'dataset').count('cars'):
            self.train_transform = tv.transforms.Compose([
                tv.transforms.Resize((getattr(self, 'input_size'), getattr(self, 'input_size'))),
                tv.transforms.RandomHorizontalFlip(),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

            self.test_transform = tv.transforms.Compose([
                tv.transforms.Resize((getattr(self, 'input_size'), getattr(self, 'input_size'))),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

        else:
            raise Exception('You may add new transform in config.py file!')


    def print(self):
        for k, _ in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k, '...', getattr(self, k))
