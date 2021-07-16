"""
Utilities functions

Created by Kunhong Yu
Date: 2021/07/15
"""

import torch as t
import torchvision as tv
import os
import shutil
import sys
import numpy as np
import matplotlib.pyplot as plt
from model import CMT_Ti, CMT_XS, CMT_S, CMT_B
from PIL import Image

##############################
#           Data             #
##############################
def get_cifar(root, dataset, transform, mode = 'train'):
    """Get cifar data set
    Args :
        --root: data root
        --dataset: dataset name
        --transform: transformation
        --mode: 'train'/'test'
    """
    assert dataset.count('cifar')

    if dataset == 'cifar10':
        dataset = tv.datasets.CIFAR10(root = root,
                                      download = True,
                                      train = True if mode == 'train' else False,
                                      transform = transform)

    elif dataset == 'cifar100':
        dataset = tv.datasets.CIFAR100(root = root,
                                       download = True,
                                       train = True if mode == 'train' else False,
                                       transform = transform)

    else:
        raise Exception('No other data sets!')

    return dataset


class ImageNetDataLoader(t.utils.data.DataLoader):
    """Define ImageNet data loader"""

    def __init__(self, data_root, transform, mode = 'train', batch_size = 16, num_workers = 1):
        """
        Args :
            --data_root
            --transform
            --mode: 'train' or 'test'
            --batch_size: default is 16
            --num_workers: default is 1
        """

        if mode == 'train':
            # reset first
            self.redir_tinyIN(data_root, 'train')

        else:
            # reset first
            self.redir_tinyIN(data_root, 'val')

        self.dataset = tv.datasets.ImageFolder(root = os.path.join(data_root, 'train' if mode == 'train' else 'val'), transform = transform)
        super(ImageNetDataLoader, self).__init__(
            dataset = self.dataset,
            batch_size = batch_size,
            shuffle = True if mode == 'train' else False,
            num_workers = num_workers,
            drop_last = False)

    def redir_tinyIN(self, data_root, mode):
        """Redir data for tiny_ImageNet
        Args :
            --data_dir, './data/tinyIN'
            --mode: 'train' or 'val'
        """
        if mode == 'train':
            dirs = os.path.join(data_root, mode)
            files = os.listdir(dirs)
            files = [os.path.join(dirs, file) for file in files]
            for file in files:
                fds = os.listdir(file)
                for fd in fds:
                    if fd.count('txt'):
                        os.remove(os.path.join(file, fd))
                        break

                if not 'images' in fds:
                    break
                all_imgs = os.listdir(os.path.join(file, 'images'))
                all_imgs = list(filter(lambda x: x.endswith('JPEG'), all_imgs))
                names = all_imgs
                all_imgs = [os.path.join(file, 'images', f) for f in all_imgs]

                for img, name in zip(all_imgs, names):
                    sys.stdout.write('\r>>ReDir class %s for %s : %s.' % (file, 'train', img))
                    sys.stdout.flush()

                    dst = os.path.join(file, name)
                    shutil.move(img, dst)

                shutil.rmtree(os.path.join(file, 'images'))

                print()
        elif mode == 'val':
            dirs = os.path.join(data_root, mode)
            dirs_imgs = os.path.join(dirs, 'images')
            if not os.path.exists(dirs_imgs):
                pass
            else:
                all_imgs = os.listdir(dirs_imgs)
                all_imgs = list(filter(lambda x : x.endswith('JPEG'), all_imgs))
                names = all_imgs
                all_imgs = [os.path.join(dirs_imgs, img) for img in all_imgs]
                all_imgs_dict = dict(zip(names, all_imgs))
                with open(os.path.join(dirs, 'val_annotations.txt'), 'r') as f:
                    count = 0
                    for line in f:
                        line = line.strip()
                        line = line.split('	')
                        name = line[0]
                        label = line[1]
                        sys.stdout.write('\r>>ReDir class %s for %s : %s.' % (label, 'val', name))
                        sys.stdout.flush()
                        label_dir = os.path.join(dirs, label)
                        if not os.path.exists(label_dir):
                            os.makedirs(label_dir)

                        shutil.move(all_imgs_dict[name], os.path.join(label_dir, name))

                        count += 1

                    shutil.rmtree(dirs_imgs)
                print()
        else:
            raise Exception('No other mode!')


class Oxford102Flowers(t.utils.data.Dataset):
    """Define Oxford 102 Flowers data set"""

    def __init__(self, data_root, transform, mode):
        """
        Args :
            --data_root: data root, './data/oxford-102-flowers'
            --transform: tv.transforms.Compose instance
            --mode: 'train' or 'test'
        """
        super(Oxford102Flowers, self).__init__()

        all_imgs_dir = os.path.join(data_root, 'jpg')
        all_imgs = os.listdir(all_imgs_dir)
        all_imgs = list(filter(lambda x : x.endswith('jpg'), all_imgs))
        names = all_imgs
        all_imgs = [os.path.join(all_imgs_dir, img) for img in all_imgs]
        all_imgs_dict = dict(zip(names, all_imgs))

        self.all_imgs_dict = all_imgs_dict

        self.transform = transform
        self.mode = mode

        labels = {}
        with open(os.path.join(data_root, mode + '.txt'), 'r') as f:
            for line in f:
                line = line.strip()
                line = line.split(' ')
                name, label = line[0], int(line[1])
                labels[name] = label

        self.labels = labels

    def __getitem__(self, index):
        name = list(self.labels.keys())[index]
        label = self.labels[name]
        img_dir = self.all_imgs_dict[name.split('/')[1]]
        img = Image.open(img_dir)

        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.labels)


class StanfordCars(t.utils.data.Dataset):
    """Define Stanford Cars data set"""

    def __init__(self, data_root, transform, mode):
        """
        Args :
            --data_root: data root, './data/stanford_cars'
            --transform: tv.transforms.Compose instance
            --mode: 'train' or 'test'
        """
        super(StanfordCars, self).__init__()

        self.data_root = data_root
        self.transform = transform
        self.mode = mode

        imgs_dir = os.path.join(data_root, 'car_ims')
        names = os.listdir(imgs_dir)
        names = list(filter(lambda x : x.endswith('jpg'), names))
        img_files = [os.path.join(imgs_dir, file) for file in names]
        img_files = dict(zip(names, img_files))

        self.train_label_mapping = {}
        self.test_label_mapping = {}
        self.train_names = []
        self.test_names = []
        self.train_img_files = []
        self.test_img_files = []
        with open(os.path.join(data_root, 'mat2txt.txt'), 'r') as f:
            for line in f:
                line = line.strip()
                line = line.split(' ')

                name, label, is_test = line[1], line[2], line[3]
                if is_test == '1' : # test
                    self.test_label_mapping[name] = label
                    self.test_names.append(name)
                    name = name.split('/')[-1]
                    self.test_img_files.append(img_files[name])
                else:
                    self.train_label_mapping[name] = label
                    self.train_names.append(name)
                    name = name.split('/')[-1]
                    self.train_img_files.append(img_files[name])

        if self.mode == 'train':
            self.names = self.train_names
            self.img_files = self.train_img_files
            self.labels = self.train_label_mapping
        else:
            self.names = self.test_names
            self.img_files = self.test_img_files
            self.labels = self.test_label_mapping

    def __getitem__(self, index):
        name = self.names[index]
        name = name.split('/')[-1]
        img_file = self.img_files[index]
        label = self.labels[index]
        label = int(label)

        img = Image.open(img_file)
        if self.transform:
            img = self.transform(img)

        return img, label


def get_data_loader(data_root, transform, dataset = 'cifar10', mode = 'train', batch_size = 16, num_workers = 1):
    """This function is used to get data loader
     Args :
        --data_root
        --transform
        --dataset: 'cifar100' and 'imagenet' and 'tiny_imagenet' and 'flowers', also 'cars',  'cifar10' as default
        --mode: 'train' or 'test'
        --batch_size: default is 16
        --num_workers: default is 1
    return :
        --dataloader: got dataloader
        --num_classes
    """

    if dataset.count('cifar'):
        data = get_cifar(root = data_root, dataset = dataset, transform = transform, mode = mode)
        dataloader = t.utils.data.DataLoader(data,
                                             shuffle = True if mode == 'train' else False,
                                             batch_size = batch_size, num_workers = num_workers,
                                             drop_last = False)
        if dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100

    elif dataset.count('imagenet'):
        dataloader = ImageNetDataLoader(data_root = data_root, transform = transform, mode = mode, batch_size = batch_size,
                                        num_workers = num_workers)
        if dataset.count('tiny'):
            num_classes = 200
        else:
            num_classes = 1000

    elif dataset.count('flower'):
        dataset = Oxford102Flowers(data_root = data_root, transform = transform, mode = mode)
        dataloader = t.utils.data.DataLoader(dataset, shuffle = True if mode == 'train' else False, batch_size = batch_size,
                                             num_workers = num_workers, drop_last = False)
        num_classes = 102

    elif dataset.count('cars'):
        dataset = StanfordCars(data_root = data_root, transform = transform, mode = mode)
        dataloader = t.utils.data.DataLoader(dataset, shuffle = True if mode == 'train' else False, batch_size = batch_size,
                                             num_workers = num_workers, drop_last = False)
        num_classes = 196

    else:
        raise Exception('No other data sets!')

    return dataloader, num_classes


#############################
#           Model           #
#############################
def get_model(model_name, in_channels = 3, input_size = 224, num_classes = 1000):
    """Get model
    Args ：
        --model_name: model's name
        --in_channels: default is 3
        --input_size: default is 224
        --num_classes: default is 1000 for ImageNet
    return :
        --model: model instance
    """

    string = model_name
    if model_name == 'cmt_ti':
        model = CMT_Ti(in_channels = in_channels, input_size = input_size, num_classes = num_classes)
    elif model_name == 'cmt_xs':
        model = CMT_XS(in_channels = in_channels, input_size = input_size, num_classes = num_classes)
    elif model_name == 'cmt_s':
        model = CMT_S(in_channels = in_channels, input_size = input_size, num_classes = num_classes)
    elif model_name == 'cmt_b':
        model = CMT_B(in_channels = in_channels, input_size = input_size, num_classes = num_classes)
    else:
        raise Exception('No other models!')

    print(string + ': \n', model)
    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.2fM" % (total / 1e6))

    return model


#############################
#           Train           #
#############################
def setup_device(use_gpu, n_gpu):
    """Set devices
    Args :
        --use_gpu: True or False
        --n_gpu: how many gpus to be used
    return :
        --device: device
        --list_ids
    """
    n_av_gpu = t.cuda.device_count()

    if use_gpu:
        n_gpu_use = n_av_gpu
        if n_gpu > 0 and n_av_gpu == 0:
            print("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu > n_av_gpu:
            print("Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu

        device = t.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
    else:
        device = t.device('cpu')
        list_ids = None

    return device, list_ids


def vis_results_simple(losses, accs1, accs5, model_name, dataset, input_size):
    """Simply visualize results using matplotlib
    Args :
        --losses: loss in the form of list
        --accs1: acc1 in the form of list
        --accs5: acc5 in the form of list
        --model_name
        --dataset
        --input_size
    """

    f, ax = plt.subplots(1, 3, figsize = (25, 5))
    f.suptitle('Train statistics')
    ax[0].plot(range(len(losses)), losses, label = 'train loss')
    ax[0].set_xlabel('Steps')
    ax[0].set_ylabel('Loss')
    ax[0].grid(True)
    ax[0].set_title('Training loss')
    ax[0].legend(loc = 'best')

    ax[1].plot(range(len(accs1)), accs1, label = 'train top@1')
    ax[1].set_xlabel('Steps')
    ax[1].set_ylabel('Top@1')
    ax[1].grid(True)
    ax[1].set_title('Training top@1')
    ax[1].legend(loc = 'best')

    ax[2].plot(range(len(accs5)), accs5, label = 'train top@5')
    ax[2].set_xlabel('Steps')
    ax[2].set_ylabel('Top@5')
    ax[2].grid(True)
    ax[2].set_title('Training top@5')
    ax[2].legend(loc = 'best')

    filename = os.path.join('./results/', f'trained_model_{dataset}_{input_size}_{model_name}.png')
    plt.savefig(filename)
    plt.close()


#############################
#         Metrics           #
#############################
def accuracy(output, target, topk = (1,)):
    """Computes the precision@k for the specified values of k
    Args : 
        --output: output tensor
        --target: gt labels
        --topk: default is (1,)
    return :
        --res: in the form of Python list
    """""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k / batch_size * 100.0)

    return res


#############################
#        Evaluation         #
#############################
def eval_main(model, dataloader, device, cost = None):
    """Test model in only one epoch
    Args :
        --model: learning model
        --dataloader: data loader
        --cost: learning cost
        --device
    return :
        --loss
        --acc1
        --acc5
    """
    losses = []
    acc1s = []
    acc5s = []
    # test loop
    with t.no_grad():
        for batch_idx, (batch_data, batch_target) in enumerate(dataloader):
            sys.stdout.write('\r>>Testing batch %d / %d.' % (batch_idx + 1, len(dataloader)))
            sys.stdout.flush()
            batch_data = batch_data.to(device)
            batch_target = batch_target.to(device)

            batch_pred = model(batch_data)
            if cost:
                loss = cost(batch_pred, batch_target)
                losses.append(loss.item())
            acc1, acc5 = accuracy(batch_pred, batch_target, topk = (1, 5))

            acc1s.append(acc1.item())
            acc5s.append(acc5.item())

    loss = 0.
    if cost:
        loss = np.mean(losses)
    acc1 = np.mean(acc1s)
    acc5 = np.mean(acc5s)

    print()

    return loss, acc1, acc5



# Run this script independently to preprocess Stanford Cars dataset
if __name__ == '__main__':
    # import scipy.io
    #
    # data = scipy.io.loadmat('./data/stanford_cars/cars_annos.mat')
    # class_names = data['class_names']
    # f_class = open('./data/stanford_cars/label_map.txt', 'w')
    #
    # num = 1
    # for j in range(class_names.shape[1]):
    #     class_name = str(class_names[0, j][0]).replace(' ', '_')
    #     print(num, class_name)
    #     f_class.write(str(num) + ' ' + class_name + '\n')
    #     num = num + 1
    # f_class.close()

    # map name to label and test indicator
    import scipy.io

    data = scipy.io.loadmat('./data/stanford_cars/cars_annos.mat')
    annotations = data['annotations']
    f_train = open('./data/stanford_cars/mat2txt.txt', 'w')

    num = 1
    for i in range(annotations.shape[1]):
        name = str(annotations[0, i][0])[2:-2]
        test = int(annotations[0, i][6])
        clas = int(annotations[0, i][5])

        name = str(name)
        clas = str(clas)
        test = str(test)
        f_train.write(str(num) + ' ' + name + ' ' + clas + ' ' + test + '\n')
        num = num + 1

    f_train.close()
