import torch
import random
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder, MNIST
import warnings
import os
import torchvision
from os import listdir
import numpy as np
import time
from os.path import isfile, join
from PIL import Image
import sys
import math
from spikingjelly.datasets import cifar10_dvs


import data.utils as utils
from data.autoaugment import CIFAR10Policy, Cutout

warnings.filterwarnings('ignore')


# ## Change to your own data dir
DIR = {
    'CIFAR10': '/data/dataset/CIFAR10',
    'CIFAR100': '/data/dataset/CIFAR100',
    'CIFAR10DVS': '/data/dataset/CIFAR10DVS',
    'MNIST': '/data/dataset/datasets/',
    'ImageNet': '/data/dataset/datasets/',
    'Tiny-ImageNet': '/data/dataset/tiny-imagenet-200/'
}


# DIR['ImageNet']
# DIR['Tiny-ImageNet']



def build_cifar(cutout=False, autoaug=False, use_cifar10=True, download=False):
    """
    trans_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),  # autoaug=True,
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        Cutout(n_holes=1, length=16)  # cutout=False
        ])
    trans_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    # cifar100:
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
    """

    aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    if autoaug:
        aug.append(CIFAR10Policy())
    aug.append(transforms.ToTensor())

    if cutout:
        aug.append(Cutout(n_holes=1, length=16))

    if use_cifar10:
        aug.append(
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), )
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = CIFAR10(
            root=DIR['CIFAR10'], train=True, download=download, transform=transform_train)
        val_dataset = CIFAR10(
            root=DIR['CIFAR10'], train=False, download=download, transform=transform_test)

    else:
        aug.append(
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        )
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        train_dataset = CIFAR100(
            root=DIR['CIFAR100'], train=True, download=download, transform=transform_train)
        val_dataset = CIFAR100(
            root=DIR['CIFAR100'], train=False, download=download, transform=transform_test)

    return train_dataset, val_dataset



def load_data_cifar(use_cifar10=True, download=True, distributed=False, autoaug=False, cutout=False):
    train_dataset, test_dataset = build_cifar(
        autoaug=autoaug, cutout=cutout, use_cifar10=use_cifar10, download=download)
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        test_sampler = torch.utils.data.SequentialSampler(test_dataset)
    return train_dataset, test_dataset, train_sampler, test_sampler



def build_mnist(download=False):
    train_dataset = MNIST(
        root=DIR['MNIST'], train=True, download=download, transform=transforms.ToTensor())
    val_dataset = MNIST(
        root=DIR['MNIST'], train=False, download=download, transform=transforms.ToTensor())
    return train_dataset, val_dataset



class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]



def build_2aug_cifar():
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_set = CIFAR10(
        root=DIR['CIFAR10'],
        train=True,
        download=True,
        transform=TwoCropsTransform(train_transforms))
    return train_set



class DVSCifar10(Dataset):
    def __init__(self, root, train=True, transform=False, target_transform=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.resize = transforms.Resize(size=(48, 48))  # 48 48
        self.tensorx = transforms.ToTensor()
        self.imgx = transforms.ToPILImage()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data, target = torch.load(self.root + '/{}.pt'.format(index))
        # print(data.shape)
        # if self.train:
        new_data = []
        for t in range(data.size(-1)):
            new_data.append(self.tensorx(self.resize(self.imgx(data[..., t]))))
        data = torch.stack(new_data, dim=0)
        if self.transform:
            flip = random.random() > 0.5
            if flip:
                data = torch.flip(data, dims=(3,))
            off1 = random.randint(-5, 5)
            off2 = random.randint(-5, 5)
            data = torch.roll(data, shifts=(off1, off2), dims=(2, 3))

        if self.target_transform:
            target = self.target_transform(target)
        return data, target.long().squeeze(-1)

    def __len__(self):
        return len(os.listdir(self.root))



def build_dvscifar():
    # path = '/data_smr/dataset/cifar-dvs/'
    path = DIR['CIFAR10DVS']
    train_path = path + '/train'
    val_path = path + '/test'
    train_dataset = DVSCifar10(root=train_path, transform=True)
    val_dataset = DVSCifar10(root=val_path)
    return train_dataset, val_dataset


def build_dvscifar_(path, time_steps=4):
    train_path = path + '/train'
    val_path = path + '/test'
    dts_cache = 'dts_cache'
    train_path = os.path.join(DIR['CIFAR10DVS'], dts_cache, f'train_set_{time_steps}.pt')
    val_path = os.path.join(DIR['CIFAR10DVS'], dts_cache, f'test_set_{time_steps}.pt')
    train_dataset = DVSCifar10(root=train_path, transform=True)
    val_dataset = DVSCifar10(root=val_path)
    return train_dataset, val_dataset



def load_data_cifardvs(distributed=False, time_steps=4):
    path = DIR['CIFAR10DVS']
    train_dataset, test_dataset = build_dvscifar(path, time_steps)
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        test_sampler = torch.utils.data.SequentialSampler(test_dataset)
    return train_dataset, test_dataset, train_sampler, test_sampler



# ## split the train and test dataset.
def split_to_train_test_set(origin_dataset, train_ratio, num_classes, random_split=False):
    """
    origin_dataset: torch.utils.data.Dataset,
    train_ratio: float,
    num_classes: int,
    random_split: bool = False

    :param origin_dataset: the origin dataset
    :param train_ratio: split the ratio of the origin dataset as the train set
    :param num_classes: total classes number, e.g., ``10`` for the CIFAR10 dataset
    :param random_split: If ``False``, the front ratio of samples in each classes will
        be included in train set, while the reset will be included in test set.
        If ``True``, this function will split samples in each classes randomly. The randomness is controlled by
        ``numpy.randon.seed``
    :return: a tuple ``(train_set, test_set)``
    """
    label_idx = [[] for i in range(num_classes)]

    for i, item in enumerate(origin_dataset):
        y = item[1]
        if isinstance(y, np.ndarray) or isinstance(y, torch.Tensor):
            y = y.item()
        label_idx[y].append(i)
    train_idx = []
    test_idx = []
    if random_split:
        for i in range(num_classes):
            np.random.shuffle(label_idx[i])

    for i in range(num_classes):
        pos = math.ceil(label_idx[i].__len__() * train_ratio)
        train_idx.extend(label_idx[i][0: pos])
        test_idx.extend(label_idx[i][pos: label_idx[i].__len__()])

    return torch.utils.data.Subset(origin_dataset, train_idx), torch.utils.data.Subset(origin_dataset, test_idx)



# #### Using spikingjelly cifar10_dvs
def DVSCifar10_Dowload(time_steps):
    """
    A `frames_number_6_split_by_number6 folder will be created and
    contain the Frame data. (e.g. time_steps=6).
    Print a sample:

    > frame, label = origin_set[0]
    > print(frame.shape)

    The outputs are:
    (6, 2, 128, 128)

    The number of channels is 2.
    """
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=(48, 48)),
        transforms.RandomCrop(48, padding=4),
        transforms.RandomHorizontalFlip()
        ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=(48, 48))
    ])

    if not os.path.isdir(DIR['CIFAR10DVS']):
        utils.mkdir_p(DIR['CIFAR10DVS'])

    origin_set = cifar10_dvs.CIFAR10DVS(
        root=DIR['CIFAR10DVS'],
        data_type='frame',
        frames_number=time_steps,
        split_by='number',
        transform=transform_train)

    train_dataset, test_dataset = split_to_train_test_set(origin_set, train_ratio=0.9, num_classes=10, random_split=False)

    dts_cache = 'dts_cache'
    if not os.path.exists(os.path.join(DIR['CIFAR10DVS'], dts_cache)):
        os.makedirs(os.path.join(DIR['CIFAR10DVS'], dts_cache))
    train_set_pth = os.path.join(DIR['CIFAR10DVS'], dts_cache, f'train_set_{time_steps}.pt')
    test_set_pth = os.path.join(DIR['CIFAR10DVS'], dts_cache, f'test_set_{time_steps}.pt')
    torch.save(train_dataset, train_set_pth)
    torch.save(test_dataset, test_set_pth)
    return train_dataset, test_dataset


def load_DVSCifar10(time_steps, distributed=False):
    # ## TODO, check if this works. (this is new)
    dts_cache = 'dts_cache'
    train_set_pth = os.path.join(DIR['CIFAR10DVS'], dts_cache, f'train_set_{time_steps}.pt')
    test_set_pth = os.path.join(DIR['CIFAR10DVS'], dts_cache, f'test_set_{time_steps}.pt')
    if os.path.exists(train_set_pth):
        train_dataset = torch.load(train_set_pth)
        test_dataset = torch.load(test_set_pth)
        train_dataset, test_dataset
    else:
        train_dataset, test_dataset = DVSCifar10_Dowload(time_steps)
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        test_sampler = torch.utils.data.SequentialSampler(test_dataset)
    return train_dataset, test_dataset, train_sampler, test_sampler




def build_MNIST():
    train_dataset = torchvision.datasets.MNIST(
        root=DIR['MNIST'], train=True, download=True, transform=transforms.ToTensor())
    test_set = torchvision.datasets.MNIST(
        root=DIR['MNIST'], train=False, download=True, transform=transforms.ToTensor())
    return train_dataset, test_set



def _get_cache_path(filepath):
    import hashlib
    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path



def load_data_imagenet(cache_dataset, distributed):
    train_dir = os.path.join(DIR['ImageNet'], 'imagenet_train')
    val_dir = os.path.join(DIR['ImageNet'], 'imagenet_validation')

    # Data loading code
    print("Loading data: imagenet")
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    print("Loading training data")
    st = time.time()
    cache_path = _get_cache_path(train_dir)
    if cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print("Loading dataset_train from {}".format(cache_path))
        dataset, _ = torch.load(cache_path)
    else:
        dataset = torchvision.datasets.ImageFolder(
            train_dir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        if cache_dataset:
            print("Saving dataset_train to {}".format(cache_path))
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset, train_dir), cache_path)
    print("Took", time.time() - st)

    print("Loading validation data")
    cache_path = _get_cache_path(val_dir)
    if cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print("Loading dataset_test from {}".format(cache_path))
        dataset_test, _ = torch.load(cache_path)
    else:
        dataset_test = torchvision.datasets.ImageFolder(
            val_dir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))
        if cache_dataset:
            print("Saving dataset_test to {}".format(cache_path))
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset_test, val_dir), cache_path)

    print("Creating data loaders")
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler



def build_imagenet():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # root = '/data/dsk/ImageNet_dataset/ImageNet/ImageNet' # 4031.44
    # root = '/data_smr/dataset/ImageNet'

    train_root = os.path.join(DIR['ImageNet'], 'imagenet_train')
    val_root = os.path.join(DIR['ImageNet'], 'imagenet_validation')
    train_dataset = ImageFolder(
        train_root,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    )
    val_dataset = ImageFolder(
        val_root,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    )
    return train_dataset, val_dataset



def load_data_imagenet_(distributed=False):
    # ### Do not use it here.
    train_dataset, test_dataset = build_imagenet()
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        test_sampler = torch.utils.data.SequentialSampler(test_dataset)
    return train_dataset, test_dataset, train_sampler, test_sampler



class CIFAR10_DVS(Dataset):
    def __init__(self, dataset_path, n_steps):
        self.path = dataset_path
        self.n_steps = n_steps
        self.samples = []
        self.labels = []
        for i in range(10):
            sample_dir = dataset_path + '/' + str(i) + '/'
            for f in listdir(sample_dir):
                filename = join(sample_dir, f)
                if isfile(filename):
                    self.samples.append(filename)
                    self.labels.append(i)

    def __getitem__(self, index):
        data_path = self.samples[index]
        label = self.labels[index]
        tmp = np.genfromtxt(data_path, delimiter=',')

        data = np.zeros((2, 42, 42, self.n_steps))
        for c in range(2):
            for y in range(42):
                for x in range(42):
                    data[c, x, y, :] = tmp[c * 42 * 42 + y * 42 + x, :]

        data = torch.FloatTensor(data)
        data = data.permute([3,0,1,2])
        # label = tmp.shape
        return data, label

    def __len__(self):
        return len(self.samples)



class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.Train = train
        self.root_dir = root
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val/images")

        # os.path.join(DIR['Tiny-ImageNet'], 'train')
        # os.path.join(DIR['Tiny-ImageNet'], 'val/images')

        if (self.Train):
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.Train)

        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()

        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(self.train_dir, d))]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images;

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}


    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            images = [d for d in os.listdir(val_image_dir) if os.path.isfile(os.path.join(self.val_dir, d))]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        # self.idx_to_class = {i:self.val_img_to_class[images[i]] for i in range(len(images))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        self.images = []
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG")):
                        path = os.path.join(root, fname)
                        if Train:
                            item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            item = (path, self.class_to_tgt_idx[self.val_img_to_class[fname]])
                        self.images.append(item)

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        img_path, tgt = self.images[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, tgt



def get_cifar10_dvs(data_path, n_steps):
    print("loading CIFAR10 DVS")
    train_path = data_path + '/train'
    test_path = data_path + '/test'

    trainset = CIFAR10_DVS(train_path, n_steps)
    testset = CIFAR10_DVS(test_path, n_steps)
    return trainset, testset



def build_tiny_imagenet():
    # data_dir = '/data_smr/dataset/tiny_ImageNet/tiny-imagenet-200/'

    data_dir = DIR['Tiny-ImageNet']
    normalize = transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
    train_transforms = transforms.Compose([
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        normalize,
    ])
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset = TinyImageNet(data_dir, train=True,transform=train_transforms)
    val_dataset = TinyImageNet(data_dir, train=False,transform=val_transforms)
    return train_dataset, val_dataset




# simclr augmentation
def simclr_augmentation():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
    return train_transform, test_transform



class Contrast_Dataset(CIFAR10):
    def __init__(self, root, train, transform, download, n_steps):
        super(Contrast_Dataset, self).__init__(root, train, transform, target_transform=None, download=download)
        self.n_steps = n_steps

    def __getitem__(self, item):
        img, target = self.data[item], self.targets[item]
        img = Image.fromarray(img)
        img_pot = []
        if self.transform is not None:
            for t in range(self.n_steps):
                img_pot.append(self.transform(img))
        img_pot = torch.stack(img_pot, dim=0)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img_pot, target




if __name__ == '__main__':
    train_dataset, val_dataset = build_tiny_imagenet()
    print(len(train_dataset))
    print(len(val_dataset))
    print(train_dataset[0][0].shape)