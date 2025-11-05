import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import WeightedRandomSampler

from skimage.io import imread, imsave
from torch.utils.data.distributed import DistributedSampler

from .transforms_factory import build_transforms
from .process_csv import merge_multiple_dataset, get_sample_weight
from .black_cut import process_fundus, pad_square

class DistributedProxySampler(DistributedSampler):

    def __init__(self, sampler, num_replicas=None, rank=None):
        '''
        description: use DistributedSampler to package randomsampler
        sampler : WeightedRandomSampler
        num_replicas: number of using gpu
        rank: this gpu id
        '''
        super(DistributedProxySampler, self).__init__(
            sampler, num_replicas=num_replicas, rank=rank, shuffle=False)
        self.sampler = sampler

    def __iter__(self):
        # deterministically shuffle based on epoch
        torch.manual_seed(self.epoch)
        indices = list(self.sampler)
        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        if len(indices) != self.total_size:
            raise RuntimeError("{} vs {}".format(len(indices), self.total_size))
        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        if len(indices) != self.num_samples:
            raise RuntimeError("{} vs {}".format(len(indices), self.num_samples))
        return iter(indices)


class BasicDataset(Dataset):
    def __init__(self, data, transforms_cfg,data_cfg, mode):
        self.img_paths = data['path'].to_numpy()
        self.labels = data['label'].to_numpy()
        self.more_col = data_cfg['more_col']
        if self.more_col!=None:
            self.img_infor = data[self.more_col].to_numpy()
        print('##############################',self.img_infor)
        if np.issubdtype(self.labels.dtype,np.float64):
            self.labels = self.labels.astype(np.float32)
            print('Convert the data type of label: np.float64 -> np.float32')

        self.tg_size = transforms_cfg['target_size']
        self.num_images = len(self.img_paths)
        self.pre_process = data_cfg.get('preprocessing',False)
        self.aug_out_dir = data_cfg.get('aug_out_dir',"")
        self.pad_square = data_cfg['pad_square']

        if isinstance(self.tg_size, int):
            self.tg_size = [self.tg_size, self.tg_size]
            transforms_cfg['target_size'] = self.tg_size

        self.aug = build_transforms(mode, transforms_cfg)
        print(f'Creating dataset with {self.num_images} examples')

    def __len__(self):
        return self.num_images

    def __getitem__(self, i):
        img_path = self.img_paths[i]
        label = int(self.labels[i])
        if self.more_col!=None:
            img_infor = self.img_infor[i]
        try:
            img = imread(img_path).astype(np.uint8)
            if self.pad_square:
                img = pad_square(img)
            assert not img is None
            assert img.shape[0] > 10 and img.shape[1] > 10 and img.shape[2] == 3 and img.ndim == 3
        except Exception as err_infor:
            print(img_path)
            print(err_infor)
            return self.__getitem__(np.random.randint(self.num_images))
        if self.pre_process:
            try:
                img = process_fundus(img, out_radius=500, square_pad=True)
            except Exception as err_infor:
                print(f'process err. image path{img_path}, {err_infor}')
                img = img
        augmented = self.aug(image=img)
        image = augmented['image']
        # for debug, view data augmentation images
        if self.aug_out_dir:
            os.makedirs(self.aug_out_dir, exist_ok=True)
            basename = os.path.basename(img_path)
            new_name = os.path.join(self.aug_out_dir, basename)
            img_id, suffix = os.path.splitext(new_name)
            original_name = img_id+'_original_' + suffix
            image_aug = (image.copy() + 1) / 2 * 255
            imsave(new_name, image_aug.astype(np.uint8))
            imsave(original_name, img)
        image = np.transpose(image, (2, 0, 1))
        # assert image.shape == (3, self.tg_size[0], self.tg_size[1])
        if self.more_col!=None:
            img_infor = torch.Tensor(img_infor)
        else:
            img_infor = None
        return {'image': torch.Tensor(image), 'label': label, 'path': img_path,'img_infor':  img_infor}


def train_loader(csv_infor, transforms_cfg, data_cfg, num_replicas=1, rank=0):
    """
    Args:
        csv_infor ( [ dict ]):  Contains multiple data information .  The keys of the dict is : 'csv_path' 'img_dir''path_col''label_col' 'scale' 'label_map'
        transforms_cfg (dict):
            transform_define(str): Name of the definition transform
            target_size (int or list): shape of input image
            aug_prob (float):data augmentation probability .range [0,1] Defaults to 0.5.
            aug_m (int): data augmentation magnitude . range [0,3], Defaults to 2.
        data_cfg (dict):
            num_samples (int): the data number of one epoch.
            replacement (bool): . Defaults to False.
            sample_distribution (float): [description]. Defaults to 1.0,uniform distribution.
            pad_square: (bool): the way image will be resized, False for directly resize, True for fill-with-black to square
            preprocessing:(bool): Whether preprocessing is required, such as cutting black edges
            sample_seed(int): Seed for sample csv . Pass the "epoch" here so that the sampling results are consistent across every DDP process 
            batch_size (int):  
            num_workers (int):  
        num_replicas (int, optional): number of using gpu. Defaults to 1.
        rank (int, optional):  gpu id. Defaults to 0.
    Returns:
        training data loader
    """
    replacement = data_cfg['data_replacement']
    data = merge_multiple_dataset(csv_infor, data_cfg['sample_seed'], data_cfg)

    final_weight = get_sample_weight(data['data_weight'], data_cfg['sample_distribution'])
    if isinstance(data_cfg['num_samples'] ,float ) :
        num_samples = int(data.shape[0] * data_cfg['num_samples'] )
    else:
        num_samples = data_cfg['num_samples'] 
    sampler = WeightedRandomSampler(weights=final_weight, num_samples= num_samples , replacement=replacement)
    print('Here the sampler is num_replicas', num_replicas)
    if num_replicas > 1:
        sampler = DistributedProxySampler(sampler, num_replicas=num_replicas, rank=rank)
        print(f'# GPU rank ={rank}, num_dataset={len(sampler)}')

    dataset = BasicDataset(data, transforms_cfg,data_cfg, mode='train')
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=data_cfg['batch_size'],
                                         shuffle=False, pin_memory=False, sampler=sampler, num_workers=data_cfg['num_workers'])
    # print(f'## transforms_cfg: {transforms_cfg}. data_cfg: {data_cfg}')
    print('Here the batch size is'), data_cfg['batch_size']
    print('the data loader iter is', len(loader))
    return loader


def _basic_loader(data, transforms_cfg, data_cfg, num_replicas=1, rank=0):
    """
    Args:
        paths (np.ndarray): 
        labels (np.ndarray): 
        transforms_cfg (dict): 'define_name', 'target_size'
        data_cfg (dict): key :pad_square,batch_size, num_workers,preprocessing
        num_replicas (int, optional): [description]. Defaults to 1.
        rank (int, optional): [description]. Defaults to 0.
    """

    dataset = BasicDataset(data, transforms_cfg,data_cfg,mode='test')
    if num_replicas > 1:
        sampler = range(data.shape[0])
        sampler = DistributedProxySampler(sampler, num_replicas=num_replicas, rank=rank)
        print(f'# GPU rank ={rank}, num_dataset={len(sampler)}')
    else:
        sampler = None
    loader = torch.utils.data.DataLoader(dataset, batch_size=data_cfg['batch_size'], shuffle=False, pin_memory=False,
                                         sampler=sampler, num_workers=data_cfg['num_workers'])
    return loader


def basic_loader(csv_infor, transforms_cfg, data_cfg, num_replicas=1, rank=0):
    """
    data_loader for validation and test

    """
    data  = merge_multiple_dataset(csv_infor, data_cfg  =data_cfg)
    return _basic_loader(data, transforms_cfg, data_cfg, num_replicas, rank)
