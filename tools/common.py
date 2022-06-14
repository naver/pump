# Copyright 2022-present NAVER Corp.
# CC BY-NC-SA 4.0
# Available only for non-commercial use

import os
import torch
import numpy as np


def mkdir_for(file_path):
    dirname = os.path.split(file_path)[0]
    if dirname: os.makedirs(dirname, exist_ok=True)
    return file_path


def model_size(model):
    ''' Computes the number of parameters of the model 
    '''
    size = 0
    for weights in model.state_dict().values():
        size += np.prod(weights.shape)
    return size


class cudnn_benchmark:
    " context manager to temporarily disable cudnn benchmark "
    def __init__(self, activate ):
        self.activate = activate
    def __enter__(self):
        self.old_bm = torch.backends.cudnn.benchmark 
        torch.backends.cudnn.benchmark = self.activate
    def __exit__(self, *args):
        torch.backends.cudnn.benchmark = self.old_bm


def todevice(x, device, non_blocking=False):
    """ Transfer some variables to another device (i.e. GPU, CPU:torch, CPU:numpy).
    x:      array, tensor, or container of such.
    device: pytorch device or 'numpy'
    """
    if isinstance(x, dict):
        return {k:todevice(v, device) for k,v in x.items()}
    
    if isinstance(x, (tuple,list)):
        return type(x)(todevice(e, device) for e in x)

    if device == 'numpy':
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    elif x is not None:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.to(device, non_blocking=non_blocking)
    return x

def nparray( x ): return todevice(x, 'numpy')
def cpu( x ): return todevice(x, 'cpu')
def cuda( x ): return todevice(x, 'cuda')


def image( img, with_trf=False ):
    " convert a torch.Tensor to a numpy image (H, W, 3) "
    def convert_image(img):
        if isinstance(img, torch.Tensor):
            if img.dtype is not torch.uint8:
                img = img * 255
                if img.min() < -10:
                    img = img.clone()
                    for i, (mean, std) in enumerate(zip([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])):
                        img[i] *= std
                        img[i] += 255*mean
                img = img.byte()
            if img.shape[0] <= 3:
                img = img.permute(1,2,0)
        return img

    if isinstance(img, tuple):
        if with_trf:
            return nparray(convert_image(img[0])), nparray(img[1])
        else:
            img = img[0]
    return nparray(convert_image(img))


def image_with_trf( img ):
    return image(img, with_trf=True)

class ToTensor:
    " numpy images to float tensors "
    def __call__(self, x):
        assert x.ndim == 4 and x.shape[3] == 3
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        assert x.dtype == torch.uint8
        return x.permute(0, 3, 1, 2).float() / 255
