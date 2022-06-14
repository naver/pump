# Copyright 2022-present NAVER Corp.
# CC BY-NC-SA 4.0
# Available only for non-commercial use

from pdb import set_trace as bb

import torch
import torch.nn as nn
import torch.nn.functional as F

""" From the ICLR22 paper: Patches are all you need
    https://openreview.net/pdf?id=TVHS5Y4dNvM
"""

class Residual(nn.Module):
    def __init__(self, fn, stride=1):
        super().__init__()
        self.fn = fn
        self.stride = stride

    def forward(self, x):
        s = slice(None,None,self.stride)
        return x[:,:,s,s] + self.fn(x)[:,:,s,s]


class ConvMixer (nn.Sequential):
    """ Modified ConvMixer with convolutional layers at the bottom.

    From the ICLR22 paper: Patches are all you need, https://openreview.net/pdf?id=TVHS5Y4dNvM
    """
    def __init__(self, output_dim, hidden_dim, 
                       depth=None, kernel_size=5, patch_size=8, group_size=1, 
                       preconv=1, faster=True, relu=nn.ReLU):

        assert kernel_size % 2 == 1, 'kernel_size must be odd'
        output_step = 1 + faster
        assert patch_size % output_step == 0, f'patch_size must be multiple of {output_step}'
        self.patch_size = patch_size

        hidden_dims = [hidden_dim//4]*preconv + [hidden_dim]*(depth+1)
        ops = [
            nn.Conv2d(3, hidden_dims[0], kernel_size=5, padding=2),
            relu(), 
            nn.BatchNorm2d(hidden_dims[0])]

        for _ in range(1,preconv):
            ops += [
                nn.Conv2d(hidden_dims.pop(0), hidden_dims[0], kernel_size=3, padding=1),
                relu(), 
                nn.BatchNorm2d(hidden_dims[0])]

        ops += [
            nn.Conv2d(hidden_dims.pop(0), hidden_dims[0], kernel_size=patch_size, stride=patch_size),
            relu(), 
            nn.BatchNorm2d(hidden_dims[0])]

        for idim, odim in zip(hidden_dims[0:], hidden_dims[1:]):
            ops += [Residual(nn.Sequential(
                        nn.Conv2d(idim, idim, kernel_size, groups=max(1,idim//group_size), padding=kernel_size//2),
                        relu(),
                        nn.BatchNorm2d(idim)
                    )),
                    nn.Conv2d(idim, odim, kernel_size=1),
                    relu(),
                    nn.BatchNorm2d(odim)]
        ops += [
            nn.Conv2d(odim, output_dim*(patch_size//output_step)**2, kernel_size=1),
            nn.PixelShuffle( patch_size//output_step ),
            nn.Upsample(scale_factor=output_step, mode='bilinear', align_corners=False)]

        super().__init__(*ops)

    def forward(self, img):
        assert img.ndim == 4
        B, C, H, W = img.shape
        desc = super().forward(img)
        return F.normalize(desc, dim=-3)


if __name__ == '__main__':
    net = ConvMixer3(128, 512, 7, patch_size=4, kernel_size=9)
    print(net)
    
    img = torch.rand(2,3,256,256)
    print('input.shape =', img.shape)
    desc = net(img)
    print('desc.shape =', desc.shape)
