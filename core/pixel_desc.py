# Copyright 2022-present NAVER Corp.
# CC BY-NC-SA 4.0
# Available only for non-commercial use

from pdb import set_trace as bb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tvf

from core.conv_mixer import ConvMixer

norm_RGB = tvf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class PixelDesc (nn.Module):
    def __init__(self, path='checkpoints/PUMP-stytrf.pt'):
        super().__init__()
        state_dict = torch.load( path, map_location='cpu' )
        self.pixel_desc = ConvMixer(output_dim=128, hidden_dim=512, depth=7, patch_size=4, kernel_size=9).eval()
        self.pixel_desc.load_state_dict(state_dict)

    def configure(self, pipeline):
        # hot-update of the default HOG-based pipeline
        pipeline.__class__ = type(type(pipeline).__name__+'_Trained', (DescPipeline, type(pipeline)), {})
        return self

    def get_atomic_patch_size(self):
        return 4

    def forward(self, img, stride=1, offset=0):
        if img.ndim == 3: img = img[None]
        trf = torch.eye(3, device=img.device)

        desc = self.pixel_desc( img )
        desc = desc[..., offset::stride, offset::stride].contiguous() # free memory
        return desc, trf


class DescPipeline:
    def extract_descs(self, img1, img2, dtype=None):
        # this will rotate the image if needed
        img1, sca1 = self.demultiplex_img_trf(img1)
        img2, sca2 = self.demultiplex_img_trf(img2)

        # convert to float and normalize std
        fimg1, fimg2 = [norm_RGB(img.type(dtype)/255) for img in (img1, img2)]

        self.pixel_desc.type(fimg1.dtype)
        desc1, trf1 = self.pixel_desc(fimg1, stride=4, offset=2)
        desc2, trf2 = self.pixel_desc(fimg2)
        return (img1, img2), (desc1.type(dtype), desc2.type(dtype)), (sca1@trf1, sca2@trf2)

    def first_level(self, desc1, desc2, **kw):
        B, C, H, W = desc1.shape
        weights = desc1.permute(0, 2, 3, 1).view(H*W, C, 1, 1) # rearrange(desc1, '1 C H W -> (H W) C 1 1')
        corr = F.conv2d(desc2, weights, padding=0, bias=None)[0]
        norms = torch.ones(desc1.shape[-2:], device=corr.device)
        return corr.view(desc1.shape[-2:]+desc2.shape[-2:]), norms
