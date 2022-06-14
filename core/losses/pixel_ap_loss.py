# Copyright 2022-present NAVER Corp.
# CC BY-NC-SA 4.0
# Available only for non-commercial use

from pdb import set_trace as bb
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ap_loss import APLoss
from datasets.utils import applyh


class PixelAPLoss (nn.Module):
    """ Computes the pixel-wise AP loss:
        Given two images and ground-truth optical flow, computes the AP per pixel.
        
        feat1:  (B, C, H, W)   pixel-wise features extracted from img1
        feat2:  (B, C, H, W)   pixel-wise features extracted from img2
        aflow:  (B, 2, H, W)   absolute flow: aflow[...,y1,x1] = x2,y2
    """
    def __init__(self, sampler, nq=20, inner_bw=False, bw_step=256):
        nn.Module.__init__(self)
        self.aploss = APLoss(nq, min=0, max=1, euc=False)
        self.name = 'pixAP'
        self.sampler = sampler
        self.inner_bw = inner_bw
        self.bw_step = bw_step

    def loss_from_ap(self, ap, rel):
        return 1 - ap

    def forward(self, desc1, desc2, homography, backward_loss=None, **kw):
        if len(desc1) == 0: return dict(ap_loss=0)
        aflow = aflow_from_H(homography, desc1)
        descriptors = (desc1, desc2)
        scores, gt, msk, qconf = self.sampler(descriptors, kw.get('reliability'), aflow)

        # compute pixel-wise AP
        n = msk.numel()
        if n == 0: return 0
        scores, gt = scores.view(n,-1), gt.view(n,-1)

        backward_loss = backward_loss or self.inner_bw
        if self.training and torch.is_grad_enabled() and backward_loss: 
            # progressive loss computation and backward, low memory but slow
            scores_, qconf_ = scores, qconf if qconf is not None else scores.new_ones(msk.shape)
            scores = scores.detach().requires_grad_(True)
            qconf = qconf_.detach().requires_grad_(True) 
            msk = msk.ravel()

            loss = 0
            for i in range(0, n, self.bw_step):
                sl = slice(i, i+self.bw_step)
                ap = self.aploss(scores[sl], gt[sl])
                pixel_loss = self.loss_from_ap(ap, qconf.ravel()[sl] if qconf is not None else None)
                l = backward_loss / msk.sum() * pixel_loss[msk[sl]].sum()
                loss += float(l)
                l.backward() # cumulate gradient
            loss = (loss, [(scores_,scores.grad)])
            if qconf_.requires_grad: loss[1].append((qconf_,qconf.grad))

        else:
            ap = self.aploss(scores, gt).view(msk.shape)
            pixel_loss = self.loss_from_ap(ap, qconf)
            loss = pixel_loss[msk].mean()

        return dict(ap_loss=loss)


def make_grid(B, H, W, device ):
    b = torch.arange(B, device=device).view(B,1,1).expand(B,H,W)
    y = torch.arange(H, device=device).view(1,H,1).expand(B,H,W)
    x = torch.arange(W, device=device).view(1,1,W).expand(B,H,W)
    return b.view(B,H*W), torch.stack((x,y),dim=-1).view(B,H*W,2)


def aflow_from_H( H_1to2, feat1 ):
    B, _, H, W = feat1.shape
    b, pos1 = make_grid(B,H,W, feat1.device)
    pos2 = applyh(H_1to2, pos1.float())
    return pos2.view(B,H,W,2).permute(0,3,1,2)
