# Copyright 2022-present NAVER Corp.
# CC BY-NC-SA 4.0
# Available only for non-commercial use

from pdb import set_trace as bb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.trainer import backward


class MultiLoss (nn.Module):
    """ This functions handles both supervised and unsupervised samples.
    """
    def __init__(self, loss_sup, loss_unsup, alpha=0.3, inner_bw=True):
        super().__init__()
        assert 0 <= alpha
        self.alpha_sup = 1 # coef of self-supervised loss
        self.loss_sup = loss_sup

        self.alpha_unsup = alpha # coef of unsupervised loss
        self.loss_unsup = loss_unsup

        self.inner_bw = inner_bw

    def forward(self, desc1, desc2, homography, **kw):
        sl_sup, sl_unsup = split_batch_sup_unsup(homography, 512 if self.inner_bw else 8)

        inner_bw = self.inner_bw and self.training and torch.is_grad_enabled()
        if inner_bw: (desc1, desc1_), (desc2, desc2_) = pause_gradient((desc1,desc2))
        kw['desc1'], kw['desc2'], kw['homography'] = desc1, desc2, homography

        (sup_name, sup_loss) ,= self.loss_sup(backward_loss=inner_bw*self.alpha_sup, **{k:v[sl_sup] for k,v in kw.items()}).items()
        if inner_bw and sup_loss: sup_loss = backward(sup_loss) # backward to desc1 and desc2

        (uns_name, uns_loss) ,= self.loss_unsup(**{k:v[sl_unsup] for k,v in kw.items()}).items()
        uns_loss = self.alpha_unsup * uns_loss
        if inner_bw and uns_loss: uns_loss = backward(uns_loss) # backward to desc1 and desc2

        loss = sup_loss + uns_loss
        return {'loss':(loss, [(desc1_,desc1.grad),(desc2_,desc2.grad)]), sup_name:float(sup_loss), uns_name:float(uns_loss)}


def pause_gradient( objs ):
    return [(obj.detach().requires_grad_(True), obj) for obj in objs]
    

def split_batch_sup_unsup(homography, max_sup=512):
    # split batch in supervised / unsupervised
    i = int(torch.isfinite(homography[:,0,0]).sum()) # first ocurence
    sl_sup, sl_unsup = slice(0, min(i,max_sup)), slice(i, None)

    assert torch.isfinite(homography[sl_sup]).all(), 'batch is not properly sorted!'
    assert torch.isnan(homography[sl_unsup]).all(), 'batch is not properly sorted!'
    return sl_sup, sl_unsup
