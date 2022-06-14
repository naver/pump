# Copyright 2022-present NAVER Corp.
# CC BY-NC-SA 4.0
# Available only for non-commercial use

from pdb import set_trace as bb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NghSampler (nn.Module):
    """ Given dense feature maps and pixel-dense flow, 
        compute a subset of all correspondences and return their scores and labels.
    
    Distance to GT =>  0 ... pos_d ... neg_d ... ngh
    Pixel label    =>  + + + + + + 0 0 - - - - - - -

    Subsample on query side: if > 0, regular grid
                                < 0, random points 
    In both cases, the number of query points is = W*H/subq**2
    """
    def __init__(self, ngh, subq=-8, subd=1, pos_d=2, neg_d=4, border=16, subd_neg=-8):
        nn.Module.__init__(self)
        assert 0 <= pos_d < neg_d <= (ngh if ngh else 99)
        self.ngh = ngh
        self.pos_d = pos_d
        self.neg_d = neg_d
        assert subd <= ngh or ngh == 0
        assert subq != 0
        self.sub_q = subq
        self.sub_d = subd
        self.sub_d_neg = subd_neg
        if border is None: border = ngh
        assert border >= ngh, 'border has to be larger than ngh'
        self.border = border
        self.precompute_offsets()

    def precompute_offsets(self):
        pos_d2 = self.pos_d**2
        neg_d2 = self.neg_d**2
        rad2 = self.ngh**2
        rad = (self.ngh//self.sub_d) * self.ngh # make an integer multiple
        pos = []
        neg = []
        for j in range(-rad, rad+1, self.sub_d):
          for i in range(-rad, rad+1, self.sub_d):
            d2 = i*i + j*j
            if d2 <= pos_d2:
                pos.append( (i,j) )
            elif neg_d2 <= d2 <= rad2: 
                neg.append( (i,j) )

        self.register_buffer('pos_offsets', torch.LongTensor(pos).view(-1,2).t())
        self.register_buffer('neg_offsets', torch.LongTensor(neg).view(-1,2).t())

    def gen_grid(self, step, aflow):
        B, two, H, W = aflow.shape
        dev = aflow.device
        b1 = torch.arange(B, device=dev)
        if step > 0:
            # regular grid
            x1 = torch.arange(self.border, W-self.border, step, device=dev)
            y1 = torch.arange(self.border, H-self.border, step, device=dev)
            H1, W1 = len(y1), len(x1)
            shape = (B, H1, W1)
            x1 = x1[None,None,:].expand(B,H1,W1).reshape(-1)
            y1 = y1[None,:,None].expand(B,H1,W1).reshape(-1)
            b1 = b1[:,None,None].expand(B,H1,W1).reshape(-1)
        else:
            # randomly spread
            n = (H - 2*self.border) * (W - 2*self.border) // step**2
            x1 = torch.randint(self.border, W-self.border, (n,), device=dev)
            y1 = torch.randint(self.border, H-self.border, (n,), device=dev)
            x1 = x1[None,:].expand(B,n).reshape(-1)
            y1 = y1[None,:].expand(B,n).reshape(-1)
            b1 = b1[:,None].expand(B,n).reshape(-1)
            shape = (B, n)
        return b1, y1, x1, shape

    def forward(self, feats, confs, aflow, **kw):
        B, two, H, W = aflow.shape
        assert two == 2, bb()
        feat1, conf1 = feats[0], (confs[0] if confs else None)
        feat2, conf2 = feats[1], (confs[1] if confs else None)
        
        # positions in the first image
        b_, y1, x1, shape = self.gen_grid(self.sub_q, aflow)

        # sample features from first image
        feat1 = feat1[b_, :, y1, x1]
        qconf = conf1[b_, :, y1, x1].view(shape) if confs else None
        
        #sample GT from second image
        xy2 = (aflow[b_, :, y1, x1] + 0.5).long().t()
        mask = (0 <= xy2[0]) * (0 <= xy2[1]) * (xy2[0] < W) * (xy2[1] < H)
        mask = mask.view(shape)

        def clamp(xy):
            torch.clamp(xy[0], 0, W-1, out=xy[0])
            torch.clamp(xy[1], 0, H-1, out=xy[1])
            return xy

        # compute positive scores
        xy2p = clamp(xy2[:,None,:] + self.pos_offsets[:,:,None])
        pscores = torch.einsum('nk,ink->ni', feat1, feat2[b_, :, xy2p[1], xy2p[0]])

        # compute negative scores
        xy2n = clamp(xy2[:,None,:] + self.neg_offsets[:,:,None])
        nscores = torch.einsum('nk,ink->ni', feat1, feat2[b_, :, xy2n[1], xy2n[0]])

        if self.sub_d_neg:
            # add distractors from a grid
            b3, y3, x3 = self.gen_grid(self.sub_d_neg, aflow)[:3]
            distractors = feat2[b3, :, y3, x3]
            dscores = torch.einsum('nk,ik->ni', feat1, distractors)
            del distractors
            
            # remove scores that corresponds to positives or nulls
            x2, y2 = xy2 = xy2.float()
            xy3 = torch.stack((x3,y3)).float()
            dis2 = torch.cdist((xy2+b_*512).T, (xy3+b3*512).T, compute_mode='donot_use_mm_for_euclid_dist')
            dscores[dis2 < self.neg_d] = 0
            
            scores = torch.cat((pscores, nscores, dscores), dim=1)

        gt = scores.new_zeros(scores.shape, dtype=torch.uint8)
        gt[:, :pscores.shape[1]] = 1

        return scores, gt, mask, qconf
