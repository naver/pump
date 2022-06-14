# Copyright 2022-present NAVER Corp.
# CC BY-NC-SA 4.0
# Available only for non-commercial use

from pdb import set_trace as bb

import torch
import torch.nn as nn
import torch.nn.functional as F

from core import functional as myF


class DeepMatchingLoss (nn.Module):
    """ This loss is based on DeepMatching (IJCV'16).
    atleast:    (int) minimum image size at which the pyramid construction stops.
    sub:        (int) prior subsampling
    way:        (str) which way to compute the asymmetric matching ('1', '2' or '12')
    border:     (int) ignore pixels too close to the border
    rectify_p:  (float) non-linear power-rectification in DeepMatching
    eps:        (float) epsilon for the L1 normalization. Kinda handles unmatched pixels.
    """
    def __init__(self, eps=0.03, atleast=5, sub=2, way='12', border=16, rectify_p=1.5):
        super().__init__()
        assert way in ('1','2','12')
        self.subsample = sub
        self.border = border
        self.way = way
        self.atleast = atleast
        self.rectify_p = rectify_p
        self.eps = eps

        self._cache = {}

    def rectify(self, corr):
        corr = corr.clip_(min=0)
        corr = corr ** self.rectify_p
        return corr
        
    def forward(self, desc1, desc2, **kw):
        # 1 --> 2
        loss1 = self.forward_oneway(desc1, desc2, **kw) \
                if '1' in self.way else 0

        # 2 --> 1
        loss2 = self.forward_oneway(desc2, desc1, **kw) \
                if '2' in self.way else 0

        return dict(deepm_loss=(loss1+loss2)/len(self.way))

    def forward_oneway(self, desc1, desc2, dbg=(), **kw):
        assert desc1.shape[:2] == desc2.shape[:2]

        # prior subsampling
        s = slice(self.border, -self.border or None, self.subsample)
        desc1, desc2 = desc1[...,s,s], desc2[...,s,s]
        desc1 = desc1[:,:,2::4,2::4] # subsample patches in 1st image
        B, D, H1, W1, H2, W2 = desc1.shape + desc2.shape[-2:]
        if B == 0: return 0 # empty batch

        # intial 4D correlation volume
        corr = torch.bmm(desc1.reshape(B,D,-1).transpose(1,2), desc2.reshape(B,D,-1)).view(B,H1,W1,H2,W2)

        # build pyramid
        pyramid = self.deep_matching(corr)
        corr = pyramid[-1] # high-level correlation
        corr = self.rectify(corr)

        # L1 norm
        B, H1, W1, H2, W2 = corr.shape
        corr = corr / (corr.reshape(B,H1*W1,-1).sum(dim=-1).view(B,H1,W1,1,1) + self.eps)

        # squared L2 norm 
        loss = - torch.square(corr).sum() / (B*H1*W1)
        return loss

    def deep_matching(self, corr):
        # print(f'level=0 {corr.shape=}')
        weights = None
        pyramid = [corr]
        for level in range(1,999):
            corr, weights = self.forward_level(level, corr, weights)
            pyramid.append(corr)
            # print(f'{level=} {corr.shape=}')
            if weights.sum() == 0: break # img1 has become too small
            if min(corr.shape[-2:]) < 2*self.atleast: break # img2 has become too small
        return pyramid

    def forward_level(self, level, corr, weights):
        B, H1, W1, H2, W2 = corr.shape

        # max-pooling
        pooled = F.max_pool2d(corr.view(B,H1*W1,H2,W2), 3, padding=1, stride=2)
        pooled = pooled.view(B, H1, W1, *pooled.shape[-2:])

        # print(f'rectifying corr at {level=}')
        pooled = self.rectify(pooled)

        # sparse conv
        key = level, H1, W1, H2, W2
        if key not in self._cache:
            B, H1, W1, H2, W2 = myF.true_corr_shape(pooled.shape, level-1)
            self._cache[key] = myF.children(level, H1, W1, H2, W2).to(corr.device)

        return sparse_conv(level, pooled, self._cache[key], weights)


def sparse_conv(level, corr, parents, weights=None, border_norm=0.9):
    B, H1, W1, H2, W2 = myF.true_corr_shape(corr.shape, level-1)
    n_cache = len(parents)

    # perform the sparse convolution 'manually'
    # since sparse convolutions are not implemented in pytorch currently
    corr = corr.view(B, -1, H2, W2)

    res = corr.new_zeros((B, n_cache+1, H2, W2)) # last one = garbage channel
    nrm = corr.new_full((n_cache+1, 3, 3), torch.finfo(corr.dtype).eps)
    ones = nrm.new_ones((corr.shape[1], 1, 1))
    ex = 1
    if weights is not None: 
        weights = weights.view(corr.shape[1],1,1)
        corr = corr * weights[None] # apply weights to correlation maps beforehand
        ones *= weights

    sl = lambda v: slice(0,-1 or None) if v < 0 else slice(1,None)
    c = 0
    for y in (-1, 1):
        for x in (-1, 1):
            src_layers = parents[:,c]; c+= 1
            # we want to do: res += corr[src_layers]  (for all children != -1)
            # but we only have 'res.index_add_()' <==> res[tgt_layers] += corr
            tgt_layers = myF.inverse_mapping(src_layers, max_elem=corr.shape[1], default=n_cache)[:-1]

            # All of corr's channels MUST be utilized. for level>1, this doesn't hold,
            # so we'll send them to a garbage channel ==> res[n_cache]
            sel = myF.good_slice( tgt_layers < n_cache )

            res[:,:,sl(-y),sl(-x)].index_add_(1, tgt_layers[sel], corr[:,sel,sl(y),sl(x)])
            nrm[  :,sl(-y),sl(-x)].index_add_(0, tgt_layers[sel], ones[sel].expand(-1,2,2))

    # normalize borders
    weights = myF.norm_borders(res, nrm, norm=border_norm)[:-1]

    res = res[:,:-1] # remove garbage channel
    return res.view(B, H1+ex, W1+ex, *res.shape[-2:]), weights

