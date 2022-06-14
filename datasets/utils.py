# Copyright 2022-present NAVER Corp.
# CC BY-NC-SA 4.0
# Available only for non-commercial use

from pdb import set_trace as bb
import numpy as np
import torch


class DatasetWithRng:
    """ Make sure that RNG is distributed properly when torch.dataloader() is used
    """

    def __init__(self, seed=None):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._rng_children = set()

    def with_same_rng(self, dataset=None):
        if dataset is not None:
            assert isinstance(dataset, DatasetWithRng) and hasattr(dataset, 'rng'), bb()
            self._rng_children.add( dataset )

        # update all registered children
        for db in self._rng_children:
            db.rng = self.rng
            db.with_same_rng() # recursive call
        return dataset

    def init_worker(self, tid):
        if self.seed is None: 
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(self.seed + tid)


class WorkerWithRngInit:
    " Dataset inherits from datasets.DatasetWithRng() and has an init_worker() function "
    def __call__(self, tid):
        torch.utils.data.get_worker_info().dataset.init_worker(tid)


def corres_from_homography(homography, W, H, grid=64):
    s = max(1, min(W, H) // grid) # at least `grid` points in smallest dim
    sx, sy = [slice(s//2, l, s) for l in (W, H)]
    grid1 = np.mgrid[sy, sx][::-1].reshape(2,-1).T # (x1,y1) grid

    grid2 = applyh(homography, grid1)
    scale = np.sqrt(np.abs(np.linalg.det(jacobianh(homography, grid1).T)))

    corres = np.c_[grid1, grid2, np.ones_like(scale), np.zeros_like(scale), scale]
    return corres


def invh( H ):
    return np.linalg.inv(H)


def applyh(H, p, ncol=2, norm=True):
    """ Apply the homography to a list of 2d points in homogeneous coordinates.

    H: Homography (...x3x3 matrix/tensor)
    p: numpy/torch/tuple of coordinates. Shape must be (...,2) or (...,3)
    
    Returns an array of projected 2d points.
    """
    if isinstance(H, np.ndarray):
        p = np.asarray(p)
    elif isinstance(H, torch.Tensor):
        p = torch.as_tensor(p, dtype=H.dtype)

    if p.shape[-1]+1 == H.shape[-1]:
        H = H.swapaxes(-1,-2) # transpose H
        p = p @ H[...,:-1,:] + H[...,-1:,:]
    else:
        p = H @ p.T
        if p.ndim >= 2: p = p.swapaxes(-1,-2)

    if norm: 
        p /= p[...,-1:]
    return p[...,:ncol]


def jacobianh(H, p):
    """ H is an homography that maps: f_H(x,y) --> (f_1, f_2)
    So the Jacobian J_H evaluated at p=(x,y) is a 2x2 matrix
    Output shape = (2, 2, N) = (f_, xy, N)

    Example of derivative:
                  numx    a*X + b*Y + c*Z
        since x = ----- = ---------------
                  denom   u*X + v*Y + w*Z

                numx' * denom - denom' * numx   a*denom - u*numx
        dx/dX = ----------------------------- = ----------------
                           denom**2                 denom**2
    """
    (a, b, c), (d, e, f), (u, v, w) = H
    numx, numy, denom = applyh(H, p, ncol=3, norm=False).T

    #                column x          column x
    J = np.float32(((a*denom - u*numx, b*denom - v*numx),  # row f_1
                    (d*denom - u*numy, e*denom - v*numy))) # row f_2
    return J / np.where(denom, denom*denom, np.nan)
