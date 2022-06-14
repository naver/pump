# Copyright 2022-present NAVER Corp.
# CC BY-NC-SA 4.0
# Available only for non-commercial use

from pdb import set_trace as bb
from itertools import starmap
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import test_singlescale as tss
from core import functional as myF
from tools.common import todevice, cpu
from tools.viz import dbgfig, show_correspondences


def arg_parser():
    parser = tss.arg_parser()
    parser.set_defaults(levels = 0, verbose=0)

    parser.add_argument('--min-scale', type=float, default=None, help='min scale ratio')
    parser.add_argument('--max-scale', type=float, default=4, help='max scale ratio')

    parser.add_argument('--min-rot', type=float, default=None, help='min rotation (in degrees) in [-180,180]')
    parser.add_argument('--max-rot', type=float, default=0, help='max rotation (in degrees) in [0,180]')
    parser.add_argument('--crop-rot', action='store_true', help='crop rotated image to prevent memory blow-up')
    parser.add_argument('--rot-step', type=int, default=45, help='rotation step (in degrees)')

    parser.add_argument('--no-swap', type=int, default=1, nargs='?', const=0, choices=[1,0,-1], help='if 0, img1 will have keypoints on a grid')
    parser.add_argument('--same-levels', action='store_true', help='use the same number of pyramid levels for all scales')

    parser.add_argument('--merge', choices='torch cpu cuda'.split(), default='cpu')
    return parser


class MultiScalePUMP (nn.Module):
    """ DeepMatching that loops over all possible {scale x rotation} combinations.
    """
    def __init__(self, matcher, 
                    min_scale=1, 
                    max_scale=1, 
                    max_rot=0, 
                    min_rot=0, 
                    rot_step=45, 
                    swap_mode=1, 
                    same_levels=False, 
                    crop_rot=False):
        super().__init__()
        min_scale = min_scale or 1/max_scale
        min_rot = min_rot or -max_rot
        assert 0.1 <= min_scale <= max_scale <= 10
        assert -180 <= min_rot <= max_rot <= 180
        self.matcher = matcher
        self.matcher.crop_rot = crop_rot

        self.min_sc = min_scale
        self.max_sc = max_scale
        self.min_rot = min_rot
        self.max_rot = max_rot
        self.rot_step = rot_step
        self.swap_mode = swap_mode
        self.merge_device = None
        self.same_levels = same_levels

    @torch.no_grad()
    def forward(self, img1, img2, dbg=()):
        img1, sca1 = img1 if isinstance(img1, tuple) else (img1, torch.eye(3, device=img1.device))
        img2, sca2 = img2 if isinstance(img2, tuple) else (img2, torch.eye(3, device=img2.device))

        # prepare correspondences accumulators
        if self.same_levels: # limit number of levels
            self.matcher.levels = self._find_max_levels(img1,img2)
        elif self.matcher.levels == 0:
            max_psize = int(min(np.mean(img1.shape[-2:]), np.mean(img2.shape[-2:])))
            self.matcher.levels = int(np.log2(max_psize / self.matcher.pixel_desc.get_atomic_patch_size()))

        all_corres = (self._make_accu(img1), self._make_accu(img2))

        for scale, ang, code, swap, swapped, (scimg1, scimg2) in self._enum_scaled_pairs(img1, img2):
            print(f"processing {scale=:g} x {ang=} {['','(swapped)'][swapped]} ({code=})...")

            # compute correspondences with rotated+scaled image
            corres, rots = self.process_one_scale(swapped, *[scimg1,scimg2], dbg=dbg)
            if dbgfig('corres-ms', dbg): viz_correspondences(img1, img2, *corres, fig='last')

            # merge correspondences in the reference frame
            self.merge_corres( corres, rots, all_corres, code )

        # final intersection
        corres = self.reciprocal( *all_corres )
        return myF.affmul(todevice((sca1,sca2),corres.device), corres) # rescaling to original image scale

    def process_one_scale(self, swapped, *imgs, dbg=()):
        return unswap(self.matcher(*imgs, ret='raw', dbg=dbg), swapped)

    def _find_max_levels(self, img1, img2):
        min_levels = self.matcher.levels or 999
        for _, _, code, _, _, (img1, img2) in self._enum_scaled_pairs(img1, img2):
            # first level when a parent dont have children: gap >= min(shape), with gap = 2**(level-2)
            img1_levels = ceil(np.log2(min(img1[0].shape[-2:])) - 1)
            # first level when img2's shape becomes smaller than self.min_shape, with shape = min(shape) / 2**level
            img2_levels = ceil(np.log2(min(img2[0].shape[-2:]) / self.matcher.min_shape))
            # print(f'predicted levels for {code=}:\timg1 --> {img1_levels},\timg2 --> {img2_levels} levels')
            min_levels = min(min_levels, img1_levels, img2_levels)
        return min_levels

    def merge_corres(self, corres, rots, all_corres, code):
        " rot : reference --> rotated "
        self.merge_one_side( corres[0], slice(0,2), rots[0], all_corres[0], code )
        self.merge_one_side( corres[1], slice(2,4), rots[1], all_corres[1], code )

    def merge_one_side(self, corres, sel, trf, all_corres, code ):
        pos, scores = corres
        grid, accu = all_corres
        accu = accu.view(-1, 6)

        # compute 4-nn in transformed image for each grid point
        best4 = torch.cdist(pos[:,sel].float(), grid).topk(4, dim=0, largest=False)
        # best4.shape = (4, len(grid))

        # update if score is better AND distance less than 2x best dist
        scale = float(torch.sqrt(torch.det(trf))) # == scale (with scale >= 1)
        dist_max = 8*scale - 1e-7 # 2x the distance between contiguous patches

        close_enough = (best4.values <= 2*best4.values[0:1]) & (best4.values < dist_max)
        neg_inf = torch.tensor(-np.inf, device=scores.device)
        best_score = torch.where(close_enough, scores.ravel()[best4.indices], neg_inf).max(dim=0)
        is_better = best_score.values > accu[:,4].ravel()

        accu[is_better,0:4] = pos[best4.indices[best_score.indices,torch.arange(len(grid))][is_better]]
        accu[is_better,4] = best_score.values[is_better]
        accu[is_better,5] = code

    def reciprocal(self, corres1, corres2 ):
        grid1, corres1 = cpu(corres1)
        grid2, corres2 = cpu(corres2)

        (H1, W1), (H2, W2) = grid1[-1]+1, grid2[-1]+1
        pos1 = corres1[:,:,0:4].view(-1,4)
        pos2 = corres2[:,:,0:4].view(-1,4)

        to_int = torch.tensor((W1*H2*W2, H2*W2, W2, 1), dtype=torch.float32)
        inter1 = myF.intersection(pos1@to_int, pos2@to_int)
        return corres1.view(-1,6)[inter1]

    def _enum_scales(self):
        for i in range(-100,101):
            scale = 2**(i/2)
            # if i != -2: continue
            if self.min_sc <= scale <= self.max_sc:
                yield i,scale

    def _enum_rotations(self):
        for i in range(-180//self.rot_step, 180//self.rot_step):
            rot = i * self.rot_step
            if self.min_rot <= rot <= self.max_rot:
                yield i,-rot

    def _enum_scaled_pairs(self, img1, img2):
        for s, scale in self._enum_scales():
            (i1,sca1), (i2,sca2) = starmap(downsample_img, [(img1, min(scale, 1)), (img2, min(1/scale, 1))])
            # set bigger image as the first one
            size1 = min(i1.shape[-2:])
            size2 = min(i2.shape[-2:])
            swapped = size1*self.swap_mode < size2*self.swap_mode
            swap = (1 - 2*swapped) # swapped ==> swap = -1
            if swapped:
                (i1,sca1), (i2,sca2) = (i2,sca2), (i1,sca1)

            for r, ang in self._enum_rotations():
                code = myF.encode_scale_rot(scale, ang)
                trf1 = (sca1, swap*ang) if ang != 0 else sca1
                yield scale, ang, code, swap, swapped, ((i1,trf1), (i2,sca2))

    def _make_accu(self, img):
        C, H, W = img.shape
        step = self.matcher.pixel_desc.get_atomic_patch_size() // 2
        h = step//2 - 1
        accu = img.new_zeros(((H+h)//step, (W+h)//step, 6), dtype=torch.float32, device=self.merge_device or img.device)
        grid = step * myF.mgrid(accu[:,:,0], device=img.device) + (step//2)
        return grid, accu


def downsample_img(img, scale=0):
    assert scale <= 1
    img, trf = img if isinstance(img, tuple) else (img, torch.eye(3, device=img.device))
    if scale == 1: return img, trf

    assert img.dtype == torch.uint8
    trf = trf.clone() # dont modify inplace
    trf[:2,:2] /= scale
    while scale <= 0.5:
        img = F.avg_pool2d(img[None].float(), 2, stride=2, count_include_pad=False)[0]
        scale *= 2
    if scale != 1:
        img = F.interpolate(img[None].float(), scale_factor=scale, mode='bicubic', align_corners=False, recompute_scale_factor=False).clamp(min=0, max=255)[0]
    return img.byte(), trf # scaled --> pxl


def ceil(i):
    return int(np.ceil(i))

def unswap( corres, swapped ):
    swap = -1 if swapped else 1
    corres, rots = corres
    corres = corres[::swap]
    rots = rots[::swap]
    if swapped:
        for pos, _ in corres:
            pos[:,0:4] = pos[:,[2,3,0,1]].clone()
    return corres, rots


def demultiplex_img_trf(self, img, force=False):
    """ img is:
        - an image
        - a tuple (image, trf) 
        - a tuple (image, (cur_trf, trf_todo)) 
    In any case, trf: cur_pix --> old_pix
    """
    img, trf = img if isinstance(img, tuple) else (img, torch.eye(3, device=img.device))

    if isinstance(trf, tuple):
        trf, todo = trf
        if isinstance(todo, (int,float)): # pure rotation
            img, trf = myF.rotate_img((img,trf), angle=todo, crop=self.crop_rot)
        else:
            img = myF.apply_trf_to_img(todo, img)
            trf = trf @ todo
    return img, trf


class Main (tss.Main):
    @staticmethod
    def get_options( args ):
        return dict(max_scale=args.max_scale, min_scale=args.min_scale, 
                    max_rot=args.max_rot, min_rot=args.min_rot, rot_step=args.rot_step, 
                    swap_mode=args.no_swap, same_levels=args.same_levels, crop_rot=args.crop_rot) 

    @staticmethod
    def tune_matcher( args, matcher, device ):
        if device == 'cpu': 
            args.merge = 'cpu'

        if args.merge == 'cpu': type(matcher).merge_corres = myF.merge_corres; matcher.merge_device = 'cpu'
        elif args.merge == 'cuda': type(matcher).merge_corres = myF.merge_corres

        return matcher.to(device)

    @staticmethod
    def build_matcher( args, device):
        # get a normal matcher
        matcher = tss.Main.build_matcher(args, device)
        type(matcher).demultiplex_img_trf = demultiplex_img_trf # update transformer

        options = Main.get_options(args)
        return Main.tune_matcher(args, MultiScalePUMP(matcher, **options), device)


if __name__ == '__main__':
    Main().run_from_args(arg_parser().parse_args())
