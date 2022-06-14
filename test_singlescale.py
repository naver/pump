# Copyright 2022-present NAVER Corp.
# CC BY-NC-SA 4.0
# Available only for non-commercial use

from pdb import set_trace as bb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core import functional as myF
from core.pixel_desc import PixelDesc
from tools.common import mkdir_for, todevice, cudnn_benchmark, nparray, image, image_with_trf
from tools.viz import dbgfig, show_correspondences


def arg_parser():
    import argparse
    parser = argparse.ArgumentParser('SingleScalePUMP on GPU with PyTorch')

    parser.add_argument('--img1', required=True, help='path to img1')
    parser.add_argument('--img2', required=True, help='path to img2')
    parser.add_argument('--resize', type=int, default=512, nargs='+', help='prior downsize of img1 and img2')

    parser.add_argument('--output', default=None, help='output path for correspondences')

    parser.add_argument('--levels', type=int, default=99, help='number of pyramid levels')
    parser.add_argument('--min-shape', type=int, default=5, help='minimum size of corr maps')
    parser.add_argument('--nlpow', type=float, default=1.5, help='non-linear activation power in [1,2]')
    parser.add_argument('--border', type=float, default=0.9, help='border invariance level in [0,1]')
    parser.add_argument('--dtype', default='float16', choices='float16 float32 float64'.split())

    parser.add_argument('--desc', default='PUMP-stytrf', help='checkpoint name')
    parser.add_argument('--first-level', choices='torch'.split(), default='torch')
    parser.add_argument('--activation', choices='torch'.split(), default='torch')
    parser.add_argument('--forward', choices='torch cuda cuda-lowmem'.split(), default='cuda-lowmem')
    parser.add_argument('--backward', choices='python torch cuda'.split(), default='cuda')
    parser.add_argument('--reciprocal', choices='cpu cuda'.split(), default='cpu')

    parser.add_argument('--post-filter', default=None, const=True, nargs='?', help='post-filtering (See post_filter.py)')

    parser.add_argument('--verbose', type=int, default=0, help='verbosity')
    parser.add_argument('--device', default='cuda', help='gpu device')
    parser.add_argument('--dbg', nargs='*', default=(), help='debug options')

    return parser


class SingleScalePUMP (nn.Module):
    def __init__(self, levels = 9, nlpow = 1.4, cutoff = 1, 
                 border_inv=0.9, min_shape=5, renorm=(),
                 pixel_desc = None, dtype = torch.float32,
                 verbose = True ):
        super().__init__()
        self.levels = levels
        self.min_shape = min_shape
        self.nlpow = nlpow
        self.border_inv = border_inv
        assert pixel_desc, 'Requires a pixel descriptor'
        self.pixel_desc = pixel_desc.configure(self)
        self.dtype = dtype
        self.verbose = verbose

    @torch.no_grad()
    def forward(self, img1, img2, ret='corres', dbg=()):
        with cudnn_benchmark(False):
            # compute descriptors
            (img1, img2), pixel_descs, trfs = self.extract_descs(img1, img2, dtype=self.dtype)

            # backward and forward passes
            pixel_corr = self.first_level(*pixel_descs, dbg=dbg)
            pixel_corr = self.backward_pass(self.forward_pass(pixel_corr, dbg=dbg), dbg=dbg)

            # recover correspondences
            corres = myF.best_correspondences( pixel_corr )

        if dbgfig('corres', dbg): viz_correspondences(img1[0], img2[0], *corres, fig='last')
        corres = [(myF.affmul(trfs,pos),score) for pos, score in corres] # rectify scaling etc.
        if ret == 'raw': return corres, trfs
        return self.reciprocal(*corres)

    def extract_descs(self, img1, img2, dtype=None):
        img1, sca1 = self.demultiplex_img_trf(img1)
        img2, sca2 = self.demultiplex_img_trf(img2)
        desc1, trf1 = self.pixel_desc(img1)
        desc2, trf2 = self.pixel_desc(img2)
        return (img1, img2), (desc1.type(dtype), desc2.type(dtype)), (sca1@trf1, sca2@trf2)

    def demultiplex_img_trf(self, img, **kw):
        return img if isinstance(img, tuple) else (img, torch.eye(3, device=img.device))

    def forward_pass(self, pixel_corr, dbg=()):
        weights = None
        if isinstance(pixel_corr, tuple):
            pixel_corr, weights = pixel_corr

        # first-level with activation
        if self.verbose: print(f'  Pyramid level {0} shape={tuple(pixel_corr.shape)}')
        pyramid = [ self.activation(0,pixel_corr) ]
        if dbgfig(f'corr0', dbg): viz_correlation_maps(*from_stack('img1','img2'), pyramid[0], fig='last')

        for level in range(1, self.levels+1):
            upper, weights = self.forward_level(level, pyramid[-1], weights)
            if weights.sum() == 0: break # img1 has become too small

            # activation
            pyramid.append( self.activation(level,upper) )

            if self.verbose: print(f'  Pyramid level {level} shape={tuple(upper.shape)}')
            if dbgfig(f'corr{level}', dbg): viz_correlation_maps(*from_stack('img1','img2'), upper, level=level, fig='last')
            if min(upper.shape[-2:]) <= self.min_shape: break # img2 has become too small

        return pyramid

    def forward_level(self, level, corr, weights):
        # max-pooling
        pooled = F.max_pool2d(corr, 3, padding=1, stride=2)

        # sparse conv
        return myF.sparse_conv(level, pooled, weights, norm=self.border_inv)

    def backward_pass(self, pyramid, dbg=()):
        # same than forward in reverse order
        for level in range(len(pyramid)-1, 0, -1):
            lower = self.backward_level(level, pyramid)
            # assert not torch.isnan(lower).any(), bb()
            if self.verbose: print(f'  Pyramid level {level-1} shape={tuple(lower.shape)}')
            del pyramid[-1] # free memory
            if dbgfig(f'corr{level}-bw', dbg): viz_correlation_maps(img1, img2, lower, fig='last')
        return pyramid[0]

    def backward_level(self, level, pyramid):
        # reverse sparse-coonv
        pooled = myF.sparse_conv(level, pyramid[level], reverse=True)

        # reverse max-pool and add to lower level
        return myF.max_unpool(pooled, pyramid[level-1])

    def activation(self, level, corr):
        assert 1 <= self.nlpow <= 3
        corr.clamp_(min=0).pow_(self.nlpow)
        return corr

    def first_level(self, desc1, desc2, dbg=()):
        assert desc1.ndim == desc2.ndim == 4
        assert len(desc1) == len(desc2) == 1, "not implemented"
        H1, W1 = desc1.shape[-2:]
        H2, W2 = desc2.shape[-2:]

        patches = F.unfold(desc1, 4, stride=4) # C*4*4, H1*W1//16
        B, C, N = patches.shape
        # rearrange(patches, 'B (C Kh Kw) H1W1 -> B H1W1 C Kh Kw', Kh=4, Kw=4)
        patches = patches.permute(0, 2, 1).view(B, H1W1, C//16, 4, 4)

        corr, norms = myF.normalized_corr(patches[0], desc2[0], ret_norms=True)
        if dbgfig('ncc',dbg):
            for j in range(0,len(corr),9):
              for i in range(9):
                pl.subplot(3,3,i+1).cla()
                i += j
                pl.imshow(corr[i], vmin=0.9, vmax=1)
                pl.plot(2+(i%16)*4, 2+(i//16)*4,'xr', ms=10)
              bb()
        return corr.view(H1//4, W1//4, H2+1, W2+1), (norms.view(H1//4, W1//4)>0).float()

    def reciprocal(self, corres1, corres2 ):
        corres1, corres2 = todevice(corres1, 'cpu'), todevice(corres2, 'cpu')
        return myF.reciprocal(self, corres1, corres2)


class Main:
    def __init__(self):
        self.post_filtering = False

    def run_from_args(self, args):
        device = args.device
        self.matcher = self.build_matcher(args, device)
        if args.post_filter:
            self.post_filtering = {} if args.post_filter is True else eval(f'dict({args.post_filter})')

        corres = self(*self.load_images(args, device), dbg=set(args.dbg))

        if args.output:
            self.save_output( args.output, corres )

    @staticmethod
    def get_options( args ):
        # configure the pipeline
        pixel_desc = PixelDesc(path=f'checkpoints/{args.desc}.pt')
        return dict(levels=args.levels, min_shape=args.min_shape, border_inv=args.border, nlpow=args.nlpow,
                    pixel_desc=pixel_desc, dtype=eval(f'torch.{args.dtype}'), verbose=args.verbose)

    @staticmethod
    def tune_matcher( args, matcher, device ):
        if device == 'cpu': 
            matcher.dtype = torch.float32
            args.forward = 'torch'
            args.backward = 'torch'
            args.reciprocal = 'cpu'

        if args.forward == 'cuda':       type(matcher).forward_level = myF.forward_cuda
        if args.forward == 'cuda-lowmem':type(matcher).forward_level = myF.forward_cuda_lowmem
        if args.backward == 'python':    type(matcher).backward_pass = legacy.backward_python
        if args.backward == 'cuda':      type(matcher).backward_level = myF.backward_cuda
        if args.reciprocal == 'cuda':    type(matcher).reciprocal = myF.reciprocal

        return matcher.to(device)

    @staticmethod
    def build_matcher(args, device):
        options = Main.get_options(args)
        matcher = SingleScalePUMP(**options)
        return Main.tune_matcher(args, matcher, device)

    def __call__(self, *imgs, dbg=()):
        corres = self.matcher( *imgs, dbg=dbg).cpu().numpy()
        if self.post_filtering is not False: 
            corres = self.post_filter( imgs, corres )

        if 'print' in dbg: print(corres)
        if dbgfig('viz',dbg):   show_correspondences(*imgs, corres)
        return corres

    @staticmethod
    def load_images( args, device='cpu' ):
        def read_image(impath):
            try:
                from torchvision.io.image import read_image, ImageReadMode
                return read_image(impath, mode=ImageReadMode.RGB)
            except RuntimeError:
                from PIL import Image
                return torch.from_numpy(np.array(Image.open(impath).convert('RGB'))).permute(2,0,1)

        if isinstance(args.resize, int): # user can provide 2 separate sizes for each image
            args.resize = (args.resize, args.resize)

        if len(args.resize) == 1: 
            args.resize = 2 * args.resize

        images = []
        for impath, size in zip([args.img1, args.img2], args.resize):
            img = read_image(impath).to(device)
            img = myF.imresize(img, size)
            images.append( img )
        return images

    def post_filter(self, imgs, corres ):
        from post_filter import filter_corres
        return filter_corres(*map(image_with_trf,imgs), corres, **self.post_filtering)

    def save_output(self, output_path, corres ):
        mkdir_for( output_path )
        np.savez(open(output_path,'wb'), corres=corres)



if __name__ == '__main__':
    Main().run_from_args(arg_parser().parse_args())
