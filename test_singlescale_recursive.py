# Copyright 2022-present NAVER Corp.
# CC BY-NC-SA 4.0
# Available only for non-commercial use

from pdb import set_trace as bb
from tqdm import tqdm
import numpy as np
import torch

import test_singlescale as tss
import core.functional as myF
from tools.viz import dbgfig, show_correspondences


def arg_parser(parser = None):
    parser = parser or tss.arg_parser()

    parser.add_argument('--rec-overlap', type=float, default=0.5, help='overlap between tiles in [0,0.5]')
    parser.add_argument('--rec-score-thr', type=float, default=1, help='corres score threshold to guide fine levels')
    parser.add_argument('--rec-fast-thr', type=float, default=0.1, help='prune block if less than `fast` corres fall in it')

    return parser


class RecursivePUMP (tss.SingleScalePUMP):
    """ Recursive PUMP: 
        1) find initial correspondences at a coarse scale, 
        2) refine them at a selection of finer scales
    """
    def __init__(self, coarse_size=512, fine_size=512, rec_overlap=0.5, rec_score_thr=1.0, 
                       rec_fast_thr = 0.1, **other_options ):
        super().__init__(**other_options)
        assert 10 < coarse_size < 1024
        assert 10 < fine_size < 1024
        assert 0 <= rec_overlap < 1
        assert 0 < rec_fast_thr < 1
        self.coarse_size = coarse_size
        self.fine_size = fine_size
        self.overlap = rec_overlap
        self.score_thr = rec_score_thr
        self.fast_thr = rec_fast_thr

    @torch.no_grad()
    def forward(self, img1, img2, ret='corres', dbg=()):
        img1, sca1 = self.demultiplex_img_trf(img1, force=True)
        img2, sca2 = self.demultiplex_img_trf(img2, force=True)
        input_trfs = (sca1, sca2)

        # coarse first level with low-res images
        corres = self.coarse_correspondences(img1, img2)

        # fine level: iterate on HQ blocks
        accu1, accu2 = (self._make_accu(img1), self._make_accu(img2))
        for block1, block2 in tqdm(list(self._enumerate_blocks(img1, img2, corres))):
            # print(f"img1[{block1[}:{}, {}:{}]"
            accus, trfs = tss.SingleScalePUMP.forward(self, block1, block2, ret='raw', dbg=dbg)
            self._update_accu( accu1, accus[0], trfs[0][:2,2] )
            self._update_accu( accu2, accus[1], trfs[1][:2,2] )

        demul = lambda accu: (accu[:,:,:4].reshape(-1,4).clone(), accu[:,:,4].clone())
        corres = demul(accu1), demul(accu2)
        if dbgfig('corres', dbg): viz_correspondences(img1, img2, *corres, fig='last')
        corres = [(myF.affmul(input_trfs,pos),score) for pos, score in corres] # rectify scaling etc.
        if ret == 'raw': return corres, input_trfs
        return self.reciprocal(*corres)

    def coarse_correspondences(self, img1, img2, **kw):
        # joint image resize, because relative size is important (multiscale)
        shape1, shape2 = img1.shape[-2:], img2.shape[-2:]
        if max(shape1 + shape2) > self.coarse_size:
            f1 = self.coarse_size / max(shape1)
            f2 = self.coarse_size / max(shape2)
            f = min(f1, f2)
            img1 = myF.imresize( img1, int(0.5+f*max(shape1)) )
            img2 = myF.imresize( img2, int(0.5+f*max(shape2)) )
        else:
            f = 1

        init_corres = tss.SingleScalePUMP.forward(self, img1, img2, **kw)
        # show_correspondences(img1, img2, init_corres, fig='last')
        corres = init_corres[init_corres[:,4] > self.score_thr]
        print(f"  keeping {len(corres)}/{len(init_corres)} corres with score > {self.score_thr} ...")
        return corres

    def _update_accu(self, accu, update, offset ):
        pos, scores = update
        H, W = scores.shape
        offx, offy = map(lambda i: int(i/4), offset)
        accu = accu[offy:offy+H, offx:offx+W]
        better = accu[:,:,4] < scores
        accu[:,:,4][better] = scores[better].float()
        accu[:,:,0:4][better] = pos.reshape(H,W,4)[better]

    def _enumerate_blocks(self, img1, img2, corres):
        H1, W1, H2, W2 = img1.shape[1:] + img2.shape[1:]
        size, step = self.fine_size, int(self.overlap * self.fine_size)
        def regular_steps(size): 
            if size <= self.fine_size: return [0]
            nb = int(np.ceil(size / step)) - 1 # garranted >= 1
            return (np.linspace(0, size-self.fine_size, nb) / 4 + 0.5).astype(int) * 4
        def translation(x,y):
            res = torch.eye(3, device=img1.device)
            res[0,2] = x
            res[1,2] = y
            return res
        def block2(x2,y2):
            return img2[:,y2:y2+size,x2:x2+size], translation(x2,y2)
        cx1, cy1 = corres[:,0:2].T

        for y1 in regular_steps(H1):
          for x1 in regular_steps(W1):
            block1 = (img1[:,y1:y1+size,x1:x1+size], translation(x1,y1))
            c2 = corres[(y1<=cy1) & (cy1<y1+size) & (x1<=cx1) & (cx1<x1+size)]
            nb_init = len(c2)
            while len(c2):
                cx2, cy2 = c2[:,2:4].T
                x2, y2 = (int(max(0,min(W2-size,cx2.median()-size//2)) / 4 + 0.5) * 4, 
                          int(max(0,min(H2-size,cy2.median()-size//2)) / 4 + 0.5) * 4)
                inside = (y2<=cy2) & (cy2<y2+size) & (x2<=cx2) & (cx2<x2+size)
                if not inside.any(): 
                    x2, y2 = c2[np.random.choice(len(c2)),2:4]
                    x2 = int(max(0,min(W2-size,x2-size//2)) / 4 + 0.5) * 4
                    y2 = int(max(0,min(H2-size,y2-size//2)) / 4 + 0.5) * 4
                    inside = (y2<=cy2) & (cy2<y2+size) & (x2<=cx2) & (cx2<x2+size)

                if inside.sum()/nb_init >= self.fast_thr:
                    yield block1, block2(x2,y2)

                c2 = c2[~inside] # remove

    def _make_accu(self, img):
        C, H, W = img.shape
        return img.new_zeros(((H+3)//4, (W+3)//4, 5), dtype=torch.float32)



class Main (tss.Main):
    @staticmethod
    def build_matcher(args, device):
        # set coarse and fine size based on now obsolete --resize argument
        if isinstance(args.resize, int): args.resize = [args.resize]
        if len(args.resize) == 1: args.resize *= 2
        args.rec_coarse_size, args.rec_fine_size = args.resize
        args.resize = 0 # disable it so that image loading does not downsize images

        options = Main.get_options( args )

        matcher = RecursivePUMP( coarse_size=args.rec_coarse_size, fine_size=args.rec_fine_size, 
            rec_overlap=args.rec_overlap, rec_score_thr=args.rec_score_thr, rec_fast_thr=args.rec_fast_thr,
            **options)

        return tss.Main.tune_matcher(args, matcher, device )


if __name__ == '__main__':
    Main().run_from_args(arg_parser().parse_args())
