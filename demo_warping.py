# Copyright 2022-present NAVER Corp.
# CC BY-NC-SA 4.0
# Available only for non-commercial use

from pdb import set_trace as bb
import os, os.path as osp

from PIL import Image
import numpy as np
from tools.viz import pl, noticks

""" This script will warp (deform) img2 so that it fits img1

>> In case of memory failure (not enough GPU memory):
   try adding '--resize 400 300' (or larger values if possible) to the _exec(...) command below.
"""

def parse_args():
    import argparse
    parser = argparse.ArgumentParser('PUMP demo script for the image warping demo')

    parser.add_argument('--img1', default='datasets/demo_warp/mountains_src.jpg')
    parser.add_argument('--img2', default='datasets/demo_warp/mountains_tgt.jpg')
    parser.add_argument('--output', default='results/demo_warp')

    parser.add_argument('--just-print', action='store_true', help='just print commands')
    return parser.parse_args()


def main( args ):
    run_pump(args) and run_demo_warp(args)


def run_pump(args):
    output_path = osp.join(args.output, args.img1, args.img2+'.corres')
    if osp.isfile(output_path): return True

    return _exec(f'''python test_singlescale_recursive.py
            --img1 {args.img1}
            --img2 {args.img2}
            --post-filter densify=True
            --output {output_path}''')


def run_demo_warp(args):
    corres_path = osp.join(args.output, args.img1, args.img2+'.corres')
    corres = np.load(corres_path)['corres']

    img1 = Image.open(args.img1).convert('RGB')
    img2 = Image.open(args.img2).convert('RGB')

    W, H = img1.size
    warped_img2 = warp_img(np.asarray(img2), corres[:,2:4].reshape(H,W,2))

    pl.figure('Warping demo')

    noticks(pl.subplot(211))
    pl.imshow( img2 )
    pl.title('Source image')

    noticks(pl.subplot(223))
    pl.imshow( img1 )
    pl.title('Target image')

    noticks(pl.subplot(224))
    pl.imshow( warped_img2 )
    pl.title('Source image warped to match target')

    pl.tight_layout()
    pl.show(block=True)


def warp_img( img, absolute_flow ):
    H1, W1, TWO = absolute_flow.shape
    H2, W2, THREE = img.shape
    assert TWO == 2 and  THREE == 3

    warp = absolute_flow.round().astype(int)
    invalid = (warp[:,:,0]<0) | (warp[:,:,0]>=W2) | (warp[:,:,1]<0) | (warp[:,:,1]>=H2)

    warp[:,:,0] = warp[:,:,0].clip(min=0, max=W2-1)
    warp[:,:,1] = warp[:,:,1].clip(min=0, max=H2-1)
    warp = warp[:,:,0] + W2*warp[:,:,1]

    warped_img = np.asarray(img).reshape(-1,3)[warp].reshape(H1,W1,3)
    return warped_img


def _exec(cmd):
    # strip & remove \n
    cmd = ' '.join(cmd.split())

    if args.just_print: 
        print(cmd)
        return False
    else:
        return os.WEXITSTATUS(os.system(cmd)) == 0


if __name__ == '__main__':
    args = parse_args()
    main( args )
