# Copyright 2022-present NAVER Corp.
# CC BY-NC-SA 4.0
# Available only for non-commercial use

from pdb import set_trace as bb
import os, os.path as osp
from tqdm import tqdm
import numpy as np

SEQUENCES = [ 'lakeside', 'sand_box', 'storage_room', 'storage_room_2', 'tunnel', 
              'delivery_area', 'electro', 'forest', 'playground', 'terrains']

RATES = [3, 5, 7, 9, 11, 13, 15]

def parse_args():
    import argparse
    parser = argparse.ArgumentParser('PUMP evaluation script for the ETH3D dataset')

    parser.add_argument('--root', default='datasets/eth3d')
    parser.add_argument('--output', default='results/eth3d')

    parser.add_argument('--just-print', action='store_true', help='just print commands')
    return parser.parse_args()


def main( args ):
    run_pump(args) and run_eval(args)


def run_pump(args):
    done = True
    for img1, img2 in tqdm(list_eth3d_pairs()):
        output_path = osp.join(args.output, img1, img2+'.corres')
        if osp.isfile(output_path): continue

        done = False
        _exec(f'''python test_multiscale_recursive.py
                --img1 {osp.join(args.root,img1)}
                --img2 {osp.join(args.root,img2)}
                --max-scale 1.5
                --desc PUMP
                --post-filter "densify=True,dense_side='right'"
                --output {output_path}''')

    return done


def run_eval( args ):
    for rate in RATES:
        mean_aepe_per_rate = 0

        for seq in SEQUENCES:
            pairs = np.load(osp.join(args.root, 'info_ETH3D_files', f'{seq}_every_5_rate_of_{rate}'), allow_pickle=True)

            mean_aepe_per_seq = 0
            for pair in pairs:
                img1, img2 = pair['source_image'], pair['target_image']
                Ys, Xs, Yt, Xt = [np.float32(pair[k]) for k in 'Ys Xs Yt Xt'.split()]

                corres_path = osp.join(args.output, img1, img2+'.corres')
                corres = np.load(corres_path, allow_pickle=True)['corres']

                # extract estimated and target flow
                W, H = np.int32(corres[-1, 2:4] + 1)
                flow = (corres[:,0:2] - corres[:,2:4]).reshape(H, W, 2)
                iYt, iXt = np.int32(np.round(Yt)), np.int32(np.round(Xt))
                if 'correct way':
                    gt_targets = np.c_[Xs - Xt, Ys - Yt]
                    est_targets = flow[iYt, iXt]
                elif 'GLU-Net way (somewhat inaccurate because of overlapping points in the mask)':
                    mask = np.zeros((H,W), dtype=bool)
                    mask[iYt, iXt] = True
                    gt_flow = np.full((H,W,2), np.nan, dtype=np.float32)
                    gt_flow[iYt, iXt, 0] = Xs - Xt
                    gt_flow[iYt, iXt, 1] = Ys - Yt
                    gt_targets = gt_flow[mask]
                    est_targets = flow[mask]

                # compute end-point error                
                aepe = np.linalg.norm(est_targets - gt_targets, axis=-1).mean()
                mean_aepe_per_seq += aepe

            mean_aepe_per_seq /= len(pairs)
            mean_aepe_per_rate += mean_aepe_per_seq
            print(f'mean AEPE for {rate=} {seq=}:', mean_aepe_per_seq)

        print(f'>> mean AEPE for {rate=}:', mean_aepe_per_rate / len(SEQUENCES))


def list_eth3d_pairs():
    path = osp.join(args.root, 'info_ETH3D_files', 'list_pairs.txt')
    try:
        lines = open(path).read().splitlines()
    except OSError:
        lines = []
        for seq in SEQUENCES:
            for rate in RATES:
                pairs = np.load(osp.join(args.root, 'info_ETH3D_files', f'{seq}_every_5_rate_of_{rate}'), allow_pickle=True)
                for pair in pairs:
                    lines.append(pair['source_image'] + ' ' + pair['target_image'])
        open(path, 'w').write('\n'.join(lines))

    pairs = [line.split() for line in lines if line[0] != '#']
    return pairs


def _exec(cmd):
    # strip & remove \n
    cmd = ' '.join(cmd.split())
    if args.just_print: 
        print(cmd)
    else:
        os.system(cmd) 


if __name__ == '__main__':
    args = parse_args()
    main( args )
