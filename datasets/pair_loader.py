# Copyright 2022-present NAVER Corp.
# CC BY-NC-SA 4.0
# Available only for non-commercial use

from pdb import set_trace as bb
from PIL import Image
import numpy as np

from core import functional as myF
from tools.common import todevice
from .transforms import instanciate_transforms
from .utils import *


class FastPairLoader (DatasetWithRng):
    """ On-the-fly generation of related image pairs
    crop:   random crop applied to both images
    scale:  random scaling applied to img2
    distort: random ditorsion applied to img2
    
    self[idx] returns: (img1, img2), dict(homography=)
        (homography: 3x3 array, can be nan)
    """
    def __init__(self, dataset, crop=256, transform='', p_flip=0, p_swap=0, scale_jitter=0, seed=None):
        super().__init__(seed)
        self.dataset = self.with_same_rng(dataset)
        self.transform = instanciate_transforms( transform, rng=self.rng )
        self.crop_size = crop
        self.p_swap = p_swap
        self.p_flip = p_flip
        self.scale_jitter = abs(np.log1p(scale_jitter))

    def __len__(self):
        return len(self.dataset)

    def __repr__(self):
        fmt_str = f'FastPairLoader({self.dataset},\n'
        short_repr = lambda s: repr(s).strip().replace('\n',', ')[14:-1].replace('    ',' ')
        fmt_str += '    Transform:\t%s\n' % short_repr(self.transform)
        fmt_str +=f'    Crop={self.crop_size}, scale_jitter=x{np.exp(self.scale_jitter):g}, p_swap={self.p_swap:g}'
        return fmt_str

    def init_worker(self, tid):
        super().init_worker(tid)
        self.dataset.init_worker(tid)

    def set_epoch(self, epoch):
        self.dataset.set_epoch(epoch)

    def __getitem__(self, idx):
        self.init_worker(idx) # preserve RNG for this pair
        (img1, img2), gt = self.dataset[idx]

        if self.rng.random() < self.p_swap:
            img1, img2 = img2, img1
            if 'homography' in gt: gt['homography'] = invh(gt['homography'])
            if 'corres' in gt: gt['corres'] = swap_corres(gt['corres'])

        if self.rng.random() < self.p_flip:
            img1, img2, gt = flip_image_pair(img1, img2, gt)

        # apply transformations to the second image
        img2 = self.transform(dict(img=img2))

        homography, corres = spatial_relationship( img1, img2, gt )

        # find a good window
        img1, img2 = map(self._pad_rgb_numpy, (img1, img2['img']))

        if not 'debug':
            from tools.viz import show_correspondences
            print(np.median(corres[:,5]))
            show_correspondences(img1, img2, corres, bb=bb)

        def windows_from_corres( idx, scale_jitter=1 ):
            c = corres[idx]
            p1, p2, scale = c[0:2], c[2:4], c[6]
            scale *= scale_jitter

            # make windows based on scaling
            win1 = window(*p1, self.crop_size, max(1, 1/scale), img1.shape)
            win2 = window(*p2, self.crop_size, max(1, scale/1), img2.shape)
            return win1, win2

        best = 0, None
        for idx in self.rng.choice(len(corres), size=min(len(corres),5), replace=False):
            # pick a correspondence at random
            win1, win2 = windows_from_corres( idx )

            # check how many matches are in the 2 windows
            score = score_windows(is_in(corres[:,0:2],win1), is_in(corres[:,2:4],win2))
            if score > best[0]: best = score, idx

        others = {}
        if None in best: # counldn't find a good window
            img1 = img2 = np.zeros((self.crop_size,self.crop_size,3), dtype=np.uint8)
            corres = np.empty((0, 6), dtype=np.float32)
        else:
            # jitter scales
            scale_jitter = np.exp(self.rng.uniform(-self.scale_jitter, self.scale_jitter))
            win1, win2 = windows_from_corres( best[1], scale_jitter )
            # print(win1, win2, img1.shape, img2.shape)
            img1, img2 = imresize(img1[win1], self.crop_size), imresize(img2[win2], self.crop_size)
            trf1, trf2 = wintrf(win1, img1), wintrf(win2, img2)

            # fix rotation if necessary
            angle_scores = np.bincount(corres[:,5].astype(int) % 8)
            rot90 = int((((angle_scores.argmax() + 4) % 8) - 4) / 2)
            if rot90: # rectify rotation
                img2, trf = myF.rotate_img_90((img2, np.eye(3)), 90*rot90)
                trf2 = invh(trf) @ trf2

            homography = trf2 @ homography @ invh(trf1)
            corres = myF.affmul((trf1,trf2), corres)

        f32c = lambda i,**kw: np.require(i, requirements='CWAE', **kw)
        return (f32c(img1), f32c(img2)), dict(homography = f32c(homography, dtype=np.float32), corres=corres, **others)

    def _pad_rgb_numpy(self, img):
        if img.mode != 'RGB': 
            img = img.convert('RGB')
        if min(img.size) < self.crop_size:
            w, h = img.size
            result = Image.new('RGB', (max(w,self.crop_size), max(h,self.crop_size)), 0)
            result.paste(img, (0, 0))
            img = result
        return np.asarray(img)



def swap_corres( corres ):  # swap img1 and img2
    res = corres.copy()
    res[:,[0,1,2,3]] = corres[:,[2,3,0,1]]
    if corres.shape[1] > 4: # invert rotation and scale
        scale, rot = myF.decode_scale_rot(corres[:,5])
        res[:,5] = myF.encode_scale_rot(1/scale, -rot)
    return res

def flip(img):
    w, h = img.size
    return img.transpose(Image.FLIP_LEFT_RIGHT), np.float32( [[-1,0,w-1],[0,1,0],[0,0,1]] )

def flip_image_pair(img1, img2, gt):
    img1, F1 = flip(img1)
    img2, F2 = flip(img2)
    res = {}
    for key, value in gt.items():
        if key == 'homography':
            res['homography'] = F2 @ value @ F1
        elif key == 'aflow':
            assert False, 'flip for aflow: todo'
        elif key == 'corres':
            new_corres = np.c_[applyh(F1,value[:,0:2]), applyh(F2,value[:,2:4])]
            if value.shape[1] == 4: pass
            elif value.shape[1] == 6:
                scale, rot = myF.decode_scale_rot(value[:,5])
                new_code = myF.encode_scale_rot(scale, -rot)
                new_corres = np.c_[new_corres,value[:,4],new_code]
            res['corres'] = new_corres
        else:
            raise ValueError(f"flip_image_pair: bad gt field '{key}'")
    return img1, img2, res


def spatial_relationship( img1, img2, gt ):
    if 'homography' in gt:
        homography = gt['homography']
        if 'homography' in img2: 
            homography = np.float32(img2['homography']) @ homography
        corres = corres_from_homography(homography, *img1.size)

    elif 'corres' in gt:
        homography = np.full((3,3), np.nan, dtype=np.float32)
        corres = gt['corres']
        if 'homography' in img2:
            corres[:,2:4] = applyh(img2['homography'], corres[:,2:4])
        else:
            img2['homography'] = np.eye(3)
        scales = np.sqrt(np.abs(np.linalg.det(jacobianh(img2['homography'], corres[:,0:2]).T)))

        if corres.shape[1] == 4:
            scales, rots = scale_rot_from_corres(corres)
            corres = np.c_[corres, np.ones_like(scales), myF.encode_scale_rot(scales,rots*180/np.pi), scales]
        elif corres.shape[1] == 6:
            corres = np.c_[corres, scales * myF.decode_scale_rot(corres[:,5])[0]]
        else:
            assert ValueError(f'bad shape for corres: {corres.shape}')

    return homography, corres


def scale_rot_from_corres( corres, sub=256, nn=16 ):
    # select a subset of relevant correspondences
    sub = np.random.choice(len(corres), size=min(len(corres),sub), replace=False)
    sub = corres[sub]

    # for each corres, find the scale change w.r.t. its NNs
    from scipy.spatial.distance import cdist
    nns = cdist(corres, sub, metric='sqeuclidean').argsort(axis=1)[:,:nn]

    # affine transform for this set of neighboring correspondences
    pts = sub[nns] # shape = npts x sub x 4
    # [P1,1] @ A = P2  with A = 3x2 matrix
    # A = [P1,1]^-1 @ P2
    P1, P2 = pts[:,:,0:2], pts[:,:,2:4] # each row = list of correspondences
    P1 = np.concatenate((P1,np.ones_like(P1[:,:,:1])),axis=-1)
    A = (np.linalg.pinv(P1) @ P2).transpose(0,2,1)

    scale, (angy,angx) = detect_scale_rotation(A.transpose(1,2,0)[:,1::-1])
    rot = np.arctan2(angy, angx)
    return scale.clip(min=0.2, max=5), rot


def window1(x, size, w):
    l = x - int(0.5 + size / 2)
    r = l + int(0.5 + size)
    if l < 0: l,r = (0, r - l)
    if r > w: l,r = (l + w - r, w)
    if l < 0: l,r = 0,w # larger than width
    return slice(l,r)

def window(cx, cy, win_size, scale, img_shape):
    return (window1(int(cy), win_size*scale, img_shape[0]), 
            window1(int(cx), win_size*scale, img_shape[1]))

def is_in( pts, window ):
    x, y = pts.T
    sly, slx = window
    return (slx.start <= x) & (x < slx.stop) & (sly.start <= y) & (y < sly.stop)

def score_windows( valid1, valid2 ):
    inter = (valid1 & valid2).sum()
    iou1 = inter / (valid1.sum() + 1e-8)
    iou2 = inter / (valid2.sum() + 1e-8)
    return inter * min(iou1, iou2)

def imresize( img, max_size, resample=Image.ANTIALIAS):
    if max(img.shape[:2]) > max_size:
        if img.shape[-1] == 2:
            img = np.stack([np.float32(Image.fromarray(img[...,i]).resize((max_size,max_size), resample=resample)) for i in range(2)], axis=-1)
        else:
            img = np.asarray(Image.fromarray(img).resize((max_size,max_size), resample=resample))
    assert img.shape[0] == img.shape[1] == max_size, bb()
    return img

def wintrf( window, final_img ):
    wy, wx = window
    H, W = final_img.shape[:2]
    T = np.float32((((wx.stop-wx.start)/W, 0, wx.start),
                    (0, (wy.stop-wy.start)/H, wy.start),
                    (0, 0, 1)) )
    return invh(T)


def collate_ordered(batch, _use_shared_memory=True):
    pairs, gt = zip(*batch)
    imgs1, imgs2 = zip(*pairs)
    assert len(imgs1) == len(imgs2) == len(gt) and isinstance(gt[0], dict)
    
    # reorder samples (supervised ones first, unsupervised ones last)
    supervised = [i for i,b in enumerate(gt) if np.isfinite(b['homography']).all()]
    unsupervsd = [i for i,b in enumerate(gt) if np.isnan(b['homography']).any()]
    order = supervised + unsupervsd

    def collate( tensors, key=None ):
        import torch
        batch = todevice([tensors[i] for i in order], 'cpu')
        if key == 'corres': return batch # cannot concat
        if _use_shared_memory: # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, dim=0, out=out)

    return (collate(imgs1), collate(imgs2)), {k:collate([b[k] for b in gt],k) for k in gt[0]}


if __name__ == '__main__':
    from datasets import *
    from tools.viz import show_random_pairs

    db = BalancedCatImagePairs(
                3125, SyntheticImagePairs(RandomWebImages(0,52),distort='RandomTilting(0.5)'),
                4875, SyntheticImagePairs(SfM120k_Images(),distort='RandomTilting(0.5)'),
                8000, SfM120k_Pairs())

    db = FastPairLoader(db, 
            crop=256, transform='RandomRotation(20), RandomScale(256,1536,ar=1.3,can_upscale=True), PixelNoise()',
            p_swap=0.5, p_flip=0.5, scale_jitter=0, seed=777)

    show_random_pairs(db)
