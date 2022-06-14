# Copyright 2022-present NAVER Corp.
# CC BY-NC-SA 4.0
# Available only for non-commercial use

from pdb import set_trace as bb
import os, os.path as osp
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch

from .image_set import ImageSet
from .transforms import instanciate_transforms
from .utils import DatasetWithRng
invh = np.linalg.inv


class ImagePairs (DatasetWithRng):
    """ Base class for a dataset that serves image pairs.
    """
    imgs = None # regular image dataset
    pairs = [] # list of (idx1, idx2), ...

    def __init__(self, image_set, pairs, trf=None, **rng):
        assert image_set and pairs, 'empty images or pairs'
        super().__init__(**rng)
        self.imgs = image_set
        self.pairs = pairs
        self.trf = instanciate_transforms(trf, rng=self.rng)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        transform = self.trf or (lambda x:x)
        pair = tuple(map(transform, self._load_pair(idx)))
        return pair, {}

    def _load_pair(self, idx):
        i,j = self.pairs[idx]
        img1 = self.imgs.get_image(i)
        return (img1, img1) if i == j else (img1, self.imgs.get_image(j))

    def __repr__(self):
        return f'{self.__class__.__name__}({len(self)} pairs from {self.imgs})'


class StillImagePairs (ImagePairs):
    """ A dataset of 'still' image pairs used for debugging purposes.
    """
    def __init__(self, image_set, pairs=None, **rng):
        if isinstance(image_set, ImagePairs):
            super().__init__(image_set.imgs, pairs or image_set.pairs, **rng)
        else:
            super().__init__(image_set, pairs or [(i,i) for i in range(len(image_set))], **rng)

    def __getitem__(self, idx):
        img1, img2 = self._load_pair(idx)
        sx, sy = img2.size / np.float32(img1.size)
        return (img1, img2), dict(homography=np.diag(np.float32([sx, sy, 1])))


class SyntheticImagePairs (StillImagePairs):
    """ A synthetic generator of image pairs.
        Given a normal image dataset, it constructs pairs using random homographies & noise.

    scale: prior image scaling.
    distort: distortion applied independently to (img1,img2) if sym=True else just img2
    sym: (bool) see above.
    """
    def __init__(self, image_set, scale='', distort='', sym=False, **rng):
        super().__init__(image_set, **rng)
        self.symmetric = sym
        self.scale = instanciate_transforms(scale, rng=self.rng)
        self.distort = instanciate_transforms(distort, rng=self.rng)

    def __getitem__(self, idx):
        (img1, img2), gt = super().__getitem__(idx)

        img1 = dict(img=img1, homography=np.eye(3,dtype=np.float32))
        if img1['img'] is img2:
            img1 = self.scale(img1)
            img2 = self.distort(dict(img1))
            if self.symmetric: img1 = self.distort(img1)
        else:
            if self.symmetric: img1 = self.distort(self.scale(img1))
            img2 = self.distort(self.scale(dict(img=img2, **gt)))

        return (img1['img'], img2['img']), dict(homography=img2['homography'] @ invh(img1['homography']))

    def __repr__(self):
        format = lambda s: ','.join(l.strip() for l in repr(s).splitlines() if l).replace(',','',1)
        return f"{self.__class__.__name__}({len(self)} images, scale={format(self.scale)}, distort={format(self.distort)})"


class CatImagePairs (DatasetWithRng):
    """ Concatenation of several ImagePairs datasets
    """
    def __init__(self, *pair_datasets, seed=torch.initial_seed()):
        assert all(isinstance(db, ImagePairs) for db in pair_datasets)
        self.pair_datasets = pair_datasets
        DatasetWithRng.__init__(self, seed=seed) # init last
        self._init()

    def _init(self):
        self._pair_offsets = np.cumsum([0] + [len(db) for db in self.pair_datasets])
        self.npairs = self._pair_offsets[-1]

    def __len__(self):
        return self.npairs

    def __repr__(self):
        fmt_str = f"{type(self).__name__}({len(self)} pairs,"
        for i,db in enumerate(self.pair_datasets):
            npairs = self._pair_offsets[i+1] - self._pair_offsets[i]
            fmt_str += f'\n\t{npairs} from '+str(db).replace("\n"," ") + ','
        return fmt_str[:-1] + ')'

    def __getitem__(self, idx):
        b, i = self._which(idx)
        return self.pair_datasets[b].__getitem__(i)

    def _which(self, i):
        pos = np.searchsorted(self._pair_offsets, i, side='right')-1
        assert pos < self.npairs, 'Bad pair index %d >= %d' % (i, self.npairs)
        return pos, i - self._pair_offsets[pos]

    def _call(self, func, i, *args, **kwargs):
        b, j = self._which(i)
        return getattr(self.pair_datasets[b], func)(j, *args, **kwargs)

    def init_worker(self, tid):
        for db in self.pair_datasets:
            db.init_worker(tid)


class BalancedCatImagePairs (CatImagePairs):
    """ Balanced concatenation of several ImagePairs datasets
    """
    def __init__(self, npairs=0, *pair_datasets, **kw):
        assert isinstance(npairs, int) and npairs >= 0, 'BalancedCatImagePairs(npairs != int)'
        assert len(pair_datasets) > 0, 'no dataset provided'

        if len(pair_datasets) >= 3 and isinstance(pair_datasets[1], int):
            assert len(pair_datasets) % 2 == 1
            pair_datasets = [npairs] + list(pair_datasets)
            npairs, pair_datasets = pair_datasets[0::2], pair_datasets[1::2]
            assert all(isinstance(n, int) for n in npairs)
            self._pair_offsets = np.cumsum([0]+npairs)
            self.npairs = self._pair_offsets[-1]
        else:
            self.npairs = npairs or max(len(db) for db in pair_datasets)
            self._pair_offsets = np.linspace(0, self.npairs, len(pair_datasets)+1).astype(int)
        CatImagePairs.__init__(self, *pair_datasets, **kw)

    def set_epoch(self, epoch):
        DatasetWithRng.init_worker(self, epoch) # random seed only depends on the epoch
        self._init() # reset permutations for this epoch

    def init_worker(self, tid):
        CatImagePairs.init_worker(self, tid) 

    def _init(self):
        self._perms = []
        for i,db in enumerate(self.pair_datasets):
            assert len(db), 'cannot balance if there is an empty dataset'
            avail = self._pair_offsets[i+1] - self._pair_offsets[i]
            idxs = np.arange(len(db))
            while len(idxs) < avail: 
                idxs = np.r_[idxs,idxs]
            if self.seed: # if not seed, then no shuffle
                self.rng.shuffle(idxs[(avail//len(db))*len(db):])
            self._perms.append( idxs[:avail] )
        # print(self._perms)

    def _which(self, i):
        pos, idx = super()._which(i)
        return pos, self._perms[pos][idx]


class UnsupervisedPairs (ImagePairs):
    """ Unsupervised image pairs obtained from SfM
    """
    def __init__(self, img_set, pair_file_path):
        assert isinstance(img_set, ImageSet), bb()
        self.pair_list = self._parse_pair_list(pair_file_path)
        self.corres_dir = osp.join(osp.split(pair_file_path)[0], 'corres')

        tag_to_idx = {n:i for i,n in enumerate(img_set.imgs)}
        img_indices = lambda pair: tuple([tag_to_idx[n] for n in pair])
        super().__init__(img_set, [img_indices(pair) for pair in self.pair_list])

    def __repr__(self):
        return f"{type(self).__name__}({len(self)} pairs from {self.imgs})"

    def _parse_pair_list(self, pair_file_path):
        res = []
        for row in open(pair_file_path).read().splitlines():
            row = row.split()
            if len(row) != 2: raise IOError()
            res.append((row[0], row[1]))
        return res

    def get_corres_path(self, pair_idx):
        img1, img2 = [osp.basename(self.imgs.imgs[i]) for i in self.pairs[pair_idx]]
        return osp.join(self.corres_dir, f'{img1}_{img2}.npy')

    def get_corres(self, pair_idx):
        return np.load(self.get_corres_path(pair_idx))

    def __getitem__(self, idx):
        img1, img2 = self._load_pair(idx)
        return (img1, img2), dict(corres=self.get_corres(idx))

'''
TODO remove

def flow2png(flow, path):
    flow = np.clip(np.around(16*flow), -2**15, 2**15-1)
    bytes = np.int16(flow).view(np.uint8)
    Image.fromarray(bytes).save(path)
    return flow / 16

def png2flow(path, scale = 1):
    try:
        img = Image.open(path)
        if scale != 1:
            new_size = tuple(int(x*scale) for x in img.size)
            img = img.resize(new_size, Image.NEAREST)
        flow = np.asarray(img).view(np.int16)
        return np.float32(flow) * (scale / 16)
    except:
        raise IOError(2, "Error loading flow for %s" % path, path)


def imsize( img ):
    " returns (width, height) "
    if isinstance(img, Image.Image):
        return img.size
    if isinstance(img, np.ndarray):
        assert img.ndim == 3
        return img.shape[1::-1]
    if isinstance(img, torch.Tensor):
        if img.ndim == 4: img = img[0]
        assert img.ndim == 3
        return img.shape[2:0:-1] if img.shape[0] <= 3 else img.shape[1::-1]


def make_grid( img ):
    W, H = imsize(img)
    grid = np.mgrid[:H,:W][::-1].transpose(1,2,0).astype(np.float32)
    if isinstance(img, torch.Tensor):
        grid = torch.from_numpy(grid).to(img)
        if img.ndim == 4: grid = grid[None].expand(len(img), *grid.shape)
    return grid

    
def make_aflow( img, axis=None, **infos ):
    if axis is None:
        if 'aflow' in infos:
            return infos['aflow']
        grid = make_grid( img )
        if 'flow' in infos:
            return infos['flow'] + grid
        if 'homography' in infos:
            return applyh(infos['homography'], grid)
        raise RuntimeError(f'could not make aflow from {infos.keys()}')
    else:
        aflow = make_aflow(img, **infos)
        if (axis % aflow.ndim) - aflow.ndim != -1: 
            return aflow if aflow.shape[axis]==2 else aflow.swapaxes(-2,-1).swapaxes(-3,-2) #.transpose(2,0,1)
        else: 
            return aflow if aflow.shape[axis]==2 else aflow.swapaxes(-3,-2).swapaxes(-2,-1) #.transpose(1,2,0)
'''

if __name__ == '__main__':
    from datasets import *
    from tools.viz import show_random_pairs

    db = BalancedCatImagePairs(
                3125, SyntheticImagePairs(RandomWebImages(0,52),distort='RandomTilting(0.5)'),
                4875, SyntheticImagePairs(SfM120k_Images(),distort='RandomTilting(0.5)'),
                8000, SfM120k_Pairs())

    show_random_pairs(db)
    
