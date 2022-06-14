# Copyright 2022-present NAVER Corp.
# CC BY-NC-SA 4.0
# Available only for non-commercial use

from pdb import set_trace as bb
import os, os.path as osp

from tqdm import trange
from .image_set import ImageSet, verify_img


class RandomWebImages (ImageSet):
    """ 1 million distractors from Oxford and Paris Revisited
        see http://ptak.felk.cvut.cz/revisitop/revisitop1m/
    """
    def __init__(self, start=0, end=52, root="datasets/revisitop1m"):
        bar = None
        imgs  = []
        for i in range(start, end):
            try: 
                # read cached list
                img_list_path = osp.join(root, "image_list_%d.txt"%i) 
                cached_imgs = [e.strip() for e in open(img_list_path)]
                assert cached_imgs, f"Cache '{img_list_path}' is empty!"
                imgs += cached_imgs

            except IOError:
                if bar is None: 
                    bar = trange(start, 4*end, desc='Caching')
                    bar.update(4*i)

                # create it
                imgs = []
                for d in range(i*4,(i+1)*4): # 4096 folders in total, on average 256 each
                    key = hex(d)[2:].zfill(3)
                    folder = osp.join(root, key)
                    if not osp.isdir(folder): continue
                    imgs += [f for f in os.listdir(folder) if verify_img(osp.join(folder, f), exts='.jpg')]
                    bar.update(1)
                assert imgs, f"No images found in {folder}/"
                open(img_list_path,'w').write('\n'.join(imgs))
                imgs += imgs

        if bar: bar.update(bar.total - bar.n)
        super().__init__(root, imgs)

    def get_image_path(self, idx):
        key = self.imgs[idx]
        return osp.join(self.root, key[:3], key)

