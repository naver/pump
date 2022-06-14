# Copyright 2022-present NAVER Corp.
# CC BY-NC-SA 4.0
# Available only for non-commercial use

from pdb import set_trace as bb
import os
from os.path import *
from PIL import Image


class ImageSet(object):
    """ Base class for an image dataset.
    """
    def __init__(self, root, imgs):
        self.root = root
        self.imgs = imgs
        assert imgs, f'Empty image set in {root}'

    def init_from_folder(self, *args, **kw):
        imset = ImageSet.from_folder(*args, **kw)
        ImageSet.__init__(self, imset.root, imset.imgs)

    def __len__(self):
        return len(self.imgs)

    def get_image_path(self, idx):
        return os.path.join(self.root, self.imgs[idx])

    def get_image(self, idx):
        fname = self.get_image_path(idx)
        try:
            return Image.open(fname).convert('RGB')
        except Exception as e:
            raise IOError("Could not load image %s (reason: %s)" % (fname, str(e)))

    __getitem__ = get_image

    @staticmethod
    def from_folder(root, exts=('.jpg','.jpeg','.png','.ppm'), recursive=False, listing=False, check_imgs=False):
        """
        recursive: bool or func. If a function, it must evaluate True to the directory name.
        """
        if listing: 
            if listing is True: listing = f"list_imgs{'_recursive' if recursive else ''}.txt"
            flist = join(root, listing)
            try: return ImageSet.from_listing(root,flist)
            except IOError: print(f'>> ImageSet.from_folder(listing=True): entering {root}...')

        if check_imgs is True: # default verif function
            check_imgs = verify_img

        for _, dirnames, dirfiles in os.walk(root):
            imgs = sorted([f for f in dirfiles if f.lower().endswith(exts)])
            if check_imgs: imgs = [img for img in imgs if check_imgs(join(root,img))]

            if recursive:
                for dirname in sorted(dirnames):
                    if callable(recursive) and not recursive(join(root,dirname)): continue
                    imset = ImageSet.from_folder(join(root,dirname), exts=exts, recursive=recursive, listing=listing, check_imgs=check_imgs)
                    imgs += [join(dirname,f) for f in imset.imgs]
            break # recursion is handled internally

        if listing: 
            try: open(flist,'w').write('\n'.join(imgs))
            except IOError: pass # write permission denied
        return ImageSet(root, imgs)

    @staticmethod
    def from_listing(root, list_path):
        return ImageSet(root, open(list_path).read().splitlines())

    def circular_pad(self, min_size):
        assert self.imgs, 'cannot pad an empty image set'
        while len(self.imgs) < min_size: 
            self.imgs += self.imgs # artifically augment size
        self.imgs = self.imgs[:min_size or None]
        return self

    def __repr__(self):
        prefix = os.path.commonprefix((self.get_image_path(0),self.get_image_path(len(self)-1)))
        return f'{self.__class__.__name__}({len(self)} images from {prefix}...)'



def verify_img(path, exts=None):
    if exts and not path.lower().endswith(exts): return False
    try: 
        Image.open(path).convert('RGB') # try to open it
        return True
    except: 
        return False
