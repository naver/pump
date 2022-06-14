# Copyright 2022-present NAVER Corp.
# CC BY-NC-SA 4.0
# Available only for non-commercial use

from pdb import set_trace as bb
import warnings

import numpy as np
from PIL import Image, ImageOps

import torch
import torch.nn as nn
from torchvision import transforms as tvf

from . import transforms_tools as F
from .utils import DatasetWithRng

'''
Example command to try out some transformation chain:

python -m pytools.transforms --trfs "Scale(384), ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1), RandomRotation(10), RandomTilting(0.5, 'all'), RandomScale(240,320), RandomCrop(224)"
'''

def instanciate_transforms(transforms, use_gpu=False, rng=None, compose=True):
    ''' Instanciate a sequence of transformations.
    
    transforms: (str, list) 
        Comma-separated list of transformations.
        Ex: "Rotate(10), Scale(256)"
    '''
    try:
        transforms = transforms or '[]'

        if isinstance(transforms, str):
            if transforms.lstrip()[0] not in '[(': transforms = f'[{transforms}]'
            if compose: transforms = f'Compose({transforms})'
            transforms = eval(transforms)

        if isinstance(transforms, list) and transforms and isinstance(transforms[0], str):
            transforms = [eval(trf) for trf in transforms]
            if compose: transforms = Compose(transforms)

        if use_gpu and not isinstance(transforms, nn.Module):
            while hasattr(transforms,'transforms') or hasattr(transforms,'transform'): 
                transforms = getattr(transforms,'transforms',getattr(transforms,'transform',None))
            transforms = [trf for trf in transforms if isinstance(trf, nn.Module)]
            transforms = nn.Sequential(*transforms) if compose else nn.ModuleList(transforms)

        if transforms and rng: 
            for trf in transforms.transforms: 
                assert hasattr(trf, 'rng'), f"Transformation {trf} has no self.rng"
                trf.rng = rng

        if isinstance(transforms, Compose) and len(transforms.transforms) == 1:
            transforms = transforms.transforms[0]
        return transforms

    except Exception as e:
        print("\nError: Cannot interpret this transform list: %s\n" % transforms)
        raise e



class Compose (DatasetWithRng):
    def __init__(self, transforms, **rng_seed):
        super().__init__(**rng_seed)
        self.transforms = [self.with_same_rng(trf) for trf in transforms]

    def __call__(self, data):
        for trf in self.transforms:
            data = trf(data)
        return data


class Scale (DatasetWithRng):
    """ Rescale the input PIL.Image to a given size.
    Copied from https://github.com/pytorch in torchvision/transforms/transforms.py
    
    The smallest dimension of the resulting image will be = size.
    
    if largest == True: same behaviour for the largest dimension.
    
    if not can_upscale: don't upscale
    if not can_downscale: don't downscale
    """
    def __init__(self, size, interpolation=Image.BILINEAR, largest=False, 
                 can_upscale=True, can_downscale=True, **rng_seed):
        super().__init__(**rng_seed)
        assert isinstance(size, int) or (len(size) == 2)
        self.size = size
        self.interpolation = interpolation
        self.largest = largest
        self.can_upscale = can_upscale
        self.can_downscale = can_downscale

    def __repr__(self):
        fmt_str = "RandomScale(%s" % str(self.size)
        if self.largest: fmt_str += ', largest=True'
        if not self.can_upscale: fmt_str += ', can_upscale=False'
        if not self.can_downscale: fmt_str += ', can_downscale=False'
        return fmt_str+')'

    def get_params(self, imsize):
        w,h = imsize
        if isinstance(self.size, int):
            cmp = lambda a,b: (a>=b) if self.largest else (a<=b)
            if (cmp(w, h) and w == self.size) or (cmp(h, w) and h == self.size):
                ow, oh = w, h
            elif cmp(w, h):
                ow = self.size
                oh = int(self.size * h / w)
            else:
                oh = self.size
                ow = int(self.size * w / h)
        else:
            ow, oh = self.size
        return ow, oh

    def __call__(self, inp):
        img = F.grab(inp,'img')
        w, h = img.size
        
        size2 = ow, oh = self.get_params(img.size)
        
        if size2 != img.size:
            a1, a2 = img.size, size2
            if (self.can_upscale and min(a1) < min(a2)) or (self.can_downscale and min(a1) > min(a2)):
                img = img.resize(size2, self.interpolation)

        return F.update(inp, img=img, homography=np.diag((ow/w,oh/h,1)))



class RandomScale (Scale):
    """Rescale the input PIL.Image to a random size.
    Copied from https://github.com/pytorch in torchvision/transforms/transforms.py

    Args:
        min_size (int): min size of the smaller edge of the picture.
        max_size (int): max size of the smaller edge of the picture.

        ar (float or tuple):
            max change of aspect ratio (width/height).

        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, min_size, max_size, ar=1, larger=False,
                 can_upscale=False, can_downscale=True, interpolation=Image.BILINEAR):
        Scale.__init__(self, (min_size,max_size), can_upscale=can_upscale, can_downscale=can_downscale, interpolation=interpolation)
        assert type(min_size) == type(max_size), 'min_size and max_size can only be 2 ints or 2 floats'
        assert isinstance(min_size, int) and min_size >= 1 or isinstance(min_size, float) and min_size>0
        assert isinstance(max_size, (int,float)) and min_size <= max_size
        self.min_size = min_size
        self.max_size = max_size
        if type(ar) in (float,int): ar = (min(1/ar,ar),max(1/ar,ar))
        assert 0.2 < ar[0] <= ar[1] < 5
        self.ar = ar
        self.larger = larger

    def get_params(self, imsize):
        w,h = imsize
        if isinstance(self.min_size, float): min_size = int(self.min_size*min(w,h) + 0.5)
        if isinstance(self.max_size, float): max_size = int(self.max_size*min(w,h) + 0.5)
        if isinstance(self.min_size, int):   min_size = self.min_size
        if isinstance(self.max_size, int):   max_size = self.max_size
        
        if not(self.can_upscale) and not(self.larger):
            max_size = min(max_size,min(w,h))

        size = int(0.5 + F.rand_log_uniform(self.rng, min_size, max_size))
        if not(self.can_upscale) and self.larger:
            size = min(size, min(w,h))

        ar = F.rand_log_uniform(self.rng, *self.ar) # change of aspect ratio

        if w < h: # image is taller
            ow = size
            oh = int(0.5 + size * h / w / ar)
            if oh < min_size:
                ow,oh = int(0.5 + ow*float(min_size)/oh),min_size
        else: # image is wider
            oh = size
            ow = int(0.5 + size * w / h * ar)
            if ow < min_size:
                ow,oh = min_size,int(0.5 + oh*float(min_size)/ow)
                
        assert ow >= min_size, 'image too small (width=%d < min_size=%d)' % (ow, min_size)
        assert oh >= min_size, 'image too small (height=%d < min_size=%d)' % (oh, min_size)
        return ow, oh



class RandomCrop (DatasetWithRng):
    """Crop the given PIL Image at a random location.
    Copied from https://github.com/pytorch in torchvision/transforms/transforms.py

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
    """

    def __init__(self, size, padding=0, **rng_seed):
        super().__init__(**rng_seed)
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __repr__(self):
        return "RandomCrop(%s)" % str(self.size)

    def get_params(self, img, output_size):
        w, h = img.size
        th, tw = output_size
        assert h >= th and w >= tw, "Image of %dx%d is too small for crop %dx%d" % (w,h,tw,th)

        y = self.rng.integers(0, h - th) if h > th else 0
        x = self.rng.integers(0, w - tw) if w > tw else 0
        return x, y, tw, th

    def __call__(self, inp):
        img = F.grab(inp,'img')

        padl = padt = 0
        if self.padding:
            if F.is_pil_image(img):
                img = ImageOps.expand(img, border=self.padding, fill=0)
            else:
                assert isinstance(img, F.DummyImg)
                img = img.expand(border=self.padding)
            if isinstance(self.padding, int):
                padl = padt = self.padding
            else:
                padl, padt = self.padding[0:2]

        i, j, tw, th = self.get_params(img, self.size)
        img = img.crop((i, j, i+tw, j+th))
        
        return F.update(inp, img=img, homography=np.float32(((1,0,padl-i),(0,1,padt-j),(0,0,1))))


class CenterCrop (RandomCrop):
    """Crops the given PIL Image at the center.
    Copied from https://github.com/pytorch in torchvision/transforms/transforms.py

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """
    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        y = int(0.5 +((h - th) / 2.))
        x = int(0.5 +((w - tw) / 2.))
        return x, y, tw, th


class RandomRotation (DatasetWithRng):
    """Rescale the input PIL.Image to a random size.
    Copied from https://github.com/pytorch in torchvision/transforms/transforms.py

    Args:
        degrees (float):
            rotation angle.

        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, degrees, interpolation=Image.BILINEAR, **rng_seed):
        super().__init__(**rng_seed)
        self.degrees = degrees
        self.interpolation = interpolation

    def __repr__(self):
        return f"RandomRotation({self.degrees})"

    def __call__(self, inp):
        img = F.grab(inp,'img')
        w, h = img.size
        
        angle = self.rng.uniform(-self.degrees, self.degrees)
        
        img = img.rotate(angle, resample=self.interpolation)
        w2, h2 = img.size

        trf = F.translate(w2/2,h2/2) @ F.rotate(-angle * np.pi/180) @ F.translate(-w/2,-h/2)
        return F.update(inp, img=img, homography=trf)


class RandomTilting (DatasetWithRng):
    """Apply a random tilting (left, right, up, down) to the input PIL.Image
    Copied from https://github.com/pytorch in torchvision/transforms/transforms.py

    Args:
        maginitude (float):
            maximum magnitude of the random skew (value between 0 and 1)
        directions (string):
            tilting directions allowed (all, left, right, up, down)
            examples: "all", "left,right", "up-down-right"
    """

    def __init__(self, magnitude, directions='all', **rng_seed):
        super().__init__(**rng_seed)
        self.magnitude = magnitude
        self.directions = directions.lower().replace(',',' ').replace('-',' ')

    def __repr__(self):
        return "RandomTilt(%g, '%s')" % (self.magnitude,self.directions)

    def __call__(self, inp):
        img = F.grab(inp,'img')
        w, h = img.size

        x1,y1,x2,y2 = 0,0,h,w
        original_plane = [(y1, x1), (y2, x1), (y2, x2), (y1, x2)]

        max_skew_amount = max(w, h)
        max_skew_amount = int(np.ceil(max_skew_amount * self.magnitude))
        skew_amount = self.rng.integers(1, max_skew_amount)

        if self.directions == 'all':
            choices = [0,1,2,3]
        else:
            dirs = ['left', 'right', 'up', 'down']
            choices = []
            for d in self.directions.split():
                try:
                    choices.append(dirs.index(d))
                except:
                    raise ValueError('Tilting direction %s not recognized' % d)

        skew_direction = self.rng.choice(choices)

        # print('randomtitlting: ', skew_amount, skew_direction) # to debug random

        if skew_direction == 0:
            # Left Tilt
            new_plane = [(y1, x1 - skew_amount),  # Top Left
                         (y2, x1),                # Top Right
                         (y2, x2),                # Bottom Right
                         (y1, x2 + skew_amount)]  # Bottom Left
        elif skew_direction == 1:
            # Right Tilt
            new_plane = [(y1, x1),                # Top Left
                         (y2, x1 - skew_amount),  # Top Right
                         (y2, x2 + skew_amount),  # Bottom Right
                         (y1, x2)]                # Bottom Left
        elif skew_direction == 2:
            # Forward Tilt
            new_plane = [(y1 - skew_amount, x1),  # Top Left
                         (y2 + skew_amount, x1),  # Top Right
                         (y2, x2),                # Bottom Right
                         (y1, x2)]                # Bottom Left
        elif skew_direction == 3:
            # Backward Tilt
            new_plane = [(y1, x1),                # Top Left
                         (y2, x1),                # Top Right
                         (y2 + skew_amount, x2),  # Bottom Right
                         (y1 - skew_amount, x2)]  # Bottom Left

        # To calculate the coefficients required by PIL for the perspective skew,
        # see the following Stack Overflow discussion: https://goo.gl/sSgJdj
        homography = F.homography_from_4pts(original_plane, new_plane)
        img =  img.transform(img.size, Image.PERSPECTIVE, homography, resample=Image.BICUBIC)

        homography = np.linalg.pinv(np.float32(homography+(1,)).reshape(3,3))
        return F.update(inp, img=img, homography=homography)


RandomHomography = RandomTilt = RandomTilting # redefinition


class Homography(object):
    """Apply a known tilting to an image
    """
    def __init__(self, *homography):
        assert len(homography) == 8
        self.homography = homography
    
    def __call__(self, inp):
        img = F.grab(inp, 'img')
        homography = self.homography
        
        img =  img.transform(img.size, Image.PERSPECTIVE, homography, resample=Image.BICUBIC)

        homography = np.linalg.pinv(np.float32(list(homography)+[1]).reshape(3,3))
        return F.update(inp, img=img, homography=homography)



class StillTransform (DatasetWithRng):
    """ Takes and return an image, without changing its shape or geometry.
    """
    def _transform(self, img):
        raise NotImplementedError()
        
    def __call__(self, inp):
        img = F.grab(inp,'img')

        # transform the image (size should not change)
        try:
            img = self._transform(img)
        except TypeError:
            pass

        return F.update(inp, img=img)



class PixelNoise (StillTransform):
    """ Takes an image, and add random white noise.
    """
    def __init__(self, ampl=20, **rng_seed):
        super().__init__(**rng_seed)
        assert 0 <= ampl < 255
        self.ampl = ampl

    def __repr__(self):
        return "PixelNoise(%g)" % self.ampl

    def _transform(self, img):
        img = np.float32(img)
        img += self.rng.uniform(0.5-self.ampl/2, 0.5+self.ampl/2, size=img.shape)
        return Image.fromarray(np.uint8(img.clip(0,255)))



class ColorJitter (StillTransform):
    """Randomly change the brightness, contrast and saturation of an image.
    Copied from https://github.com/pytorch in torchvision/transforms/transforms.py

    Args:
    brightness (float): How much to jitter brightness. brightness_factor
    is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
    contrast (float): How much to jitter contrast. contrast_factor
    is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
    saturation (float): How much to jitter saturation. saturation_factor
    is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
    hue(float): How much to jitter hue. hue_factor is chosen uniformly from
    [-hue, hue]. Should be >=0 and <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __repr__(self):
        return "ColorJitter(%g,%g,%g,%g)" % (
            self.brightness, self.contrast, self.saturation, self.hue)
    
    def get_params(self, brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Returns:
        Transform which randomly adjusts brightness, contrast and
        saturation in a random order.
        """
        transforms = []
        if brightness > 0:
            brightness_factor = self.rng.uniform(max(0, 1 - brightness), 1 + brightness)
            transforms.append(tvf.Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast > 0:
            contrast_factor = self.rng.uniform(max(0, 1 - contrast), 1 + contrast)
            transforms.append(tvf.Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation > 0:
            saturation_factor = self.rng.uniform(max(0, 1 - saturation), 1 + saturation)
            transforms.append(tvf.Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue > 0:
            hue_factor = self.rng.uniform(-hue, hue)
            transforms.append(tvf.Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        # print('colorjitter: ', brightness_factor, contrast_factor, saturation_factor, hue_factor) # to debug random seed
        self.rng.shuffle(transforms)
        transform = tvf.Compose(transforms)
        return transform

    def _transform(self, img):
        transform = self.get_params(self.brightness, self.contrast, self.saturation, self.hue)
        return transform(img)


def pil_loader(path, mode='RGB'):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with (path if hasattr(path,'read') else open(path, 'rb')) as f:
            img = Image.open(f)
            return img.convert(mode)

def torchvision_loader(path, mode='RGB'):
    from torchvision.io import read_file, decode_image, read_image, image
    return read_image(getattr(path,'name',path), mode=getattr(image.ImageReadMode,mode))



if __name__ == '__main__':
    from matplotlib import pyplot as pl
    import argparse

    parser = argparse.ArgumentParser("Script to try out and visualize transformations")
    parser.add_argument('--img', type=str, default='imgs/test.png', help='input image')
    parser.add_argument('--trfs', type=str, required=True, help='list of transformations')
    parser.add_argument('--layout', type=int, nargs=2, default=(3,3), help='nb of rows,cols')
    args = parser.parse_args()
    
    img = dict(img=pil_loader(args.img))

    trfs = instanciate_transforms(args.trfs)

    pl.subplots_adjust(0,0,1,1)
    nr,nc = args.layout

    while True:
        t0 = now()
        imgs2 = [trfs(img) for _ in range(nr*nc)]

        for j in range(nr):
            for i in range(nc):
                pl.subplot(nr,nc,i+j*nc+1)
                img2 = img if i==j==0 else imgs2.pop() #trfs(img)
                img2 = img2['img']
                pl.imshow(img2)
                pl.xlabel("%d x %d" % img2.size)
        print(f'Took {now() - t0:.2f} seconds')
        pl.show()
