# Copyright 2022-present NAVER Corp.
# CC BY-NC-SA 4.0
# Available only for non-commercial use

from pdb import set_trace as bb
from os.path import *

from .image_set import ImageSet
from .pair_dataset import UnsupervisedPairs


class SfM120k_Images (ImageSet):
    def __init__(self, root='datasets/sfm120k'): 
        self.init_from_folder(join(root,'ims'), recursive=True, listing=True, exts='')


class SfM120k_Pairs (UnsupervisedPairs):
    def __init__(self, root='datasets/sfm120k'):
        super().__init__(SfM120k_Images(root=root), join(root,'list_pairs.txt'))


if __name__ == '__main__':
    from tools.viz import show_random_pairs

    db = SfM120k_Pairs()

    show_random_pairs(db)
