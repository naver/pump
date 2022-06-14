# Copyright 2022-present NAVER Corp.
# CC BY-NC-SA 4.0
# Available only for non-commercial use

from pdb import set_trace as bb
import numpy as np
from PIL import Image, ImageOps, ImageEnhance


def grab( data, *fields ):
    ''' Called to extract fields from a dictionary
    '''
    if isinstance(data, dict):
        res = []
        for f in fields:
            res.append( data[f] )
        return res[0] if len(fields) == 1 else tuple(res)

    else: # or it must be the img directly
        assert fields == ('img',) and isinstance(data, (np.ndarray, Image.Image)), \
            f"data should be an image, not {type(data)}!"
        return data


def update( data, **fields):
    ''' Called to update the img_and_label
    '''
    if isinstance( data, dict):
        if 'homography' in fields and 'homography' in data:
            data['homography'] = fields.pop('homography') @ data['homography']
        data.update(fields)
        if 'img' in fields: 
            data['imsize'] = data['img'].size
        return data

    else: # or it must be the img directly
        return fields['img']


def rand_log_uniform(rng, a, b):
    return np.exp(rng.uniform(np.log(a),np.log(b)))


def translate(tx, ty):
    return np.float32(((1,0,tx),(0,1,ty,),(0,0,1)))

def rotate(angle):
    return np.float32(((np.cos(angle),-np.sin(angle),0),(np.sin(angle),np.cos(angle),0),(0,0,1)))


def is_pil_image(img):
    return isinstance(img, Image.Image)


def homography_from_4pts(pts_cur, pts_new):
    "pts_cur and pts_new = 4x2 point array, in [(x,y),...] format"
    matrix = []
    for p1, p2 in zip(pts_new, pts_cur):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])
    A = np.matrix(matrix, dtype=np.float)
    B = np.array(pts_cur).reshape(8)

    homography = np.dot(np.linalg.pinv(A), B)
    homography = tuple(np.array(homography).reshape(8))
    #print(homography)
    return homography




