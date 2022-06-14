# Copyright 2022-present NAVER Corp.
# CC BY-NC-SA 4.0
# Available only for non-commercial use

import pdb, sys, os
import argparse
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, triu, csgraph

import core.functional as myF
from tools.common import image, image_with_trf
from tools.viz import dbgfig, show_correspondences


def arg_parser():
    parser = argparse.ArgumentParser("Post-filtering of Deep matching correspondences")

    parser.add_argument("--img1", required=True, help="path to first image")
    parser.add_argument("--img2", required=True, help="path to second image")
    parser.add_argument("--resize", default=0, type=int, help="prior image downsize (0 if recursive)")
    parser.add_argument("--corres", required=True, help="input path")
    parser.add_argument("--output", default="", help="filtered corres output")

    parser.add_argument("--locality", type=float, default=2, help="tolerance to deformation")
    parser.add_argument("--min-cc-size", type=int, default=50, help="min connex-component size")
    parser.add_argument("--densify", default='no', choices=['no','full','cc','convex'], help="output pixel-dense corres field")
    parser.add_argument("--dense-side", default='left', choices=['left','right'], help="img to densify")

    parser.add_argument("--verbose", "-v", type=int, default=0, help="verbosity level")
    parser.add_argument("--dbg", type=str, nargs='+', default=(), help="debug options")
    return parser


def main(args):
    import test_singlescale as pump
    corres = np.load(args.corres)['corres']
    imgs = tuple(map(image, pump.Main.load_images(args)))

    if dbgfig('raw',args.dbg):
        show_correspondences(*imgs, corres)

    corres = filter_corres( *imgs, corres, 
        locality=args.locality, min_cc_size=args.min_cc_size, 
        densify=args.densify, dense_side=args.dense_side,
        verbose=args.verbose, dbg=args.dbg)

    if dbgfig('viz',args.dbg):
        show_correspondences(*imgs, corres)

    return pump.save_output( args, corres )


def filter_corres( img0, img1, corres, 
            locality = None, # graph edge locality
            min_cc_size = None, # min CC size
            densify = None, 
            dense_side = None,
            verbose = 0, dbg=()):

    if None in (locality, min_cc_size, densify, dense_side):
        default_params = arg_parser()
        locality = locality or default_params.get_default('locality')
        min_cc_size = min_cc_size or default_params.get_default('min_cc_size')
        densify = densify or default_params.get_default('densify')
        dense_side = dense_side or default_params.get_default('dense_side')

    img0, trf0 = img0 if isinstance(img0,tuple) else (img0, np.eye(3))
    img1, trf1 = img1 if isinstance(img1,tuple) else (img1, np.eye(3))
    assert isinstance(img0, np.ndarray) and isinstance(img1, np.ndarray)

    corres = myF.affmul((np.linalg.inv(trf0),np.linalg.inv(trf1)), corres)
    n_corres = len(corres)
    if verbose: print(f'>> input: {len(corres)} correspondences')

    graph = compute_graph(corres, max_dis=locality*4)
    if verbose: print(f'>> {locality=}: {graph.nnz} nodes in graph')

    cc_sizes = measure_connected_components(graph)
    corres[:,4] += np.log2(cc_sizes)
    corres = corres[cc_sizes > min_cc_size]
    if verbose: print(f'>> {min_cc_size=}: remaining {len(corres)} correspondences')

    final = myF.affmul((trf0,trf1), corres)

    if densify != 'no':
        # densify correspondences
        if dense_side == 'right': # temporary swap
            final = final[:,[2,3,0,1]]
            H = round(img1.shape[0] / trf1[1,1])
            W = round(img1.shape[1] / trf1[0,0])
        else:
            H = round(img0.shape[0] / trf0[1,1])
            W = round(img0.shape[1] / trf0[0,0])
        
        if densify == 'cc':
            assert False, 'todo'
        elif densify in (True, 'full', 'convex'):
            # recover true image0's shape
            final = densify_corres( final, (H, W), full=(densify!='convex') )
        else:
            raise ValueError(f'Bad mode for {densify=}')

        if dense_side == 'right': # undo temporary swap
            final = final[:,[2,3,0,1]]

    return final


def compute_graph(corres, max_dis=10, min_ang=90):
    """ 4D distances (corres can only be connected to same scale)
        using sparse matrices for efficiency

    step1: build horizontal and vertical binning, binsize = max_dis
           add in each bin all neighbor bins
    step2: for each corres, we can intersect 2 bins to get a short list of candidates
    step3: verify euclidean distance < maxdis (optional?)
    """
    def bin_positions(pos):
        # every corres goes into a single bin
        bin_indices = np.int32(pos.clip(min=0) // max_dis) + 1
        cols = np.arange(len(pos))
        
        # add the cell before and the cell after, to handle border effects
        res = csr_matrix((np.ones(len(bin_indices)*3,dtype=np.float32), 
            (np.r_[bin_indices-1, bin_indices, bin_indices+1], np.r_[cols,cols,cols])),
            shape=(bin_indices.max()+2 if bin_indices.size else 1, len(pos)))

        return res, bin_indices

    # 1-hot matrices of shape = nbins x n_corres
    x1_bins = bin_positions(corres[:,0])
    y1_bins = bin_positions(corres[:,1])
    x2_bins = bin_positions(corres[:,2])
    y2_bins = bin_positions(corres[:,3])    

    def row_indices(ngh):
        res = np.bincount(ngh.indptr[1:-1], minlength=ngh.indptr[-1])[:-1]
        return res.cumsum()

    def compute_dist( ngh, pts, scale=None ):
        # pos from the second point
        x_pos = pts[ngh.indices,0]
        y_pos = pts[ngh.indices,1]

        # subtract pos from the 1st point
        rows = row_indices(ngh)
        x_pos -= pts[rows, 0]
        y_pos -= pts[rows, 1]
        dis = np.sqrt(np.square(x_pos) + np.square(y_pos))
        if scale is not None: 
            # there is a scale for each of the 2 pts, we encline to choose the worst one
            dis *= (scale[rows] + scale[ngh.indices]) / 2 # so we use arithmetic instead of geometric mean

        return normed(np.c_[x_pos, y_pos]), dis

    def Rot( ngh, degrees ):
        rows = row_indices(ngh)
        rad = degrees * np.pi / 180
        rad = (rad[rows] + rad[ngh.indices]) / 2 # average angle between 2 corres
        cos, sin = np.cos(rad), np.sin(rad)
        return np.float32(((cos, -sin), (sin,cos))).transpose(2,0,1)

    def match(xbins, ybins, pt1, pt2, way):
        xb, ixb = xbins
        yb, iyb = ybins
        
        # gets for each corres a list of potential matches
        ngh = xb[ixb].multiply( yb[iyb] ) # shape = n_corres x n_corres    
        ngh = triu(ngh, k=1).tocsr() # remove mirrored matches
        # ngh = matches of matches, shape = n_corres x n_corres

        # verify locality and flow
        vec1, d1 = compute_dist(ngh, pt1) # for each match, distance and orientation in img1
        # assert d1.max()**0.5 < 2*max_dis*1.415, 'cannot be larger than 2 cells in diagonals, or there is a bug'+bb()
        scale, rot = myF.decode_scale_rot(corres[:,5])
        vec2, d2 = compute_dist(ngh, pt2, scale=scale**(-way))
        ang = np.einsum('ik,ik->i', (vec1[:,None] @ Rot(ngh,way*rot))[:,0], vec2)

        valid = (d1 <= max_dis) & (d2 <= max_dis) & (ang >= np.cos(min_ang*np.pi/180))
        res = csr_matrix((valid, ngh.indices, ngh.indptr), shape=ngh.shape)
        res.eliminate_zeros()
        return res

    # find all neihbors within each xy bin
    ngh1 = match(x1_bins, y1_bins, corres[:,0:2], corres[:,2:4], way=+1)
    ngh2 = match(x2_bins, y2_bins, corres[:,2:4], corres[:,0:2], way=-1).T

    return ngh1 + ngh2 # union
    

def measure_connected_components(graph, dbg=()):
    # compute connected components
    nc, labels = csgraph.connected_components(graph, directed=False)

    # filter and remove all small components
    count = np.bincount(labels)
    
    return count[labels]

def normed( mat ):
    return mat / np.linalg.norm(mat, axis=-1, keepdims=True).clip(min=1e-16)


def densify_corres( corres, shape, full=True ):
    from scipy.interpolate import LinearNDInterpolator
    from scipy.spatial import cKDTree as KDTree

    assert len(corres) > 3, 'Not enough corres for densification'
    H, W = shape

    interp = LinearNDInterpolator(corres[:,0:2], corres[:,2:4])
    X, Y = np.mgrid[0:H, 0:W][::-1] # H x W, H x W
    p1 = np.c_[X.ravel(), Y.ravel()]
    p2 = interp(X, Y) # H x W x 2

    p2 = p2.reshape(-1,2)
    invalid = np.isnan(p2).any(axis=1)

    if full:
        # interpolate pixels outside of the convex hull
        badp = p1[invalid]
        tree = KDTree(corres[:,0:2])
        _, nn = tree.query(badp, 3) # find 3 closest neighbors
        corflow = corres[:,2:4] - corres[:,0:2]
        p2.reshape(-1,2)[invalid] = corflow[nn].mean(axis=1) + p1[invalid]
    else:
        # remove nans, i.e. remove points outside of convex hull
        p1, p2 = p1[~invalid], p2[~invalid]

    # return correspondence field
    return np.c_[p1, p2]


if __name__ == '__main__':
    main(arg_parser().parse_args())
