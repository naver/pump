# Copyright 2022-present NAVER Corp.
# CC BY-NC-SA 4.0
# Available only for non-commercial use

from pdb import set_trace as bb
import numpy as np
import torch
import torch.nn.functional as F


def affmul( aff, vecs ):
    """ affine multiplication: 
        computes aff @ vecs.T """
    if aff is None: return vecs        
    if isinstance(aff, (tuple,list)) or aff.ndim==3:
        assert len(aff) == 2
        assert 4 <= vecs.shape[-1], bb()
        vecs = vecs.clone() if isinstance(vecs, torch.Tensor) else vecs.copy()
        vecs[...,0:2] = affmul(aff[0], vecs[...,0:2])
        vecs[...,2:4] = affmul(aff[1], vecs[...,2:4])
        return vecs
    else:
        assert vecs.shape[-1] == 2, bb()
        assert aff.shape == (2,3) or (aff.shape==(3,3) and
               aff[2,0] == aff[2,1] == 0 and aff[2,2] == 1), bb()
        return (vecs @ aff[:2,:2].T) + aff[:2,2]


def imresize( img, max_size, mode='area' ):
    # trf: cur_pix --> old_pix
    img, trf = img if isinstance(img,tuple) else (img, torch.eye(3,device=img.device))
    
    shape = img.shape[-2:]
    if max_size > 0 and max(shape) > max_size:
        new_shape = tuple(i * max_size // max(shape) for i in shape)
        img = F.interpolate( img[None].float(), size=new_shape, mode=mode )[0]
        img.clamp_(min=0, max=255)
        sca = torch.diag(torch.tensor((shape[0]/new_shape[0],shape[1]/new_shape[1],1), device=img.device))
        img = img.byte()
        trf = trf @ sca # undo sca first

    return img, trf


def rotate_img( img, angle, crop=False ):
    if angle in (0, 90, 180, 270):
        return rotate_img_90(img,angle)

    img, trf = img
    assert trf.shape == (3,3)

    def centered_rotation(rotation, shape, **device):
        # rotation matrix
        # pt_in_original_image = rot * pt_in_rotated_image
        angle = rotation * np.pi / 180
        c, s = np.cos(angle), np.sin(angle)
        rot = torch.tensor([(c, -s, 0), (s, c, 0), (0, 0, 1)], dtype=torch.float32, **device)

        # determine center of rotation before
        H, W = shape
        c_before = torch.tensor((W,H), **device) / 2
        if crop:
            c_after = c_before
            rot_size = (W,H)
        else:
            # enlarge image to fit everything
            corners = torch.tensor([(0, W, W, 0), (0, 0, H, H)], dtype=torch.float32, **device)
            corners = affmul(rot, corners.T).T
            rot_size = (corners.max(dim=1).values - corners.min(dim=1).values + 0.5).int()
            rot_size = (rot_size // 4) * 4 # legacy
            c_after = rot_size / 2

        rot[:2,2] = c_before - affmul(rot, c_after) # fix translation
        return rot, tuple(rot_size)[::-1]

    C, H, W = img.shape
    rot, (OH, OW) = centered_rotation(angle, (H,W), device=img.device)

    # pt_in_original_image = rot * pt_in_rotated_image
    # but pytorch works in [-1,1] coordinates... annoying
    # pt_in_original_1_1 = orig_px_to_1_1 * rot * rotated_1_1_to_px * pt_in_rotated_1_1
    _1_1_to_px = lambda W,H: torch.tensor(((W/2, 0, W/2), (0, H/2, H/2), (0, 0, 1)), device=img.device)
    theta = torch.inverse(_1_1_to_px(W-1,H-1)) @ rot @ _1_1_to_px(OW-1,OH-1)

    grid = F.affine_grid(theta[None,:2], (1, C, OH, OW), align_corners=True)
    res = F.grid_sample(img[None].float(), grid, align_corners=True).to(dtype=img.dtype)[0]
    return res, trf @ rot



def rotate_img_90( img, angle ):
    """ Rotate an image by a multiple of 90 degrees using simple transpose and flip ops.
    img = tuple( image, existing_trf )
        existing_trf: current --> old
    """
    angle = angle % 360
    assert angle in (0, 90, 180, 270), 'cannot handle rotation other than multiple of 90 degrees'
    img, trf = img
    assert trf.shape == (3,3)
    
    if isinstance(img, np.ndarray):
        assert img.ndim == 3 and 1 <= img.shape[2] <= 3
        new, x, y = np.float32, 1, 0
        flip = lambda i,d: np.flip(i,axis=d)
    elif isinstance(img, torch.Tensor):
        assert img.ndim == 3 and 1 <= img.shape[0] <= 3
        new, x, y = trf.new, -1, -2
        flip = lambda i,d: i.flip(dims=[d])
    H, W = img.shape[y], img.shape[x]

    if angle == 90:
        # point 0,0 --> (0, H-1); W-1,0 --> 0,0
        img = flip(img.swapaxes(x,y),y)
        trf = trf @ new([[0,-1,W-1],[1,0,0],[0,0,1]]) # inverse transform: new --> current
    if angle == 180:
        # point 0,0 --> (W-1, H-1)
        img = flip(flip(img,x),y)
        trf = trf @ new([[-1,0,W-1],[0,-1,H-1],[0,0,1]]) # inverse transform: new --> current
    if angle == 270:
        # point 0,0 --> (H-1, 0); 0,H-1 --> 0,0
        img = flip(img.swapaxes(x,y),x)
        trf = trf @ new([[0,1,0],[-1,0,H-1],[0,0,1]]) # inverse transform: new --> current
    return img, trf


def encode_scale_rot(scale, rot):
    s = np.int32(np.rint(np.log(scale) / (0.5*np.log(2))))
    r = np.int32(np.rint(((-rot) % 360) / 45)) % 8
    return 8*s + (r%8)

def decode_scale_rot( code ):
    s = code // 8
    r = (code % 8)
    return 2 ** (s/2), -((45 * r + 180) % 360 - 180)


def normalized_corr(patches, img, padding='ncc', extra_patch=False, ret_norms=False):
    assert patches.ndim == 4, 'patches shape must be (H*W, C, K, K)'
    P, C, K, K = patches.shape
    assert img.ndim == 3 and img.shape[0] == C, 'img shape must be (C, W, H)'
    eps = torch.finfo(patches.dtype).tiny

    # normalize on patches side
    norms = patches.view(P,-1).norm(dim=-1)
    patches = patches / norms[:,None,None,None].clamp(min=eps)

    # convolve normalized patches on unnormalized image    
    ninth = 0
    if padding == 'ninth': 
        ninth = img[:,-1].mean() # ninth dimension    
    img = F.pad(img[None], (K//2,K//2)*2, mode='constant', value=ninth)[0]

    corr = F.conv2d(img[None], patches, padding=0, bias=None)[0]

    # normalize on img's side
    ones = patches.new_ones((1, C, K, K))
    local_norm = torch.sqrt(F.conv2d(img[None]**2, ones))[0]
    corr /= local_norm

    # normalize on patches' side (image borders)
    if padding == 'ncc':
        local_norm = torch.sqrt(F.conv2d(ones, patches**2, padding=2))[0]
        local_norm.clamp_(min=eps)
        for j in range(-2, 3):
          for i in range(-2,3):
            if i == j == 2: continue # normal case is already normalized
            if i == 2: i = slice(2,-2)
            if j == 2: j = slice(2,-2)
            corr[:,j,i] /= local_norm[:,j,i]

    return (corr, norms) if ret_norms else corr


def true_corr_shape( corr_shape, level ):
    H1, W1, H2, W2 = corr_shape[-4:]
    if level > 0: # recover true size
        H1, W1 = H1-1, W1-1
    return corr_shape[:-4] + (H1, W1, H2, W2)

def children(level, H1, W1, H2, W2):
    """ level: parent level (> 1) """
    gap = 2**(level-2)
    # @ level 1: gap=0.5   (parent at x=1 has children at x=[0.5, 1.5])
    # @ level 2: gap=1     (parent at x=1 has children at x=[0, 2])
    # @ level 3: gap=2     (parent at x=2 has children at x=[0, 4])
    #   etc.

    def ravel_child(x, y):
        # x,y is he center of the child patch
        inside = (0 <= x <= W1) and (0 <= y <= H1)
        if gap < 1:
            assert x % 1 == y % 1 == 0.5, bb()
            return int((x-0.5) + (y-0.5) * W1) if inside else -1
        else:
            assert x % 1 == y % 1 == 0, bb()
            return int(x + y * (W1+1)) if inside else -1
    
    # 4 children for each parent patch (top-left, top-right, bot-left, bot-right, -1 = None)
    parents = []
    for h in range(H1+1):
      for w in range(W1+1):
        # enumerate the 4 children for this patch
        children = [ravel_child(w + gap*tx, h + gap*ty) for ty in (-1,1) for tx in (-1,1)]
        parents.append(children)

    return torch.tensor(parents, dtype=torch.int64)


def sparse_conv(level, corr, weights=None, reverse=False, norm=0.9):
    H1, W1, H2, W2 = true_corr_shape(corr.shape, level-1 + reverse)
    parents = children(level, H1, W1, H2, W2).to(corr.device)
    n_parents = len(parents)

    # perform the sparse convolution 'manually'
    # since sparse convolutions are not implemented in pytorch currently
    corr = corr.view(-1, *corr.shape[-2:])
    if not reverse:
        res = corr.new_zeros((n_parents+1,)+corr.shape[-2:]) # last one = garbage channel
        nrm = corr.new_full((n_parents+1,3,3), 1e-8)
        ones = nrm.new_ones((len(corr),1,1))
        ex = 1
        if weights is not None: 
            weights = weights.view(len(corr),1,1)
            corr *= weights # apply weights to correlation maps without increasing memory footprint
            ones *= weights
    else:
        assert corr._base is not None and corr._base.shape[0] == n_parents+1
        corr._base[-1] = 0 # reset garbage layer
        ex = 1 if level > 1 else 0
        n_children = (H1+ex) * (W1+ex)
        res = corr.new_zeros((n_children,)+corr.shape[-2:]) 

    sl = lambda v: slice(0,-1 or None) if v < 0 else slice(1,None)
    c = 0
    for y in (-1, 1):
        for x in (-1, 1):
            src_layers = parents[:,c]; c+= 1
            # we want to do: res += corr[src_layers]  (for all children != -1)
            # but we only have 'res.index_add_()' <==> res[tgt_layers] += corr
            tgt_layers = inverse_mapping(src_layers, max_elem=len(corr), default=n_parents)[:-1]

            if not reverse:
                # All of corr's channels MUST be utilized. for level>1, this doesn't hold,
                # so we'll send them to a garbage channel ==> res[n_parents]
                sel = good_slice( tgt_layers < n_parents )

                res[:,sl(-y),sl(-x)].index_add_(0, tgt_layers[sel], corr[sel,sl(y),sl(x)])
                nrm[:,sl(-y),sl(-x)].index_add_(0, tgt_layers[sel], ones[sel].expand(-1,2,2))
            else:
                ''' parent=199=11*17+12 @ (x=48, y=44) at level=1
                    |-- child=171 @ (x=46,y=42) at level0
                    |-- child=172 @ (x=50,y=42) at level0
                    |-- child=187 @ (x=46,y=46) at level0
                    |-- child=188 @ (x=50,y=46) at level0
                '''
                out = res[:,sl(y),sl(x)]
                sel = tgt_layers[:n_children]
                torch.maximum(out, corr._base[sel,sl(-y),sl(-x)], out=out)

    if not reverse: 
        if weights is not None: corr /= weights.clamp(min=1e-12) # cancel weights
        weights = norm_borders(res, nrm, norm=norm)[:-1]
        res = res[:-1] # remove garbage channel
    res = res.view(H1+ex, W1+ex, *res.shape[-2:])
    return res if reverse else (res, weights)
    
def norm_borders( res, nrm, norm=0.9 ):
    """ apply some border normalization, modulated by `norm`
        - if norm=0: no normalization at all
        - if norm=1: full normalization
    Formula: nrm = k * (nrm/k)**p = k**(1-p) * nrm**p, 
        with k=nrm[:,1,1] and p=norm
    """
    new_weights = nrm[...,1,1].clone()
    nrm = (nrm[...,1:2,1:2] ** (1-norm)) * (nrm ** norm)
    # assert not torch.isnan(nrm).any()
    
    # normalize results on the borders 
    res[...,0   ,0   ] /= nrm[...,0  ,0  ]
    res[...,0   ,1:-1] /= nrm[...,0  ,1:2]
    res[...,0   ,  -1] /= nrm[...,0  ,2  ]
    res[...,1:-1,0   ] /= nrm[...,1:2,0  ]
    res[...,1:-1,1:-1] /= nrm[...,1:2,1:2]
    res[...,1:-1,  -1] /= nrm[...,1:2,2  ]
    res[...,  -1,0   ] /= nrm[...,2  ,0  ]
    res[...,  -1,1:-1] /= nrm[...,2  ,1:2]
    res[...,  -1,  -1] /= nrm[...,2  ,2  ]
    return new_weights


def inverse_mapping( map, max_elem=None, default=None):
    """ given a mapping {i:j} we output {j:i}
        (the mapping is a torch array)
    """
    assert isinstance(map, torch.Tensor) and map.ndim == 1
    if max_elem is None: max_elem = map.max()
    if default is None:
        index = torch.empty(max_elem+1, dtype=torch.int64, device=map.device) # same size as corr, last elem == garbage
    else:
        index = torch.full((max_elem+1,), default, dtype=torch.int64, device=map.device) # same size as corr, last elem == garbage
    index[map] = torch.arange(len(map), device=map.device)
    return index


def good_slice( nonzero ):
    good = nonzero.nonzero().ravel()
    return slice(good.min().item(), good.max().item()+1)


def max_unpool(upper, lower, exclude_border=True):
    # re-compute max-pool indices
    if exclude_border:
        # apparently, we cannot unpool on the bottom and right borders in legacy code (local_argmax with ex=1)
        _, pos = F.max_pool2d(lower[:,:,:-1,:-1], 3, padding=1, stride=2, return_indices=True, ceil_mode=True)
        W1 = lower.shape[-1]
        pos = (pos//(W1-1))*W1 + (pos%(W1-1)) # fix the shortening
    else:
        _, pos = F.max_pool2d(lower, 3, padding=1, stride=2, return_indices=True)

    # because there are potential collisions between overlapping 3x3 cells,
    # that pytorch does not handle, we unpool in 4 successive non-overlapping steps.
    for i in range(2):
      for j in range(2):
        # stride=0 instead of 1 because pytorch does some size checking, this is a hack
        tmp = F.max_unpool2d(upper[:,:,i::2,j::2], pos[:,:,i::2,j::2], kernel_size=3, padding=0, stride=4, output_size=lower.shape[-2:])
        if i == j == 0:
            res = tmp
        else:
            torch.maximum(res, tmp, out=res)

    # add scores to existing lower correlation map
    lower += res
    return lower


def mgrid( shape, **kw ):
    """ Returns in (x, y) order (contrary to numpy which is (y,x)  """
    if isinstance(shape, torch.Tensor): shape = shape.shape
    res = torch.meshgrid(*[torch.arange(n, dtype=torch.float32, **kw) for n in shape], indexing='ij')
    return torch.stack(res[::-1], dim=-1).view(-1,2)


def check_corres( corres, step, rot=None ):
    H, W, two = corres.shape
    assert two == 2
    if isinstance(corres, np.ndarray): 
        corres = torch.from_numpy(corres)
    if rot is not None:
        corres = affmul(rot, corres)
    gt = mgrid(corres.shape[:2]).view(H,W,2)
    assert ((gt - corres // step).abs() <= 2).float().mean() > 0.99, bb()


def best_correspondences(corr):
    """ All positions are returned as x1, y1, x2, y2
    """
    if isinstance(corr, tuple): return corr # for legacy
    H1, W1, H2, W2 = corr.shape
    fix1 = lambda arr: 4*arr+2 # center of cells in img1
    div = lambda a,b: torch.div(a, b, rounding_mode='trunc') # because of warning in pytorch 1.9+

    # best scores in img1
    score1, pos1 = corr.view(H1, W1, H2*W2).max(dim=-1)
    pos1 = torch.cat((fix1(mgrid(score1, device=pos1.device)), pos1.view(-1,1)%W2, div(pos1.view(-1,1),W2)), dim=-1)

    # best scores in img2
    score2, pos2 = max_pool3d( corr, kernel_size=4, stride=4 )
    pos2, score2 = pos2.view(-1,1), score2.squeeze()
    pos2 = torch.cat((fix1(div(pos2,W2*H2)%W1), fix1(div(pos2,(W1*H2*W2))), pos2%W2, div(pos2,W2)%H2), dim=-1).float()

    return (pos1, score1), (pos2, score2)


def intersection( set1_, set2_ ):
    """ Returns the indices of values in set1 that are duplicated in set2
    """
    set1, map1 = set1_.squeeze().unique(return_inverse=True) # map1: i1 -> j1
    set2 = set2_.squeeze().unique()
    combined = torch.cat((set1, set2))

    uniques, inverse, counts = combined.unique(return_counts=True, return_inverse=True)
    # j -> u, i -> j, j -> n
    # we are interested only in (j -> i) for n > 1: 
    # assert counts.max() <= 2, 'there were non-unique values in either set1 or set2'+bb()
    # intersected_values = uniques[counts > 1]
    inverse1 = inverse_mapping(inverse[:len(set1)], max_elem=len(uniques)-1)
    intersected_indices1 = inverse1[counts>1]
    return inverse_mapping(map1, max_elem=len(set1)-1)[intersected_indices1]


def reciprocal(self, corres1, corres2 ):
    pos1, score1 = corres1 
    pos2, score2 = corres2
    (H1, W1), (H2, W2) = score1.shape, map(lambda i: 4*i+1, score2.shape)

    to_int = pos1.new_tensor((W1*H2*W2, H2*W2, W2, 1), dtype=torch.float32)
    inter1 = intersection(pos1@to_int, pos2@to_int)
    res = torch.cat((pos1[inter1], score1.view(-1,1)[inter1], 0*score1.view(-1,1)[inter1]), dim=-1)
    return res


def max_pool3d( corr, kernel_size=4, stride=4 ):
    H1, W1, H2, W2 = corr.shape
    ks, st = kernel_size, stride
    if corr.numel() >= 2**31 and corr.device != torch.device('cpu'):
        # re-implementation due to a bug in pytorch
        import core.cuda_deepm as kernels
        return kernels.max_pool3d( corr.view(1, H1*W1, H2, W2), kernel_size, stride)
    else:
        return F.max_pool3d( corr.view(1, 1, H1*W1, H2, W2), kernel_size=(H1*W1,ks,ks), stride=(1,st,st), return_indices=True)


def forward_cuda(self, level, lower, weights=None, pooled=False):
    import core.cuda_deepm as kernels # must be imported after torch_set_gpu()
    assert lower.numel() < 2**31, 'please use cuda-lowmem, pytorch cannot handle big tensors'
    pooled = lower if pooled else F.max_pool2d(lower, 3, padding=1, stride=2)
    return kernels.forward_agg(level, self.border_inv, pooled, weights)

def forward_cuda_lowmem(self, level, lower, weights=None):
    import core.cuda_deepm as kernels # must be imported after torch_set_gpu()
    return kernels.forward_pool_agg(level, self.border_inv, lower, weights)

def backward_cuda(self, level, pyramid):
    import core.cuda_deepm as kernels # must be imported after torch_set_gpu()
    kernels.backward_agg_unpool(level, pyramid[level], pyramid[level-1], True)
    # assert not torch.isnan(pyramid[level-1]).any(), bb()
    return pyramid[level-1]

def merge_corres(self, corres, rots, all_corres, code):
    " rot : reference --> rotated "
    all_step = self.matcher.pixel_desc.get_atomic_patch_size() // 2 # step size in all_corres
    dev = all_corres[0][1].device

    # stack correspondences
    corres = [torch.cat((p.view(*s.shape,4),s[:,:,None],torch.full_like(s[:,:,None],code)),dim=2) for (p,s) in corres]

    import core.cuda_deepm as kernels # must be imported after torch_set_gpu()
    kernels.merge_corres_one_side( corres[0].to(dev), 0, rots[0].to(dev), all_corres[0][1], all_step )
    kernels.merge_corres_one_side( corres[1].to(dev), 2, rots[1].to(dev), all_corres[1][1], all_step )

