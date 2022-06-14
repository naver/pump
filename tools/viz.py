# Copyright 2022-present NAVER Corp.
# CC BY-NC-SA 4.0
# Available only for non-commercial use

import sys
from pdb import set_trace as bb
from PIL import Image
import numpy as np

import matplotlib.pyplot as pl; pl.ion()
import torch
import torch.nn.functional as F

from core import functional as myF
from .common import cpu, nparray, image, image_with_trf


def dbgfig(*args, **kwargs):
    assert len(args) >= 2
    dbg = args[-1]
    if isinstance(dbg, str): 
        dbg = dbg.split()
    for name in args[:-1]:
        if {name,'all'} & set(dbg):
            return pl.figure(name, **kwargs)
    return False


def noticks(ax=None):
    if ax is None: ax = pl.gca()
    ax.set_xticks(())
    ax.set_yticks(())
    return ax


def plot_grid( corres, ax1, ax2=None, marker='+' ):
    """ corres = Nx2 or Nx4 list of correspondences
    """
    if marker is True: marker = '+'

    corres = nparray(corres)
    # make beautiful colors
    center = corres[:,[1,0]].mean(axis=0)
    colors = np.arctan2(*(corres[:,[1,0]] - center).T)
    colors = np.int32(64*colors/np.pi) % 128

    all_colors = np.unique(colors)
    palette = {m:pl.cm.hsv(i/float(len(all_colors))) for i,m in enumerate(all_colors)}

    for m in all_colors:
        x, y = corres[colors==m,0:2].T
        ax1.plot(x, y, marker, ms=10, mew=2, color=palette[m], scalex=0, scaley=0)

    if not ax2: return
    for m in all_colors:
        x, y = corres[colors==m,2:4].T
        ax2.plot(x, y, marker, ms=10, mew=2, color=palette[m], scalex=0, scaley=0)


def show_correspondences( img0, img1, corres, F=None, fig='last', show_grid=True, bb=None, clf=False):
    img0, trf0 = img0 if isinstance(img0, tuple) else (img0, torch.eye(3))
    img1, trf1 = img1 if isinstance(img1, tuple) else (img1, torch.eye(3))
    if not bb: pl.ioff()
    fig, axes = pl.subplots(2, 2, num=fig_num(fig, 'viz_corres'))
    for i, ax in enumerate(axes.ravel()):
        if clf: ax.cla()
        noticks(ax).numaxis = i % 2
        ax.imshow( [image(img0),image(img1)][i%2] )

    if corres.shape == (3,3): # corres is an homography matrix
        from pytools.hfuncs import applyh
        H, W = axes[0,0].images[0].get_size()
        pos1 = np.mgrid[:H,:W].reshape(2,-1)[::-1].T
        pos2 = applyh(corres, pos1)
        corres = np.concatenate((pos1,pos2), axis=-1)

    inv = np.linalg.inv
    corres = myF.affmul((inv(nparray(trf0)),inv(nparray(trf1))), nparray(corres)) # image are already downscaled
    print(f">> Displaying {len(corres)} correspondences (move you mouse over the images)")

    (ax1, ax2), (ax3, ax4) = axes
    if corres.shape[-1] > 4:
        corres = corres[corres[:,4]>0,:] # select non-null correspondences
    if show_grid: plot_grid(corres, ax3, ax4, marker=show_grid)

    def mouse_move(event):
        if event.inaxes==None: return
        numaxis = event.inaxes.numaxis
        if numaxis<0: return
        x,y = event.xdata, event.ydata
        ax1.lines.clear()
        ax2.lines.clear()
        sl = slice(2*numaxis, 2*(numaxis+1))
        n = np.sum((corres[:,sl] - [x,y])**2,axis=1).argmin() # find nearest point
        print("\rdisplaying #%d (%d,%d) --> (%d,%d), score=%g, code=%g" % (n,
            corres[n,0],corres[n,1],corres[n,2],corres[n,3],
            corres[n,4] if corres.shape[-1] > 4 else np.nan,
            corres[n,5] if corres.shape[-1] > 5 else np.nan), end=' '*7);sys.stdout.flush()
        x,y = corres[n,0:2]
        ax1.plot(x, y, '+', ms=10, mew=2, color='blue', scalex=False, scaley=False)
        x,y = corres[n,2:4]
        ax2.plot(x, y, '+', ms=10, mew=2, color='red', scalex=False, scaley=False)
        if F is not None:
            ax = None
            if numaxis == 0:
                line = corres[n,0:2] @ F[:2] + F[2]
                ax = ax2
            if numaxis == 1:
                line = corres[n,2:4] @ F.T[:2] + F.T[2]
                ax = ax1
            if ax:
                x = np.linspace(-10000,10000,2)
                y = (line[2]+line[0]*x) / -line[1]
                ax.plot(x, y, '-', scalex=0, scaley=0)

        # we redraw only the concerned axes
        renderer = fig.canvas.get_renderer()
        ax1.draw(renderer)
        ax2.draw(renderer)
        fig.canvas.blit(ax1.bbox)
        fig.canvas.blit(ax2.bbox)

    cid_move = fig.canvas.mpl_connect('motion_notify_event',mouse_move)
    pl.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0.02, hspace=0.02)
    bb() if bb else pl.show()
    fig.canvas.mpl_disconnect(cid_move)
    

def closest( grid, event ):
    query = (event.xdata, event.ydata)
    n = np.linalg.norm(grid.reshape(-1,2) - query, axis=1).argmin()
    return np.unravel_index(n, grid.shape[:2])


def local_maxima( arr2d, top=5 ):
    maxpooled = F.max_pool2d( arr2d[None, None], 3, padding=1, stride=1)[0,0]
    local_maxima = (arr2d == maxpooled).nonzero()
    order = arr2d[local_maxima.split(1,dim=1)].ravel().argsort()
    return local_maxima[order[-5:]].T


def fig_num( fig, default, clf=False ):
    if fig == 'last': num = pl.gcf().number
    elif fig: num = fig.number
    else: num = default
    if clf: pl.figure(num).clf()
    return num


def viz_correlation_maps( img1, img2, corr, level=0, fig=None, grid1=None, grid2=None, show_grid=False, bb=bb, **kw ):
    fig, ((ax1, ax2), (ax4, ax3)) = pl.subplots(2, 2, num=fig_num(fig, 'viz_correlation_maps', clf=True))
    img1 = image(img1)
    img2 = image(img2)
    noticks(ax1).imshow( img1 )
    noticks(ax2).imshow( img2 )
    ax4.hist(corr.ravel()[7:7777777:7].cpu().numpy(), bins=50)

    if isinstance(corr, tuple):
        H1, W1 = corr.grid.shape[:2]
        corr = torch.from_numpy(corr.res_map).view(H1,W1,*corr.res_map.shape[-2:])

    if grid1 is None:
        s1 = int(0.5 + np.sqrt(img1.size / (3 * corr[...,0,0].numel()))) # scale factor between img1 and corr
        grid1 = nparray(torch.ones_like(corr[:,:,0,0]).nonzero()*s1)[:,1::-1]
        if level == 0: grid1 += s1//2
    if show_grid: plot_grid(grid1, ax1)
    grid1 = nparray(grid1).reshape(*corr[:,:,0,0].shape,2)

    if grid2 is None:
        s2 = int(0.5 + np.sqrt(img2.size / (3 * corr[0,0,...].numel()))) # scale factor between img2 and corr
        grid2 = nparray(torch.ones_like(corr[0,0]).nonzero()*s2)[:,::-1]
    grid2 = nparray(grid2).reshape(*corr.shape[2:],2)

    def mouse_move(ev):
        if ev.inaxes is ax1:
            ax3.images.clear()
            n = closest(grid1, ev)
            ax3.imshow(corr[n].cpu().float(), vmin=0, **kw)

            # find local maxima
            lm = nparray(local_maxima(corr[n]))
            for ax in (ax3, ax2):
                if ax is ax2 and not show_grid: 
                    ax1.lines.clear()
                    ax1.plot(*grid1[n], 'xr', ms=10, scalex=0, scaley=0)
                ax.lines.clear()
                x, y = grid2[y,x].T if ax is ax2 else lm[::-1]
                if ax is not ax3:
                    ax.plot(x, y, 'xr', ms=10, scalex=0, scaley=0, label='local maxima')
            print(f"\rCorr channel {n}. Min={corr[n].min():g}, Avg={corr[n].mean():g}, Max={corr[n].max():g}   ", end='')

    mouse_move(FakeEvent(0,0,inaxes=ax1))
    cid_move = fig.canvas.mpl_connect('motion_notify_event', mouse_move)
    pl.subplots_adjust(0,0,1,1,0,0)
    pl.sca(ax4)
    if bb: bb(); fig.canvas.mpl_disconnect(cid_move)

def viz_correspondences( img1, img2, corres1, corres2, fig=None ):
    img1, img2 = map(image, (img1, img2))
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = pl.subplots(3,2, num=fig_num(fig, 'viz_correspondences'))
    for ax in fig.axes: noticks(ax)
    ax1.imshow( img1 )
    ax2.imshow( img2 )
    ax3.imshow( img1 )
    ax4.imshow( img2 )
    corres1, corres2 = map(cpu, (corres1, corres2))
    plot_grid( corres1[0], ax1, ax2 )
    plot_grid( corres2[0], ax3, ax4 )

    corres1, corres2 = corres1[1].float(), corres2[1].float()
    ceiling = np.ceil(max(corres1.max(), corres2.max()).item())
    ax5.imshow( corres1, vmin=0, vmax=ceiling )
    ax6.imshow( corres2, vmin=0, vmax=ceiling )
    bb()


class FakeEvent:
    def __init__(self, xdata, ydata, **kw):
        self.xdata = xdata
        self.ydata = ydata
        for name, val in kw.items():
            setattr(self, name, val)


def show_random_pairs( db, pair_idxs=None, **kw ):
    print('Showing random pairs from', db)

    if pair_idxs is None:
        pair_idxs = np.random.permutation(len(db))

    for pair_idx in pair_idxs:
        print(f'{pair_idx=}')
        try:
            img1_path, img2_path = map(db.imgs.get_image_path, db.pairs[pair_idx])
            print(f'{img1_path=}\n{img2_path=}')
            if hasattr(db, 'get_corres_path'):
                print(f'corres_path = {db.get_corres_path(pair_idx)}')
        except: pass
        (img1, img2), gt = db[pair_idx]

        if 'corres' in gt:
            corres = gt['corres']
        else: 
            # make corres from homography
            from datasets.utils import corres_from_homography
            corres = corres_from_homography(gt['homography'], *img1.size)

        show_correspondences(img1, img2, corres, **kw)


if __name__=='__main__':
    import argparse
    import test_singlescale as pump

    parser = argparse.ArgumentParser('Correspondence visualization')
    parser.add_argument('--img1', required=True, help='path to first image')
    parser.add_argument('--img2', required=True, help='path to second image')
    parser.add_argument('--corres', required=True, help='path to correspondences')
    args = parser.parse_args()

    corres = np.load(args.corres)['corres']

    args.resize = 0 # don't resize images
    imgs = tuple(map(image, pump.Main.load_images(args)))

    show_correspondences(*imgs, corres)
