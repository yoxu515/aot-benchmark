import numpy as np
from PIL import Image
import cv2

cv2.setNumThreads(0)

import numpy as np

class TPS:       
    @staticmethod
    def fit(c, lambd=0., reduced=False):        
        n = c.shape[0]

        U = TPS.u(TPS.d(c, c))
        K = U + np.eye(n, dtype=np.float32)*lambd

        P = np.ones((n, 3), dtype=np.float32)
        P[:, 1:] = c[:, :2]

        v = np.zeros(n+3, dtype=np.float32)
        v[:n] = c[:, -1]

        A = np.zeros((n+3, n+3), dtype=np.float32)
        A[:n, :n] = K
        A[:n, -3:] = P
        A[-3:, :n] = P.T

        theta = np.linalg.solve(A, v) # p has structure w,a
        return theta[1:] if reduced else theta
        
    @staticmethod
    def d(a, b):
        return np.sqrt(np.square(a[:, None, :2] - b[None, :, :2]).sum(-1))

    @staticmethod
    def u(r):
        return r**2 * np.log(r + 1e-6)

    @staticmethod
    def z(x, c, theta):
        x = np.atleast_2d(x)
        U = TPS.u(TPS.d(x, c))
        w, a = theta[:-3], theta[-3:]
        reduced = theta.shape[0] == c.shape[0] + 2
        if reduced:
            w = np.concatenate((-np.sum(w, keepdims=True), w))
        b = np.dot(U, w)
        return a[0] + a[1]*x[:, 0] + a[2]*x[:, 1] + b

def uniform_grid(shape):
    '''Uniform grid coordinates.
    
    Params
    ------
    shape : tuple
        HxW defining the number of height and width dimension of the grid
    Returns
    -------
    points: HxWx2 tensor
        Grid coordinates over [0,1] normalized image range.
    '''

    H,W = shape[:2]    
    c = np.empty((H, W, 2))
    c[..., 0] = np.linspace(0, 1, W, dtype=np.float32)
    c[..., 1] = np.expand_dims(np.linspace(0, 1, H, dtype=np.float32), -1)

    return c
    
def tps_theta_from_points(c_src, c_dst, reduced=False):
    delta = c_src - c_dst
    
    cx = np.column_stack((c_dst, delta[:, 0]))
    cy = np.column_stack((c_dst, delta[:, 1]))
        
    theta_dx = TPS.fit(cx, reduced=reduced)
    theta_dy = TPS.fit(cy, reduced=reduced)

    return np.stack((theta_dx, theta_dy), -1)


def tps_grid(theta, c_dst, dshape):    
    ugrid = uniform_grid(dshape)

    reduced = c_dst.shape[0] + 2 == theta.shape[0]

    dx = TPS.z(ugrid.reshape((-1, 2)), c_dst, theta[:, 0]).reshape(dshape[:2])
    dy = TPS.z(ugrid.reshape((-1, 2)), c_dst, theta[:, 1]).reshape(dshape[:2])
    dgrid = np.stack((dx, dy), -1)

    grid = dgrid + ugrid
    
    return grid # H'xW'x2 grid[i,j] in range [0..1]

def tps_grid_to_remap(grid, sshape):
    '''Convert a dense grid to OpenCV's remap compatible maps.
    
    Params
    ------
    grid : HxWx2 array
        Normalized flow field coordinates as computed by compute_densegrid.
    sshape : tuple
        Height and width of source image in pixels.
    Returns
    -------
    mapx : HxW array
    mapy : HxW array
    '''

    mx = (grid[:, :, 0] * sshape[1]).astype(np.float32)
    my = (grid[:, :, 1] * sshape[0]).astype(np.float32)

    return mx, my

def pick_random_points(h, w, n_samples):
    y_idx = np.random.choice(np.arange(h), size=n_samples, replace=False)
    x_idx = np.random.choice(np.arange(w), size=n_samples, replace=False)
    return y_idx/h, x_idx/w


def warp_dual_cv(img, mask, c_src, c_dst):
    dshape = img.shape
    theta = tps_theta_from_points(c_src, c_dst, reduced=True)
    grid = tps_grid(theta, c_dst, dshape)
    mapx, mapy = tps_grid_to_remap(grid, img.shape)
    return cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR), cv2.remap(mask, mapx, mapy, cv2.INTER_NEAREST)


def random_tps_warp(img, mask, scale, n_ctrl_pts=12,return_array=False):
    """
    Apply a random TPS warp of the input image and mask
    Uses randomness from numpy
    """
    img = np.asarray(img)
    mask = np.asarray(mask)

    h, w = mask.shape
    points = pick_random_points(h, w, n_ctrl_pts)
    c_src = np.stack(points, 1)
    c_dst = c_src + np.random.normal(scale=scale, size=c_src.shape)
    warp_im, warp_gt = warp_dual_cv(img, mask, c_src, c_dst)
    
    if return_array:
        return warp_im,warp_gt
    else:
        return Image.fromarray(warp_im), Image.fromarray(warp_gt)

