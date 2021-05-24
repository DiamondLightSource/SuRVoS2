import numpy as np
from itertools import combinations_with_replacement
from loguru import logger
from .blur import gaussian_blur_kornia
import torch
import kornia
import numbers
from .base import rescale_denan


def compute_hessian(data, sigma):
    """Compute hessian matrix using pytorch Gaussian smoothing

    Parameters
    ----------
    data : np.array
        (D,H,W) array of floats
    sigma : Vector of 3 floats
        Smoothing kernel size

    Returns
    -------
    np.ndarray(6,D,H,W)
        Hessian matrix
    """
    logger.info("+ Computing Hessian Matrix")
    gaussian_filtered = gaussian_blur_kornia(data, sigma=sigma)
    gradients = [np.gradient(gaussian_filtered, axis=i) for i in range(3)]
    axes = range(data.ndim)
    H_elems = [
        np.gradient(gradients[2 - ax0], axis=2 - ax1)
        for ax0, ax1 in combinations_with_replacement(axes, 2)
    ]
    if not isinstance(sigma, numbers.Number):
        sigma = max(sigma)

    if sigma > 1:
        H_elems = [elem * sigma ** 2 for elem in H_elems]
    return H_elems


def compute_hessian_determinant(data, sigma, bright=False):
    """Hessian determinant

    Parameters
    ----------
    data : (D,H,W) np.array
        Input image array
    sigma: Vector of 3 floats
        Kernel size for smoothing used for hessian
    bright : bool, optional
        Negative of determinant, by default False

    Returns
    -------
    np.ndarray(D,H,W)
        Determinant of hessian calculated at each voxel value
    """
    logger.info("+ Computing Hessian Determinant")
    Hxx, Hxy, Hxz, Hyy, Hyz, Hzz = compute_hessian(data, sigma)

    det = (
        Hxx * (Hyy * Hzz - Hyz * Hyz)
        - Hxy * (Hxy * Hzz - Hyz * Hxz)
        + Hxz * (Hxy * Hyz - Hyy * Hxz)
    )

    if bright:
        return -det

    return det


def hessian_eigvals_cython(data, sigma, correct=False):
    """Hessian eigenvalues cython

    Parameters
    ----------
    data : np.ndarray(D,H,W)
        Input image
    sigma : Vector of 3 float
        Kernel size
    correct : bool, optional
        Adjust sigma to square of input sigma, by default False

    Returns
    -------
    np.array
        (D,H,W,3) Hessian eigenvalues
    """
    H = compute_hessian(data, sigma)

    if correct:
        if not isinstance(sigma, numbers.Number):
            s = max(sigma) ** 2
        else:
            s = sigma ** 2
        Hxx, Hxy, Hxz, Hyy, Hyz, Hzz = [h * s for h in H]
    else:
        Hxx, Hxy, Hxz, Hyy, Hyz, Hzz = H

    logger.info(f"+ Computing Hessian Eigenvalues with sigma {sigma}")
    from survos2.improc.features._symeigval import symmetric_eig

    response = symmetric_eig(
        Hxx.copy(order="C"),
        Hxy.copy(order="C"),
        Hxz.copy(order="C"),
        Hyy.copy(order="C"),
        Hyz.copy(order="C"),
        Hzz.copy(order="C"),
    )

    logger.debug(f"Calculated hessian_eigvals reponse of shape {response.shape}")
    
    return rescale_denan(response)


def hessian_eigvals(data, sigma, correct=False):
    """Hessian eigenvalues using Pytorch

    Parameters
    ----------
    data : np.ndarray(D,H,W)
        Input image
    sigma : Vector of 3 float
        Kernel size
    correct : bool, optional
        Adjust sigma to square of input sigma, by default False

    Returns
    -------
    np.array
        (D,H,W,3) Hessian eigenvalues
    """
    H = compute_hessian(data, sigma)

    if correct:
        if not isinstance(sigma, numbers.Number):
            s = max(sigma) ** 2
        else:
            s = sigma ** 2
        Hxx, Hxy, Hxz, Hyy, Hyz, Hzz = [h * s for h in H]
    else:
        Hxx, Hxy, Hxz, Hyy, Hyz, Hzz = H

    logger.info(f"+ Computing Hessian Eigenvalues with sigma {sigma}")

    Hs = torch.FloatTensor([[Hzz, Hyz, Hxz], [Hyz, Hyy, Hxy], [Hxz, Hxy, Hxx]])
    Hs_batch = Hs.reshape((3, 3, Hs.shape[-3] * Hs.shape[-2] * Hs.shape[-1]))
    e, v = torch.symeig(Hs_batch.permute((2, 0, 1)))
    e = e.reshape((Hs.shape[-3], Hs.shape[-2], Hs.shape[-1], 3))

    img: np.ndarray = kornia.tensor_to_image(e)
    img = np.transpose(img, (0, 3, 1, 2))

    logger.debug(f"Calculated hessian_eigvals reponse of shape {img.shape}")

    
    return np.nan_to_num(img)


def hessian_eigvals_image(data, sigma, correct=False):
    """Hessian eigenvalues using Pytorch

    Parameters
    ----------
    data : np.ndarray(D,H,W)
        Input image
    sigma : Vector of 3 float
        Kernel size
    correct : bool, optional
        Adjust sigma to square of input sigma, by default False

    Returns
    -------
    np.array
        (D,H,W,3) Hessian eigenvalues
    """
    H = compute_hessian(data, sigma)

    if correct:
        if not isinstance(sigma, numbers.Number):
            s = max(sigma) ** 2
        else:
            s = sigma ** 2
        Hxx, Hxy, Hxz, Hyy, Hyz, Hzz = [h * s for h in H]
    else:
        Hxx, Hxy, Hxz, Hyy, Hyz, Hzz = H

    logger.info(f"+ Computing Hessian Eigenvalues with sigma {sigma}")

    Hs = torch.FloatTensor([[Hzz, Hyz, Hxz], [Hyz, Hyy, Hxy], [Hxz, Hxy, Hxx]])
    Hs_batch = Hs.reshape((3, 3, Hs.shape[-3] * Hs.shape[-2] * Hs.shape[-1]))
    e, v = torch.symeig(Hs_batch.permute((2, 0, 1)))
    e = e.reshape((Hs.shape[-3], Hs.shape[-2], Hs.shape[-1], 3))

    img: np.ndarray = kornia.tensor_to_image(e)
    img = np.transpose(img, (0, 3, 1, 2))
    
    logger.debug(f"Calculated hessian_eigvals reponse of shape {img.shape}")


    img = img[:,:,:,0] # return primay eigenvalue

    
    return np.nan_to_num(img)


def hessian_eigvals_p(data, sigma, correct=False):
    """Hessian eigenvalue image
    Returns primary eigenvalue

    Parameters
    ----------
    data : np.ndarray(D,H,W)
        Input image
    sigma : Vector of 3 float
        Kernel size
    correct : bool, optional
        Adjust sigma to square of input sigma, by default False

    Returns
    -------
    np.array
        (D,H,W,3) Hessian eigenvalues
    """
    H = compute_hessian(data, sigma)

    if correct:
        if not isinstance(sigma, numbers.Number):
            s = max(sigma) ** 2
        else:
            s = sigma ** 2
        Hxx, Hxy, Hxz, Hyy, Hyz, Hzz = [h * s for h in H]
    else:
        Hxx, Hxy, Hxz, Hyy, Hyz, Hzz = H

    logger.info(f"+ Computing Hessian Eigenvalues with sigma {sigma}")
    from survos2.improc.features._symeigval import symmetric_eig_pytorch

    response = symmetric_eig_pytorch(
        Hxx.copy(order="C"),
        Hxy.copy(order="C"),
        Hxz.copy(order="C"),
        Hyy.copy(order="C"),
        Hyz.copy(order="C"),
        Hzz.copy(order="C"),
    )[:, :, :, 0]

    logger.debug(f"Calculated hessian_eigvals reponse of shape {response.shape}")

    
    return rescale_denan(response)


def make_gaussian_1d(sigma=1.0, size=None, order=0, trunc=3):
    if size is None:
        size = sigma * trunc * 2 + 1
    x = np.arange(-(size // 2), (size // 2) + 1)
    if order > 2:
        raise ValueError("Only orders up to 2 are supported")
    # compute unnormalized Gaussian response
    response = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    if order == 1:
        response = -response * x
    elif order == 2:
        response = response * (x ** 2 - sigma ** 2)
    # normalize
    response /= np.abs(response).sum()

    return response.astype(np.float32)


def compute_structure_tensor(data, sigma):

    logger.info("+ Computing Structure Tensor")
    data = gaussian_blur_kornia(data, sigma=sigma)
    gradients = np.gradient(data)

    H_elems = [
        gradients[2 - ax0] * gradients[2 - ax1]
        for ax0, ax1 in combinations_with_replacement(range(3), 2)
    ]

    return H_elems


def compute_structure_tensor_determinant(data, sigma=1):
    """Structure tensor determinant filter

    https://en.wikipedia.org/wiki/Structure_tensor

    Parameters
    ----------
    data : np.ndarray(D,H,W)
        Input image, by default None
    sigma :  Vector of 3 float
        Kernel size, by default None

    Returns
    -------
    np.ndarray(D,H,W)
        Each voxel of the filtered image is the determiant of the structure tensor at that voxel.
    """
    Sxx, Sxy, Sxz, Syy, Syz, Szz = compute_structure_tensor(data=data, sigma=sigma)

    logger.info("+ Computing Structure Tensor Determinant")
    determinant = (
        Sxx * (Syy * Szz - Syz * Syz)
        - Sxy * (Sxy * Szz - Syz * Sxz)
        + Sxz * (Sxy * Syz - Syy * Sxz)
    )
    
    
    return rescale_denan(determinant)



def compute_structure_tensor_eigvals(data, sigma):
    """Structure tensor eigenvalues

    Parameters
    ----------
    data : np.ndarray(D,H,W)
        Input image
    sigma : Vector of 3 floats
        Kernel size

    Returns
    -------
    np.ndarray()
        [description]
    """
    Sxx, Sxy, Sxz, Syy, Syz, Szz = compute_structure_tensor(data, sigma)
    logger.info("+ Computing Structure Tensor Eigenvalues")

    img_t = (
        kornia.utils.image_to_tensor(np.array(img_gray))
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
    )

    R, eigvec = torch.symeig(img_t, eigenvectors=False)
    R: np.ndarray = kornia.tensor_to_image(R.float())

    return R


def divide_nonzero(array1, array2):
    denominator = np.copy(array2)
    denominator[denominator == 0] = 1e-10
    return np.divide(array1, denominator)


def compute_frangi(
    data,
    scale_range=(1.0, 4.0),
    scale_step=1.0,
    alpha=0.5,
    beta=0.5,
    gamma=15,
    dark_response=False,
):
    """Frangi filter for pulling out elongated structures

    Iteratively calculates hessian eigenvalues for a range of sigma values and takes the max eigenvalue per voxel.

    Frangi, A. F., Niessen, W. J., Vincken, K. L., & Viergever, M. A. (1998, October). Multiscale vessel enhancement filtering.

    Parameters
    ----------
    data : np.ndarray(D,H,W)
        Input image
    scale_range : tuple, optional
        Range of values to use as kernel size, by default (1.0, 4.0)
    scale_step : float, optional
        Step of values for kernel, by default 1.0
    alpha : float, optional
        by default 0.5
    beta : float, optional
        by default 0.5
    gamma : int, optional
        by default 15
    dark_response : bool, optional
        Pull out dark blobs on light, by default False

    Returns
    -------
    np.ndarray(D,H,W)
        Frangi-filtered image

    """
    logger.info(f"+ Computing frangi on input data of shape {data.shape}")

    sigmas = np.arange(scale_range[0], scale_range[1], scale_step)

    if np.any(np.asarray(sigmas) < 0.0):
        raise ValueError("Sigma values less than zero are not valid")

    filtered_array = np.zeros(sigmas.shape + data.shape)
    logger.info(f"+ Created frangi filter array of shape {filtered_array.shape}")

    for i, sigma in enumerate(sigmas):
        R = hessian_eigvals(data, sigma, correct=True)

        e1 = R[..., 0]
        e2 = R[..., 1]
        e3 = R[..., 2]

        ae1 = np.abs(e1)
        ae2 = np.abs(e2)
        ae3 = np.abs(e3)

        ae1sq = ae1 * ae1
        ae2sq = ae2 * ae2
        ae3sq = ae3 * ae3

        Ra = divide_nonzero(ae2sq, ae3sq)
        Rb = divide_nonzero(ae1sq, (ae2 * ae3))
        S = np.sqrt(ae1sq + ae2sq + ae3sq)

        A = 2 * alpha ** 2
        B = 2 * beta ** 2
        C = 2 * gamma ** 2

        plate_factor = 1 - np.exp(-(Ra ** 2) / A)
        blob_factor = np.exp(-(Rb) / B)
        background_factor = 1 - np.exp(-(S ** 2) / C)

        tmp = plate_factor * blob_factor * background_factor

        # filter out background
        if dark_response:
            tmp[e2 < 0] = 0
            tmp[e3 < 0] = 0
        else:
            tmp[e2 > 0] = 0
            tmp[e3 > 0] = 0

        filtered_array[i] = np.nan_to_num(tmp)
        logger.info(f"+ Computed frangi filtered array of shape {filtered_array.shape}")

    return np.max(filtered_array, axis=0)
