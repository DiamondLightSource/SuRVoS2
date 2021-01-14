import numpy as np
from itertools import combinations_with_replacement
from loguru import logger
from .blur import gaussian_blur_kornia
import torch
import kornia


def compute_hessian(data, sigma):
    logger.info("+ Computing Hessian Matrix")
    gaussian_filtered = gaussian_blur_kornia(data, sigma=sigma)
    gradients = [np.gradient(gaussian_filtered, axis=i) for i in range(3)]
    axes = range(data.ndim)
    H_elems = [
        np.gradient(gradients[2 - ax0], axis=2 - ax1)
        for ax0, ax1 in combinations_with_replacement(axes, 2)
    ]
    sigma = max(sigma)
    if sigma > 1:
        H_elems = [elem * sigma ** 2 for elem in H_elems]
    return H_elems


def compute_hessian_determinant(data, sigma, bright=False):
    Hxx, Hxy, Hxz, Hyy, Hyz, Hzz = compute_hessian(data, sigma)

    logger.info("+ Computing Hessian Determinant")

    det = (
        Hxx * (Hyy * Hzz - Hyz * Hyz)
        - Hxy * (Hxy * Hzz - Hyz * Hxz)
        + Hxz * (Hxy * Hyz - Hyy * Hxz)
    )

    if bright:
        return -det

    return det


def hessian_eigvals(data, sigma, correct=False):  # TODO: GPU THIS
    H = compute_hessian(data, sigma)

    if correct:
        s = max(sigma) ** 2
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
    )[:,:,:,0]

    logger.debug(f"Calculated hessian_eigvals reponse of shape {response.shape}")
    # img_t =  kornia.utils.image_to_tensor(np.array(Hxx)).float().unsqueeze(0).unsqueeze(0)
    # response, eigvec = torch.symeig(img_t, eigenvectors=False)
    # response: np.ndarray = kornia.tensor_to_image(response.float())

    from .base import rescale_denan

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


def compute_structure_tensor(data=None, sigma=None):
    logger.info("+ Computing Structure Tensor")

    data = gaussian_blur_kornia(data, sigma=sigma)

    gradients = np.gradient(data)

    H_elems = [
        gradients[2 - ax0] * gradients[2 - ax1]
        for ax0, ax1 in combinations_with_replacement(range(3), 2)
    ]

    return H_elems


def compute_structure_tensor_determinant(data=None, sigma=1):
    Sxx, Sxy, Sxz, Syy, Syz, Szz = compute_structure_tensor(data=data, sigma=sigma)

    logger.info("+ Computing Structure Tensor Determinant")
    return (
        Sxx * (Syy * Szz - Syz * Syz)
        - Sxy * (Sxy * Szz - Syz * Sxz)
        + Sxz * (Sxy * Syz - Syy * Sxz)
    )


def compute_structure_tensor_eigvals(data, params):
    Sxx, Sxy, Sxz, Syy, Syz, Szz = compute_structure_tensor(data, sigma)
    logger.info("+ Computing Structure Tensor Eigenvalues")

    img_t = (
        kornia.utils.image_to_tensor(np.array(img_gray))
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
    )

    # R = symmetric_eigvals3S_gpu(Sxx, Sxy, Sxz, Syy, Syz, Szz, gpu=DM.selected_gpu)
    R, eigvec = torch.symeig(img_t, eigenvectors=False)
    R: np.ndarray = kornia.tensor_to_image(R.float())

    return R  # R[..., params['Eigen Value']].copy()


def compute_frangi(data=None, sigma=1.0, max_sigma=2.0, sincr=0.2):
    logger.info("+ Computing frangi")
    result = None

    while max(sigma) < max_sigma:
        R = hessian_eigvals(
            data=data, params=dict(Sigma=sigma), correct=True, doabs=True
        )
        e1 = R[..., 0]
        e2 = R[..., 1]
        e3 = R[..., 2]

        ae1 = np.abs(e1)
        ae2 = np.abs(e2)
        ae3 = np.abs(e3)

        ae1sq = ae1 * ae1
        ae2sq = ae2 * ae2
        ae3sq = ae3 * ae3

        Ra = ae2sq / ae3sq
        Rb = ae1sq / (ae2 * ae3)
        S = ae1sq + ae2sq + ae3sq

        A = B = 2 * (params["Lamda"] ** 2)
        C = 2 * S.max()

        expRa = 1 - np.exp(-Ra / A)
        expRb = np.exp(-Rb / B)
        expS = 1 - np.exp(-S / C)

        tmp = expRa * expRb * expS

        if params["Response"] == "Dark":
            tmp[e2 < 0] = 0
            tmp[e3 < 0] = 0
        else:
            tmp[e2 > 0] = 0
            tmp[e3 > 0] = 0

        tmp[np.isnan(tmp)] = 0

        if result is None:
            result = tmp
        else:
            np.maximum(result, tmp, out=result)

        sigma += sincr

    return result
