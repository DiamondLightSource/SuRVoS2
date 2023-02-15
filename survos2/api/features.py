import numpy as np
from loguru import logger

from survos2.api import workspace as ws

from survos2.api.utils import dataset_repr, get_function_api
from survos2.api.utils import save_metadata, dataset_repr
from survos2.improc import map_blocks
from survos2.data_io import dataset_from_uri
from survos2.utils import encode_numpy, decode_numpy, encode_numpy_slice
from survos2.model import DataModel
from survos2.improc.utils import DatasetManager
from survos2.model import DataModel

from typing import Union, List, Optional
import pickle
from fastapi import APIRouter, Body, Query, File, UploadFile


features = APIRouter()


__feature_group__ = "features"
__feature_dtype__ = "float32"
__feature_fill__ = 0


def rescale_denan(img):
    img = img - np.min(img)
    img = img / np.max(img)
    img = np.nan_to_num(img)
    return img


@features.post("/upload")
def upload(file: UploadFile = File(...)):
    """Upload features layer as an array (via Launcher) to the current workspace.
    After unpickling there is a dictionary with a 'name' key and a
       'data' key. The 'data' key contains a numpy array of labels.
    """
    encoded_buffer = file.file.read()
    d = pickle.loads(encoded_buffer)

    feature = np.array(d["data"])

    params = dict(feature_type="raw", workspace=DataModel.g.current_workspace)
    result = create(**params)
    fid = result["id"]
    ftype = result["kind"]
    fname = result["name"]
    logger.debug(f"Created new object in workspace {fid}, {ftype}, {fname}")
    result = DataModel.g.dataset_uri(fid, group="features")
    with DatasetManager(result, out=result, dtype="float32", fillvalue=0) as DM:
        DM.out[:] = feature


@features.get("/get_volume")
def get_volume(src: str):
    logger.debug("Getting feature volume")
    ds = dataset_from_uri(src, mode="r")
    data = ds[:]
    return encode_numpy(data)


@features.get("/get_crop")
def get_crop(src: str, roi: list):
    logger.debug("Getting feature crop")
    ds = dataset_from_uri(src, mode="r")
    data = ds[roi[0] : roi[1], roi[2] : roi[3], roi[4] : roi[5]]
    return encode_numpy(data)


@features.get("/get_slice")
def get_slice(src: str, slice_idx: int, order: tuple):
    order = np.array(order)
    ds = dataset_from_uri(src, mode="r")[:]
    ds = np.transpose(ds, order).astype(np.float32)
    data = ds[slice_idx]
    return encode_numpy_slice(data.astype(np.float32))


def simple_norm(dst):
    with DatasetManager(dst, out=None, dtype="float32", fillvalue=0) as DM:
        arr = DM.sources[0][:]
        arr -= np.min(arr)
        arr /= np.max(arr)
    map_blocks(pass_through, arr, out=dst, normalize=False)


@features.get("/feature_composite")
@save_metadata
def feature_composite(
    src: str,
    dst: str,
    workspace: str,
    feature_A: str,
    feature_B: str,
    op: str,
):
    DataModel.g.current_workspace = workspace
    src_A = DataModel.g.dataset_uri(feature_A, group="features")
    with DatasetManager(src_A, out=None, dtype="uint16", fillvalue=0) as DM:
        src_A_dataset = DM.sources[0]
        src_A_arr = src_A_dataset[:]
        logger.info(f"Obtained src A with shape {src_A_arr.shape}")

    src_B = DataModel.g.dataset_uri(feature_B, group="features")
    with DatasetManager(src_B, out=None, dtype="uint16", fillvalue=0) as DM:
        src_B_dataset = DM.sources[0]
        src_B_arr = src_B_dataset[:]
        logger.info(f"Obtained src B with shape {src_B_arr.shape}")
    if op == "+":
        result = src_A_arr + src_B_arr
    else:
        result = src_A_arr * src_B_arr

    map_blocks(pass_through, result, out=dst, normalize=False)


@features.get("/structure_tensor_determinant")
@save_metadata
def structure_tensor_determinant(src: str, dst: str, sigma: List[int] = Query()) -> "BLOB":
    from ..server.filtering.blob import compute_structure_tensor_determinant

    map_blocks(
        compute_structure_tensor_determinant,
        src,
        out=dst,
        sigma=sigma,
        pad=max(4, int((max(sigma) * 2))),
        normalize=True,
    )

    simple_norm(dst)


@features.get("/frangi")
@save_metadata
def frangi(
    src: str,
    dst: str,
    scale_min: float = 1.0,
    scale_max: float = 4.0,
    scale_step: float = 1.0,
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma=15,
) -> "BLOB":
    from ..server.filtering.blob import compute_frangi

    map_blocks(
        compute_frangi,
        src,
        out=dst,
        scale_range=(scale_min, scale_max),
        scale_step=1.0,
        alpha=0.5,
        beta=0.5,
        gamma=15,
        dark_response=True,
        normalize=True,
        pad=max(4, int((scale_max * 2))),
    )

    simple_norm(dst)


@features.get("/hessian_eigenvalues")
@save_metadata
def hessian_eigenvalues(src: str, dst: str, sigma: List[int] = Query()) -> "BLOB":
    from ..server.filtering.blob import hessian_eigvals_image

    map_blocks(
        hessian_eigvals_image,
        src,
        out=dst,
        pad=max(4, int((max(sigma) * 2))),
        sigma=sigma,
        normalize=True,
    )

    simple_norm(dst)


def pass_through(x):
    return x


@features.get("/raw")
@save_metadata
def raw(src: str, dst: str) -> "BASE":
    map_blocks(pass_through, src, out=dst, normalize=True)


@features.get("/simple_invert")
@save_metadata
def simple_invert(src: str, dst: str) -> "BASE":
    from ..server.filtering import simple_invert

    map_blocks(simple_invert, src, out=dst, normalize=True)


@features.get("/invert_threshold")
@save_metadata
def invert_threshold(src: str, dst: str, thresh: float = 0.5) -> "BASE":
    from ..server.filtering import invert_threshold

    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        src_dataset_arr = DM.sources[0][:]
        filtered = invert_threshold(src_dataset_arr, thresh=thresh)

    map_blocks(pass_through, filtered, out=dst, normalize=False)


@features.get("/threshold")
@save_metadata
def threshold(src: str, dst: str, threshold: float = 0.5) -> "BASE":
    from ..server.filtering import threshold as threshold_fn

    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        src_dataset_arr = DM.sources[0][:]
        filtered = threshold_fn(src_dataset_arr, thresh=threshold)

    map_blocks(pass_through, filtered, out=dst, normalize=False)


@features.get("/rescale")
@save_metadata
def rescale(src: str, dst: str) -> "BASE":

    logger.debug(f"Rescaling src {src}")
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        src_dataset = DM.sources[0][:]

        filtered = rescale_denan(src_dataset)

    map_blocks(pass_through, filtered, out=dst, normalize=False)


@features.get("/gamma_correct")
@save_metadata
def gamma_correct(src: str, dst: str, gamma: float = 1.0) -> "BASE":
    from ..server.filtering import gamma_adjust

    map_blocks(gamma_adjust, src, gamma=gamma, out=dst, normalize=True)


@features.get("/dilation")
@save_metadata
def dilation(src: str, dst: str, num_iter: int = 1) -> "MORPHOLOGY":
    from ..server.filtering import dilate

    map_blocks(
        dilate,
        src,
        num_iter=num_iter,
        out=dst,
        normalize=True,
        pad=max(4, int(num_iter * 2)),
    )


@features.get("/erosion")
@save_metadata
def erosion(src: str, dst: str, num_iter: int = 1) -> "MORPHOLOGY":
    from ..server.filtering import erode

    map_blocks(
        erode,
        src,
        num_iter=num_iter,
        out=dst,
        normalize=True,
        pad=max(4, int(num_iter * 2)),
    )


@features.get("/opening")
@save_metadata
def opening(src: str, dst: str, num_iter: int = 1) -> "MORPHOLOGY":
    from ..server.filtering import opening

    map_blocks(
        opening,
        src,
        num_iter=num_iter,
        out=dst,
        normalize=True,
        pad=max(4, int(num_iter * 2)),
    )


@features.get("/closing")
@save_metadata
def closing(src: str, dst: str, num_iter: int = 1) -> "MORPHOLOGY":
    from ..server.filtering import closing

    map_blocks(
        closing,
        src,
        num_iter=num_iter,
        out=dst,
        normalize=True,
        pad=max(4, int(num_iter * 2)),
    )


@features.get("/distance_transform_edt")
@save_metadata
def distance_transform_edt(src: str, dst: str) -> "MORPHOLOGY":
    from ..server.filtering import distance_transform_edt

    logger.debug(f"Calculating distance transform")
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        src_dataset_arr = DM.sources[0][:]
        filtered = distance_transform_edt(src_dataset_arr)

    map_blocks(pass_through, filtered, out=dst, normalize=False)


@features.get("/skeletonize")
@save_metadata
def skeletonize(src: str, dst: str) -> "MORPHOLOGY":
    from ..server.filtering import skeletonize

    logger.debug(f"Calculating medial axis")
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        src_dataset_arr = DM.sources[0][:]
        filtered = skeletonize(src_dataset_arr)

    map_blocks(pass_through, filtered, out=dst, normalize=False)


@features.get("/tvdenoise")
@save_metadata
def tvdenoise(
    src: str,
    dst: str,
    regularization_amount: float = 0.001,
    pad: int = 8,
    max_iter: int = 100,
) -> "DENOISING":
    from ..server.filtering.blur import tvdenoise_kornia

    map_blocks(
        tvdenoise_kornia,
        src,
        out=dst,
        regularization_amount=regularization_amount,
        max_iter=max_iter,
        pad=pad,
        normalize=True,
    )


@features.get("/spatial_gradient_3d")
@save_metadata
def spatial_gradient_3d(src: str, dst: str, dim: int = 0) -> "EDGES":
    from ..server.filtering import spatial_gradient_3d

    map_blocks(
        spatial_gradient_3d,
        src,
        out=dst,
        dim=dim,
        normalize=True,
    )

    simple_norm(dst)


@features.get("/difference_of_gaussians")
@save_metadata
def difference_of_gaussians(
    src: str, dst: str, sigma: List[int] = Query(), sigma_ratio: float = 2
) -> "EDGES":
    from ..server.filtering.edge import compute_difference_gaussians

    if isinstance(sigma, float) or isinstance(sigma, int):
        sigma = np.array([sigma] * 3)
    map_blocks(
        compute_difference_gaussians,
        src,
        out=dst,
        sigma=sigma,
        sigma_ratio=sigma_ratio,
        pad=max(4, int((np.max(sigma) * 3))),
        normalize=True,
    )
    simple_norm(dst)


@features.get("/gaussian_blur")
@save_metadata
def gaussian_blur(src: str, dst: str, sigma: List[int] = Query()) -> "DENOISING":
    from ..server.filtering import gaussian_blur_kornia

    if isinstance(sigma, float) or isinstance(sigma, int):
        sigma = np.array([sigma] * 3)

    if sigma[0] == 0:
        from skimage.filters import gaussian

        map_blocks(
            gaussian,
            src,
            out=dst,
            sigma=(1.0, sigma[1], sigma[2]),
            pad=0,
            normalize=False,
        )

    else:
        map_blocks(
            gaussian_blur_kornia,
            src,
            out=dst,
            sigma=sigma,
            pad=max(4, int(max(sigma))),
            normalize=False,
        )


@features.get("/laplacian")
@save_metadata
def laplacian(src: str, dst: str, sigma: List[int] = Query()) -> "EDGES":
    from ..server.filtering import ndimage_laplacian

    map_blocks(
        ndimage_laplacian,
        src,
        out=dst,
        kernel_size=sigma,
        # pad=max(4, int(max(np.array(kernel_size))) * 3),
        pad=max(4, int(np.max(sigma))),
        normalize=False,
    )

    simple_norm(dst)


@features.get("/gaussian_norm")
@save_metadata
def gaussian_norm(src: str, dst: str, sigma: List[int] = Query()) -> "NEIGHBORHOOD":
    from ..server.filtering.blur import gaussian_norm

    map_blocks(
        gaussian_norm,
        src,
        out=dst,
        sigma=sigma,
        pad=max(4, int((max(sigma) * 2))),
        normalize=False,
    )

    simple_norm(dst)


@features.get("/gaussian_center")
@save_metadata
def gaussian_center(src: str, dst: str, sigma: List[int] = Query()) -> "NEIGHBORHOOD":

    from ..server.filtering.blur import gaussian_center

    map_blocks(
        gaussian_center,
        src,
        out=dst,
        sigma=sigma,
        pad=max(4, int((max(sigma) * 2))),
        normalize=False,
    )
    simple_norm(dst)


@features.get("/median")
@save_metadata
def median(src: str, dst: str, median_size: int = 1, num_iter: int = 1) -> "DENOISING":
    from ..server.filtering import median

    map_blocks(
        median,
        src,
        median_size=median_size,
        num_iter=num_iter,
        out=dst,
        pad=max(4, int((median_size * 2))),
        normalize=False,
    )


@features.get("/wavelet")
@save_metadata
def wavelet(
    src: str,
    dst: str,
    threshold: float = 64.0,
    level: int = 1,
    wavelet: str = "sym3",
    hard: bool = True,
) -> "WAVELET":
    from ..server.filtering import wavelet as wavelet_fn

    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        src_dataset_arr = DM.sources[0][:]

    result = wavelet_fn(
        src_dataset_arr, level=level, wavelet=str(wavelet), threshold=threshold, hard=hard
    )
    map_blocks(pass_through, result, out=dst, normalize=False)


@features.get("/create")
def create(workspace: str, feature_type: str):
    ds = ws.auto_create_dataset(
        workspace,
        feature_type,
        __feature_group__,
        __feature_dtype__,
        fill=__feature_fill__,
    )
    ds.set_attr("kind", feature_type)
    logger.debug(f"Created (empty) feature of kind {feature_type}")
    return dataset_repr(ds)


# @hug.local()
@features.get("/existing")
def existing(workspace: str, full: bool = False, filter: bool = True):
    datasets = ws.existing_datasets(workspace, group=__feature_group__)
    if full:
        datasets = {
            "{}/{}".format(__feature_group__, k): dataset_repr(v) for k, v in datasets.items()
        }
    else:
        datasets = {k: dataset_repr(v) for k, v in datasets.items()}
    if filter:
        datasets = {k: v for k, v in datasets.items() if v["kind"] != "unknown"}
    return datasets


@features.get("/remove")
def remove(workspace: str, feature_id: str):
    ws.delete_dataset(workspace, feature_id, group=__feature_group__)
    return {"done": "ok"}


@features.get("/rename")
def rename(workspace: str, feature_id: str, new_name: str):
    ws.rename_dataset(workspace, feature_id, __feature_group__, new_name)
    return {"done": "ok"}


@features.get("/group")
def group():
    return __feature_group__


@features.get("/available")
def available():
    h = features  # hug.API(__name__)
    all_features = []
    for r in h.routes:
        name = r.name
        # method = r.methods
        if name in [
            "available",
            "create",
            "existing",
            "remove",
            "rename",
            "group",
            "get_volume",
            "get_slice",
            "get_crop",
            "upload",
        ]:
            continue
        func = r.endpoint
        desc = get_function_api(func)
        category = desc["returns"] or "Others"
        desc = dict(name=name, params=desc["params"], category=category)
        all_features.append(desc)
    return all_features
