import numbers
import os.path as op

import hug
import numpy as np
from loguru import logger

from survos2.api import workspace as ws
from survos2.api.types import (
    DataURI,
    Float,
    FloatOrVector,
    Int,
    IntList,
    SmartBoolean,
    String,
    IntOrVector,
)
from survos2.api.utils import dataset_repr, get_function_api
from survos2.api.utils import save_metadata, dataset_repr
from survos2.improc import map_blocks
from survos2.io import dataset_from_uri
from survos2.utils import encode_numpy, decode_numpy
from survos2.model import DataModel
from survos2.improc.utils import DatasetManager
from survos2.server.state import cfg
from survos2.frontend.control.launcher import Launcher
from survos2.model import DataModel

__feature_group__ = "features"
__feature_dtype__ = "float32"
__feature_fill__ = 0

def rescale_denan(img):
    img = img - np.min(img)
    img = img / np.max(img)
    img = np.nan_to_num(img)
    return img

@hug.post()
def upload(body, request, response):
    print(f"Request: {request}")
    print(f"Response: {response}")
    encoded_feature = body['file']
    array_shape = body['shape']
    print(f"shape {array_shape}")
    feature = np.frombuffer(encoded_feature, dtype="float32")
    from ast import literal_eval 
    feature.shape = literal_eval(array_shape)
    print(f"Uploaded feature of shape {feature.shape}")
    
    params = dict(feature_type="raw", workspace=DataModel.g.current_workspace)
    result = create(**params)
    fid = result["id"]
    ftype = result["kind"]
    fname = result["name"]
    logger.debug(f"Created new object in workspace {fid}, {ftype}, {fname}")
    result = DataModel.g.dataset_uri(fid, group="features")
    with DatasetManager(result, out=result, dtype="float32", fillvalue=0) as DM:
        DM.out[:] = feature



@hug.get()
def get_volume(src: DataURI):
    logger.debug("Getting feature volume")
    ds = dataset_from_uri(src, mode="r")
    data = ds[:]
    return encode_numpy(data)


@hug.get()
def get_crop(src: DataURI, roi: IntList):
    logger.debug("Getting feature crop")
    ds = dataset_from_uri(src, mode="r")
    data = ds[roi[0] : roi[1], roi[2] : roi[3], roi[4] : roi[5]]
    return encode_numpy(data)


@hug.get()
def get_slice(src: DataURI, slice_idx: Int, order: tuple):
    order = np.array(order)
    ds = dataset_from_uri(src, mode="r")[:]
    ds = np.transpose(ds, order).astype(np.float32)
    data = ds[slice_idx]
    return encode_numpy(data.astype(np.float32))


@hug.get()
@save_metadata
def structure_tensor_determinant(
    src: DataURI, dst: DataURI, sigma: FloatOrVector = 1
) -> "BLOB":
    from ..server.filtering.blob import compute_structure_tensor_determinant

    map_blocks(
        compute_structure_tensor_determinant,
        src,
        out=dst,
        sigma=sigma,
        pad=max(4, int((max(sigma) * 2))),
        normalize=True,
    )


@hug.get()
@save_metadata
def frangi(
    src: DataURI,
    dst: DataURI,
    scale_min: Float = 1.0,
    scale_max: Float = 4.0,
    scale_step: Float = 1.0,
    alpha: Float = 0.5,
    beta: Float = 0.5,
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


@hug.get()
@save_metadata
def hessian_eigenvalues(src: DataURI, dst: DataURI, sigma: FloatOrVector = 1) -> "BLOB":
    from ..server.filtering.blob import hessian_eigvals_image

    
    map_blocks(
        hessian_eigvals_image,
        src,
        out=dst,
        pad=max(4, int((max(sigma) * 2))),
        sigma=sigma,
        normalize=True,
    )


def pass_through(x):
    return x


@hug.get()
@save_metadata
def raw(src: DataURI, dst: DataURI) -> "BASE":
    map_blocks(pass_through, src, out=dst, normalize=True)


@hug.get()
@save_metadata
def simple_invert(src: DataURI, dst: DataURI) -> "BASE":
    from ..server.filtering import simple_invert

    map_blocks(simple_invert, src, out=dst, normalize=True)


@hug.get()
@save_metadata
def invert_threshold(src: DataURI, dst: DataURI, thresh: Float = 0.5) -> "BASE":
    from ..server.filtering import invert_threshold

    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        src_dataset_arr = DM.sources[0][:]
        filtered = invert_threshold(src_dataset_arr, thresh=thresh)

    map_blocks(pass_through, src, out=dst, thresh=thresh, normalize=True)


@hug.get()
@save_metadata
def threshold(src: DataURI, dst: DataURI, thresh: Float = 0.5) -> "BASE":
    from ..server.filtering import threshold as threshold_fn

    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        src_dataset_arr = DM.sources[0][:]
        filtered = threshold_fn(src_dataset_arr, thresh=thresh)

    map_blocks(pass_through, filtered, out=dst, normalize=False)


@hug.get()
@save_metadata
def rescale(src: DataURI, dst: DataURI) -> "BASE":

    logger.debug(f"Rescaling src {src}")
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        src_dataset = DM.sources[0][:]

        filtered = rescale_denan(src_dataset)

    map_blocks(pass_through, filtered, out=dst, normalize=False)


@hug.get()
@save_metadata
def gamma_correct(src: DataURI, dst: DataURI, gamma: Float = 1.0) -> "BASE":
    from ..server.filtering import gamma_adjust

    map_blocks(gamma_adjust, src, gamma=gamma, out=dst, normalize=True)


@hug.get()
@save_metadata
def dilation(src: DataURI, dst: DataURI, num_iter: Int = 1) -> "MORPHOLOGY":
    from ..server.filtering import dilate

    map_blocks(
        dilate,
        src,
        num_iter=num_iter,
        out=dst,
        normalize=True,
        pad=max(4, int(num_iter * 2)),
    )


@hug.get()
@save_metadata
def erosion(src: DataURI, dst: DataURI, num_iter: Int = 1) -> "MORPHOLOGY":
    from ..server.filtering import erode

    map_blocks(
        erode,
        src,
        num_iter=num_iter,
        out=dst,
        normalize=True,
        pad=max(4, int(num_iter * 2)),
    )


@hug.get()
@save_metadata
def opening(src: DataURI, dst: DataURI, num_iter: Int = 1) -> "MORPHOLOGY":
    from ..server.filtering import opening

    map_blocks(
        opening,
        src,
        num_iter=num_iter,
        out=dst,
        normalize=True,
        pad=max(4, int(num_iter * 2)),
    )


@hug.get()
@save_metadata
def closing(src: DataURI, dst: DataURI, num_iter: Int = 1) -> "MORPHOLOGY":
    from ..server.filtering import closing

    map_blocks(
        closing,
        src,
        num_iter=num_iter,
        out=dst,
        normalize=True,
        pad=max(4, int(num_iter * 2)),
    )


@hug.get()
@save_metadata
def distance_transform_edt(src: DataURI, dst: DataURI) -> "MORPHOLOGY":
    from ..server.filtering import distance_transform_edt

    logger.debug(f"Calculating distance transform")
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        src_dataset_arr = DM.sources[0][:]
        filtered = distance_transform_edt(src_dataset_arr)

    map_blocks(pass_through, filtered, out=dst, normalize=False)


@hug.get()
@save_metadata
def skeletonize(src: DataURI, dst: DataURI) -> "MORPHOLOGY":
    from ..server.filtering import skeletonize

    logger.debug(f"Calculating medial axis")
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        src_dataset_arr = DM.sources[0][:]
        filtered = skeletonize(src_dataset_arr)

    map_blocks(pass_through, filtered, out=dst, normalize=False)


@hug.get()
@save_metadata
def tvdenoise(
    src: DataURI,
    dst: DataURI,
    regularization_amount: Float = 0.001,
    pad: Int = 8,
    max_iter: Int = 100,
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


@hug.get()
@save_metadata
def spatial_gradient_3d(src: DataURI, dst: DataURI, dim: Int = 0) -> "EDGES":
    from ..server.filtering import spatial_gradient_3d

    map_blocks(
        spatial_gradient_3d,
        src,
        out=dst,
        dim=dim,
        normalize=True,
    )


@hug.get()
@save_metadata
def difference_of_gaussians(
    src: DataURI, dst: DataURI, sigma: FloatOrVector = 1, sigma_ratio: Float = 2
) -> "EDGES":
    from ..server.filtering.edge import compute_difference_gaussians

    map_blocks(
        compute_difference_gaussians,
        src,
        out=dst,
        sigma=sigma,
        sigma_ratio=sigma_ratio,
        pad=max(4, int((max(sigma) * 3))),
        normalize=False,
    )


@hug.get()
@save_metadata
def gaussian_blur(src: DataURI, dst: DataURI, sigma: FloatOrVector = 1) -> "DENOISING":
    from ..server.filtering import gaussian_blur_kornia

    if isinstance(sigma, numbers.Number):
        sigma = (sigma, sigma, sigma)
    map_blocks(
        gaussian_blur_kornia,
        src,
        out=dst,
        sigma=sigma,
        pad=max(4, int(max(sigma))),
        normalize=False,
    )


@hug.get()
@save_metadata
def laplacian(src: DataURI, dst: DataURI, kernel_size: FloatOrVector = 1) -> "EDGES":
    from ..server.filtering import ndimage_laplacian

    map_blocks(
        ndimage_laplacian,
        src,
        out=dst,
        kernel_size=kernel_size,
        pad=max(4, int(max(kernel_size)) * 3),
        normalize=False,
    )


@hug.get()
@save_metadata
def gaussian_norm(
    src: DataURI, dst: DataURI, sigma: FloatOrVector = 1
) -> "NEIGHBORHOOD":
    from ..server.filtering.blur import gaussian_norm

    map_blocks(
        gaussian_norm,
        src,
        out=dst,
        sigma=sigma,
        pad=max(4, int((max(sigma) * 2))),
        normalize=True,
    )


@hug.get()
@save_metadata
def gaussian_center(
    src: DataURI, dst: DataURI, sigma: FloatOrVector = 1
) -> "NEIGHBORHOOD":

    from ..server.filtering.blur import gaussian_center

    map_blocks(
        gaussian_center,
        src,
        out=dst,
        sigma=sigma,
        pad=max(4, int((max(sigma) * 2))),
        normalize=True,
    )


@hug.get()
@save_metadata
def median(
    src: DataURI, dst: DataURI, median_size: Int = 1, num_iter: Int = 1
) -> "DENOISING":
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


@hug.get()
@save_metadata
def wavelet(
    src: DataURI,
    dst: DataURI,
    threshold: Float = 64.0,
    level: Int = 1,
    wavelet: String = "sym3",
    hard: SmartBoolean = True,
) -> "WAVELET":
    from ..server.filtering import wavelet

    map_blocks(
        wavelet,
        src,
        level=level,
        out=dst,
        normalize=True,
        wavelet="sym3",
        threshold=threshold,
        hard=hard,
        pad=max(4, int(level * 2)),
    )


@hug.get()
def create(workspace: String, feature_type: String):
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


@hug.get()
@hug.local()
def existing(
    workspace: String, full: SmartBoolean = False, filter: SmartBoolean = True
):
    datasets = ws.existing_datasets(workspace, group=__feature_group__)
    if full:
        datasets = {
            "{}/{}".format(__feature_group__, k): dataset_repr(v)
            for k, v in datasets.items()
        }
    else:
        datasets = {k: dataset_repr(v) for k, v in datasets.items()}
    if filter:
        datasets = {k: v for k, v in datasets.items() if v["kind"] != "unknown"}
    return datasets


@hug.get()
def remove(workspace: String, feature_id: String):
    ws.delete_dataset(workspace, feature_id, group=__feature_group__)


@hug.get()
def rename(workspace: String, feature_id: String, new_name: String):
    ws.rename_dataset(workspace, feature_id, __feature_group__, new_name)


@hug.get()
def group():
    return __feature_group__


@hug.get()
def available():
    h = hug.API(__name__)
    all_features = []
    for name, method in h.http.routes[""].items():
        if name[1:] in [
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
        name = name[1:]
        func = method["GET"][None].interface.spec
        desc = get_function_api(func)
        category = desc["returns"] or "Others"
        desc = dict(name=name, params=desc["params"], category=category)
        all_features.append(desc)
    return all_features


