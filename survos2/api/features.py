import numpy as np
from loguru import logger

from survos2.api import workspace as ws

from survos2.api.utils import dataset_repr, get_function_api
from survos2.api.utils import save_metadata, dataset_repr, pass_through
from survos2.improc import map_blocks
from survos2.data_io import dataset_from_uri
from survos2.utils import encode_numpy, encode_numpy_slice
from survos2.config import Config
from survos2.model import DataModel
from survos2.improc.utils import DatasetManager
from survos2.api._features.wavelet import features as wavelet
from survos2.api._features.blur import features as blur
from survos2.api._features.base import features as base
from survos2.api._features.morph import features as morph
from survos2.api._features.composite import features as composite
from survos2.api._features.blob import features as blob



from typing import List
import pickle
from fastapi import APIRouter, Query, File, UploadFile


features = APIRouter()
features.include_router(wavelet)
features.include_router(blur)
features.include_router(base)
features.include_router(morph)
features.include_router(composite)
features.include_router(blob)


__feature_group__ = "features"
__feature_dtype__ = "float32"
__feature_fill__ = 0

CHUNK_SIZE =  Config["computing.chunk_size_sparse"]


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





@features.get("/create")
def create(workspace: str, feature_type: str):
    ds = ws.auto_create_dataset(
        workspace,
        feature_type,
        __feature_group__,
        __feature_dtype__,
        fill=__feature_fill__,
        chunks=CHUNK_SIZE
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
