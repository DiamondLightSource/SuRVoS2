import numpy as np
from loguru import logger
import ast
from survos2.api import workspace as ws
from survos2.api.utils import APIException, dataset_repr
from survos2.config import Config
from survos2.data_io import dataset_from_uri
from survos2.utils import encode_numpy
from survos2.model import DataModel
from survos2.improc.utils import DatasetManager
from survos2.frontend.view_fn import get_level_from_server
from survos2.api.annotate import annotate_voxels as _annotate_voxels
from survos2.api.annotate import annotate_regions as _annotate_regions
from survos2.api.annotate import annotate_from_slice as _annotate_from_slice

import pickle
from fastapi import APIRouter, Body, File, UploadFile, Query

__level_fill__ = 0
__level_dtype__ = "uint32"
__group_pattern__ = "annotations"


def pass_through(x):
    return x


annotations = APIRouter()


CHUNK_SIZE = Config["computing.chunk_size_sparse"]


def to_label(idx=0, name="Label", color="#000000", visible=True, **kwargs):
    return dict(idx=idx, name=name, color=color, visible=visible)


@annotations.post("/upload")
def upload(file: UploadFile = File(...)):
    """Upload annotations layer as an array (via Launcher) to the current workspace.
    After unpickling there is a dictionary with a 'name' key and a
       'data' key. The 'data' key contains a numpy array of labels.
    """
    encoded_buffer = file.file.read()
    d = pickle.loads(encoded_buffer)

    anno_id = d["name"]
    level_arr = np.array(d["data"])
    logger.debug(f"Uploaded feature of shape {level_arr.shape}")
    dst = DataModel.g.dataset_uri(anno_id, group="annotations")

    with DatasetManager(dst, out=dst, dtype="uint32", fillvalue=0) as DM:
        DM.out[:] = level_arr

    modified_ds = dataset_from_uri(dst, mode="r")
    modified = [1]
    modified_ds.set_attr("modified", modified)


@annotations.get("/set_volume")
def set_volume(src: str, vol_array):
    ds = dataset_from_uri(src, mode="rw")
    if ds[:].shape == vol_array.shape:
        ds[:] = vol_array


@annotations.get("/get_volume")
def get_volume(src: str):
    ds = dataset_from_uri(src, mode="r")
    data = ds[:]
    return encode_numpy(data)


@annotations.get("/get_slice")
def get_slice(src: str, slice_idx: int, order: tuple = Query()):
    ds = dataset_from_uri(src, mode="r")[:]
    order = tuple(int(e) for e in order)
    ds = np.transpose(ds, order)
    data = ds[slice_idx]
    return encode_numpy(data)


@annotations.get("/get_crop")
def get_crop(src: str, roi: list):
    ds = dataset_from_uri(src, mode="r")
    data = ds[roi[0] : roi[1], roi[2] : roi[3], roi[4] : roi[5]]
    return encode_numpy(data)


@annotations.get("/set_label_parent")
def set_label_parent(
    workspace: str,
    level: str,
    label_idx: int,
    parent_level: str,
    parent_label_idx: int,
    parent_color: str,
):
    ds = ws.get_dataset(workspace, level, group=__group_pattern__)
    labels = ds.get_metadata("labels", {})
    logger.debug(f"Setting label parent using dataset {ds} and with labels {labels}")

    if label_idx in labels:
        labels[label_idx]["parent_level"] = parent_level
        labels[label_idx]["parent_label"] = parent_label_idx
        labels[label_idx]["parent_color"] = parent_color
    ds.set_metadata("labels", labels)


@annotations.get("/get_label_parent")
def get_label_parent(workspace: str, level: str, label_idx: int):
    ds = get_level(workspace, level)
    labels = ds.get_metadata("labels", {})
    logger.debug(
        f"get_label_parent with level {level}, label_idx {label_idx}, result labels: {labels}"
    )
    parent_level = -1
    parent_label_idx = -1
    parent_color = None
    if label_idx in labels:
        if "parent_level" in labels[label_idx]:
            parent_level = labels[label_idx]["parent_level"]
            parent_label_idx = int(labels[label_idx]["parent_label"])
            if "parent_color" in labels[label_idx]:
                parent_color = labels[label_idx]["parent_color"]

    return parent_level, parent_label_idx, parent_color


@annotations.get("/add_level")
def add_level(workspace: str):
    ds = ws.auto_create_dataset(
        workspace,
        "level",
        __group_pattern__,
        __level_dtype__,
        fill=__level_fill__,
        chunks=CHUNK_SIZE,
    )
    logger.debug(ds)
    ds.set_attr("kind", "level")
    ds.set_attr("modified", [0] * ds.total_chunks)

    return dataset_repr(ds)


@annotations.get("/get_level")
def get_level(workspace: str, level: str, full: bool = False):
    if full == False:
        return ws.get_dataset(workspace, level, group=__group_pattern__, session="default")
    return ws.get_dataset(workspace, level, session="default")


@annotations.get("/get_single_level")
def get_single_level(workspace: str, level: str):
    ds = ws.get_dataset(workspace, level, group=__group_pattern__, session="default")
    return dataset_repr(ds)


@annotations.get("/get_levels")
def get_levels(workspace: str, full: bool = False):
    datasets = ws.existing_datasets(workspace, group=__group_pattern__)
    datasets = [dataset_repr(v) for k, v in datasets.items()]

    if full:
        for ds in datasets:
            ds["id"] = "{}/{}".format(__group_pattern__, ds["id"])

    return datasets


@annotations.get("/rename_level")
def rename_level(workspace: str, level: str, name: str, full: bool = False):
    ds = get_level(workspace, level, full)
    ds.set_metadata("name", name)


@annotations.get("/delete_level")
def delete_level(workspace: str, level: str, full: bool = False):
    if full:
        ws.delete_dataset(workspace, level)
    else:
        ws.delete_dataset(workspace, level, group=__group_pattern__)
    return dict(done=True)


@annotations.get("/add_label")
def add_label(workspace: str, level: str, full: bool = False):
    from survos2.api.annotate import erase_label
    from survos2.improc.utils import map_blocks

    ds = get_level(workspace, level, full)
    labels = ds.get_metadata("labels", {})
    idx = max(1, (max(l for l in labels) if labels else 1)) + 1  # change min label to 1
    # idx = max(0, (max(l for l in labels) if labels else 0))   # change min label to 1

    if idx >= 16:
        existing_idx = set(labels.keys())
        for i in range(1, 16):
            if i not in existing_idx:
                idx = i
                break
        if idx >= 16:
            raise ValueError("Only 15 labels can be created")

    new_label = to_label(idx=idx)
    labels[idx] = new_label
    ds.set_metadata("labels", labels)

    # Erase label from dataset
    erase_label(ds, label=idx)

    return new_label


@annotations.get("/get_labels")
def get_labels(workspace: str, level: str, full: bool = False):
    ds = get_level(workspace, level, full)
    labels = ds.get_metadata("labels", {})
    return {k: to_label(**v) for k, v in labels.items()}


@annotations.get("/update_label")
def update_label(
    workspace: str,
    level: str,
    idx: int,
    name: str = None,
    color: str = None,
    visible: bool = None,
    full: bool = False,
):
    ds = get_level(workspace, level, full)
    labels = ds.get_metadata("labels", {})
    if idx in labels:
        for k, v in (("name", name), ("color", color), ("visible", visible)):
            if v is not None:
                labels[idx][k] = v
        ds.set_metadata("labels", labels)
        return to_label(**labels[idx])
    raise APIException("Label {}::{} does not exist".format(level, idx))


@annotations.get("/delete_label")
def delete_label(workspace: str, level: str, idx: int, full: bool = False):
    ds = get_level(workspace, level, full)
    labels = ds.get_metadata("labels", {})
    if idx in labels:
        del labels[idx]
        ds.set_metadata("labels", labels)
        return dict(done=True)
    raise APIException("Label {}::{} does not exist".format(level, idx))


@annotations.get("/delete_all_labels")
def delete_all_labels(workspace: str, level: str, full: bool = False):
    ds = get_level(workspace, level, full)
    ds.set_metadata("labels", {})
    return dict(done=True)


@annotations.get("/annotate_voxels")
def annotate_voxels(
    workspace: str = Body(),
    level: str = Body(),
    slice_idx: int = Body(),
    yy: list = Body(),
    xx: list = Body(),
    label: int = Body(),
    full: bool = Body(),
    parent_level: str = Body(),
    parent_label_idx: int = Body(),
    viewer_order: tuple = Body(),
):

    ds = get_level(workspace, level, full)

    from survos2.frontend.frontend import get_level_from_server

    if parent_level != "-1" and parent_level != -1:
        parent_arr, parent_annotations_dataset = get_level_from_server(
            {"level_id": parent_level}, retrieval_mode="volume"
        )

        if isinstance(parent_arr, np.ndarray):
            parent_arr = parent_arr & 15
            parent_mask = parent_arr == parent_label_idx
        else:
            parent_arr = None
            parent_mask = None
    else:
        parent_arr = None
        parent_mask = None

    _annotate_voxels(
        ds,
        slice_idx=slice_idx,
        yy=yy,
        xx=xx,
        label=label,
        parent_mask=parent_mask,
        viewer_order=viewer_order,
    )
    DataModel.g.current_workspace = workspace
    dst = DataModel.g.dataset_uri(level, group="annotations")
    modified_ds = dataset_from_uri(dst, mode="r")
    modified = [1]
    modified_ds.set_attr("modified", modified)

@annotations.get("/annotate_from_slice")
def annotate_from_slice(
    workspace: str = Body(),
    target_level: str = Body(),
    source_level: str = Body(),
    region: str = Body(),
    slice_num : int = Body(),
    viewer_order: tuple = Body(),
):
    DataModel.g.current_workspace = workspace
    target_ds = get_level(workspace, target_level, False)
    source_ds = get_level(workspace, source_level, False)

    region_uri = DataModel.g.dataset_uri(region, group="superregions")
    region_ds = dataset_from_uri(region_uri, mode="r")
    source_std_label = source_ds[:] & 15
    source_slice = source_std_label[slice_num,:]

    anno = _annotate_from_slice(target_ds,
                                region_ds, 
                                source_slice,
                                slice_num, 
                                viewer_order)
    
    dst = DataModel.g.dataset_uri(target_level, group="annotations")
    target_ds[:] = anno
    modified_ds = dataset_from_uri(dst, mode="rw")
    modified = [1]
    modified_ds.set_attr("modified", modified)

@annotations.get("/annotate_regions")
def annotate_regions(
    workspace: str = Body(),
    level: str = Body(),
    region: str = Body(),
    r: list = Body(),
    label: int = Body(),
    full: bool = Body(),
    parent_level: str = Body(),
    parent_label_idx: int = Body(),
    bb: list = Body(),
    viewer_order: tuple = Body(),
):
    DataModel.g.current_workspace = workspace
    ds = get_level(workspace, level, full)
    region = dataset_from_uri(region, mode="r")

    if parent_level != "-1" and parent_level != -1:
        parent_arr, parent_annotations_dataset = get_level_from_server(
            {"level_id": parent_level}, retrieval_mode="volume"
        )
        parent_arr = parent_arr & 15
        parent_mask = parent_arr == parent_label_idx
    else:
        parent_arr = None
        parent_mask = None

    anno = _annotate_regions(
        ds,
        region,
        r=r,
        label=label,
        parent_mask=parent_mask,
        bb=bb,
        viewer_order=viewer_order,
    )

    dst = DataModel.g.dataset_uri(level, group="annotations")
    ds[:] = anno
    modified_ds = dataset_from_uri(dst, mode="rw")
    modified = [1]
    modified_ds.set_attr("modified", modified)


@annotations.get("/annotate_undo")
def annotate_undo(workspace: str, level: str, full: bool = False):
    from survos2.api.annotate import undo_annotation

    ds = get_level(workspace, level, full)
    undo_annotation(ds)
