import os.path as op

import hug
import numpy as np
import parse
from loguru import logger

from survos2.api import workspace as ws
from survos2.api.types import DataURI, Float, Int, IntList, SmartBoolean, String
from survos2.api.utils import APIException, dataset_repr
from survos2.config import Config
from survos2.improc import map_blocks
from survos2.io import dataset_from_uri
from survos2.utils import encode_numpy
from survos2.model import DataModel
from survos2.improc.utils import DatasetManager

__level_fill__ = 0
__level_dtype__ = "uint32"
__group_pattern__ = "annotations"


CHUNK_SIZE = Config["computing.chunk_size_sparse"]


def to_label(idx=0, name="Label", color="#000000", visible=True, **kwargs):
    return dict(idx=idx, name=name, color=color, visible=visible)

@hug.post()
def upload(body, request, response):
    print(f"Request: {request}")
    print(f"Response: {response}")
    
    encoded_array = body['file']
    array_shape = body['shape']
    anno_id = body['name']
    print(f"shape {array_shape} name {anno_id}")
    
    level_arr = np.frombuffer(encoded_array, dtype="uint32")
    
    print(f"level_arr: {level_arr.shape}")
    from ast import literal_eval 
    level_arr.shape = literal_eval(array_shape)
    print(f"Uploaded feature of shape {level_arr.shape}")

    dst = DataModel.g.dataset_uri(anno_id, group="annotations")

    with DatasetManager(dst, out=dst, dtype="uint32", fillvalue=0) as DM:
        DM.out[:] = level_arr


    modified_ds = dataset_from_uri(dst, mode="r")    
    modified = [1]
    modified_ds.set_attr("modified", modified)

            
    
@hug.get()
def get_volume(src: DataURI):
    logger.debug("Getting annotation volume")
    ds = dataset_from_uri(src, mode="r")
    data = ds[:]
    return encode_numpy(data)


@hug.get()
def get_slice(src: DataURI, slice_idx: Int, order: tuple):
    ds = dataset_from_uri(src, mode="r")[:]
    ds = np.transpose(ds, order)
    data = ds[slice_idx]
    return encode_numpy(data)


@hug.get()
def get_crop(src: DataURI, roi: IntList):
    logger.debug("Getting anno crop")
    ds = dataset_from_uri(src, mode="r")
    data = ds[roi[0] : roi[1], roi[2] : roi[3], roi[4] : roi[5]]
    return encode_numpy(data)


@hug.get()
def set_label_parent(
    workspace: String,
    level: String,
    label_idx: Int,
    parent_level: String,
    parent_label_idx: Int,
    parent_color: String,
):
    ds = ws.get_dataset(workspace, level, group=__group_pattern__)
    labels = ds.get_metadata("labels", {})
    logger.debug(f"Setting label parent using dataset {ds} and with labels {labels}")

    if label_idx in labels:
        labels[label_idx]["parent_level"] = parent_level
        labels[label_idx]["parent_label"] = parent_label_idx
        labels[label_idx]["parent_color"] = parent_color
    ds.set_metadata("labels", labels)


@hug.get()
def get_label_parent(workspace: String, level: String, label_idx: Int):
    ds = get_level(workspace, level)
    labels = ds.get_metadata("labels", {})
    print(
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


@hug.get()
def add_level(workspace: String):
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


@hug.local()
def get_level(workspace: String, level: String, full: SmartBoolean = False):
    if full == False:
        return ws.get_dataset(workspace, level, group=__group_pattern__)
    return ws.get_dataset(workspace, level)

@hug.get()
@hug.local()
def get_single_level(workspace: String, level: String):
    ds = ws.get_dataset(workspace, level, group=__group_pattern__)
    return dataset_repr(ds)


@hug.get()
@hug.local()
def get_levels(workspace: String, full: SmartBoolean = False):
    datasets = ws.existing_datasets(workspace, group=__group_pattern__)
    datasets = [dataset_repr(v) for k, v in datasets.items()]

    # TODO: unreached
    if full:
        for ds in datasets:
            ds["id"] = "{}/{}".format(__group_pattern__, ds["id"])

    return datasets


@hug.get()
def rename_level(
    workspace: String, level: String, name: String, full: SmartBoolean = False
):
    ds = get_level(workspace, level, full)
    ds.set_metadata("name", name)


@hug.get()
def delete_level(workspace: String, level: String, full: SmartBoolean = False):
    if full:
        ws.delete_dataset(workspace, level)
    else:
        ws.delete_dataset(workspace, level, group=__group_pattern__)


@hug.get()
def add_label(workspace: String, level: String, full: SmartBoolean = False):
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


@hug.get()
def get_labels(workspace: String, level: String, full: SmartBoolean = False):
    ds = get_level(workspace, level, full)
    labels = ds.get_metadata("labels", {})
    return {k: to_label(**v) for k, v in labels.items()}


@hug.get()
def update_label(
    workspace: String,
    level: String,
    idx: Int,
    name: String = None,
    color: String = None,
    visible: SmartBoolean = None,
    full: SmartBoolean = False,
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


@hug.get()
def delete_label(
    workspace: String, level: String, idx: Int, full: SmartBoolean = False
):
    ds = get_level(workspace, level, full)
    labels = ds.get_metadata("labels", {})
    if idx in labels:
        del labels[idx]
        ds.set_metadata("labels", labels)
        return dict(done=True)
    raise APIException("Label {}::{} does not exist".format(level, idx))

@hug.get()
def delete_all_labels(
    workspace: String, level: String, full: SmartBoolean = False
):
    ds = get_level(workspace, level, full)
    ds.set_metadata("labels", {})
    return dict(done=True)

@hug.get()
def annotate_voxels(
    workspace: String,
    level: String,
    slice_idx: Int,
    yy: IntList,
    xx: IntList,
    label: Int,
    full: SmartBoolean,
    parent_level: String,
    parent_label_idx: Int,
    viewer_order: tuple,
    three_dim: SmartBoolean,
    brush_size: Int,
    centre_point: tuple
):
    from survos2.api.annotate import annotate_voxels

    ds = get_level(workspace, level, full)

    from survos2.frontend.frontend import get_level_from_server

    if parent_level != "-1" and parent_level != -1:
        parent_arr, parent_annotations_dataset = get_level_from_server(
            {"level_id": parent_level}, retrieval_mode="volume"
        )
        parent_arr = parent_arr & 15
        parent_mask = parent_arr == parent_label_idx
    else:
        parent_arr = None
        parent_mask = None

    logger.info(f"slice_idx {slice_idx} Viewer order: {viewer_order}")
    annotate_voxels(
        ds,
        slice_idx=slice_idx,
        yy=yy,
        xx=xx,
        label=label,
        parent_mask=parent_mask,
        viewer_order=viewer_order,
        three_dim = three_dim,
        brush_size=brush_size,
        centre_point=centre_point
    )

    dst = DataModel.g.dataset_uri(level, group="annotations")
    modified_ds = dataset_from_uri(dst, mode="r")
    modified = [1]
    modified_ds.set_attr("modified", modified)


@hug.get()
def annotate_regions(
    workspace: String,
    level: String,
    region: DataURI,
    r: IntList,
    label: Int,
    full: SmartBoolean,
    parent_level: String,
    parent_label_idx: Int,
    bb: IntList,
    viewer_order: tuple,
):
    from survos2.api.annotate import annotate_regions

    ds = get_level(workspace, level, full)
    region = dataset_from_uri(region, mode="r")

    from survos2.frontend.frontend import get_level_from_server

    if parent_level != "-1" and parent_level != -1:
        parent_arr, parent_annotations_dataset = get_level_from_server(
            {"level_id": parent_level}, retrieval_mode="volume"
        )
        parent_arr = parent_arr & 15
        # print(f"Using parent dataset for masking {parent_annotations_dataset}")
        parent_mask = parent_arr == parent_label_idx
    else:
        # print("Not masking using parent level")
        parent_arr = None
        parent_mask = None

    logger.debug(f"BB in annotate_regions {bb}")
    anno = annotate_regions(
        ds,
        region,
        r=r,
        label=label,
        parent_mask=parent_mask,
        bb=bb,
        viewer_order=viewer_order,
    )

    def pass_through(x):
        return x

    dst = DataModel.g.dataset_uri(level, group="annotations")
    map_blocks(pass_through, anno, out=dst, normalize=False)
    modified_ds = dataset_from_uri(dst, mode="r")
    modified = [1]
    modified_ds.set_attr("modified", modified)


@hug.get()
def annotate_undo(workspace: String, level: String, full: SmartBoolean = False):
    from survos2.api.annotate import undo_annotation
    ds = get_level(workspace, level, full)
    undo_annotation(ds)
