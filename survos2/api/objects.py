import logging
import os.path as op
from ast import literal_eval 
import os
import dask.array as da
import hug
import numpy as np
from loguru import logger
import tempfile

from survos2.api import workspace as ws
from survos2.api.types import (
    DataURI,
    DataURIList,
    Float,
    FloatList,
    FloatOrVector,
    Int,
    IntList,
    SmartBoolean,
    String,
)
from survos2.api.utils import dataset_repr, get_function_api, save_metadata
from survos2.improc import map_blocks
from survos2.improc.utils import DatasetManager
from survos2.io import dataset_from_uri
from survos2.model import DataModel
from survos2.utils import encode_numpy
from survos2.frontend.components.entity import setup_entity_table
from survos2.entity.entities import load_entities_via_file, make_entity_df

__objects_fill__ = 0
__objects_dtype__ = "uint32"
__objects_group__ = "objects"
__objects_names__ = ["points", "boxes", "patches"]



def load_entities(entities_arr, flipxy=True):
    entities_df = make_entity_df(entities_arr, flipxy=flipxy)
    tmp_fullpath = os.path.abspath(
        os.path.join(tempfile.gettempdir(), os.urandom(24).hex() + ".csv")
    )
    print(entities_df)
    print(f"Creating temp file: {tmp_fullpath}")
    entities_df.to_csv(tmp_fullpath, line_terminator="")

    object_scale = 1.0
    object_offset = (0.0, 0.0, 0.0)
    object_crop_start = (0.0, 0.0, 0.0)
    object_crop_end = (1e9, 1e9, 1e9)

    objects_type = __objects_names__[0]
    ds = ws.auto_create_dataset(
        DataModel.g.current_session + "@" + DataModel.g.current_workspace,
        objects_type,
        __objects_group__,
        __objects_dtype__,
        fill=__objects_fill__,
    )

    ds.set_attr("kind", objects_type)
    ds.set_attr("fullname", tmp_fullpath)
    
    src = DataModel.g.dataset_uri("__data__")
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        src_dataset = DM.sources[0]
        img_volume = src_dataset[:]
        logger.info(f"Got __data__ volume of size {img_volume.shape}")

    ds[:] = np.zeros_like(img_volume)
    ds.set_attr("scale", object_scale)
    ds.set_attr("offset", list(object_offset))
    ds.set_attr("crop_start", list(object_crop_start))
    ds.set_attr("crop_end", list(object_crop_end))

    csv_saved_fullname = ds.save_file(tmp_fullpath)
    logger.info(f"Saving {tmp_fullpath} to {csv_saved_fullname}")
    ds.set_attr("fullname", csv_saved_fullname)
    os.remove(tmp_fullpath)

@hug.post()
def upload(body, request, response):
    print(f"Request: {request}")
    print(f"Response: {response}")
    encoded_buffer = body['file']
    array_shape = body['shape']
    print(f"shape {array_shape}")
    entities_arr = np.frombuffer(encoded_buffer, dtype="float32")
    print(entities_arr)
    entities_arr.shape = literal_eval(array_shape)
    print(f"Uploaded array of entities of shape {entities_arr.shape}")
    load_entities(entities_arr)

@hug.get()
def get_entities(src: DataURI):
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        ds_objects = DM.sources[0]
        logger.debug(f"Using dataset {ds_objects}")
        
        objects_fullname = ds_objects.get_metadata("fullname")
        objects_scale = ds_objects.get_metadata("scale")
        objects_offset = ds_objects.get_metadata("offset")
        objects_crop_start = ds_objects.get_metadata("crop_start")
        objects_crop_end = ds_objects.get_metadata("crop_end")

    logger.info(f"Setting up entities {objects_fullname}")
    tabledata, entities_df = setup_entity_table(
        objects_fullname,
        entities_df=None,
        scale=objects_scale,
        offset=objects_offset,
        crop_start=objects_crop_start,
        crop_end=objects_crop_end,
        flipxy=False
    )
    return encode_numpy(np.array(entities_df))

@hug.get()
def get_entities_metadata(src: DataURI):
    ds = dataset_from_uri(src, mode="r")[:]
    
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        ds_objects = DM.sources[0]
        logger.debug(f"Using dataset {ds_objects}")

    entities_metadata = {'fullname' : ds_objects.get_metadata("fullname"),
                         'scale' : ds_objects.get_metadata("scale"),
                         'offset' : ds_objects.get_metadata("offset"),
                         'crop_start' : ds_objects.get_metadata("crop_start"),
                         'crop_end' : ds_objects.get_metadata("crop_end")}

    return entities_metadata

@hug.get()
def points(
    dst: DataURI,
    fullname: String,
    scale: float,
    offset: FloatOrVector,
    crop_start: FloatOrVector,
    crop_end: FloatOrVector,
) -> "GEOMETRY":
    src = DataModel.g.dataset_uri("__data__")
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        src_dataset = DM.sources[0]
        img_volume = src_dataset[:]
        logger.info(f"Got __data__ volume of size {img_volume.shape}")
    # store in dst
    logger.info(f"Storing in dataset {dst}")

    with DatasetManager(dst, out=dst, dtype="float32", fillvalue=0) as DM:
        DM.out[:] = np.zeros_like(img_volume)
        dst_dataset = DM.sources[0]
        dst_dataset.set_attr("scale", scale)
        dst_dataset.set_attr("offset", offset)
        dst_dataset.set_attr("crop_start", crop_start)
        dst_dataset.set_attr("crop_end", crop_end)

        csv_saved_fullname = dst_dataset.save_file(fullname)
        logger.info(f"Saving {fullname} to {csv_saved_fullname}")
        dst_dataset.set_attr("fullname", csv_saved_fullname)


@hug.get()
def boxes(
    dst: DataURI,
    fullname: String,
    scale: float,
    offset: FloatOrVector,
    crop_start: FloatOrVector,
    crop_end: FloatOrVector,
) -> "GEOMETRY":
    src = DataModel.g.dataset_uri("__data__")
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        src_dataset = DM.sources[0]
        img_volume = src_dataset[:]
        logger.info(f"Got __data__ volume of size {img_volume.shape}")

    # store in dst
    logger.info(f"Storing in dataset {dst}")

    with DatasetManager(dst, out=dst, dtype="float32", fillvalue=0) as DM:
        DM.out[:] = np.zeros_like(img_volume)
        dst_dataset = DM.sources[0]
        dst_dataset.set_attr("scale", scale)
        dst_dataset.set_attr("offset", offset)
        dst_dataset.set_attr("crop_start", crop_start)
        dst_dataset.set_attr("crop_end", crop_end)

        csv_saved_fullname = dst_dataset.save_file(fullname)
        logger.info(f"Saving {fullname} to {csv_saved_fullname}")
        dst_dataset.set_attr("fullname", csv_saved_fullname)


@hug.get()
def patches(
    dst: DataURI,
    fullname: String,
    scale: float,
    offset: FloatOrVector,
    crop_start: FloatOrVector,
    crop_end: FloatOrVector,
) -> "GEOMETRY":
    src = DataModel.g.dataset_uri("__data__")
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        src_dataset = DM.sources[0]
        img_volume = src_dataset[:]
        logger.info(f"Got __data__ volume of size {img_volume.shape}")
    # store in dst
    logger.info(f"Storing in dataset {dst}")

    with DatasetManager(dst, out=dst, dtype="float32", fillvalue=0) as DM:
        DM.out[:] = np.zeros_like(img_volume)
        dst_dataset = DM.sources[0]
        dst_dataset.set_attr("scale", scale)
        dst_dataset.set_attr("offset", offset)
        dst_dataset.set_attr("crop_start", crop_start)
        dst_dataset.set_attr("crop_end", crop_end)

        csv_saved_fullname = dst_dataset.save_file(fullname)
        logger.info(f"Saving {fullname} to {csv_saved_fullname}")
        dst_dataset.set_attr("fullname", csv_saved_fullname)

@hug.get()
def update_metadata(dst: DataURI,fullname: String,
    scale: float,
    offset: FloatOrVector,
    crop_start: FloatOrVector,
    crop_end: FloatOrVector):
    with DatasetManager(dst, out=dst, dtype="float32", fillvalue=0) as DM:
        dst_dataset = DM.sources[0]
        dst_dataset.set_attr("scale", scale)
        dst_dataset.set_attr("offset", offset)
        dst_dataset.set_attr("crop_start", crop_start)
        dst_dataset.set_attr("crop_end", crop_end)

        csv_saved_fullname = dst_dataset.save_file(fullname)
        logger.info(f"Saving {fullname} to {csv_saved_fullname}")
        dst_dataset.set_attr("fullname", csv_saved_fullname)



@hug.get()
def create(workspace: String, fullname: String, order: Int = 0):
    objects_type = __objects_names__[order]
    ds = ws.auto_create_dataset(
        workspace,
        objects_type,
        __objects_group__,
        __objects_dtype__,
        fill=__objects_fill__,
    )

    ds.set_attr("kind", objects_type)
    ds.set_attr("fullname", fullname)
    return dataset_repr(ds)


@hug.get()
@hug.local()
def existing(
    workspace: String,
    full: SmartBoolean = False,
    filter: SmartBoolean = True,
    order: Int = 0,
):
    datasets = ws.existing_datasets(workspace, group=__objects_group__)

    if full:
        datasets = {
            "{}/{}".format(__objects_group__, k): dataset_repr(v)
            for k, v in datasets.items()
        }
    else:
        datasets = {k: dataset_repr(v) for k, v in datasets.items()}

    if filter:
        datasets = {k: v for k, v in datasets.items() if v["kind"] != "unknown"}

    return datasets


@hug.get()
def remove(workspace: String, objects_id: String):
    ws.delete_dataset(workspace, objects_id, group=__objects_group__)


@hug.get()
def rename(workspace: String, objects_id: String, new_name: String):
    ws.rename_dataset(workspace, objects_id, __objects_group__, new_name)


@hug.get()
def available():
    h = hug.API(__name__)
    all_features = []
    for name, method in h.http.routes[""].items():
        if name[1:] in ["available", "create", "existing", "remove", "rename", "group", "upload", "get_entities", "get_entities_metadata", "update_metadata"]:
            continue
        logger.debug(f"Object types available {name}")
        name = name[1:]
        func = method["GET"][None].interface.spec
        desc = get_function_api(func)
        category = desc["returns"] or "Others"
        desc = dict(name=name, params=desc["params"], category=category)
        all_features.append(desc)
    return all_features
