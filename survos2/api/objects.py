import pickle
import os
import numpy as np
from loguru import logger
import tempfile

from fastapi import APIRouter, Query, Body, File, UploadFile, Form, Depends

from survos2.api import workspace as ws

from survos2.api.utils import dataset_repr, get_function_api, save_metadata
from survos2.improc.utils import DatasetManager
from survos2.data_io import dataset_from_uri
from survos2.model import DataModel
from survos2.utils import encode_numpy
from survos2.frontend.components.entity import setup_entity_table
from survos2.entity.entities import load_entities_via_file, make_entity_df, make_entity_bvol


__objects_fill__ = 0
__objects_dtype__ = "uint32"
__objects_group__ = "objects"
__objects_names__ = ["points", "boxes", "patches"]


objects = APIRouter()


def load_bvols(bvols_arr, flipxy=True):
    entities_df = make_entity_bvol(bvols_arr, flipxy=flipxy)
    tmp_fullpath = os.path.abspath(
        os.path.join(tempfile.gettempdir(), os.urandom(24).hex() + ".csv")
    )
    logger.debug(entities_df)
    logger.debug(f"Creating temp file: {tmp_fullpath}")
    entities_df.to_csv(tmp_fullpath, line_terminator="")

    object_scale = 1.0
    object_offset = (0.0, 0.0, 0.0)
    object_crop_start = (0.0, 0.0, 0.0)
    object_crop_end = (1e9, 1e9, 1e9)

    objects_type = __objects_names__[1]
    ds = ws.auto_create_dataset(
        DataModel.g.current_session + "@" + DataModel.g.current_workspace,
        objects_type,
        __objects_group__,
        __objects_dtype__,
        fill=__objects_fill__,
    )

    ds.set_attr("kind", objects_type)
    ds.set_attr("fullname", tmp_fullpath)
    ds.set_attr("scale", object_scale)
    ds.set_attr("offset", list(object_offset))
    ds.set_attr("crop_start", list(object_crop_start))
    ds.set_attr("crop_end", list(object_crop_end))

    csv_saved_fullname = ds.save_file(tmp_fullpath)
    logger.debug(f"Saving {tmp_fullpath} to {csv_saved_fullname}")
    ds.set_attr("fullname", csv_saved_fullname)
    os.remove(tmp_fullpath)


def load_entities(entities_arr, flipxy=True):
    ws_object = ws.get(DataModel.g.current_workspace)
    tmp_fullpath = os.path.join(ws_object.path, os.urandom(24).hex() + ".csv")

    entities_df = make_entity_df(entities_arr, flipxy=flipxy)

    #    tmp_fullpath = os.path.abspath(os.path.join(tempfile.gettempdir(), os.urandom(24).hex() + ".csv"))
    logger.debug(entities_df)
    logger.debug(f"Creating temp file: {tmp_fullpath}")
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
    ds.set_attr("scale", object_scale)
    ds.set_attr("offset", list(object_offset))
    ds.set_attr("crop_start", list(object_crop_start))
    ds.set_attr("crop_end", list(object_crop_end))

    csv_saved_fullname = ds.save_file(tmp_fullpath)
    logger.debug(f"Saving {tmp_fullpath} to {csv_saved_fullname}")
    basename = os.path.basename(csv_saved_fullname)
    ds.set_attr("fullname", basename)
    os.remove(tmp_fullpath)


@objects.post("/upload")
def upload(file: UploadFile = File(...)):
    """Upload a set of entities (from Launcher) to the current workspace
    After unpickling there is a dictionary with a 'name' key and a
    'data' key. The 'data' key contains a numpy array of entities.
    """
    encoded_buffer = file.file.read()
    d = pickle.loads(encoded_buffer)
    entities_arr = np.array(d["data"])
    load_entities(entities_arr)


@objects.get("/get_entities")
def get_entities(src: str, basename: bool = True):
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        ds_objects = DM.sources[0]
        logger.debug(f"Using dataset {ds_objects}")
        objects_name = ds_objects.get_metadata("fullname")
        objects_scale = ds_objects.get_metadata("scale")
        objects_offset = ds_objects.get_metadata("offset")
        objects_crop_start = ds_objects.get_metadata("crop_start")
        objects_crop_end = ds_objects.get_metadata("crop_end")

    if basename:
        fname = os.path.basename(objects_name)
        objects_dataset_fullpath = ds_objects._path
        objects_fullname = os.path.join(objects_dataset_fullpath, fname)
    else:
        objects_fullname = objects_name
    logger.debug(f"Setting up entities {objects_fullname}")

    tabledata, entities_df = setup_entity_table(
        objects_fullname,
        entities_df=None,
        scale=objects_scale,
        offset=objects_offset,
        crop_start=objects_crop_start,
        crop_end=objects_crop_end,
        flipxy=False,
    )
    return encode_numpy(np.array(entities_df))


@objects.get("/get_entities_metadata")
def get_entities_metadata(src: str):
    ds = dataset_from_uri(src, mode="r")[:]

    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        ds_objects = DM.sources[0]
        logger.debug(f"Using dataset {ds_objects}")

    entities_metadata = {
        "fullname": ds_objects.get_metadata("fullname"),
        "scale": ds_objects.get_metadata("scale"),
        "offset": ds_objects.get_metadata("offset"),
        "crop_start": ds_objects.get_metadata("crop_start"),
        "crop_end": ds_objects.get_metadata("crop_end"),
    }

    return entities_metadata


@objects.get("/points")
def points(
    dst: str,
    fullname: str,
    scale: float,
) -> "GEOMETRY":
    with DatasetManager(dst, out=dst, dtype="float32", fillvalue=0) as DM:
        # DM.out[:] = np.zeros_like(img_volume)
        dst_dataset = DM.sources[0]
        dst_dataset.set_attr("scale", scale)

        offset = [0, 0, 0]
        crop_start = [0, 0, 0]
        crop_end = [1e9, 1e9, 1e9]

        dst_dataset.set_attr("offset", offset)
        dst_dataset.set_attr("crop_start", crop_start)
        dst_dataset.set_attr("crop_end", crop_end)

        basename = os.path.basename(fullname)
        # csv_saved_fullname = dst_dataset.save_file(fullname)
        csv_saved_fullname = dst_dataset.save_file(fullname)
        logger.debug(f"Saving {fullname} to {csv_saved_fullname}")
        dst_dataset.set_attr("fullname", csv_saved_fullname)

        return dataset_repr(dst_dataset)


@objects.get("/boxes")
def boxes(
    dst: str,
    fullname: str,
    scale: float,
    offset: list,
    crop_start: list,
    crop_end: list,
) -> "GEOMETRY":
    with DatasetManager(dst, out=dst, dtype="float32", fillvalue=0) as DM:
        dst_dataset = DM.sources[0]
        dst_dataset.set_attr("scale", scale)
        dst_dataset.set_attr("offset", offset)
        dst_dataset.set_attr("crop_start", crop_start)
        dst_dataset.set_attr("crop_end", crop_end)

        csv_saved_fullname = dst_dataset.save_file(fullname)
        logger.debug(f"Saving {fullname} to {csv_saved_fullname}")
        dst_dataset.set_attr("fullname", csv_saved_fullname)


@objects.get("/patches")
def patches(
    dst: str,
    fullname: str,
    scale: float,
    offset: list,
    crop_start: list,
    crop_end: list,
) -> "GEOMETRY":
    with DatasetManager(dst, out=dst, dtype="float32", fillvalue=0) as DM:
        dst_dataset = DM.sources[0]
        dst_dataset.set_attr("scale", scale)
        dst_dataset.set_attr("offset", offset)
        dst_dataset.set_attr("crop_start", crop_start)
        dst_dataset.set_attr("crop_end", crop_end)

        csv_saved_fullname = dst_dataset.save_file(fullname)
        logger.debug(f"Saving {fullname} to {csv_saved_fullname}")
        dst_dataset.set_attr("fullname", csv_saved_fullname)


@objects.get("/import_entities")
def import_entities(dst: str, fullname: str):
    with DatasetManager(dst, out=dst, dtype="float32", fillvalue=0) as DM:
        dst_dataset = DM.sources[0]
        scale = 1.0
        offset = [0, 0, 0]
        crop_start = [0, 0, 0]
        crop_end = [1e9, 1e9, 1e9]

        dst_dataset.set_attr("scale", scale)
        dst_dataset.set_attr("offset", offset)
        dst_dataset.set_attr("crop_start", crop_start)
        dst_dataset.set_attr("crop_end", crop_end)

        csv_saved_fullname = dst_dataset.save_file(fullname)
        logger.debug(f"Saving {fullname} to {csv_saved_fullname}")
        dst_dataset.set_attr("fullname", csv_saved_fullname)

        return csv_saved_fullname


@objects.get("/update_metadata")
def update_metadata(
    dst: str,
    fullname: str,
    scale: float,
    # offset: list,
    # crop_start: list,
    # crop_end: list,
):
    with DatasetManager(dst, out=dst, dtype="float32", fillvalue=0) as DM:
        dst_dataset = DM.sources[0]
        dst_dataset.set_attr("scale", scale)

        offset = [0, 0, 0]
        crop_start = [0, 0, 0]
        crop_end = [1e9, 1e9, 1e9]

        dst_dataset.set_attr("offset", offset)
        dst_dataset.set_attr("crop_start", crop_start)
        dst_dataset.set_attr("crop_end", crop_end)

        csv_saved_fullname = dst_dataset.save_file(fullname)
        logger.debug(f"Saving {fullname} to {csv_saved_fullname}")
        dst_dataset.set_attr("fullname", csv_saved_fullname)


@objects.get("/create")
def create(workspace: str, fullname: str, order: int = 0):
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
    ds.set_attr("basename", os.path.basename(fullname))
    return dataset_repr(ds)


# @hug.local()
@objects.get("/existing")
def existing(
    workspace: str,
    full: bool = False,
    filter: bool = True,
    order: int = 0,
):
    datasets = ws.existing_datasets(workspace, group=__objects_group__)

    if full:
        datasets = {
            "{}/{}".format(__objects_group__, k): dataset_repr(v) for k, v in datasets.items()
        }
    else:
        datasets = {k: dataset_repr(v) for k, v in datasets.items()}

    if filter:
        datasets = {k: v for k, v in datasets.items() if v["kind"] != "unknown"}

    return datasets


@objects.get("/remove")
def remove(workspace: str, objects_id: str):
    ws.delete_dataset(workspace, objects_id, group=__objects_group__)
    return {"done": "ok"}


@objects.get("/rename")
def rename(workspace: str, objects_id: str, new_name: str):
    ws.rename_dataset(workspace, objects_id, __objects_group__, new_name)
    return {"done": "ok"}


@objects.get("/available")
def available():
    h = objects  # hug.API(__name__)
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
            "upload",
            "get_entities",
            "get_entities_metadata",
            "import_entities",
            "update_metadata",
        ]:
            continue
        func = r.endpoint
        desc = get_function_api(func)
        category = desc["returns"] or "Others"
        desc = dict(name=name, params=desc["params"], category=category)
        all_features.append(desc)
    return all_features
