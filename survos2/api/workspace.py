import os.path as op
from typing import List, Union

import parse
from fastapi import APIRouter, Body, File, Query, UploadFile
from loguru import logger

from survos2.api.types import DataURI, Int, IntList, IntOrNone, String
from survos2.api.utils import APIException
from survos2.config import Config
from survos2.data_io import dataset_from_uri
from survos2.frontend.main import roi_ws
from survos2.improc.utils import DatasetManager
from survos2.model import DataModel, Dataset, Workspace

workspace = APIRouter()


@workspace.get("/create")
def create(workspace: str):
    workspace, session = parse_workspace(workspace)
    return Workspace.create(workspace)


def get_dataset(workspace: str, dataset: str, group: str = None):
    workspace, session = parse_workspace(workspace)
    if group:
        dataset = "{}/{}".format(group, dataset)
    return get(workspace).get_dataset(dataset, session=session)


@workspace.get("/get")
def get(workspace: str):
    workspace, session = parse_workspace(workspace)
    if Workspace.exists(workspace):
        return Workspace(workspace)
    else:
        return False
    #raise APIException("Workspace '%s' does not exist." % workspace)


@workspace.get("/make_roi_ws")
def make_roi_ws(feature_id: str, current_workspace_name: str, roi: List[int] = Query()):
    # get raw from current workspace
    src = DataModel.g.dataset_uri(feature_id, group="features")
    logger.debug(f"Getting features {src} and cropping roi {roi}")
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        src_dataset = DM.sources[0][:]

    # print(roi)
    # rc_dataset = src_dataset[roi[0] : roi[3], roi[1] : roi[4], roi[2] : roi[5]]

    src_dataset = src_dataset[roi[0] : roi[1], roi[2] : roi[3], roi[4] : roi[5]]

    # print(src_dataset.shape)

    # make new ws from roi crop of raw data
    roi_name = (
        DataModel.g.current_workspace
        + "_roi_"
        + str(roi[0])
        + "_"
        + str(roi[1])
        + "_"
        + str(roi[2])
        + "_"
        + str(roi[3])
        + "_"
        + str(roi[4])
        + "_"
        + str(roi[5])
    )

    roi_ws(src_dataset, roi_name)
    # src_ws = get(current_workspace_name)
    # target_ws = get(roi_name)
    # src_ws.replicate_workspace(target_ws.path)

    return roi_name


@workspace.get("/set_workspace")
def set_workspace(workspace: str):
    logger.debug(f"Setting workspace to {workspace}")
    DataModel.g.current_workspace = workspace


@workspace.get("/set_workspace_shape")
def set_workspace_shape(shape: list):
    logger.debug(f"Setting workspace shape to {shape}")
    DataModel.g.current_workspace_shape = shape
    return DataModel.g.current_workspace_shape


@workspace.get("/delete")
def delete(workspace: str):
    workspace, session = parse_workspace(workspace)
    Workspace.remove(workspace)
    return dict(done=True)


### Data


@workspace.get("/add_data")
def add_data(workspace: str, data_fname: str):
    import dask.array as da

    from survos2.improc.utils import optimal_chunksize

    ws = get(workspace)
    logger.info(f"Adding data to workspace {ws}")

    with dataset_from_uri(data_fname, mode="r") as data:
        chunk_size = optimal_chunksize(data, Config["computing.chunk_size"])
        logger.debug(
            f'Calculating optimal chunk size using chunk_size {Config["computing.chunk_size"]}: {chunk_size}'
        )

        data = da.from_array(data, chunks=chunk_size)
        data -= da.min(data)
        data /= da.max(data)
        ds = ws.add_data(data)
        # ds.set_attr("chunk_size", chunk_size)
    return ds


# @hug.local()
@workspace.get("/get_data")
def get_data(workspace: str):
    workspace, session = parse_workspace(workspace)
    data = get(workspace).get_data()
    return data.tojson()
    # return data


### Sessions


@workspace.get("/list_sessions")
def list_sessions(workspace: str):
    return get(workspace).available_sessions()


@workspace.get("/add_session")
def add_session(workspace: str, session: str):
    return get(workspace).add_session(session)


@workspace.get("/delete_session")
def delete_session(workspace: str, session: str):
    get(workspace).remove_session(session)
    return dict(done=True)


@workspace.get("/get_session")
def get_session(workspace: str, session: str):
    return get(workspace).get_session(session)


### Datasets


@workspace.get("/list_datasets")
def list_datasets(workspace: str):
    workspace, session = parse_workspace(workspace)
    return get(workspace).available_datasets(session)


@workspace.get("/add_dataset")
def add_dataset(
    workspace: str,
    dataset_name: str,
    dtype: str,
    fillvalue: int = 0,
    group: str = None,
    chunks: Union[int, None] = None,
):
    workspace, session = parse_workspace(workspace)
    if group:
        dataset_name = "{}/{}".format(group, dataset_name)
    return get(workspace).add_dataset(
        dataset_name, dtype, session=session, fillvalue=fillvalue, chunks=chunks
    )


# @hug.local()
@workspace.get("/delete_dataset")
def delete_dataset(workspace: str, dataset: str, group: str = None):
    workspace, session = parse_workspace(workspace)
    if group:
        dataset = "{}/{}".format(group, dataset)
    get(workspace).remove_dataset(dataset, session=session)
    return dict(done=True)


# @hug.local()
@workspace.get("/get_dataset")
def get_dataset(
    workspace: str,
    dataset: str,
    session: str,
    group: str = None,
):
    workspace, session = parse_workspace(workspace)
    if group:
        dataset = "{}/{}".format(group, dataset)
    return get(workspace).get_dataset(dataset, session=session)


def existing_datasets(workspace: str, group: str = None, filter: str = None):
    ws, session = parse_workspace(workspace)
    ws = get(ws)
    all_ds = sorted(ws.available_datasets(session=session, group=group))
    result = {}
    for dsname in all_ds:
        if filter is None or filter in dsname:
            ds = ws.get_dataset(dsname, session=session)
            result[dsname.split(op.sep)[1]] = ds
    return result


@workspace.get("/metadata")
def metadata(workspace: str = Body(), dataset: str = Body()):
    workspace, session = parse_workspace(workspace)
    if dataset:
        ds = get_dataset(workspace, dataset, session=session)
    else:
        ds = get_data(workspace)
    return ds.get_metadata()


def auto_create_dataset(
    workspace: str,
    name: str,
    group: str,
    dtype: str,
    fill: int = 0,
    chunks: Union[int, None] = None,
):
    all_ds = existing_datasets(workspace, group)
    max_idx = 0
    pattern = "{:03d}_{}"
    for dsname in all_ds:
        idx = parse.parse(pattern, dsname)
        if idx:
            max_idx = max(max_idx, idx[0])
    dataset_id = pattern.format(max_idx + 1, name)
    dataset_name = dataset_id.replace("_", " ").title()
    dataset_file = "{}/{}".format(group, dataset_id)
    ds = add_dataset(workspace, dataset_file, dtype, fillvalue=fill, chunks=chunks)
    ds.set_attr("name", dataset_name)
    return ds


def rename_dataset(workspace: str, feature_id: str, group: str, new_name: str):
    ds = get_dataset(workspace, feature_id, group=group, session="default")
    ds.set_attr("name", new_name)


def parse_workspace(workspace: str):
    if "@" in workspace:
        session, workspace = workspace.split("@")
        # logger.debug(f"Using session {session}")
        return workspace, session
    else:
        return workspace, "default"
