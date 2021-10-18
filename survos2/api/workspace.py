import os.path as op

import hug
import parse
from loguru import logger

from survos2.api.types import Int, IntOrNone, String, IntList, DataURI
from survos2.api.utils import APIException
from survos2.config import Config
from survos2.io import dataset_from_uri
from survos2.model import Dataset, Workspace
from survos2.model import DataModel
from survos2.improc.utils import DatasetManager
from survos2.frontend.main import roi_ws


@hug.get()
def make_roi_ws(feature_id: DataURI, roi: IntList, current_workspace_name: String):
    # get raw from current workspace
    src = DataModel.g.dataset_uri(feature_id, group="features")
    logger.debug(f"Getting features {src} and cropping roi {roi}")
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        src_dataset = DM.sources[0][:]

    src_dataset = src_dataset[roi[0] : roi[3], roi[1] : roi[4], roi[2] : roi[5]]

    # make new ws from roi crop of raw data
    roi_name = (
        DataModel.g.current_workspace
        + "_roi_"
        + str(roi[0])
        + "_"
        + str(roi[3])
        + "_"
        + str(roi[1])
        + "_"
        + str(roi[4])
        + "_"
        + str(roi[2])
        + "_"
        + str(roi[5])
    )
    roi_ws(src_dataset, roi_name)
    #src_ws = get(current_workspace_name)
    #target_ws = get(roi_name)
    # src_ws.replicate_workspace(target_ws.path)

    return roi_name


@hug.get()
def set_workspace(workspace: String):
    logger.debug(f"Setting workspace to {workspace}")
    DataModel.g.current_workspace = workspace

@hug.get()
def set_workspace_shape(shape: IntList):
    logger.debug(f"Setting workspace shape to {shape}")
    DataModel.g.current_workspace_shape = shape
    return DataModel.g.current_workspace_shape

@hug.get()
def create(workspace: String):
    workspace, session = parse_workspace(workspace)
    return Workspace.create(workspace)

@hug.get()
def delete(workspace: String):
    workspace, session = parse_workspace(workspace)
    Workspace.remove(workspace)
    return dict(done=True)


@hug.get()
@hug.local()
def get(workspace: String):
    workspace, session = parse_workspace(workspace)
    if Workspace.exists(workspace):
        return Workspace(workspace)
    raise APIException("Workspace '%s' does not exist." % workspace)


### Data
@hug.get()
def add_data(workspace: String, data_fname: String):
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


@hug.get()
@hug.local()
def get_data(workspace: String):
    workspace, session = parse_workspace(workspace)
    return get(workspace).get_data()


### Sessions


@hug.get()
def list_sessions(workspace: String):
    return get(workspace).available_sessions()


@hug.get()
def add_session(workspace: String, session: String):
    return get(workspace).add_session(session)


@hug.get()
def delete_session(workspace: String, session: String):
    get(workspace).remove_session(session)
    return dict(done=True)


@hug.get()
def get_session(workspace: String, session: String):
    return get(workspace).get_session(session)


### Datasets


@hug.get()
def list_datasets(workspace: String):
    workspace, session = parse_workspace(workspace)
    return get(workspace).available_datasets(session)


@hug.get()
def add_dataset(
    workspace: String,
    dataset_name: String,
    dtype: String,
    fillvalue: Int = 0,
    group: String = None,
    chunks: IntOrNone = None,
):
    workspace, session = parse_workspace(workspace)
    if group:
        dataset_name = "{}/{}".format(group, dataset_name)
    return get(workspace).add_dataset(
        dataset_name, dtype, session=session, fillvalue=fillvalue, chunks=chunks
    )


@hug.get()
@hug.local()
def delete_dataset(workspace: String, dataset: String, group: String = None):
    workspace, session = parse_workspace(workspace)
    if group:
        dataset = "{}/{}".format(group, dataset)
    get(workspace).remove_dataset(dataset, session=session)
    return dict(done=True)


@hug.get()
@hug.local()
def get_dataset(workspace: String, dataset: String, group: String = None):
    workspace, session = parse_workspace(workspace)
    if group:
        dataset = "{}/{}".format(group, dataset)
    return get(workspace).get_dataset(dataset, session=session)


### Get Metadata


@hug.get()
@hug.local()
def metadata(workspace: String, dataset: String = None):
    workspace, session = parse_workspace(workspace)
    if dataset:
        ds = get_dataset(workspace, dataset, session=session)
    else:
        ds = get_data(workspace)
    return ds.get_metadata()


### Local utils for plugins

# @hug.get()
@hug.local()
def existing_datasets(workspace: String, group: String = None, filter: String = None):
    ws, session = parse_workspace(workspace)
    ws = get(ws)
    all_ds = sorted(ws.available_datasets(session=session, group=group))
    result = {}
    for dsname in all_ds:
        if filter is None or filter in dsname:
            ds = ws.get_dataset(dsname, session=session)
            result[dsname.split(op.sep)[1]] = ds
    return result


@hug.local()
def auto_create_dataset(
    workspace: String,
    name: String,
    group: String,
    dtype: String,
    fill: Int = 0,
    chunks: IntOrNone = None,
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


@hug.local()
def rename_dataset(
    workspace: String, feature_id: String, group: String, new_name: String
):
    ds = get_dataset(workspace, feature_id, group=group)
    ds.set_attr("name", new_name)


@hug.local()
def parse_workspace(workspace: String):
    if "@" in workspace:
        session, workspace = workspace.split("@")
        logger.debug(f"Using session {session}")

        return workspace, session
    else:
        return workspace, "default"
