

#import hug
import parse
import os.path as op

#from survos2.api.utils import APIException
#from survos2.api.types import String, int, intOrNone
from survos2.io import dataset_from_uri
from survos2.config import Config
from survos2.model import Workspace, Dataset
from pydantic import BaseModel

class IntOrNone(BaseModel):

    def __call__(self, value):
        if value is None or type(value) == int:
            return value
        return int(value)


### Workspace

def create(workspace:str):
    workspace, session = parse_workspace(workspace)
    return Workspace.create(workspace)

def delete(workspace:str):
    workspace, session = parse_workspace(workspace)
    Workspace.remove(workspace)
    return dict(done=True)


#hug.local()
def get(workspace:str):
    workspace, session = parse_workspace(workspace)
    if Workspace.exists(workspace):
        return Workspace(workspace)
    #raise APIException('Workspace \'%s\' does not exist.' % workspace)

### Data

def add_data(workspace:str, dataset:str):
    import dask.array as da
    from survos2.improc.utils import optimal_chunksize
    ws = get(workspace)
    with dataset_from_uri(dataset, mode='r') as data:
        chunk_size = optimal_chunksize(data, Config['computing.chunk_size'])
        data = da.from_array(data, chunks=chunk_size)
        data -= da.min(data)
        data /= da.max(data)
        ds = ws.add_data(data)
    logger.info(type(ds))
    return ds



#hug.local()
def get_data(workspace:str):
    workspace, session = parse_workspace(workspace)
    return get(workspace).get_data()

### Sessions

def list_sessions(workspace:str):
    return get(workspace).available_sessions()


def add_session(workspace:str, session:str):
    return get(workspace).add_session(session)


def delete_session(workspace:str, session:str):
    get(workspace).remove_session(session)
    return dict(done=True)

def get_session(workspace:str, session:str):
    return get(workspace).get_session(session)

### Datasets


def list_datasets(workspace:str):
    workspace, session = parse_workspace(workspace)
    return get(workspace).available_datasets(session)


def add_dataset(workspace:str, dataset:str, dtype:str,
                fillvalue:int=0, group:str=None,
                chunks:IntOrNone=None):
    workspace, session = parse_workspace(workspace)
    if group:
        dataset = '{}/{}'.format(group, dataset)
    return get(workspace).add_dataset(dataset, dtype, session=session,
                                      fillvalue=fillvalue, chunks=chunks)



#hug.local()
def delete_dataset(workspace:str, dataset:str, group:str=None):
    workspace, session = parse_workspace(workspace)
    if group:
        dataset = '{}/{}'.format(group, dataset)
    get(workspace).remove_dataset(dataset, session=session)
    return dict(done=True)



#hug.local()
def get_dataset(workspace:str, dataset:str, group:str=None):
    workspace, session = parse_workspace(workspace)
    if group:
        dataset = '{}/{}'.format(group, dataset)
    return get(workspace).get_dataset(dataset, session=session)

### Get Metadata


#hug.local()
def metadata(workspace:str, dataset:str=None):
    workspace, session = parse_workspace(workspace)
    if dataset:
        ds = get_dataset(workspace, dataset, session=session)
    else:
        ds = get_data(workspace)
    return ds.get_metadata()

### Local utils for plugins

#hug.local()
def existing_datasets(workspace:str, group:str=None, filter:str=None):
    ws, session = parse_workspace(workspace)
    print(f"Existing datasets {ws} {session}")
    ws = get(ws)
    all_ds = sorted(ws.available_datasets(session=session, group=group))
    result = {}
    for dsname in all_ds:
        if filter is None or filter in dsname:
            ds = ws.get_dataset(dsname, session=session)
            result[dsname.split(op.sep)[1]] = ds
    return result


#hug.local()
def auto_create_dataset(workspace:str, name:str, group:str,
                        dtype:str, fill:int=0, chunks:IntOrNone=None):

    print(f"auto_create_dataset {workspace} {name} {group}")
    all_ds = existing_datasets(workspace, group)
    max_idx = 0
    pattern = '{:03d}_{}'
    for dsname in all_ds:
        idx = parse.parse(pattern, dsname)
        if idx:
            max_idx = max(max_idx, idx[0])
    dataset_id = pattern.format(max_idx + 1, name)
    dataset_name = dataset_id.replace('_', ' ').title()
    dataset_file = '{}/{}'.format(group, dataset_id)
    ds = add_dataset(workspace, dataset_file, dtype, fillvalue=fill, chunks=chunks)
    ds.set_attr('name', dataset_name)
    return ds


#hug.local()
def rename_dataset(workspace:str, feature_id:str, group:str, new_name:str):
    ds = get_dataset(workspace, feature_id, group=group)
    ds.set_attr('name', new_name)


#hug.local()
def parse_workspace(workspace:str):
    if '@' in workspace:
        session, workspace = workspace.split('@')
        return workspace, session
    else:
        return workspace, 'default'

