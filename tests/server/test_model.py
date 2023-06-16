from survos2.model.workspace import Workspace
from survos2.data_io import dataset_from_uri
from survos2.config import Config
import dask.array as da
from survos2.improc.utils import optimal_chunksize


def test_datamodel():
    ws = Workspace(".")
    workspace_fpath = "./newws1"
    ws = ws.create(workspace_fpath)
    data_fname = "./tmp/testvol_4x4x4b.h5"

    with dataset_from_uri(data_fname, mode="r") as data:
        chunk_size = optimal_chunksize(data, Config["computing.chunk_size"])
        data = da.from_array(data, chunks=chunk_size)
        data -= da.min(data)
        data /= da.max(data)
        ds = ws.add_data(data)

    ws.add_dataset("testds", "float32")
    assert ws.exists(workspace_fpath)
    assert ws.has_data()
    assert ws.available_datasets() == ["testds"]
    ws.add_session("newsesh")
    assert ws.has_session("newsesh")

    ws.delete()
