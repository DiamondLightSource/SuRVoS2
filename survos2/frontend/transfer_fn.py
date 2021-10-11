from survos2.entity.entities import make_entity_df
from loguru import logger
import numpy as np
from survos2.improc.utils import DatasetManager
from survos2.model import DataModel, Workspace
from survos2.frontend.control.launcher import Launcher
from survos2.server.state import cfg
from survos2.utils import decode_numpy
from survos2.frontend.utils import get_array_from_dataset, get_color_mapping
from survos2.frontend.components.entity import setup_entity_table
from skimage.segmentation import find_boundaries
import seaborn as sns
from napari.qt import progress




def _transfer_features_http(selected_layer):
    print(selected_layer.data.shape)
    result = Launcher.g.post_array(selected_layer.data, 
    group='features', 
    workspace=DataModel.g.current_workspace, 
    name="feature")

def _transfer_points(selected_layer):
    logger.debug("Transferring Points layer to Objects.")
    # points from napari have no class code so add column of zeros
    entities_arr = np.concatenate(
        (selected_layer.data, np.zeros((len(selected_layer.data), 1))),
        axis=1,
    ).astype(np.float32)
    result = Launcher.g.post_array(entities_arr, 
    group='objects', 
    workspace=DataModel.g.current_workspace, 
    name="objects")

def _transfer_features(selected_layer):
    logger.debug("Transferring Image layer to Features.")
    params = dict(feature_type="raw", workspace=True)
    result = Launcher.g.run("features", "create", **params)
    fid = result["id"]
    ftype = result["kind"]
    fname = result["name"]
    logger.debug(f"Created new object in workspace {fid}, {ftype}, {fname}")
    result = DataModel.g.dataset_uri(fid, group="features")
    with DatasetManager(result, out=result, dtype="float32", fillvalue=0) as DM:
        DM.out[:] = selected_layer.data
    return result

def _transfer_labels(selected_layer):
    logger.debug("Transferring Labels layer to Annotations.")
    
    # add level
    result = Launcher.g.run("annotations", "add_level", workspace=True)
    new_level_id = result["id"]
    # add labels to newly created level
    label_values = np.unique(selected_layer.data)
    for v in label_values:
        if v != 0:
            params = dict(
                level=result["id"],
                idx=int(v) - 2,
                name=str(int(v) - 2),
                color="#11FF11",
                workspace=True,
            )
            label_result = Launcher.g.run("annotations", "add_label", **params)
    params = dict(level=new_level_id, workspace=DataModel.g.current_workspace)
    # get added level and set label parameters to values from label layer being transferred in
    levels_result = Launcher.g.run("annotations", "get_single_level", **params)
    
    for v in levels_result["labels"].keys():
        label_rgba = np.array(selected_layer.get_color(int(v) - 1))
        label_rgba = (255 * label_rgba).astype(np.uint8)
        label_hex = "#{:02x}{:02x}{:02x}".format(*label_rgba)
        label = dict(
            idx=int(v),
            name=str(int(v) - 1),
            color=label_hex,
        )
        params = dict(level=result["id"], workspace=True)
        label_result = Launcher.g.run(
            "annotations", "update_label", **params, **label
        )

    if levels_result:
        fid = result["id"]
        ftype = result["kind"]
        fname = result["name"]
        logger.debug(f"Created new object in workspace {fid}, {ftype}, {fname}")
        
        print(f"anno_layer: {selected_layer.data.shape}")
        result = Launcher.g.post_array(selected_layer.data.astype(np.uint32), 
        group='annotations', 
        workspace=DataModel.g.current_workspace, 
        name=result["id"])
