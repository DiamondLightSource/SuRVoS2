import numpy as np
from magicgui import magicgui
from napari import layers
import enum
from skimage import img_as_ubyte
from loguru import logger

from survos2.server.features import (
    prepare_prediction_features,
    generate_features,
    features_factory,
)
from survos2.server.model import SRData, SRFeatures
from survos2.server.config import cfg
from survos2.improc.utils import DatasetManager
from survos2.model import DataModel
from survos2.frontend.control import Launcher
from survos2.frontend.model import ClientData


class Operation(enum.Enum):
    mask_pipeline = "mask_pipeline"
    saliency_pipeline = "saliency_pipeline"
    superprediction_pipeline = "prediction_pipeline"
    survos_pipeline = "survos_pipeline"


class TransferOperation(enum.Enum):
    features = "features"
    regions = "regions"
    annotations = "annotations"


@magicgui(auto_call=True)  # call_button="Set Pipeline")
def pipeline_gui(pipeline_option: Operation):
    cfg["pipeline_option"] = pipeline_option.value


@magicgui(
    call_button="Assign ROI",
    layout="horizontal",
    x_st={"maximum": 5000},
    y_st={"maximum": 5000},
    z_st={"maximum": 5000},
    x_end={"maximum": 5000},
    y_end={"maximum": 5000},
    z_end={"maximum": 5000},
)
def roi_gui(z_st: int, z_end: int, x_st: int, x_end: int, y_st: int, y_end: int):
    cfg["roi_crop"] = (z_st, z_end, x_st, x_end, y_st, y_end)


@magicgui(call_button="Update annotation", layout="vertical")
def save_annotation_gui():
    cfg.ppw.clientEvent.emit(
        {"source": "save_annotation", "data": "save_annotation", "value": None}
    )
    cfg.ppw.clientEvent.emit(
        {"source": "save_annotation", "data": "refresh", "value": None}
    )


@magicgui(call_button="Save to workspace", layout="vertical")
def workspace_gui(Layer: layers.Image, Group: TransferOperation):
    logger.debug(f"Selected layer name: {Layer.name} and shape: {Layer.data.shape} ")

    params = dict(feature_type="raw", workspace=True)

    if Group.name == "features":
        result = Launcher.g.run("features", "create", **params)
    elif Group.name == "annotations":

        params = dict(level=Layer.name, workspace=True)
        result = Launcher.g.run("annotations", "add_level", workspace=True)
        # result = Launcher.g.run("annotations", "get_levels", **params)[0]
        label_values = np.unique(Layer.data)
        for v in label_values:
            params = dict(
                level=result["id"],
                idx=int(v),
                name=str(v),
                color="#11FF11",
                workspace=True,
            )
            label_result = Launcher.g.run("annotations", "add_label", **params)

            print(label_result)

        levels_result = Launcher.g.run("annotations", "get_levels", **params)[0]
        print(levels_result)

        for v in levels_result["labels"].keys():
            print(v)
            label_rgba = np.array(Layer.get_color(int(v)))
            label_rgba = (255 * label_rgba).astype(np.uint8)
            label_hex = "#{:02x}{:02x}{:02x}".format(*label_rgba)
            label = dict(idx=int(v), name=str(v), color=label_hex,)
            params = dict(level=result["id"], workspace=True)
            label_result = Launcher.g.run(
                "annotations", "update_label", **params, **label
            )

    elif Group.name == "regions":
        result = Launcher.g.run("regions", "create", **params)

    if result:
        fid = result["id"]
        ftype = result["kind"]
        fname = result["name"]
        logger.debug(f"Created new object in workspace {fid}, {ftype}, {fname}")

        dst = DataModel.g.dataset_uri(fid, group=Group.name)

        if Group.name == "features":
            with DatasetManager(dst, out=dst, dtype="float32", fillvalue=0) as DM:
                DM.out[:] = Layer.data
        elif Group.name == "annotations":
            with DatasetManager(dst, out=dst, dtype="uint32", fillvalue=0) as DM:
                DM.out[:] = Layer.data
        elif Group.name == "regions":
            with DatasetManager(dst, out=dst, dtype="uint32", fillvalue=0) as DM:
                DM.out[:] = Layer.data

        cfg.ppw.clientEvent.emit(
            {"source": "workspace_gui", "data": "refresh", "value": None}
        )
