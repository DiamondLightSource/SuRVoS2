import ntpath
import warnings
from functools import partial

import numpy as np
import seaborn as sns
from loguru import logger
from napari.layers import Image
from napari.layers.image.image import Image
from napari.layers.labels.labels import Labels
from napari.layers.points.points import Points
from napari.qt.progress import progress
from qtpy.QtCore import QSize
from skimage.segmentation import find_boundaries
from skimage.util.dtype import img_as_ubyte

from survos2.config import Config
from survos2.frontend.control.launcher import Launcher
from survos2.frontend.paint_strokes import paint_strokes
from survos2.frontend.panels import ButtonPanelWidget, PluginPanelWidget
from survos2.frontend.transfer_fn import _transfer_features_http, _transfer_labels, _transfer_points
from survos2.frontend.utils import get_array_from_dataset, get_color_mapping
from survos2.frontend.view_fn import (
    get_level_from_server,
    remove_layer,
    view_feature,
    view_objects,
    view_pipeline,
    view_regions,
)
from survos2.frontend.workflow import run_workflow
from survos2.helpers import AttrDict
from survos2.improc.utils import DatasetManager
from survos2.model import DataModel, Workspace
from survos2.server.state import cfg
from survos2.utils import decode_numpy

warnings.filterwarnings("ignore")


def frontend(viewer):
    logger.info(f"Frontend loading workspace: {DataModel.g.current_workspace}")
    cfg.base_dataset_shape = (100, 100, 100)
    cfg.slice_max = 100
    cfg.current_mode = "paint"
    cfg.label_ids = [
        0,
    ]
    cfg.retrieval_mode = Config["volume_mode"]  # "volume"  # volume_http | volume | slice
    cfg.current_slice = 0
    cfg.current_orientation = 0

    cfg.current_feature_name = "001_raw"
    cfg.current_annotation_name = None
    cfg.current_pipeline_name = None
    cfg.current_regions_name = None
    cfg.current_analyzers_name = None
    cfg.emptying_viewer = False

    cfg.current_regions_dataset = None
    cfg.current_supervoxels = None
    cfg.supervoxels_cache = None
    cfg.supervoxels_cached = False
    cfg.supervoxel_size = 10
    cfg.brush_size = 10
    cfg.viewer_order = (0, 1, 2)

    # controls whether annotation updates the server with every stroke
    cfg.remote_annotation = True

    cfg.object_offset = (0, 0, 0)
    cfg.num_undo = 0

    label_dict = {
        "level": "001_level",
        "idx": 1,
        "color": "#000000",
    }
    cfg.label_value = label_dict

    cfg.order = (0, 1, 2)
    cfg.group = "main"

    # SuRVoS controls
    dw = AttrDict()
    dw.ppw = PluginPanelWidget()  # Main SuRVoS panel
    dw.bpw = ButtonPanelWidget()  # Additional controls
    dw.ppw.setMinimumSize(QSize(600, 500))
    dw.bpw.setMinimumSize(QSize(600, 250))
    if DataModel.g.current_workspace != "":
        ws = Workspace(DataModel.g.current_workspace)
        dw.ws = ws

    dw.datamodel = DataModel.g
    dw.Launcher = Launcher
    viewer.theme = "dark"

    def set_paint_params(msg):
        if not cfg.emptying_viewer:
            logger.debug(f"set_paint_params {msg['paint_params']}")
            paint_params = msg["paint_params"]
            label_value = paint_params["label_value"]

            if label_value is not None and len(viewer.layers.selection) > 0:
                _set_anno_layer_params(label_value, paint_params)

    def _set_anno_layer_params(label_value, paint_params):
        anno_layer = [v for v in viewer.layers if v.name == cfg.current_annotation_name]

        if anno_layer:
            if not cfg.remote_annotation:
                update_annotation_layer_in_viewer(cfg.current_annotation_name, cfg.anno_data)

            # cfg.local_sv = False
            anno_layer = anno_layer[0]
            cfg.label_ids = list(np.unique(anno_layer))
            anno_layer.mode = "paint"
            cfg.current_mode = "paint"
            anno_layer.selected_label = int(label_value["idx"]) - 1
            cfg.label_value = label_value
            anno_layer.brush_size = int(paint_params["brush_size"])

            if paint_params["current_supervoxels"] is not None:
                sv_name = ntpath.basename(paint_params["current_supervoxels"])
                existing_sv_layer = [v for v in viewer.layers if v.name == sv_name]
                if not existing_sv_layer:
                    view_regions(viewer, {"region_id": sv_name})
            else:
                logger.debug("paint_params['current_supervoxels'] is None")

            viewer.layers.selection.active = anno_layer

    def update_annotations(msg):
        logger.debug(f"update_annotation {msg}")
        if not cfg.remote_annotation:
            print("Remote annotation")
            update_annotation_layer_in_viewer(msg["level_id"], cfg.anno_data)
            cfg.ppw.clientEvent.emit(
                {
                    "source": "save_annotation",
                    "data": "save_annotation",
                    "value": None,
                }
            )
        else:
            src = DataModel.g.dataset_uri(msg["level_id"], group="annotations")
            with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
                src_annotations_dataset = DM.sources[0][:]
                src_arr = get_array_from_dataset(src_annotations_dataset)
            update_annotation_layer_in_viewer(msg["level_id"], src_arr)

    def update_annotation_layer_in_viewer(layer_name, src_arr):
        existing_layer = [v for v in viewer.layers if v.name == layer_name]
        if existing_layer:
            existing_layer[0].data = src_arr.astype(np.int32) & 15

    def refresh_annotations_in_viewer(msg):
        print(f"refresh_annotation {msg['level_id']}")

        if not cfg.emptying_viewer:
            try:
                cfg.current_annotation_name = msg["level_id"]
                if cfg.remote_annotation:
                    print("Remote annotation, getting level from server")
                    src_arr, src_annotations_dataset = get_level_from_server(
                        msg, retrieval_mode=cfg.retrieval_mode
                    )
                else:
                    # annotation level is not in viewer, needs to be loaded from server
                    anno_layer = [v for v in viewer.layers if v.name == cfg.current_annotation_name]
                    if len(anno_layer) == 0:
                        print("Number of annotation layers is 0")
                        src_arr, src_annotations_dataset = get_level_from_server(
                            msg, retrieval_mode=cfg.retrieval_mode
                        )
                        cfg.anno_data = src_arr

                    print("Refresh from server paused, updating annotation from cfg.anno_data")

                    src_arr = cfg.anno_data

                    if cfg.retrieval_mode != "slice":
                        print("Saving annotation")
                        cfg.ppw.clientEvent.emit(
                            {
                                "source": "save_annotation",
                                "data": "save_annotation",
                                "value": None,
                            }
                        )

                result = Launcher.g.run(
                    "annotations", "get_levels", workspace=DataModel.g.current_workspace
                )

                if result:
                    return _refresh_annotations_in_viewer(result, msg, src_arr)

            except Exception as e:
                print(f"Exception {e}")

    def _refresh_annotations_in_viewer(result, msg, src_arr):
        cmapping, label_ids = get_color_mapping(result, msg["level_id"])
        cfg.label_ids = label_ids
        existing_layer = [v for v in viewer.layers if v.name == msg["level_id"]]
        # some defaults
        sel_label = 1
        brush_size = 10

        if existing_layer and cfg.current_annotation_name:
            existing_layer[0].data = src_arr.astype(np.int32) & 15
            label_layer = existing_layer[0]
            existing_layer[0].color = cmapping
        elif cfg.current_annotation_name:
            label_layer = viewer.add_labels(src_arr & 15, name=msg["level_id"], color=cmapping)
            label_layer.mode = cfg.current_mode
            label_layer.brush_size = brush_size

        if cfg.label_value is not None:
            label_layer.selected_label = int(cfg.label_value["idx"]) - 1

        return label_layer

    def setup_paint_undo(label_layer):
        @label_layer.bind_key("Control-Z", overwrite=True)
        def undo():
            logger.info("Undoing annotation")
            if cfg.num_undo == 0:
                cfg.anno_data = cfg.prev_arr.copy()
                cfg.num_undo += 1
                existing_layer = [v for v in viewer.layers if v.name == cfg.current_annotation_name]
                if existing_layer:
                    existing_layer[0].data = cfg.anno_data.astype(np.int32) & 15
                label_layer.undo()
                if cfg.retrieval_mode != "slice":
                    logger.debug("Saving annotation")
                    cfg.ppw.clientEvent.emit(
                        {
                            "source": "save_annotation",
                            "data": "save_annotation",
                            "value": None,
                        }
                    )

    def setup_painting_layer(label_layer, msg, parent_level, parent_label_idx):
        if "anno_data" in cfg:
            logger.debug(f"Anno data is of shape {cfg.anno_data}")
        else:
            cfg.anno_data = label_layer.data
            logger.debug(f"Anno data is initialized to shape {cfg.anno_data}")

        if not hasattr(label_layer, "already_init"):

            @label_layer.mouse_drag_callbacks.append
            def painting_layer(layer, event):
                logger.debug(cfg.current_annotation_name)
                cfg.prev_arr = label_layer.data.copy()
                drag_pts = []
                coords = np.round(layer.world_to_data(viewer.cursor.position)).astype(np.int32)
                cfg.current_slice = coords[0]
                try:
                    drag_pt = [coords[0], coords[1], coords[2]]
                    drag_pts.append(drag_pt)
                    yield

                    if layer.mode in ["fill", "pick"]:
                        layer.mode = "paint"
                    if layer.mode in ["paint", "erase"]:
                        while event.type == "mouse_move":
                            coords = np.round(layer.world_to_data(viewer.cursor.position)).astype(
                                np.int32
                            )
                            drag_pt = [coords[0], coords[1], coords[2]]
                            drag_pts.append(drag_pt)
                            yield

                        if len(drag_pts) >= 0:
                            anno_layer = [
                                v for v in viewer.layers if v.name == cfg.current_annotation_name
                            ]
                            if len(anno_layer) > 0:
                                anno_layer = anno_layer[0]

                                def update_anno(msg):
                                    if cfg.remote_annotation:
                                        src_arr, _ = get_level_from_server(
                                            msg, retrieval_mode=cfg.retrieval_mode
                                        )
                                    else:
                                        src_arr = cfg.anno_data
                                        logger.debug(
                                            f"replaced src array with array of shape {src_arr.shape}"
                                        )
                                    update_annotation_layer_in_viewer(msg["level_id"], src_arr)

                                update = partial(update_anno, msg=msg)
                                viewer_order = viewer.window.qt_viewer.viewer.dims.order
                                cfg.viewer_order = viewer_order
                                paint_strokes_worker = paint_strokes(
                                    msg,
                                    drag_pts,
                                    anno_layer,
                                    cfg.parent_level,
                                    cfg.parent_label_idx,
                                    viewer_order,
                                )
                                paint_strokes_worker.returned.connect(update)
                                paint_strokes_worker.start()

                                cfg.num_undo = 0
                        else:
                            cfg.ppw.clientEvent.emit(
                                {
                                    "source": "annotations",
                                    "data": "update_annotations",
                                    "level_id": msg["level_id"],
                                }
                            )
                except ValueError as e:
                    print(e)

        label_layer.already_init = 1

    def paint_annotations(msg):
        # if not in the process of clearing viewer
        if not cfg.emptying_viewer:
            try:
                label_layer = refresh_annotations_in_viewer(msg)

                sel_label = int(cfg.label_value["idx"]) if cfg.label_value is not None else 1
                if msg["level_id"] is not None:
                    params = dict(workspace=True, level=msg["level_id"], label_idx=sel_label)
                    result = Launcher.g.run("annotations", "get_label_parent", **params)
                    parent_level = result[0]
                    parent_label_idx = result[1]
                    cfg.parent_level = parent_level
                    cfg.parent_label_idx = parent_label_idx
                    setup_paint_undo(label_layer)
                    setup_painting_layer(label_layer, msg, parent_level, parent_label_idx)

            except Exception as e:
                print(f"Exception: {e}")

    def set_session(msg):
        logger.debug(f"Set session to {msg['session']}")
        DataModel.g.current_session = msg["session"]

    def set_workspace(msg):
        logger.debug(f"Set workspace to {msg['workspace']}")
        DataModel.g.current_workspace = msg["workspace"]
        # set on server
        params = dict(workspace=msg["workspace"])
        result = Launcher.g.run("workspace", "set_workspace", **params)
        viewer.title = msg["workspace"]

    def view_patches(msg):
        from survos2.entity.patches import load_patch_vols

        logger.debug(f"view_patches {msg['patches_fullname']}")
        img_vols, label_vols = load_patch_vols(msg["patches_fullname"])
        viewer.add_image(img_vols, name="Patch Image")
        viewer.add_image(label_vols, name="Patch Label")

    def save_annotation(msg):
        logger.info(f"Save annotation {msg}")
        annotation_layer = [v for v in viewer.layers if v.name == cfg.current_annotation_name]
        if len(annotation_layer) == 1:
            logger.info(
                f"Updating annotation {cfg.current_annotation_name} with label image {annotation_layer}"
            )

            result = Launcher.g.post_array(
                annotation_layer[0].data,
                group="annotations",
                workspace=DataModel.g.current_workspace,
                name=cfg.current_annotation_name,
            )
        else:
            logger.info("save_annotation couldn't find annotation in viewer")

    def transfer_layer(msg):
        params = dict(workspace=DataModel.g.current_workspace)
        result = Launcher.g.run("workspace", "set_workspace", **params)

        with progress(total=1) as pbar:
            pbar.set_description("Transferring layer")
            logger.debug(f"transfer_layer {msg}")
            selected_layer = viewer.layers.selection.pop()

            if isinstance(selected_layer, Labels):
                transfer_function = _transfer_labels
                plugin_name = "annotations"
            elif isinstance(selected_layer, Points):
                transfer_function = _transfer_points
                plugin_name = "objects"
            elif isinstance(selected_layer, Image):
                transfer_function = _transfer_features_http
                plugin_name = "features"
            else:
                logger.debug("Unsupported layer type.")
                return

            transfer_function(selected_layer)
            pbar.update(1)
            processEvents({"data": "refresh_plugin", "plugin_name": plugin_name})

    def make_roi_ws(msg):
        logger.debug(f"Goto roi: {msg}")
        if "feature_id" in msg:
            feature_id = msg["feature_id"]
            print(f"Making ROI from feature id {feature_id}")
        else:
            feature_id = "001_raw"
        params = dict(
            workspace=True,
            current_workspace_name=DataModel.g.current_workspace,
            feature_id=feature_id,
            roi=msg["roi"],
        )
        result = Launcher.g.run("workspace", "make_roi_ws", **params)
        if result:
            logger.debug(f"Switching to make_roi_ws created workspace {result}")

    def get_crop(msg):
        logger.debug(f"Getting crop roi: {msg}")
        features_src = DataModel.g.dataset_uri(msg["feature_id"], group="features")
        params = dict(workpace=True, src=features_src, roi=(0, 0, 0, 100, 100, 100))
        result = Launcher.g.run("features", "get_crop", **params)
        if result:
            src_arr = decode_numpy(result)
            viewer.add_image(src_arr, name=msg["feature_id"])

    def show_roi(msg):
        logger.info(f"Showing ROI {msg['selected_roi']}")
        z, y, x = msg["selected_roi"]
        existing_feature_layer = [v for v in viewer.layers if v.name == cfg.current_feature_name]
        viewer.camera.center = (int(float(z)), int(float(y)), int(float(x)))
        viewer.dims.set_current_step(0, z)
        viewer.camera.zoom = 4

    def processEvents(msg):
        "Main event handling function uses the message to update the viewer"
        if msg["data"] == "refesh_annotations":
            refresh_annotations_in_viewer(msg)
        elif msg["data"] == "paint_annotations":
            paint_annotations(msg)
        elif msg["data"] == "update_annotations":
            update_annotations(msg)
        elif msg["data"] == "remove_layer":
            layer_name = msg["layer_name"]
            remove_layer(viewer, layer_name)
        elif msg["data"] == "view_feature":
            view_feature(viewer, msg)
        elif msg["data"] == "view_pipeline":
            if msg["source"] == "analyzer":
                view_pipeline(viewer, msg, analyzers=True)
            else:
                view_pipeline(viewer, msg)
        elif msg["data"] == "view_regions":
            view_regions(viewer, msg)
        elif msg["data"] == "view_objects":
            view_objects(viewer, msg)
        elif msg["data"] == "view_patches":
            view_patches(msg)
        elif msg["data"] == "show_roi":
            show_roi(msg)
        elif msg["data"] == "run_workflow":
            run_workflow_worker = run_workflow(msg)
            run_workflow_worker.start()
            processEvents({"data": "faster_refresh"})
        elif msg["data"] == "refresh":
            params = dict(workspace=DataModel.g.current_workspace)
            result = Launcher.g.run("workspace", "set_workspace", **params)
            logger.debug("Refreshing plugin panel")
            dw.ppw.setup()
        elif msg["data"] == "faster_refresh":
            logger.debug("Faster refresh")
            dw.ppw.setup_fast()
        elif msg["data"] == "refresh_chroot":
            logger.debug("Refresh chroot")
            dw.bpw.refresh_workspaces()
        elif msg["data"] == "refresh_plugin":
            plugin_name = msg["plugin_name"]
            logger.debug("Refresh plugin")
            dw.ppw.setup_named_plugin(plugin_name)
        elif msg["data"] == "faster_refresh_plugin":
            plugin_name = msg["plugin_name"]
            logger.debug("Refresh plugin")
            dw.ppw.faster_setup_named_plugin(plugin_name)
        elif msg["data"] == "empty_viewer":
            logger.debug("\n\nEmptying viewer")
            for l in viewer.layers:
                viewer.layers.remove(l)
            # some bug does not allow to remove all layers at once
            for l in viewer.layers:
                viewer.layers.remove(l)
            cfg.current_feature_name = "001_raw"
            cfg.current_annotation_name = None
            cfg.current_pipeline_name = None
            cfg.current_regions_name = None
            cfg.current_analyzers_name = None
        elif msg["data"] == "save_annotation":
            save_annotation(msg)
        elif msg["data"] == "set_paint_params":
            set_paint_params(msg)
        elif msg["data"] == "set_session":
            set_session(msg)
        elif msg["data"] == "set_workspace":
            set_workspace(msg)
        elif msg["data"] == "get_crop":
            get_crop(msg)
        elif msg["data"] == "make_roi_ws":
            make_roi_ws(msg)
        elif msg["data"] == "transfer_layer":
            transfer_layer(msg)

    # setup message based event handling mechanism and return it to the dockwidget
    dw.ppw.clientEvent.connect(lambda x: processEvents(x))
    cfg.ppw = dw.ppw
    cfg.bpw = dw.bpw
    cfg.processEvents = processEvents
    cfg.viewer = viewer
    dw.Config = Config
    dw.cfg = cfg

    return dw
