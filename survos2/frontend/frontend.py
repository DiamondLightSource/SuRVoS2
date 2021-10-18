import ntpath

from skimage.util.dtype import img_as_ubyte
from survos2.config import Config
from functools import partial
import numpy as np
import seaborn as sns
from loguru import logger

from napari.layers import Image
from napari.layers.image.image import Image
from napari.layers.points.points import Points
from napari.layers.labels.labels import Labels
from napari.qt import progress

from qtpy.QtCore import QSize

from skimage.segmentation import find_boundaries

from survos2.frontend.panels import ButtonPanelWidget, PluginPanelWidget
from survos2.frontend.utils import get_array_from_dataset, get_color_mapping
from survos2.helpers import AttrDict
from survos2.improc.utils import DatasetManager
from survos2.model import DataModel, Workspace
from survos2.frontend.control.launcher import Launcher
from survos2.server.state import cfg
from survos2.config import Config
from survos2.utils import decode_numpy, decode_numpy_slice
from survos2.frontend.paint_strokes import paint_strokes
from survos2.frontend.workflow import run_workflow
from survos2.frontend.view_fn import (
    view_feature,
    view_regions,
    remove_layer,
    view_objects,
    view_pipeline,
)

from survos2.frontend.transfer_fn import (
    _transfer_features,
    _transfer_labels,
    _transfer_features_http,
    _transfer_points
)
import warnings

warnings.filterwarnings("ignore")


_MaskSize = 4  # 4 bits per history label
_MaskCopy = 15  # 0000 1111
_MaskPrev = 240  # 1111 0000


def get_level_from_server(msg, retrieval_mode="volume"):
    if retrieval_mode == "slice":  # get a slice over http
        src_annotations_dataset = DataModel.g.dataset_uri(
            msg["level_id"], group="annotations"
        )
        params = dict(
            workpace=True,
            src=src_annotations_dataset,
            slice_idx=cfg.current_slice,
            order=cfg.order,
        )
        result = Launcher.g.run("annotations", "get_slice", **params)
        if result:
            src_arr = decode_numpy(result)
    elif retrieval_mode == "volume_http":  # get a slice over http
        src_annotations_dataset = DataModel.g.dataset_uri(
            msg["level_id"], group="annotations"
        )
        params = dict(workpace=True, src=src_annotations_dataset)
        result = Launcher.g.run("annotations", "get_volume", **params)
        if result:
            src_arr = decode_numpy(result)
    elif retrieval_mode == "volume":  # get entire volume
        src = DataModel.g.dataset_uri(msg["level_id"], group="annotations")
        with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
            src_annotations_dataset = DM.sources[0][:]
            src_arr = get_array_from_dataset(src_annotations_dataset)

    return src_arr, src_annotations_dataset


def frontend(viewer):
    logger.info(f"Frontend loading workspace: {DataModel.g.current_workspace}")
    cfg.base_dataset_shape = (100, 100, 100)
    cfg.slice_max = 100

    cfg.current_mode = "paint"
    cfg.label_ids = [
        0,
    ]
    cfg.retrieval_mode =  Config["volume_mode"] # "volume"  # volume_http | volume | slice
    cfg.current_slice = 0
    cfg.current_orientation = 0

    cfg.current_feature_name = "001_raw"
    cfg.current_annotation_name = None
    cfg.current_pipeline_name = None
    cfg.current_regions_name = None
    cfg.current_analyzers_name = None

    cfg.emptying_viewer = False
    cfg.three_dim = False

    cfg.current_regions_dataset = None
    cfg.current_supervoxels = None
    cfg.supervoxels_cache = None
    cfg.supervoxels_cached = False
    cfg.supervoxel_size = 10
    cfg.local_sv = True
    cfg.pause_save = False
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
    dw.ppw.setMinimumSize(QSize(450, 750))

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
                _set_params(label_value, paint_params)

    def _set_params(label_value, paint_params):
        anno_layer = [v for v in viewer.layers if v.name == cfg.current_annotation_name]

        if len(anno_layer) > 0:
            if not cfg.remote_annotation:
                if cfg.retrieval_mode != 'slice':
                    cfg.ppw.clientEvent.emit(
                                {
                                    "source": "save_annotation",
                                    "data": "save_annotation",
                                    "value": None,
                                }
                            )
            cfg.local_sv = False
        
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
        
        if cfg.local_sv:
            update_annotation_layer_in_viewer(msg["level_id"], cfg.anno_data)
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
        logger.debug(f"refresh_annotation {msg['level_id']}")
        if not cfg.emptying_viewer:
            try:
                cfg.current_annotation_name = msg["level_id"]

                if cfg.local_sv:
                    anno_layer = [v for v in viewer.layers if v.name == cfg.current_annotation_name]
                    if len(anno_layer) == 0:
                        src_arr, src_annotations_dataset = get_level_from_server(
                            msg, retrieval_mode=cfg.retrieval_mode
                        )    
                        cfg.anno_data = src_arr

                    print("Refresh from server paused, updating annotation from cfg.anno_data")
                    if 'prev_arr' in cfg: 
                        src_arr = cfg.prev_arr
                    else:
                        src_arr = cfg.anno_data
                        
                    if cfg.retrieval_mode != 'slice':
                        print("Saving annotation")
                        cfg.ppw.clientEvent.emit({
                                    "source": "save_annotation",
                                    "data": "save_annotation",
                                    "value": None,
                                }
                            )
                else:
                    src_arr, src_annotations_dataset = get_level_from_server(
                        msg, retrieval_mode=cfg.retrieval_mode
                    )

                result = Launcher.g.run(
                    "annotations", "get_levels", workspace=DataModel.g.current_workspace
                )
  
                if result:
                    return _refresh_annotations_in_viewer(
                        result, msg, src_arr
                    )

  

            except Exception as e:
                print(f"Exception {e}")

    def _refresh_annotations_in_viewer(result, msg, src_arr):
        print(f"Refresh annotations in viewer {src_arr.shape}")
        cmapping, label_ids = get_color_mapping(result, msg["level_id"])
        logger.debug(f"Label ids {label_ids}")
        cfg.label_ids = label_ids
        existing_layer = [v for v in viewer.layers if v.name == msg["level_id"]]
        
        # some defaults
        sel_label = 1
        brush_size = 10

        if existing_layer and cfg.current_annotation_name:
            existing_layer[0].data = src_arr.astype(np.int32) & 15
            label_layer = existing_layer[0]
            existing_layer[0].color = cmapping
        else:
            print(f"Adding labels {src_arr.shape} {cmapping }")
            label_layer = viewer.add_labels(
            src_arr & 15, name=msg["level_id"], color=cmapping
            )
            label_layer.mode = cfg.current_mode
            label_layer.brush_size = brush_size
        
        if cfg.label_value is not None:
            label_layer.selected_label = int(cfg.label_value["idx"]) - 1

 

        print(f"Returning label layer {label_layer}")
        return label_layer

    def setup_paint_undo_remote(label_layer):
        @label_layer.bind_key("Control-Z", overwrite=True)
        def undo(v):
            logger.info("Undoing")
            if cfg.num_undo == 0:
                level = cfg.current_annotation_name
                params = dict(workspace=True, level=level)
                result = Launcher.g.run("annotations", "annotate_undo", **params)
                cfg.ppw.clientEvent.emit(
                    {
                        "source": "annotations",
                        "data": "update_annotations",
                        "level_id": level,
                    }
                )
               
                cfg.num_undo += 1

    def setup_paint_undo_local(label_layer):
        @label_layer.bind_key("Control-Z", overwrite=True)
        def undo(v):
            logger.info("Undoing local annotation")
            if cfg.num_undo == 0:
                print(cfg.anno_data)
                #cfg.anno_data = cfg.anno_data >> _MaskSize
                cfg.anno_data = cfg.prev_arr.copy()
                cfg.num_undo += 1
                existing_layer = [v for v in viewer.layers if v.name == cfg.current_annotation_name]
                if existing_layer:
                    existing_layer[0].data = cfg.anno_data.astype(np.int32) & 15
                label_layer.undo()
                if cfg.retrieval_mode != 'slice':
                    logger.debug("Saving annotation")
                    cfg.ppw.clientEvent.emit({
                                "source": "save_annotation",
                                "data": "save_annotation",
                                "value": None,
                    })


    def setup_painting_layer(label_layer, msg, parent_level, parent_label_idx):
        if not hasattr(label_layer, 'already_init'):
            @label_layer.mouse_drag_callbacks.append
            def painting_layer(layer, event):
                cfg.prev_arr = label_layer.data.copy()
                drag_pts = []
                coords = np.round(layer.world_to_data(viewer.cursor.position)).astype(
                    np.int32
                )
                try:
                    if cfg.retrieval_mode == "slice":
                        drag_pt = [coords[0], coords[1]]
                    elif cfg.retrieval_mode == "volume" or cfg.retrieval_mode == "volume_http":
                        drag_pt = [coords[0], coords[1], coords[2]]
                    drag_pts.append(drag_pt)
                    yield

                    if layer.mode == "fill" or layer.mode == "pick":
                        layer.mode = "paint"
                    if layer.mode == "paint" or layer.mode == "erase":
                        while event.type == "mouse_move":
                            
                            coords = np.round(
                                layer.world_to_data(viewer.cursor.position)
                            ).astype(np.int32)

                            if cfg.retrieval_mode == "slice":
                                drag_pt = [coords[0], coords[1]]
                            elif cfg.retrieval_mode == "volume" or cfg.retrieval_mode == "volume_http":
                                drag_pt = [coords[0], coords[1], coords[2]]
                            drag_pts.append(drag_pt)
                            yield

                        if len(drag_pts) >= 0:
                            #top_layer = viewer.layers[-1]
                            #layer_name = top_layer.name  # get last added layer name
                            anno_layer = [v for v in viewer.layers if v.name == cfg.current_annotation_name]
                            if len(anno_layer) > 0:
                                anno_layer = anno_layer[0]
                                #anno_layer = next(l for l in viewer.layers if l.name == cfg.current_annotation_name)
                    
                                def update_anno(msg):
                                    if cfg.local_sv:
                                        
                                        src_arr = cfg.anno_data 
                                    else:
                                        src_arr, _ = get_level_from_server(
                                            msg, retrieval_mode=cfg.retrieval_mode
                                        )
                                    update_annotation_layer_in_viewer(msg["level_id"], src_arr)

                                update = partial(update_anno, msg=msg)
                                viewer_order = viewer.window.qt_viewer.viewer.dims.order
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
        if not cfg.emptying_viewer:
            logger.debug(f"paint_annotation {msg['level_id']}")
            
            try:
                label_layer = refresh_annotations_in_viewer(msg)
                print(f"paint_annotations label_layer {label_layer}")
                sel_label = (
                    int(cfg.label_value["idx"]) if cfg.label_value is not None else 1
                )
                if msg["level_id"] is not None:
                    params = dict(
                        workspace=True, level=msg["level_id"], label_idx=sel_label
                    )
                    result = Launcher.g.run("annotations", "get_label_parent", **params)
                    parent_level = result[0]
                    parent_label_idx = result[1]
                    print(f"Got label parent: level {parent_level} label {parent_label_idx}")
                    cfg.parent_level = parent_level
                    cfg.parent_label_idx = parent_label_idx
                    setup_paint_undo_local(label_layer)
                    setup_painting_layer(label_layer, msg, parent_level, parent_label_idx)

            except Exception as e:
                print(f"Exception: {e}")
    

    def set_session(msg):
        logger.debug(f"Set session to {msg['session']}")
        DataModel.g.current_session = msg["session"]

    def set_workspace(msg):
        logger.debug(f"Set workspace to {msg['workspace']}")
        # set on client
        DataModel.g.current_workspace = msg["workspace"]
        # set on server
        params = dict(workspace=msg["workspace"])
        result = Launcher.g.run("workspace", "set_workspace", **params)

    def view_patches(msg):
        from entityseg.training.patches import load_patch_vols
        logger.debug(f"view_patches {msg['patches_fullname']}")
        img_vols, label_vols = load_patch_vols(msg["patches_fullname"])
        viewer.add_image(img_vols, name="Patch Image")
        viewer.add_image(label_vols, name="Patch Label")

    def save_annotation(msg):
        logger.info(f"Save annotation {msg}")
        annotation_layer = [
            v for v in viewer.layers if v.name == cfg.current_annotation_name
        ]
        if len(annotation_layer) == 1:
            logger.info(
                f"Updating annotation {cfg.current_annotation_name} with label image {annotation_layer}"
            )
            
            result = Launcher.g.post_array(annotation_layer[0].data, 
                group='annotations', 
                workspace=DataModel.g.current_workspace, 
                name=cfg.current_annotation_name)
        else:
            logger.info("save_annotation couldn't find annotation in viewer")


    def transfer_layer(msg):
        with progress(total=1) as pbar:
            pbar.set_description("Transferring layer")
            logger.debug(f"transfer_layer {msg}")
            selected_layer = viewer.layers.selection.pop()
            if isinstance(selected_layer, Labels):
                _transfer_labels(selected_layer)
            elif isinstance(selected_layer, Points):
                _transfer_points(selected_layer)
            elif isinstance(selected_layer, Image):
                _transfer_features_http(selected_layer)
            else:
                logger.debug("Unsupported layer type.")
            pbar.update(1)
            processEvents({"data": "refresh"})

    def jump_to_slice(msg):
        cfg.supervoxels_cached = False
        logger.debug(f"jump_to_slice Using order {cfg.order}")
        cfg.retrieval_mode = "slice"
        cfg.current_slice = int(msg["frame"])
        logger.debug(f"Jump around to {cfg.current_slice}, msg {msg['frame']}")

        existing_feature_layer = [
            v for v in viewer.layers if v.name == cfg.current_feature_name
        ]

        if existing_feature_layer:
            features_src = DataModel.g.dataset_uri(
                cfg.current_feature_name, group="features"
            )
            params = dict(
                workpace=True,
                src=features_src,
                slice_idx=cfg.current_slice,
                order=cfg.order,
            )
            result = Launcher.g.run("features", "get_slice", **params)
            if result:
                src_arr = decode_numpy_slice(result)
                existing_feature_layer[0].data = src_arr.copy()

        existing_regions_layer = [
            v for v in viewer.layers if v.name == cfg.current_regions_name
        ]
        if existing_regions_layer:
            regions_src = DataModel.g.dataset_uri(
                cfg.current_regions_name, group="superregions"
            )
            params = dict(
                workpace=True,
                src=regions_src,
                slice_idx=cfg.current_slice,
                order=cfg.order,
            )
            result = Launcher.g.run("superregions", "get_slice", **params)
            if result:
                src_arr = decode_numpy(result)
                src_arr = find_boundaries(src_arr) * 1.0
                existing_regions_layer[0].data = src_arr.copy()
                existing_regions_layer[0].opacity = 0.3

        existing_level_layer = [
            v for v in viewer.layers if v.name == cfg.current_annotation_name
        ]
        if existing_level_layer and cfg.current_annotation_name is not None:
            paint_annotations({"level_id": cfg.current_annotation_name})

        existing_pipeline_layer = [
            v for v in viewer.layers if v.name == cfg.current_pipeline_name
        ]

        if existing_pipeline_layer:
            print(f"loading pipeline {cfg.current_pipeline_name}")
            pipeline_src = DataModel.g.dataset_uri(
                cfg.current_pipeline_name, group="pipelines"
            )
            params = dict(
                workpace=True,
                src=pipeline_src,
                slice_idx=cfg.current_slice,
                order=cfg.order,
            )
            result = Launcher.g.run("features", "get_slice", **params)
            if result:
                src_arr = decode_numpy(result).astype(np.int32)
                existing_pipeline_layer[0].data = src_arr.copy()
    
        existing_analyzers_layer = [
            v for v in viewer.layers if v.name == cfg.current_analyzers_name
        ]

        if existing_analyzers_layer:
            print(f"Jumping to analyzer slice {cfg.current_analyzers_name}")
            analyzers_src = DataModel.g.dataset_uri(
                cfg.current_analyzers_name, group="analyzer"
            )
            params = dict(
                workpace=True,
                src=analyzers_src,
                slice_idx=cfg.current_slice,
                order=cfg.order,
            )
            result = Launcher.g.run("features", "get_slice", **params)
            if result:
                src_arr = decode_numpy(result).astype(np.int32)
                existing_analyzers_layer[0].data = src_arr.copy()

    def make_roi_ws(msg):
        logger.debug(f"Goto roi: {msg}")
        params = dict(
            workspace=True,
            current_workspace_name=DataModel.g.current_workspace,
            feature_id="001_raw",
            roi=msg["roi"],
        )
        result = Launcher.g.run("workspace", "make_roi_ws", **params)
        if result:
            logger.debug(f"Switching to make_roi_ws created workspace {result}")
            DataModel.g.current_workspace = result
            processEvents({"data": "refresh"})

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
        z, x, y = msg["selected_roi"]
        existing_feature_layer = [
            v for v in viewer.layers if v.name == cfg.current_feature_name
        ]
        viewer.camera.center = (z, x, y)
        viewer.dims.set_current_step(0, z)
        viewer.camera.zoom = 4

    def processEvents(msg):
        logger.debug(msg)
        if msg["data"] == "jump_to_slice":
            jump_to_slice(msg)
        elif msg["data"] == "slice_mode":
            logger.debug(f"Slice mode changing from: {cfg.retrieval_mode}")
            if cfg.retrieval_mode != "slice":
                cfg.local_sv = True
                _switch_to_slice_mode_and_jump()
            else:
                try:
                    logger.debug(f"In slice mode changing to volume mode {viewer.layers}")
                    cfg.retrieval_mode = "volume"
                    cfg.local_sv = True
                    for _ in range(len(viewer.layers)):
                        viewer.layers.pop(0)
                    view_feature(viewer, {"feature_id": cfg.current_feature_name})
                except KeyError as e:
                    print(e)
        elif msg["data"] == "refesh_annotations":
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
            if msg["source"] == 'analyzer':
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
            processEvents({"data": "refresh"})
        elif msg["data"] == "refresh":
            logger.debug("Refreshing plugin panel")
            dw.ppw.setup()
        elif msg["data"] == "empty_viewer":
            logger.debug("\n\nEmptying viewer")
            for l in viewer.layers:
                viewer.layers.remove(l)
            
            cfg.current_feature_name = "001_raw"
            
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

    def _switch_to_slice_mode_and_jump():

        cfg.retrieval_mode = "slice"
        viewer_order = viewer.window.qt_viewer.viewer.dims.order
        for l in viewer.layers:
            viewer.layers.remove(l)
        
        if len(viewer_order) == 3:
            cfg.order = [int(d) for d in viewer_order]
            logger.debug(f"Setting order to {cfg.order}")
        else:
            cfg.order = [0, 1, 2]
            logger.debug(f"Viewer order {viewer_order} Resetting order to {cfg.order}")
        cfg.slice_max = cfg.base_dataset_shape[cfg.order[0]]
        logger.debug(f"Setting slice max to {cfg.slice_max}")
        view_feature(viewer, {"feature_id": cfg.current_feature_name})
        try:
            jump_to_slice({"frame": 0})
        except AttributeError as e:
            print(e)
    dw.ppw.clientEvent.connect(lambda x: processEvents(x))
    cfg.ppw = dw.ppw
    cfg.processEvents = processEvents
    cfg.viewer = viewer
    dw.Config = Config
    dw.cfg = cfg
    return dw
