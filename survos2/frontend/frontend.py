import ntpath
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
from survos2.utils import decode_numpy
from survos2.frontend.paint_strokes import paint_strokes
from survos2.frontend.workflow import run_workflow
from survos2.frontend.view_fn import (
    view_feature,
    view_regions,
    remove_layer,
    view_objects,
    view_pipeline,
)
import warnings

warnings.filterwarnings("ignore")


def get_annotation_array(msg, retrieval_mode="volume"):
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
    logger.info(f"Frontend loading workspace {DataModel.g.current_workspace}")
    cfg.base_dataset_shape = (100, 100, 100)
    cfg.slice_max = 100

    cfg.current_mode = "paint"
    cfg.label_ids = [
        0,
    ]
    cfg.retrieval_mode = "volume"  # volume | slice
    cfg.current_slice = 0
    cfg.current_orientation = 0

    cfg.current_feature_name = "001_raw"
    cfg.current_annotation_name = None
    cfg.current_pipeline_name = None
    cfg.current_regions_name = None
    cfg.current_supervoxels = None
    cfg.supervoxels_cache = None
    cfg.supervoxels_cached = False
    cfg.current_regions_dataset = None
    cfg.object_scale = 1.0
    cfg.object_offset = (0, 0, 0)  # (-16, 170, 165)  # (-16,-350,-350)
    cfg.num_undo = 0

    label_dict = {
        "level": "001_level",
        "idx": 1,
        "color": "#000000",
    }
    cfg.label_value = label_dict

    cfg.order = (0, 1, 2)
    cfg.group = "main"

    viewer.theme = "dark"

    # SuRVoS controls
    dw = AttrDict()
    dw.ppw = PluginPanelWidget()  # Main SuRVoS panel
    dw.bpw = ButtonPanelWidget()  # Additional controls
    dw.ppw.setMinimumSize(QSize(400, 600))

    ws = Workspace(DataModel.g.current_workspace)
    dw.ws = ws
    dw.datamodel = DataModel.g
    dw.Launcher = Launcher

    def set_paint_params(msg):
        logger.debug(f"set_paint_params {msg['paint_params']}")
        paint_params = msg["paint_params"]
        label_value = paint_params["label_value"]

        if label_value is not None and len(viewer.layers.selection) > 0:
            _set_params(label_value, paint_params)

    def _set_params(label_value, paint_params):
        anno_layer = viewer.layers.selection
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

    def update_annotation_layer(layer_name, src_arr):
        existing_layer = [v for v in viewer.layers if v.name == layer_name]
        if existing_layer:
            existing_layer[0].data = src_arr.astype(np.int32) & 15

    def update_annotations(msg):
        logger.debug(f"refresh_annotation {msg}")
        src = DataModel.g.dataset_uri(msg["level_id"], group="annotations")
        with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
            src_annotations_dataset = DM.sources[0][:]
            src_arr = get_array_from_dataset(src_annotations_dataset)
        update_annotation_layer(msg["level_id"], src_arr)

    def refresh_annotations(msg):
        logger.debug(f"refresh_annotation {msg['level_id']}")
        try:
            cfg.current_annotation_name = msg["level_id"]

            src_arr, src_annotations_dataset = get_annotation_array(
                msg, retrieval_mode=cfg.retrieval_mode
            )
            result = Launcher.g.run(
                "annotations", "get_levels", workspace=DataModel.g.current_workspace
            )

            if cfg.label_ids is not None:
                label_ids = cfg.label_ids

            if result:
                return _refresh_annotations(
                    result, msg, src_arr, src_annotations_dataset
                )

        except Exception as e:
            print(f"Exception {e}")

    def _refresh_annotations(result, msg, src_arr, src_annotations_dataset):
        cmapping, label_ids = get_color_mapping(result, msg["level_id"])
        logger.debug(f"Label ids {label_ids}")
        cfg.label_ids = label_ids
        existing_layer = [v for v in viewer.layers if v.name == msg["level_id"]]
        sel_label = 1
        brush_size = 10

        if existing_layer:
            viewer.layers.remove(existing_layer[0])
            sel_label = existing_layer[0].selected_label
            brush_size = existing_layer[0].brush_size
            logger.debug(f"Removed existing layer {existing_layer[0]}")
        label_layer = viewer.add_labels(
            src_arr & 15, name=msg["level_id"], color=cmapping
        )
        label_layer.mode = cfg.current_mode
        label_layer.brush_size = brush_size

        if cfg.label_value is not None:
            label_layer.selected_label = int(cfg.label_value["idx"]) - 1

        return label_layer, src_annotations_dataset

    def setup_paint_undo(label_layer):
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

    def setup_view_location(label_layer, msg, parent_level, parent_label_idx):
        @label_layer.mouse_drag_callbacks.append
        def view_location(layer, event):
            drag_pts = []
            coords = np.round(layer.world_to_data(viewer.cursor.position)).astype(
                np.int32
            )
            if cfg.retrieval_mode == "slice":
                drag_pt = [coords[0], coords[1]]
            elif cfg.retrieval_mode == "volume":
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
                    elif cfg.retrieval_mode == "volume":
                        drag_pt = [coords[0], coords[1], coords[2]]
                    drag_pts.append(drag_pt)
                    yield

                if len(drag_pts) >= 0:
                    top_layer = viewer.layers[-1]
                    layer_name = top_layer.name  # get last added layer name
                    anno_layer = next(l for l in viewer.layers if l.name == layer_name)

                    def update_anno(msg):
                        src_arr, _ = get_annotation_array(
                            msg, retrieval_mode=cfg.retrieval_mode
                        )
                        update_annotation_layer(msg["level_id"], src_arr)

                    update = partial(update_anno, msg=msg)
                    viewer_order = viewer.window.qt_viewer.viewer.dims.order
                    paint_strokes_worker = paint_strokes(
                        msg,
                        drag_pts,
                        layer,
                        top_layer,
                        anno_layer,
                        parent_level,
                        parent_label_idx,
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

    def paint_annotations(msg):
        logger.debug(f"paint_annotation {msg['level_id']}")
        cfg.current_annotation_name = msg["level_id"]

        try:
            label_layer, current_annotation_ds = refresh_annotations(msg)
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
                setup_paint_undo(label_layer)
                setup_view_location(label_layer, msg, parent_level, parent_label_idx)

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
        from survos2.entity.entities import load_entities_via_file
        print(entities_arr.shape)
        #load_entities_via_file(entities_arr)
        result = Launcher.g.post_array(entities_arr, 
        group='objects', 
        workspace=DataModel.g.current_workspace, 
        name="objects")

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
        #params = dict(level=selected_layer.name, workspace=True)

        # add level
        result = Launcher.g.run("annotations", "add_level", workspace=True)
        new_level_id = result["id"]
        print(f"Added new level: {new_level_id}")
        # add labels to newly created level
        label_values = np.unique(selected_layer.data)
        print(f"Label values: {label_values}")
        for v in label_values:
            print(f"Label value to add {v}")
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
        print(levels_result)
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
            #dst = DataModel.g.dataset_uri(fid, group="annotations")

        #with DatasetManager(dst, out=dst, dtype="uint32", fillvalue=0) as DM:
        #    DM.out[:] = selected_layer.data
            print(selected_layer.data.shape)
            result = Launcher.g.post_array(selected_layer.data, 
            group='annotations', 
            workspace=DataModel.g.current_workspace, 
            name=result["id"])

    def save_annotation(msg):
        annotation_layer = [
            v for v in viewer.layers if v.name == cfg.current_annotation_name
        ]
        if len(annotation_layer) == 1:
            logger.debug(
                f"Updating annotation {cfg.current_annotation_name} with label image {annotation_layer}"
            )
            params = dict(level=cfg.current_annotation_name, workspace=True)
            result = Launcher.g.run("annotations", "get_levels", **params)[0]

            if result:
                _extracted_from_save_annotation(result, annotation_layer)
        else:
            logger.debug("save_annotation couldn't find annotation in viewer")

    def _extracted_from_save_annotation(result, annotation_layer):
        fid = result["id"]
        ftype = result["kind"]
        fname = result["name"]
        logger.debug(f"Transfering to workspace {fid}, {ftype}, {fname}")

        dst = DataModel.g.dataset_uri(fid, group="annotations")

        with DatasetManager(dst, out=dst, dtype="uint32", fillvalue=0) as DM:
            DM.out[:] = annotation_layer[0].data


    def jump_to_slice(msg):
        logger.debug(f"jump_to_slice Using order {cfg.order}")
        if cfg.retrieval_mode == "slice":
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
                    src_arr = decode_numpy(result)
                    existing_feature_layer[0].data = src_arr.copy()

            existing_regions_layer = [
                v for v in viewer.layers if v.name == cfg.current_regions_name
            ]
            if existing_regions_layer:
                regions_src = DataModel.g.dataset_uri(
                    cfg.current_regions_name, group="regions"
                )
                params = dict(
                    workpace=True,
                    src=regions_src,
                    slice_idx=cfg.current_slice,
                    order=cfg.order,
                )
                result = Launcher.g.run("regions", "get_slice", **params)
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
        else:
            logger.debug("jump_to_slice but in volume mode")

    def goto_roi(msg):
        logger.debug(f"Goto roi: {msg}")
        params = dict(
            workspace=True,
            current_workspace_name=DataModel.g.current_workspace,
            feature_id="001_raw",
            roi=msg["roi"],
        )
        result = Launcher.g.run("workspace", "goto_roi", **params)
        if result:
            logger.debug(f"Switching to goto_roi created workspace {result}")
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
                _jump_to_slice()
            else:
                logger.debug(f"In slice mode changing to volume mode {viewer.layers}")
                cfg.retrieval_mode = "volume"
                for _ in range(len(viewer.layers)):
                    viewer.layers.pop(0)
                view_feature(viewer, {"feature_id": cfg.current_feature_name})
        elif msg["data"] == "refesh_annotations":
            refresh_annotations(msg)
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
            for l in viewer.layers:
                viewer.layers.remove(l)
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
        elif msg["data"] == "goto_roi":
            goto_roi(msg)
        elif msg["data"] == "transfer_layer":
            transfer_layer(msg)

    def _jump_to_slice():
        cfg.retrieval_mode = "slice"
        for l in viewer.layers:
            viewer.layers.remove(l)
        viewer_order = viewer.window.qt_viewer.viewer.dims.order
        if len(viewer_order) == 3:
            cfg.order = [int(d) for d in viewer_order]
            logger.debug(f"Setting order to {cfg.order}")
        else:
            cfg.order = [0, 1, 2]
            logger.debug(f"Resetting order to {cfg.order}")
        cfg.slice_max = cfg.base_dataset_shape[cfg.order[0]]
        logger.debug(f"Setting slice max to {cfg.slice_max}")
        view_feature(viewer, {"feature_id": cfg.current_feature_name})
        jump_to_slice({"frame": 0})

    dw.ppw.clientEvent.connect(lambda x: processEvents(x))
    cfg.ppw = dw.ppw
    cfg.processEvents = processEvents
    dw.Config = Config
   
    return dw
