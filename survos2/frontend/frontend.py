import ntpath
import os
import sys
import threading
import time
from enum import Enum
from typing import List
from functools import partial
import napari
import numpy
import numpy as np
import seaborn as sns
import yaml
from loguru import logger
from matplotlib.colors import Normalize
import tempfile

from napari.layers import Image
from napari.qt.threading import thread_worker

from qtpy import QtCore
from qtpy.QtCore import QSize, QThread, QTimer
from qtpy.QtWidgets import QApplication, QPushButton, QTabWidget, QVBoxLayout, QWidget

from scipy import ndimage
from skimage import img_as_float, img_as_ubyte
from skimage.draw import line
from skimage.segmentation import find_boundaries
from skimage.morphology import disk
from scipy.ndimage import binary_dilation

from survos2 import survos
from survos2.entity.entities import make_entity_df
from survos2.entity.sampler import crop_vol_and_pts_centered, sample_roi
from survos2.frontend.components.entity import (
    SmallVolWidget,
    TableWidget,
    setup_entity_table,
)
from survos2.frontend.control.launcher import Launcher
from survos2.frontend.panels import ButtonPanelWidget, PluginPanelWidget

# from survos2.frontend.panels_magicgui import save_annotation_gui, workspace_gui
from survos2.frontend.utils import (
    coords_in_view,
    get_array_from_dataset,
    get_color_mapping,
    hex_string_to_rgba,
)
from survos2.helpers import AttrDict, simple_norm
from survos2.improc.utils import DatasetManager
from survos2.model import DataModel, Workspace
from survos2.server.state import cfg
from survos2.utils import decode_numpy
from survos2.frontend.paint_strokes import paint_strokes
from survos2.frontend.workflow import run_workflow
from napari_plugin_engine import napari_hook_implementation
from qtpy.QtWidgets import QWidget
from napari_plugin_engine import napari_hook_implementation


def get_annotation_array(msg, retrieval_mode='volume'):
    if retrieval_mode == 'slice':  # get a slice over http
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
    elif retrieval_mode == 'volume':  # get entire volume
        src = DataModel.g.dataset_uri(msg["level_id"], group="annotations")
        with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
            src_annotations_dataset = DM.sources[0][:]
            src_arr = get_array_from_dataset(src_annotations_dataset)

    return src_arr, src_annotations_dataset


def frontend():
    logger.info(f"Frontend loading workspace {DataModel.g.current_workspace}")

    DataModel.g.current_session = "default"

    src = DataModel.g.dataset_uri("__data__", None)
    dst = DataModel.g.dataset_uri("001_raw", group="features")

    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        logger.debug(f"Workspace has data of shape {DM.sources[0].shape}")
        orig_dataset = DM.sources[0]

    src = DataModel.g.dataset_uri("__data__")

    params = dict(workspace=DataModel.g.current_workspace)
    cfg.sessions = Launcher.g.run("workspace", "list_sessions", **params)

    # Frontend state params
    cfg.current_mode = "paint"
    cfg.label_ids = [
        0,
    ]
    cfg.retrieval_mode = "volume"  # volume | slice | crop
    cfg.current_slice = 0
    cfg.current_orientation = 0
    cfg.base_dataset_shape = orig_dataset.shape
    cfg.slice_max = orig_dataset.shape[0]

    cfg.current_feature_name = None
    cfg.current_annotation_name = None
    cfg.current_pipeline_name = None
    cfg.object_scale = 1.0
    cfg.object_offset = (0, 0, 0)  # (-16, 170, 165)  # (-16,-350,-350)
    cfg.current_regions_name = None
    cfg.current_supervoxels = None
    cfg.supervoxels_cache = None
    cfg.supervoxels_cached = False
    cfg.current_regions_dataset = None

    label_dict = {
        "level": "001_level",
        "idx": 1,
        "color": "#000000",
    }
    cfg.label_value = label_dict

    cfg.order = (0, 1, 2)
    cfg.group = "main"

    with napari.gui_qt():
        viewer = napari.Viewer(title="SuRVoS - " + DataModel.g.current_workspace)
        viewer.theme = "dark"
        viewer.window._qt_window.setGeometry(100, 200, 1280, 720)
        
        # SuRVoS controls
        dw = AttrDict()
        dw.ppw = PluginPanelWidget() # Main SuRVoS panel
        dw.bpw = ButtonPanelWidget() # Additional controls
        dw.ppw.setMinimumSize(QSize(400, 600))

        ws = Workspace(DataModel.g.current_workspace)
        dw.ws = ws
        dw.datamodel = DataModel.g
        dw.Launcher = Launcher

        def remove_layer(layer_name):
            logger.debug(f"Removing layer {layer_name}")
            existing_layer = [v for v in viewer.layers if v.name == layer_name]
            if len(existing_layer) > 0:
                viewer.layers.remove(existing_layer[0])

        def view_feature(msg, new_name=None):
            logger.debug(f"view_feature {msg['feature_id']}")
            existing_feature_layer = [
                v for v in viewer.layers if v.name == msg["feature_id"]
            ]

            if cfg.retrieval_mode == "slice":
                features_src = DataModel.g.dataset_uri(
                    msg["feature_id"], group="features"
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

                    if len(existing_feature_layer) > 0:
                        existing_feature_layer[0].data = src_arr.copy()
                    else:
                        if new_name:
                            viewer.add_image(src_arr, name=new_name)
                        else:
                            viewer.add_image(src_arr, name=msg["feature_id"])

            elif cfg.retrieval_mode == "volume":
                # use DatasetManager to load feature from workspace as array and then add it to viewer
                src = DataModel.g.dataset_uri(msg["feature_id"], group="features")
                remove_layer(cfg.current_feature_name)
                cfg.current_feature_name = msg["feature_id"]

                with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
                    src_dataset = DM.sources[0][:]
                    src_arr = get_array_from_dataset(src_dataset)
                    cfg.supervoxels_cache = src_arr
                    cfg.supervoxels_cached = True
                    if new_name:
                        viewer.add_image(src_arr, name=new_name)
                    else:
                        viewer.add_image(src_arr, name=msg["feature_id"])
        def view_regions(msg):
            logger.debug(f"view_feature {msg['region_id']}")
            region_name = msg["region_id"]
            existing_regions_layer = [
                v for v in viewer.layers if v.name == cfg.current_regions_name
            ]
            
            region_opacity = 0.3
            if len(existing_regions_layer) > 0:
                region_opacity = existing_regions_layer[0].opacity
                remove_layer(cfg.current_regions_name)
                cfg.current_regions_name = None


            if cfg.retrieval_mode == "slice":                
                regions_src = DataModel.g.dataset_uri(region_name, group="regions")
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
                    if len(existing_regions_layer) > 0:
                        existing_regions_layer[0].data = src_arr.copy()
                        existing_regions_layer[0].opacity = region_opacity
                    else:
                        sv_layer = viewer.add_image(src_arr, name=region_name)
                        sv_layer.opacity = region_opacity
                        sv_layer.colormap = "bop orange"
            elif cfg.retrieval_mode == "volume":
                src = DataModel.g.dataset_uri(region_name, group="regions")
                with DatasetManager(src, out=None, dtype="uint32", fillvalue=0) as DM:
                    src_dataset = DM.sources[0][:]
                    src_arr = get_array_from_dataset(src_dataset)
                    existing_layer = [v for v in viewer.layers if v.name == region_name]

                    if len(existing_layer) > 0:
                        existing_layer[0].data = src_arr
                    else:
                        sv_image = find_boundaries(src_arr, mode="inner")
                        sv_layer = viewer.add_image(sv_image, name=region_name)
                        sv_layer.opacity = region_opacity
                        sv_layer.colormap = "bop orange"

            cfg.current_regions_name = region_name
            cfg.supervoxels_cached = False

        def set_paint_params(msg):
            logger.debug(f"set_paint_params {msg['paint_params']}")
            paint_params = msg["paint_params"]
            label_value = paint_params["label_value"]

            if label_value is not None:
                if len(viewer.layers.selected) > 0:
                    anno_layer = viewer.layers.selected[0]
                    cfg.label_ids = list(np.unique(anno_layer))
                    anno_layer.mode = "paint"
                    cfg.current_mode = "paint"
                    anno_layer.selected_label = int(label_value["idx"]) - 1
                    cfg.label_value = label_value
                    anno_layer.brush_size = int(paint_params["brush_size"])
                    if paint_params["current_supervoxels"] is not None:
                        view_regions(
                            {
                                "region_id": ntpath.basename(
                                    paint_params["current_supervoxels"]
                                )
                            }
                        )
                        cfg.supervoxels_cached = False

        def update_annotation_layer(layer_name, src_arr):
            existing_layer = [v for v in viewer.layers if v.name == layer_name]
            if len(existing_layer) > 0:
                existing_layer[0].data = src_arr.astype(np.int) & 15

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
                    cmapping, label_ids = get_color_mapping(result, msg["level_id"])
                    logger.debug(f"Label ids {label_ids}")
                    cfg.label_ids = label_ids

                    existing_layer = [v for v in viewer.layers if v.name == msg["level_id"]]
                    sel_label = 1
                    brush_size = 10

                    if len(existing_layer) > 0:
                        viewer.layers.remove(existing_layer[0])
                        sel_label = existing_layer[0].selected_label
                        brush_size = existing_layer[0].brush_size
                        logger.debug(f"Removed existing layer {existing_layer[0]}")
                        label_layer = viewer.add_labels(
                            src_arr & 15, name=msg["level_id"], color=cmapping
                        )
                    else:
                        label_layer = viewer.add_labels(
                            src_arr & 15, name=msg["level_id"], color=cmapping
                        )

                    label_layer.mode = cfg.current_mode
                    label_layer.brush_size = brush_size

                    if cfg.label_value is not None:
                        label_layer.selected_label = int(cfg.label_value["idx"]) - 1

                    return label_layer, src_annotations_dataset
            except Exception as e:
                print(f"Exception {e}")

        def paint_annotations(msg):
            logger.debug(f"paint_annotation {msg['level_id']}")
            cfg.current_annotation_name = msg["level_id"]
            
            try:
                label_layer, current_annotation_ds = refresh_annotations(msg)
                if cfg.label_value is not None:
                    sel_label = int(cfg.label_value["idx"])
                else:
                    sel_label = 1

                if msg["level_id"] is not None:
                    params = dict(
                        workspace=True, level=msg["level_id"], label_idx=sel_label
                    )
                    result = Launcher.g.run("annotations", "get_label_parent", **params)
                    
                    parent_level = result[0]
                    parent_label_idx = result[1]

                    @label_layer.bind_key("Control-Z", overwrite=True)
                    def undo(v):
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

                    @label_layer.mouse_drag_callbacks.append
                    def view_location(layer, event):
                        drag_pts = []
                        coords = np.round(layer.coordinates).astype(int)

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
                                coords = np.round(layer.coordinates).astype(int)

                                if cfg.retrieval_mode == "slice":
                                    drag_pt = [coords[0], coords[1]]
                                elif cfg.retrieval_mode == "volume":
                                    drag_pt = [coords[0], coords[1], coords[2]]

                                drag_pts.append(drag_pt)
                                yield

                            if len(drag_pts) >= 0:
                                top_layer = viewer.layers[-1]
                                layer_name = top_layer.name  # get last added layer name
                                anno_layer = next(
                                    l for l in viewer.layers if l.name == layer_name
                                )

                                def update_anno(msg):
                                    src_arr, _ = get_annotation_array(
                                        msg, retrieval_mode=cfg.retrieval_mode
                                    )
                                    update_annotation_layer(msg["level_id"], src_arr)

                                update = partial(update_anno, msg=msg)
                                paint_strokes_worker = paint_strokes(
                                    msg,
                                    drag_pts,
                                    layer,
                                    top_layer,
                                    anno_layer,
                                    parent_level,
                                    parent_label_idx,
                                )
                                paint_strokes_worker.returned.connect(update)
                                paint_strokes_worker.start()
                            else:
                                cfg.ppw.clientEvent.emit(
                                    {
                                        "source": "annotations",
                                        "data": "update_annotations",
                                        "level_id": msg["level_id"],
                                    }
                                )
            except Exception as e:
                print(f"Exception: {e}")


        def view_pipeline(msg):
            logger.debug(f"view_pipeline {msg['pipeline_id']} using {msg['level_id']}")

            result = Launcher.g.run(
                "annotations", "get_levels", workspace=DataModel.g.current_workspace
            )
            logger.debug(f"Result of annotations get_levels: {result}")

            if result:
                cmapping, _ = get_color_mapping(result, level_id=msg["level_id"])

            existing_pipeline_layer = [
                v for v in viewer.layers if v.name == msg["pipeline_id"]
            ]
            cfg.current_pipeline_name = msg["pipeline_id"]

            if cfg.retrieval_mode == "slice":
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
                    src_arr = decode_numpy(result)
                    if len(existing_pipeline_layer) > 0:
                        existing_pipeline_layer[0].data = src_arr.copy()
                    else:
                        viewer.add_labels(
                            src_arr.astype(np.uint32),
                            name=msg["pipeline_id"],
                            color=cmapping,
                        )
            elif cfg.retrieval_mode == "volume":
                src = DataModel.g.dataset_uri(msg["pipeline_id"], group="pipelines")
                with DatasetManager(src, out=None, dtype="uint32", fillvalue=0) as DM:
                    src_dataset = DM.sources[0][:]
                    src_arr = get_array_from_dataset(src_dataset)

                    existing_layer = [
                        v for v in viewer.layers if v.name == msg["pipeline_id"]
                    ]

                    if len(existing_layer) > 0:
                        logger.debug(
                            f"Removing existing layer and re-adding it with new colormapping. {existing_layer}"
                        )

                        viewer.layers.remove(existing_layer[0])
                        viewer.add_labels(
                            src_arr.astype(np.uint32),
                            name=msg["pipeline_id"],
                            color=cmapping,
                        )
                    else:
                        viewer.add_labels(
                            src_arr.astype(np.uint32),
                            name=msg["pipeline_id"],
                            color=cmapping,
                        )

        def view_objects(msg):
            logger.debug(f"view_objects {msg['objects_id']}")
            src = DataModel.g.dataset_uri(msg["objects_id"], group="objects")
            with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
                ds_objects = DM.sources[0]

            logger.debug(f"Using dataset {ds_objects}")
            objects_fullname = ds_objects.get_metadata("fullname")
            objects_scale = ds_objects.get_metadata("scale")
            objects_offset = ds_objects.get_metadata("offset")
            objects_crop_start = ds_objects.get_metadata("crop_start")
            objects_crop_end = ds_objects.get_metadata("crop_end")

            logger.info(f"Viewing entities {objects_fullname}")
            tabledata, entities_df = setup_entity_table(
                objects_fullname,
                scale=objects_scale,
                offset=objects_offset,
                crop_start=objects_crop_start,
                crop_end=objects_crop_end,
            )
            sel_start, sel_end = 0, len(entities_df)

            centers = np.array(
                [
                    [
                        np.int((np.float(entities_df.iloc[i]["z"]) * 1.0) + 0),
                        np.int((np.float(entities_df.iloc[i]["x"]) * 1.0) + 0),
                        np.int((np.float(entities_df.iloc[i]["y"]) * 1.0) + 0),
                    ]
                    for i in range(sel_start, sel_end)
                ]
            )

            num_classes = max(9, len(np.unique(entities_df["class_code"]))) + 2
            logger.debug(f"Number of entity classes {num_classes}")
            palette = np.array(sns.color_palette("hls", num_classes))
            face_color_list = [
                palette[class_code] for class_code in entities_df["class_code"]
            ]

            entity_layer = viewer.add_points(
                centers,
                size=[10] * len(centers),
                opacity=0.5,
                face_color=face_color_list,
                n_dimensional=True,
            )

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
                    fid = result["id"]
                    ftype = result["kind"]
                    fname = result["name"]
                    logger.debug(f"Transfering to workspace {fid}, {ftype}, {fname}")

                    dst = DataModel.g.dataset_uri(fid, group="annotations")

                    with DatasetManager(
                        dst, out=dst, dtype="uint32", fillvalue=0
                    ) as DM:
                        DM.out[:] = annotation_layer[0].data
            else:
                logger.debug("save_annotation couldn't find annotation in viewer")

        def set_session(msg):
            logger.debug(f"Set session to {msg['session']}")
            DataModel.g.current_session = msg["session"]

        def transfer_layer(msg):
            logger.debug(f"transfer_layer {msg}")
            selected_layer = viewer.layers.selected[0]
            from napari.layers.image.image import Image
            from napari.layers.points.points import Points
            from napari.layers.labels.labels import Labels
            print(type(selected_layer))

            if isinstance(selected_layer, Labels):
                logger.debug("Transferring Labels layer to Annotations.")
                params = dict(level=selected_layer.name, workspace=True)
                result = Launcher.g.run("annotations", "add_level", workspace=True)
                label_values = np.unique(selected_layer.data)
                print(f"Label values: {label_values}")
                for v in label_values:
                    print(f"Label value to add {v}")
                    if v != 0:
                        params = dict(
                        level=result["id"],
                        idx=int(v)-2,
                        name=str(int(v)-2),
                        color="#11FF11",
                        workspace=True,
                        )
                        label_result = Launcher.g.run("annotations", "add_label", **params)
                        
                levels_result = Launcher.g.run("annotations", "get_levels", **params)[0]
    
                for v in levels_result["labels"].keys():
                    label_rgba = np.array(selected_layer.get_color(int(v)-1))
                    label_rgba = (255 * label_rgba).astype(np.uint8)
                    label_hex = "#{:02x}{:02x}{:02x}".format(*label_rgba)
                    label = dict(
                        idx=int(v),
                        name=str(int(v)-1),
                        color=label_hex,
                    )
                    params = dict(level=result["id"], workspace=True)
                    label_result = Launcher.g.run(
                        "annotations", "update_label", **params, **label)
                
                    if levels_result:
                        fid = result["id"]
                        ftype = result["kind"]
                        fname = result["name"]
                        logger.debug(f"Created new object in workspace {fid}, {ftype}, {fname}")
                        dst = DataModel.g.dataset_uri(fid, group="annotations")

                    with DatasetManager(dst, out=dst, dtype="uint32", fillvalue=0) as DM:
                        DM.out[:] = selected_layer.data

            elif isinstance(selected_layer, Points):
                logger.debug("Transferring Points layer to Objects.")
                entities_arr = np.concatenate((selected_layer.data, np.zeros((len(selected_layer.data),1))), axis=1)
                
                entities_df = make_entity_df(entities_arr, flipxy=True)
                
                tmp_fullpath = os.path.abspath(os.path.join(tempfile.gettempdir(), os.urandom(24).hex() + ".csv"))
                print(tmp_fullpath)

                entities_df.to_csv(tmp_fullpath, line_terminator="")
                
                object_scale = 1.0
                object_offset = (0.0,0.0,0.0)
                object_crop_start = (0.0,0.0,0.0)
                object_crop_end = (1e5,1e5,1e5)
                
                params = dict(
                    order=0, workspace=True, fullname=tmp_fullpath,
                )
                result = Launcher.g.run("objects", "create", **params)
                if result:
                    dst = DataModel.g.dataset_uri(result["id"], group="objects")
                    params = dict(dst=dst, fullname=tmp_fullpath, scale=object_scale, 
                                  offset=object_offset, crop_start=object_crop_start, crop_end=object_crop_end)
                    logger.debug(f"Getting objects with params {params}")
                    Launcher.g.run("objects", "points", **params)

                os.remove(tmp_fullpath)
       

            elif isinstance(selected_layer, Image):
                logger.debug("Transferring Image layer to Features.")
                params = dict(feature_type="raw", workspace=True)                
                result = Launcher.g.run("features", "create", **params)
                if result:
                    fid = result["id"]
                    ftype = result["kind"]
                    fname = result["name"]
                    logger.debug(f"Created new object in workspace {fid}, {ftype}, {fname}")
                    dst = DataModel.g.dataset_uri(fid, group="features")
                    with DatasetManager(dst, out=dst, dtype="float32", fillvalue=0) as DM:
                        DM.out[:] = selected_layer.data
            else:
                logger.debug("Unsupported layer type.")
   
            processEvents({'data':'refresh'})


        def jump_to_slice(msg):
            logger.debug(f"Using order {cfg.order}")
            if cfg.retrieval_mode == "slice":
                cfg.current_slice = int(msg["frame"])
                logger.debug(f"Jump around to {cfg.current_slice}, msg {msg['frame']}")

                print(cfg.current_feature_name)                
                existing_feature_layer = [
                    v for v in viewer.layers if v.name == cfg.current_feature_name
                ]

                if len(existing_feature_layer) > 0:
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
                if len(existing_regions_layer) > 0:
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
                if len(existing_level_layer) > 0:
                    if cfg.current_annotation_name is not None:
                        paint_annotations({"level_id": cfg.current_annotation_name})

                existing_pipeline_layer = [
                    v for v in viewer.layers if v.name == cfg.current_pipeline_name
                ]

                if len(existing_pipeline_layer) > 0:
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
                        src_arr = decode_numpy(result)
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
            selected_roi_idx = cfg.entity_table.w.selected_row
            logger.info(f"Showing ROI {msg['selected_roi']}")

            existing_feature_layer = [
                v for v in viewer.layers if v.name == cfg.current_feature_name
            ]

            vol1 = sample_roi(
                existing_feature_layer[0].data,
                cfg.tabledata,
                selected_roi_idx,
                vol_size=(32, 32, 32),
            )

            # viewer.dw.smallvol_control.set_vol(np.transpose(vol1, (0,2,1)))
            cfg.smallvol_control.set_vol(vol1)
            logger.info(f"Sampled ROI vol of shape: {vol1.shape}")


        def processEvents(msg):
            if msg["data"] == "refesh_annotations":
                refresh_annotations(msg)
            elif msg["data"] == "paint_annotations":
                paint_annotations(msg)
            elif msg["data"] == "update_annotations":
                update_annotations(msg)
            elif msg["data"] == "remove_layer":
                layer_name = msg["layer_name"]
                remove_layer(layer_name)
            elif msg["data"] == "view_feature":
                view_feature(msg)
            elif msg["data"] == "view_pipeline":
                view_pipeline(msg)
            elif msg["data"] == "view_regions":
                view_regions(msg)
            elif msg["data"] == "view_objects":
                view_objects(msg)
            elif msg["data"] == "show_roi":
                show_roi(msg)
            elif msg["data"] == "run_workflow":
                run_workflow_worker = run_workflow(msg)
                run_workflow_worker.start()
                processEvents({"data": "refresh"})
            elif msg["data"] == "refresh":
                logger.debug("Refreshing plugin panel")
                dw.ppw.setup()
            elif msg["data"] == "save_annotation":
                save_annotation(msg)
            elif msg["data"] == "set_paint_params":
                set_paint_params(msg)
            elif msg["data"] == "jump_to_slice":
                jump_to_slice(msg)
            elif msg["data"] == "set_session":
                set_session(msg)
            elif msg["data"] == "get_crop":
                get_crop(msg)
            elif msg["data"] == "goto_roi":
                goto_roi(msg)
            elif msg["data"] == "transfer_layer":
                transfer_layer(msg)
            elif msg["data"] == "slice_mode":
                logger.debug(f"Slice mode: {cfg.retrieval_mode}")
                if cfg.retrieval_mode != "slice":
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
                    view_feature({"feature_id": "001_raw"})
                    jump_to_slice({"frame": 0})
                elif cfg.retrieval_mode == "slice":
                    cfg.retrieval_mode = "volume"
                    for l in viewer.layers:
                        viewer.layers.remove(l)
                    view_feature({"feature_id": "001_raw"})

        #
        # Add widgets to viewer
        #
        # Plugin Panel widget
        dw.ppw.clientEvent.connect(lambda x: processEvents(x))
        cfg.ppw = dw.ppw
        cfg.processEvents = processEvents
        pluginwidget_dockwidget = viewer.window.add_dock_widget(
            dw.ppw,
            area="right",
            name="Workspace"
        )
        pluginwidget_dockwidget.setWindowTitle("Workspace")
        #view_feature({"feature_id": "001_raw"})

        view_feature({"feature_id": "001_raw"}, new_name="Raw")


        smallvol_control = SmallVolWidget(np.zeros((32, 32, 32)))
        cfg.smallvol_control = smallvol_control
        smallvol_control_dockwidget = viewer.window.add_dock_widget(
            smallvol_control.imv, area="right",
            name="Patch viewer"
        )
        smallvol_control_dockwidget.setVisible(False)
        smallvol_control_dockwidget.setWindowTitle("Patch viewer")

        #Button panel (beta panel)
        bpw_control_dockwidget = viewer.window.add_dock_widget(dw.bpw, area="left", name="Beta menu")
        dw.bpw.clientEvent.connect(lambda x: processEvents(x))
        bpw_control_dockwidget.setVisible(False)
        bpw_control_dockwidget.setWindowTitle("Beta menu")

