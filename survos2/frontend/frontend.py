import os
import sys
import threading
import time
from enum import Enum
from typing import List
import ntpath
import napari
import numpy
import numpy as np
import seaborn as sns

import yaml
from loguru import logger
from magicgui import magicgui
from matplotlib.colors import Normalize
from napari.layers import Image
from pyqtgraph.widgets.ProgressDialog import ProgressDialog
from qtpy import QtCore
from qtpy.QtCore import QSize, QThread, QTimer
from qtpy.QtWidgets import (
    QApplication, QPushButton, QTabWidget, QVBoxLayout, QWidget)
from scipy import ndimage
from scipy.ndimage import binary_dilation
from skimage import img_as_float, img_as_ubyte
from skimage.draw import line
from skimage.morphology import disk
from skimage.segmentation import find_boundaries
from survos2.entity.entities import make_entity_df
from survos2.entity.sampler import crop_vol_and_pts_centered, sample_roi
from survos2.frontend.components.entity import (SmallVolWidget, TableWidget,
                                                setup_entity_table)
from survos2.frontend.control.launcher import Launcher
from survos2.model import Workspace

from survos2.frontend.panels import ButtonPanelWidget, PluginPanelWidget
#from survos2.frontend.panels_magicgui import save_annotation_gui, workspace_gui
from survos2.frontend.utils import (coords_in_view, get_color_mapping,
                                    hex_string_to_rgba)
from survos2.helpers import AttrDict, simple_norm
from survos2.improc.utils import DatasetManager
from survos2.model import DataModel
from survos2.server.state import cfg
from survos2.utils import decode_numpy
from survos2 import survos

def frontend():
    logger.info(
        f"Frontend loading workspace {DataModel.g.current_workspace}"
    )

    DataModel.g.current_session = "default"

    existing_features = survos.run_command('features', 'existing', uri=None, workspace=DataModel.g.current_workspace)[0]
    existing_feature_names= [f for f in existing_features.keys()]  
    
    if '001_raw' not in existing_feature_names:
        survos.run_command('features', 'create', uri=None, workspace=DataModel.g.current_workspace,                   
                        feature_type='raw')
    
    src = DataModel.g.dataset_uri('__data__', None)
    dst = DataModel.g.dataset_uri('001_raw', group='features')

    with DatasetManager(src, out=dst, dtype='float32', fillvalue=0) as DM:
        print(DM.sources[0].shape)
        orig_dataset = DM.sources[0]
        dst_dataset = DM.out
        src_arr = orig_dataset[:]
        dst_dataset[:] = src_arr

    src = DataModel.g.dataset_uri("__data__")

    # Entity loading
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        entity_dataset = DM.sources[0]
        entity_fullname = entity_dataset.get_metadata("entities_name")
    logger.info(f"entity_fullname {entity_fullname}")
    if entity_fullname is None:
        use_entities = False
    else:
        use_entities = True


    params = dict(workspace=DataModel.g.current_workspace)
    cfg.sessions = Launcher.g.run("workspace", "list_sessions", **params)

    # Frontend state params
    cfg.current_mode = "paint"
    cfg.label_ids = [
        0,
    ]
    cfg.slice_mode = False
    cfg.current_slice = 0
    cfg.current_orientation = 0
    cfg.base_dataset_shape = orig_dataset.shape
    cfg.slice_max = orig_dataset.shape[0]

    cfg.current_feature_name = None
    cfg.current_annotation_name = None
    cfg.current_pipeline_name = None

    cfg.current_regions_name = None
    cfg.current_supervoxels = None
    cfg.supervoxels_cache = None
    cfg.supervoxels_cached = False
    cfg.current_regions_dataset = None

    cfg.order = (0,1,2)

    cfg.group = "main"

    with napari.gui_qt():
        viewer = napari.Viewer(title="SuRVoS - " + DataModel.g.current_workspace )
        viewer.theme = "dark"
        viewer.window._qt_window.setGeometry(100, 200, 1280, 720)

        # SuRVoS controls
        viewer.dw = AttrDict()
        viewer.dw.ppw = PluginPanelWidget()
        viewer.dw.bpw = ButtonPanelWidget()
        viewer.dw.ppw.setMinimumSize(QSize(600, 500))

        # provide state variables to viewer for interactive debugging
        viewer.cfg = cfg
        ws = Workspace(DataModel.g.current_workspace)
        viewer.dw.ws = ws
        viewer.dw.datamodel = DataModel.g
        viewer.dw.Launcher = Launcher

        def remove_layer(layer_name):
            logger.debug(f"Removing layer {layer_name}")
            existing_layer = [v for v in viewer.layers if v.name == layer_name]
            if len(existing_layer) > 0:
                viewer.layers.remove(existing_layer[0])

        def view_feature(msg):
            logger.debug(f"view_feature {msg['feature_id']}")

            # use DatasetManager to load feature from workspace as array and then add it to viewer
            src = DataModel.g.dataset_uri(msg["feature_id"], group="features")
            remove_layer(cfg.current_feature_name)
            cfg.current_feature_name = msg["feature_id"]

            with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
                src_dataset = DM.sources[0][:]
                src_arr = get_array_from_dataset(src_dataset)
                
                existing_layer = [
                    v for v in viewer.layers if v.name == msg["feature_id"]
                ]
                cfg.supervoxels_cache = src_arr
                cfg.supervoxels_cached = True

                if len(existing_layer) > 0:
                    existing_layer[0].data = src_arr
                else:
                    viewer.add_image(src_arr, name=msg["feature_id"])

        def update_regions(region_name):
            logger.debug(f"update regions {region_name}")

            existing_regions_layer = [
                v for v in viewer.layers if v.name == cfg.current_regions_name
            ]

            remove_layer(cfg.current_regions_name)

            if cfg.slice_mode:
                #
                regions_src = DataModel.g.dataset_uri(region_name, group="regions")
                params = dict(workpace=True, src=regions_src, slice_idx=cfg.current_slice, order=cfg.order)
                result = Launcher.g.run("regions", "get_slice", **params)
                if result:
                    src_arr = decode_numpy(result)
                    src_arr = find_boundaries(src_arr) * 1.0
                    if len(existing_regions_layer) > 0:
                        existing_regions_layer[0].data = src_arr.copy()
                        existing_regions_layer[0].opacity = 0.3
                    else:
                        sv_layer = viewer.add_image(src_arr, name=region_name)
                        sv_layer.opacity = 0.3
            else:
                src = DataModel.g.dataset_uri(region_name, group="regions")
                with DatasetManager(src, out=None, dtype="uint32", fillvalue=0) as DM:
                    src_dataset = DM.sources[0][:]
                    src_arr = get_array_from_dataset(src_dataset)
                    existing_layer = [v for v in viewer.layers if v.name == region_name]

                    if len(existing_layer) > 0:
                        existing_layer[0].data = src_arr
                    else:
                        sv_image = find_boundaries(src_arr)
                        sv_layer = viewer.add_image(sv_image, name=region_name)
                        sv_layer.opacity = 0.3

            cfg.current_regions_name = region_name
            cfg.supervoxels_cached = False

        def view_regions(msg):
            logger.debug(f"view_feature {msg['region_id']}")
            update_regions(msg["region_id"])

        def set_paint_params(msg):
            logger.debug(f"set_paint_params {msg['paint_params']}")
            paint_params = msg["paint_params"]
            anno_layer = viewer.layers.selected[0]
            cfg.label_ids = list(np.unique(anno_layer))

            if anno_layer.dtype == "uint32":
                anno_layer.mode = "paint"
                cfg.current_mode = "paint"
                label_value = paint_params["label_value"]
                anno_layer.selected_label = int(label_value["idx"]) - 1
                cfg.label_value = label_value
                anno_layer.brush_size = int(paint_params["brush_size"])
                update_regions(ntpath.basename(paint_params["current_supervoxels"]))
                cfg.supervoxels_cached = False

        def update_layer(layer_name, src_arr):
            existing_layer = [v for v in viewer.layers if v.name == layer_name]
            if len(existing_layer) > 0:
                existing_layer[0].data = src_arr & 15

        def update_annotations(msg):
            logger.debug(f"refresh_annotation {msg}")

            src = DataModel.g.dataset_uri(msg["level_id"], group="annotations")
            with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
                src_annotations_dataset = DM.sources[0][:]
                src_arr = get_array_from_dataset(src_annotations_dataset)

            update_layer(msg["level_id"], src_arr)

        def get_annotation_array(msg):
            if cfg.slice_mode: # just get a slice over http
                src_annotations_dataset = DataModel.g.dataset_uri(msg["level_id"], group="annotations")
                params = dict(workpace=True, src=src_annotations_dataset, slice_idx=cfg.current_slice, order=cfg.order)
                result = Launcher.g.run("annotations", "get_slice", **params)
                if result:
                    src_arr = decode_numpy(result)
            else: # get entire volume
                src = DataModel.g.dataset_uri(msg["level_id"], group="annotations")
                with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
                    src_annotations_dataset = DM.sources[0][:]
                    src_arr = get_array_from_dataset(src_annotations_dataset)

            return src_arr, src_annotations_dataset

        def refresh_annotations(msg):
            logger.debug(f"refresh_annotation {msg['level_id']}")

            cfg.current_annotation_name = msg["level_id"]
            src_arr, src_annotations_dataset = get_annotation_array(msg)
            result = Launcher.g.run(
                "annotations", "get_levels", workspace=DataModel.g.current_workspace
            )

            if cfg.label_ids is not None:
                label_ids = cfg.label_ids

            if result:
                # label_ids = list(np.unique(src_arr))
                cmapping, label_ids = get_color_mapping(result)
                logger.debug(f"Label ids {label_ids}")
                cfg.label_ids = label_ids

                existing_layer = [v for v in viewer.layers if v.name == msg["level_id"]]
                sel_label = 1
                brush_size = 10

                if len(existing_layer) > 0:
                    viewer.layers.remove(existing_layer[0])
                    sel_label = existing_layer[0].selected_label
                    brush_size = existing_layer[0].brush_size
                    print(f"Removed existing layer {existing_layer[0]}")
                    label_layer = viewer.add_labels(
                        src_arr & 15, name=msg["level_id"], color=cmapping
                    )
                else:
                    label_layer = viewer.add_labels(
                        src_arr & 15, name=msg["level_id"], color=cmapping
                    )

                label_layer.mode = cfg.current_mode
                label_layer.selected_label = int(cfg.label_value["idx"]) - 1
                label_layer.brush_size = brush_size

                return label_layer, src_annotations_dataset


        def paint_strokes(msg, drag_pts, layer):
            level = msg["level_id"]
            layer_name = viewer.layers[-1].name  # get last added layer name
            anno_layer = next(
                l for l in viewer.layers if l.name == layer_name
            )

            sel_label = int(cfg.label_value["idx"]) - 1   
            anno_layer.selected_label = sel_label
            anno_layer.brush_size = int(cfg.brush_size)

            if layer.mode == "erase":
                sel_label = 0
                cfg.current_mode = "erase"
            else:
                cfg.current_mode = "paint"

            line_x = []
            line_y = []

            if cfg.slice_mode:
                px, py = drag_pts[0]
                z = cfg.current_slice
            else:
                z, px, py = drag_pts[0]

            # depending on the slice mode we need to handle either 2 or 3 coordinates
            if cfg.slice_mode:
                for x, y in drag_pts[1:]:
                    yy, xx = line(py, px, y, x)
                    line_x.extend(xx)
                    line_y.extend(yy)
                    py, px = y, x
            else:
                for _, x, y in drag_pts[1:]:
                    yy, xx = line(py, px, y, x)
                    line_x.extend(xx)
                    line_y.extend(yy)
                    py, px = y, x

            line_y = np.array(line_y)
            line_x = np.array(line_x)

            from survos2.frontend.plugins.annotations import \
                dilate_annotations

            all_regions = set()

            # Check if we are painting using supervoxels, if not, annotate voxels
            if cfg.current_supervoxels == None:
                line_y, line_x = dilate_annotations(
                    line_x,
                    line_y,
                    (anno_layer.data.shape[1], anno_layer.data.shape[2]),
                    viewer.layers[-1].brush_size,
                )

                params = dict(workspace=True, level=level, label=sel_label)
                yy,xx = list(line_y), list(line_x)
                yy = [int(e) for e in yy]
                xx = [int(e) for e in xx]

                params.update(slice_idx=int(z), yy=yy, xx=xx)
                result = Launcher.g.run(
                    "annotations", "annotate_voxels", **params
                )                
            # we are painting with supervoxels, so check if we have a current supervoxel cache
            # if not, get the supervoxels from the server
            else:
                if cfg.supervoxels_cached == False:
                    regions_dataset = DataModel.g.dataset_uri(
                        cfg.current_regions_name, group="regions"
                    )

                    with DatasetManager(
                        regions_dataset,
                        out=None,
                        dtype="uint32",
                        fillvalue=0,
                    ) as DM:
                        src_dataset = DM.sources[0]
                        sv_arr = src_dataset[:]

                    print(
                        f"Loaded superregion array of shape {sv_arr.shape}"
                    )
                    cfg.supervoxels_cache = sv_arr
                    cfg.supervoxels_cached = True
                    cfg.current_regions_dataset = regions_dataset
                else:
                    sv_arr = cfg.supervoxels_cache

                    print(
                        f"Used cached supervoxels of shape {sv_arr.shape}"
                    )

                for x, y in zip(line_x, line_y):
                    sv = sv_arr[z, x, y]
                    all_regions |= set([sv])

                print(f"Painted regions {all_regions}")

                # Commit annotation to server
                params = dict(workspace=True, level=level, label=sel_label)

                params.update(
                    region=cfg.current_regions_dataset,
                    r=list(map(int, all_regions)),
                    modal=False,
                )
                result = Launcher.g.run(
                    "annotations", "annotate_regions", **params
                )

            # Update client annotation
            src_arr, _ = get_annotation_array(msg)
            update_layer(msg["level_id"], src_arr)

        def paint_annotations(msg):
            logger.debug(f"view_annotation {msg['level_id']}")
            cfg.current_annotation_name = msg["level_id"]
            label_layer, current_annotation_ds = refresh_annotations(msg)

            @label_layer.bind_key("Control-Z", overwrite=True)
            def undo(v):
                level = cfg.current_annotation
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
                dragged = False
                drag_pts = []
                coords = np.round(layer.coordinates).astype(int)

                if cfg.slice_mode:
                    drag_pt = [coords[0], coords[1]]
                else:
                    drag_pt = [coords[0], coords[1], coords[2]]
                drag_pts.append(drag_pt)
                yield

                if layer.mode == "paint" or layer.mode == "erase":                    
                    while event.type == "mouse_move":
                        coords = np.round(layer.coordinates).astype(int)

                        if cfg.slice_mode:
                            drag_pt = [coords[0], coords[1]]
                        else:
                            drag_pt = [coords[0], coords[1], coords[2]]

                        drag_pts.append(drag_pt)
                        dragged = True
                        yield

                    paint_strokes(msg, drag_pts, layer)
        
        def get_array_from_dataset(src_dataset, axis=0):
            
            if cfg.slice_mode:
                print(f"src_dataset shape {src_dataset.shape}")
                dataset = src_dataset.copy(order='C')
                dataset = np.transpose(dataset, np.array(cfg.order)).astype(np.float32)
                src_arr = dataset[cfg.current_slice, :, :]
            else:
                src_arr = src_dataset[:]

            return src_arr

        def view_pipeline(msg):
            logger.debug(f"view_pipeline {msg['pipeline_id']}")
            
            result = Launcher.g.run(
                "annotations", "get_levels", workspace=DataModel.g.current_workspace
            )
            logger.debug(f"Result of annotations get_levels: {result}")
            if result:
                cmapping, _ = get_color_mapping(result)
                
            existing_pipeline_layer = [
                v for v in viewer.layers if v.name == msg['pipeline_id']
            ]
            cfg.current_pipeline_name = msg["pipeline_id"]

            if cfg.slice_mode:                
                pipeline_src = DataModel.g.dataset_uri(cfg.current_pipeline_name, group="pipeline")
                params = dict(workpace=True, src=pipeline_src, slice_idx=cfg.current_slice,  order=cfg.order)
                result = Launcher.g.run("features", "get_slice", **params)
                if result:
                    src_arr = decode_numpy(result)
                    if len(existing_pipeline_layer) > 0:
                        existing_pipeline_layer[0].data = src_arr.copy()
                    else:
                        viewer.add_labels(src_arr, name=msg["pipeline_id"], color=cmapping)
            else:
                src = DataModel.g.dataset_uri(msg["pipeline_id"], group="pipeline")
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
                        viewer.add_labels(src_arr, name=msg["pipeline_id"], color=cmapping)
                    else:
                        viewer.add_labels(src_arr, name=msg["pipeline_id"], color=cmapping)

        def view_objects(msg, scale=1.0):
            logger.debug(f"view_objects {msg['objects_id']}")
            dsname = "objects\\" + msg["objects_id"]
            ds = viewer.dw.ws.get_dataset(dsname)
            logger.debug(f"Using dataset {ds}")
            entities_fullname = ds.get_metadata("fullname")
            logger.info(f"Viewing entities {entities_fullname}")
            tabledata, entities_df = setup_entity_table(entities_fullname)
            sel_start, sel_end = 0, len(entities_df)

            centers = np.array(
                [
                    [
                        np.int(np.float(entities_df.iloc[i]["z"]) * scale),
                        np.int(np.float(entities_df.iloc[i]["x"]) * scale),
                        np.int(np.float(entities_df.iloc[i]["y"]) * scale),
                    ]
                    for i in range(sel_start, sel_end)
                ]
            )

            num_classes = len(np.unique(entities_df["class_code"])) + 2
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
                v for v in viewer.layers if v.name == cfg.current_annotation
            ]
            if len(annotation_layer) == 1:
                logger.debug(
                    f"Updating annotation in workspace {cfg.current_annotation} with label image {annotation_layer}"
                )
                params = dict(level=cfg.current_annotation, workspace=True)
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

        def show_roi(msg):
            selected_roi_idx = cfg.entity_table.w.selected_row
            logger.info(f"Showing ROI {msg['selected_roi']}")
            
            existing_feature_layer = [
                    v for v in viewer.layers 
                    if v.name == cfg.current_feature_name]
            
            vol1 = sample_roi(
                existing_feature_layer[0].data,
                cfg.tabledata,
                selected_roi_idx,
                vol_size=(32, 32, 32),
            )
            
            # viewer.dw.smallvol_control.set_vol(np.transpose(vol1, (0,2,1)))
            cfg.smallvol_control.set_vol(vol1)
            logger.info(f"Sampled ROI vol of shape: {vol1.shape}")

        def run_workflow(msg):
            workflow_file = msg["workflow_file"]
            if not os.path.isabs(workflow_file):
                fworkflows = os.path.join(os.getcwd(), workflow_file)
            with open(fworkflows) as f:
                workflows = yaml.safe_load(f.read())

            num_workflow_steps = len(workflows.keys())
            minVal, maxVal = 0, num_workflow_steps
            with ProgressDialog(
                f"Processing pipeline {workflow_file}", minVal, maxVal
            ) as dlg:

                if dlg.wasCanceled():
                    raise Exception("Processing canceled")

                for step_idx, k in enumerate(workflows):
                    workflow = workflows[k]
                    action = workflow.pop("action")
                    plugin, command = action.split(".")
                    params = workflow.pop("params")
                    
                    src_name = workflow.pop("src")
                    dst_name = workflow.pop("dst")

                    src = DataModel.g.dataset_uri(src_name, group=plugin)
                    dst = DataModel.g.dataset_uri(dst_name, group=plugin)

                    all_params = dict(src=src, dst=dst, modal=True)
                    all_params.update(params)
                    logger.info(f"Executing workflow {all_params}")

                    print(
                        f"Running {plugin}, {command} on {src}\n to dst {dst} {all_params}\n"
                    )

                    Launcher.g.run(plugin, command, **all_params)
                    dlg.setValue(step_idx)

            cfg.ppw.clientEvent.emit(
                {"source": "workspace_gui", "data": "refresh", "value": None}
            )

        def set_session(msg):
            logger.debug(f"Set session to {msg['session']}")
            DataModel.g.current_session = msg['session']

        def jump_to_slice(msg):            

            print(f"Using order {cfg.order}")
            if cfg.slice_mode:
                cfg.current_slice = int(msg["frame"])
                logger.debug(f"Jump around to {cfg.current_slice}, msg {msg['frame']}")

                existing_feature_layer = [
                    v for v in viewer.layers if v.name == cfg.current_feature_name
                ]

                if len(existing_feature_layer) > 0:                    
                    features_src = DataModel.g.dataset_uri(cfg.current_feature_name, group="features")
                    params = dict(workpace=True, src=features_src, slice_idx=cfg.current_slice, order=cfg.order)
                    print(params)
                    result = Launcher.g.run("features", "get_slice", **params)
                    if result:
                        src_arr = decode_numpy(result)
                        existing_feature_layer[0].data = src_arr.copy()

                existing_regions_layer = [
                    v for v in viewer.layers if v.name == cfg.current_regions_name
                ]
                if len(existing_regions_layer) > 0:
                    regions_src = DataModel.g.dataset_uri(cfg.current_regions_name, group="regions")
                    params = dict(workpace=True, src=regions_src, slice_idx=cfg.current_slice,  order=cfg.order)
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
                    pipeline_src = DataModel.g.dataset_uri(cfg.current_pipeline_name, group="pipeline")
                    params = dict(workpace=True, src=pipeline_src, slice_idx=cfg.current_slice, order=cfg.order)
                    result = Launcher.g.run("features", "get_slice", **params)
                    if result:
                        src_arr = decode_numpy(result)
                        existing_pipeline_layer[0].data = src_arr.copy()
            else:
                pass

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
                run_workflow(msg)
            elif msg["data"] == "refresh":
                logger.debug("Refreshing plugin panel")
                viewer.dw.ppw.setup()
            elif msg["data"] == "save_annotation":
                save_annotation(msg)
            elif msg["data"] == "set_paint_params":
                set_paint_params(msg)
            elif msg["data"] == "jump_to_slice":
                jump_to_slice(msg)
            elif msg["data"] == "set_session":
                set_session(msg)
            elif msg["data"] == "slice_mode":
                logger.debug(f"Slice mode: {cfg.slice_mode}")

                if not cfg.slice_mode:
                    for l in viewer.layers:
                        viewer.layers.remove(l)
                    viewer_order = viewer.window.qt_viewer.viewer.dims.order
                    print(len(viewer_order))
                    if len(viewer_order) == 3:
                        cfg.order = [int(d) for d in viewer_order]
                        print(f"Setting order to {cfg.order}")
                    else:
                        cfg.order = [0,1,2]
                        print(f"Resetting order to {cfg.order}")
                    cfg.slice_max = cfg.base_dataset_shape[cfg.order[0]]
                    print(f"Setting slice max to {cfg.slice_max}")

                    view_feature({'feature_id' : '001_raw'})        
                    jump_to_slice({'frame' : cfg.current_slice})
                else:
                    for l in viewer.layers:
                        viewer.layers.remove(l)
                    view_feature({'feature_id' : '001_raw'})        
   
                cfg.slice_mode = not cfg.slice_mode

        #
        # Add widgets to viewer
        #
        # Plugin Panel widget
        viewer.dw.ppw.clientEvent.connect(lambda x: processEvents(x))
        cfg.ppw = viewer.dw.ppw

        smallvol_control = SmallVolWidget(np.zeros((32, 32, 32)))
        cfg.smallvol_control = smallvol_control
        smallvol_control_dockwidget = viewer.window.add_dock_widget(
            smallvol_control.imv, area="right"
        )
        smallvol_control_dockwidget.setVisible(False)
        smallvol_control_dockwidget.setWindowTitle("Patch viewer")

        bpw_control_widget = viewer.window.add_dock_widget(viewer.dw.bpw, area="left")
        viewer.dw.bpw.clientEvent.connect(lambda x: processEvents(x))

        pluginwidget_dockwidget = viewer.window.add_dock_widget(
            viewer.dw.ppw,
            area="right",
        )
        pluginwidget_dockwidget.setWindowTitle("Workspace")
        view_feature({'feature_id' : '001_raw'})

        # if use_entities:
        #     from survos2.entity.sampler import sample_region_at_pt
        #     sample_region_at_pt(cData.vol_stack[0], [50, 50, 50], (32, 32, 32))
        #     @entity_layer.mouse_drag_callbacks.append
        #     def view_location(layer, event):
        #         coords = np.round(layer.coordinates).astype(int)
        #         if coords_in_view(coords, cData.vol_stack[0].shape):
        #             vol1 = sample_region_at_pt(cData.vol_stack[0], coords, (32, 32, 32))
        #             logger.debug(f"Sampled from {coords} a vol of shape {vol1.shape}")
        #             cfg.smallvol_control.set_vol(np.transpose(vol1, (0, 2, 1)))
        #             msg = f"Displaying vol of shape {vol1.shape}"
        #             viewer.status = msg
        #         else:
        #             print("Coords out of view")
