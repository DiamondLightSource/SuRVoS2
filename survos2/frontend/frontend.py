import os
import sys
import numpy as np
import napari
import time
from typing import List
import seaborn as sns

from matplotlib.colors import Normalize
from pyqtgraph.widgets.ProgressDialog import ProgressDialog
from qtpy import QtCore
from qtpy.QtWidgets import QTabWidget, QVBoxLayout, QWidget
from qtpy.QtCore import QSize
from loguru import logger

from skimage import img_as_float, img_as_ubyte
from scipy import ndimage

from survos2.frontend.components.entity import (
    TableWidget,
    SmallVolWidget,
    setup_entity_table,
)
from survos2.frontend.panels import ButtonPanelWidget, PluginPanelWidget
from survos2.model import DataModel
from survos2.frontend.control.launcher import Launcher
from survos2.frontend.panels_magicgui import (
    workspace_gui,
    save_annotation_gui,
)
from survos2.entity.entities import make_entity_df
from survos2.entity.sampler import (
    sample_roi,
    crop_vol_and_pts_centered,
)
from survos2.helpers import AttrDict
from survos2.improc.utils import DatasetManager
from survos2.server.config import cfg
from survos2.helpers import simple_norm

from survos2.frontend.utils import coords_in_view, hex_string_to_rgba, get_color_mapping

import threading
from PyQt5.QtCore import QThread, QTimer
from PyQt5.QtWidgets import QApplication, QPushButton, QWidget
from skimage.draw import line
from skimage.morphology import disk
from scipy.ndimage import binary_dilation


# def update_ui():
#     QtCore.QCoreApplication.processEvents()
#     time.sleep(0.1)


def frontend(cData):
    logger.info(
        f"Frontend loaded image volume of shape {DataModel.g.current_workspace_shape}"
    )
    src = DataModel.g.dataset_uri("__data__")

    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        src_dataset = DM.sources[0]
        entity_fullname = src_dataset.get_metadata("entities_name")
    logger.info(f"entity_fullname {entity_fullname}")

    if entity_fullname is None:
        use_entities = False
    else:
        use_entities = True

    # cfg.timer = WorkerThread()
    cfg.current_supervoxels = None
    cfg.current_mode = 'paint'
    with napari.gui_qt():
        viewer = napari.Viewer(title="SuRVoS")

        viewer.theme = "dark"
        viewer.window._qt_window.setGeometry(100, 200, 1280, 720)
        
        # Napari ui modification
        
        #viewer.window.qt_viewer.layerButtons.hide() 
        # remove in order to rearrange standard Napari layer widgets
        #viewer.window.remove_dock_widget(viewer.window.qt_viewer.dockLayerList)
        #viewer.window.remove_dock_widget(viewer.window.qt_viewer.dockLayerControls)
        # viewer.window.qt_viewer.setVisible(False)
        
        
        # Load data into viewer
        viewer.add_image(cData.vol_stack[0], name=cData.layer_names[0])

        # SuRVoS controls

        viewer.dw = AttrDict()
        # viewer.dw.bpw = ButtonPanelWidget()
        viewer.dw.ppw = PluginPanelWidget()
        viewer.dw.ppw.setMinimumSize(QSize(400, 400))
        viewer.cfg = cfg
        # attach workspace to viewer for interactive debugging
        from survos2.model import Workspace

        ws = Workspace(DataModel.g.current_workspace)
        viewer.dw.ws = ws
        viewer.dw.datamodel = DataModel.g

        def setup_pipeline():
            pipeline_ops = [
                "make_masks",
                "make_features",
                "make_sr",
                "make_seg_sr",
                "make_seg_cnn",
            ]

        def set_paint_params(msg):
            logger.debug(f"set_paint_params {msg['paint_params']}")
            paint_params = msg['paint_params']
            #logger.debug(paint_params)
            anno_layer = viewer.layers.selected[0]
            if anno_layer.dtype == 'uint32':
                anno_layer.mode = 'paint'
                cfg.current_mode = 'paint'
            
                label_value = paint_params['label_value']
                anno_layer.selected_label = int(label_value['idx']) - 1
                cfg.label_value = label_value
                anno_layer.brush_size = int(paint_params['brush_size'])
                 

        def paint_annotations(msg):
            logger.debug(f"view_annotation {msg['level_id']}")

            ws = Workspace(DataModel.g.current_workspace)
            dataset_name = "annotations\\" + msg["level_id"]
            ds = ws.get_dataset(dataset_name)
            src_arr = ds[:]

            result = Launcher.g.run(
                "annotations", "get_levels", workspace=DataModel.g.current_workspace
            )
            label_ids = list(np.unique(src_arr))
            print(label_ids)

            if result:
                cmapping = get_color_mapping(result)
                print(cmapping)

                existing_layer = [v for v in viewer.layers if v.name == msg["level_id"]]
                
                sel_label = 1
                brush_size = 10

                if len(existing_layer) > 0:
                    viewer.layers.remove(existing_layer[0])
                    sel_label = existing_layer[0].selected_label
                    brush_size = existing_layer[0].brush_size
                    print(f"Removed existing layer {existing_layer[0]}")


                if cfg.current_supervoxels == None:
                    pass
                else:
                    sv_name = (
                        cfg.current_supervoxels
                    )  # e.g. "regions/001_supervoxels"
                    regions_dataset = DataModel.g.dataset_uri(sv_name)

                    with DatasetManager(
                        regions_dataset, out=None, dtype="uint32", fillvalue=0
                    ) as DM:
                        src_dataset = DM.sources[0]
                        sv_arr = src_dataset[:]

                    print(f"Loaded superregion array of shape {sv_arr.shape}")

                label_layer = viewer.add_labels(
                    src_arr & 15, name=msg["level_id"], color=cmapping
                )

                label_layer.mode = cfg.current_mode
                label_layer.selected_label = sel_label
                label_layer.brush_size = brush_size

                @label_layer.bind_key("Control-Z")
                def undo(v):
                    level = cfg.current_annotation
                    params = dict(workspace=True, level=level)
                    result = Launcher.g.run("annotations", "annotate_undo", **params)
                    cfg.ppw.clientEvent.emit(
                        {
                            "source": "annotations",
                            "data": "view_annotations",
                            "level_id": level,
                        }
                    )

                # viewer.layers[-1].corner_pixels
                @label_layer.mouse_drag_callbacks.append
                def view_location(layer, event):
                    dragged = False

                    drag_pts = []
                    coords = np.round(layer.coordinates).astype(int)
                    drag_pt = [coords[0], coords[1], coords[2]]
                    drag_pts.append(drag_pt)
                    yield

                    all_regions = set()

                    if layer.mode == "paint" or layer.mode == "erase":

                        while event.type == "mouse_move":
                            coords = np.round(layer.coordinates).astype(int)
                            drag_pt = [coords[0], coords[1], coords[2]]
                            drag_pts.append(drag_pt)
                            dragged = True
                            yield

                        if dragged:

                            level = msg["level_id"]
                            layer_name = viewer.layers[-1].name  # get last added layer name
                            anno_layer = next(
                                l for l in viewer.layers if l.name == layer_name
                            )

                            sel_label = int(cfg.label_value['idx']) - 1


                            if layer.mode == 'erase':
                                sel_label = 0
                                cfg.current_mode = 'erase'
                            else:
                                cfg.current_mode = 'paint'

                            anno_layer.selected_label = sel_label 
                            anno_layer.brush_size = int(cfg.brush_size)

                            line_x = []
                            line_y = []
                            z, px, py = drag_pts[0]

                            for _, x, y in drag_pts[1:]:
                                yy, xx = line(py, px, y, x)
                                line_x.extend(xx)
                                line_y.extend(yy)
                                py, px = y, x

                            line_y = np.array(line_y)
                            line_x = np.array(line_x)

                            from survos2.frontend.plugins.annotations import (
                                dilate_annotations,
                            )

                            line_y, line_x = dilate_annotations(
                                line_y,
                                line_x,
                                src_arr[0, :],
                                viewer.layers[-1].brush_size,
                            )

                            

                            if cfg.current_supervoxels == None:
                                params = dict(workspace=True, level=level, label=sel_label)
                             
                                xx,yy = list(line_y), list(line_x)
                                yy = [int(e) for e in yy]
                                xx = [int(e) for e in xx]
                                params.update(slice_idx=int(z), yy=yy, xx=xx)
                                print(params)
                                result = Launcher.g.run("annotations", "annotate_voxels", **params)
                                print(result)
                            else:
                                for x, y in zip(line_x, line_y):
                                    sv = sv_arr[z, x, y]
                                    all_regions |= set([sv])

                                print(f"Painted regions {all_regions}")

                                params = dict(workspace=True, level=level, label=sel_label)
                                region = DataModel.g.dataset_uri(sv_name)

                                params.update(
                                    region=region,
                                    r=list(map(int, all_regions)),
                                    modal=False,
                                )
                                result = Launcher.g.run(
                                    "annotations", "annotate_regions", **params
                                )
                            
                            cfg.ppw.clientEvent.emit(
                                {
                                    "source": "annotations",
                                    "data": "view_annotations",
                                    "level_id": level,
                                }
                            )

                        else:
                            print("single click")

        def view_pipeline(msg):
            logger.debug(f"view_annotation {msg['pipeline_id']}")

            result = Launcher.g.run(
                "annotations", "get_levels", workspace=DataModel.g.current_workspace
            )
            logger.debug(f"Result of annotations get_levels: {result}")
            if result:
                cmapping = get_color_mapping(result)
                print(cmapping)

            src = DataModel.g.dataset_uri(msg["pipeline_id"], group="pipeline")

            with DatasetManager(src, out=None, dtype="uint32", fillvalue=0) as DM:
                src_dataset = DM.sources[0]
                src_arr = src_dataset[:]

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

        def view_feature(msg):
            logger.debug(f"view_feature {msg['feature_id']}")

            # use DatasetManager to load feature from workspace as array and then add it to viewer
            src = DataModel.g.dataset_uri(msg["feature_id"], group="features")
            with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
                src_dataset = DM.sources[0]
                src_arr = src_dataset[:]

                existing_layer = [
                    v for v in viewer.layers if v.name == msg["feature_id"]
                ]

                if len(existing_layer) > 0:
                    existing_layer[0].data = src_arr
                else:
                    viewer.add_image(src_arr, name=msg["feature_id"])

        def view_supervoxels(msg):
            logger.debug(f"view_feature {msg['region_id']}")
            src = DataModel.g.dataset_uri(msg["region_id"], group="regions")

            with DatasetManager(src, out=None, dtype="uint32", fillvalue=0) as DM:
                src_dataset = DM.sources[0]
                src_arr = src_dataset[:]

                existing_layer = [
                    v for v in viewer.layers if v.name == msg["region_id"]
                ]
                if len(existing_layer) > 0:
                    existing_layer[0].data = src_arr
                else:
                    from skimage.segmentation import find_boundaries

                    sv_image = find_boundaries(src_arr)
                    sv_layer = viewer.add_image(sv_image, name=msg["region_id"])
                    sv_layer.opacity = 0.3

        def view_entitys(msg):
            logger.debug(f"view_entitys {msg['entitys_id']}")
            dsname = "entitys\\" + msg["entitys_id"]
            ds = viewer.dw.ws.get_dataset(dsname)
            logger.debug(f"Using dataset {ds}")

            entities_fullname = ds.get_metadata("fullname")
            logger.info(f"Viewing entities {entities_fullname}")
            tabledata, entities_df = setup_entity_table(entities_fullname)

            sel_start, sel_end = 0, len(entities_df)

            centers = np.array(
                [
                    [
                        np.int(np.float(entities_df.iloc[i]["z"]) * 0.25),
                        np.int(np.float(entities_df.iloc[i]["x"]) * 0.25),
                        np.int(np.float(entities_df.iloc[i]["y"]) * 0.25),
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
            vol1 = sample_roi(
                cData.vol_stack[0],
                cfg.tabledata,
                selected_roi_idx,
                vol_size=(32, 32, 32),
            )
            # viewer.dw.smallvol_control.set_vol(np.transpose(vol1, (0,2,1)))
            cfg.smallvol_control.set_vol(vol1)
            logger.info(f"Sampled ROI vol of shape: {vol1.shape}")

        def process_pipeline(pipeline):
            minVal, maxVal = 0, len(pipeline.pipeline_ops)
            with ProgressDialog(
                f"Processing pipeline {cData.cfg['pipeline_option']}", minVal, maxVal
            ) as dlg:

                if dlg.wasCanceled():
                    raise Exception("Processing canceled")

                for step_idx, step in enumerate(pipeline):
                    logger.debug(step)
                    dlg.setValue(step_idx)

        def get_patch():
            entity_pts = np.array(make_entity_df(np.array(cData.entities)))
            z_st, z_end, x_st, x_end, y_st, y_end = cData.cfg["roi_crop"]
            crop_coord = [z_st, x_st, y_st]
            precropped_vol_size = z_end - z_st, x_end - x_st, y_end - y_st

            logger.info(
                "Applying pipeline to cropped vol at loc {crop_coord} and size {precropped_vol_size}"
            )

            cropped_vol, precropped_pts = crop_vol_and_pts_centered(
                cData.vol_stack[0],
                entity_pts,
                location=crop_coord,
                patch_size=precropped_vol_size,
                debug_verbose=True,
                offset=True,
            )

            patch = Patch({"in_array": cropped_vol}, precropped_pts)

            return patch

        def processEvents(msg):
            if msg["data"] == "view_pipeline":
                view_pipeline(msg)
            elif msg["data"] == "view_annotations":
                paint_annotations(msg)
            elif msg["data"] == "view_feature":
                view_feature(msg)
            elif msg["data"] == "view_supervoxels":
                view_supervoxels(msg)
            elif msg["data"] == "view_entitys":
                view_entitys(msg)
            elif msg["data"] == "show_roi":
                show_roi(msg)
            elif msg["data"] == "refresh":
                logger.debug("Refreshing plugin panel")
                viewer.dw.ppw.setup()
            elif msg["data"] == "save_annotation":
                save_annotation(msg)
            elif msg["data"] == "set_paint_params":
                set_paint_params(msg)

        #
        # Add widgets to viewer
        #

        # Plugin Panel widget
        viewer.dw.ppw.clientEvent.connect(lambda x: processEvents(x))
        cData.cfg.ppw = viewer.dw.ppw

        from survos2.frontend.slice_paint import MainWidget

        #viewer.classic_widget = MainWidget()
        # viewer.window.qt_viewer.setFixedSize(200,200)
        # viewer.window.qt_viewer.setVisible(False)
        #viewer.dims.ndisplay = 3 # start on 3d view

        # classic_dockwidget = viewer.window.add_dock_widget(
        #     viewer.classic_widget, area="right"
        # )
        # classic_dockwidget.setVisible(False)

        
        #
        # Tabs
        #
        pluginwidget_dockwidget = viewer.window.add_dock_widget(
            viewer.dw.ppw, area="right"
        )
        pluginwidget_dockwidget.setWindowTitle("Workspace")

        # viewer.window.add_dock_widget(
        #     viewer.window.qt_viewer.dockLayerList, area="right"
        # )
        # viewer.window.add_dock_widget(
        #     viewer.window.qt_viewer.dockLayerControls, area="right"
        # )
        # viewer.window.qt_viewer.dockLayerControls.setVisible(False)

        workspace_gui_widget = workspace_gui.Gui()
        workspace_gui_dockwidget = viewer.window.add_dock_widget(
            workspace_gui_widget, area="left"
        )
        workspace_gui_dockwidget.setVisible(False)
        workspace_gui_dockwidget.setWindowTitle("Save to workspace")

        viewer.layers.events.changed.connect(
            lambda x: workspace_gui_widget.refresh_choices()
        )
        workspace_gui_widget.refresh_choices()

        save_annotation_gui_widget = save_annotation_gui.Gui()
        save_annotation_dockwidget = viewer.window.add_dock_widget(
            save_annotation_gui_widget, area="right"
        )
        viewer.layers.events.changed.connect(
            lambda x: save_annotation_gui_widget.refresh_choices()
        )
        save_annotation_gui_widget.refresh_choices()
        save_annotation_dockwidget.setWindowTitle("Update annotation")
        save_annotation_dockwidget.setVisible(False)

        if use_entities:
            from survos2.entity.sampler import sample_region_at_pt

            sample_region_at_pt(cData.vol_stack[0], [50, 50, 50], (32, 32, 32))

            @entity_layer.mouse_drag_callbacks.append
            def view_location(layer, event):
                coords = np.round(layer.coordinates).astype(int)
                if coords_in_view(coords, cData.vol_stack[0].shape):
                    vol1 = sample_region_at_pt(cData.vol_stack[0], coords, (32, 32, 32))
                    logger.debug(f"Sampled from {coords} a vol of shape {vol1.shape}")
                    cfg.smallvol_control.set_vol(np.transpose(vol1, (0, 2, 1)))
                    msg = f"Displaying vol of shape {vol1.shape}"
                    viewer.status = msg
                else:
                    print("Coords out of view")
