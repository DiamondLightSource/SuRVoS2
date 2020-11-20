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
    update_annotation_gui,
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


import threading
from PyQt5.QtCore import QThread, QTimer
from PyQt5.QtWidgets import QApplication, QPushButton, QWidget


class WorkerThread(QThread):
    def run(self):
        def work():
            cfg.ppw.clientEvent.emit(
                {
                    "source": "update_annotation",
                    "data": "update_annotation",
                    "value": None,
                }
            )
            cfg.ppw.clientEvent.emit(
                {"source": "update_annotation", "data": "refresh", "value": None}
            )
            QThread.sleep(5)

        timer = QTimer()
        timer.timeout.connect(work)
        timer.start(50000)
        self.exec_()


def update_ui():
    QtCore.QCoreApplication.processEvents()
    time.sleep(0.1)


def hex_string_to_rgba(hex_string):
    hex_value = hex_string.lstrip("#")
    rgba_array = (
        np.append(np.array([int(hex_value[i : i + 2], 16) for i in (0, 2, 4)]), 255.0)
        / 255.0
    )
    return rgba_array


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

    cfg.timer = WorkerThread()

    with napari.gui_qt():
        viewer = napari.Viewer(title="SuRVoS")
        
        viewer.theme = "dark"
        viewer.window._qt_window.setGeometry(100, 200, 1280, 720)

        # remove in order to rearrange standard Napari layer widgets
        viewer.window.remove_dock_widget(viewer.window.qt_viewer.dockLayerList)
        viewer.window.remove_dock_widget(viewer.window.qt_viewer.dockLayerControls)
        #viewer.window.qt_viewer.setVisible(False)
        # Load data into viewer
        viewer.add_image(cData.vol_stack[0], name=cData.layer_names[0])

        viewer.dw = AttrDict()
        #viewer.dw.bpw = ButtonPanelWidget()
        viewer.dw.ppw = PluginPanelWidget()
        viewer.dw.ppw.setMinimumSize(QSize(400,400))

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
                logger.debug(f"view_annotation {msg['pipeline_id']}")

                result = Launcher.g.run(
                    "annotations", "get_levels", workspace=DataModel.g.current_workspace
                )
                logger.debug(f"Result of annotations get_levels: {result}")
                # if result:
                #     for r in result:
                #         level_name = r["name"]
                #         if r["kind"] == "level":
                #             cmapping = {}
                #             for ii, (k, v) in enumerate(r["labels"].items()):
                #                 remapped_label = int(k) + (16 * ii)
                #                 print(remapped_label)
                #                 cmapping[remapped_label] = hex_string_to_rgba(v["color"])

                src = DataModel.g.dataset_uri(msg["pipeline_id"], group="pipeline")

                with DatasetManager(src, out=None, dtype="uint32", fillvalue=0) as DM:
                    src_dataset = DM.sources[0]
                    src_arr = src_dataset[:]

                    existing_layer = [
                        (i, v)
                        for i, v in enumerate(viewer.layers)
                        if v.name == "001_level"
                    ]

                    if len(existing_layer) > 0:
                        logger.debug(
                            f"Removing existing layer and re-adding it with new colormapping. {existing_layer}"
                        )

                        viewer.layers.remove(existing_layer[0][0])
                        viewer.add_labels(
                            src_arr, name=msg["pipeline_id"]) #, color=cmapping)
                    else:
                        viewer.add_labels(
                            src_arr, name=msg["pipeline_id"]) #, color=cmapping)

            elif msg["data"] == "view_annotations":
                logger.debug(f"view_annotation {msg['level_id']}")

                ws = Workspace(DataModel.g.current_workspace)
                dataset_name = "annotations\\" + msg["level_id"]
                ds = ws.get_dataset(dataset_name)
                src_arr = ds[:]

                # src = DataModel.g.dataset_uri(msg["level_id"], group="annotations")
                # with DatasetManager(src, out=None, dtype="uint32", fillvalue=0) as DM:
                #    src_dataset = DM.sources[0]
                #    src_arr = src_dataset[:]

                result = Launcher.g.run(
                    "annotations", "get_levels", workspace=DataModel.g.current_workspace
                )
                label_ids = list(np.unique(src_arr))
                label_ids.remove(0)
                #label_ids.reverse()
                print(label_ids)


                if result:
                    # for r in result:
                    #     level_name = r["name"]
                    #     if r["kind"] == "level":
                    #         cmapping = {}
                    #         print(r["labels"].items())
                    #         for ii, (k, v) in enumerate(r["labels"].items()):
                    #             remapped_label = label_ids[ii]
                    #             print(remapped_label)
                    #             cmapping[remapped_label] = hex_string_to_rgba(v["color"])
                    #             #cmapping[ii+1] = hex_string_to_rgba(v["color"])
                    # print(cmapping)
                    # #cmapping[0] = (0.0,0.0,0.0,1.0)

                    existing_layer = [
                        v for v in viewer.layers if v.name == msg["level_id"]
                    ]

                    regions_dataset = DataModel.g.dataset_uri("regions/002_supervoxels")            
                    with DatasetManager(regions_dataset, out=None, dtype="uint32", fillvalue=0) as DM:
                        src_dataset = DM.sources[0]
                        sv_arr = src_dataset[:]
                    print(f"SV array shape {sv_arr.shape}")


                    if len(existing_layer) > 0:
                        existing_layer[0].data = src_arr
                    else:
                        label_layer = viewer.add_labels(src_arr, name=msg["level_id"])#, color=cmapping)
                        
                        @label_layer.mouse_drag_callbacks.append
                        def view_location(layer, event):
                            print('mouse down')
                            dragged = False
                            yield
                            
                            drag_pts = []
                            all_regions = set() 
                            
                            while event.type == 'mouse_move':
                                coords = np.round(layer.coordinates).astype(int)
                                drag_pt = [viewer.dims.current_step[0], coords[0], coords[1]]
                                drag_pts.append(drag_pt)
                                dragged = True
                                yield
                            
                            if dragged:
                                print('drag end')
                                print(drag_pts)
                                
                                level = msg["level_id"]
                                label = 1

                                #params = dict(workpace=True, src=region, slice_idx=viewer.dims.current_step[0])
                                #result = Launcher.g.run("regions", "get_slice", **params)
                                
                                from survos2.utils import decode_numpy
                                #region = decode_numpy(result)
                                #print(region.shape)
                                
                                for z,x,y in drag_pts:
                                    print(z,x,y)
                                    sv = sv_arr[z, y, x]
                                    all_regions |= set([sv])
                                
                                print(all_regions)      

                                params = dict(workspace=True, level=level, label=label)
                                region = DataModel.g.dataset_uri("regions/002_supervoxels")
                                
                                params.update(region=region, r=list(map(int, all_regions)), modal=False)
                                result = Launcher.g.run("annotations", "annotate_regions", **params)
                            
                            else:
                                print('single click')
                        # coords = np.round(layer.coordinates).astype(int)
                        # viewer.status = str(coords)
                        # logger.debug(f"{coords} {event}")
                        

            elif msg["data"] == "view_feature":
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

            elif msg["data"] == "view_supervoxels":
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
                        viewer.add_image(find_boundaries(src_arr), name=msg["region_id"])

            elif msg["data"] == "view_entitys":
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
                            np.int(np.float(entities_df.iloc[i]["z"])),
                            np.int(np.float(entities_df.iloc[i]["x"])),
                            np.int(np.float(entities_df.iloc[i]["y"])),
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

            elif msg["data"] == "flip_coords":
                logger.debug(f"flip coords {msg['axis']}")
                layers_selected = [
                    viewer.layers[i].selected for i in range(len(viewer.layers))
                ]
                logger.info(f"Selected layers {layers_selected}")
                selected_layer_idx = int(np.where(layers_selected)[0][0])
                selected_layer = viewer.layers[selected_layer_idx]
                if viewer.layers[selected_layer_idx]._type_string == "points":
                    pts = viewer.layers[selected_layer_idx].data

                    pts = pts[:, [0, 2, 1]]
                    entity_layer = viewer.add_points(
                        pts, size=[10] * len(pts), n_dimensional=True
                    )

            elif msg["data"] == "spatial_cluster":
                logger.debug(f"spatial cluster")
                layers_selected = [
                    viewer.layers[i].selected for i in range(len(viewer.layers))
                ]
                logger.info(f"Selected layers {layers_selected}")
                selected_layer_idx = int(np.where(layers_selected)[0][0])
                selected_layer = viewer.layers[selected_layer_idx]

                if viewer.layers[selected_layer_idx]._type_string == "points":
                    pts = viewer.layers[selected_layer_idx].data

                    entities_pts = np.zeros((pts.shape[0], 4))
                    entities_pts[:, 0:3] = pts[:, 0:3]
                    cc = [0] * len(entities_pts)
                    entities_pts[:, 3] = cc

                    logger.debug(
                        f"Added class code to bare pts producing {entities_pts}"
                    )

                    from survos2.entity.anno.point_cloud import chip_cluster

                    refined_entities = chip_cluster(
                        entities_pts,
                        cData.vol_stack[0].copy(),
                        0,
                        0,
                        MIN_CLUSTER_SIZE=2,
                        method="dbscan",
                        debug_verbose=False,
                        plot_all=False,
                    )

                    pts = np.array(refined_entities)
                    logger.info(
                        f"Clustered with result of {refined_entities.shape} {refined_entities}"
                    )

                    cData.entities = make_entity_df(pts, flipxy=False)
                    entity_layer, tabledata = setup_entity_table(viewer, cData)

            elif msg["data"] == "show_roi":
                if use_entities:
                    selected_roi_idx = viewer.dw.table_control.w.selected_row
                    logger.info(f"Showing ROI {msg['selected_roi']}")
                    vol1 = sample_roi(
                        cData.vol_stack[0],
                        cData.cfg.tabledata,
                        selected_roi_idx,
                        vol_size=(32, 32, 32),
                    )
                    # viewer.dw.smallvol_control.set_vol(np.transpose(vol1, (0,2,1)))
                    viewer.dw.smallvol_control.set_vol(vol1)
                    logger.info(f"Sampled ROI vol of shape: {vol1.shape}")

            elif msg["data"] == "refresh":
                logger.debug("Refreshing plugin panel")
                viewer.dw.ppw.setup()

            elif msg["data"] == "update_annotation":
                
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
                        logger.debug(
                            f"Transfering to workspace {fid}, {ftype}, {fname}"
                        )

                        dst = DataModel.g.dataset_uri(fid, group="annotations")

                        with DatasetManager(
                            dst, out=dst, dtype="uint16", fillvalue=0
                        ) as DM:
                            DM.out[:] = annotation_layer[0].data
                else:
                    logger.debug("update_annotation couldn't find annotation in viewer")

        #
        # Add widgets to viewer
        #

        # Plugin Panel widget
        viewer.dw.ppw.clientEvent.connect(lambda x: processEvents(x))
        cData.cfg.ppw = viewer.dw.ppw

        from survos2.frontend.scratch_f2 import MainWidget
        viewer.classic_widget = MainWidget()
        #viewer.window.qt_viewer.setFixedSize(200,200)
        #viewer.window.qt_viewer.setVisible(False)
        viewer.dims.ndisplay = 3

        classic_dockwidget = viewer.window.add_dock_widget(viewer.classic_widget, area="right")
        classic_dockwidget.setVisible(False)
        
        #main_dockwidget = viewer.window.add_dock_widget(viewer.window.qt_viewer, area="right")
        #viewer.dw.bpw = ButtonPanelWidget() 
        #bpw_control_widget = viewer.window.add_dock_widget(viewer.dw.bpw, area='top')                
        #viewer.dw.bpw.clientEvent.connect(lambda x: processEvents(x)  )  

        
        #
        # Tabs
        #

        
        pluginwidget_dockwidget = viewer.window.add_dock_widget(
            viewer.dw.ppw, area="left"
        )
        pluginwidget_dockwidget.setWindowTitle("Workspace")

        viewer.window.add_dock_widget(viewer.window.qt_viewer.dockLayerList, area="right")
        viewer.window.add_dock_widget(viewer.window.qt_viewer.dockLayerControls, area="right")
        #viewer.window.qt_viewer.dockLayerControls.setVisible(False)
                
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

        update_annotation_gui_widget = update_annotation_gui.Gui()
        update_annotation_dockwidget = viewer.window.add_dock_widget(
            update_annotation_gui_widget, area="right"
        )
        viewer.layers.events.changed.connect(
            lambda x: update_annotation_gui_widget.refresh_choices()
        )
        update_annotation_gui_widget.refresh_choices()
        update_annotation_dockwidget.setWindowTitle("Update annotation")
        update_annotation_dockwidget.setVisible(False)
        


        def coords_in_view(coords, image_shape):
            if (
                coords[0] >= 0
                and coords[1] >= 0
                and coords[0] < image_shape[0]
                and coords[1] < image_shape[1]
            ):
                return True
            else:
                return False

        # if use_entities:
        #     from survos2.entity.sampler import sample_region_at_pt
        #     sample_region_at_pt(cData.vol_stack[0], [50, 50, 50], (32, 32, 32))
        #     @entity_layer.mouse_drag_callbacks.append
        #     def view_location(layer, event):
        #         coords = np.round(layer.coordinates).astype(int)
        #         if coords_in_view(coords, cData.vol_stack[0].shape):
        #             vol1 = sample_region_at_pt(cData.vol_stack[0], coords, (32, 32, 32))
        #             logger.debug(f"Sampled from {coords} a vol of shape {vol1.shape}")
        #             viewer.dw.smallvol_control.set_vol(np.transpose(vol1, (0, 2, 1)))
        #             msg = f"Displaying vol of shape {vol1.shape}"
        #             viewer.status = msg
        #         else:
        #             print("Coords out of view")

        @napari.Viewer.bind_key("Shift-D")
        def dilate(viewer):
            logger.debug("Erode")
            str_3D = ndimage.morphology.generate_binary_structure(3, 1)

            img = viewer.layers.selected[0].data.copy()
            img = simple_norm(img)

            img = ndimage.morphology.binary_dilation(img_as_ubyte(img), str_3D)
            viewer.add_image(img_as_float(img), name="Dilation")

        @napari.Viewer.bind_key("Shift-E")
        def erode(viewer):
            logger.debug("Erode")
            str_3D = ndimage.morphology.generate_binary_structure(3, 1)
            img = viewer.layers.selected[0].data.copy()
            img = img / np.max(img)
            img = simple_norm(img)

            img = ndimage.morphology.binary_dilation(img_as_ubyte(img), str_3D)
            viewer.add_image(img_as_float(img), name="Erosion")

        @napari.Viewer.bind_key("Shift-M")
        def median(viewer, median_size=2):
            logger.debug("Median, size {median_size}")
            img = viewer.layers.selected[0].data.copy()
            img = img_as_ubyte(
                ndimage.median_filter(simple_norm(img), size=median_size)
            )

            viewer.add_image(img, name="Median")
