import os
import sys
import numpy as np
import napari
import time
from typing import List
import seaborn as sns
import skimage

from matplotlib.colors import Normalize

import pyqtgraph
from pyqtgraph.widgets.ProgressDialog import ProgressDialog
from qtpy import QtWidgets, QtCore, QtGui
from qtpy.QtCore import QSize, Signal
from loguru import logger

import survos2
from survos2.model import Workspace, Dataset
import survos2.api.workspace as ws
from survos2.frontend.gui_napari import NapariWidget
from survos2.frontend.datawidgets import TableWidget, SmallVolWidget 
from survos2.frontend.controlpanel import ButtonPanelWidget, PluginPanelWidget
from survos2.model import DataModel
from survos2.frontend.control.launcher import Launcher
from survos2.frontend.modal import ModalManager
from survos2.frontend.classic_views import ViewContainer
from survos2.frontend.magicgui_widgets import roi_gui, pipeline_gui, workspace_gui
from survos2.entity.entities import make_entity_df
from survos2.entity.anno import geom
from survos2.entity.sampler import sample_roi, sample_bvol, crop_vol_in_bbox, crop_vol_and_pts_centered
from survos2.helpers import AttrDict
from .views import list_views, get_view
from survos2.improc.utils import DatasetManager
from survos2.config import Config 




def update_ui():
    logger.info('Updating UI')
    QtCore.QCoreApplication.processEvents()
    time.sleep(0.1)


def setup_entity_table(viewer, cData):
    tabledata = []
    
    for i in range(len(cData.entities)):
        entry = (i, 
            cData.entities.iloc[i]['z'],
            cData.entities.iloc[i]['x'],
            cData.entities.iloc[i]['y'],
            cData.entities.iloc[i]['class_code'])
        tabledata.append(entry)

    tabledata = np.array(tabledata, 
                        dtype=[('index', int), 
                                ('z', int), 
                                ('x', int),
                                ('y', int),
                                ('class_code', int),])   

    logger.debug(f"Loaded {len(tabledata)} entities.")
    sel_start, sel_end = 0, len(cData.entities)

    centers = np.array([ [np.int(np.float(cData.entities.iloc[i]['z'])), 
                        np.int(np.float(cData.entities.iloc[i]['x'])), 
                        np.int(np.float(cData.entities.iloc[i]['y']))] 
                                for i in range(sel_start, sel_end)])

    num_classes = len(np.unique(cData.entities["class_code"])) + 5
    logger.debug(f"Number of entity classes {num_classes}")
    palette = np.array(sns.color_palette("hls", num_classes) )# num_classes))
    norm = Normalize(vmin=0, vmax=num_classes)

    face_color_list = [palette[class_code] for class_code in cData.entities["class_code"]]    
    entity_layer = viewer.add_points(centers, size=[10] * len(centers), opacity=0.5, 
    face_color=face_color_list, n_dimensional=True)

    cData.tabledata = tabledata
    
    return entity_layer, tabledata


def hex_string_to_rgba(hex_string):
    hex_value = hex_string.lstrip('#')
    rgba_array = np.append(np.array([int(hex_value[i:i+2], 16) for i in (0, 2, 4)]), 255.0) / 255.0
    return rgba_array

def frontend(cData):

    logger.info(f"Connected to launcher {Launcher.g.connected}")    
    default_uri = '{}:{}'.format(Config['api.host'], Config['api.port'])
    Launcher.g.set_remote(default_uri)
    
    with napari.gui_qt():
        viewer = napari.Viewer(title="SuRVoS")
        viewer.theme = 'dark'
        
        # remove in order to rearrange standard Napari layer widgets
        #viewer.window.remove_dock_widget(viewer.window.qt_viewer.dockLayerList)
        #viewer.window.remove_dock_widget(viewer.window.qt_viewer.dockLayerControls)
        
        # Load data into viewer
        viewer.add_image(cData.vol_stack[0], name=cData.layer_names[0])   
        #labels_from_pts = viewer.add_labels(cData.vol_anno, name='labels')
        #labels_from_pts.visible = False

        #
        # Entities
        #

        logger.debug("Creating entities.")    
        entity_layer, tabledata = setup_entity_table(viewer, cData)

        viewer.dw = AttrDict()
        viewer.dw.bpw = ButtonPanelWidget()        
        viewer.dw.ppw = PluginPanelWidget()
        viewer.dw.ppw.setMinimumSize(QSize(400, 500))

        viewer.dw.table_control = TableWidget()        
        viewer.dw.table_control.set_data(tabledata)
        
        vol1 = sample_roi(cData.vol_stack[0], tabledata, vol_size=(32,32,32))
        logger.debug(f"Sampled ROI vol of shape {vol1.shape}")
                
        viewer.dw.smallvol_control = SmallVolWidget(vol1)
        cData.scfg.object_table = viewer.dw.table_control
        
        def setup_pipeline():
            pipeline_ops = ['make_masks', 'make_features', 'make_sr', 'make_seg_sr', 'make_seg_cnn']

        def process_pipeline(pipeline):
            minVal, maxVal = 0, len(pipeline.pipeline_ops)    
            with ProgressDialog(f"Processing pipeline {cData.scfg['pipeline_option']}", minVal, maxVal) as dlg:                    
            
                if dlg.wasCanceled():
                    raise Exception("Processing canceled")

                for step_idx, step in enumerate(pipeline):
                    logger.debug(step)
                    dlg.setValue(step_idx)  

        def get_patch():
            entity_pts = np.array(make_entity_df(np.array(cData.entities)))
            z_st, z_end, x_st, x_end, y_st, y_end = cData.scfg['roi_crop']
            crop_coord = [z_st, x_st, y_st]
            precropped_vol_size = z_end-z_st,x_end-x_st,y_end-y_st

            logger.info("Applying pipeline to cropped vol at loc {crop_coord} and size {precropped_vol_size}")
            
            cropped_vol, precropped_pts = crop_vol_and_pts_centered(cData.vol_stack[0],entity_pts,
                    location=crop_coord,
                    patch_size=precropped_vol_size,
                    debug_verbose=True,
                    offset=True)

            patch = Patch(
                        {'in_array': cropped_vol},
                        precropped_pts)
                
            return patch

        def processEvents(x):
            logger.info(f"Received event {x}")

            if x['data'] == 'view_pipeline':
                logger.debug(f"view_annotation {x['pipeline_id']}")

                src = DataModel.g.dataset_uri(x['pipeline_id'], group='pipeline')
                with DatasetManager(src, out=None, dtype='uint32', fillvalue=0) as DM:
                    src_dataset = DM.sources[0]
                    src_arr = src_dataset[:]
                    viewer.add_labels(src_arr, name=x['pipeline_id'])

            elif x['data'] == 'view_annotations':
                logger.debug(f"view_annotation {x['level_id']}")

                src = DataModel.g.dataset_uri(x['level_id'], group='annotations')
                with DatasetManager(src, out=None, dtype='uint32', fillvalue=0) as DM:
                    src_dataset = DM.sources[0]
                    src_arr = src_dataset[:]

                result = Launcher.g.run('annotations', 'get_levels', workspace=DataModel.g.current_workspace)
                logger.debug(f"Result of regions existing: {result}")
                
                if result:
                    for r in result:
                        level_name = r['name']
                        if r['kind'] == 'level':
                            cmapping = {}
                            for k, v in r['labels'].items():
                                cmapping[int(k)] = hex_string_to_rgba(v['color'])
                    
                    viewer.add_labels(src_arr, name=x['level_id'],  color=cmapping)
                    
            elif x['data'] == 'view_feature':
                logger.debug(f"view_feature {x['feature_id']}")

                # use DatasetManager to load feature from workspace as array and then add it to viewer
                src = DataModel.g.dataset_uri(x['feature_id'], group='features')
                with DatasetManager(src, out=None, dtype='float32', fillvalue=0) as DM:
                    src_dataset = DM.sources[0]
                    src_arr = src_dataset[:]
                    viewer.add_image(src_arr, name=x['feature_id'])
                    
            elif x['data'] == 'view_supervoxels':
                logger.debug(f"view_feature {x['region_id']}")

                #DataModel.g.current_workspace = test_workspace_name
                src = DataModel.g.dataset_uri(x['region_id'], group='regions')

                with DatasetManager(src, out=None, dtype='uint32', fillvalue=0) as DM:
                    src_dataset = DM.sources[0]
                    src_arr = src_dataset[:]
                    viewer.add_labels(src_arr, name=x['region_id'])
                    
            elif x['data']=='flip_coords':
                logger.debug(f"flip coords {x['axis']}")
                layers_selected = [viewer.layers[i].selected for i in range(len(viewer.layers))]
                logger.info(f"Selected layers {layers_selected}")
                selected_layer_idx = int(np.where(layers_selected)[0][0])
                selected_layer = viewer.layers[selected_layer_idx]
                if viewer.layers[selected_layer_idx]._type_string == 'points':
                    pts = viewer.layers[selected_layer_idx].data
                    
                    pts = pts[:,[0,2,1]]
                    entity_layer = viewer.add_points(pts, size=[10] * len(pts),  n_dimensional=True)
            
            elif x['data']=='spatial_cluster':
                logger.debug(f"spatial cluster")
                layers_selected = [viewer.layers[i].selected for i in range(len(viewer.layers))]
                logger.info(f"Selected layers {layers_selected}")
                selected_layer_idx = int(np.where(layers_selected)[0][0])
                selected_layer = viewer.layers[selected_layer_idx]
                if viewer.layers[selected_layer_idx]._type_string == 'points':
                    pts = viewer.layers[selected_layer_idx].data
                                    
                    entities_pts = np.zeros((pts.shape[0],4))
                    entities_pts[:, 0:3] = pts[:,0:3]
                    cc = [0] * len(entities_pts)
                    entities_pts[:, 3] =  cc
                    
                    logger.debug(f"Added class code to bare pts producing {entities_pts}")
                
                    from survos2.entity.anno.point_cloud import chip_cluster
                    refined_entities = chip_cluster(entities_pts,cData.vol_stack[0].copy(), 0, 0, 
                                                        MIN_CLUSTER_SIZE=2, method='dbscan',
                                                        debug_verbose=False, plot_all=False)

                    pts = np.array(refined_entities)
                    logger.info(f"Clustered with result of {refined_entities.shape} {refined_entities}")
                    
                    cData.entities = make_entity_df(pts, flipxy=False)
                    entity_layer, tabledata = setup_entity_table(viewer, cData)
                    
            elif x['data']=='show_roi':
                selected_roi_idx = viewer.dw.table_control.w.selected_row 
                logger.info(f"Showing ROI {x['selected_roi']}")
                vol1 = sample_roi(cData.vol_stack[0], cData.tabledata, selected_roi_idx, vol_size=(32,32,32))
                #viewer.dw.smallvol_control.set_vol(np.transpose(vol1, (0,2,1)))                        
                viewer.dw.smallvol_control.set_vol(vol1)
                logger.info(f"Sampled ROI vol of shape: {vol1.shape}")
        
            elif x['data']=='refresh':
                logger.debug("Refreshing plugin panel")
                viewer.dw.ppw.setup()
        #
        # Add widgets to viewer
        #

        # Plugin Panel widget
        viewer.dw.ppw.clientEvent.connect(lambda x: processEvents(x)  )  
        
        # todo: fix hack connection
        cData.scfg.ppw = viewer.dw.ppw
        
        # Button panel widget 
        #bpw_control_widget = viewer.window.add_dock_widget(viewer.dw.bpw, area='left')                
        viewer.dw.bpw.clientEvent.connect(lambda x: processEvents(x)  )  
                
        #viewer.dw.table_control.w.events.subscribe(lambda x: processEvents(x)  ) 
        viewer.dw.table_control.clientEvent.connect(lambda x: processEvents(x)  ) 

        #
        # magicgui
        #

        #roi_gui_widget = roi_gui.Gui()
        #viewer.window.add_dock_widget(roi_gui_widget, area='top')
        #viewer.layers.events.changed.connect(lambda x: roi_gui_widget.refresh_choices())
        #roi_gui_widget.refresh_choices()
        #workspace_layer_controls_dockwidget = viewer.window.add_dock_widget(viewer.window.qt_viewer.layerButtons, area='right')
        #workspace_layer_controls_dockwidget = viewer.window.add_dock_widget(viewer.window.qt_viewer.viewerButtons, area='left')
                
        #from survos2.frontend.views.layer_manager import WorkspaceLayerManager
        #workspace_layer_manager_widget = WorkspaceLayerManager()
        #viewer.window.add_dock_widget(workspace_layer_manager_widget, area='right')
        
        #workspace_layer_controls_dockwidget = viewer.window.add_dock_widget(viewer.window.qt_viewer.dockLayerControls, area='left')
        #workspace_layer_manager_dockwidget = viewer.window.add_dock_widget(viewer.window.qt_viewer.dockLayerList, area='left')
        
        #
        # Tabs
        #

        from qtpy.QtWidgets import QLabel, QTabWidget, QVBoxLayout, QPushButton, QWidget
        tabwidget = QTabWidget()
        tab1 = QWidget()
        tab2 = QWidget()
        tab3 = QWidget()

        tabwidget.addTab(tab1, "Segmentation")
        tabwidget.addTab(tab2, "Objects")
        #tabwidget.addTab(tab2, "Analyze")

        tab1.layout = QVBoxLayout()
        tab1.setLayout(tab1.layout)
        tab1.layout.addWidget(viewer.dw.ppw)
        
        tab2.layout = QVBoxLayout()
        tab2.setLayout(tab2.layout)
        tab2.layout.addWidget(viewer.dw.table_control.w)
        tab2.layout.addWidget(viewer.dw.smallvol_control.imv)

        viewer.window.add_dock_widget(tabwidget, area='right')

        workspace_gui_widget = workspace_gui.Gui()
        workspace_gui_dockwidget = viewer.window.add_dock_widget(workspace_gui_widget, area='left')
        viewer.layers.events.changed.connect(lambda x: workspace_gui_widget.refresh_choices())
        workspace_gui_widget.refresh_choices()

        
        def coords_in_view(coords, image_shape):
            if coords[0] > 0 and coords[1] > 0:
                return True
            else:
                return False
        
        from survos2.entity.sampler import sample_region_at_pt
        sample_region_at_pt(cData.vol_stack[0], [50,50,50], (32,32,32))
        @entity_layer.mouse_drag_callbacks.append
        def get_connected_component_shape(layer, event):            
            logger.debug(f"Clicked: {layer}, {event}")
            coords = np.round(layer.coordinates).astype(int)
            logger.debug(f"Sampling region at ")
            if coords_in_view(cData.vol_stack[0].shape, coords):
                vol1 = sample_region_at_pt(cData.vol_stack[0], coords, (32,32,32))
                logger.debug(f'Sampled from {coords} a vol of shape {vol1.shape}')
                viewer.dw.smallvol_control.set_vol(np.transpose(vol1, (0,2,1)))            
            #msg = f'Displaying vol of shape {vol1.shape}'
            

