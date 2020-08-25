import os
import sys
import numpy as np
import napari
import time
from typing import List

import seaborn as sns
import skimage

import matplotlib.cm as cm
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
from survos2.frontend.control import DataModel
from survos2.frontend.control.launcher import Launcher
from survos2.frontend.modal import ModalManager
from survos2.frontend.classic_views import ViewContainer
from survos2.frontend.pipeline import sv_gui, features_gui, roi_gui, prediction_gui, pipeline_gui, workspace_gui
from survos2.entity.entities import make_entity_df
from survos2.entity.anno import geom
from survos2.entity.sampler import sample_roi, sample_bvol, crop_vol_in_bbox, crop_vol_and_pts_centered
from survos2.helpers import AttrDict
from .views import list_views, get_view

#from survos2.improc.features import gaussian, tvdenoising3d, gaussian_norm
#from survos2.improc.regions.slic import postprocess


#from survos2.server.supervoxels import generate_supervoxels
#from survos2.server.features import generate_features, prepare_prediction_features

#from survos2.server.cnn_models import setup_cnn_model
#from survos2.server.pipeline import Patch


from survos2.improc.utils import DatasetManager


from survos2.config import Config 
from survos2.server.config import appState
scfg = appState.scfg


fmt = "{time} - {name} - {level} - {message}"
logger.remove() # remove default logger
logger.add(sys.stderr, level="DEBUG")
#logger.add(sys.stderr, level="ERROR", format=fmt)  #minimal stderr logger
#logger.add("logs/main.log", level="DEBUG", format=fmt) #compression='zip')


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


        
def frontend(cData):

    logger.info(f"Connected to launcher {Launcher.g.connected}")    
    default_uri = '{}:{}'.format(Config['api.host'], Config['api.port'])
    Launcher.g.set_remote(default_uri)
    
    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.appState = appState
        viewer.theme = 'light'

        #try:
        #    if len(cData.opacities) != len(cData.vol_stack):
         #       cData.opacities = [1] * len(cData.layer_names)

            #if len(cData.layer_names) != len(cData.vol_stack):
            #    cData.layer_names = [str(t) for t in list(range(len(cData.vol_stack)))]

        #except Exception as err:
        #    logger.error(f"Exception at Napari layer setup {err}")
        
        #
        # Load data into viewer
        #
        viewer.add_image(cData.vol_stack[0], name=cData.layer_names[0])
        #viewer.add_labels(cData.vol_anno, name='Active Annotation')
        #viewer.add_labels(cData.vol_supervoxels, name='Superregions')
        
        #for ii,f in enumerate(cData.features.filtered_layers):
        #    viewer.add_image(f, name=str(ii))

        #viewer.layers['Active Annotation'].opacity = 0.9
   
        #labels_from_pts = viewer.add_labels(cData.vol_anno, name='labels')
        #labels_from_pts.visible = False

        #
        # Entities
        #
        logger.debug("Creating entities.")    
        entity_layer, tabledata = setup_entity_table(viewer, cData)

        #pipeline_gui_widget = pipeline_gui.Gui()
        #viewer.window.add_dock_widget(pipeline_gui_widget, area='left')
        #viewer.layers.events.changed.connect(lambda x: pipeline_gui_widget.refresh_choices())
        #pipeline_gui_widget.refresh_choices()

        viewer.dw = AttrDict()
        viewer.dw.bpw = ButtonPanelWidget()
        viewer.dw.ppw = PluginPanelWidget()

        viewer.dw.table_control = TableWidget()        
        viewer.dw.table_control.set_data(tabledata)
        
        vol1 = sample_roi(cData.vol_stack[0], tabledata, vol_size=(32,32,32))
        logger.debug(f"Sampled ROI vol of shape {vol1.shape}")
                
        viewer.dw.smallvol_control = SmallVolWidget(vol1)
        appState.scfg.object_table = viewer.dw.table_control
        
        def setup_pipeline():
            pipeline_ops = ['make_masks', 'make_features', 'make_sr', 'make_seg_sr', 'make_seg_cnn']

        def process_pipeline(pipeline):
            minVal, maxVal = 0, len(pipeline.pipeline_ops)    
            with ProgressDialog(f"Processing pipeline {appState.scfg['pipeline_option']}", minVal, maxVal) as dlg:                    
            
                if dlg.wasCanceled():
                    raise Exception("Processing canceled")

                for step_idx, step in enumerate(pipeline):
                    logger.debug(step)
                    dlg.setValue(step_idx)  

        def get_patch():
            entity_pts = np.array(make_entity_df(np.array(cData.entities)))
            z_st, z_end, x_st, x_end, y_st, y_end = appState.scfg['roi_crop']
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
            

            # crop the vol, apply the pipeline, display the result in the roi viewer and add
            # an image that is blank except for the roi region back to the viewer
            # when the pipeline is ready to be applied to the whole image, set the 
            # roi to the size of the image and reapply
            if x['data']=='pipeline':
                logger.info(f"Pipeline: {appState.scfg['pipeline_option']}")
        
                pipeline = Pipeline(scfg.pipeline_ops)
                pipeline.init_payload(p)
                process_pipeline(pipeline)
                result = pipeline.get_result()

                viewer.add_labels(result.layers['total_mask'], name='Masks')
                viewer.add_labels(result.superregions.supervoxel_vol, name='SR')

            if x['data']=='oldpipeline':                 
                    if appState.scfg['pipeline_option']=='saliency_pipeline':
                        logger.info("Saliency pipeline")
                        models = {}
                        models['saliency_model'] = setup_cnn_model()

                        patch = Patch({'in_array': cropped_vol},
                                precropped_pts,
                                None)

                        viewer.add_image(payload.layers['saliency_bb'], name='BB', colormap='cyan')
                                                
                    elif appState.scfg['pipeline_option']=='prediction_pipeline':
                        logger.info(f"Pipeline: {appState.scfg['pipeline_option']}")
                        patch = Patch(
                                {'in_array': cropped_vol},
                                precropped_pts)
                        
                        pipeline_ops = ['make_masks', 'make_features', 'make_sr', 'make_seg_sr', 'make_seg_cnn']
                        pipeline = Pipeline(scfg.pipeline.pipeline_ops)
                        pipeline.init_payload(p)
                        process_pipeline(pipeline)
                        result = pipeline.get_result()

                        viewer.add_labels(result.layers['total_mask'], name='Masks')
                        viewer.add_labels(result.superregions.supervoxel_vol, name='SR')

                    elif appState.scfg['pipeline_option']=='mask_pipeline':
                        logger.info("Mask pipeline")
                        payload = Patch({'in_array': cropped_vol}, 
                                        precropped_pts)
                        pipeline_ops = ['make_masks',]
                        result = process_pipeline(pipeline)
                        viewer.add_labels(result.layers['core_mask'], name='Masks')
                    
                    result = payload.layers['result']
                    logger.info(f"Produced pipeline result: {result.shape}")                
                    viewer.dw.smallvol_control.set_vol(result)
                    result = np.zeros_like(cData.vol_stack[0])
                    
                    result[crop_coord[0]:crop_coord[0]+precropped_vol_size[0],
                    crop_coord[1]:crop_coord[1]+precropped_vol_size[1],
                    crop_coord[2]:crop_coord[2]+precropped_vol_size[2]] = payload.layers['result']
                    
                    viewer.add_image(result, name='Pipeline Result')
                    
                    dlg.setValue(10)         

            elif x['data'] == 'view_annotations':
                logger.debug(f"view_annotation {x['level_id']}")

                src = DataModel.g.dataset_uri(x['level_id'], group='annotations')
                with DatasetManager(src, out=None, dtype='uint32', fillvalue=0) as DM:
                    src_dataset = DM.sources[0]
                    src_arr = src_dataset[:]
                    viewer.add_labels(src_arr, name=x['level_id'])
                    

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

                with DatasetManager(src, out=None, dtype='float32', fillvalue=0) as DM:
                    src_dataset = DM.sources[0]
                    src_arr = src_dataset[:]
                    viewer.add_image(src_arr, name=x['region_id'])
                    
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
        
        #
        # Add widgets to viewer
        #
        #bpw_control_widget = viewer.window.add_dock_widget(viewer.dw.bpw, area='left')                
        viewer.dw.bpw.clientEvent.connect(lambda x: processEvents(x)  )  
        viewer.dw.ppw.clientEvent.connect(lambda x: processEvents(x)  )  
        
        # todo: fix hack connection
        appState.scfg.ppw = viewer.dw.ppw
        
        #viewer.dw.table_control.w.events.subscribe(lambda x: processEvents(x)  ) 
        viewer.dw.table_control.clientEvent.connect(lambda x: processEvents(x)  ) 

        #
        # magicgui
        #
        #roi_gui_widget = roi_gui.Gui()
        #viewer.window.add_dock_widget(roi_gui_widget, area='top')
        # sync dropdowns with layer model
        #viewer.layers.events.changed.connect(lambda x: roi_gui_widget.refresh_choices())
        #roi_gui_widget.refresh_choices()

        #workspace_gui_widget = workspace_gui.Gui()
        #viewer.window.add_dock_widget(workspace_gui_widget, area='left')
        #viewer.layers.events.changed.connect(lambda x: workspace_gui_widget.refresh_choices())
        #workspace_gui_widget.refresh_choices()
 
        #sv_gui_widget = sv_gui.Gui()
        #viewer.window.add_dock_widget(sv_gui_widget, area='right')
        #viewer.layers.events.changed.connect(lambda x: sv_gui_widget.refresh_choices())
        #sv_gui_widget.refresh_choices()
 
        prediction_gui_widget = prediction_gui.Gui()
        #viewer.window.add_dock_widget(prediction_gui_widget, area='right')
        viewer.layers.events.changed.connect(lambda x: prediction_gui_widget.refresh_choices())
        prediction_gui_widget.refresh_choices()


        #
        # Tabs
        #

        from qtpy.QtWidgets import QLabel, QTabWidget, QVBoxLayout, QPushButton, QWidget
        tabwidget = QTabWidget()
        tab1 = QWidget()
        tab2 = QWidget()
        tab3 = QWidget()
        tab4 = QWidget()
        tab5 = QWidget()

        tabwidget.addTab(tab1, "Workspace")
        tabwidget.addTab(tab2, "Predict")
        tabwidget.addTab(tab3, "Entities")
        #tabwidget.addTab(tab4, "Close up")

        tab1.layout = QVBoxLayout()
        tab1.setLayout(tab1.layout)
        tab1.layout.addWidget(viewer.dw.ppw)
        
        tab2.layout = QVBoxLayout()
        tab2.setLayout(tab2.layout)
        tab2.layout.addWidget(prediction_gui_widget)

        tab3.layout = QVBoxLayout()
        tab3.setLayout(tab3.layout)
        tab3.layout.addWidget(viewer.dw.table_control.w)
        tab3.layout.addWidget(viewer.dw.smallvol_control.imv)


        viewer.window.add_dock_widget(tabwidget, area='right')
            
        from survos2.entity.sampler import sample_region_at_pt
        sample_region_at_pt(cData.vol_stack[0], [50,50,50], (32,32,32))
        @entity_layer.mouse_drag_callbacks.append
        def get_connected_component_shape(layer, event):            
            logger.debug(f"Clicked: {layer}, {event}")
            coords = np.round(layer.coordinates).astype(int)
            logger.debug(f"Sampling region at ")
            vol1 = sample_region_at_pt(cData.vol_stack[0], coords, (32,32,32))
            logger.debug(f'Sampled from {coords} a vol of shape {vol1.shape}')
            viewer.dw.smallvol_control.set_vol(np.transpose(vol1, (0,2,1)))            
            #msg = f'Displaying vol of shape {vol1.shape}'
            

