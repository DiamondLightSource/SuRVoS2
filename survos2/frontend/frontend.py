import os
import sys
import numpy as np
import napari
import pandas as pd
import time
from typing import List
from dataclasses import dataclass
import seaborn as sns
from scipy import ndimage as ndi

import skimage
from skimage import img_as_float
from skimage import data
from skimage.morphology import binary_dilation, binary_erosion

import matplotlib.cm as cm
from matplotlib.colors import Normalize

import pyqtgraph
from pyqtgraph.widgets.ProgressDialog import ProgressDialog
#from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.parametertree import Parameter, ParameterTree
import pyqtgraph.parametertree.parameterTypes as pTypes

from qtpy import QtWidgets, QtCore, QtGui
from qtpy.QtCore import QSize, Signal

from vispy import scene
from vispy.color import Colormap
#import qdarkstyle
from loguru import logger

import survos2
from survos2.frontend.gui_napari import NapariWidget
from survos2.frontend.datawidgets import (TableWidget, TreeControl, 
                                          Cluster2dWidget, #ImageGridWidget,
                                            ParameterTreeWidget, DataTreeControl, SmallVolWidget,
                                            generate_test_data, parameter_tree_change,) 

from survos2.frontend.model import SegSubject
from survos2.frontend.controlpanel import ButtonPanelWidget, PluginPanelWidget
from survos2.frontend.cluster import ClusterWidget
from survos2.frontend.pipeline import sv_gui, features_gui, roi_gui, prediction_gui, pipeline_gui, workspace_gui

from survos2.model import Workspace, Dataset
import survos2.server.workspace as ws
import survos2.server.segmentation as sseg
#import survos2.server.test as stest
#from survos2.server.test import SegData

from survos2.server.config import appState
from survos2.server.model import Features
from survos2.server.supervoxels import generate_supervoxels
from survos2.server.filtering import generate_features, prepare_prediction_features
from survos2.server.config import appState
from survos2.server.pipeline import PipelinePayload
from survos2.server.pipeline import make_features
from survos2.server.pipeline import prediction_pipeline, mask_pipeline, saliency_pipeline
from survos2.helpers import AttrDict
#from survos2.utils import logger
from survos2.improc.features import gaussian, tvdenoising3d, gaussian_norm
from survos2.improc.regions.slic import postprocess
from survos2.entity.entities import make_entity_df
from survos2.entity.anno import geom
from survos2.entity.sampler import sample_roi, sample_bvol, crop_vol_in_bbox,  crop_vol_and_pts

from survos2.frontend.control.model import DataModel
from survos2.frontend.control.launcher import Launcher
from survos2.frontend.modal import ModalManager
from survos2.frontend.model import ClientData
from survos2.server.pipeline import setup_cnn_model
from survos2.frontend.control import DataModel
from survos2.improc.utils import DatasetManager
            

scfg = appState.scfg
#sns.set_style('darkgrid')

sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

fmt = "{time} - {name} - {level} - {message}"
logger.remove() # remove default logger
logger.add(sys.stderr, level="INFO")
#logger.add(sys.stderr, level="ERROR", format=fmt)  #minimal stderr logger
logger.add("logs/main.log", level="DEBUG", format=fmt) #compression='zip')





def make_shapes_test():    
    bsize = 20
    polygons = [[[c[1],c[2]], [c[1],c[2]+bsize], 
                [c[1]+10,c[2]+bsize], [c[1]+bsize,c[2]] ] 
            for c in centers ]
    
    print(polygons[0:2])

    layer1 = viewer.add_shapes(
        polygons,
        shape_type='polygon',
        edge_width=1,
        edge_color='coral',
        face_color='royalblue',
        name='shapes',
    )

    #layer1.selected_data = set(range(layer1.nshapes))
    layer1.current_edge_width = 5
    layer1.current_opacity = 0.75
    #layer1.selected_data = set()

def setup_entity_table(viewer, cData):
    # Setup Entity Table    
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

    #class_codes = [ entities['class_zode'].iloc[i] for i in range(sel_start, sel_end)]
    face_color_list = [palette[class_code] for class_code in cData.entities["class_code"]]
    
    entity_layer = viewer.add_points(centers, size=[10] * len(centers), opacity=0.5, 
        face_color=face_color_list, n_dimensional=True)

    cData.tabledata = tabledata
    return entity_layer, tabledata


def filter(viewer):
    ## RASTERIZATION (V->R)
    #payload = make_masks(payload)

    cropped_vol = viewer.layers['Original Image'].data

    payload = PipelinePayload(cropped_vol,
                        {'in_array': cropped_vol},
                        np.array([0,0,0,0]),
                        None,
                        None,
                        None,
                        appState.scfg)

    feature_params = [ [gaussian, appState.scfg.filter1['gauss_params']]]

     #   [gaussian, scfg.filter2['gauss_params'] ],
     #   [simple_laplacian, scfg.filter4['laplacian_params'] ],
     #   [tvdenoising3d, scfg.filter3['tvdenoising3d_params'] ]]
    # feature_params = [ 
    # [gaussian, scfg.filter2['gauss_params'] ]]
    # FEATURES (R -> List[R])

    payload = make_features(payload, feature_params)
    logger.info(f"Filtered layer to produce {payload.features.filtered_layers}")
    
    return payload.features.features_stack[0]

        
def frontend(cData):
    #l = Launcher.instance()
    logger.info(f"Connected to launcher {Launcher.g.connected}")
    #print(l.set_remote('localhost:8123'))
    
    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.appState = appState
        viewer.theme = 'light'

        try:
            if len(cData.opacities) != len(cData.vol_stack):
                cData.opacities = [1] * len(cData.layer_names)

            if len(cData.layer_names) != len(cData.vol_stack):
                cData.layer_names = [str(t) for t in list(range(len(cData.vol_stack)))]

            #for vl, layer_name, opacity in zip(cData.vol_stack, cData.layer_names, cData.opacities):
            #    logger.debug(f"Adding layer: {layer_name} of shape {vl.shape}")

#                image_layer = viewer.add_image(vl, name=layer_name, opacity=opacity, colormap='inferno')
 #               image_layer.visible = True

        except Exception as err:
            logger.error(f"Exception at Napari layer setup {err}")
        
        #
        # Load data into viewer
        #
        viewer.add_image(cData.vol_stack[0], name='Original Image')
        viewer.add_labels(cData.vol_anno, name='Active Annotation')
        viewer.add_labels(cData.vol_supervoxels, name='Superregions')
        
        for ii,f in enumerate(cData.features.filtered_layers):
            viewer.add_image(f, name=str(ii))

        viewer.layers['Active Annotation'].opacity = 0.9
   
        #labels_from_pts = layer1.to_labels([viewer.img.shape[1], viewer.img.shape[2]])
        labels_from_pts = viewer.add_labels(cData.vol_anno, name='labels')
        labels_from_pts.visible = False

        #
        # Entities
        #
        logger.debug("Creating entities.")    
        entity_layer, tabledata = setup_entity_table(viewer, cData)

        pipeline_gui_widget = pipeline_gui.Gui()
        viewer.window.add_dock_widget(pipeline_gui_widget, area='right')
        viewer.layers.events.changed.connect(lambda x: pipeline_gui_widget.refresh_choices())
        pipeline_gui_widget.refresh_choices()

        viewer.dw = AttrDict()
        viewer.dw.bpw = ButtonPanelWidget()
        viewer.dw.ppw = PluginPanelWidget()
        #viewer.dw.datatree_control = DataTreeControl(appState.scfg)
        viewer.dw.table_control = TableWidget()        
        viewer.dw.table_control.set_data(tabledata)
        
        vol1 = sample_roi(cData.vol_stack[0], tabledata, vol_size=(32,32,32))
        logger.debug(f"Sampled ROI vol of shape {vol1.shape}")
                
        viewer.dw.smallvol_control = SmallVolWidget(vol1)
        appState.scfg.object_table = viewer.dw.table_control
        
        #cluster_control = ClusterWidget(scfg)
        #cluster2d_control = Cluster2dWidget(pts, colors, cData.classnames)

        def processEvents(x):
            logger.info(f"Received event {x}")
            #
            # todo: move into api and call via launcher
            #

            if x['data']=='predict':
                predicted = np.zeros_like(cData.vol_stack[0])
                viewer.add_image(predicted, name='Prediction')
                #scfg.segSubject.notify() 
                #segment(viewer)
 
            # crop the vol, apply the pipeline, display the result in the roi viewer and add
            # an image that is blank except for the roi region back to the viewer
            # when the pipeline is ready to be applied to the whole image, set the 
            # roi to the size of the image and reapply

            elif x['data']=='pipeline':                 
                minVal, maxVal = 0, 10
                with ProgressDialog("Processing..", minVal, maxVal) as dlg:                    
                    dlg.setValue(1)         
                
                    if dlg.wasCanceled():
                        raise Exception("Processing canceled")
            
                    entity_pts = np.array(make_entity_df(np.array(cData.entities))) #np.array(entities_df)
                    crop_coords = [(0,0,0), (16,450,450),(32,700,700),(32,850,550),(32,500,500),(32,1450,1450),]
                    z_st, z_end, x_st, x_end, y_st, y_end = appState.scfg['roi_crop']

                    crop_coord = crop_coords[0]
                    crop_coord = [z_st, x_st, y_st]
                    #st_z, st_x, st_y = crop_coord
                    #precropped_vol_size = 128, 1450,1450
                    precropped_vol_size=z_end-z_st,x_end-x_st,y_end-y_st
                    #overall_offset=(st_z,st_x,st_y)
                    logger.info("Applying pipeline to cropped vol at loc {crop_coord} and size {precropped_vol_size}")
                    
                    cropped_vol, precropped_pts = crop_vol_and_pts(cData.vol_stack[0],entity_pts,
                            location=crop_coord,
                            patch_size=precropped_vol_size,
                            debug_verbose=True,
                            offset=True)

                    dlg.setValue(3)         

                    if appState.scfg['pipeline_option']=='saliency_pipeline':
                        logger.info("Saliency pipeline")
                        models = {}
                        models['saliency_model'] = setup_cnn_model()

                        payload = PipelinePayload(cropped_vol,
                                {'in_array': cropped_vol},
                                precropped_pts,
                                None,
                                None,
                                None,
                                models,
                                appState.scfg)

                        payload = saliency_pipeline(payload)
                        viewer.add_image(payload.layers['saliency_bb'], name='BB', colormap='cyan')
                                                
                    elif appState.scfg['pipeline_option']=='prediction_pipeline':
                        logger.info("Prediction pipeline")
                        payload = PipelinePayload(cropped_vol,
                                {'in_array': cropped_vol},
                                precropped_pts,
                                None,
                                None,
                                None,
                                None,
                                appState.scfg)
                        payload = prediction_pipeline(payload)
                        viewer.add_labels(payload.layers['total_mask'], name='Masks')
                        viewer.add_labels(payload.superregions.supervoxel_vol, name='SR')


                    elif appState.scfg['pipeline_option']=='mask_pipeline':
                        logger.info("Mask pipeline")
                        payload = PipelinePayload(cropped_vol,
                                {'in_array': cropped_vol},
                                precropped_pts,
                                None,
                                None,
                                None,
                                None,
                                appState.scfg)
                        payload = mask_pipeline(payload)
                        viewer.add_labels(payload.layers['core_mask'], name='Masks')

                    dlg.setValue(9)         
   
                    result = payload.layers['result']
                    logger.info(f"Produced pipeline result: {result.shape}")                
                    viewer.dw.smallvol_control.set_vol(result)
                    result = np.zeros_like(cData.vol_stack[0])
                    
                    result[crop_coord[0]:crop_coord[0]+precropped_vol_size[0],
                    crop_coord[1]:crop_coord[1]+precropped_vol_size[1],
                    crop_coord[2]:crop_coord[2]+precropped_vol_size[2]] = payload.layers['result']
                    
                    viewer.add_image(result, name='Pipeline Result')
                    
                    dlg.setValue(10)         
            
            elif x['data']=='calc_supervoxels':
                logger.debug(f"calc_supervoxels clicked {x['compactness']}")
                appState.scfg.slic_params['compactness'] = x['compactness']
                minVal, maxVal = 0, 10
                with ProgressDialog("Processing..", minVal, maxVal) as dlg:                    

                    superregions = generate_supervoxels(np.array(cData.features.dataset_feats), 
                                                        cData.features.features_stack, 
                                                        appState.scfg.feats_idx, appState.scfg.slic_params)

                    for i in range(maxVal):
                        time.sleep(0.1)
                        dlg.setValue(i)         
                    if dlg.wasCanceled():
                        raise Exception("Processing canceled")
            
                viewer.layers['Superregions'].data = superregions.supervoxel_vol.astype(np.uint16)
                logger.info("Calculated superregions: {superregions")

            elif x['data']=='calc_features':
                logger.debug(f"Calc features with params {x['sigma']}")
                logger.debug(f"predict clicked {x['count']} times")
                
                result = filter(viewer)
                logger.info(f"Processed image with result {result.shape}")
                viewer.add_image(result, name='Filtered Image')
                #features = generate_features(cData.vol_supervoxels, feature_params, scfg.roi_crop, scfg.resample_amt)
           
            #elif x['data'] == 'predict_saliency':
            #   logger.debug(f"detect clicked {x['count']} times")

            elif x['data'] == 'view_feature':
                logger.debug(f"view_feature {x['feature_id']}")

                #Launcher.g.run('workspace','get_dataset', workspace="test_s12", dataset='biovol')
                #Launcher.g.run('workspace','list_datasets', workspace=DataModel.g.current_workspace)

                # use DatasetManager to load feature from workspace as array and then add it to viewer
                src = DataModel.g.dataset_uri(x['feature_id'], group='features')
                with DatasetManager(src, out=None, dtype='float32', fillvalue=0) as DM:
                    print(DM.sources[0].shape)
                    src_dataset = DM.sources[0]
                    src_arr = src_dataset[:]
                    viewer.add_image(src_arr, name=x['feature_id'])
                    
            elif x['data'] == 'view_supervoxels':
                logger.debug(f"view_feature {x['region_id']}")

                #Launcher.g.run('workspace','get_dataset', workspace="test_s12", dataset='biovol')
                #Launcher.g.run('workspace','list_datasets', workspace=DataModel.g.current_workspace)

                #DataModel.g.current_workspace = test_workspace_name
                src = DataModel.g.dataset_uri(x['region_id'], group='regions')

                with DatasetManager(src, out=None, dtype='float32', fillvalue=0) as DM:
                    print(DM.sources[0].shape)
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
                    
                    #pts = pts[:,[0,2,1]]
                    #entity_layer = viewer.add_points(pts, size=[10] * len(pts),  n_dimensional=True)
                    cData.entities = make_entity_df(pts, flipxy=False)
                    entity_layer, tabledata = setup_entity_table(viewer, cData)
                    
            
            elif x['data']=='show_roi':
                selected_roi_idx = viewer.dw.table_control.w.selected_row 
                logger.info(f"Showing ROI {x['selected_roi']}")
                vol1 = sample_roi(cData.vol_stack[0], cData.tabledata, selected_roi_idx, vol_size=(32,32,32))
                #vol1 = sample_region_at_pt(cData.vol_stack[0], [64,64,64], (32,32,32))
                print(f'Sampled vol: {vol1.shape}')

                #viewer.dw.smallvol_control.set_vol(np.transpose(vol1, (0,2,1)))                        
                viewer.dw.smallvol_control.set_vol(vol1)
                logger.info(f"Sampled ROI vol of shape: {vol1.shape}")
        
        #
        # Add widgets to viewer
        #
        bpw_control_widget = viewer.window.add_dock_widget(viewer.dw.bpw, area='left')                
        #ppw_control_widget = viewer.window.add_dock_widget(viewer.dw.ppw, area='right')                
        
        #viewer.dw.bpw.events.subscribe(lambda x: processEvents(x)  )
        viewer.dw.bpw.clientEvent.connect(lambda x: processEvents(x)  )  
        viewer.dw.ppw.clientEvent.connect(lambda x: processEvents(x)  )  
        
        # todo: fix hack connection
        appState.scfg.ppw = viewer.dw.ppw
        
        #viewer.dw.table_control.w.events.subscribe(lambda x: processEvents(x)  ) 
        viewer.dw.table_control.clientEvent.connect(lambda x: processEvents(x)  ) 
        #table_control_widget = viewer.window.add_dock_widget(viewer.dw.table_control.w, area='bottom')        
        #datatree_control_widget = viewer.window.add_dock_widget(viewer.dw.datatree_control.w, area='right')
        #smallvol_control_widget = viewer.window.add_dock_widget(viewer.dw.smallvol_control.imv, area='left')

        #cluster_control_widget = viewer.window.add_dock_widget(cluster_control.w, area='right')                
        #cluster2d_control_widget = viewer.window.add_dock_widget(cluster2d_control.w, area='right')
        #tree_control_widget = viewer.window.add_dock_widget(tree_control.w, area='left')
        #parametertree_control_widget = viewer.window.add_dock_widget(viewer.dw.parametertree_control.w, area='right')
                
        #masks = layer1.to_masks([512, 512])
        #masks_layer = viewer.add_image(masks.astype(float), name='masks')
        #masks_layer.opacity = 0.7
        #masks_layer.colormap = Colormap(
        #    [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0]]
        #)

   
        #
        # magicgui
        #
        roi_gui_widget = roi_gui.Gui()
        viewer.window.add_dock_widget(roi_gui_widget, area='top')
        # sync dropdowns with layer model
        viewer.layers.events.changed.connect(lambda x: roi_gui_widget.refresh_choices())
        roi_gui_widget.refresh_choices()
 
        #feats_gui_widget = features_gui.Gui()
        #viewer.window.add_dock_widget(feats_gui_widget, area='left')
        #viewer.layers.events.changed.connect(lambda x: feats_gui_widget.refresh_choices())
        #feats_gui_widget.refresh_choices()
 
        workspace_gui_widget = workspace_gui.Gui()
        viewer.window.add_dock_widget(workspace_gui_widget, area='left')
        viewer.layers.events.changed.connect(lambda x: workspace_gui_widget.refresh_choices())
        workspace_gui_widget.refresh_choices()
 
        #sv_gui_widget = sv_gui.Gui()
        #viewer.window.add_dock_widget(sv_gui_widget, area='right')
        #viewer.layers.events.changed.connect(lambda x: sv_gui_widget.refresh_choices())
        #sv_gui_widget.refresh_choices()
 
        prediction_gui_widget = prediction_gui.Gui()
        viewer.window.add_dock_widget(prediction_gui_widget, area='right')
        viewer.layers.events.changed.connect(lambda x: prediction_gui_widget.refresh_choices())
        prediction_gui_widget.refresh_choices()

        #layout = QGridLayout()
        #self.setLayout(layout)
        
        from qtpy.QtWidgets import QLabel, QTabWidget, QVBoxLayout, QPushButton, QWidget
        tabwidget = QTabWidget()
        tab1 = QWidget()
        tab2 = QWidget()
        tab3 = QWidget()
        tab4 = QWidget()
        
        tabwidget.addTab(tab1, "SV")
        #tabwidget.addTab(tab2, "Commit")
        tabwidget.addTab(tab3, "Entities")
        tabwidget.addTab(tab4, "Entity Viewer")
       
        tab1.layout = QVBoxLayout()
        tab1.setLayout(tab1.layout)
        tab1.layout.addWidget(viewer.dw.ppw)
        
        #tab2.layout = QVBoxLayout()
        #tab2.setLayout(tab2.layout)
        #tab2.layout.addWidget(workspace_gui_widget)

        tab3.layout = QVBoxLayout()
        tab3.setLayout(tab3.layout)
        tab3.layout.addWidget(viewer.dw.table_control.w)

        tab4.layout = QVBoxLayout()
        tab4.setLayout(tab4.layout)
        tab4.layout.addWidget(viewer.dw.smallvol_control.imv)

        #layout.addWidget(tabwidget, 0, 0)
        viewer.window.add_dock_widget(tabwidget, area='right')
      
         
        from survos2.entity.sampler import sample_region_at_pt
        sample_region_at_pt(cData.vol_stack[0], [50,50,50], (32,32,32))
        @entity_layer.mouse_drag_callbacks.append
        def get_connected_component_shape(layer, event):            
            print(f"Clicked: {layer}, {event}")
            coords = np.round(layer.coordinates).astype(int)
            #if 'Alt' in event.modifiers:
            print(f"Sampling region at {coords}")
            vol1 = sample_region_at_pt(cData.vol_stack[0], coords, (32,32,32))
            print(f'Sampled vol {vol1.shape}')
            viewer.dw.smallvol_control.set_vol(np.transpose(vol1, (0,2,1)))            
            msg = f'Displaying vol of shape {vol1.shape}'
            print(msg)
            
            #layer.status = msg
            #else:
            #    msg = f'clicked on background at {coords}'
            #    print(msg)
