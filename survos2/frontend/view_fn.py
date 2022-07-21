from survos2.entity.entities import make_entity_df
from loguru import logger
import numpy as np
from survos2.improc.utils import DatasetManager
from survos2.model import DataModel, Workspace
from survos2.frontend.control.launcher import Launcher
from survos2.server.state import cfg
from survos2.utils import decode_numpy, decode_numpy_slice
from survos2.frontend.utils import get_array_from_dataset, get_color_mapping
from survos2.frontend.components.entity import setup_entity_table
from skimage.segmentation import find_boundaries
from skimage import img_as_ubyte
import seaborn as sns


def remove_layer(viewer, layer_name):
    logger.debug(f"Removing layer {layer_name}")
    existing_layer = [v for v in viewer.layers if v.name == layer_name]
    if len(existing_layer) > 0:
        viewer.layers.remove(existing_layer[0])


def view_feature(viewer, msg, new_name=None):
    logger.debug(f"view_feature {msg['feature_id']}")
    existing_feature_layer = [v for v in viewer.layers if v.name == msg["feature_id"]]

    if cfg.retrieval_mode == "slice":
        features_src = DataModel.g.dataset_uri(msg["feature_id"], group="features")
        remove_layer(viewer, cfg.current_feature_name)
        cfg.current_feature_name = msg["feature_id"]
        params = dict(
            workpace=True,
            src=features_src,
            slice_idx=cfg.current_slice,
            order=cfg.order,
        )

        result = Launcher.g.run("features", "get_slice", **params)
        if result:
            src_arr = decode_numpy_slice(result)

            if len(existing_feature_layer) > 0:
                existing_feature_layer[0].data = src_arr.copy()
            else:
                if new_name:
                    viewer.add_image(src_arr, name=new_name)
                else:
                    viewer.add_image(src_arr, name=msg["feature_id"])

    elif cfg.retrieval_mode == "volume_http":
        features_src = DataModel.g.dataset_uri(msg["feature_id"], group="features")
        remove_layer(viewer, cfg.current_feature_name)
        cfg.current_feature_name = msg["feature_id"]
        params = dict(
            workpace=True,
            src=features_src,
        )

        result = Launcher.g.run("features", "get_volume", **params)
        if result:
            src_arr = decode_numpy(result)
            cfg.base_dataset_shape = src_arr.shape
            cfg.slice_max = cfg.base_dataset_shape[0]
            if len(existing_feature_layer) > 0:
                existing_feature_layer[0].data = src_arr.copy()
            else:
                if new_name:
                    viewer.add_image(src_arr, name=new_name)
                else:
                    viewer.add_image(src_arr, name=msg["feature_id"])
                cfg.bpw.display_histogram_plot(src_arr.ravel())

    elif cfg.retrieval_mode == "volume":
        # use DatasetManager to load feature from workspace as array and then add it to viewer
        src = DataModel.g.dataset_uri(msg["feature_id"], group="features")
        remove_layer(viewer, cfg.current_feature_name)
        cfg.current_feature_name = msg["feature_id"]

        with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
            src_dataset = DM.sources[0][:]
            src_arr = get_array_from_dataset(src_dataset)
            src_arr = np.nan_to_num(src_arr)
            cfg.base_dataset_shape = src_arr.shape
            cfg.slice_max = cfg.base_dataset_shape[0]
            if new_name:
                viewer.add_image(src_arr, name=new_name)
            else:
                viewer.add_image(src_arr, name=msg["feature_id"])

            cfg.bpw.display_histogram_plot(src_arr.ravel())
                


def view_regions(viewer, msg):
    logger.debug(f"view_feature {msg['region_id']}")
    region_name = msg["region_id"]
    existing_regions_layer = [
        v for v in viewer.layers if v.name == cfg.current_regions_name
    ]
    region_opacity = 0.3
    if len(existing_regions_layer) > 0:
        region_opacity = existing_regions_layer[0].opacity
        remove_layer(viewer, cfg.current_regions_name)
        cfg.current_regions_name = None
    if cfg.retrieval_mode == "slice":
        regions_src = DataModel.g.dataset_uri(region_name, group="superregions")
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
            if len(existing_regions_layer) > 0:
                existing_regions_layer[0].data = src_arr.copy()
                existing_regions_layer[0].opacity = region_opacity
            else:
                sv_layer = viewer.add_image(src_arr, name=region_name)
                sv_layer.opacity = region_opacity
                sv_layer.colormap = "bop orange"
    elif cfg.retrieval_mode == "volume_http":
        regions_src = DataModel.g.dataset_uri(region_name, group="superregions")
        params = dict(
            workpace=True,
            src=regions_src,
            slice_idx=cfg.current_slice,
            order=cfg.order,
        )
        result = Launcher.g.run("superregions", "get_volume", **params)
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
        src = DataModel.g.dataset_uri(region_name, group="superregions")
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


def view_pipeline(viewer, msg, analyzers=False):
    try:
        logger.debug(f"view_pipeline {msg['pipeline_id']} using {msg['level_id']}")
        source = msg["source"]

        result = Launcher.g.run(
            "annotations", "get_levels", workspace=DataModel.g.current_workspace
        )

        if result:
            cmapping, _ = get_color_mapping(result, level_id=msg["level_id"])

        existing_pipeline_layer = [v for v in viewer.layers if v.name == msg["pipeline_id"]]
        
        if analyzers:
            source = 'analyzer'
            remove_layer(viewer, cfg.current_analyzers_name)
            cfg.current_analyzers_name = msg["pipeline_id"]
            
        else:
            remove_layer(viewer, cfg.current_pipeline_name)
            cfg.current_pipeline_name = msg["pipeline_id"]

        pipeline_src = DataModel.g.dataset_uri(msg["pipeline_id"], group=source)
            
        if cfg.retrieval_mode == "slice":
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
                    existing_pipeline_layer[0].data = src_arr.astype(np.uint32).copy()
                else:
                    viewer.add_labels(
                        src_arr.astype(np.uint32),
                        name=msg["pipeline_id"],
                        color=cmapping,
                    )
        elif cfg.retrieval_mode == "volume_http":
            params = dict(workpace=True, src=pipeline_src)
            result = Launcher.g.run("features", "get_volume", **params)
            if result:
                src_arr = decode_numpy(result)
                if len(existing_pipeline_layer) > 0:
                    existing_pipeline_layer[0].data = src_arr.astype(np.uint32).copy()
                else:
                    viewer.add_labels(
                        src_arr.astype(np.uint32),
                        name=msg["pipeline_id"],
                        color=cmapping,
                    )
        elif cfg.retrieval_mode == "volume":
            
            with DatasetManager(pipeline_src, out=None, dtype="uint32", fillvalue=0) as DM:
                src_dataset = DM.sources[0][:]
                src_arr = get_array_from_dataset(src_dataset)
                existing_layer = [v for v in viewer.layers if v.name == msg["pipeline_id"]]

                if len(existing_layer) > 0:
                    logger.debug(
                        f"Removing existing layer and re-adding it with new colormapping. {existing_layer}"
                    )
                    existing_layer[0].data = src_arr.astype(np.uint32).copy()
                else:
                    viewer.add_labels(
                        src_arr.astype(np.uint32),
                        name=msg["pipeline_id"],
                        color=cmapping,
                    )

                cfg.bpw.display_histogram_plot(src_arr.ravel())
    except AttributeError as e:
        print(e)


def view_objects(viewer, msg):
    logger.debug(f"view_objects {msg['objects_id']}")
    objects_src = DataModel.g.dataset_uri(msg["objects_id"], group="objects")
    
    params = dict(
        workpace=True,
        src=objects_src,
    )
    result = Launcher.g.run("objects", "get_entities", **params)
    if result:
        entities_arr = decode_numpy(result)
      
    #with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
    #    ds_objects = DM.sources[0]

    logger.debug(f"Got entities_arr of shape {entities_arr.shape}")
    
    # objects_fullname = entities_metadata["fullname"]
    # objects_scale = entities_metadata["scale"]
    # objects_offset = entities_metadata["offset"]
    # objects_crop_start = entities_metadata["crop_start"]
    # objects_crop_end = entities_metadata["crop_end"]

    entities_df = make_entity_df(entities_arr)

    tabledata, entities_df = setup_entity_table(
        entities_fullname=None,
        entities_df=entities_df,
        # scale=objects_scale,
        # offset=objects_offset,
        # crop_start=objects_crop_start,
        # crop_end=objects_crop_end,
        flipxy=msg["flipxy"],
    )
    sel_start, sel_end = 0, len(entities_df)

    centers = np.array(
        [
            [
                int((float(entities_df.iloc[i]["z"]) * 1.0) + 0),
                int((float(entities_df.iloc[i]["x"]) * 1.0) + 0),
                int((float(entities_df.iloc[i]["y"]) * 1.0) + 0),
            ]
            for i in range(sel_start, sel_end)
        ]
    )

    num_classes = max(9, len(np.unique(entities_df["class_code"]))) + 2

    logger.debug(f"Number of entity classes {num_classes}")
    palette = np.array(sns.color_palette("hls", num_classes))
    face_color_list = [palette[class_code] for class_code in entities_df["class_code"]]

    entity_layer = viewer.add_points(
        centers, size=[10] * len(centers), opacity=0.5, face_color=face_color_list
    )



def view_entities(viewer, msg):
    logger.debug(f"view_entities {msg['objects_id']}")
    
    entities_arr = decode_numpy(msg['entities'])
      
    logger.debug(f"Got entities_arr of shape {entities_arr.shape}")

    entities_df = make_entity_df(entities_arr)

    tabledata, entities_df = setup_entity_table(
        entities_fullname=None,
        entities_df=entities_df,
        # scale=objects_scale,
        # offset=objects_offset,
        # crop_start=objects_crop_start,
        # crop_end=objects_crop_end,
        flipxy=msg["flipxy"],
    )
    sel_start, sel_end = 0, len(entities_df)

    centers = np.array(
        [
            [
                int((float(entities_df.iloc[i]["z"]) * 1.0) + 0),
                int((float(entities_df.iloc[i]["x"]) * 1.0) + 0),
                int((float(entities_df.iloc[i]["y"]) * 1.0) + 0),
            ]
            for i in range(sel_start, sel_end)
        ]
    )

    num_classes = max(9, len(np.unique(entities_df["class_code"]))) + 2

    logger.debug(f"Number of entity classes {num_classes}")
    palette = np.array(sns.color_palette("hls", num_classes))
    face_color_list = [palette[class_code] for class_code in entities_df["class_code"]]

    entity_layer = viewer.add_points(
        centers, size=[10] * len(centers), opacity=0.5, face_color=face_color_list
    )


