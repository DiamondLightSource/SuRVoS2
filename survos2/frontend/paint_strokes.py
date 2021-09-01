import numpy as np
from loguru import logger
from napari.qt.threading import thread_worker
from skimage.draw import line

from survos2.frontend.control.launcher import Launcher
from survos2.improc.utils import DatasetManager
from survos2.model import DataModel
from survos2.server.state import cfg
from survos2.frontend.plugins.annotations import dilate_annotations
from survos2.utils import decode_numpy

@thread_worker
def paint_strokes(
    msg,
    drag_pts,
    layer,
    top_layer,
    anno_layer,
    parent_level=None,
    parent_label_idx=None,
    viewer_order=(0, 1, 2),
):
    level = msg["level_id"]
    anno_layer_shape = anno_layer.data.shape
    anno_layer_shape = [anno_layer_shape[i] for i in viewer_order]

    if len(drag_pts) == 0:
        return

    try:
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

        if len(drag_pts[0]) == 2:
            px, py = drag_pts[0]
            z = cfg.current_slice
        else:
            logger.info(f"drag_pts[0] {drag_pts}")
            pt_data = drag_pts[0]
            pt_data = [pt_data[i] for i in viewer_order]
            z, px, py = pt_data
            logger.debug(f"z px py {z} {px} {py}")

        if len(drag_pts[0]) == 2:
            # depending on the slice mode we need to handle either 2 or 3 coordinates
            for x, y in drag_pts[1:]:
                if x < anno_layer.data.shape[0] and y < anno_layer.data.shape[1]:
                    yy, xx = line(py, px, y, x)
                    line_x.extend(xx)
                    line_y.extend(yy)
                    py, px = y, x
                    anno_shape = (anno_layer.data.shape[0], anno_layer.data.shape[1])
                    
        else:  # cfg.retrieval_mode == 'volume':
            for zz, x, y in drag_pts[1:]:
                pt_data = [zz, x, y]
                pt_data = [pt_data[i] for i in viewer_order]
                zz, x, y = pt_data

                if x < anno_layer_shape[1] and y < anno_layer_shape[2]:
                    yy, xx = line(py, px, y, x)
                    line_x.extend(xx)
                    line_y.extend(yy)
                    py, px = y, x

            anno_shape = (anno_layer_shape[1], anno_layer_shape[2])

        line_x = np.array(line_x)
        line_y = np.array(line_y)

        if len(line_y) > 0:

            all_regions = set()

            # Check if we are painting using supervoxels, if not, annotate voxels
            if cfg.current_supervoxels == None:
                line_y, line_x = dilate_annotations(
                    line_x,
                    line_y,
                    anno_shape,
                    top_layer.brush_size,
                )
                params = dict(workspace=True, level=level, label=sel_label)
                yy, xx = list(line_y), list(line_x)
                yy = [int(e) for e in yy]
                xx = [int(e) for e in xx]

                # todo: preserve existing
                params.update(
                    slice_idx=int(z),
                    yy=yy,
                    xx=xx,
                    parent_level=parent_level,
                    parent_label_idx=parent_label_idx,
                    full=False,
                    viewer_order=viewer_order,
                )

                result = Launcher.g.run("annotations", "annotate_voxels", **params)

            # we are painting with supervoxels, so check if we have a current supervoxel cache
            # if not, get the supervoxels from the server
            else:
                line_x, line_y = dilate_annotations(
                    line_x,
                    line_y,
                    anno_shape,
                    top_layer.brush_size,
                )

                supervoxel_size = 40
                bb = np.array(
                    [
                        max(0, int(z) - supervoxel_size),
                        max(0, min(line_x) - supervoxel_size),
                        max(0, min(line_y) - supervoxel_size),
                        int(z) + supervoxel_size,
                        max(line_x) + supervoxel_size,
                        max(line_y) + supervoxel_size,
                    ]
                )
                bb = bb.tolist()
                logger.debug(f"BB: {bb}")

                if cfg.supervoxels_cached == False:                    
                    if cfg.retrieval_mode == 'volume':
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
                    else:
                        regions_dataset = DataModel.g.dataset_uri(
                            cfg.current_regions_name, group="regions"
                        )
                        params = dict(
                            workpace=True,
                            src=regions_dataset,
                            slice_idx=cfg.current_slice,
                            order=cfg.order,
                        )
                        result = Launcher.g.run("regions", "get_slice", **params)
                        if result:
                            sv_arr = decode_numpy(result)
                        
                    logger.debug(f"Loaded superregion array of shape {sv_arr.shape}")

                    cfg.supervoxels_cache = sv_arr
                    cfg.supervoxels_cached = True
                    cfg.current_regions_dataset = regions_dataset
                else:
                    sv_arr = cfg.supervoxels_cache

                if cfg.retrieval_mode != 'slice':
                    viewer_order_str = "".join(map(str, viewer_order))
                    if viewer_order_str != "012" and len(viewer_order_str) == 3:
                        sv_arr = np.transpose(sv_arr, viewer_order)
                        logger.debug(f"After viewer_order transform {sv_arr.shape}")

                for x, y in zip(line_x, line_y):
                    if cfg.retrieval_mode=='slice':
                        if (
                            x >= 0
                            and x < sv_arr.shape[0]
                            and y >= 0
                            and y < sv_arr.shape[1]
                        ):
                            sv = sv_arr[x, y]
                            all_regions |= set([sv])
                    else:
                        if (
                            z >= 0
                            and z < sv_arr.shape[0]
                            and x >= 0
                            and x < sv_arr.shape[1]
                            and y >= 0
                            and y < sv_arr.shape[2]
                        ):
                            sv = sv_arr[z, x, y]
                            all_regions |= set([sv])

                # Commit annotation to server
                params = dict(workspace=True, level=level, label=sel_label)

                params.update(
                    region=cfg.current_regions_dataset,
                    r=list(map(int, all_regions)),
                    modal=False,
                    parent_level=parent_level,
                    parent_label_idx=parent_label_idx,
                    full=False,
                    bb=bb,
                    viewer_order=viewer_order,
                )
                result = Launcher.g.run("annotations", "annotate_regions", **params)
    except Exception as e:
        logger.debug(f"paint_strokes Exception: {e}")
