import numpy as np
from loguru import logger
from napari.qt.threading import thread_worker
from skimage.draw import line

from survos2.frontend.control.launcher import Launcher
from survos2.improc.utils import DatasetManager
from survos2.model import DataModel
from survos2.server.state import cfg

@thread_worker
def paint_strokes(msg, drag_pts, layer, top_layer, anno_layer, parent_level=None, parent_label_idx=None):
    level = msg["level_id"]

    if len(drag_pts) == 0:
        return
    
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
        z, px, py = drag_pts[0]

    if len(drag_pts[0]) == 2:
    # depending on the slice mode we need to handle either 2 or 3 coordinates
    #if cfg.retrieval_mode == 'slice':
        for x, y in drag_pts[1:]:
            yy, xx = line(py, px, y, x)
            line_x.extend(xx)
            line_y.extend(yy)
            py, px = y, x
            anno_shape = (anno_layer.data.shape[0], anno_layer.data.shape[1])
    else: # cfg.retrieval_mode == 'volume':
        for _, x, y in drag_pts[1:]:
            yy, xx = line(py, px, y, x)
            line_x.extend(xx)
            line_y.extend(yy)
            py, px = y, x
            anno_shape = (anno_layer.data.shape[1], anno_layer.data.shape[2])

    line_y = np.array(line_y)
    line_x = np.array(line_x)
    
    if len(line_y) > 0:
        from survos2.frontend.plugins.annotations import dilate_annotations

        all_regions = set()

        # Check if we are painting using supervoxels, if not, annotate voxels
        if cfg.current_supervoxels == None:
            line_y, line_x = dilate_annotations(
                line_x, line_y, anno_shape, top_layer.brush_size,
            )
            params = dict(workspace=True, level=level, label=sel_label)
            yy, xx = list(line_y), list(line_x)
            yy = [int(e) for e in yy]
            xx = [int(e) for e in xx]

            n_dimensional = top_layer.n_dimensional

            # todo: preserve existing

            params.update(slice_idx=int(z), 
                yy=yy, 
                xx=xx,
                parent_level=parent_level,
                parent_label_idx=parent_label_idx,
                full=False)

            result = Launcher.g.run("annotations", "annotate_voxels", **params)
        # we are painting with supervoxels, so check if we have a current supervoxel cache
        # if not, get the supervoxels from the server
        else:
            line_x, line_y = dilate_annotations(
                line_x, line_y, anno_shape, top_layer.brush_size,
            )

            if cfg.supervoxels_cached == False:
                regions_dataset = DataModel.g.dataset_uri(
                    cfg.current_regions_name, group="regions"
                )

                with DatasetManager(
                    regions_dataset, out=None, dtype="uint32", fillvalue=0,
                ) as DM:
                    src_dataset = DM.sources[0]
                    sv_arr = src_dataset[:]

                logger.debug(f"Loaded superregion array of shape {sv_arr.shape}")
                cfg.supervoxels_cache = sv_arr
                cfg.supervoxels_cached = True
                cfg.current_regions_dataset = regions_dataset
            else:
                sv_arr = cfg.supervoxels_cache

                print(f"Used cached supervoxels of shape {sv_arr.shape}")

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
                parent_level=parent_level,
                parent_label_idx=parent_label_idx,
                full=False
            )
            result = Launcher.g.run("annotations", "annotate_regions", **params)

 