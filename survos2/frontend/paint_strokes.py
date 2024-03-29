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
from survos2.api.annotate import get_order

_MaskSize = 4  # 4 bits per history label
_MaskCopy = 15  # 0000 1111
_MaskPrev = 240  # 1111 0000


def get_line_points(drag_pts, anno_layer_shape, viewer_order):
    line_x, line_y = [], []

    pt_data = drag_pts[0]
    pt_data = [pt_data[i] for i in viewer_order]
    z, px, py = pt_data

    for zz, x, y in drag_pts[1:]:
        pt_data = [zz, x, y]
        pt_data = [pt_data[i] for i in viewer_order]
        zz, x, y = pt_data

        if x < anno_layer_shape[1] and y < anno_layer_shape[2]:
            yy, xx = line(py, px, y, x)
            line_x.extend(xx)
            line_y.extend(yy)
            py, px = y, x
    return np.array(line_x), np.array(line_y), z


def annotate_voxels(
    line_x,
    line_y,
    z,
    level,
    sel_label,
    brush_size,
    anno_shape,
    parent_level,
    parent_label_idx,
    viewer_order,
):
    ########################################################################
    line_y, line_x = dilate_annotations(
        line_x,
        line_y,
        anno_shape,
        brush_size,
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
        workspace=DataModel.g.current_workspace,
    )

    result = Launcher.g.run("annotations", "annotate_voxels", json_transport=True, **params)


def annotate_regions(
    line_x,
    line_y,
    z,
    level,
    sel_label,
    brush_size,
    anno_shape,
    parent_level,
    parent_label_idx,
    viewer_order,
):
    ################################################################################
    # we are painting with supervoxels, so check if we have a current supervoxel cache
    # if not, get the supervoxels from the server

    line_x, line_y = dilate_annotations(
        line_x,
        line_y,
        anno_shape,
        brush_size,
    )

    supervoxel_size = cfg.supervoxel_size * 2
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

    if cfg.supervoxels_cached == False:
        regions_dataset = DataModel.g.dataset_uri(cfg.current_regions_name, group="superregions")
        with DatasetManager(
            regions_dataset,
            out=None,
            dtype="uint32",
            fillvalue=0,
        ) as DM:
            src_dataset = DM.sources[0]
            sv_arr = src_dataset[:]

        cfg.supervoxels_cache = sv_arr
        cfg.supervoxels_cached = True
        cfg.current_regions_dataset = regions_dataset
    else:
        sv_arr = cfg.supervoxels_cache

    viewer_order_str = "".join(map(str, viewer_order))
    if viewer_order_str != "012" and len(viewer_order_str) == 3:
        sv_arr = np.transpose(sv_arr, viewer_order)

    all_regions = set()
    for x, y in zip(line_x, line_y):
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

    if cfg.remote_annotation:
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
        result = Launcher.g.run("annotations", "annotate_regions", json_transport=True, **params)
    else:
        _annotate_regions_local(
            cfg.anno_data.copy(),
            sv_arr,
            list(map(int, all_regions)),
            label=sel_label,
            parent_level=parent_level,
            parent_label_idx=parent_label_idx,
            bb=bb,
            viewer_order=viewer_order,
        )


@thread_worker
def paint_strokes(
    msg,
    drag_pts,
    anno_layer,
    parent_level=None,
    parent_label_idx=None,
    viewer_order=(0, 1, 2),
):
    """
    Gather all information required from viewer and call annotation function
    """
    level = msg["level_id"]
    anno_layer_shape = anno_layer.data.shape
    anno_layer_shape = [anno_layer_shape[i] for i in viewer_order]

    if len(viewer_order) == 2:
        viewer_order = (0, 1, 2)

    if len(drag_pts) == 0:
        return

    try:
        sel_label = int(cfg.label_value["idx"]) - 1
        anno_layer.selected_label = sel_label
        anno_layer.brush_size = int(cfg.brush_size)

        if anno_layer.mode == "erase":
            sel_label = 0
            cfg.current_mode = "erase"
        else:
            cfg.current_mode = "paint"

        line_x, line_y, z = get_line_points(drag_pts, anno_layer_shape, viewer_order)

        if len(line_y) > 0:
            anno_shape = (anno_layer_shape[1], anno_layer_shape[2])

            # Check if we are painting using supervoxels, if not, annotate voxels
            if cfg.current_supervoxels is None:
                annotate_voxels(
                    line_x,
                    line_y,
                    z,
                    level,
                    sel_label,
                    anno_layer.brush_size,
                    anno_shape,
                    parent_level,
                    parent_label_idx,
                    viewer_order,
                )
            else:
                annotate_regions(
                    line_x,
                    line_y,
                    z,
                    level,
                    sel_label,
                    anno_layer.brush_size,
                    anno_shape,
                    parent_level,
                    parent_label_idx,
                    viewer_order,
                )
    except Exception as e:
        logger.debug(f"paint_strokes Exception: {e}")


def _annotate_regions_local(
    level: np.ndarray,
    region: np.ndarray,
    r: list,
    label: int,
    parent_level: str,
    parent_label_idx: int,
    bb: list,
    viewer_order=(0, 1, 2),
):
    from survos2.frontend.frontend import get_level_from_server

    if parent_level != "-1" and parent_level != -1:
        parent_arr, parent_annotations_dataset = get_level_from_server(
            {"level_id": parent_level}, retrieval_mode="volume"
        )
        parent_arr = parent_arr & 15
        parent_mask = parent_arr == parent_label_idx
    else:
        parent_arr = None
        parent_mask = None

    logger.debug(f"BB in annotate_regions {bb}")

    if label >= 16 or label < 0 or type(label) != int:
        raise ValueError("Label has to be in bounds [0, 15]")
    if r is None or len(r) == 0:
        return

    cfg.modified = [0]
    ds = level[:]
    reg = region[:]
    logger.debug(f"reg shape {reg.shape} ds shape {ds.shape}")

    viewer_order_str = "".join(map(str, viewer_order))
    if viewer_order_str != "012" and len(viewer_order_str) == 3:
        # print(f"Viewer order {viewer_order} Performing viewer_order transform {ds.shape}")
        ds_t = np.transpose(ds, viewer_order)

    else:
        # print(f"Viewer order {viewer_order} No viewer_order transform {ds.shape}")
        ds_t = ds

    mask = np.zeros_like(reg)

    # print(f"BB: {bb}")
    try:
        if not bb:
            # print("No mask")
            bb = [0, 0, 0, ds_t.shape[0], ds_t.shape[1], ds_t.shape[2]]

        # else:
        # print(f"Masking using bb: {bb}")
        for r_idx in r:
            mask[bb[0] : bb[3], bb[1] : bb[4], bb[2] : bb[5]] += (
                reg[bb[0] : bb[3], bb[1] : bb[4], bb[2] : bb[5]] == r_idx
            )
    except Exception as e:
        logger.debug(f"__annotate_regions_local exception {e}")

    if parent_mask is not None:
        parent_mask_t = np.transpose(parent_mask, viewer_order)
        # print(f"Using parent mask of shape: {parent_mask.shape}")
        mask = mask * parent_mask_t

    mask = (mask > 0) * 1.0
    mask = mask > 0
    # if not np.any(mask):
    #    modified[i] = (modified[i] << 1) & mbit

    ds_t = (ds_t & _MaskCopy) | (ds_t << _MaskSize)
    ds_t[mask] = (ds_t[mask] & _MaskPrev) | label
    logger.debug(f"Returning annotated region ds {ds.shape}")

    if viewer_order_str != "012" and len(viewer_order_str) == 3:
        new_order = get_order(viewer_order)
        # logger.info(
        #    f"new order {new_order} Dataset before second transpose: {ds_t.shape}"
        # )
        cfg.anno_data = np.transpose(ds_t, new_order)  # .reshape(original_shape)
        # logger.info(f"Dataset after second transpose: {ds_o.shape}")

    else:
        cfg.anno_data = ds_t

    cfg.modified = [1]
