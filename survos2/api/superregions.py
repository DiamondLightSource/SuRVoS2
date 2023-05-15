import ntpath
import numpy as np
from loguru import logger
from skimage.segmentation import slic
from survos2.api import workspace as ws
from survos2.api.utils import dataset_repr, save_metadata, get_function_api
from survos2.data_io import dataset_from_uri
from survos2.improc import map_blocks
from survos2.improc.utils import DatasetManager
from survos2.model import DataModel
from survos2.utils import encode_numpy
from fastapi import APIRouter
from pathlib import Path
from tqdm import tqdm
import urllib.request

__region_fill__ = 0
__region_dtype__ = "uint32"
__region_group__ = "superregions"
__region_names__ = [None, "supervoxels", "sam"]

superregions = APIRouter()


@superregions.get("/get_volume")
def get_volume(src: str):
    logger.debug("Getting region volume")
    ds = dataset_from_uri(src, mode="r")
    data = ds[:]
    return encode_numpy(data)


@superregions.get("/get_slice")
def get_slice(src: str, slice_idx: int, order: tuple):
    ds = dataset_from_uri(src, mode="r")[:]
    ds = np.transpose(ds, order)
    data = ds[slice_idx]
    return encode_numpy(data)


@superregions.get("/get_crop")
def get_crop(src: str, roi: list):
    logger.debug("Getting regions crop")
    ds = dataset_from_uri(src, mode="r")
    data = ds[roi[0] : roi[1], roi[2] : roi[3], roi[4] : roi[5]]
    return encode_numpy(data)


@superregions.get("/supervoxels")
@save_metadata
def supervoxels(
    src: str,
    dst: str,
    mask_id: str,
    n_segments: int = 10,
    compactness: float = 20,
    spacing: list = [1, 1, 1],
    multichannel: bool = False,
    enforce_connectivity: bool = False,
    out_dtype: str = "int",
    zero_parameter: bool = False,
    max_num_iter: int = 10,
):
    with DatasetManager(src, out=None, dtype=out_dtype, fillvalue=0) as DM:
        src_data_arr = DM.sources[0][:]

    # get image feature for mask, if any

    if mask_id == "None":
        mask_feature = None
    else:
        src = DataModel.g.dataset_uri(ntpath.basename(mask_id), group="features")
        logger.debug(f"Getting features {src}")
        with DatasetManager(src, out=None, dtype="uint32", fillvalue=0) as DM:
            mask_feature = DM.sources[0][:].astype(np.uint32)
            logger.debug(f"Feature to use as mask shape {mask_feature.shape}")

    supervoxel_image = slic(
        src_data_arr,
        n_segments=n_segments,
        spacing=spacing,
        compactness=compactness,
        multichannel=False,
        max_num_iter=max_num_iter,
        slic_zero=zero_parameter,
        mask=mask_feature,
    )

    def pass_through(x):
        return x

    map_blocks(pass_through, supervoxel_image, out=dst, normalize=False)



# taken from napari-sam
def download_with_progress(url, output_file):
    req = urllib.request.urlopen(url)
    content_length = int(req.headers.get('Content-Length'))
    progress_bar = tqdm(total=content_length, unit='B', unit_scale=True)

    with open(output_file, 'wb') as f:
        downloaded_bytes = 0
        while True:
            buffer = req.read(8192)
            if not buffer:
                break
            downloaded_bytes += len(buffer)
            f.write(buffer)
            progress_bar.update(len(buffer))
    progress_bar.close()
    req.close()

@superregions.get("/sam")
@save_metadata
def sam(
    src: str,
    dst: str,
    points_per_side: int = 32,
    pred_iou_thresh: float = 0.86,
    stability_score_thresh: float = 0.5,
    crop_n_layers: int = 1,
    crop_n_points_downscale_factor: int = 1,
    min_mask_region_area: int = 1000,
    MAX_NUM_LABELS_PER_SLICE: int = 1000,
    skip: int = 1,
):
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        feature_array = DM.sources[0][:]

    weights_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    
    cache_dir = Path.home() / ".cache/survos"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    #cache_dir = str(Path.home() / ".cache/napari-segment-anything")
    sam_checkpoint = cache_dir / weights_url.split("/")[-1]

    if not sam_checkpoint.exists():
        print("Downloading {} to {} ...".format(weights_url, sam_checkpoint))
        download_with_progress(weights_url, sam_checkpoint)

    print(f"SAM CHECKPOINT: {sam_checkpoint}")
    model_type = "vit_h"
    #sam_checkpoint = "/ceph/users/fot15858/libs/sam_vit_h_4b8939.pth"
    device = "cuda"

    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    from scipy import ndimage as ndi
    from skimage.measure import label, regionprops
    import cv2
    from skimage import img_as_ubyte

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        crop_n_layers=crop_n_layers,
        crop_n_points_downscale_factor=crop_n_points_downscale_factor,
        min_mask_region_area=min_mask_region_area,  # Requires open-cv to run post-processing
    )
    
    MAX_Z = feature_array.shape[0]
    total_vol = np.ones((MAX_Z, feature_array.shape[1], feature_array.shape[2])) * -1
    for slice_num in range(0,MAX_Z, skip):
        print(f"Slice num: {slice_num}")
        img = feature_array[slice_num:slice_num+1,:].T
        img = img_as_ubyte(img)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        masks = mask_generator.generate(image)
        total = np.zeros_like(masks[0]['segmentation'] * 1)
        base_slice_num= (slice_num * MAX_NUM_LABELS_PER_SLICE)
        for i,m in enumerate(masks):
            _slice = np.zeros_like(masks[0]['segmentation'] * 1)
            _slice += ((m['segmentation'] * 1) * (i*10))
            total += _slice + base_slice_num
        total_lab = label(total)
        total_vol[slice_num,:] = total_lab + (MAX_NUM_LABELS_PER_SLICE * slice_num)

    total_vol = total_vol.astype(np.uint32)
    total_vol = np.transpose(total_vol.T, [2,0,1])

    def pass_through(x):
        return x

    map_blocks(pass_through, total_vol, out=dst, normalize=False)



@superregions.get("/supervoxels_chunked")
@save_metadata
def supervoxels_chunked(
    src: list,
    dst: str,
    n_segments: int = 10,
    compactness: float = 20,
    spacing: list = [1, 1, 1],
    multichannel: bool = False,
    enforce_connectivity: bool = False,
    out_dtype="int",
):

    map_blocks(
        slic,
        *src,
        out=dst,
        n_segments=n_segments,
        spacing=spacing,
        compactness=compactness,
        multichannel=False,
        enforce_connectivity=True,
        stack=False,
        timeit=True,
        uses_gpu=False,
        out_dtype=out_dtype,
        relabel=True,
    )

    with DatasetManager(dst, out=None, dtype=out_dtype, fillvalue=0) as DM:
        dst_dataset = DM.sources[0]
        supervoxel_image = dst_dataset[:]

    num_sv = len(np.unique(supervoxel_image))
    logger.debug(f"Number of supervoxels created: {num_sv}")

    dst_dataset.set_attr("num_supervoxels", num_sv)


@superregions.get("/create")
def create(workspace: str, order: int = 1, big: bool = False):
    region_type = __region_names__[order]
    if big:
        logger.debug("Creating int64 regions")
        ds = ws.auto_create_dataset(
            workspace,
            region_type,
            __region_group__,
            __region_dtype__,
            dtype=np.uint64,
            fill=__region_fill__,
        )
    else:
        logger.debug("Creating int32 regions")
        ds = ws.auto_create_dataset(
            workspace,
            region_type,
            __region_group__,
            __region_dtype__,
            # dtype=np.uint32,
            fill=__region_fill__,
        )

    ds.set_attr("kind", region_type)
    return dataset_repr(ds)


@superregions.get("/existing")
def existing(workspace: str, full: bool = False, order: int = 1):
    datasets = ws.existing_datasets(workspace, group=__region_group__)
    if full:
        return {"{}/{}".format(__region_group__, k): dataset_repr(v) for k, v in datasets.items()}
    return {k: dataset_repr(v) for k, v in datasets.items()}


@superregions.get("/remove")
def remove(workspace: str, region_id: str):
    ws.delete_dataset(workspace, region_id, group=__region_group__)

    return {"done": True}


@superregions.get("/rename")
def rename(workspace: str, region_id: str, new_name: str):
    ws.rename_dataset(workspace, region_id, __region_group__, new_name)


@superregions.get("/available")
def available():
    h = superregions  
    all_features = []
    for r in h.routes:
        name = r.name
        if name in [
            "available",
            "create",
            "existing",
            "remove",
            "rename",
            "group",
            "supervoxels_chunked",
            "get_volume",
            "get_slice",
            "get_crop"
        ]:
            continue
        func = r.endpoint
        desc = get_function_api(func)
        category = desc["returns"] or "Others"
        desc = dict(name=name, params=desc["params"], category=category)
        all_features.append(desc)
    return all_features
