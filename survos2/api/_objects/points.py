import pickle
import os
import numpy as np
from loguru import logger
import tempfile


from survos2.api import workspace as ws

from survos2.api.utils import dataset_repr, get_function_api, save_metadata
from survos2.improc.utils import DatasetManager
from survos2.data_io import dataset_from_uri
from survos2.model import DataModel
from survos2.utils import encode_numpy
from survos2.frontend.components.entity import setup_entity_table
from survos2.entity.entities import load_entities_via_file, make_entity_df, make_entity_bvol
from fastapi import APIRouter



objects = APIRouter()



@objects.get("/points", response_model=None)
def points(
    dst: str,
    fullname: str,
    scale: float,
) -> "GEOMETRY":
    with DatasetManager(dst, out=dst, dtype="float32", fillvalue=0) as DM:
        # DM.out[:] = np.zeros_like(img_volume)
        dst_dataset = DM.sources[0]
        dst_dataset.set_attr("scale", scale)

        offset = [0, 0, 0]
        crop_start = [0, 0, 0]
        crop_end = [1e9, 1e9, 1e9]

        dst_dataset.set_attr("offset", offset)
        dst_dataset.set_attr("crop_start", crop_start)
        dst_dataset.set_attr("crop_end", crop_end)

        basename = os.path.basename(fullname)
        # csv_saved_fullname = dst_dataset.save_file(fullname)
        csv_saved_fullname = dst_dataset.save_file(fullname)
        logger.debug(f"Saving {fullname} to {csv_saved_fullname}")
        dst_dataset.set_attr("fullname", csv_saved_fullname)

        return dataset_repr(dst_dataset)