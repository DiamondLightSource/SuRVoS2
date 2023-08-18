from survos2.api.workspace import workspace
from survos2.api.features import features
from survos2.api.annotations import annotations
from survos2.api.superregions import superregions
from survos2.api.objects import objects

from survos2.api.roi import roi
from survos2.api.pipelines import pipelines
from survos2.api.analyzer import analyzer
from survos2.config import Config
import colorama
import logging

from fastapi import FastAPI
import sys
import warnings
from loguru import logger
from numba.core.errors import NumbaWarning
import warnings


colorama.init()  # allows colors in command line output of uvicorn
warnings.simplefilter("ignore", category=NumbaWarning)
numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.ERROR)
# fmt = "{time} - {name} - {level} - {message}"
fmt = "<level>{level} - {message} </level><green> - {name}</green>:<cyan>{function}</cyan>:<cyan>{line}</cyan>"
logger.remove()  # remove default logger
# logger.add(sys.stderr, level="DEBUG", format=fmt, colorize=True)
logger.add(sys.stderr, level=Config["logging.overall_level"], format=fmt, colorize=True)

app = FastAPI(debug=True)
app.include_router(workspace, prefix="/workspace")
app.include_router(features, prefix="/features")
app.include_router(annotations, prefix="/annotations")
app.include_router(superregions, prefix="/superregions")
app.include_router(objects, prefix="/objects")
app.include_router(roi, prefix="/roi")
app.include_router(pipelines, prefix="/pipelines")
app.include_router(analyzer, prefix="/analyzer")
