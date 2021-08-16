from napari_plugin_engine import napari_hook_implementation
from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton
from magicgui import magic_factory
from survos2.model import DataModel
import numpy
import survos2
import sys
import warnings
from loguru import logger 

warnings.filterwarnings("ignore")



# fmt = "{time} - {name} - {level} - {message}"
fmt = "<level>{level} - {message} </level><green> - {name}</green>:<cyan>{function}</cyan>:<cyan>{line}</cyan>"
logger.remove()  # remove default logger
logger.add(
    sys.stderr, level="DEBUG", format=fmt, colorize=True
)  
# logger.add("logs/main.log", level="DEBUG", format=fmt) #compression='zip')
#logger.level("CUSTOM", no=15, color="<blue>", icon="@")


class Workspace(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        self.setLayout(QVBoxLayout())
        from survos2.frontend.frontend import frontend

        self.dw = frontend(napari_viewer)
        self.layout().addWidget(self.dw.ppw)
        self.layout().addWidget(self.dw.bpw)


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return [Workspace]
