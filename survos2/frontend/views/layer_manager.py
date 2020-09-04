from collections import OrderedDict
from qtpy.QtWidgets import QRadioButton, QPushButton
from qtpy.QtCore import QSize, Signal

from survos2.utils import decode_numpy
from survos2.frontend.views.viewer import Viewer
from survos2.frontend.views.base import register_view

from survos2.frontend.components.base import *
from survos2.frontend.components import PushButton
from survos2.frontend.utils import resource
from survos2.frontend.control import Launcher
from survos2.frontend.plugins.features import FeatureComboBox
from survos2.frontend.plugins.regions import RegionComboBox
from survos2.frontend.plugins.annotations import MultiAnnotationComboBox, AnnotationComboBox
from survos2.frontend.plugins.pipelines import PipelinesComboBox

from survos2.model import DataModel

class CmapComboBox(LazyComboBox):

    def __init__(self, parent=None):
        super().__init__(select=1, parent=parent)

    def fill(self):
        #result = Launcher.g.run('render', 'cmaps')
        result = None
        if result:
            for category in result:
                self.addCategory(category)
                for cmap in result[category]:
                    if cmap == 'Greys':
                        cmap = 'Greys_r'
                    self.addItem(cmap)


class Layer(QCSWidget):

    updated = Signal(object)

    def __init__(self, name='layer', source=None, cmap='gray',
                 parent=None):
        super().__init__(parent=parent)
        self.name = name
        self.source = source or ComboBox()
        self.cmap = cmap or CmapComboBox()
        self.slider = Slider(value=100, label=False, auto_accept=False)
        #self.checkbox = CheckBox(checked=True)
        self.view_btn = PushButton('View')
        self.commit_btn = PushButton('Commit')

        hbox = HBox(self, spacing=5, margin=(5, 0, 5, 0))
        hbox.addWidget(self.source, 1)
        #hbox.addWidget(self.cmap, 1)
        hbox.addWidget(self.slider)
        #hbox.addWidget(self.checkbox)
        hbox.addWidget(self.view_btn)
        hbox.addWidget(self.commit_btn)
        
        #if hasattr(self.source, 'currentIndexChanged'):
        #    self.source.currentIndexChanged.connect(self._params_updated)
        #elif hasattr(self.source, 'valueChanged'):
        #    self.source.valueChanged.connect(self._params_updated)
        
        if hasattr(self.cmap, 'currentIndexChanged'):
            self.cmap.currentIndexChanged.connect(self._params_updated)
        elif hasattr(self.cmap, 'colorChanged'):
            self.cmap.colorChanged.connect(self._params_updated)

        self.slider.setMinimumWidth(150)
        self.slider.valueChanged.connect(self._params_updated)
        #self.checkbox.toggled.connect(self._params_updated)
        self.commit_btn.clicked.connect(self._commit)

    def _commit(self):
        logger.debug(f"_commit: {self.name}")

    def value(self):
        return (self.source.value(), self.cmap.value(),
                self.slider.value(), self.checkbox.value())

    def _params_updated(self):
        self.updated.emit()

    def accept(self):
        self.slider.accept()

    def select(self, view):
        self.source.select(view)
        self.updated.emit()


class DataLayer(Layer):
    def __init__(self, name):
        super().__init__(name, Label('Raw Data'))

    def value(self):
        return ('__data__', self.cmap.value(),
                self.slider.value(), self.checkbox.value())


class FeatureLayer(Layer):
    def __init__(self, name):
        super().__init__(name, FeatureComboBox(full=True))


class RegionLayer(Layer):
    def __init__(self, name):
        region = RegionComboBox(full=True)
        color = ColorButton('#0D47A1')
        super().__init__(name, region, color)


class PipelinesLayer(Layer):
    def __init__(self, name):
        pipeline = PipelinesComboBox(full=True)
        color = ColorButton('#0D47A1')
        super().__init__(name, pipeline, color)


class AnnotationsLayer(Layer):
    def __init__(self, name):
        #super().__init__(name, MultiAnnotationComboBox(full=True), Spacing(0))
        super().__init__(name, AnnotationComboBox(full=True), Spacing(0))

    def select(self, view):
        if self.source.select_prefix(view):
            self.updated.emit()


class LayerManager(QtWidgets.QMenu):

    __all_layers__ = {}

    paramsUpdated = Signal(object) 

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        vbox = VBox(self, spacing=5, margin=5)
        #self.vmin = RealSlider(value=0, vmin=-1, vmax=1, auto_accept=False)
        #self.vmax = RealSlider(value=1, vmin=0, vmax=1, auto_accept=False)
        #vbox.addWidget(self.vmin)
        #vbox.addWidget(self.vmax)
        self._layers = OrderedDict()
        for key, title, cls in self.__all_layers__:
            layer = cls(key)
            #layer.updated.connect(self.on_layer_updated)
            vbox.addWidget(SubHeader(title))
            vbox.addWidget(layer)
            self._layers[key] = layer
        #self.vmin.valueChanged.connect(self.on_layer_updated)
        #self.vmax.valueChanged.connect(self.on_layer_updated)

    def mousePressEvent(self, event):
        print(f"mousePressEvent {event}")
    def refresh(self):
        logger.debug("layer_manger refresh")
        for layer in self._layers.values():
            layer.update()
        return self

    def on_layer_updated(self):
        logger.debug("layer_manger on_layer_updated")
        self.paramsUpdated.emit()

    def show_layer(self, layer, view):
        if layer in self._layers:
            self._layers[layer].select(view)

    def value(self):
        logger.debug("layer_manger value")
        params = {k: v.value() for k, v in self._layers.items()}
        params['clim'] = (self.vmin.value(), self.vmax.value())
        return params

    def accept(self):
        logger.debug("layer_manger accept")
        #self.vmin.accept()
        #self.vmax.accept()
        for layer in self._layers.values():
            layer.accept()


class WorkspaceLayerManager(LayerManager):

    __all_layers__ = (
        #('data', 'Data', DataLayer),
        ('feature', 'Feature', FeatureLayer),
        ('regions', 'Regions', RegionLayer),
        ('annotations', 'Annotations', AnnotationsLayer),
        ('pipelines', 'Segmentations', PipelinesLayer)
    )


