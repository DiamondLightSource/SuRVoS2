
import os.path as op

from .utils import resource
from .qtcompat import QtWidgets, QtCore, QtGui
from .plugins import list_plugins, get_plugin
from .views import list_views, get_view
from .components import *
from .modal import ModalManager
from .control import Launcher

from survos2.utils import get_logger
from survos2.config import Config
import time
from multiprocessing import Process
from vispy.color import Colormap

import numpy as np
import numba
import math


logger = get_logger()


class ViewContainer(QCSWidget):

    __empty_view__ = dict(idx=0, title=None)

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.header = TabBar()
        self.container = QtWidgets.QStackedWidget()
        self.container.addWidget(QtWidgets.QWidget())

        vbox = VBox(self, margin=(1, 0, 0, 0), spacing=5)
        vbox.addWidget(self.header)
        vbox.addWidget(self.container, 1)

        self.views = {}
        self.current_view = None
        self.header.tabSelected.connect(self.select_view)

    def set_available_views(self, views):
        if not self.current_view in views:
            self.container.setCurrentIndex(0)
            self.current_view = None

        self.header.clear()
        views = [v for v in views if v in self.views]

        for view in views:
            self.header.addTab(view, self.views[view]['title'])

        if self.current_view is None:
            self.select_view(views[0])
        else:
            widget = self.views[self.current_view]['widget']
            if hasattr(widget, 'setup'):
                widget.setup()

    def select_view(self, name):
        if name in self.views:
            self.container.setCurrentIndex(self.views[name]['idx'])
            self.current_view = name
            self.header.setSelected(name)
            widget = self.views[self.current_view]['widget']
            if hasattr(widget, 'setup'):
                widget.setup()

    def load_view(self, name, title, cls):
        if name in self.views:
            return
        idx = len(self.views) + 1
        widget = cls()
        self.container.addWidget(widget)
        self.views[name] = dict(title=title, idx=idx, widget=widget)
        return widget

    def unload_view(self, name):
        pass

    def propagate_keybinding(self, evt):
        if self.current_view is not None:
            widget = self.views[self.current_view]['widget']
            if hasattr(widget, 'triggerKeybinding'):
                widget.triggerKeybinding(evt.key(), evt.modifiers())

        if not evt.isAccepted():
            evt.accept()

def update_ui():
    logger.info('Updating UI')
    QtCore.QCoreApplication.processEvents()
    time.sleep(0.1)



"""
        from .views import list_views, get_view

        _loaded_views = {}
        view_container_widget = ViewContainer()

        def _load_view(name):
            view = get_view(name)
            name, cls, title = view['name'], view['cls'], view['title']
            return view_container_widget.load_view(name, title, cls)

        def _unload_view(name):
            view_container.unload_view(name)

        def _show_view(name, **kwargs):
            view_container.select_view(name, **kwargs)

        def _load_views(views=None):
            all_views = list_views() if views is None else views
            for vname in all_views:
                view = _load_view(vname)
                _loaded_views[vname] = view

        _load_views()

        tab5.layout = QVBoxLayout()
        tab5.setLayout(tab5.layout)
        tab5.layout.addWidget(view_container_widget)
        
"""
