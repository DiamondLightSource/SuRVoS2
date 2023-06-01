"""
QT Plugins Base


"""
from collections import OrderedDict
from qtpy import QtWidgets, QtCore, QtGui
from qtpy.QtCore import QSize, Signal
from qtpy.QtWidgets import QPushButton, QRadioButton, QTabWidget, QVBoxLayout, QWidget

from survos2.frontend.components.base import Header, QCSWidget, ScrollPane, VBox
from survos2.config import config

__available_plugins__ = OrderedDict()


class Plugin(QCSWidget):
    change_view = QtCore.Signal(str, dict)

    __icon__ = "square"
    __pname__ = "plugin"
    __title__ = None
    __views__ = []
    __tab__ = "workspace"

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self._loaded_views = dict()

    def show_view(self, name, **kwargs):
        if name not in self.__views__:
            raise ValueError("View `{}` was not preloaded.".format(name))
        self.change_view.emit(name, kwargs)

    def register_view(self, view, widget):
        if not view in self.__views__:
            return
        self._loaded_views[view] = widget

    def on_created(self):
        pass

    def __getitem__(self, name):
        return self._loaded_views[name]


def register_plugin(cls):
    name = cls.__pname__
    icon = cls.__icon__
    title = cls.__title__
    views = cls.__views__
    tab = cls.__tab__

    if name in __available_plugins__:
        raise ValueError("Plugin {} already registered.".format(name))

    if title is None:
        title = name.capitalize()

    desc = dict(cls=cls, name=name, icon=icon, title=title, views=views, tab=tab)
    __available_plugins__[name] = desc

    return cls


def get_plugin(name):
    if name not in __available_plugins__:
        raise ValueError("Plugin {} not registered".format(name))
    return __available_plugins__[name]


def list_plugins():
    return list(__available_plugins__.keys())


class PluginContainer(QCSWidget):
    view_requested = QtCore.Signal(str, dict)
    __sidebar_width__ = 500

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.tabwidget = QTabWidget()
        vbox = VBox(self, margin=(1, 1, 2, 0), spacing=2)
        vbox.addWidget(self.tabwidget, 1)
        self.tabwidget.setStyleSheet("QTabBar::tab { padding: 10px; color: 	#AAFFFF}")
        

        self.tabs = [
            (QWidget(), t)
            for t in config["api"]["plugins"]
            if (t != "workspace") and (t != "render")
        ]
        # 'workspace' and 'render' are internal plugins with no gui
        for t in self.tabs:
            self.tabwidget.addTab(t[0], t[1].capitalize())
            t[0].layout = QVBoxLayout()
            t[0].setLayout(t[0].layout)

        self.title = Header("Plugin")

        self.containers = {}
        for t in self.tabs:
            pane = ScrollPane(parent=self)
            t[0].layout.addWidget(pane)
            self.containers[t[1]] = pane

        self.plugins = {}
        self.selected_name = None
        self.selected = None

    def load_plugin(self, name, title, cls):
        if name in self.plugins:
            return
        widget = cls()
        widget.change_view.connect(self.view_requested)
        self.plugins[name] = dict(widget=widget, title=title)
        return widget

    def unload_plugin(self, name):
        self.plugins.pop(name, None)

    def show_plugin(self, name, tab):
        if name in self.plugins:
            self.selected_name = name
            self.selected = self.plugins[name]
            self.title.setText(self.selected["title"].capitalize())

            for t in self.tabs:
                if tab == t[1]:
                    self.containers[t[1]].addWidget(self.selected["widget"], 1)

            if hasattr(self.selected["widget"], "setup"):
                self.selected["widget"].clear()
                self.selected["widget"].setup()

    def show_plugin_fast(self, name, tab):
        if name in self.plugins:
            self.selected = self.plugins[name]
            if hasattr(self.selected["widget"], "setup"):
                self.selected["widget"].setup()
