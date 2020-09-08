from qtpy import QtWidgets
from qtpy.QtWidgets import QRadioButton, QPushButton
from qtpy.QtCore import QSize
from qtpy import QtCore
from qtpy.QtCore import QSize, Signal
from qtpy import QtGui
import qtawesome as qta


class IconButton(QtWidgets.QPushButton):
    def __init__(
        self,
        icon,
        title="",
        checkable=False,
        accent=False,
        flat=False,
        error=False,
        parent=None,
        toggleable=False,
    ):
        super().__init__(title, parent=parent)
        self.icon = icon
        self.altIcon = icon
        self.flat = flat
        self.accent = accent
        self.error = error
        self.toggeable = toggleable
        if accent:
            self.setProperty("accent", True)
        self.setFlat(flat)
        self.setIconColor("white")
        self.setCheckable(checkable)
        self.leaveEvent()
        if toggleable and flat:
            self.setStyleSheet("background-color: transparent;")

    def setIconColor(self, color):
        icon = qta.icon(self.icon, color=color)
        self.setIcon(icon)

    def enterEvent(self, evt=None):
        if self.flat and not self.isChecked():
            color = "#4527A0" if self.accent else "#009688"
            self.setIconColor(color)

    def leaveEvent(self, evt=None):
        if self.flat and (not self.isChecked() or self.toggeable):
            if self.error:
                color = "red"
            else:
                color = "#009688" if self.accent else "#0D47A1"
            self.setIconColor(color)
        elif self.flat:
            self.setIconColor("white")

    def setChecked(self, flag):
        self.leaveEvent()
        super().setChecked(flag)


class DelIconButton(IconButton):
    def __init__(self, secondary=False, **kwargs):
        kwargs.setdefault("flat", True)
        kwargs.setdefault("error", True)
        icon = "fa.times-circle-o" if secondary else "fa.times"
        super().__init__(icon, **kwargs)


class AddIconButton(IconButton):
    def __init__(self, **kwargs):
        kwargs.setdefault("flat", True)
        super().__init__("fa.plus", **kwargs)


class ViewIconButton(IconButton):
    def __init__(self, **kwargs):
        kwargs.setdefault("flat", True)
        super().__init__("fa.chevron-down", toggleable=True, **kwargs)
        self.setCheckable(True)
        self.toggled.connect(self.update_icon)
        self.setChecked(True)
        self.update_icon()

    def update_icon(self):
        self.icon = "fa.chevron-down" if self.isChecked() else "fa.chevron-up"
        super().leaveEvent()

    def setChecked(self, flag):
        super().setChecked(flag)
        self.update_icon()


class ToolIconButton(QtWidgets.QToolButton):
    def __init__(
        self, icon, text="", size=None, checkable=False, parent=None, **kwargs
    ):
        super().__init__(parent)
        self.setStyleSheet(
            """
            QToolButton {
                margin-left: 0px;
            }
            QToolButton::menu-indicator {
                width: 0px;
                border: none;
                image: none;
            }
        """
        )
        self.setText(text)
        self.setIcon(qta.icon(icon, **kwargs))
        if size is not None:
            self.setIconSize(QtCore.QSize(*size))
        self.setCheckable(checkable)
        self.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
