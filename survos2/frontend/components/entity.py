import numpy as np
import seaborn as sns
import pandas as pd
from loguru import logger

from qtpy import QtWidgets
from qtpy.QtCore import QSize, Signal

from survos2.frontend.model import ClientData
from survos2.entity.entities import make_entity_df
from survos2.server.state import cfg
import pyqtgraph as pg


def setup_entity_table(entities_fullname, scale=1.0):

    entities_df = pd.read_csv(entities_fullname)
    entities_df.drop(
        entities_df.columns[entities_df.columns.str.contains("unnamed", case=False)],
        axis=1,
        inplace=True,
    )
    entities_df = make_entity_df(np.array(entities_df), flipxy=True)
    logger.debug(f"Loaded entities {entities_df.shape}")

    tabledata = []

    for i in range(len(entities_df)):
        entry = (
            i,
            entities_df.iloc[i]["z"] * scale,
            entities_df.iloc[i]["x"] * scale,
            entities_df.iloc[i]["y"] * scale,
            entities_df.iloc[i]["class_code"],
        )
        tabledata.append(entry)

    tabledata = np.array(
        tabledata,
        dtype=[
            ("index", int),
            ("z", int),
            ("x", int),
            ("y", int),
            ("class_code", int),
        ],
    )

    logger.debug(f"Loaded {len(tabledata)} entities.")

    return tabledata, entities_df


class SmallVolWidget:
    def __init__(self, smallvol):

        self.imv = pg.ImageView()
        self.imv.setImage(smallvol, xvals=np.linspace(1.0, 3.0, smallvol.shape[0]))

    def set_vol(self, smallvol):
        self.imv.setImage(smallvol, xvals=np.linspace(1.0, 3.0, smallvol.shape[0]))
        self.imv.jumpFrames(smallvol.shape[0] // 2)


# have to inherit from QGraphicsObject in order for signal to work
class TableWidget(QtWidgets.QGraphicsObject):
    clientEvent = Signal(object)

    def __init__(self):
        super().__init__()
        self.w = pg.TableWidget()

        self.w.show()
        self.w.resize(500, 500)
        self.w.setWindowTitle("Entity table")

        self.w.cellClicked.connect(self.cell_clicked)
        self.w.doubleClicked.connect(self.double_clicked)
        self.w.selected_row = 0

        stylesheet = "QHeaderView::section{Background-color:rgb(30,60,80)}"
        self.w.setStyleSheet(stylesheet)

    def set_data(self, data):
        self.w.setData(data)

    def double_clicked(self):
        cfg.ppw.clientEvent.emit(
            {"source": "table", "data": "show_roi", "selected_roi": self.w.selected_row}
        )

        for index in self.w.selectedIndexes():
            logger.debug(f"Retrieved item from table: {self.w.model().data(index)}")

    def cell_clicked(self, row, col):
        logger.debug("Row %d and Column %d was clicked" % (row, col))
        self.w.selected_row = row
