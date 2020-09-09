import numpy as np


from loguru import logger

from qtpy import QtWidgets
from qtpy.QtCore import QSize, Signal

from survos2.frontend.model import ClientData


def setup_entity_table(viewer, cData):
    tabledata = []

    for i in range(len(cData.entities)):
        entry = (
            i,
            cData.entities.iloc[i]["z"],
            cData.entities.iloc[i]["x"],
            cData.entities.iloc[i]["y"],
            cData.entities.iloc[i]["class_code"],
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
    sel_start, sel_end = 0, len(cData.entities)

    centers = np.array(
        [
            [
                np.int(np.float(cData.entities.iloc[i]["z"])),
                np.int(np.float(cData.entities.iloc[i]["x"])),
                np.int(np.float(cData.entities.iloc[i]["y"])),
            ]
            for i in range(sel_start, sel_end)
        ]
    )

    num_classes = len(np.unique(cData.entities["class_code"])) + 5
    logger.debug(f"Number of entity classes {num_classes}")
    palette = np.array(sns.color_palette("hls", num_classes))  # num_classes))
    # norm = Normalize(vmin=0, vmax=num_classes)

    face_color_list = [
        palette[class_code] for class_code in cData.entities["class_code"]
    ]

    entity_layer = viewer.add_points(
        centers,
        size=[10] * len(centers),
        opacity=0.5,
        face_color=face_color_list,
        n_dimensional=True,
    )
    cData.tabledata = tabledata

    return entity_layer, tabledata


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

    def set_data(self, data):
        self.w.setData(data)

    def double_clicked(self):
        self.clientEvent.emit(
            {"source": "table", "data": "show_roi", "selected_roi": self.w.selected_row}
        )

        for index in self.w.selectedIndexes():
            logger.debug(f"Retrieved item from table: {self.w.model().data(index)}")

    def cell_clicked(self, row, col):
        logger.debug("Row %d and Column %d was clicked" % (row, col))
        self.w.selected_row = row
