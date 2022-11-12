from loguru import logger
from qtpy import QtWidgets
from qtpy.QtCore import QSize, Signal
from survos2.frontend.components.base import (
    ComboBox,
    HWidgets,
    LineEdit,
)
from survos2.model import DataModel
from survos2.frontend.plugins.pipeline.base import PipelineCardBase
from napari.qt.progress import progress
from survos2.frontend.plugins.pipeline.base import PipelineCardBase


class SVMWidget(QtWidgets.QWidget):
    predict = Signal(dict)

    def __init__(self, parent=None):
        super(SVMWidget, self).__init__(parent=parent)

        vbox = QtWidgets.QVBoxLayout()
        vbox.setContentsMargins(0, 0, 0, 0)
        self.setLayout(vbox)

        self.type_combo = ComboBox()
        self.type_combo.addCategory("Kernel Type:")
        self.type_combo.addItem("linear")
        self.type_combo.addItem("poly")
        self.type_combo.addItem("rbf")
        self.type_combo.addItem("sigmoid")
        vbox.addWidget(self.type_combo)

        self.penaltyc = LineEdit(default=1.0, parse=float)
        self.gamma = LineEdit(default=1.0, parse=float)

        vbox.addWidget(
            HWidgets(
                QtWidgets.QLabel("Penalty C:"),
                self.penaltyc,
                QtWidgets.QLabel("Gamma:"),
                self.gamma,
                stretch=[0, 1, 0, 1],
            )
        )

    def on_predict_clicked(self):
        params = {
            "clf": "svm",
            "kernel": self.type_combo.currentText(),
            "C": self.penaltyc.value(),
            "gamma": self.gamma.value(),
        }

        self.predict.emit(params)

    def get_params(self):
        params = {
            "clf": "svm",
            "kernel": self.type_combo.currentText(),
            "C": self.penaltyc.value(),
            "gamma": self.gamma.value(),
            "type": self.type_combo.value(),
        }

        return params


class EnsembleWidget(QtWidgets.QWidget):
    train_predict = Signal(dict)

    def __init__(self, parent=None):
        super(EnsembleWidget, self).__init__(parent=parent)

        vbox = QtWidgets.QVBoxLayout()
        vbox.setContentsMargins(0, 0, 0, 0)
        self.setLayout(vbox)

        self.type_combo = ComboBox()
        self.type_combo.addCategory("Ensemble Type:")
        self.type_combo.addItem("Random Forest")
        self.type_combo.addItem("ExtraRandom Forest")
        self.type_combo.addItem("AdaBoost")
        self.type_combo.addItem("GradientBoosting")

        self.type_combo.currentIndexChanged.connect(self.on_ensemble_changed)
        vbox.addWidget(self.type_combo)

        self.ntrees = LineEdit(default=100, parse=int)
        self.depth = LineEdit(default=15, parse=int)
        self.lrate = LineEdit(default=1.0, parse=float)
        self.subsample = LineEdit(default=1.0, parse=float)

        vbox.addWidget(
            HWidgets(
                QtWidgets.QLabel("# Trees:"),
                self.ntrees,
                QtWidgets.QLabel("Max Depth:"),
                self.depth,
                stretch=[0, 1, 0, 1],
            )
        )

        vbox.addWidget(
            HWidgets(
                QtWidgets.QLabel("Learn Rate:"),
                self.lrate,
                QtWidgets.QLabel("Subsample:"),
                self.subsample,
                stretch=[0, 1, 0, 1],
            )
        )

        self.n_jobs = LineEdit(default=10, parse=int)
        vbox.addWidget(HWidgets("Num Jobs", self.n_jobs))

    def on_ensemble_changed(self, idx):
        if idx == 2:
            self.ntrees.setDefault(50)
        else:
            self.ntrees.setDefault(100)

        if idx == 3:
            self.lrate.setDefault(0.1)
            self.depth.setDefault(3)
        else:
            self.lrate.setDefault(1.0)
            self.depth.setDefault(15)

    def on_train_predict_clicked(self):
        ttype = ["rf", "erf", "ada", "gbf", "xgb"]
        params = {
            "clf": "ensemble",
            "type": ttype[self.type_combo.currentIndex()],
            "n_estimators": self.ntrees.value(),
            "max_depth": self.depth.value(),
            "learning_rate": self.lrate.value(),
            "subsample": self.subsample.value(),
            "n_jobs": self.n_jobs.value(),
        }
        self.train_predict.emit(params)

    def get_params(self):
        ttype = ["rf", "erf", "ada", "gbf", "xgb"]
        if self.type_combo.currentIndex() == 1:
            current_index = 0
        else:
            current_index = self.type_combo.currentIndex() - 1
        logger.debug(f"Ensemble type_combo index: {current_index}")
        params = {
            "clf": "ensemble",
            "type": ttype[current_index],
            "n_estimators": self.ntrees.value(),
            "max_depth": self.depth.value(),
            "learning_rate": self.lrate.value(),
            "subsample": self.subsample.value(),
            "n_jobs": self.n_jobs.value(),
        }

        return params


class SuperregionSegment(PipelineCardBase):
    def __init__(self, fid, ftype, fname, fparams, parent=None):
        super().__init__(fid=fid, ftype=ftype, fname=fname, fparams=fparams)

    def setup(self):
        logger.debug("Adding a superregion_segment pipeline")
        self._add_features_source()
        self._add_annotations_source()
        self._add_constrain_source()
        self._add_regions_source()

        self.ensembles = EnsembleWidget()
        self.ensembles.train_predict.connect(self.compute_pipeline)
        self.svm = SVMWidget()
        self.svm.predict.connect(self.compute_pipeline)

        self._add_classifier_choice()
        self._add_projection_choice()
        self._add_param("refine", type="SmartBoolean", default=True)
        self._add_param("lam", type="FloatSlider", default=0.15)
        self._add_confidence_choice()

    def compute_pipeline(self):
        feature_names_list = [n.rsplit("/", 1)[-1] for n in self.features_source.value()]
        src_grp = None if self.annotations_source.currentIndex() == 0 else "pipelines"
        src = DataModel.g.dataset_uri(
            self.annotations_source.value().rsplit("/", 1)[-1],
            group="annotations",
        )
        all_params = dict(src=src, modal=True)
        all_params["workspace"] = DataModel.g.current_workspace

        logger.info(f"Setting src to {self.annotations_source.value()} ")
        all_params["region_id"] = str(self.regions_source.value().rsplit("/", 1)[-1])
        all_params["feature_ids"] = feature_names_list
        all_params["anno_id"] = str(self.annotations_source.value().rsplit("/", 1)[-1])
        if self.constrain_mask_source.value() != None:
            all_params["constrain_mask"] = self.constrain_mask_source.value()  # .rsplit("/", 1)[-1]
        else:
            all_params["constrain_mask"] = "None"

        all_params["dst"] = self.dst
        all_params["refine"] = self.widgets["refine"].value()
        all_params["lam"] = self.widgets["lam"].value()
        all_params["classifier_type"] = self.classifier_type.value()
        all_params["projection_type"] = self.projection_type.value()
        all_params["confidence"] = self.confidence_checkbox.value()

        if self.classifier_type.value() == "Ensemble":
            all_params["classifier_params"] = self.ensembles.get_params()
        else:
            all_params["classifier_params"] = self.svm.get_params()

        all_params["json_transport"] = True
        return all_params
