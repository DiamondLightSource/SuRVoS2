from loguru import logger
from qtpy import QtWidgets
import logging
import ast
import os
from loguru import logger
from qtpy import QtWidgets

from qtpy.QtWidgets import QLabel, QRadioButton

from survos2.frontend.control import Launcher
from survos2.model import DataModel
from survos2.frontend.components.base import (
    LineEdit,
    ComboBox,
    HWidgets,
    PushButton,
    Spacing,
    Label,
    ComboBox,
    DataTableWidgetItem,
)
from survos2.frontend.plugins.pipeline.base import PipelineCardBase
from survos2.model import DataModel
from survos2.server.state import cfg
from survos2.frontend.plugins.pipeline.base import PipelineCardBase


class TrainMultiaxisCNN(PipelineCardBase):
    def __init__(self, fid, ftype, fname, fparams, parent=None, pipeline_notifier=None):
        super().__init__(fid=fid, ftype=ftype, fname=fname, fparams=fparams, pipeline_notifier=pipeline_notifier)

    def setup(self):
        self._add_multi_ax_cnn_training_ws_widget()
        self._add_multi_ax_cnn_annotations_features_from_ws(DataModel.g.current_workspace)
        self._add_volseg_model_type()
        self._add_multi_ax_cnn_data_table()
        self._add_multi_ax_2d_training_params()
        self.adv_train_fields.hide()

    def compute_pipeline(self):
        # Retrieve params from table
        num_rows = self.table.rowCount()
        if num_rows == 0:
            logging.error("No data selected for training!")
            return
        else:
            workspace_list = []
            data_list = []
            label_list = []
            for i in range(num_rows):
                workspace_list.append(
                    (self.table.item(i, 0).get_hidden_field(), self.table.item(i, 0).text())
                )
                data_list.append(
                    (self.table.item(i, 1).get_hidden_field(), self.table.item(i, 1).text())
                )
                label_list.append(
                    (self.table.item(i, 2).get_hidden_field(), self.table.item(i, 2).text())
                )
        # Can the src parameter be removed?
        src = DataModel.g.dataset_uri("001_raw", group="features")
        all_params = dict(src=src, dst=self.dst, modal=True)
        all_params["workspace"] = workspace_list
        all_params["feature_id"] = data_list
        all_params["anno_id"] = label_list
        all_params["multi_ax_train_params"] = dict(
            model_type=self.volseg_model_type.key(),
            encoder_type=self.volseg_encoder_type.key(),
            cyc_frozen=self.cycles_frozen.value(),
            cyc_unfrozen=self.cycles_unfrozen.value(),
            patience=self.train_patience_linedt.value(),
            loss_criterion=self.loss_type_combo.key(),
            bce_dice_alpha=self.bce_dice_alpha_linedt.value(),
            bce_dice_beta=self.bce_dice_beta_linedt.value(),
            training_axes=self.train_idx_to_axis[self.train_axis_r_group.checkedId()],
        )
        all_params["json_transport"] = True
        return all_params

    def _add_multi_ax_cnn_data_table(self):
        columns = ["Workspace", "Data", "Labels", ""]
        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(columns)
        table_fields = QtWidgets.QGroupBox("Training Datasets")
        table_layout = QtWidgets.QVBoxLayout()
        table_layout.addWidget(QLabel('Press "Compute" when list is complete.'))
        table_layout.addWidget(self.table)
        table_fields.setLayout(table_layout)
        self.add_row(table_fields, max_height=200)

    def table_delete_clicked(self):
        button = self.sender()
        if button:
            row = self.table.indexAt(button.pos()).row()
            self.table.removeRow(row)

    def _add_item_to_data_table(self, item):
        ws = DataTableWidgetItem(self.workspaces_list.currentText())
        ws.hidden_field = self.workspaces_list.value()
        data = DataTableWidgetItem(self.feature_source.currentText())
        data.hidden_field = self.feature_source.value()
        labels = DataTableWidgetItem(self.annotations_source.currentText())
        labels.hidden_field = self.annotations_source.value()
        self._add_data_row(ws, data, labels)

    def _add_data_row(self, ws, data, labels):
        row_pos = self.table.rowCount()
        self.table.insertRow(row_pos)
        self.table.setItem(row_pos, 0, ws)
        self.table.setItem(row_pos, 1, data)
        self.table.setItem(row_pos, 2, labels)
        delete_button = QtWidgets.QPushButton("Delete")
        delete_button.clicked.connect(self.table_delete_clicked)
        self.table.setCellWidget(row_pos, 3, delete_button)

    def _update_data_table_from_dict(self, data_dict):
        for ws, ds, lbl in zip(data_dict["Workspaces"], data_dict["Data"], data_dict["Labels"]):
            # ws = ast.literal_eval(ws)
            ws_item = DataTableWidgetItem(ws[1])
            ws_item.hidden_field = ws[0]
            # ds = ast.literal_eval(ds)
            ds_item = DataTableWidgetItem(ds[1])
            ds_item.hidden_field = ds[0]
            # lbl = ast.literal_eval(lbl)
            lbl_item = DataTableWidgetItem(lbl[1])
            lbl_item.hidden_field = lbl[0]
            self._add_data_row(ws_item, ds_item, lbl_item)

    def _update_multi_axis_cnn_train_params(self, frozen, unfrozen):
        self.cycles_frozen.setText(str(frozen))
        self.cycles_unfrozen.setText(str(unfrozen))

    def _add_multi_ax_cnn_training_ws_widget(self):
        data_label = QtWidgets.QLabel("Select training data:")
        self.add_row(data_label)
        self.workspaces_list = self._get_workspaces_list()
        self.workspaces_list
        self.workspaces_list.currentTextChanged.connect(self.on_ws_combobox_changed)
        ws_widget = HWidgets("Workspace:", self.workspaces_list, stretch=1)
        self.add_row(ws_widget)

    def _add_multi_ax_cnn_annotations_source(self):
        self.annotations_source = ComboBox()
        self.annotations_source.setMaximumWidth(250)
        anno_widget = HWidgets("Annotation (Labels):", self.annotations_source, stretch=1)
        self.add_row(anno_widget)

    def _add_multi_ax_cnn_feature_source(self):
        self.feature_source = ComboBox()
        self.feature_source.setMaximumWidth(250)
        feature_widget = HWidgets("Feature (Data):", self.feature_source, stretch=1)
        self.add_row(feature_widget)

    def _update_features_from_ws(self, workspace):
        self.feature_source.clear()
        workspace = "default@" + workspace
        params = {"workspace": workspace}
        logger.debug(f"Filling features from session: {params}")
        result = Launcher.g.run("features", "existing", **params)
        if result:
            for fid in result:
                self.feature_source.addItem(fid, result[fid]["name"])

    def _add_multi_ax_cnn_annotations_features_from_ws(self, workspace):
        self._add_multi_ax_cnn_feature_source()
        self._update_features_from_ws(workspace)
        self._add_multi_ax_cnn_annotations_source()
        self._update_annotations_from_ws(workspace)
        train_data_btn = PushButton("Add to list", accent=True)
        train_data_btn.clicked.connect(self._add_item_to_data_table)
        widget = HWidgets(None, train_data_btn, stretch=0)
        self.add_row(widget)

    def _get_workspaces_list(self):
        workspaces = [d for d in next(os.walk(DataModel.g.CHROOT))[1]]
        workspaces.sort()
        workspaces_list = ComboBox()
        workspaces_list.setMaximumWidth(250)
        for s in workspaces:
            workspaces_list.addItem(key=s)
        workspaces_list.setCurrentText(DataModel.g.current_workspace)
        return workspaces_list

    def on_ws_combobox_changed(self, workspace):
        self._update_annotations_from_ws(workspace)
        self._update_features_from_ws(workspace)

    def _add_model_type(self):
        self.model_type = ComboBox()
        self.model_type.addItem(key="unet3d")
        self.model_type.addItem(key="fpn3d")
        widget = HWidgets("Model type:", self.model_type, stretch=0)
        self.add_row(widget)

    def _add_volseg_model_type(self):
        self.volseg_model_type = ComboBox()
        self.volseg_model_type.addItem(key="U_Net", value="U-Net")
        self.volseg_model_type.addItem(key="U_Net_Plus_plus", value="U-Net++")
        self.volseg_model_type.addItem(key="FPN", value="FPN")
        self.volseg_model_type.addItem(key="DeepLabV3", value="DeepLabV3")
        self.volseg_model_type.addItem(key="DeepLabV3_Plus", value="DeepLabV3+")
        self.volseg_model_type.addItem(key="MA_Net", value="MA-Net")
        self.volseg_model_type.addItem(key="LinkNet", value="LinkNet")
        self.volseg_model_type.addItem(key="PAN", value="PAN")
        widget = HWidgets("Model type:", self.volseg_model_type, Spacing(35), stretch=0)
        self.add_row(widget)

    def _add_multi_ax_2d_training_params(self):
        self.multi_ax_train_settings = cfg["volume_segmantics"]["train_settings"]
        self.add_row(HWidgets("Training Parameters:", stretch=1))
        self.cycles_frozen = LineEdit(default=8, parse=int)
        self.cycles_unfrozen = LineEdit(default=5, parse=int)
        train_advanced_button = QRadioButton("Advanced")
        self.setup_adv_train_fields()
        self.adv_train_fields.hide()
        refresh_label = Label(
            'Please: 1. "Compute", 2. "Refresh Data", 3. Reopen dialog and "View".'
        )
        self.multi_ax_train_refresh_btn = PushButton("Refresh Data", accent=True)
        self.multi_ax_train_refresh_btn.clicked.connect(self.refresh_multi_ax_data)
        self.multi_ax_pred_refresh_btn = None
        self.add_row(
            HWidgets(
                "No. Cycles Frozen:",
                self.cycles_frozen,
                "No. Cycles Unfrozen",
                self.cycles_unfrozen,
                stretch=1,
            )
        )
        self.add_row(HWidgets(train_advanced_button, Spacing(35), stretch=1))
        self.add_row(self.adv_train_fields, max_height=250)
        self.add_row(HWidgets(refresh_label, stretch=1))
        self.add_row(HWidgets(self.multi_ax_train_refresh_btn, stretch=1))
        train_advanced_button.toggled.connect(self.toggle_advanced_train)

    def setup_adv_train_fields(self):
        """Sets up the QGroupBox that displays the advanced option for 2d deep
        learning training."""
        self.adv_train_fields = QtWidgets.QGroupBox("Advanced Training Settings:")
        adv_train_layout = QtWidgets.QGridLayout()
        cuda_device = str(self.multi_ax_train_settings["cuda_device"])
        patience = str(self.multi_ax_train_settings["patience"])
        bce_dice_alpha = str(self.multi_ax_train_settings["alpha"])
        bce_dice_beta = str(self.multi_ax_train_settings["beta"])
        self.train_cuda_dev_linedt = LineEdit(cuda_device)
        self.train_patience_linedt = LineEdit(patience)
        self.volseg_encoder_type = ComboBox()
        self.volseg_encoder_type.addItem(key="resnet34", value="ResNet34 (Pre-trained)")
        self.volseg_encoder_type.addItem(key="resnet50", value="ResNet50 (Pre-trained)")
        self.volseg_encoder_type.addItem(
            key="resnext50_32x4d", value="ResNeXt50 (32x4d Pre-trained)"
        )
        self.volseg_encoder_type.addItem(
            key="efficientnet-b3", value="EfficientNetB3 (Pre-trained)"
        )
        self.volseg_encoder_type.addItem(
            key="efficientnet-b4", value="EfficientNetB4 (Pre-trained)"
        )
        self.loss_type_combo = ComboBox()
        self.loss_type_combo.addItem(key="DiceLoss", value="Dice Loss")
        self.loss_type_combo.addItem(key="CrossEntropyLoss", value="Cross Entropy Loss")
        self.loss_type_combo.addItem(key="GeneralizedDiceLoss", value="Generalised Dice Loss")
        self.loss_type_combo.addItem(key="BCELoss", value="Binary Cross Entropy Loss")
        self.loss_type_combo.addItem(key="BCEDiceLoss", value="Binary Cross Entropy and Dice Loss")
        self.loss_type_combo.currentTextChanged.connect(self.on_loss_function_combo_changed)
        self.bce_dice_alpha_linedt = LineEdit(bce_dice_alpha)
        self.bce_dice_beta_linedt = LineEdit(bce_dice_beta)
        adv_train_layout.addWidget(QLabel("CUDA Device:"), 0, 0)
        adv_train_layout.addWidget(self.train_cuda_dev_linedt, 0, 1)
        adv_train_layout.addWidget(QLabel("Early stopping patience:"), 1, 0)
        adv_train_layout.addWidget(self.train_patience_linedt, 1, 1)
        adv_train_layout.addWidget(QLabel("Loss function:"), 2, 0)
        adv_train_layout.addWidget(self.loss_type_combo, 2, 1)
        self.bce_dice_alpha_label = QLabel("BCE Weight")
        self.bce_dice_beta_label = QLabel("Dice Weight")
        adv_train_layout.addWidget(self.bce_dice_alpha_label, 3, 0)
        adv_train_layout.addWidget(self.bce_dice_alpha_linedt, 3, 1)
        adv_train_layout.addWidget(self.bce_dice_beta_label, 4, 0)
        adv_train_layout.addWidget(self.bce_dice_beta_linedt, 4, 1)
        adv_train_layout.addWidget(QLabel("Encoder:"), 5, 0)
        adv_train_layout.addWidget(self.volseg_encoder_type, 5, 1)
        self.bce_dice_alpha_label.hide()
        self.bce_dice_alpha_linedt.hide()
        self.bce_dice_beta_label.hide()
        self.bce_dice_beta_linedt.hide()
        self.train_axis_r_group = QtWidgets.QButtonGroup()
        self.train_axis_r_group.setExclusive(True)
        all_axes_rb = QRadioButton("All")
        all_axes_rb.setChecked(True)
        self.train_axis_r_group.addButton(all_axes_rb, 0)
        z_axis_rb = QRadioButton("Z")
        self.train_axis_r_group.addButton(z_axis_rb, 1)
        y_axis_rb = QRadioButton("Y")
        self.train_axis_r_group.addButton(y_axis_rb, 2)
        x_axis_rb = QRadioButton("X")
        self.train_axis_r_group.addButton(x_axis_rb, 3)
        self.train_idx_to_axis = {
            0: "All",
            1: "Z",
            2: "Y",
            3: "X",
        }
        adv_train_layout.addWidget(QLabel("Train Axis:"), 6, 0)
        adv_train_layout.addWidget(HWidgets(all_axes_rb, z_axis_rb, y_axis_rb, x_axis_rb), 6, 1)
        self.adv_train_fields.setLayout(adv_train_layout)

    def toggle_advanced_train(self):
        """Controls displaying/hiding the advanced train fields on radio button toggle."""
        rbutton = self.sender()
        if rbutton.isChecked():
            self.adv_train_fields.show()
        else:
            self.adv_train_fields.hide()

    def on_loss_function_combo_changed(self, value):
        if value == "Binary Cross Entropy and Dice Loss":
            self.bce_dice_alpha_label.show()
            self.bce_dice_alpha_linedt.show()
            self.bce_dice_beta_label.show()
            self.bce_dice_beta_linedt.show()


class PredictMultiaxisCNN(PipelineCardBase):
    def __init__(self, fid, ftype, fname, fparams, parent=None, pipeline_notifier=None):
        super().__init__(fid=fid, ftype=ftype, fname=fname, fparams=fparams, pipeline_notifier=pipeline_notifier)

    def setup(self):
        self._add_feature_source()
        self._add_multi_ax_2d_prediction_params()
        self._add_annotations_source()

    def compute_pipeline(self):
        src = DataModel.g.dataset_uri(self.feature_source.value(), group="features")
        all_params = dict(src=src, dst=self.dst, modal=True)
        all_params["workspace"] = DataModel.g.current_workspace
        all_params["feature_id"] = str(self.feature_source.value())
        model_path = str(self.model_file_line_edit.value())
        if len(model_path) > 0:
            all_params["model_path"] = model_path
        else:
            raise ValueError("No model filepath selected!")
        all_params["no_of_planes"] = self.radio_group.checkedId()
        all_params["cuda_device"] = int(self.pred_cuda_dev_linedt.value())
        all_params["prediction_axis"] = self.pred_idx_to_axis[self.pred_axis_r_group.checkedId()]
        return all_params

    def _add_multi_ax_2d_prediction_params(self):
        self.multi_ax_pred_settings = cfg["volume_segmantics"]["predict_settings"]
        self.model_file_line_edit = LineEdit(default="Filepath", parse=str)
        model_input_btn = PushButton("Select Model", accent=True)
        model_input_btn.clicked.connect(self.get_model_path)
        self.radio_group = QtWidgets.QButtonGroup()
        self.radio_group.setExclusive(True)
        single_pp_rb = QRadioButton("Single plane")
        single_pp_rb.setChecked(True)
        self.radio_group.addButton(single_pp_rb, 1)
        triple_pp_rb = QRadioButton("Three plane")
        self.radio_group.addButton(triple_pp_rb, 3)
        twelve_pp_rb = QRadioButton("3 planes, 4 rotations")
        self.radio_group.addButton(twelve_pp_rb, 12)
        advanced_button = QRadioButton("Advanced")
        self.setup_adv_pred_fields()
        self.adv_pred_fields.hide()
        refresh_label = Label(
            'Please: 1. "Compute", 2. "Refresh Data", 3. Reopen dialog and "View".'
        )
        self.multi_ax_pred_refresh_btn = PushButton("Refresh Data", accent=True)
        self.multi_ax_pred_refresh_btn.clicked.connect(self.refresh_multi_ax_data)
        self.multi_ax_train_refresh_btn = None
        self.add_row(HWidgets(self.model_file_line_edit, model_input_btn, Spacing(35)))
        self.add_row(HWidgets("Prediction Parameters:", stretch=1))
        self.add_row(HWidgets(single_pp_rb, triple_pp_rb, twelve_pp_rb))
        self.add_row(HWidgets(advanced_button, stretch=1))
        self.add_row(HWidgets(self.adv_pred_fields), max_height=75)
        self.add_row(HWidgets(refresh_label, stretch=1))
        self.add_row(HWidgets(self.multi_ax_pred_refresh_btn, stretch=1))
        advanced_button.toggled.connect(self.toggle_advanced_pred)
        advanced_button.setChecked(True)
        advanced_button.setChecked(False)

    def setup_adv_pred_fields(self):
        """Sets up the QGroupBox that displays the advanced option for 2d deep
        learning prediction."""
        self.adv_pred_fields = QtWidgets.QGroupBox("Advanced Prediction Settings:")
        adv_pred_layout = QtWidgets.QGridLayout()
        adv_pred_layout.addWidget(QLabel("CUDA Device:"), 0, 0)
        cuda_device = str(self.multi_ax_pred_settings["cuda_device"])
        self.pred_cuda_dev_linedt = LineEdit(cuda_device)
        adv_pred_layout.addWidget(self.pred_cuda_dev_linedt, 0, 1)
        self.pred_axis_r_group = QtWidgets.QButtonGroup()
        self.pred_axis_r_group.setExclusive(True)
        z_axis_rb = QRadioButton("Z")
        z_axis_rb.setChecked(True)
        self.pred_axis_r_group.addButton(z_axis_rb, 0)
        y_axis_rb = QRadioButton("Y")
        self.pred_axis_r_group.addButton(y_axis_rb, 1)
        x_axis_rb = QRadioButton("X")
        self.pred_axis_r_group.addButton(x_axis_rb, 2)
        self.pred_idx_to_axis = {
            0: "Z",
            1: "Y",
            2: "X",
        }
        adv_pred_layout.addWidget(QLabel("Single plane predict axis:"), 1, 0)
        adv_pred_layout.addWidget(HWidgets(z_axis_rb, y_axis_rb, x_axis_rb), 1, 1)
        self.adv_pred_fields.setLayout(adv_pred_layout)

    def toggle_advanced_pred(self):
        """Controls displaying/hiding the advanced predict fields on radio button toggle."""
        rbutton = self.sender()
        if rbutton.isChecked():
            self.adv_pred_fields.show()
        else:
            self.adv_pred_fields.hide()
