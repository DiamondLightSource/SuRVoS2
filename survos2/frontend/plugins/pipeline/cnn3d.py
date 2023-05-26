from survos2.model import DataModel
from survos2.frontend.components.base import LineEdit, ComboBox, HWidgets
from survos2.frontend.plugins.pipeline.base import PipelineCardBase
from survos2.frontend.utils import FileWidget


class Train3DCNN(PipelineCardBase):
    def __init__(self, fid, ftype, fname, fparams, parent=None):
        super().__init__(fid=fid, ftype=ftype, fname=fname, fparams=fparams)

    def setup(self):
        self._add_annotations_source()
        self._add_feature_source()
        self._add_objects_source()
        self._add_fcn_choice()
        self._add_overlap_choice()
        self._add_param("cuda_device", type="Int", default=0)
        self._add_param("num_samples", type="Int", default=100)
        self._add_param("num_augs", type="Int", default=1)
        self._add_param("num_epochs", type="Int", default=2)
        self._add_param("bce_to_dice_weight", type="Float", default=0.3)
        self._add_param("patch_overlap", type="IntOrVector", default=(32, 32, 32))
        self._add_param("patch_size", type="IntOrVector", default=(64, 64, 64))
        self._add_param("threshold", type="Float", default=0.5)

    def compute_pipeline(self):
        src = DataModel.g.dataset_uri(self.feature_source.value(), group="features")
        all_params = dict(src=src, dst=self.dst, modal=True)
        workspace_id = DataModel.g.current_workspace
        feature_id = str(self.feature_source.value())
        anno_id = str(self.annotations_source.value().rsplit("/", 1)[-1])
        all_params["workspace"] = [[workspace_id, workspace_id]]
        all_params["feature_id"] = [[feature_id, feature_id]]
        all_params["anno_id"] = [[anno_id, anno_id]]
        if self.objects_source.value() != None:
            objects_id = str(self.objects_source.value().rsplit("/", 1)[-1])
            all_params["objects_id"] = [[objects_id, objects_id]]
            # all_params["objects_id"] = str(self.objects_source.value()

        else:
            #     all_params["objects_id"] = str(self.objects_source.value())
            all_params["objects_id"] = [["None", "None"]]

        all_params["fcn_type"] = self.fcn_type.value()
        # all_params["bce_to_dice_weight"] = 0.3
        # all_params["num_epochs"] = 2
        # all_params["num_augs"] = 0
        # all_params["num_samples"] = 400
        # all_params["threshold"] = 0.5
        # all_params["patch_overlap"] = [16,16,16]
        # all_params["patch_size"] = [64,64,64]
        all_params["overlap_mode"] = self.overlap_type.value()
        all_params["json_transport"] = True
        all_params["plot_figures"] = False
        print(all_params)

        return all_params


class Predict3DCNN(PipelineCardBase):
    def __init__(self, fid, ftype, fname, fparams, parent=None):
        super().__init__(fid=fid, ftype=ftype, fname=fname, fparams=fparams)

    def setup(self):
        self._add_annotations_source()
        self._add_feature_source()
        self._add_model_file()
        self._add_fcn_choice()
        self._add_overlap_choice()
        self._add_param("patch_overlap", type="IntOrVector", default=(32, 32, 32))
        self._add_param("patch_size", type="IntOrVector", default=(64, 64, 64))
        self._add_param("threshold", type="Float", default=0.5)
        self._add_param("cuda_device", type="Int", default=0)

    def compute_pipeline(self):
        src = DataModel.g.dataset_uri(self.feature_source.value(), group="features")
        all_params = dict(src=src, dst=self.dst, modal=True)
        all_params["workspace"] = DataModel.g.current_workspace
        all_params["anno_id"] = str(self.annotations_source.value().rsplit("/", 1)[-1])
        all_params["feature_id"] = self.feature_source.value()
        all_params["model_fullname"] = self.model_fullname
        all_params["model_type"] = self.fcn_type.value()
        all_params["overlap_mode"] = self.overlap_type.value()
        # all_params["threshold"] = 0.5
        # all_params["patch_overlap"] = [16,16,16]
        # all_params["patch_size"] = [64,64,64]

        return all_params

    def _add_model_file(self):
        self.filewidget = FileWidget(extensions="*.pt", save=False)
        self.add_row(self.filewidget)
        self.filewidget.path_updated.connect(self.load_data)

    def _add_overlap_choice(self):
        self.overlap_type = ComboBox()
        self.overlap_type.addItem(key="crop")
        self.overlap_type.addItem(key="average")
        widget = HWidgets("Overlap:", self.overlap_type, stretch=0)
        self.add_row(widget)

    def load_data(self, path):
        self.model_fullname = path
        print(f"Setting model fullname: {self.model_fullname}")
