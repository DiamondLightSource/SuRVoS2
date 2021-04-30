import yaml
import pprint
from survos2.helpers import AttrDict
from survos2.config import config

# config object used as a part of the application state

cfg = {}
cfg["current_annotation"] = "001_level"
cfg["torch_models_fullpath"] = "../experiments"
cfg["filter_cfg"] = config["filters"].copy()
cfg["pipeline"] = config["pipeline"].copy()
cfg["retrieval_mode"] = "volume"
cfg = AttrDict(cfg)
