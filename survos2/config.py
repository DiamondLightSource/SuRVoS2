import yaml
import os
import os.path as op


class _Config(type):
    __data__ = {  # Defaults
        "title": "SuRVoS",
        "api": {
            "host": "127.0.0.1",
            "port": 8123,
            "plugins": [],
            "renderer": "mpl",
        },
        "computing": {
            "chunks": True,
            "chunk_size": 32,
            "chunk_padding": 8,
            "chunk_size_sparse": 10,
            "scale": False,
            "stretch": False,
            "device": 0,
        },
        "model": {
            "chroot": "/",  # default location to store data
            "dbtype": "yaml",
        },
        "logging": {
            "overall_level": "INFO",
            "file": "",
            "level": "error",
            "std": True,
            "std_format": "%(levelname)8s | %(message)s",
            "file_format": "%(asctime)s - | %(levelname)8s | %(message)s",
        },
        "qtui": {"maximized": False, "menuKey": "\\"},
        "filters": {},
        "pipeline": {},
        "slic": "skimage",
        "volume_mode": "volume_http",
        "volume_segmantics": {
            "train_settings": {
                "end_lr": 50,
                "training_set_proportion": 0.8,
                "eval_metric": "MeanIoU",
                "num_cyc_frozen": 8,
                "num_cyc_unfrozen": 5,
                "image_size": 256,
                "data_hdf5_path": "/data",
                "patience": 3,
                "beta": 0.25,
                "model_output_fn": "trained_2d_model",
                "starting_lr": "1e-6",
                "data_im_dirname": "data",
                "alpha": 0.75,
                "st_dev_factor": 2.575,
                "cuda_device": 0,
                "loss_criterion": "DiceLoss",
                "lr_reduce_factor": 500,
                "downsample": False,
                "seg_im_out_dirname": "seg",
                "seg_hdf5_path": "/data",
                "clip_data": True,
                "model": {
                    "type": "U_Net",
                    "encoder_weights": "imagenet",
                    "encoder_name": "resnet34",
                },
                "pct_lr_inc": 0.3,
                "lr_find_epochs": 1,
            },
            "predict_settings": {
                "cuda_device": 0,
                "output_probs": False,
                "downsample": False,
                "st_dev_factor": 2.575,
                "data_hdf5_path": "/data",
                "clip_data": True,
                "quality": "medium",
                "one_hot": False,
            },
        },
    }

    def __getitem__(self, key):
        return self.get(key)

    def get(self, key):
        keys = key.split(".")
        data = self.__data__
        for i, key in enumerate(keys):
            if key in data:
                data = data[key]
            else:
                raise KeyError("Config does not contain key `{}`".format(".".join(keys[: i + 1])))
        return data

    def __contains__(self, key):
        try:
            self.get(key)
        except KeyError:
            return False
        return True


class Config(object, metaclass=_Config):
    @staticmethod
    def update(data):
        for k, v in data.items():
            if k == "environments":
                continue
            if type(v) == dict:
                _Config.__data__[k].update(v)
            else:
                _Config.__data__[k] = v

    def __repr__(self):
        return repr(_Config.__data__)

    def print_contents():
        print(_Config.__data__)

    @staticmethod
    def update_yaml():
        settings_yaml = op.join(op.dirname(__file__), "..", "settings.yaml")
        print(settings_yaml)
        if op.isfile(settings_yaml):
            with open(settings_yaml, "r") as __f:
                print("Updating Config")
                yaml_dict = yaml.safe_load(__f)
                for k, v in yaml_dict.items():
                    print(k, v)
                    if k == "environments":
                        continue
                    if type(v) == dict:
                        print("Updating dict")
                        _Config.__data__[k].update(v)
                    else:
                        print("Updating single key")
                        _Config.__data__[k] = v

    def write_yaml():
        settings_yaml = op.join(op.dirname(__file__), "..", "settings.yaml")
        with open(settings_yaml, "w") as outfile:
            yaml.dump(_Config.__data__, outfile, default_flow_style=False)


__default_config_files__ = [
    op.join(op.dirname(__file__), "..", "settings.yaml"),
    op.join(op.expanduser("~"), ".survosrc"),
]

#
# Load available config from environment
#
for __config_file in __default_config_files__:
    configs = []
    if op.isfile(__config_file):
        with open(__config_file, "r") as __f:
            configs.append(yaml.safe_load(__f))

    # Load all the default config
    for config in configs:
        Config.update(config)

    # Overwrite with the enviromental config
    # e.g. activate test environment with SURVOS_ENV=test
    for config in configs:
        envs = config.get("environments", [])
        if envs and "SURVOS_ENV" in os.environ and os.environ["SURVOS_ENV"] in envs:
            Config.update(envs[os.environ["SURVOS_ENV"]])

    # Overwrite with `all` special environment
    for config in configs:
        envs = config.get("environments", [])
        if envs and "all" in envs:
            Config.update(envs["all"])


# Overwrite config with enviromental variables SURVOS_$section_$setting
for k1, v in _Config.__data__.items():
    if type(v) == dict:
        for k2 in v:
            env_name = "SURVOS_{}_{}".format(k1.upper(), k2.upper())
            if env_name in os.environ:
                try:
                    dtype = type(Config[k1][k2])
                    Config[k1][k2] = dtype(os.environ[env_name])
                except ValueError:
                    raise ValueError(
                        "Error updating config {}.{} to {}.".format(k1, k2, os.environ[env_name])
                    )

