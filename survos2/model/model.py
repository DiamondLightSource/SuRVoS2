from .singleton import Singleton

from survos2.config import Config

from loguru import logger
import tempfile
import os


@Singleton
class DataModel(object):
    def __init__(self):
        self.server_uri = None
        self.current_session = "default"
        self.current_workspace = ""
        self.current_workspace_shape = (1, 1, 1)
        self.CHROOT = Config["model.chroot"]
        self.DATABASE = Config["model.dbtype"]
        self.CHUNK_DATA = Config["computing.chunks"]
        self.CHUNK_SIZE = Config["computing.chunk_size"]
        self.device = Config["computing.device"]

    def load_settings(self):
        print("Loading settings from yaml")
        Config.update_yaml()

        self.CHROOT = Config["model.chroot"]
        self.DATABASE = Config["model.dbtype"]
        self.CHUNK_DATA = Config["computing.chunks"]
        self.CHUNK_SIZE = Config["computing.chunk_size"]

        if self.CHROOT in ["tmp", "temp"]:
            tmp = tempfile.gettempdir()
            self.CHROOT = os.path.join(tmp, "tmp_survos_chroot")

        os.makedirs(self.CHROOT, exist_ok=True)
        logger.info(f"CHROOT is {self.CHROOT}")

    def dataset_uri(self, dataset, group=None):
        session = self.current_session
        workspace = self.current_workspace
        if group:
            params = session, workspace, group, dataset
            return "survos://{}@{}:{}/{}".format(*params)
        return "survos://{}@{}:{}".format(session, workspace, dataset)

    def dataset_name(self, dataset_uri):
        return dataset_uri.split(":")[-1]

