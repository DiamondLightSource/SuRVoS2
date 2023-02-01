import hug
import multiprocessing
from requests.exceptions import ConnectTimeout, ConnectionError
from qtpy import QtCore, QtWidgets
from hug.use import Local  # HTTP,
import pickle
from survos2.model import DataModel
from survos2.model.singleton import Singleton
from survos2.utils import encode_numpy, format_yaml, Timer
from survos2.survos import remote_client, parse_response, init_api
from loguru import logger
import requests
from hug import _empty as empty
from cgi import parse_header
import requests
from io import BytesIO


# needed as separate function rather than method for multiprocessing
def _run_command(plugin, command, client, uri=None, out=None, **kwargs):
    logger.debug(f"bg _run_command: {plugin} {command}")
    response = client.get("{}/{}".format(plugin, command), **kwargs)
    logger.debug(f"bg parsing response")
    result = parse_response(plugin, command, response, log=False)
    if out is not None:
        out.put(result)
    else:
        return result


def parse_content_type(content_type):
    """Separates out the parameters from the content_type and returns both in a tuple (content_type, parameters)"""
    if content_type is not None and ";" in content_type:
        return parse_header(content_type)
    return (content_type, empty.dict)


input_format = {
    "application/json": hug.input_format.json,
    "application/x-www-form-urlencoded": hug.input_format.urlencoded,
    "multipart/form-data": hug.input_format.multipart,
    "text/plain": hug.input_format.text,
    "text/css": hug.input_format.text,
    "text/html": hug.input_format.text,
}

from collections import namedtuple

Response = namedtuple("Response", ("data", "status_code", "headers"))


def _parse_uri(uri):
    if type(uri) == str:

        pattern = r"@?(?P<host>\w+):(?P<port>[0-9]+)"
        result = re.search(pattern, uri)

        ip = re.findall(r"[0-9]+(?:\.[0-9]+){3}", uri)

        if result is not None:
            return ip[0], result["port"]
    elif len(uri) == 2:
        return uri[0], uri[1]
    raise ValueError("Not a valid [host:port] URI.")


def remote_url(uri, plugin, command):
    host, port = _parse_uri(uri)
    endpoint = "http://{}:{}/{}/{}".format(host, port, plugin, command)
    logger.debug(f"endpoint: {endpoint}")
    return endpoint


@Singleton
class Launcher(QtCore.QObject):
    def __init__(self):
        super().__init__()
        self.connected = True
        self.terminated = False
        self.queue = None
        self.client = None
        self.remote_ip_port = None

    def set_remote(self, uri):
        logger.info("Launcher setting remote to {}".format(uri))
        self.client = remote_client(uri)
        self.modal = False
        self.pending = []
        self.remote_ip_port = uri

    def set_current_workspace(self, workspace_name):
        logger.debug(f"Setting workspace to {workspace_name}")
        DataModel.g.current_workspace = workspace_name

    def set_current_workspace_shape(self, workspace_shape, **kwargs):
        logger.debug(f"Setting workspace shape to {workspace_shape}")
        DataModel.g.current_workspace_shape = workspace_shape

    def post_array(self, arr, group, name="ndarray", **kwargs):
        logger.debug(f"Posting array of shape {arr.shape}")
        data = {"name": name, "data": arr}

        data_as_bytes = pickle.dumps(data)
        response = requests.post(
            "http://" + self.remote_ip_port + "/" + group + "/upload",
            files={"file": data_as_bytes},
        )

    def post_file(self, fullname, group, **kwargs):
        with open(fullname, "rb") as file_handle:
            response = requests.post(
                "http://" + self.remote_ip_port + "/" + group + "/upload",
                files={"file": file_handle},
            )

    def run(self, plugin, command, modal=False, **kwargs):
        if self.client:
            if self.terminated:
                return False

            workspace = kwargs.pop("workspace", None)
            logger.debug(f"Running command {command} with plugin {plugin} in workspace {workspace}")

            self.modal = modal
            self.title = "{}::{}".format(plugin.capitalize(), command.capitalize())
            self.setup(self.title)

            # use default workspace or instead use the passed in workspace

            logger.debug(f"Running using workspace {workspace}")
            if workspace == True:
                if DataModel.g.current_workspace:
                    kwargs["workspace"] = DataModel.g.current_workspace
                else:
                    return self.process_error("Workspace required but not loaded.")
            elif workspace is not None:
                kwargs["workspace"] = workspace

            func = self._run_background if modal else self._run_command
            success = False
            result = ""
            error = False
            cnt = 0
            while not success and cnt < 100:
                try:
                    if kwargs.pop("timeit", False):
                        with Timer(self.title):
                            result, error = func(plugin, command, **kwargs)
                    else:
                        result, error = func(plugin, command, **kwargs)

                except (ConnectTimeout, ConnectionError):
                    self.connected = False
                    cnt += 1
                    logger.info("ConnectionError - delayed")

                    if self.terminated:
                        return False

                else:
                    success = True

            if error:
                return self.process_error(result)

            self.cleanup()

            return result

    def _run_command(self, plugin, command, uri=None, out=None, **kwargs):
        logger.debug(f"_run_command: {plugin} {command}")
        response = self.client.get("{}/{}".format(plugin, command), timeout=12, **kwargs)
        result = parse_response(plugin, command, response, log=False)
        if out is not None:
            out.put(result)
        else:
            return result

    def _run_background(self, plugin, command, **kwargs):
        logger.debug("Running command in background.")
        queue = multiprocessing.Queue()
        kwargs.update(out=queue)

        p = multiprocessing.Process(
            target=_run_command, args=[plugin, command, self.client], kwargs=kwargs
        )
        p.daemon = True
        p.start()
        while p.is_alive():
            QtWidgets.QApplication.processEvents()
            p.join(0.1)
        return queue.get()

    def reconnect(self):
        try:
            params = dict(workspace=DataModel.g.current_workspace)
            self._run_command("workspace", "list_datasets", **params)

        except (ConnectTimeout, ConnectionError):
            pass
        else:
            self.connected = True

    def setup(self, caption):
        logger.debug("### {} ###".format(caption))

    def cleanup(self):
        QtWidgets.QApplication.processEvents()

    def process_error(self, error):
        if not isinstance(error, str):
            error = format_yaml(error, explicit_start=False, explicit_end=False, flow=False)

        logger.error("{} :: {}".format(self.title, error))
        QtWidgets.QApplication.processEvents()
        return False

    def terminate(self):
        pass
