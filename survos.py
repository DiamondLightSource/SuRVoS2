#!/usr/bin/env python
"""
List of commands


"""
import sys
import os
import begin
from loguru import logger

from survos2.config import Config
from survos2.utils import format_yaml
from survos2.survos import init_api, run_command
from survos2.model.model import DataModel


default_uri = "{}:{}".format(Config["api.host"], Config["api.port"])

# fmt = "{time} - {name} - {level} - {message}"
fmt = " <green>{name}</green> - <level>{level} - {message}</level>"
logger.remove()  # remove default logger
# logger.add(sys.stderr, level="DEBUG")
logger.add(sys.stderr, level="INFO", format=fmt, colorize=True)  # minimal stderr logger
# logger.add("logs/main.log", level="DEBUG", format=fmt) #compression='zip')
# logger.add(send_udp, format="{time} {level} {message}", filter="my_module", level="DEBUG")


@begin.subcommand
def start_server(
    workspace: "Workspace path (full or chrooted) to load",
    port: "port like URI to start the server at",
    CHROOT: "Path to Image Data",
):
    """
    Start a SuRVoS API Server for requests.
    """
    from hug.store import InMemoryStore
    from hug.middleware import SessionMiddleware, CORSMiddleware
    from survos2.model import DataModel

    full_ws_path = os.path.join(CHROOT, workspace)

    if not os.path.isdir(full_ws_path):
        logger.error(f"No workspace can be found at {full_ws_path}, aborting.")
        sys.exit(1)
    logger.info(f"Full workspace path is {full_ws_path}")

    DataModel.g.CHROOT = CHROOT
    DataModel.g.current_workspace = workspace

    api, __plugins = init_api(return_plugins=True)

    session_store = InMemoryStore()
    middleware = SessionMiddleware(
        session_store, cookie_name="survos_sid", cookie_secure=False, cookie_path="/"
    )
    api.http.add_middleware(middleware)
    api.http.add_middleware(CORSMiddleware(api, max_age=10))

    logger.debug(f"Starting server on port {port} with workspace {workspace}")
    api.http.serve(port=int(port), display_intro=False)


@begin.subcommand
def run_server(
    command: "Command to execute in `plugin.action` format.",
    server: "URI to the remote SuRVoS API Server",
    *args: "Extra keyword arguments in the form of `key=value` "
    "required by plugin/commands. Please refer to SuRVoS API.",
):

    """
    Run a plugin/command from terminal. If remote is `None` it will use
    the local SuRVoS installation.
    """

    # workaround issue with hug function args
    if server == "server=0:0":  # use server=0:0 for local
        server = None

    plugin, command = command.split(".")
    args = [k.split("=") for k in args]
    params = {k: v for k, v in args}

    print(f"Running command: {plugin} {command} on {server}")
    result = run_command(plugin, command, uri=server, **params)[0]

    if type(result) == dict:
        result = format_yaml(result)

    logger.info(result)


@begin.subcommand
def nu_gui(
    workspace: "Workspace path (full or chrooted) to load",
    server: "URI to the remote SuRVoS API Server",
):

    from survos2.frontend import main
    from napari import gui_qt
    import napari

    logger.debug(f"Starting nu_gui frontend with workspace {workspace}")

    from survos2.frontend.control import Launcher
    from survos2.model import DataModel

    DataModel.g.current_workspace = workspace

    logger.info(f"Connecting to server: {server}")
    resp = Launcher.g.set_remote(server)
    logger.info(f"Response from server: {resp}")

    if Launcher.g.connected:
        main.start_client()


@begin.subcommand
def list_plugins():
    """
    List available plugins in SuRVoS.
    """
    logger.info("Available plugins:")
    plugins = init_api(return_plugins=True)[1]
    for plugin in plugins:
        logger.info(" - " + plugins[plugin]["name"])


@begin.subcommand
def load_metadata(source: "Dataset URI to load in HDF5, MRC or SuRVoS format"):
    from survos2.io import dataset_from_uri, supports_metadata

    if supports_metadata(source):
        source = dataset_from_uri(source)
        logger.info(format_yaml(source.metadata()))
        source.close()
    else:
        logger.info("Dataset `{}` has no metadata.".format(source))


@begin.subcommand
def process(
    pfile: "Process file with the plugin+command instructions",
    remote: "Execute the commands in a remote server" = False,
    uri: "URI to the remote SuRVoS API Server" = default_uri,
):
    import yaml

    uri = uri if remote else None

    print(pfile)

    if not os.path.isabs(pfile):
        fworkflows = os.path.join(os.getcwd(), pfile)
    else:
        fworkflows = pfile

    with open(fworkflows) as f:
        workflows = yaml.safe_load(f.read())

    for workflow in workflows:
        name = workflow.pop("name", "Workflow")
        plugin = workflow.pop("plugin")
        command = workflow.pop("command")

        print("Running workflow:", name)

        run_command(plugin, command, workflow, remote=args.remote)


@begin.start(auto_convert=True, cmd_delim="--")
def run(name="SuRVoS", quest="Segmentation"):
    "SuRVoS"
