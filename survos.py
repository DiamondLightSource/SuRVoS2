#!/usr/bin/env python
"""
List of commands


"""
import sys
import os
import logging
import begin

from survos2.config import Config
from survos2.utils import format_yaml, get_logger
from survos2.survos import init_api, run_command

import numpy  as np
from napari import gui_qt
import napari

from survos2.frontend import main

logger = get_logger(level=logging.INFO)
default_uri = '{}:{}'.format(Config['api.host'], Config['api.port'])


######################################################################
# COMMANDS MANAGEMENT
######################################################################


@begin.subcommand
def start_server(port:'port like URI to start the server at'=Config['api.port']):
    """
    Start a SuRVoS API Server listeting for requests.
    """
    from hug.store import InMemoryStore
    from hug.middleware import SessionMiddleware, CORSMiddleware

    api, __plugins = init_api(return_plugins=True)
    logging.info(f"Initializing api: {api}")
    logging.info(f"Loaded plugins: {__plugins}")
    
    
    session_store = InMemoryStore()
    middleware = SessionMiddleware(session_store,
                                   cookie_name='survos_sid',
                                   cookie_secure=False,
                                   cookie_path='/')
    api.http.add_middleware(middleware)
    CORSMiddleware(api)

    api.http.serve(port=port, display_intro=False)



@begin.subcommand
def run_server(command:'Command to execute in `plugin.action` format.',
        server:'URI to the remote SuRVoS API Server',
        *args:'Extra keyword arguments in the form of `key=value` '
              'required by plugin/commands. Please refer to SuRVoS API.'):
    
    """
    Run a plugin/command from terminal. If remote is `None` it will use
    the local SuRVoS installation.
    """
    
    #workaround issue with hug function args
    if server=='server=0:0':   #use server=0:0 for local
        server = None

    plugin, command = command.split('.')
    args = [k.split('=') for k in args]
    params = {k: v for k, v in args}
    
    print(f'Running command: {plugin} {command} on {server}')
    result = run_command(plugin, command, uri=server, **params)[0]
    
    if type(result) == dict:
        result = format_yaml(result)
    
    logger.info(result)


@begin.subcommand
def nu_gui(workspace:'Workspace path (full or chrooted) to load',
        project_file:'JSON project file',
       server:'URI to the remote SuRVoS API Server'=default_uri):

    logger.debug(f"Starting nu_gui frontend with workspace {workspace}")

    from survos2.frontend.control import Launcher, DataModel
    DataModel.g.current_workspace = workspace
    
    logger.info(f"Connecting to server: {server}")
    resp = Launcher.g.set_remote(server)

    logger.info(f"Response from server: {resp}")
    if Launcher.g.connected:
        main.startup(name='brain', project_file=project_file)


@begin.subcommand
def classic_gui(workspace:'Workspace path (full or chrooted) to load',
       server:'URI to the remote SuRVoS API Server'=default_uri):
    """
    Show Classic SuRVoS QT user interface
    """
    #from survos2.ui.qt import QtWidgets
    from qtpy import QtWidgets
    #from survos2.frontend.mainwindow2 import MainWindow
    #from survos2.frontend.control import Launcher, DataModel
    #from survos2.ui.qt import QtWidgets
    from survos2.ui.qt import MainWindow
    from survos2.ui.qt.control import Launcher, DataModel

    DataModel.g.current_workspace = workspace

    logger.info(f"Connecting to server: {server}")
    resp = Launcher.g.set_remote(server)
    logger.info(f"Response from server: {resp}")
    
    app = QtWidgets.QApplication([])
    window = MainWindow(maximize=bool(Config['qtui.maximized']))

    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    sys.exit(app.exec_())




# old
@begin.subcommand
def process(pfile:'Process file with the plugin+command instructions',
            remote:'Execute the commands in a remote server'=False,
            uri:'URI to the remote SuRVoS API Server'=default_uri):
    import yaml
    uri = uri if remote else None

    print(pfile)


    if not os.path.isabs(args.workflow):
        fworkflows = os.path.join(os.getcwd(), args.workflow)
    else:
        fworkflows = args.workflow

    with open(fworkflows) as f:
        workflows = yaml.safe_load(f.read())

    for workflow in workflows:
        name = workflow.pop('name', 'Workflow')
        plugin = workflow.pop('plugin')
        command = workflow.pop('command')

        print("Running workflow:", name)

        run_command(plugin, command, workflow, remote=args.remote)



@begin.subcommand
def list_plugins():
    """
    List available plugins in SuRVoS.
    """
    logger.info("Available plugins:")
    plugins = init_api(return_plugins=True)[1]
    for plugin in plugins:
        logger.info(' - ' + plugins[plugin]['name'])


@begin.subcommand
def load_metadata(source:'Dataset URI to load in HDF5, MRC or SuRVoS format'):
    from survos2.io import dataset_from_uri, supports_metadata
    if supports_metadata(source):
        source = dataset_from_uri(source)
        logger.info(format_yaml(source.metadata()))
        source.close()
    else:
        logger.info('Dataset `{}` has no metadata.'.format(source))


@begin.subcommand
def view_data(source:'Dataset URI to load in HDF5, MRC or SuRVoS format',
         boundaries:'Boundaries to show on top of the `source`'=None,
         overlay:'Overlay dataset to show on top of the `source`'=None,
         bcolor:'Color of the overlaid boundaries'='#000099',
         balpha:'Alpha of the overlaid boundaries'=0.7,
         oalpha:'Overlay alpha.'=0.5):
    """
    Visualizes a 3D volume with sliders.
    Allows to overlay a segmentation / other image and to overlay
    boundaries, from either supervoxels or other boundary extraction
    method.
    """
    from survos2.io import dataset_from_uri
    from survos2.volume_utils import view

    logger.info(f'Loading source volume {source}')
    source = dataset_from_uri(source)
    if boundaries:
        logger.info('Loading boundaries')
        boundaries = dataset_from_uri(boundaries)
    if overlay:
        logger.info('Loading overlay')
        overlay = dataset_from_uri(overlay)

    view(source, boundaries=boundaries, overlay=overlay,
         bcolor=bcolor, balpha=balpha, oalpha=oalpha)

    source.close()
    if boundaries:
        boundaries.close()
    if overlay:
        overlay.close()




@begin.subcommand
def test_feats():
    from survos2.model import Workspace
    from survos2.data import mrbrain, embrain
    from survos2.io import dataset_from_uri
    from survos2.api.regions import get_slice

    if Workspace.exists('test_survos'):
        ws = Workspace('test_survos')
        print("Found workspace")
        #logger.info('Removing existing test workspace')
        #Workspace.remove('test_survos')
        #print(Workspace.has_dataset())
    else:
        logger.info('Creating test workspace')
        ws = Workspace.create('test_survos')
    
    
    with dataset_from_uri(mrbrain()) as data:
        ws.add_data(data)

    print(ws.tojson())

    #if not (ws.has_dataset('features/gauss')):
    #    ws.add_dataset('features/gauss', np.float32)
    #ds = ws.get_dataset('features/gauss')
    
    with dataset_from_uri(mrbrain()) as data:
        ds.set_data(data)
        ds[...] = data
        ds.load(data)

    print(ws.path())
    print(ws.tojson())


@begin.subcommand
def test_workspace():
    
    from survos2.model import Workspace
    from survos2.data import mrbrain, embrain
    from survos2.io import dataset_from_uri

    if Workspace.exists('test_survos'):
        logger.info('Removing existing test workspace')
        Workspace.remove('test_survos')
    logger.info('Creating test workspace')
    ws = Workspace.create('test_survos')
    
    with dataset_from_uri(mrbrain()) as data:
        ws.add_data(data)


@begin.start(auto_convert=True, cmd_delim='--')
def run(name='Gawain', quest='Holy Grail', colour='greenblue'):
    "We are the k.ni.ghts of the round table..."
