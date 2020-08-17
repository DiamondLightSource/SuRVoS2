import hug
import time
import multiprocessing
from requests.exceptions import ConnectTimeout, ConnectionError
from qtpy import QtCore, QtWidgets
from hug.use import HTTP, Local

from survos2.frontend.control.model import DataModel
from survos2.frontend.control.singleton import Singleton
#from survos2.ui.qt.qtcompat import QtCore, QtWidgets
from survos2.frontend.modal import ModalManager
from survos2.utils import format_yaml, get_logger, Timer
from survos2.survos import remote_client, parse_response, init_api
from survos2.config import Config
from survos2.api.utils import APIException, handle_exceptions, handle_api_exceptions

from importlib import import_module
import logging

logger = get_logger()

from survos2.survos import run_command



@Singleton
class Launcher(QtCore.QObject):
    def __init__(self):
        super().__init__()
        self.connected = True
        self.terminated = False

    def set_remote(self, uri):
        self.client = remote_client(uri)
        self.modal = False
        self.pending = []

    def run(self, plugin, command, modal=False, **kwargs):
        if self.terminated:
            return False

        self.modal = modal
        self.title = '{}::{}'.format(plugin.capitalize(),
                                     command.capitalize())
        self.setup(self.title)

        # use default workspace or instead use the passed in workspace
        workspace = kwargs.pop('workspace', None)
        logger.info(f'Running using workspace {workspace}')        
        if workspace == True:
            if DataModel.g.current_workspace:
                kwargs['workspace'] = DataModel.g.current_workspace
            else:
                return self.process_error('Workspace required but not loaded.')
        elif workspace is not None:
            kwargs['workspace'] = workspace

        func = self._run_background if modal else self._run_command
        success = False

        while not success:
            try:
                if kwargs.pop('timeit', False):
                    with Timer(self.title):
                        result, error = func(plugin, command, **kwargs)
                else:
                    result, error = func(plugin, command, **kwargs)
        
            except (ConnectTimeout, ConnectionError):
                
                self.connected = False
                logger.info('ConnectionError - delayed')
                #ModalManager.g.connection_lost()

                if self.terminated:
                    return False
            else:
                success = True

        if error:
            return self.process_error(result)

        self.cleanup()
        return result

    def _run_command(self, plugin, command, uri=None, out=None, **kwargs):
        if uri is None:
            return run_command(plugin, command, uri=None, **kwargs)    
        else:
            # remote client, send (plugin, command)
            response = self.client.get('{}/{}'.format(plugin, command), **kwargs)        
            result = parse_response(plugin, command, response, log=False)        
            if out is not None:
                out.put(result)
            else:
                return result

    def _run_background(self, plugin, command, **kwargs):
        queue = multiprocessing.Queue()
        kwargs.update(out=queue)

        p = multiprocessing.Process(target=self._run_command,
                                    args=[plugin, command],
                                    kwargs=kwargs)
        p.daemon = True
        p.start()
        while p.is_alive():
            QtWidgets.QApplication.processEvents()
            p.join(0.1)
        return queue.get()

    def reconnect(self):
        try:
            params = dict(workspace=DataModel.g.current_workspace)
            self._run_command('workspace', 'list_datasets', **params)
        except (ConnectTimeout, ConnectionError):
            pass
        else:
            self.connected = True
    def setup(self, caption):
        logger.info('### {} ###'.format(caption))
        #if self.modal:
        #    ModalManager.g.show_loading(caption)

    def cleanup(self):
        #if self.modal:
        #    ModalManager.g.hide()
        QtWidgets.QApplication.processEvents()

    def process_error(self, error):
        if not isinstance(error, str):
            error = format_yaml(error, explicit_start=False, explicit_end=False, flow=False)
        try:
            traceback.print_last()
        except Exception as e:
            pass
        
        logger.error('{} :: {}'.format(self.title, error))
        ModalManager.g.show_error(error)
        QtWidgets.QApplication.processEvents()
        return False

    def terminate(self):
        pass