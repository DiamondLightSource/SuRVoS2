"""
Survos server functions

"""

import re
import hug
from hug.use import HTTP, Local

import logging

from loguru import logger
from importlib import import_module

from .config import Config
from .api.utils import APIException, handle_exceptions, handle_api_exceptions

global __plugins
__plugins = {}
global __api_started
__api_started = False


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


def init_api(return_plugins=False):
    global __api_started

    if __api_started:
        logger.info("Api already started.")
        return

    # Init app
    api = hug.API(__name__)
    logger.debug(f"API: {api}\n")

    # Set exception handlers
    hug.exception(api=api)(handle_exceptions)
    hug.exception(APIException, api=api)(handle_api_exceptions)

    # Load plugins
    __plugins.clear()

    logger.info(f"Config plugins: {Config['api.plugins']}")

    if Config["api.plugins"]:
        logger.info("Configuring api plugins")
        for plugin_name in Config["api.plugins"]:
            logger.debug(f"Loading plugin {plugin_name}")
            plugin_module = "survos2.api.{}".format(plugin_name)
            plugin_path = "/" + plugin_name
            plugin = import_module(plugin_module)

            api.extend(plugin, plugin_path)

            __plugins[plugin_name] = dict(
                name=plugin_name, modname=plugin_module, path=plugin_path, module=plugin
            )

    __api_started = True

    if return_plugins:
        return api, __plugins

    logger.debug(f"Init api returned {api}")
    return api


def remote_client(uri):
    host, port = _parse_uri(uri)
    endpoint = "http://{}:{}/".format(host, port)
    logger.info(f"Contacting endpoint {endpoint}")
    return HTTP(endpoint)


def run_command(plugin, command, uri=None, **kwargs):
    global __api_started
    global plugins

    if uri is None:
        if not __api_started:
            api, plugins_loaded = init_api(return_plugins=True)
            __api_started = True
            logger.debug(api)
            logger.debug(f"plugins: {__plugins}")

        client = Local(__plugins[plugin]["module"])
        logger.debug(f"Using client {client}")

        try:
            logger.info(f"get request to client: {command}")
            response = client.get(command, **kwargs)
            logger.info(f"Local client gave response {response}")

        except APIException as e:
            if not e.critical:
                e.critical = Config["logging.level"].lower() == "debug"
            handle_api_exceptions(e)
            return str(e), True
    else:
        client = remote_client(uri)
        logger.info(f"Connecting to remote client {client}")
        logger.info('{}/{}'.format(plugin, command))
        response = client.get("{}/{}".format(plugin, command), **kwargs)
        logger.info(f"Received response {response}")

    return parse_response(plugin, command, response)


def parse_response(plugin, command, response, log=True):
    if response.data == "Not Found" or "404" in response.data:
        errmsg = "API {}/{} does not exist.".format(plugin, command)
        if log:
            logging.critical(errmsg)
        response = errmsg
        return response, True

    elif response.data["error"]:
        errmsg = response.data["error_message"]
        if log:
            if response.data["critical"]:
                logging.critical(errmsg)
            else:
                logging.error(errmsg)
        return errmsg, True

    elif "errors" in response.data["data"]:
        errmsg = response.data["data"]["errors"]
        if log:
            logging.critical(errmsg)
        return response.data["data"]["errors"], True

    return response.data["data"], False
