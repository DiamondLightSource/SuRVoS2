"""
Survos server functions

Service and HTTP class taken from Hug
"""

import re

import hug
from hug import _empty as empty
from hug.defaults import input_format
from hug.format import parse_content_type

from loguru import logger
import logging
from survos2.model.model import DataModel
import warnings
from cgi import parse_header
import requests
from io import BytesIO
from collections import namedtuple


warnings.filterwarnings("ignore")

global __plugins
__plugins = {}
global __api_started
__api_started = False


def load_settings():
    DataModel.g.load_settings()


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


def init_api():
    pass


def remote_client(uri, json_transport=False):
    host, port = _parse_uri(uri)
    endpoint = "http://{}:{}/".format(host, port)
    logger.debug(f"Contacting endpoint: {endpoint}")
    return HTTP(endpoint, timeout=5, json_transport=json_transport)


def remote_url(uri, plugin, command):
    host, port = _parse_uri(uri)
    endpoint = "http://{}:{}/{}/{}".format(host, port, plugin, command)
    return endpoint


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


Response = namedtuple("Response", ("data", "status_code", "headers"))
Request = namedtuple("Request", ("content_length", "stream", "params"))


class Service(object):
    """Defines the base concept of a consumed service.
    This is to enable encapsulating the logic of calling a service so usage can be independant of the interface
    """

    __slots__ = ("timeout", "raise_on", "version")

    def __init__(self, version=None, timeout=None, raise_on=(500,), **kwargs):
        self.version = version
        self.timeout = timeout
        self.raise_on = raise_on if type(raise_on) in (tuple, list) else (raise_on,)

    def request(
        self, method, url, url_params=empty.dict, headers=empty.dict, timeout=None, **params
    ):
        """Calls the service at the specified URL using the "CALL" method"""
        raise NotImplementedError("Concrete services must define the request method")

    def get(self, url, url_params=empty.dict, headers=empty.dict, timeout=None, **params):
        """Calls the service at the specified URL using the "GET" method"""
        return self.request("GET", url=url, headers=headers, timeout=timeout, **params)

    def post(self, url, url_params=empty.dict, headers=empty.dict, timeout=None, **params):
        """Calls the service at the specified URL using the "POST" method"""
        return self.request("POST", url=url, headers=headers, timeout=timeout, **params)

    def delete(self, url, url_params=empty.dict, headers=empty.dict, timeout=None, **params):
        """Calls the service at the specified URL using the "DELETE" method"""
        return self.request("DELETE", url=url, headers=headers, timeout=timeout, **params)

    def put(self, url, url_params=empty.dict, headers=empty.dict, timeout=None, **params):
        """Calls the service at the specified URL using the "PUT" method"""
        return self.request("PUT", url=url, headers=headers, timeout=timeout, **params)

    def trace(self, url, url_params=empty.dict, headers=empty.dict, timeout=None, **params):
        """Calls the service at the specified URL using the "TRACE" method"""
        return self.request("TRACE", url=url, headers=headers, timeout=timeout, **params)

    def patch(self, url, url_params=empty.dict, headers=empty.dict, timeout=None, **params):
        """Calls the service at the specified URL using the "PATCH" method"""
        return self.request("PATCH", url=url, headers=headers, timeout=timeout, **params)

    def options(self, url, url_params=empty.dict, headers=empty.dict, timeout=None, **params):
        """Calls the service at the specified URL using the "OPTIONS" method"""
        return self.request("OPTIONS", url=url, headers=headers, timeout=timeout, **params)

    def head(self, url, url_params=empty.dict, headers=empty.dict, timeout=None, **params):
        """Calls the service at the specified URL using the "HEAD" method"""
        return self.request("HEAD", url=url, headers=headers, timeout=timeout, **params)

    def connect(self, url, url_params=empty.dict, headers=empty.dict, timeout=None, **params):
        """Calls the service at the specified URL using the "CONNECT" method"""
        return self.request("CONNECT", url=url, headers=headers, timeout=timeout, **params)


class HTTP(Service):
    __slots__ = ("endpoint", "session", "json_transport")

    def __init__(
        self,
        endpoint,
        auth=None,
        version=None,
        headers=empty.dict,
        timeout=None,
        raise_on=(500,),
        json_transport=True,
        **kwargs,
    ):
        super().__init__(timeout=timeout, raise_on=raise_on, version=version, **kwargs)
        self.endpoint = endpoint
        self.session = requests.Session()
        self.session.auth = auth
        self.session.headers.update(headers)
        self.json_transport = json_transport

    def request(
        self,
        method,
        url,
        url_params=empty.dict,
        headers=empty.dict,
        timeout=None,
        json_transport=False,
        **params,
    ):
        url = "{0}/{1}".format(self.version, url.lstrip("/")) if self.version else url

        kwargs = {"json" if json_transport else "params": params}
        # kwargs = params
        response = self.session.request(
            method, self.endpoint + url.format(url_params), headers=headers, **kwargs
        )
        data = BytesIO(response.content)
        content_type, content_params = parse_content_type(response.headers.get("content-type", ""))
        if content_type in input_format:
            data = input_format[content_type](data, **content_params)

        if response.status_code in self.raise_on:
            raise requests.HTTPError(
                "{0} {1} occured for url: {2}".format(response.status_code, response.reason, url)
            )

        return Response(data, response.status_code, response.headers)


def run_command(plugin, command, uri=None, json_transport=False, **kwargs):
    if uri is None:
        # same as Launcher.run
        client = remote_client(DataModel.g.server_uri)
        logger.debug(f"Connecting to remote client set in DataModel {client}")
        logger.debug("{}/{}".format(plugin, command))
        response = client.get(
            "{}/{}".format(plugin, command), json_transport=json_transport, **kwargs
        )
        logger.debug(f"Received response {response}")
    else:
        client = remote_client(uri)
        logger.debug(f"Connecting to remote client {client}")
        logger.debug("{}/{}".format(plugin, command))
        response = client.get(
            "{}/{}".format(plugin, command), json_transport=json_transport, **kwargs
        )
        logger.debug(f"Received response {response}")

    return parse_response(plugin, command, response)


def parse_response(plugin, command, response, log=True):
    if response.data:
        if response.data == "Not Found" or "404" in response.data:
            errmsg = "API {}/{} does not exist.".format(plugin, command)
            if log:
                logging.critical(errmsg)
            response = errmsg
            return response, True

    return response.data, False
