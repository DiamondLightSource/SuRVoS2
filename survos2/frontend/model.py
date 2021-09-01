from dataclasses import dataclass
from survos2.helpers import AttrDict


@dataclass
class ClientData:
    cfg: AttrDict
