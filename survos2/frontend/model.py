
from dataclasses import dataclass
import numpy as np
from survos2.server.model import SRFeatures
from typing import List
import pandas as pd

@dataclass
class ClientData:
    vol_stack : np.ndarray
    layer_names : List[str]
    opacities : List[int]
    entities : pd.DataFrame
    classnames : np.ndarray


class SegSubject:
    def __init__(self):
        self.listeners = []

    def __iadd__(self, listener):
        self.listeners.append(listener)
        return self

    def notify(self, *args, **kwargs):
        for listener in self.listeners:
            listener(*args, **kwargs)

