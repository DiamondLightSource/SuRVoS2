from dataclasses import dataclass
import numpy as np
from typing import List
import pandas as pd
from survos2.helpers import AttrDict

@dataclass
class ClientData:
    vol_stack : np.ndarray
    layer_names : List[str]
    opacities : List[int]
    entities : pd.DataFrame
    classnames : np.ndarray
    scfg: AttrDict



