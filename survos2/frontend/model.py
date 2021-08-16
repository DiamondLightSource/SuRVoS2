from dataclasses import dataclass
from survos2.helpers import AttrDict


@dataclass
class ClientData:
    # vol_stack: np.ndarray
    # layer_names: List[str]
    # opacities: List[int]
    # entities : pd.DataFrame
    # classnames : np.ndarray
    cfg: AttrDict
