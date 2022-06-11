from pydantic.types import PositiveInt

from schemas.environment import EnvironmentArgs
from schemas.data import DataArgs
from schemas.utils import ImmutableArgs, MyBaseModel


class IntDimArgs(ImmutableArgs):
    patch_size: PositiveInt = 5
    n_points: PositiveInt = 1048576
    min_n_centroids: PositiveInt = 2
    max_n_centroids: PositiveInt = 1024
    whitening_regularization_factor: float = 0.001
    zca_whitening: bool = False


class Args(MyBaseModel):
    env = EnvironmentArgs()
    data = DataArgs()
    int_dim = IntDimArgs()
