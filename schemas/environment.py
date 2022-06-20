import datetime
from pathlib import Path
from typing import List, Literal, Optional, Union

import torch
from pydantic import validator
from pydantic.types import PositiveInt, DirectoryPath

from consts import DATETIME_STRING_FORMAT
from schemas.utils import ImmutableArgs, NonNegativeInt


class EnvironmentArgs(ImmutableArgs):

    #: On which device to train.
    device: Literal['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cuda:4', 'cuda:5', 'cuda:6', 'cuda:7'] = 'cpu'

    #: Output path for the experiment - a sub-directory named with the date & time will be created within.
    path: DirectoryPath = './experiments'

    #: The name to give the run in wandb
    wandb_run_name: Optional[str] = None

    #: The name to give the run in wandb
    wandb_project_name: str = 'patch-based-learning'

    #: Debug mode means limiting the number of batches during training, etc.
    debug: bool = False

    #: Debug mode means limiting the number of batches during training, etc.
    multi_gpu: Union[NonNegativeInt, List[NonNegativeInt]] = 0

    #: Whether to use faiss for running k-means clustering (much faster than sklearn).
    use_faiss: bool = False

    @validator('path', pre=True)
    def create_parent_out_dir_if_not_exists(cls, v: str):
        if not Path(v).exists():
            Path(v).mkdir()
        return v

    @validator('path', always=True)
    def create_out_dir(cls, v: Path):
        datetime_string = datetime.datetime.now().strftime(DATETIME_STRING_FORMAT)
        out_dir = v / datetime_string
        out_dir.mkdir(exist_ok=True)  # exist_ok because this validator is being called multiple times (I think)
        return out_dir

    @validator('device')
    def validate_device_exists(cls, v):
        if v.startswith('cuda:'):
            assert torch.cuda.is_available(), f"CUDA is not available, so can't use device {v}"
            assert int(v[-1]) < torch.cuda.device_count(), f"GPU index {v[-1]} is higher than the number of GPUs."
        return v

    @property
    def is_cuda(self) -> bool:
        return self.device.startswith('cuda:')

    @property
    def device_num(self) -> int:
        assert self.is_cuda, "When asking for device_num it must be on CUDA and not CPU."
        return int(self.device.replace('cuda:', ''))
