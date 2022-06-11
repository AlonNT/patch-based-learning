from typing import Literal, Optional, List, Union
from pydantic import validator
from pydantic.types import PositiveInt

from schemas.utils import ImmutableArgs, NonNegativeInt, NonOneFraction
from vgg import configs


class ArchitectureArgs(ImmutableArgs):

    #: The model name for the network architecture.
    model_name: str = 'VGG11c'

    #: How many hidden layers the final MLP at the end of the convolution blocks.
    final_mlp_n_hidden_layers: NonNegativeInt = 1

    #: Dimension of each hidden layer the final MLP at the end of the convolution blocks.
    final_mlp_hidden_dim: PositiveInt = 128

    #: Dropout probability (will be added after each non-linearity).
    dropout_prob: NonOneFraction = 0

    input_channels: PositiveInt = 3
    input_spatial_size: PositiveInt = 32

    #: The kernel-size to use in each convolution-layer.
    kernel_size: Union[PositiveInt, List[PositiveInt]] = 5

    #: Stride to use in the convolution-layer.
    stride: Union[PositiveInt, List[PositiveInt]] = 1

    #: The padding amount to use in each convolution-layer.
    padding: Union[NonNegativeInt, List[NonNegativeInt]] = 0

    #: The pooling size and stride to use in the AvgPool / MaxPool layers.
    pool_size: Union[PositiveInt, List[PositiveInt]] = 4
    pool_stride: Union[PositiveInt, List[PositiveInt]] = 4

    #: Whether to use batch-normalization layer after the Conv -> ReLU (and possible pool) part in the block.
    use_batch_norm: Union[bool, List[bool]] = True

    #: If it's greater than zero, adds a 1x1 convolution layer ("bottleneck") in the end of the block.
    bottle_neck_dimension: Union[NonNegativeInt, List[NonNegativeInt]] = 32

    #: The size for the bottleneck layer(s).
    bottle_neck_kernel_size: Union[PositiveInt, List[PositiveInt]] = 1

    #: Padding mode for the convolution layers.
    padding_mode: Literal['zeros', 'circular'] = 'zeros'

    @validator('model_name', always=True)
    def validate_model_name(cls, v):
        assert v == 'mlp' or v in configs.keys(), f"model_name {v} is not supported, " \
                                                  f"should be 'mlp' or one of {list(configs.keys())}"
        return v
