from pydantic import validator
from pydantic.types import PositiveInt
from typing import List, Union
from schemas.utils import ImmutableArgs, ProperFraction, NonNegativeFloat


class OptimizationArgs(ImmutableArgs):

    #: Number of epochs to train.
    epochs: Union[PositiveInt, List[PositiveInt]] = 200

    #: Mini batch size to use in each training-step.
    batch_size: Union[PositiveInt, List[PositiveInt]] = 64

    #: Momentum to use in SGD optimizer.
    momentum: ProperFraction = 0.9

    #: Amount of weight decay (regularization).
    weight_decay: NonNegativeFloat = 0

    # The initial learning-rate which might later be decayed.
    learning_rate: Union[ProperFraction, List[ProperFraction]] = 0.003

    #: Decay the learning-rate at these steps by a factor of `learning_rate_decay_gamma`.
    learning_rate_decay_steps: List[PositiveInt] = [100, 150]

    #: The factor gamma to multiply the learning-rate at the decay steps.
    learning_rate_decay_gamma: ProperFraction = 0.1

    @validator('learning_rate_decay_steps', each_item=True)
    def validate_learning_rate_decay_steps_below_epochs(cls, v, values):
        assert v < values['epochs'], "Each decay step must be lower than the total number of epoch."
        return v

    @validator('learning_rate_decay_steps')
    def validate_learning_rate_decay_steps_are_ascending(cls, v):
        assert all(v[i] <= v[i+1] for i in range(len(v)-1)), "Decay steps should be ascending."
        return v
