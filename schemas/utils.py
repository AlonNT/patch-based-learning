import flatten_dict
from pydantic import BaseModel, Extra
from pydantic.types import ConstrainedFloat, ConstrainedInt


class NonNegativeInt(ConstrainedInt):
    ge = 0


class NonNegativeFloat(ConstrainedFloat):
    ge = 0


class Fraction(ConstrainedFloat):
    ge = 0
    le = 1


class ProperFraction(ConstrainedFloat):
    gt = 0
    lt = 1


class NonZeroFraction(ConstrainedFloat):
    gt = 0
    le = 1


class NonOneFraction(ConstrainedFloat):
    ge = 0
    lt = 1


class ImmutableArgs(BaseModel):
    class Config:
        allow_mutation = True
        extra = Extra.forbid


class MyBaseModel(BaseModel):
    def flattened_dict(self):
        """
        Returns the arguments as a flattened dictionary, without the category name (i.e. opt, arch, env, data).
        It's assumed that there is no field with the same name among different categories.
        """
        return {k[1]: v for k, v in flatten_dict.flatten(self.dict()).items()}
