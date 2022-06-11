from copy import deepcopy
from math import ceil
from typing import List, Optional, Union, get_origin

from pydantic import root_validator
from pydantic.types import PositiveInt
from pydantic.fields import ModelField

from schemas.architecture import ArchitectureArgs
from schemas.data import DataArgs
from schemas.environment import EnvironmentArgs
from schemas.optimization import OptimizationArgs
from schemas.utils import ImmutableArgs, MyBaseModel, ProperFraction, NonNegativeFloat, NonNegativeInt


class PatchesArgs(ImmutableArgs):

    #: Whether to train a locally linear network.
    train_locally_linear_network: bool = False

    #: Number of patches to uniformly sample (which might get clustered later).
    n_patches: Union[PositiveInt, List[PositiveInt]] = 262144

    #: How many clusters to have in the final patches' dictionary.
    n_clusters: Union[PositiveInt, List[PositiveInt]] = 1024

    #: If it's true, the patches will NOT be taken from the dataset,
    #: they will be uniformly sampled from [-1,+1]
    random_uniform_patches: Union[bool, List[bool]] = False

    #: If it's true, the patches will NOT be taken from the dataset,
    #: they will be uniformly sampled from [-1,+1]
    random_gaussian_patches: Union[bool, List[bool]] = False

    #: The k-th nearest-neighbor will be used for the k-NN imitator, or in the locally linear model.
    k: Union[PositiveInt, List[PositiveInt]] = 256

    #: If it's true, when calculating the k-nearest-neighbors mask, there will be ones in the indices of the neighbors
    #: 1, 2, ..., k. If it's false, there will be a single one in the index of the k-th nearest-neighbor.
    up_to_k: Union[bool, List[bool]] = True

    #: Use convolution layer multiplied by the patch-based embedding.
    #: If it's false the resulting model is the same as Thiry et al.
    use_conv: Union[bool, List[bool]] = True

    #: Initialize the patches dictionary randomly from the same random distribution as PyTorch default for Conv2D.
    random_embedding: Union[bool, List[bool]] = False

    #: Taken from page 4 from Coates et al (2011)
    #: An Analysis of Single-Layer Networks in Unsupervised Feature Learning
    #: https://cs.stanford.edu/~acoates/papers/coatesleeng_aistats_2011.pdf
    #: This activation function outputs 0 for any patch where the distance to the centroid  is “above average”.
    #: In practice, this means that roughly half of the features will be set to 0.
    kmeans_triangle: Union[bool, List[bool]] = False

    #: If it's true, the embedding with be replaced with a Conv->ReLU
    replace_embedding_with_regular_conv_relu: Union[bool, List[bool]] = False

    #: If it's true, the embedding will have gradients and will change during training.
    learnable_embedding: Union[bool, List[bool]] = False

    #: The regularization factor (a.k.a. lambda) of the whitening matrix.
    use_whitening: Union[bool, List[bool]] = True

    #: If it's false, use PCA-whitening. Otherwise, use ZCA whitening (which is a rotation of the PCA-whitening).
    zca_whitening: Union[bool, List[bool]] = False

    #: If it's true, calculates whitening from the sampled patches, and not from all patches in the dataset.
    calc_whitening_from_sampled_patches: Union[bool, List[bool]] = False

    #: The regularization factor (a.k.a. lambda) of the whitening matrix.
    whitening_regularization_factor: Union[NonNegativeFloat, List[NonNegativeFloat]] = 0.001

    #: Whether rto normalize the patches to unit vectors (divide by its l2 norm). 
    #: This cause the metric between patches to be minimal-angle instead of euclidean-distance.
    normalize_patches_to_unit_vectors: Union[bool, List[bool]] = False

    #: If it's true, sample the patches to obtain the patches-dictionary from the original
    #: pixel values which are in [0,1]. If it's false, sample the patches after the normalization 
    #: to [-1,+1] (or after subtracting the mean and dividing by the standard deviation).
    sample_patches_from_original_zero_one_values: Union[bool, List[bool]] = True

    #: Whether to use avg-pool after the embedding.
    use_avg_pool: Union[bool, List[bool]] = True
    #: Whether to use adaptive-avg-pool after the avg-pool.
    use_adaptive_avg_pool: Union[bool, List[bool]] = False
    adaptive_pool_output_size: Union[PositiveInt, List[PositiveInt]] = 6

    #: Whether to decompose the final linear layer into 1x1 convolution followed by an actual linear layer.
    use_bottle_neck: Union[bool, List[bool]] = True
    use_relu_after_bottleneck: Union[bool, List[bool]] = False

    #: If depth > 1 it means that we build model on top of other models.
    depth: int = 1

    #: If it's true, when building deep models (i.e., depth > 1) the logits of all of the different modules 
    #: will be summed up, so the last module (that is being trained, since all other modules are frozen) 
    #: will have to predict the "residual" predictions.
    sum_logits: bool = False

    #: Whether the input embedding to the deep model should have a residual connection (addition / concatenation).
    residual_add: Union[bool, List[bool]] = False
    residual_cat: Union[bool, List[bool]] = False

    #: Number of values to output for each patch's linear-classifier (used in the other constrained linear
    # model experimented but not added to the final paper).
    c: Union[PositiveInt, List[PositiveInt]] = 1

    #: Whether to use the Phi_full embedding without any constraint (see Section 5.3 in the paper).
    full_embedding: Union[bool, List[bool]] = False

    def extract_single_depth_args(self, i: int):
        assert i < self.depth, f'i is {i} and self.depth is {self.depth}, i should be smaller.'
        self_i = deepcopy(self)
        fields = [k for k, v in PatchesArgs.__fields__.items() if get_origin(v.type_) is Union]
        for field in fields:
            setattr(self_i, field, getattr(self, field)[i])
        
        self_i.depth = 1  # To make the validators of the pydantic class not crash.
        return self_i

    @root_validator
    def set_args_for_deep_locally_linear_networks(cls, values):
        depth = values['depth']
        if depth > 1:
            fields = [k for k, v in cls.__fields__.items() if get_origin(v.type_) is Union]
            for field in fields:
                if isinstance(values[field], list):
                    assert len(values[field]) == depth, \
                        f'Given a list of length={len(values[field])} for field {field} but depth={depth}'
                else:
                    values[field] = [values[field]] * depth
        return values

    @root_validator
    def validate_replace_embedding_with_regular_conv_relu(cls, values):
        k1 = 'replace_embedding_with_regular_conv_relu'
        k2 = 'use_conv'
        if isinstance(values[k1], list) or isinstance(values[k2], list):
            for i in range(len(values[k1])):
                assert not (values[k1][i] and values[k2][i]), f"Don't use both {k1} and {k2}"
        else:    
            assert not (values[k1] and values[k2]), f"Don't use both {k1} and {k2}"
        return values

    @root_validator
    def validate_c_larger_than_1_requires_conv(cls, values):
        if isinstance(values['c'], list):
            for i in range(len(values['c'])):
                if values['c'][i] > 1:
                    assert values['use_conv'][i], "It doesn't make sense to use c > 1 without a conv layer."
        else:
            if values['c'] > 1:
                assert values['use_conv'], "It doesn't make sense to use c > 1 without a conv layer."
        return values
    
    @root_validator
    def validate_full_embedding(cls, values):
        if isinstance(values['full_embedding'], list):
            for i in range(len(values['full_embedding'])):
                if values['full_embedding'][i]:
                    for k in ['use_conv', 'use_avg_pool', 'use_adaptive_avg_pool', 'use_bottle_neck',
                              'use_relu_after_bottleneck', 'residual_add', 'residual_cat']:
                        assert not values[k][i], f"Can't use full embedding and {k}"
                    assert values['c'][i] == 1, "Can't use full embedding and c > 1"
        else:
            if values['full_embedding']:
                for k in ['use_conv', 'use_avg_pool', 'use_adaptive_avg_pool', 'use_bottle_neck',
                          'use_relu_after_bottleneck', 'residual_add', 'residual_cat']:
                    assert not values[k], f"Can't use full embedding and {k}"
                assert values['c'] == 1, "Can't use full embedding and c > 1"
        return values
        

class Args(MyBaseModel):
    opt: OptimizationArgs = OptimizationArgs()
    arch: ArchitectureArgs = ArchitectureArgs()
    env: EnvironmentArgs = EnvironmentArgs()
    data: DataArgs = DataArgs()
    patches: PatchesArgs = PatchesArgs()

    def extract_single_depth_args(self, i: int):
        assert i < self.patches.depth, f'i is {i} and depth is {self.patches.depth}, i should be smaller.'
        new_self = deepcopy(self)

        fields = Args.__fields__
        categories = list(fields.keys())
        for category in categories:
            new_category = getattr(new_self, category)
            old_category = getattr(self, category)

            category_fields = fields[category].default.__fields__
            for arg_name, arg_field in category_fields.items():
                if (get_origin(arg_field.type_) is Union) and (isinstance(getattr(old_category, arg_name), list)):
                    setattr(new_category, arg_name, getattr(old_category, arg_name)[i])
            
            setattr(new_self, category, new_category)
        
        new_self.patches.depth = 1  # To make the validators of the pydantic class not crash.
        return new_self
