import torch.nn as nn
from typing import List, Union, Tuple


# Configurations for the VGG models family.
# A number indicates number of channels in a convolution block, and M/A denotes a MaxPool/AvgPool layer.
configs = {
    'VGGc16d1': [16],
    'VGGc32d1': [32],
    'VGGc64d1': [64],
    'VGGc128d1': [128],
    'VGGc256d1': [256],
    'VGGc512d1': [512],
    'VGGc1024d1': [1024],
    'VGGc1024d1A': [1024, 'A'],

    'VGGc16d2': [16, 16],
    'VGGc32d2': [32, 32],
    'VGGc64d2': [64, 64],
    'VGGc128d2': [128, 128],
    'VGGc256d2': [256, 256],
    'VGGc512d2': [512, 512],
    'VGGc1024d2': [1024, 1024],
    'VGGc1024d2A': [1024, 'A', 1024],

    'VGGc16d3': [16, 16, 16],
    'VGGc32d3': [32, 32, 32],
    'VGGc64d3': [64, 64, 64],
    'VGGc128d3': [128, 128, 128],
    'VGGc256d3': [256, 256, 256],
    'VGGc512d3': [512, 512, 512],
    'VGGc1024d3': [1024, 1024, 1024],
    'VGGc1024d3A': [1024, 'A', 1024, 1024],

    'VGGc16d4': [16, 16, 16, 16],
    'VGGc32d4': [32, 32, 32, 32],
    'VGGc64d4': [64, 64, 64, 64],
    'VGGc128d4': [128, 128, 128, 128],
    'VGGc256d4': [256, 256, 256, 256],
    'VGGc512d4': [512, 512, 512, 512],
    'VGGc1024d4': [1024, 1024, 1024, 1024],
    'VGGc1024d4A': [1024, 'A', 1024, 1024, 1024],

    'VGGs': [8, 'M', 16, 'M', 32, 'M'],  # 's' for shallow.
    'VGGsw': [64, 'M', 128, 'M', 256, 'M'],  # 'w' for wide.
    'VGGsxw': [128, 'M', 256, 'M', 512, 'M'],  # 'x' for extra.

    'VGG8c': [64, 128, 'M', 256, 256, 'M', 512, 512, 'M'],

    # Models taken from the paper "Training Neural Networks with Local Error Signals"
    # https://github.com/anokland/local-loss/blob/master/train.py#L1276
    'VGG8b': [128, 256, 'M', 256, 512, 'M', 512, 'M', 512, 'M'],
    'VGG11b': [128, 128, 128, 256, 'M', 256, 512, 'M', 512, 512, 'M', 512, 'M'],

    # These are versions similar to the original VGG models but with less down-sampling,
    # reaching final spatial size of 4x4 instead of 1x1 in the original VGG architectures.
    # 'c' stands for CIFAR, i.e. models that are suited to CIFAR instead of ImageNet.
    'VGG11c': [64, 128, 'M', 256, 256, 'M', 512, 512, 512, 512, 'M'],
    'VGG13c': [64, 64, 128, 128, 'M', 256, 256, 'M', 512, 512, 512, 512, 'M'],
    'VGG16c': [64, 64, 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 512, 512, 512, 'M'],
    'VGG19c': [64, 64, 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 512, 512, 512, 512, 'M'],

    # Original VGG architectures (built for ImageNet images of size 224x224)
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def get_list_of_arguments_for_config(config: List[Union[int, str]], arg) -> list:
    """Given an argument `arg`, returns a list of arguments for a given config of a VGG model,
    where this argument is repeated for conv blocks (and None in the positions of non-conv blocks).
    """
    if isinstance(arg, list):
        non_conv_indices = [i for i, l in enumerate(config) if isinstance(l, str)]
        for non_conv_index in non_conv_indices:
            arg.insert(non_conv_index, None)
    else:
        arg = [(arg if isinstance(l, int) else None) for l in range(len(config))]
    
    return arg


def get_vgg_blocks(config: List[Union[int, str]],
                   in_channels: int = 3,
                   spatial_size: int = 32,
                   kernel_size: Union[int, List[int]] = 3,
                   padding: Union[int, List[int]] = 1,
                   use_batch_norm: Union[bool, List[bool]] = False,
                   bottleneck_dim: Union[int, List[int]] = 0) -> Tuple[List[nn.Module], int]:
    """Gets a list containing the blocks of the given VGG model config.

    Args:
        config: One of the lists in the dictionary `configs` above, describing the architecture of the network.
        in_channels: Number of input channels (3 for RGB images).
        spatial_size: The size of the input tensor for the network (32 for CIFAR10, 28 for MNIST, etc).
        kernel_size: The kernel size to use in each conv block. If it's a single variable, the same one is used.
        padding: The amount of padding to use in each conv block. If it's a single variable, the same one is used.
        use_batch_norm: Whether to use batch-norm in each conv block. If it's a single variable, the same one is used.
        bottleneck_dim: The dimension of the bottleneck layer to use in the end of each conv block
            (0 means no bottleneck is added). If it's a single variable, the same one is used.

    Returns:
        A tuple containing the list of nn.Modules, and an integers which is the number of input features
        (will be useful later when feeding to a linear layer).
    """
    blocks: List[nn.Module] = list()
    
    kernel_size = get_list_of_arguments_for_config(config, kernel_size)
    padding = get_list_of_arguments_for_config(config, padding)
    use_batch_norm = get_list_of_arguments_for_config(config, use_batch_norm)
    bottleneck_dim = get_list_of_arguments_for_config(config, bottleneck_dim)

    for i in range(len(config)):
        if config[i] == 'M':
            blocks.append(nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2)))
        elif config[i] == 'A':
            continue  # The average-pooling was already added in the else section...
        else:
            out_channels = config[i]

            block_layers = [nn.Conv2d(in_channels, out_channels, kernel_size[i], padding=padding[i]),
                            nn.ReLU()]

            if (i+1 < len(config)) and (config[i+1] == 'A'):
                block_layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            
            if use_batch_norm[i]:
                block_layers.append(nn.BatchNorm2d(out_channels))

            if bottleneck_dim[i] > 0:
                block_layers.append(nn.Conv2d(out_channels, bottleneck_dim[i], kernel_size=1))
                out_channels = bottleneck_dim[i]
            
            blocks.append(nn.Sequential(*block_layers))
            
            spatial_size = spatial_size + 2*padding[i] - kernel_size[i] + 1
            if (i+1 < len(config)) and (config[i+1] == 'A'):
                spatial_size /= 2
            in_channels = out_channels  # The input channels of the next convolution layer.

    n_features = int(out_channels * (spatial_size ** 2))
    return blocks, n_features


def get_vgg_model_kernel_size(model, block_index: int):
    """Get the kernel-size for a specific block in the given model.

    Args:
        model: A VGG model to query.
        block_index: THe index of the block to query.

    Returns:
        The kernel-size for block number `block_index` in `model`.
    """
    if not (0 <= block_index < len(model.features)):
        raise IndexError(f"block_index {block_index} is out-of-bounds (len={len(model.features)})")

    block = model.features[block_index]

    if isinstance(block, nn.MaxPool2d):
        pool_layer = block
        return pool_layer.kernel_size

    if not isinstance(block, nn.Sequential):
        raise ValueError(f"block_index {block_index} is not a sequential module (i.e. \'block\'), it's {type(block)}.")

    conv_layer = block[0]

    if not isinstance(conv_layer, nn.Conv2d):
        raise ValueError(f"first layer of the block is not a conv layer, it's {type(conv_layer)}")

    return conv_layer.kernel_size
