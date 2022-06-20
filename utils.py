import argparse
import math
import os
import sys
import warnings
import yaml
import torch
import torchvision

import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from functools import partial
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm
from typing import List, Optional, Callable, Tuple
from loguru import logger
from torch.utils.data import DataLoader

from consts import LOGGER_FORMAT


def log_args(args):
    """Logs the given arguments to the logger's output.
    """
    logger.info(f'Running with the following arguments:')
    longest_arg_name_length = max(len(k) for k in args.flattened_dict().keys())
    pad_length = longest_arg_name_length + 4
    for arg_name, value in args.flattened_dict().items():
        logger.info(f'{f"{arg_name} ":-<{pad_length}} {value}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Main script for running the experiments with arguments from the corresponding pydantic schema'
    )
    parser.add_argument('--yaml_path', help=f'(Optional) path to a YAML file with the arguments')
    return parser.parse_known_args()


def get_args(args_class):
    """Gets arguments as an instance of the given pydantic class,
    according to the argparse object (possibly including the yaml config).
    """
    known_args, unknown_args = parse_args()
    args_dict = None
    if known_args.yaml_path is not None:
        with open(known_args.yaml_path, 'r') as f:
            args_dict = yaml.load(f, Loader=yaml.FullLoader)

    if args_dict is None:  # This happens when the yaml file is empty, or no yaml file was given.
        args_dict = dict()

    while len(unknown_args) > 0:
        arg_name = unknown_args.pop(0).replace('--', '')
        values = list()
        while (len(unknown_args) > 0) and (not unknown_args[0].startswith('--')):
            values.append(unknown_args.pop(0))
        if len(values) == 0:
            raise ValueError(f'Argument {arg_name} given in command line has no corresponding value.')
        value = values[0] if len(values) == 1 else values

        categories = list(args_class.__fields__.keys())
        found = False
        for category in categories:
            category_args = list(args_class.__fields__[category].default.__fields__.keys())
            if arg_name in category_args:
                if category not in args_dict:
                    args_dict[category] = dict()
                args_dict[category][arg_name] = value
                found = True

        if not found:
            raise ValueError(f'Argument {arg_name} is not recognized.')

    args = args_class.parse_obj(args_dict)

    return args


def power_minus_1(a: torch.Tensor):
    """Raises the input tensor to the power of minus 1.
    """
    return torch.divide(torch.ones_like(a), a)


def get_mlp(input_dim: int, output_dim: int, n_hidden_layers: int = 0, hidden_dim: int = 0,
            use_batch_norm: bool = False, organize_as_blocks: bool = False) -> torch.nn.Sequential:
    """Create an MLP (i.e. Multi-Layer-Perceptron) and return it as a PyTorch's sequential model.

    Args:
        input_dim: The dimension of the input tensor.
        output_dim: The dimension of the output tensor.
        n_hidden_layers: Number of hidden layers.
        hidden_dim: The dimension of each hidden layer.
        use_batch_norm: Whether to use BatchNormalization after each layer or not.
        organize_as_blocks: Whether to organize the model as blocks of Linear->(BatchNorm)->ReLU.

    Returns:
        A sequential model which is the constructed MLP.
    """
    layers: List[torch.nn.Module] = list()

    for i in range(n_hidden_layers):
        current_layers: List[torch.nn.Module] = list()

        in_features = input_dim if i == 0 else hidden_dim
        out_features = hidden_dim

        # Begins with a `Flatten` layer. It's useful when the input is 4D from a conv layer, and harmless otherwise.
        if i == 0:
            current_layers.append(nn.Flatten())

        current_layers.append(torch.nn.Linear(in_features, out_features))
        if use_batch_norm:
            current_layers.append(torch.nn.BatchNorm1d(hidden_dim))
        current_layers.append(torch.nn.ReLU())

        if organize_as_blocks:
            block = torch.nn.Sequential(*current_layers)
            layers.append(block)
        else:
            layers.extend(current_layers)

    final_layer = torch.nn.Linear(in_features=input_dim if n_hidden_layers == 0 else hidden_dim,
                                  out_features=output_dim)
    if organize_as_blocks:
        final_layer = torch.nn.Sequential(final_layer)

    layers.append(final_layer)

    return torch.nn.Sequential(*layers)


@torch.no_grad()
def calc_aggregated_patch(dataloader,
                          patch_size,
                          agg_func: Callable,
                          existing_model: Optional[nn.Module] = None):
    """Calculate the aggregated patch across all patches in the dataloader.

    Args:
        dataloader: dataloader to iterate on.
        patch_size: The patch-size to feed into the aggregate function.
        agg_func: The aggregate function, which gets a single argument which is a NumPy array,
            and return a single argument which is a NumPy array
        existing_model: An (optionally) existing model to call on each image in the data.
    """
    total_size = 0
    mean = None
    device = get_model_device(existing_model)
    for inputs, _ in tqdm(dataloader, total=len(dataloader), desc='Calculating mean patch'):
        inputs = inputs.to(device)
        if existing_model is not None:
            inputs = existing_model(inputs)

        # Unfold the input batch to its patches - shape (N, C*H*W, M) where M is the number of patches per image.
        patches = F.unfold(inputs, patch_size)

        # Transpose to (N, M, C*H*W) and then reshape to (N*M, C*H*W) to have collection of vectors
        # Also make contiguous in memory
        patches = patches.transpose(1, 2).flatten(0, 1).contiguous().double()

        # Perform the aggregation function over the batch-size and number of patches per image.
        # For example, when calculating mean it'll a (C*H*W)-dimensional vector,
        # and when calculating the covariance it will be a square matrix of shape (C*H*W, C*H*W)
        aggregated_patch = agg_func(patches)

        if mean is None:
            mean = torch.zeros_like(aggregated_patch)

        batch_size = inputs.size(0)
        mean = ((total_size / (total_size + batch_size)) * mean +
                (batch_size / (total_size + batch_size)) * aggregated_patch)

        total_size += batch_size

    return mean


def calc_covariance(data, mean=None):
    """Calculates the covariance-matrix of the given data.

    This function assumes the data matrix is ordered as rows-vectors
    (i.e. shape (n,d) so n data-points in d dimensions).

    Args:
        data: The given data, a 2-dimensional NumPy array ordered as rows-vectors
            (i.e. shape (n,d) so n data-points in d dimensions).
        mean: The mean of the data, if not given the mean will be calculated.
            It's useful when the mean is the mean of some larger distribution, and not only the mean of the
            given data array (as done when calculating the covariance matrix of the whole patches distribution).

    Returns:
        The covariance-matrix of the given data.
    """
    if mean is None:
        mean = data.mean(axis=0)
    centered_data = data - mean
    return (1 / data.shape[0]) * (centered_data.T @ centered_data)


def calc_whitening_from_dataloader(dataloader: DataLoader,
                                   patch_size: int,
                                   whitening_regularization_factor: float,
                                   zca_whitening: bool = False,
                                   existing_model: Optional[nn.Module] = None) -> np.ndarray:
    """Calculates the whitening matrix from the given data.

    Denote the data matrix by X (i.e. collection of patches) with shape N x D.
    N is the number of patches, and D is the dimension of each patch (channels * spatial_size ** 2).
    This function returns the whitening operator as a columns-vectors matrix of shape D x D,
    so it needs to be multiplied by the target data matrix X' of shape N' x D from the right (X' @ W)
    [and NOT from the left, i.e. NOT W @ X'].

    Args:
        dataloader: The given data to iterate on.
        patch_size: The size of the patches to calculate the whitening on.
        whitening_regularization_factor: The regularization factor used when calculating the whitening,
            which is some small constant positive float added to the denominator.
        zca_whitening: Whether it's ZCA whitening (or PCA whitening).
        existing_model: An (optionally) existing model to call on each image in the data.

    Returns:
        The whitening matrix.
    """
    logger.debug('Performing a first pass over the dataset to calculate the mean patch...')
    mean_patch = calc_aggregated_patch(dataloader, patch_size, agg_func=partial(torch.mean, dim=0),
                                       existing_model=existing_model)

    logger.debug('Performing a second pass over the dataset to calculate the covariance...')
    covariance_matrix = calc_aggregated_patch(dataloader, patch_size,
                                              agg_func=partial(calc_covariance, mean=mean_patch),
                                              existing_model=existing_model)

    logger.debug('Calculating eigenvalues decomposition to get the whitening matrix...')
    whitening_matrix = get_whitening_matrix_from_covariance_matrix(
        covariance_matrix.cpu(), whitening_regularization_factor, zca_whitening
    )

    logger.debug('Done.')
    return whitening_matrix


def configure_logger(out_dir: str, level='INFO', print_sink=sys.stdout):
    """
    Configure the logger:
    (1) Remove the default logger (to stdout) and use a one with a custom format.
    (2) Adds a log file named `run.log` in the given output directory.
    """
    logger.remove()
    logger.remove()
    logger.add(sink=print_sink, format=LOGGER_FORMAT, level=level)
    logger.add(sink=os.path.join(out_dir, 'run.log'), format=LOGGER_FORMAT, level=level)


def get_dataloaders(batch_size: int = 64,
                    normalize_to_unit_gaussian: bool = False,
                    normalize_to_plus_minus_one: bool = False,
                    random_crop: bool = False,
                    random_horizontal_flip: bool = False,
                    random_erasing: bool = False,
                    random_resized_crop: bool = False):
    """Gets dataloaders for the CIFAR10 dataset, including data augmentations as requested by the arguments.

    Args:
        batch_size: The size of the mini-batches to initialize the dataloaders.
        normalize_to_unit_gaussian: If true, normalize the values to be a unit gaussian.
        normalize_to_plus_minus_one: If true, normalize the values to be in the range [-1,1] (instead of [0,1]).
        random_crop: If true, performs padding of 4 followed by random crop.
        random_horizontal_flip: If true, performs random horizontal flip.
        random_erasing: If true, erase a random rectangle in the image. See https://arxiv.org/pdf/1708.04896.pdf.
        random_resized_crop: If true, performs random resized crop.

    Returns:
        A dictionary mapping "train"/"test" to its dataloader.
    """
    transforms = {'train': list(), 'test': list()}

    if random_horizontal_flip:
        transforms['train'].append(torchvision.transforms.RandomHorizontalFlip())
    if random_crop:
        transforms['train'].append(torchvision.transforms.RandomCrop(size=32, padding=4))
    if random_resized_crop:
        transforms['train'].append(torchvision.transforms.RandomResizedCrop(size=32, scale=(0.75, 1.), ratio=(1., 1.)))
    for t in ['train', 'test']:
        transforms[t].append(torchvision.transforms.ToTensor())
    if random_erasing:
        transforms['train'].append(torchvision.transforms.RandomErasing())
    if normalize_to_plus_minus_one or normalize_to_unit_gaussian:
        # For the different normalization values see:
        # https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457/7
        if normalize_to_unit_gaussian:
            # These normalization values are taken from https://github.com/kuangliu/pytorch-cifar/issues/19
            # normalization_values = [(0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)]

            # These normalization values are taken from https://github.com/louity/patches
            # and also https://stackoverflow.com/questions/50710493/cifar-10-meaningless-normalization-values
            normalization_values = [(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)]
        else:
            normalization_values = [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)]
        for t in ['train', 'test']:
            transforms[t].append(torchvision.transforms.Normalize(*normalization_values))

    datasets = {t: torchvision.datasets.CIFAR10(root='./data',
                                                train=(t == 'train'),
                                                transform=torchvision.transforms.Compose(transforms[t]),
                                                download=False)
                for t in ['train', 'test']}

    dataloaders = {t: torch.utils.data.DataLoader(datasets[t],
                                                  batch_size=batch_size,
                                                  shuffle=(t == 'train'),
                                                  num_workers=2)
                   for t in ['train', 'test']}

    return dataloaders


def get_model_device(model: Optional[torch.nn.Module]):
    """Returns the device of the given model
    """
    default_device = torch.device('cpu')

    # If the model is None, assume the model's device is CPU.
    if model is None:
        return default_device

    try:
        device = next(model.parameters()).device
    except StopIteration:  # If the model has no parameters, assume the model's device is CPU.
        device = default_device

    return device


@torch.no_grad()
def get_model_output_shape(model: nn.Module, dataloader: Optional[DataLoader] = None):
    """Gets the output shape of the given model, on images from the given dataloader.
    """
    if dataloader is None:
        clean_dataloaders = get_dataloaders(batch_size=1)
        dataloader = clean_dataloaders["train"]

    inputs, _ = next(iter(dataloader))
    inputs = inputs.to(get_model_device(model))
    outputs = model(inputs)
    outputs = outputs.cpu().numpy()
    return outputs.shape[1:]  # Remove the first dimension corresponding to the batch


@torch.no_grad()
def sample_random_patches(dataloader,
                          n_patches,
                          patch_size,
                          existing_model: Optional[nn.Module] = None,
                          visualize: bool = False,
                          random_uniform_patches: bool = False,
                          random_gaussian_patches: bool = False,
                          verbose: bool = False):
    """Sample random patches from the data.

    Args:
        dataloader: The dataloader to sample patches from.
        n_patches: Number of patches to sample.
        patch_size: The size of the patches to sample.
        existing_model: (Possibly) an existing model to transform each image from the dataloader.
        visualize: Whether to visualize the sampled patches (for debugging purposes).
        random_uniform_patches: Whether to avoid sampling and simply return patches from the uniform distribution.
        random_gaussian_patches: Whether to avoid sampling and simply return patches from the Gaussian distribution.
        verbose: Whether to print progress using tqdm.

    Returns:
        The sampled patches as a NumPy array.
    """
    batch_size = dataloader.batch_size
    n_images = len(dataloader.dataset)

    # We need the shape of the images in the data.
    # In relatively small datasets (CIFAR, MNIST) the data itself is stored in `dataloader.dataset.data`
    # and in ImageNet it's not the case since the data is too large.
    # This is why the shape of ImageNet images is hard-coded.
    images_shape = dataloader.dataset.data.shape[1:] if hasattr(dataloader.dataset, 'data') else (224, 224, 3)

    if len(images_shape) == 2:  # When the dataset contains grayscale images, 
        images_shape += (1,)    # add dimension of channels which will be 1.

    images_shape = np.roll(images_shape, shift=1)  # In the dataset it's H x W x C but in the model it's C x H x W
    if existing_model is not None:
        device = get_model_device(existing_model)
        images_shape = get_model_output_shape(existing_model, dataloader)

    if len(images_shape) > 1:
        assert len(images_shape) == 3 and (images_shape[1] == images_shape[2]), "Should be C x H x W where H = W"
        spatial_size = images_shape[-1]
        if patch_size == -1:  # -1 means the patch size is the whole size of the image.
            patch_size = spatial_size
        n_patches_per_row_or_col = spatial_size - patch_size + 1
        patch_shape = (images_shape[0],) + 2 * (patch_size,)
    else:
        assert patch_size == -1, "When working with fully-connected the patch 'size' must be -1 i.e. the whole size."
        n_patches_per_row_or_col = 1
        patch_shape = images_shape

    n_patches_per_image = n_patches_per_row_or_col ** 2
    n_patches_in_dataset = n_images * n_patches_per_image

    if n_patches >= n_patches_in_dataset:
        n_patches = n_patches_in_dataset

    patches_indices_in_dataset = np.random.default_rng().choice(n_patches_in_dataset, size=n_patches, replace=False)

    images_indices = patches_indices_in_dataset % n_images
    patches_indices_in_images = patches_indices_in_dataset // n_images
    patches_x_indices_in_images = patches_indices_in_images % n_patches_per_row_or_col
    patches_y_indices_in_images = patches_indices_in_images // n_patches_per_row_or_col

    batches_indices = images_indices // batch_size
    images_indices_in_batches = images_indices % batch_size

    patches = np.empty(shape=(n_patches,) + patch_shape, dtype=np.float32)

    if random_uniform_patches:
        return np.random.default_rng().uniform(low=-1, high=+1, size=patches.shape).astype(np.float32)
    if random_gaussian_patches:
        patch_dim = math.prod(patch_shape)
        return np.random.default_rng().multivariate_normal(
            mean=np.zeros(patch_dim), cov=np.eye(patch_dim), size=n_patches).astype(np.float32).reshape(patches.shape)

    iterator = enumerate(dataloader)
    if verbose:
        iterator = tqdm(iterator, total=len(dataloader), desc='Sampling patches from the dataset')
    for batch_index, (inputs, _) in iterator:
        if batch_index not in batches_indices:
            continue

        relevant_patches_mask = (batch_index == batches_indices)
        relevant_patches_indices = np.where(relevant_patches_mask)[0]

        if existing_model is not None:
            inputs = inputs.to(device)
            inputs = existing_model(inputs)
        inputs = inputs.cpu().numpy()

        for i in relevant_patches_indices:
            image_index_in_batch = images_indices_in_batches[i]
            if len(patch_shape) > 1:
                patch_x_start = patches_x_indices_in_images[i]
                patch_y_start = patches_y_indices_in_images[i]
                patch_x_slice = slice(patch_x_start, patch_x_start + patch_size)
                patch_y_slice = slice(patch_y_start, patch_y_start + patch_size)

                patches[i] = inputs[image_index_in_batch, :, patch_x_slice, patch_y_slice]

                if visualize:
                    visualize_image_patch_pair(image=inputs[image_index_in_batch], patch=patches[i],
                                               patch_x_start=patch_x_start, patch_y_start=patch_y_start)
            else:
                patches[i] = inputs[image_index_in_batch]

    return patches


def visualize_image_patch_pair(image, patch, patch_x_start, patch_y_start):
    """Visualize the given image and the patch in it, with rectangle in the location of the patch.
    """
    patch_size = patch.shape[-1]
    rect = Rectangle(xy=(patch_y_start, patch_x_start),  # x and y are reversed on purpose...
                     width=patch_size, height=patch_size,
                     linewidth=1, edgecolor='red', facecolor='none')

    plt.figure()
    ax = plt.subplot(2, 1, 1)
    ax.imshow(np.transpose(image, axes=(1, 2, 0)))
    ax.add_patch(rect)
    ax = plt.subplot(2, 1, 2)
    ax.imshow(np.transpose(patch, axes=(1, 2, 0)))
    plt.show()


def get_whitening_matrix_from_covariance_matrix(covariance_matrix: np.ndarray,
                                                whitening_regularization_factor: float,
                                                zca_whitening: bool = False) -> np.ndarray:
    """Calculates the whitening matrix from the given covariance matrix.

    Args:
        covariance_matrix: The covariance matrix.
        whitening_regularization_factor: The regularization factor used when calculating the whitening,
            which is some small constant positive float added to the denominator.
        zca_whitening: Whether it's ZCA whitening (or PCA whitening).

    Returns:
        The whitening matrix.
    """
    eigenvectors, eigenvalues, eigenvectors_transposed = np.linalg.svd(covariance_matrix, hermitian=True)
    inv_sqrt_eigenvalues = np.diag(1. / (np.sqrt(eigenvalues) + whitening_regularization_factor))
    whitening_matrix = eigenvectors.dot(inv_sqrt_eigenvalues)
    if zca_whitening:
        whitening_matrix = whitening_matrix @ eigenvectors.T
    whitening_matrix = whitening_matrix.astype(np.float32)
    return whitening_matrix


def whiten_data(data, whitening_regularization_factor=1e-05, zca_whitening=False):
    """Whiten the given data.

    Note that the data is assumed to be of shape (n_samples, n_features), meaning it's a collection of row-vectors.

    Args:
        data: The given data to whiten.
        whitening_regularization_factor: The regularization factor used when calculating the whitening,
            which is some small constant positive float added to the denominator.
        zca_whitening: Whether it's ZCA whitening (or PCA whitening).

    Returns:
        The whitened data.
    """
    covariance_matrix = calc_covariance(data)
    whitening_matrix = get_whitening_matrix_from_covariance_matrix(covariance_matrix,
                                                                   whitening_regularization_factor,
                                                                   zca_whitening)
    centered_data = data - data.mean(axis=0)
    whitened_data = centered_data @ whitening_matrix
    return whitened_data


def normalize_data(data, epsilon=1e-05):
    """Normalize the given data (making it centered (zero mean) and each feature have unit variance).

    Note that the data is assumed to be of shape (n_samples, n_features), meaning it's a collection of row-vectors.

    Args:
        data: The data to normalize.
        epsilon: Some small positive number to add to the denominator,
            to avoid getting NANs (if the data-point has a small std).

    Returns:
        The normalized data.
    """
    centered_data = data - data.mean(axis=0)
    normalized_data = centered_data / (centered_data.std(axis=0) + epsilon)
    return normalized_data


def get_random_initialized_conv_kernel_and_bias(in_channels: int,
                                                out_channels: int,
                                                kernel_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Returns randomly initialized kernel and bias for a conv layer, as in PyTorch default initialization (Xavier).

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: The kernel size.

    Returns:
        A tuple of two numpy arrays which are the randomly initialized kernel and bias for a conv layer,
        as in PyTorch default initialization (Xavier).
    """
    tmp_conv = nn.Conv2d(in_channels, out_channels, kernel_size)
    kernel = tmp_conv.weight.data.cpu().numpy().copy()
    bias = tmp_conv.bias.data.cpu().numpy().copy()
    return kernel, bias


def run_kmeans_clustering(data: np.ndarray, k: int, try_to_use_faiss: bool):
    """Runs k-means clustering on the given data. Returns the centroids, the assignments of data-points to centroids,
    and the distances between each data-point to its assigned centroid.

    Args:
        data: ndarray of shape (n_samples, n_features)
            The data to cluster.
        k: int
            The number of clusters to form as well as the number of centroids to generate.
        try_to_use_faiss: boolean, default=False
            Whether to use faiss library for faster run-time (requires faiss library installed).

    Returns:
        centroids: ndarray of shape (n_clusters, n_features)
            The clusters centers.
        indices: ndarray of shape (n_samples,)
            Labels of each point (i.e., the index of the closest centroid).
        distances: ndarray of shape (n_samples,)
            The distance of each data-point to its closest centroid.
    """
    if try_to_use_faiss:
        try:
            from faiss import Kmeans as KMeans
            kmeans = KMeans(d=data.shape[1], k=k)
            kmeans.train(data)
            centroids = kmeans.centroids
            distances, indices = kmeans.assign(data)
            return centroids, distances, indices
        except ImportError:
            warnings.warn(f'use_faiss is True, but failed to import faiss. Using sklearn instead.')
            from sklearn.cluster import KMeans

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=k)
    distances_to_each_centroid = kmeans.fit_transform(data)
    indices = kmeans.labels_
    distances = np.take_along_axis(distances_to_each_centroid, indices[:, np.newaxis], axis=1).squeeze(axis=1)
    centroids = kmeans.cluster_centers_
    return centroids, indices, distances
