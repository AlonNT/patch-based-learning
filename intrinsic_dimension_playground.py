import os
import torchvision
import wandb
import tikzplotlib

import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

from tqdm import tqdm
from loguru import logger
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Resize

from schemas.data import DataArgs
from utils import sample_random_patches, run_kmeans_clustering
from main import DataModule
from schemas.intrinsic_dimension_playground import Args
from utils import get_args, configure_logger, log_args, calc_whitening_from_dataloader


def create_data_dict(patches: np.ndarray, whitening_matrix: np.ndarray):
    """Creates a dictionary containing the different datasets (original, whitened, whitened_shuffled, shuffled, random).

    Args:
        patches: The "data" - a NumPy array of shape (n, d) containing flattened patches.
        whitening_matrix: The whitening matrix defining the whitening operator.

    Returns:
        The dictionary containing the different datasets.
    """
    assert patches.ndim == 2, f'patches has shape {patches.shape}, it should have been flattened.'
    n_patches, d = patches.shape
    rng = np.random.default_rng()
    patches_shuffled = rng.permuted(patches, axis=1)
    patches_whitened = patches @ whitening_matrix
    patches_whitened_shuffled = rng.permuted(patches_whitened, axis=1)
    random_data = rng.multivariate_normal(mean=np.zeros(d), cov=np.eye(d), size=n_patches)

    data_dict = {
        'original': patches,
        'whitened': patches_whitened,
        'whitened_shuffled': patches_whitened_shuffled,
        'shuffled': patches_shuffled,
        'random': random_data
    }

    data_dict = {data_name: data.astype(np.float32) for data_name, data in data_dict.items()}

    return data_dict


def normalize_data(data_dict):
    """Normalize each data in the given data dictionary to be a unit vector.

    Note that the data-points with extremely low norms (percentile 0.1%) are filtered out,
    since their norms cause numeric issues (they can even be actually zero so we can divide by it later).

    Args:
        data_dict: A dictionary containing datasets, each is a NumPy array of shape (n, d).

    Returns:
        A dictionary where each dataset is normalized.
    """
    norms_dict = {data_name: np.linalg.norm(data, axis=1)
                  for data_name, data in data_dict.items()}

    low_norms_masks_dict = {data_name: (norms < np.percentile(norms, q=1))
                            for data_name, norms in norms_dict.items()}

    filtered_data_dict = {data_name: data[np.logical_not(low_norms_masks_dict[data_name])]
                          for data_name, data in data_dict.items()}

    filtered_norms_dict = {data_name: norms[np.logical_not(low_norms_masks_dict[data_name])]
                           for data_name, norms in norms_dict.items()}

    normalized_data_dict = {data_name: (filtered_data_dict[data_name] / filtered_norms_dict[data_name][:, np.newaxis])
                            for data_name in data_dict.keys()}

    return normalized_data_dict


def get_dataloader(data_args: DataArgs):
    """Get the relevant dataloader to operate on.

    Args:
        data_args: The data's arguments schema.
    """
    # Relatively small datasets can be used with the LightningDataModule class used in the training-process.
    if data_args.dataset_name in ['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100']:
        datamodule = DataModule(data_args, batch_size=128)
        datamodule.prepare_data()
        datamodule.setup(stage='validate')
        dataloader = datamodule.val_dataloader_clean()

    # Since ImageNet is a quite large dataset, it's being read differently.
    # Furthermore, each image needs to be resized to 224x224 (images in the validation dataset are not always 224x224).
    elif data_args.dataset_name == 'ImageNet':
        dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_args.data_dir, 'val'),
                                                   transform=Compose([Resize((224, 224)), ToTensor()]))
        dataloader = DataLoader(dataset, batch_size=128, num_workers=4)
    else:
        raise NotImplementedError(f'Dataset {data_args.dataset_name} is not implemented.')

    return dataloader


def create_elbow_graphs(args: Args):
    """The main function, producing the different graphs (logging to wandb and to locally .tex files).
    """
    dataloader = get_dataloader(args.data)

    figures = dict()
    n_centroids_list = list(range(args.int_dim.min_n_centroids, args.int_dim.max_n_centroids + 1))

    logger.info(f'Calculating the whitening-matrix for patch-size '
                f'{args.int_dim.patch_size}x{args.int_dim.patch_size}.')
    whitening_matrix = calc_whitening_from_dataloader(dataloader,
                                                      args.int_dim.patch_size,
                                                      args.int_dim.whitening_regularization_factor,
                                                      args.int_dim.zca_whitening)
    prefix = f'{args.int_dim.n_points:,}_{args.int_dim.patch_size}x{args.int_dim.patch_size}_patches'
    logger.info(f'Starting with {prefix}')

    patches = sample_random_patches(dataloader, args.int_dim.n_points, args.int_dim.patch_size, verbose=True)
    patches_flat = patches.reshape(patches.shape[0], -1)
    n_patches, patch_dim = patches_flat.shape

    data_dict = create_data_dict(patches_flat, whitening_matrix)
    normalized_data_dict = normalize_data(data_dict)

    norms_dict = {data_name: np.linalg.norm(data, axis=1)
                  for data_name, data in data_dict.items()}
    low_norms_masks_dict = {data_name: (norms < np.percentile(norms, q=0.1))
                            for data_name, norms in norms_dict.items()}

    norms_df = pd.DataFrame(norms_dict)
    for col in norms_df.columns:
        figures[f'{prefix}_{col}_norms'] = px.histogram(norms_df, x=col, marginal='box')

    # `dist_to_centroid` will be in the following structure:
    #     {'mean','max','median'}:
    #         {'normalized-data', 'data'}:
    #             {'normalized-distance', 'distance'}:
    #                 {'original', 'shuffled', ...}:
    #                     [---------------- values ----------------]
    dist_to_centroid = dict()
    norm_data_names = ['normalized-data']  # ['normalized-data', 'data']
    norm_dist_names = ['distance']  # ['normalized-distance', 'distance']
    agg_funcs = {'mean': np.mean}  # {'mean': np.mean, 'max': np.max, 'median': np.median}
    for agg_name in agg_funcs.keys():
        dist_to_centroid[agg_name] = dict()
        for norm_data_name in norm_data_names:
            dist_to_centroid[agg_name][norm_data_name] = dict()
            for norm_dist_name in norm_dist_names:
                dist_to_centroid[agg_name][norm_data_name][norm_dist_name] = dict()
                for data_name in data_dict.keys():
                    dist_to_centroid[agg_name][norm_data_name][norm_dist_name][data_name] = list()

    for data_name in data_dict.keys():
        for norm_data_name in norm_data_names:
            for k in tqdm(n_centroids_list,
                          desc=f'Running k-means on {data_name} {norm_data_name} for different values of k'):
                data = normalized_data_dict[data_name] if norm_data_name == 'normalized-data' else data_dict[data_name]

                _, _, distances = run_kmeans_clustering(data, k, args.env.use_faiss)

                for norm_dist_name in norm_dist_names:
                    # If the data is already normalized, no need to divide by the norm.
                    if (norm_dist_name == 'normalized-distance') and (norm_data_name != 'normalized-data'):
                        not_low_norm_mask = np.logical_not(low_norms_masks_dict[data_name])
                        distances_filtered = distances[not_low_norm_mask]
                        norms_filtered = norms_dict[data_name][not_low_norm_mask]
                        distances = distances_filtered / norms_filtered

                    for agg_name, agg_func in agg_funcs.items():
                        dist_to_centroid[agg_name][norm_data_name][norm_dist_name][data_name].append(
                            agg_func(distances))

    for agg_name in agg_funcs.keys():
        for norm_data_name in norm_data_names:
            for norm_dist_name in norm_dist_names:
                df = pd.DataFrame(data=dist_to_centroid[agg_name][norm_data_name][norm_dist_name],
                                  index=n_centroids_list)
                df.name = f'{args.data.dataset_name}-{norm_data_name}-{agg_name}-{norm_dist_name}-distance-to-centroid'
                df.index.name = 'k'
                df.columns.name = 'type-of-data'
                figures[f'{prefix}-{df.name}'] = px.line(df)

                plt.figure()
                plt.style.use("ggplot")
                for col in df.columns:
                    plt.plot(n_centroids_list, df[col], label=col)
                plt.legend()
                plt.xlabel('k')
                plt.ylabel('mean-distance')
                plt.grid(True)
                tikzplotlib.save(os.path.join('figures', f'{df.name}.tex'))

    wandb.log(figures, step=0)


def main():
    args = get_args(args_class=Args)

    configure_logger(args.env.path)
    log_args(args)
    wandb.init(project='thesis', name=args.env.wandb_run_name, config=args.flattened_dict())

    create_elbow_graphs(args)


if __name__ == '__main__':
    main()
