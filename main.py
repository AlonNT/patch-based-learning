import math
import wandb
import torch
import faiss

import numpy as np
import torchmetrics as tm
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, List, Tuple, Union, Dict, Type

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100
from torchvision.transforms import ToTensor, RandomCrop, RandomHorizontalFlip, Normalize, Compose
from torchvision.transforms.functional import center_crop
from pytorch_lightning import LightningDataModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary

from consts import N_CLASSES
from schemas.architecture import ArchitectureArgs
from schemas.data import DataArgs
from schemas.optimization import OptimizationArgs
from schemas.patches import PatchesArgs, Args
from utils import (sample_random_patches, get_args, get_model_device, get_mlp,
                   get_dataloaders, calc_whitening_from_dataloader,
                   get_whitening_matrix_from_covariance_matrix, get_model_output_shape,
                   get_random_initialized_conv_kernel_and_bias, calc_covariance)
from vgg import get_vgg_model_kernel_size, get_vgg_blocks, configs


class KNearestPatchesEmbedding(nn.Module):
    def __init__(self, kernel: np.ndarray, bias: np.ndarray, stride: int, padding: int, args: PatchesArgs):
        """Calculate the k-nearest-neighbors for each patch in the input image.

        Calculating the k-nearest-neighbors is implemented as a convolution layer, as in
        The Unreasonable Effectiveness of Patches in Deep Convolutional Kernels Methods
        (https://arxiv.org/pdf/2101.07528.pdf)
        Details can be found in Appendix B (page 13).

        Args:
            kernel: The kernel that will be used during the embedding calculation.
                For example, when using the embedding on the original patches (not-whitened) the kernel will
                be the patches themselves in the patches-dictionary. if we use whitening then the kernel is
                the patches multiplied by WW^T and the bias is the squared-norm of patches multiplied by W (no W^T).
            bias: The bias that will be used during the embedding calculation.
                For example, when using the embedding on the original patches (not-whitened) the bias will
                be the squared-norms of the patches in the patches-dictionary.
            stride: The stride to use.
            padding: The amount of padding to use.
            args: Various arguments, in the form of a pydantic class.
                See the docs in the class itself for each argument used for more information.
        """
        super(KNearestPatchesEmbedding, self).__init__()

        self.k: int = args.k
        self.up_to_k: bool = args.up_to_k
        self.stride: int = stride
        self.padding: int = padding
        self.kmeans_triangle: bool = args.kmeans_triangle
        self.random_embedding: bool = args.random_embedding
        self.learnable_embedding: bool = args.learnable_embedding

        if self.random_embedding:
            out_channels, in_channels, kernel_height, kernel_width = kernel.shape
            assert kernel_height == kernel_width, "the kernel should be square"
            kernel, bias = get_random_initialized_conv_kernel_and_bias(in_channels, out_channels, kernel_height)

        self.kernel = nn.Parameter(torch.Tensor(kernel), requires_grad=self.learnable_embedding)
        self.bias = nn.Parameter(torch.Tensor(bias), requires_grad=self.learnable_embedding)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass.

        Args:
            images: The input tensor of shape (B, C, H, W) to calculate the k-nearest-neighbors.

        Returns:
            A tensor of shape (B, N, H, W) where N is the size of the patches-dictionary.
            The ij spatial location will hold the mask indicating the k nearest neighbors of the patch centered at ij.
        """
        # In every spatial location ij, we'll have a vector containing the squared distances to all the patches.
        # Note that it's not really the squared distance, but the squared distance minus the squared-norm of the
        # input patch in that location, but minimizing this value will minimize the distance
        # (since the norm of the input patch is the same among all patches in the bank).
        distances = F.conv2d(images, self.kernel, self.bias, self.stride, self.padding)
        values, indices = distances.kthvalue(k=self.k, dim=1, keepdim=True)

        if self.kmeans_triangle:
            mask = F.relu(torch.mean(distances, dim=1, keepdim=True) - distances)
        elif self.up_to_k:
            mask = torch.le(distances, values).float()
        else:
            mask = torch.zeros_like(distances).scatter_(dim=1, index=indices, value=1)

        return mask


class PatchBasedNetwork(pl.LightningModule):
    def __init__(self, args: Args, input_shape: Optional[Tuple[int, int, int]] = None):
        """The patch-based model, named A_Patch in the paper.

        This model contains a linear-network on top of a data-dependent embedding of the image.
        The embedding can be either Phi_full or Phi_hard (notations from the paper).

        Args:
            args: The arguments-schema.
            input_shape: If given, overrides the input shape of the data itself (used in deep models).
        """
        super(PatchBasedNetwork, self).__init__()
        self.args: Args = args
        self.save_hyperparameters(args.flattened_dict())

        self.whitening_matrix: Optional[np.ndarray] = None
        self.patches_covariance_matrix: Optional[np.ndarray] = None
        self.whitened_patches_covariance_matrix: Optional[np.ndarray] = None

        self.input_shape = self.init_input_shape(input_shape)
        self.input_channels = self.input_shape[0]
        self.input_spatial_size = self.input_shape[1]

        self.embedding = self.init_embedding()
        self.conv = self.init_conv()
        self.avg_pool = self.init_avg_pool()
        self.adaptive_avg_pool = self.init_adaptive_avg_pool()
        self.batch_norm = self.init_batch_norm()
        self.bottle_neck = self.init_bottleneck()
        self.bottle_neck_relu = self.init_bottleneck_relu()
        self.linear = nn.Linear(self.calc_linear_in_features(), N_CLASSES)

        self.loss = nn.CrossEntropyLoss()

        self.train_accuracy = tm.Accuracy()
        self.validate_accuracy = tm.Accuracy()
        self.accuracy = {RunningStage.TRAINING: self.train_accuracy,
                         RunningStage.VALIDATING: self.validate_accuracy}

        self.logits_prediction_mode: bool = True

    def embedding_mode(self):
        """Sets the network in 'embedding-mode', meaning removing the last linear layer predicting the logits.

        This is useful when stacking one model on top of the other to create a deep model.
        """
        self.set_logits_prediction_mode(False)

    def logits_mode(self):
        """Sets the network in "logits-mode", meaning that the last linear layer predicting the logits will be used.
        """
        self.set_logits_prediction_mode(True)

    def set_logits_prediction_mode(self, mode: bool):
        """Changes logits_prediction_mode to be the given mode.
        """
        self.logits_prediction_mode = mode

    def init_input_shape(self, input_shape: Optional[Tuple[int, int, int]]) -> Tuple[int, int, int]:
        """Initializes the input shape.

        Args:
            input_shape: The given input shape.

        Returns:
            The given input shape as is (if actually given, i.e. not None), or the data input shape.
        """
        if input_shape is not None:
            return input_shape
        return (self.args.data.n_channels,) + (self.args.data.spatial_size,) * 2

    def init_embedding(self) -> Union[KNearestPatchesEmbedding, nn.Sequential]:
        """Initializes the embedding layer.

        Returns:
            Typically an instance of KNearestPatchesEmbedding, but can also be a sequential
            model containing learnable Conv2d followed by ReLU, when training vanilla CNN as a baseline.
        """
        conv_args = dict(in_channels=self.input_channels,
                         out_channels=self.args.patches.n_clusters,
                         kernel_size=self.args.arch.kernel_size)

        if self.args.patches.replace_embedding_with_regular_conv_relu:
            return nn.Sequential(nn.Conv2d(**conv_args), nn.ReLU())

        kernel, bias = get_random_initialized_conv_kernel_and_bias(**conv_args)
        return KNearestPatchesEmbedding(kernel, bias, self.args.arch.stride, self.args.arch.padding, self.args.patches)

    def calculate_embedding_from_data(self, dataloader: DataLoader, pre_model: Optional[nn.Module] = None):
        """Calculates the embedding from the given dataloader.

        The embedding will be calculated by sampling patches uniformly-at-random,
        performing whitening, and finally running k-means clustering.

        Args:
            dataloader: The dataloader to sample patches from.
            pre_model: If given, each input image is transformed using this model before sampling patches.
                It's used when creating deep models, where the input to the model is the output of the previous one.
        """
        assert not self.args.patches.replace_embedding_with_regular_conv_relu
        kernel, bias = self.get_kernel_and_bias_from_data(dataloader, pre_model)
        self.embedding = KNearestPatchesEmbedding(kernel, bias,
                                                  self.args.arch.stride, self.args.arch.padding, self.args.patches)

    def init_whitening_matrix(self,
                              dataloader: DataLoader,
                              pre_model: Optional[nn.Module] = None) -> Optional[np.ndarray]:
        """Calculates the whitening-matrix from the given dataloader.

        Args:
            dataloader: The dataloader to sample patches from.
            pre_model: If given, each input image is transformed using this model before sampling patches.
                It's used when creating deep models, where the input to the model is the output of the previous one.

        Returns:
            The whitening matrix.
        """
        if not self.args.patches.use_whitening:
            return None

        if (self.args.patches.random_gaussian_patches or
                self.args.patches.random_uniform_patches or
                self.args.patches.calc_whitening_from_sampled_patches):
            return get_whitening_matrix_from_covariance_matrix(self.patches_covariance_matrix,
                                                               self.args.patches.whitening_regularization_factor,
                                                               self.args.patches.zca_whitening)

        return calc_whitening_from_dataloader(dataloader,
                                              self.args.arch.kernel_size,
                                              self.args.patches.whitening_regularization_factor,
                                              self.args.patches.zca_whitening,
                                              pre_model)

    def init_conv(self) -> Optional[nn.Conv2d]:
        """Initializes the convolution layer.

        Returns:
            A convolution layer which calculates the linear-functions corresponding to each patch in the
            patches-dictionary, which are later multiplied element-wise to create the vector psi
            (see Section 4.3 in the paper).
        """
        if not self.args.patches.use_conv:
            return None
        return nn.Conv2d(in_channels=self.input_channels,
                         out_channels=self.args.patches.n_clusters * self.args.patches.c,
                         kernel_size=self.args.arch.kernel_size,
                         stride=self.args.arch.stride,
                         padding=self.args.arch.padding)

    def init_adaptive_avg_pool(self) -> Optional[nn.AdaptiveAvgPool2d]:
        """Initializes the adaptive-average-pooling layer (by default it's None, i.e. not used).

        This layer is used to re-produce the model suggested in
        "The Unreasonable Effectiveness of Patches in Deep Convolutional Kernels Methods"
        where they had an adaptive-average-pooling layer (right after the regular average-pooling layer,
        before feeding it to the linear layer).

        Returns:
            An adaptive-average-pooling layer, reducing the spatial dimension of the average-pooling output.
        """
        if not self.args.patches.use_adaptive_avg_pool:
            return None
        return nn.AdaptiveAvgPool2d(output_size=self.args.patches.adaptive_pool_output_size)

    def init_avg_pool(self) -> Optional[nn.AvgPool2d]:
        """Initializes the average-pooling layer.

        Returns:
            An average-pooling layer, reducing the spatial dimension of the output of the first layer.
            Its affect is discussed in Section 5.3 in the paper (specifically Table 3).
        """
        if not self.args.patches.use_avg_pool:
            return None
        return nn.AvgPool2d(kernel_size=self.args.arch.pool_size, stride=self.args.arch.pool_stride, ceil_mode=True)

    def init_batch_norm(self) -> Optional[nn.BatchNorm2d]:
        """Initializes the batch normalization layer.

        This layer is used on top of the output of the average-pooling layer, 
        which proved helpful for the optimization.

        Returns:
            A BatchNorm layer.
        """
        if not self.args.arch.use_batch_norm:
            return None
        num_features = self.args.patches.n_clusters if (self.args.patches.c == 1) else self.args.patches.c
        return nn.BatchNorm2d(num_features)

    def init_bottleneck(self) -> Optional[nn.Conv2d]:
        """Initializes the bottleneck layer.

        Returns:
            A convolution-layer (typically 1x1, but this can be modified), which comes right before the
            final linear layer. Its affect is discussed in Section 5.3 in the paper.
        """
        if not self.args.patches.use_bottle_neck:
            return None

        in_channels = self.args.patches.n_clusters if (self.args.patches.c == 1) else self.args.patches.c
        return nn.Conv2d(in_channels=in_channels,
                         out_channels=self.args.arch.bottle_neck_dimension,
                         kernel_size=self.args.arch.bottle_neck_kernel_size)

    def init_bottleneck_relu(self) -> Optional[nn.ReLU]:
        """Initializes the ReLU layer (by default it's None, i.e. not used).

        This layer comes after the bottleneck layer (before the final linear layer).
        It's used to compare the performance of Phi_full against Phi_hard,
        in the non-linear model suggested in Thiry et al.

        Returns:
            A ReLU module.
        """
        if not self.args.patches.use_relu_after_bottleneck:
            return None
        return nn.ReLU()

    def calc_linear_in_features(self) -> int:
        """
        Returns:
            The number of input features for the linear layer.
        """
        # Inspiration is taken from PyTorch Conv2d docs regarding the output shape
        # https://pytorch.org/docs/1.10.1/generated/torch.nn.Conv2d.html
        embedding_spatial_size = math.floor(
            1 + ((self.input_spatial_size + 2 * self.args.arch.padding - self.args.arch.kernel_size) /
                 self.args.arch.stride)
        )

        if self.args.patches.use_adaptive_avg_pool:
            intermediate_spatial_size = self.args.patches.adaptive_pool_output_size
        elif self.args.patches.use_avg_pool:  # ceil and not floor, because we used `ceil_mode=True` in AvgPool2d
            intermediate_spatial_size = math.ceil(1 + ((embedding_spatial_size - self.args.arch.pool_size) /
                                                       self.args.arch.pool_stride))
        else:
            intermediate_spatial_size = embedding_spatial_size

        if self.args.patches.full_embedding:
            patch_dim = self.input_channels * self.args.arch.kernel_size ** 2
            embedding_n_channels = self.args.patches.n_clusters * patch_dim
        elif self.args.patches.c > 1:
            embedding_n_channels = self.args.patches.c
        else:
            embedding_n_channels = self.args.patches.n_clusters

        intermediate_n_features = embedding_n_channels * (intermediate_spatial_size ** 2)
        bottleneck_output_spatial_size = intermediate_spatial_size - self.args.arch.bottle_neck_kernel_size + 1
        if self.args.patches.residual_cat:
            bottle_neck_dimension = self.args.arch.bottle_neck_dimension + self.input_channels
        else:
            bottle_neck_dimension = self.args.arch.bottle_neck_dimension
        bottleneck_output_n_features = bottle_neck_dimension * (bottleneck_output_spatial_size ** 2)

        if self.args.patches.use_bottle_neck:
            linear_in_features = bottleneck_output_n_features
        else:
            linear_in_features = intermediate_n_features

        return linear_in_features

    def get_kernel_and_bias_from_data(self,
                                      dataloader: DataLoader,
                                      pre_model: Optional[nn.Module] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates the kernel and bias for the embedding layer.

        Args:
            dataloader: The dataloader to sample patches from.
            pre_model: If given, each input image is transformed using this model before sampling patches.
                It's used when creating deep models, where the input to the model is the output of the previous one.

        Returns:
            The kernel and bias to use in the embedding layer.
        """
        # Set the kernel and the bias of the embedding. Note that if we use whitening then the kernel is the patches
        # multiplied by WW^T and the bias is the squared-norm of patches multiplied by W (no W^T).
        kernel = self.get_clustered_patches(dataloader, pre_model)
        bias = 0.5 * np.linalg.norm(kernel.reshape(kernel.shape[0], -1), axis=1) ** 2
        if self.args.patches.use_whitening:
            kernel_flat = kernel.reshape(kernel.shape[0], -1)
            kernel_flat = kernel_flat @ self.whitening_matrix.T
            kernel = kernel_flat.reshape(kernel.shape)
        kernel *= -1  # According to the formula as page 13 in https://arxiv.org/pdf/2101.07528.pdf

        return kernel, bias

    def get_clustered_patches(self, dataloader: DataLoader, pre_model: Optional[nn.Module] = None) -> np.ndarray:
        """Gets the clustered (possibly whitened) patches which define the "patches-dictionary".

        Args:
            dataloader: The dataloader to sample patches from.
            pre_model: If given, each input image is transformed using this model before sampling patches.
                It's used when creating deep models, where the input to the model is the output of the previous one.

        Returns:
            The clustered patches, defining the patches-dictionary.
        """
        # Sample a bit more (1% more), because some are removed layer in `remove_low_norm_patches`
        # and eventually we want to have exactly `n_patches` patches.
        n_patches_extended = math.ceil(1.01 * self.args.patches.n_patches)
        patches = sample_random_patches(dataloader, n_patches_extended, self.args.arch.kernel_size,
                                        existing_model=pre_model,
                                        random_uniform_patches=self.args.patches.random_uniform_patches,
                                        random_gaussian_patches=self.args.patches.random_gaussian_patches)
        patch_shape = patches.shape[1:]
        patches = patches.reshape(patches.shape[0], -1)

        patches = self.remove_low_norm_patches(patches)

        self.patches_covariance_matrix = calc_covariance(patches)
        self.whitening_matrix = self.init_whitening_matrix(dataloader, pre_model)

        if self.args.patches.use_whitening:
            patches = patches @ self.whitening_matrix
            self.whitened_patches_covariance_matrix = calc_covariance(patches)

        if self.args.patches.n_patches > self.args.patches.n_clusters:
            kmeans = faiss.Kmeans(d=patches.shape[1], k=self.args.patches.n_clusters, verbose=True)
            kmeans.train(patches)
            patches = kmeans.centroids

        if self.args.patches.normalize_patches_to_unit_vectors:
            patches /= np.linalg.norm(patches, axis=1)[:, np.newaxis]

        return patches.reshape(-1, *patch_shape)

    def remove_low_norm_patches(self, patches: np.ndarray) -> np.ndarray:
        """Removed patches with extremely low norm from the given patches.

        Patches with extremely low norm become problematic later, if we divide by the norm to get unit-vectors.
        """
        minimal_norm = 0.01
        low_norm_patches_mask = (np.linalg.norm(patches, axis=1) < minimal_norm)
        patches = patches[np.logical_not(low_norm_patches_mask)]
        patches = patches[:self.args.patches.n_patches]  # Don't leave more than the requested patches

        return patches

    def forward(self, x: torch.Tensor):
        """Performs a forward pass.

        Args:
            x: The input tensor.

        Returns:
            The output of the model, which is logits in case the model is in `logits_prediction_mode`,
            and the intermediate embedding in case `logits_prediction_mode` is False (used in deep models).
        """
        args = self.args.patches

        features = self.embedding(x)

        if args.full_embedding:
            features = self.get_full_embedding(x, features)
        if args.c > 1:
            features = torch.repeat_interleave(features, repeats=args.c, dim=1)
        if args.use_conv:
            features *= self.conv(x)
        if args.c > 1:
            features = features.view(features.shape[0], args.n_clusters, args.c, *features.shape[2:])
            features = torch.sum(features, dim=1) / args.k
        if args.use_avg_pool:
            features = self.avg_pool(features)
        if args.use_adaptive_avg_pool:
            features = self.adaptive_avg_pool(features)
        if self.args.arch.use_batch_norm:
            features = self.batch_norm(features)
        if args.use_bottle_neck:
            features = self.bottle_neck(features)
        if args.use_relu_after_bottleneck:
            features = self.bottle_neck_relu(features)
        if args.residual_add or args.residual_cat:
            if x.shape[-2:] != features.shape[-2:]:  # Spatial size might be slightly different due to lack of padding
                x = center_crop(x, output_size=features.shape[-1])
            features = torch.add(features, x) if args.residual_add else torch.cat((features, x), dim=1)

        if not self.logits_prediction_mode:
            return features

        logits = self.linear(torch.flatten(features, start_dim=1))

        return logits

    def get_full_embedding(self, x: torch.Tensor, nearest_neighbors_mask: torch.Tensor):
        """Calculates the Phi_full embedding, which has the patch values scattered in the k neighbors indices

        For more information, see Section 5.3 in the paper.

        Args:
            x: The input tensor.
            nearest_neighbors_mask: A tensor containing in each spatial location a mask
                indicating the k-nearest-neighbors of the patches centered in this location.

        Returns:
            A tensor of shape (B, n_clusters * patch_dim, H, W) there the q-th block in the ij-spatial location contains
            the values of the patch centered in ij, or zero if the q-th patch is not in the k-nearest-neighbors.
        """
        x = F.unfold(x,
                     kernel_size=self.args.arch.kernel_size,
                     stride=self.args.arch.stride,
                     padding=self.args.arch.padding)
        batch_size, patch_dim, n_patches = x.shape
        spatial_size = int(math.sqrt(n_patches))  # This is the spatial size (e.g. 28 in CIFAR10 with patch-size 5)

        x = x.reshape(batch_size, patch_dim, spatial_size, spatial_size)
        x = torch.repeat_interleave(x, repeats=self.args.patches.n_clusters, dim=1)
        x = x.reshape(batch_size, patch_dim, self.args.patches.n_clusters, spatial_size, spatial_size)
        x = x.transpose(1, 2)
        x = x.reshape(batch_size, patch_dim * self.args.patches.n_clusters, spatial_size, spatial_size)

        return x * torch.repeat_interleave(nearest_neighbors_mask, repeats=patch_dim, dim=1)

    def shared_step(self, batch: Tuple[torch.Tensor, torch.Tensor], stage: RunningStage):
        """Performs train/validation step, depending on the given `stage`.

        Note that when training in multi-GPU setting, in `DataParallel` strategy, the input `batch` will actually
        be only a portion of the input batch.
        We also return the logits and the labels to calculate the accuracy in `shared_step_end`.

        Args:
            batch: The batch to process (containing a tuple of tensors - inputs and labels).
            stage: Indicating if this is a training-step or a validation-step.

        Returns:
            A dictionary containing the loss, logits and labels.
        """
        assert self.logits_prediction_mode, 'Can not do train/validation step when logits prediction mode is off'

        x, labels = batch
        logits = self(x)
        loss = self.loss(logits, labels)

        self.log(f'{stage.value}_loss', loss)

        return {'loss': loss, 'logits': logits, 'labels': labels}

    def shared_step_end(self, batch_parts_outputs: Dict[str, torch.Tensor], stage: RunningStage):
        """Finalize the train/validation step, depending on the given `stage`.

        When using multi-GPU (`DataParallel` strategy), there is an error due to accumulating the Accuracy metric
        in different devices. The solution (as proposed in the following link) is the call the Accuracy in *_step_end.
        https://github.com/PyTorchLightning/pytorch-lightning/issues/4353#issuecomment-716224855

        Args:
            batch_parts_outputs: The outputs of each GPU in the multi-GPU setting (`DataParallel` strategy).
            stage: Indicating if this is a training-step or a validation-step.

        Returns:
            The mean loss of the batch part losses
        """
        loss = batch_parts_outputs['loss']
        logits = batch_parts_outputs['logits']
        labels = batch_parts_outputs['labels']

        accuracy = self.accuracy[stage]
        accuracy(logits, labels)

        self.log(f'{stage.value}_accuracy', accuracy, metric_attribute=f'{stage.value}_accuracy')

        return loss.mean()

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Performs a training-step.

        Args:
            batch: The batch to process (containing a tuple of tensors - inputs and labels).
            batch_idx: The index of the batch in the dataset.

        Returns:
            A dictionary containing the loss, logits and labels.
        """
        return self.shared_step(batch, RunningStage.TRAINING)

    def training_step_end(self, batch_parts_outputs):
        """Finalize the training step.

        Args:
            batch_parts_outputs: The outputs of each GPU in the multi-GPU setting (`DataParallel` strategy).

        Returns:
            The mean loss of the batch part losses
        """
        return self.shared_step_end(batch_parts_outputs, RunningStage.TRAINING)

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Performs a validation-step.

        Args:
            batch: The batch to process (containing a tuple of tensors - inputs and labels).
            batch_idx: The index of the batch in the dataset.

        Returns:
            A dictionary containing the loss, logits and labels.
        """
        return self.shared_step(batch, RunningStage.VALIDATING)

    def validation_step_end(self, batch_parts_outputs):
        """Finalize the validation step.

        Args:
            batch_parts_outputs: The outputs of each GPU in the multi-GPU setting (`DataParallel` strategy).

        Returns:
            The mean loss of the batch part losses
        """
        return self.shared_step_end(batch_parts_outputs, RunningStage.VALIDATING)

    def configure_optimizers(self):
        """Configure the optimizer and the learning-rate scheduler for the training process.

        Returns:
            A dictionary containing the optimizer and learning-rate scheduler.
        """
        optimizer = torch.optim.SGD(self.parameters(),
                                    self.args.opt.learning_rate,
                                    self.args.opt.momentum,
                                    weight_decay=self.args.opt.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=self.args.opt.learning_rate_decay_steps,
                                                         gamma=self.args.opt.learning_rate_decay_gamma)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def on_train_start(self):
        """Visualization at the beginning of the training process (after the model's initialization).

        Visualizes the patches (if they are 3-dimensional, i.e. RGB images),
        as well as the covariance matrix of the sampled patches, before and after whitening.
        """
        if self.input_channels == 3:
            self.visualize_patches()
        m = 64  # this is the maximal number of elements to take when calculating in covariance matrix
        if self.patches_covariance_matrix is not None:
            self.log_covariance_matrix(self.patches_covariance_matrix[:m, :m], name='patches')
        if self.args.patches.use_whitening and (self.whitened_patches_covariance_matrix is not None):
            self.log_covariance_matrix(self.whitened_patches_covariance_matrix[:m, :m], name='whitened_patches')

    def on_train_end(self):
        """Visualizes the patches (if they are 3-dimensional, i.e. RGB images) at the end of the training process.
        """
        if self.input_channels == 3:
            self.visualize_patches()

    def visualize_patches(self, n: int = 8):
        """Visualize patches from the model's patches-dictionary.

        The chosen patches are those that had the highest "effect" on the network's output, done by measuring the norm
        of the bottleneck layer. If there is no bottleneck layer, patches are chosen uniformly at random.

        Args:
            n: The size of the grid of patches to visualize (total n^2 patches will be visualized).
        """
        if self.bottle_neck is not None:
            weights = self.bottle_neck.weight.data
            weights = weights.mean((2, 3)) if self.args.arch.bottle_neck_kernel_size > 1 else weights.squeeze(
                3).squeeze(2)
            norms = torch.linalg.norm(weights, ord=2, dim=0).cpu().numpy()
        else:
            norms = np.random.default_rng().uniform(size=self.args.patches.n_clusters).astype(np.float32)

        if n ** 2 >= len(norms) / 2:  # Otherwise, we'll crash when asking for n**2 best/worth norms.
            n = math.floor(math.sqrt(len(norms) / 2))

        worst_patches_indices, best_patches_indices = self.get_extreme_patches_indices(norms, n ** 2)
        worst_patches_unwhitened, best_patches_unwhitened = self.get_extreme_patches_unwhitened(
            worst_patches_indices, best_patches_indices)
        worst_patches_whitened, best_patches_whitened = self.get_extreme_patches_unwhitened(
            worst_patches_indices, best_patches_indices, only_inv_t=True)

        # Normalize the values to be in [0, 1] for plotting (matplotlib just clips the values).
        for a in [worst_patches_unwhitened, best_patches_unwhitened, worst_patches_whitened, best_patches_whitened]:
            a[:] = (a - a.min()) / (a.max() - a.min())

        worst_patches_fig, worst_patches_axs = plt.subplots(n, n)
        best_patches_fig, best_patches_axs = plt.subplots(n, n)
        worst_patches_whitened_fig, worst_patches_whitened_axs = plt.subplots(n, n)
        best_patches_whitened_fig, best_patches_whitened_axs = plt.subplots(n, n)

        for k in range(n ** 2):
            i = k // n  # Row index
            j = k % n  # Columns index
            worst_patches_axs[i, j].imshow(worst_patches_unwhitened[k].transpose(1, 2, 0), vmin=0, vmax=1)
            best_patches_axs[i, j].imshow(best_patches_unwhitened[k].transpose(1, 2, 0), vmin=0, vmax=1)
            worst_patches_whitened_axs[i, j].imshow(worst_patches_whitened[k].transpose(1, 2, 0), vmin=0, vmax=1)
            best_patches_whitened_axs[i, j].imshow(best_patches_whitened[k].transpose(1, 2, 0), vmin=0, vmax=1)

            worst_patches_axs[i, j].axis('off')
            best_patches_axs[i, j].axis('off')
            worst_patches_whitened_axs[i, j].axis('off')
            best_patches_whitened_axs[i, j].axis('off')

        self.trainer.logger.experiment.log({
            'worst_patches': worst_patches_fig, 'best_patches': best_patches_fig,
            'worst_patches_whitened': worst_patches_whitened_fig, 'best_patches_whitened': best_patches_whitened_fig
        }, step=self.trainer.global_step)

        plt.close('all')  # Avoid memory consumption

    @staticmethod
    def get_extreme_patches_indices(norms, n):
        """Gets the "extreme" (possibly whitened) patches indices, meaning they have the lowest/highest norms.

        Args:
            norms: The norms of the weights of the different patches.
            n: number of extreme patches to extract.

        Returns:
            A tuple containing the worst and best patches indices.
        """
        partitioned_indices = np.argpartition(norms, n)
        worst_patches_indices = partitioned_indices[:n]
        partitioned_indices = np.argpartition(norms, len(norms) - n)
        best_patches_indices = partitioned_indices[-n:]

        return worst_patches_indices, best_patches_indices

    def get_extreme_patches_unwhitened(self,
                                       worst_patches_indices: np.ndarray,
                                       best_patches_indices: np.ndarray,
                                       only_inv_t: bool = False):
        """Gets the "extreme" (possibly whitened) patches, meaning they have the lowest/highest norms.

        Note that in order to gets the whitened patches we still need to multiply the patches by W^T,
        because the way our model works is to multiply the patches by WW^T (so later convolving with
        these patches gives the distance between the whitened input patches and the whitened patches).

        Args:
            worst_patches_indices: The indices of the worst indices.
            best_patches_indices: The indices of the best indices.
            only_inv_t: Whether to multiply the patches by W or WW^T (if we want the whitened or un-whitened patches).

        Returns:
            A tuple containing the worst and best patches, possible un-whitened.
        """
        if self.args.patches.replace_embedding_with_regular_conv_relu:
            kernel = self.embedding[0].weight
        else:
            kernel = self.embedding.kernel

        all_patches = kernel.data.cpu().numpy()
        worst_patches = all_patches[worst_patches_indices]
        best_patches = all_patches[best_patches_indices]

        both_patches = np.concatenate([worst_patches, best_patches])
        both_patches_unwhitened = self.unwhiten_patches(both_patches, only_inv_t)

        worst_patches_unwhitened = both_patches_unwhitened[:len(worst_patches)]
        best_patches_unwhitened = both_patches_unwhitened[len(best_patches):]

        return worst_patches_unwhitened, best_patches_unwhitened

    def unwhiten_patches(self, patches: np.ndarray, only_inv_t: bool = False) -> np.ndarray:
        """Un-whiten the given patches, by multiplying by W or WW^T.

        Args:
            patches: The indices of the worst indices.
            only_inv_t: Whether to multiply the patches by W or WW^T (if we want the whitened or un-whitened patches).

        Returns:
            The patches after multiplying with W or WW^T.
        """
        patches_flat = patches.reshape(patches.shape[0], -1)
        whitening_matrix = np.eye(patches_flat.shape[1]) if (self.whitening_matrix is None) else self.whitening_matrix
        matrix = whitening_matrix.T if only_inv_t else (whitening_matrix @ whitening_matrix.T)
        patches_orig_flat = np.dot(patches_flat, np.linalg.inv(matrix))
        patches_orig = patches_orig_flat.reshape(patches.shape)

        return patches_orig

    def log_covariance_matrix(self, covariance_matrix: np.ndarray, name: str):
        """Log the given covariance to wandb.

        Args:
            covariance_matrix: The covariance matrix.
            name: The name to add to the visualization key as a prefix.
        """
        labels = [f'{i:0>3}' for i in range(covariance_matrix.shape[0])]
        name = f'{name}_cov_matrix'
        self.logger.experiment.log(
            {name: wandb.plots.HeatMap(x_labels=labels, y_labels=labels, matrix_values=covariance_matrix)},
            step=self.trainer.global_step)
        self.logger.experiment.summary[f'{name}_ratio'] = self.get_diagonal_sum_vs_total_sum_ratio(covariance_matrix)

    @staticmethod
    def get_diagonal_sum_vs_total_sum_ratio(matrix: np.ndarray) -> float:
        """
        Args:
            matrix: A 2-dimensional matrix.
        Returns:
            The ratio between the diagonal sum and the total sum of the given matrix
        """
        non_diagonal_elements_sum = np.sum(np.abs(matrix - np.diag((np.diagonal(matrix)))))
        elements_sum = np.sum(np.abs(matrix))
        ratio = non_diagonal_elements_sum / elements_sum
        return ratio


class DeepPatchBasedNetwork(pl.LightningModule):
    def __init__(self, args: Args):
        """The deep patch-based model, studied in Section 5.4 ("How our model scales with depth").

        This model contains several modules of `PatchBasedModel` stacked one on top of the other,
        trained in a layer-wise fashion.

        Args:
            args: The arguments-schema.
        """
        super(DeepPatchBasedNetwork, self).__init__()
        self.args: Args = args
        self.save_hyperparameters(args.flattened_dict())
        self.input_channels = self.args.data.n_channels
        self.input_spatial_size = self.args.data.spatial_size
        self.input_shape = (self.input_channels,) + 2 * (self.input_spatial_size,)

        self.layers = nn.ModuleList()

        self.loss = nn.CrossEntropyLoss()

        self.train_accuracy = tm.Accuracy()
        self.validate_accuracy = tm.Accuracy()
        self.accuracy = {RunningStage.TRAINING: self.train_accuracy,
                         RunningStage.VALIDATING: self.validate_accuracy}

    def add_layer(self, dataloader: DataLoader):
        """Adds a layer on top of the current model.

        The last layer is frozen (the previous ones have already been frozen before,
        when the last layer was added), and another PatchBasedModel is built on top of that.

        Args:
            dataloader: The loader to sample patches from, creating the new PatchBasedModel's patches-dictionary.
        """
        if len(self.layers) > 0:
            self.layers[-1].eval()
            self.layers[-1].requires_grad_(False)

        pre_model = nn.Sequential(*self.layers)

        args = self.args.extract_single_depth_args(i=len(self.layers))
        new_layer = PatchBasedNetwork(args, input_shape=get_model_output_shape(pre_model))

        if not args.patches.replace_embedding_with_regular_conv_relu:
            original_device = get_model_device(pre_model)
            pre_model.to(self.args.env.device)
            new_layer.calculate_embedding_from_data(dataloader, pre_model)
            pre_model.to(original_device)

        new_layer.embedding_mode()  # we'll run the linear layer explicitly in the `forward` function.

        # Remove these from the layer since we are not planning to train them separately.
        # (The loss and accuracy will be those of the DeepPatchBasedNetwork).
        new_layer.loss = None
        new_layer.train_accuracy = None
        new_layer.validate_accuracy = None

        self.layers.append(new_layer)

    def forward(self, x: torch.Tensor):
        """Performs a forward pass.

        Args:
            x: The input tensor.

        Returns:
            The output of the model, which is logits for the different classes.
        """
        assert len(self.layers) > 0, 'Can not perform a forward-pass until at least one layer is added.'

        logits = list()
        for layer in self.layers:
            x = layer(x)
            if self.args.patches.sum_logits:
                logits.append(layer.linear(x.flatten(start_dim=1)))

        if self.args.patches.sum_logits:
            return sum(logits)
        else:
            return self.layers[-1].linear(x.flatten(start_dim=1))

    def shared_step(self, batch: Tuple[torch.Tensor, torch.Tensor], stage: RunningStage):
        """Performs train/validation step, depending on the given `stage`.

        Args:
            batch: The batch to process (containing a tuple of tensors - inputs and labels).
            stage: Indicating if this is a training-step or a validation-step.

        Returns:
            The loss.
        """
        x, labels = batch
        logits = self(x)
        loss = self.loss(logits, labels)

        accuracy = self.accuracy[stage]
        accuracy(logits, labels)

        self.log(f'{stage.value}_loss', loss)
        self.log(f'{stage.value}_accuracy', accuracy)

        return loss

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Performs a training step.

        Args:
            batch: The batch to process (containing a tuple of tensors - inputs and labels).
            batch_idx: The index of the batch in the dataset.

        Returns:
            The loss.
        """
        loss = self.shared_step(batch, RunningStage.TRAINING)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Performs a validation step.

        Args:
            batch: The batch to process (containing a tuple of tensors - inputs and labels).
            batch_idx: The index of the batch in the dataset.
        """
        self.shared_step(batch, RunningStage.VALIDATING)

    def configure_optimizers(self):
        """Configure the optimizer and the learning-rate scheduler for the training process.

        Note that only the last PatchBasedModel parameters are being optimized, everything else is frozen.

        Returns:
            A dictionary containing the optimizer and learning-rate scheduler.
        """
        args: OptimizationArgs = self.layers[-1].args.opt
        optimizer = torch.optim.SGD(self.layers[-1].parameters(),
                                    args.learning_rate,
                                    args.momentum,
                                    weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=args.learning_rate_decay_steps,
                                                         gamma=args.learning_rate_decay_gamma)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}


class DataModule(LightningDataModule):
    def __init__(self, args: DataArgs, batch_size: int):
        """A datamodule to be used with PyTorch Lightning modules.

        Args:
            args: The data's arguments-schema.
            batch_size: The batch-size.
        """
        super().__init__()

        self.dataset_class = self.get_dataset_class(args.dataset_name)
        self.n_channels = args.n_channels
        self.spatial_size = args.spatial_size
        self.data_dir = args.data_dir
        self.batch_size = batch_size

        transforms_list_no_aug, transforms_list_with_aug = DataModule.get_transforms_lists(args)
        self.transforms = {'aug': Compose(transforms_list_with_aug),
                           'no_aug': Compose(transforms_list_no_aug),
                           'clean': ToTensor()}
        self.datasets = {f'{stage}_{aug}': None
                         for stage in ('fit', 'validate')
                         for aug in ('aug', 'no_aug', 'clean')}

    def get_dataset_class(self, dataset_name: str) -> Union[Type[MNIST],
                                                            Type[FashionMNIST],
                                                            Type[CIFAR10],
                                                            Type[CIFAR100]]:
        """Gets the class of the dataset, according to the given dataset name.

        Args:
            dataset_name: name of the dataset (CIFAR10, MNIST or FashionMNIST).

        Returns:
            The dataset class.
        """
        if dataset_name == 'CIFAR10':
            return CIFAR10
        elif dataset_name == 'MNIST':
            return MNIST
        elif dataset_name == 'FashionMNIST':
            return FashionMNIST
        elif dataset_name == 'CIFAR100':
            return CIFAR100
        else:
            raise NotImplementedError(f'Dataset {dataset_name} is not implemented.')

    @staticmethod
    def get_transforms_lists(args: DataArgs) -> Tuple[list, list]:
        """Gets the transformations list to be used in the dataloader.

        Args:
            args: The data's arguments-schema.

        Returns:
            One list is the transformations without augmentation,
            and the other is the transformations with augmentations.
        """
        augmentations = DataModule.get_augmentations_transforms(args.random_horizontal_flip,
                                                                args.random_crop,
                                                                args.spatial_size)
        normalization = DataModule.get_normalization_transform(args.normalization_to_plus_minus_one,
                                                               args.normalization_to_unit_gaussian,
                                                               args.n_channels)
        normalizations_list = list() if (normalization is None) else [normalization]
        crucial_transforms = [ToTensor()]
        post_transforms = [ShufflePixels(args.keep_rgb_triplets_intact)] if args.shuffle_images else list()
        transforms_list_no_aug = crucial_transforms + normalizations_list + post_transforms
        transforms_list_with_aug = augmentations + crucial_transforms + normalizations_list + post_transforms

        return transforms_list_no_aug, transforms_list_with_aug

    @staticmethod
    def get_augmentations_transforms(random_flip: bool, random_crop: bool, spatial_size: int) -> list:
        """Gets the augmentations transformations list to be used in the dataloader.

        Args:
            random_flip: Whether to use random-flip augmentation.
            random_crop: Whether to use random-crop augmentation.
            spatial_size: The spatial-size of the input images (needed for the target-size of the random-crop).

        Returns:
            A list containing the augmentations transformations.
        """
        augmentations_transforms = list()

        if random_flip:
            augmentations_transforms.append(RandomHorizontalFlip())
        if random_crop:
            augmentations_transforms.append(RandomCrop(size=spatial_size, padding=4))

        return augmentations_transforms

    @staticmethod
    def get_normalization_transform(plus_minus_one: bool, unit_gaussian: bool, n_channels: int) -> Optional[Normalize]:
        """Gets the normalization transformation to be used in the dataloader (or None, if no normalization is needed).

        Args:
            plus_minus_one: Whether to normalize the input-images from [0,1] to [-1,+1].
            unit_gaussian: Whether to normalize the input-images to have zero mean and std one (channels-wise).
            n_channels: Number of input-channels for each input-image (3 for CIFAR10, 1 for MNIST/FashionMNIST).

        Returns:
            The normalization transformation (or None, if no normalization is needed).
        """
        assert not (plus_minus_one and unit_gaussian), 'Only one should be given'

        if unit_gaussian:
            if n_channels != 3:
                raise NotImplementedError('Normalization for MNIST / FashionMNIST is not supported. ')
            normalization_values = [(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)]
        elif plus_minus_one:
            normalization_values = [(0.5,) * n_channels] * 2  # times 2 because one is mean and one is std
        else:
            return None

        return Normalize(*normalization_values)

    def prepare_data(self):
        """Download the dataset if it's not already in `self.data_dir`.
        """
        for train_mode in [True, False]:
            self.dataset_class(self.data_dir, train=train_mode, download=True)

    def setup(self, stage: Optional[str] = None):
        """Create the different datasets.
        """
        if stage is None:
            return

        for s in ('fit', 'validate'):
            for aug in ('aug', 'no_aug', 'clean'):
                k = f'{s}_{aug}'
                if self.datasets[k] is None:
                    self.datasets[k] = self.dataset_class(self.data_dir,
                                                          train=(s == 'fit'),
                                                          transform=self.transforms[aug])

    def train_dataloader(self):
        """
        Returns:
             The train dataloader, which is the train-data with augmentations.
        """
        return DataLoader(self.datasets['fit_aug'], batch_size=self.batch_size, num_workers=4, shuffle=True)

    def train_dataloader_no_aug(self):
        """
        Returns:
             The train dataloader without augmentations.
        """
        return DataLoader(self.datasets['fit_no_aug'], batch_size=self.batch_size, num_workers=4, shuffle=True)

    def train_dataloader_clean(self):
        """
        Returns:
             The train dataloader without augmentations and normalizations (i.e. the original images in [0,1]).
        """
        return DataLoader(self.datasets['fit_clean'], batch_size=self.batch_size, num_workers=4, shuffle=True)

    def val_dataloader(self):
        """
        Returns:
             The validation dataloader, which is the validation-data without augmentations
             (but possibly has normalization, if the training-dataloader has one).
        """
        return DataLoader(self.datasets['validate_no_aug'], batch_size=self.batch_size, num_workers=4)

    def val_dataloader_clean(self):
        """
        Returns:
             The train dataloader without augmentations and normalizations (i.e. the original images in [0,1]).
        """
        return DataLoader(self.datasets['validate_clean'], batch_size=self.batch_size, num_workers=4)


class ShufflePixels:
    def __init__(self, keep_rgb_triplets_intact: bool = True):
        """A data transformation which shuffles the pixels of the input image.

        Args:
            keep_rgb_triplets_intact: If it's true, shuffle the RGB triplets and not each value separately.
        """
        self.keep_rgb_triplets_intact = keep_rgb_triplets_intact

    def __call__(self, img):
        assert img.ndim == 3 and img.shape[0] == 3, "The input-image is expected to be of shape 3 x H x W"
        start_dim = 1 if self.keep_rgb_triplets_intact else 0
        img_flat = torch.flatten(img, start_dim=start_dim)
        permutation = torch.randperm(img_flat.shape[-1])
        permuted_img_flat = img_flat[..., permutation]
        permuted_img = torch.reshape(permuted_img_flat, shape=img.shape)
        return permuted_img


class VGG(pl.LightningModule):
    def __init__(self, arch_args: ArchitectureArgs, opt_args: OptimizationArgs, data_args: DataArgs):
        """A basic CNN, based on the VGG architecture (and some variants).

        Args:
            arch_args: The arguments for the architecture.
            opt_args: The arguments for the optimization process.
            data_args: The arguments for the input data.
        """
        super(VGG, self).__init__()
        layers, n_features = get_vgg_blocks(configs[arch_args.model_name],
                                            data_args.n_channels,
                                            data_args.spatial_size,
                                            arch_args.kernel_size,
                                            arch_args.padding,
                                            arch_args.use_batch_norm,
                                            arch_args.bottle_neck_dimension)
        self.features = nn.Sequential(*layers)
        self.mlp = get_mlp(input_dim=n_features,
                           output_dim=N_CLASSES,
                           n_hidden_layers=arch_args.final_mlp_n_hidden_layers,
                           hidden_dim=arch_args.final_mlp_hidden_dim,
                           use_batch_norm=arch_args.use_batch_norm,
                           organize_as_blocks=True)
        self.loss = torch.nn.CrossEntropyLoss()

        self.arch_args: ArchitectureArgs = arch_args
        self.opt_args: OptimizationArgs = opt_args

        self.save_hyperparameters(arch_args.dict())
        self.save_hyperparameters(opt_args.dict())
        self.save_hyperparameters(data_args.dict())

        self.num_blocks = len(self.features) + len(self.mlp)

        self.train_accuracy = tm.Accuracy()
        self.validate_accuracy = tm.Accuracy()
        self.accuracy = {RunningStage.TRAINING: self.train_accuracy,
                         RunningStage.VALIDATING: self.validate_accuracy}

        self.kernel_sizes: List[int] = self.init_kernel_sizes()
        self.shapes: List[tuple] = self.init_shapes()

    def forward(self, x: torch.Tensor):
        """Performs a forward pass.

        Args:
            x: The input tensor.

        Returns:
            The output of the model, which is logits for the different classes.
        """
        features = self.features(x)
        logits = self.mlp(features.flatten(start_dim=1))
        return logits

    def shared_step(self, batch: Tuple[torch.Tensor, torch.Tensor], stage: RunningStage):
        """Performs train/validation step, depending on the given `stage`.

        Args:
            batch: The batch to process (containing a tuple of tensors - inputs and labels).
            stage: Indicating if this is a training-step or a validation-step.

        Returns:
            The loss.
        """
        inputs, labels = batch
        logits = self(inputs)
        loss = self.loss(logits, labels)

        self.accuracy[stage](logits, labels)

        self.log(f'{stage.value}_loss', loss)
        self.log(f'{stage.value}_accuracy', self.accuracy[stage])

        return loss

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Performs a training step.

        Args:
            batch: The batch to process (containing a tuple of tensors - inputs and labels).
            batch_idx: The index of the batch in the dataset.

        Returns:
            The loss.
        """
        loss = self.shared_step(batch, RunningStage.TRAINING)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Performs a validation step.

        Args:
            batch: The batch to process (containing a tuple of tensors - inputs and labels).
            batch_idx: The index of the batch in the dataset.
        """
        self.shared_step(batch, RunningStage.VALIDATING)

    def get_sub_model(self, i: int) -> nn.Sequential:
        """Extracts a sub-model up to the given layer index.

        Args:
            i: The maximal index to take in the sub-model

        Returns:
            The sub-model.
        """
        if i < len(self.features):
            sub_model = self.features[:i]
        else:
            j = len(self.features) - i  # This is the index in the mlp
            sub_model = nn.Sequential(*(list(self.features) + list(self.mlp[:j])))

        return sub_model

    def configure_optimizers(self):
        """Configure the optimizer and the learning-rate scheduler for the training process.

        Returns:
            A dictionary containing the optimizer and learning-rate scheduler.
        """
        optimizer = torch.optim.SGD(self.parameters(),
                                    self.opt_args.learning_rate,
                                    self.opt_args.momentum,
                                    weight_decay=self.opt_args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=self.opt_args.learning_rate_decay_steps,
                                                         gamma=self.opt_args.learning_rate_decay_gamma)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def init_kernel_sizes(self) -> List[int]:
        """Initialize the kernel-size of each block in the model.

        Returns:
            A list of integers with the same length as `self.features`,
            where the i-th element is the kernel size of the i-th block.
        """
        kernel_sizes = list()
        for i in range(len(self.features)):
            kernel_size = get_vgg_model_kernel_size(self, i)
            if isinstance(kernel_size, tuple):
                assert kernel_size[0] == kernel_size[1], "Only square patches are supported"
                kernel_size = kernel_size[0]
            kernel_sizes.append(kernel_size)
        return kernel_sizes

    @torch.no_grad()
    def init_shapes(self) -> List[Tuple[int, int, int]]:
        """Initialize the input shapes of each block in the model.

        Returns:
            A list of shapes, with the same length as the sum of `self.features` and `self.mlp`.
        """
        shapes = list()

        dataloader = get_dataloaders(batch_size=8)["train"]
        x, _ = next(iter(dataloader))
        x = x.to(self.device)

        for block in self.features:
            shapes.append(tuple(x.shape[1:]))
            x = block(x)

        x = x.flatten(start_dim=1)

        for block in self.mlp:
            shapes.append(tuple(x.shape[1:]))
            x = block(x)

        shapes.append(tuple(x.shape[1:]))  # This is the output shape

        return shapes


class MLP(pl.LightningModule):
    def __init__(self, arch_args: ArchitectureArgs, opt_args: OptimizationArgs, data_args: DataArgs):
        """A basic MLP, which consists of multiple linear layers with ReLU in-between.

        Args:
            arch_args: The arguments for the architecture.
            opt_args: The arguments for the optimization process.
            data_args: The arguments for the input data.
        """
        super(MLP, self).__init__()
        self.input_dim = data_args.n_channels * data_args.spatial_size ** 2
        self.output_dim = N_CLASSES
        self.n_hidden_layers = arch_args.final_mlp_n_hidden_layers
        self.hidden_dim = arch_args.final_mlp_hidden_dim
        self.mlp = get_mlp(self.input_dim, self.output_dim, self.n_hidden_layers, self.hidden_dim,
                           use_batch_norm=True, organize_as_blocks=True)
        self.loss = torch.nn.CrossEntropyLoss()

        self.arch_args = arch_args
        self.opt_args = opt_args
        self.save_hyperparameters(arch_args.dict())
        self.save_hyperparameters(opt_args.dict())
        self.save_hyperparameters(data_args.dict())

        self.num_blocks = len(self.mlp)

        # Apparently the Metrics must be an attribute of the LightningModule, and not inside a dictionary.
        # This is why we have to set them separately here and then the dictionary will map to the attributes.
        self.train_accuracy = tm.Accuracy()
        self.validate_accuracy = tm.Accuracy()
        self.accuracy = {RunningStage.TRAINING: self.train_accuracy,
                         RunningStage.VALIDATING: self.validate_accuracy}

    def forward(self, x: torch.Tensor):
        """Performs a forward pass.

        Args:
            x: The input tensor.

        Returns:
            The output of the model, which is logits for the different classes.
        """
        return self.mlp(x)

    def shared_step(self, batch, stage: RunningStage):
        """Performs train/validation step, depending on the given `stage`.

        Args:
            batch: The batch to process (containing a tuple of tensors - inputs and labels).
            stage: Indicating if this is a training-step or a validation-step.

        Returns:
            The loss.
        """
        inputs, labels = batch
        logits = self(inputs)
        loss = self.loss(logits, labels)

        self.accuracy[stage](logits, labels)

        self.log(f'{stage.value}_loss', loss)
        self.log(f'{stage.value}_accuracy', self.accuracy[stage])

        return loss

    def training_step(self, batch, batch_idx):
        """Performs a training step.

        Args:
            batch: The batch to process (containing a tuple of tensors - inputs and labels).
            batch_idx: The index of the batch in the dataset.

        Returns:
            The loss.
        """
        loss = self.shared_step(batch, RunningStage.TRAINING)
        return loss

    def validation_step(self, batch, batch_idx):
        """Performs a validation step.

        Args:
            batch: The batch to process (containing a tuple of tensors - inputs and labels).
            batch_idx: The index of the batch in the dataset.
        """
        self.shared_step(batch, RunningStage.VALIDATING)

    def get_sub_model(self, i: int) -> nn.Sequential:
        """Extracts a sub-model up to the given layer index.

        Args:
            i: The maximal index to take in the sub-model

        Returns:
            The sub-model.
        """
        return self.mlp[:i]

    def configure_optimizers(self):
        """Configure the optimizer and the learning-rate scheduler for the training process.

        Returns:
            A dictionary containing the optimizer and learning-rate scheduler.
        """
        optimizer = torch.optim.SGD(self.parameters(),
                                    self.opt_args.learning_rate,
                                    self.opt_args.momentum,
                                    weight_decay=self.opt_args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=self.opt_args.learning_rate_decay_steps,
                                                         gamma=self.opt_args.learning_rate_decay_gamma)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}


def initialize_model(args: Args) -> Union[VGG, MLP]:
    """
    Returns:
        The VGG / MLP model, initialized according to the given arguments-schema.
    """
    model_class = VGG if args.arch.model_name.startswith('VGG') else MLP
    model = model_class(args.arch, args.opt, args.data)
    return model


def initialize_wandb_logger(args: Args, name_suffix: str = '') -> WandbLogger:
    """
    Returns:
        The wandb logger, logging to a project named `wandb_project_name` 
        and a run named `wandb_run_name` with the given suffix.
    """
    return WandbLogger(
        project=args.env.wandb_project_name,
        config=args.flattened_dict(),
        name=(args.env.wandb_run_name + name_suffix) if (args.env.wandb_run_name is not None) else name_suffix,
        log_model=True
    )


def initialize_trainer(args: Args, wandb_logger: WandbLogger):
    """
    Returns:
        The trainer, logging to the given wandb logger, 
        with callbacks saving checkpoints and printing the model's summary.
    """
    checkpoint_callback = ModelCheckpoint(monitor='validate_accuracy', mode='max')
    model_summary_callback = ModelSummary(max_depth=3)

    if isinstance(args.env.multi_gpu, list) or (args.env.multi_gpu != 0):
        trainer_kwargs = dict(gpus=args.env.multi_gpu, strategy="dp")
    else:
        trainer_kwargs = dict(gpus=[args.env.device_num]) if args.env.is_cuda else dict()

    if args.env.debug:
        trainer_kwargs.update({f'limit_{t}_batches': 3 for t in ['train', 'val']})
        trainer_kwargs.update({'log_every_n_steps': 1})

    return pl.Trainer(logger=wandb_logger, 
                      callbacks=[checkpoint_callback, model_summary_callback], 
                      max_epochs=args.opt.epochs,
                      **trainer_kwargs)


def initialize_datamodule(args: DataArgs, batch_size: int) -> DataModule:
    """
    Returns:
        data-module used for the training process, with the given batch_size.
    """
    datamodule = DataModule(args, batch_size)
    datamodule.prepare_data()
    datamodule.setup(stage='fit')
    datamodule.setup(stage='validate')

    return datamodule


def get_dataloader_for_patches_sampling(args: Args, datamodule: DataModule) -> DataLoader:
    """
    Returns:
        The data-loader used for patches sampling (resulting in the patches-dictionary used by the patch-based-model).
    """
    if args.patches.sample_patches_from_original_zero_one_values:
        dataloader = datamodule.train_dataloader_clean()
    else:
        dataloader = datamodule.train_dataloader_no_aug()

    return dataloader


def unwatch_model(model: nn.Module):
    """Unwatch a model, to be watched in the next training-iteration.

    Prevents wandb error which they have a TO-DO for fixing in wandb/sdk/wandb_watch.py:123
    "  TO-DO: we should also remove recursively model._wandb_watch_called  "
    ValueError: You can only call `wandb.watch` once per model.
    Pass a new instance of the model if you need to call wandb.watch again in your code.

    Args:
        model: The model to unwatch.
    """
    wandb.unwatch(model)
    for module in model.modules():
        if hasattr(module, "_wandb_watch_called"):
            delattr(module, "_wandb_watch_called")
    if hasattr(model, "_wandb_watch_called"):
        delattr(model, "_wandb_watch_called")
    wandb.finish()


def train_patch_based_network(args: Args, dataloader: DataLoader, wandb_logger: WandbLogger, datamodule: DataModule):
    """Train a patch-based network.

    Args:
        args: The arguments-schema.
        dataloader: The data-loader for patches sampling, obtaining the patches-dictionary.
        wandb_logger: The wandb logger to use during training.
        datamodule: The data-module to use during training.
    """
    model = PatchBasedNetwork(args)
    if not args.patches.replace_embedding_with_regular_conv_relu:
        model.to(args.env.device)
        model.calculate_embedding_from_data(dataloader)
        model.cpu()
    wandb_logger.watch(model, log='all')
    trainer = initialize_trainer(args, wandb_logger)
    trainer.fit(model, datamodule=datamodule)


def train_deep_patch_based_network(args: Args, dataloader: DataLoader, datamodule: DataModule):
    """Train a deep patch-based network.

    Args:
        args: The arguments-schema.
        dataloader: The data-loader for patches sampling, obtaining the patches-dictionary.
        datamodule: The data-module to use during training.
    """
    model = DeepPatchBasedNetwork(args)
    for i in range(args.patches.depth):
        model.add_layer(dataloader)
        wandb_logger = initialize_wandb_logger(args, name_suffix=f'_layer_{i + 1}')
        wandb_logger.watch(model, log='all')
        trainer = initialize_trainer(args, wandb_logger)
        trainer.fit(model, datamodule=datamodule)
        unwatch_model(model)


def train_model(args: Args, wandb_logger: WandbLogger, datamodule: DataModule):
    """Train a 'regular' VGG / MLP model

    Args:
        args: The arguments-schema.
        wandb_logger: The wandb logger to use during training.
        datamodule: The data-module to use during training.
    """
    model = initialize_model(args)
    wandb_logger.watch(model, log='all')
    trainer = initialize_trainer(args, wandb_logger)
    trainer.fit(model, datamodule=datamodule)


def main():
    """The main function running everything.
    """
    args: Args = get_args(args_class=Args)
    datamodule = initialize_datamodule(args.data, args.opt.batch_size)
    wandb_logger = initialize_wandb_logger(args)

    if args.patches.train_locally_linear_network:
        dataloader = get_dataloader_for_patches_sampling(args, datamodule)
        if args.patches.depth == 1:
            train_patch_based_network(args, dataloader, wandb_logger, datamodule)
        else:
            train_deep_patch_based_network(args, dataloader, datamodule)
    else:
        train_model(args, wandb_logger, datamodule)


if __name__ == '__main__':
    main()
