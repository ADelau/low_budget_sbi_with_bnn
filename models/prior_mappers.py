import math
import os
from itertools import cycle
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import wandb
from autoclip.torch import QuantileClip
from lampe.data import H5Dataset
from torch import Tensor

from .np_priors import GPPrior
from .stein_gradient_estimator import SpectralSteinEstimator
from .variational_distributions import (
    GaussianNNParametersDistribution,
    HierarchicalGaussianNNParametersDistribution,
)


class DatasetMeasurementGenerator:
    def __init__(self, train_set: H5Dataset) -> None:
        """Constructor.

        Parameters
        ----------
        train_set : H5Dataset
            The training set.
        """
        self.train_iterator = iter(train_set)

        self.data_iterator = cycle(self.train_iterator)

    def get(self, n_data: int) -> Tuple[Tensor, Tensor]:
        """Get items.

        Parameters
        ----------
        n_data : int
            The number of data points to get.

        Returns
        -------
        Tuple[Tensor, Tensor]
            Simulator's parameters and observations.
        """
        theta = None
        x = None

        # Create samples from the joint
        while (
            x is None
            or theta is None
            or x.shape[0] < n_data / 2
            or theta.shape[0] < n_data / 2
        ):
            theta_cur, x_cur = next(self.data_iterator)

            if theta is None:
                theta = theta_cur
            else:
                theta = torch.cat((theta, theta_cur), dim=0)

            if x is None:
                x = x_cur
            else:
                x = torch.cat((x, x_cur), dim=0)

        # random permute in order to not always select the same samples in a batch.
        perm = torch.randperm(x.shape[0])
        x = x[perm, ...]
        theta = theta[perm, ...]

        # Cut to only keep the required amount of samples
        x = x[: math.ceil(n_data / 2), ...]
        theta = theta[: math.ceil(n_data / 2), ...]

        # Create samples from the marginal
        theta_prime = torch.roll(theta, 1, dims=0)

        theta = torch.cat((theta, theta_prime), dim=0)
        x = torch.cat((x, x), dim=0)

        return theta, x


class UniformMeasurementGenerator:
    def __init__(
        self, theta_domain: Tuple[Tensor, Tensor], x_domain: Tuple[Tensor, Tensor]
    ) -> None:
        """Constructor

        Parameters
        ----------
        theta_domain : Tuple[Tensor, Tensor]
            Lower and upper bound of the simulator's parameters domain.
        x_domain : Tuple[Tensor, Tensor]
            Lower and upper bound of the observations domain.
        """
        self.theta_distribution = torch.distributions.uniform.Uniform(
            theta_domain[0], theta_domain[1]
        )
        self.x_distribution = torch.distributions.uniform.Uniform(
            x_domain[0], x_domain[1]
        )

    def get(self, n_data: int) -> Tuple[Tensor, Tensor]:
        """Get items.

        Parameters
        ----------
        n_data : int
            The number of data points to get.

        Returns
        -------
        Tuple[Tensor, Tensor]
            Simulator's parameters and observations.
        """
        theta = self.theta_distribution.rsample((n_data,))
        x = self.x_distribution.rsample((n_data,))

        return theta, x


class HybridMeasurementGenerator:
    def __init__(
        self, theta_domain: Tuple[Tensor, Tensor], train_set: H5Dataset
    ) -> None:
        """Constructor

        Parameters
        ----------
        theta_domain : Tuple[Tensor, Tensor]
            Lower and upper bound of the simulator's parameters domain.
        train_set : H5Dataset
            Training set.
        """
        self.theta_distribution = torch.distributions.uniform.Uniform(
            theta_domain[0], theta_domain[1]
        )

        self.train_iterator = iter(train_set)

        self.data_iterator = cycle(self.train_iterator)

    def get(self, n_data: int) -> Tuple[Tensor, Tensor]:
        """Get items.

        Parameters
        ----------
        n_data : int
            The number of data points to get.

        Returns
        -------
        Tuple[Tensor, Tensor]
            Simulator's parameters and observations.
        """
        theta = self.theta_distribution.rsample((n_data,))

        x = None
        while x is None or x.shape[0] < n_data:
            _, x_cur = next(self.data_iterator)

            if x is None:
                x = x_cur
            else:
                x = torch.cat((x, x_cur), dim=0)

        # random permute in order to not always select the same samples in a batch.
        perm = torch.randperm(x.shape[0])
        x = x[perm, ...]

        # Cut to only keep the required amount of samples
        x = x[:n_data, ...]

        return theta, x


def weights_init(m):
    """Reproduced from https://github.com/tranbahien/you-need-a-good-prior"""
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.normal_(m.bias.data)


class SteinKLDistance:
    def __init__(
        self,
        np_model: GPPrior,
        output_dim: int,
        num_eigs: int,
        eta: float,
        gpu_np_model: bool = False,
        device: str = "cpu",
        joint_entropy: bool = False,
        loss_divider: float = 1.0,
    ) -> None:
        """Constructor

        Parameters
        ----------
        np_model : GPPrior
            The non parametric model.
        output_dim : int
           The output dimension.
        num_eigs : int
            Number of eigen values to use in the SSGE method.
        eta : float
            Hyperparameter of the SSGE method
        gpu_np_model : bool, optional
            The device of the non parametric model, by default False
        device : str, optional
            The device on which to run the computations, by default "cpu"
        joint_entropy : bool, optional
            Whether to estimator the entropy jointly, by default False
        loss_divider : float, optional
            Divide the loss by this factor, by default 1.
        """
        self.np_model = np_model
        self.output_dim = output_dim
        self.gpu_np_model = gpu_np_model
        self.device = device
        self.score_estimator = SpectralSteinEstimator(eta, num_eigs)
        self.joint_entropy = joint_entropy
        self.loss_divider = loss_divider

    def calculate(self, theta: Tensor, x: Tensor, nnet_samples: Tensor) -> Tensor:
        """Calculate the distance

        Parameters
        ----------
        theta : Tensor
            The simulator's parameters.
        x : Tensor
            The observations.
        nnet_samples : Tensor
            The function output samples from the neural network distribution.

        Returns
        -------
        Tensor
            The distance.
        """
        n_data = nnet_samples.shape[1]

        #  It was of size: [n_func, n_data, n_out]
        # will be of size: [n_data, n_func, n_out]
        nnet_samples = nnet_samples.transpose(0, 1)
        cross_entropy = 0.0
        for dim in range(self.output_dim):
            log_probs = self.np_model.functions_log_prob(
                theta, x, nnet_samples[:, :, dim]
            )

            cross_entropy -= log_probs.mean()

        entropy = 0.0

        if self.joint_entropy:
            #  It was of size: [n_data, n_func, n_out]
            # will be of size: [n_func, n_data, n_out]
            nnet_samples = nnet_samples.transpose(0, 1)
            for dim in range(self.output_dim):
                with torch.no_grad():
                    dlog_q = self.score_estimator(nnet_samples[:, :, dim])
                scores = -dlog_q * nnet_samples[:, :, dim]
                entropy += scores.sum(dim=1).mean()
        else:
            for data_index in range(nnet_samples.shape[0]):
                for dim in range(self.output_dim):
                    with torch.no_grad():
                        dlog_q = self.score_estimator(
                            nnet_samples[data_index, :, dim].unsqueeze(1)
                        )
                    scores = -dlog_q * nnet_samples[data_index, :, dim].unsqueeze(1)
                    scores = scores.squeeze()
                    entropy += scores.mean()

        entropy = entropy / (n_data * self.loss_divider)
        cross_entropy = cross_entropy / (n_data * self.loss_divider)
        print("entropy = {} cross_entropy = {}".format(entropy, cross_entropy))
        return -entropy + cross_entropy


class PriorMapper(object):
    def __init__(
        self,
        np_model: GPPrior,
        bnn: Union[
            GaussianNNParametersDistribution,
            HierarchicalGaussianNNParametersDistribution,
        ],
        data_generator: Union[
            DatasetMeasurementGenerator,
            UniformMeasurementGenerator,
            HybridMeasurementGenerator,
        ],
        out_dir: str,
        log_space: bool,
        output_dim: int = 1,
        n_data: int = 256,
        gpu_np_model: bool = False,
        logger: bool = None,
        device: str = "cpu",
        clip_gradient: bool = False,
        clipping_norm: Optional[float] = None,
        clipping_quantile: Optional[float] = None,
        use_wandb: bool = False,
        validation_data_generator: Optional[
            Union[
                DatasetMeasurementGenerator,
                UniformMeasurementGenerator,
                HybridMeasurementGenerator,
            ]
        ] = None,
        nb_val_steps: Optional[int] = None,
        val_steps_every: Optional[int] = None,
        schedule_lr: bool = False,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        np_model : GPPrior
            The non parametric model.
        bnn : Union[GaussianNNParametersDistribution, HierarchicalGaussianNNParametersDistribution]
            The bayesian neural network
        data_generator : Union[DatasetMeasurementGenerator, UniformMeasurementGenerator, HybridMeasurementGenerator]
            The measurement data generator
        out_dir : str
            directory to save debugging info
        log_space : bool
            Whether the models make prediction in log space
        output_dim : int, optional
            The output dim, by default 1
        n_data : int, optional
            The number of measurement samples per iteration, by default 256
        gpu_np_model : bool, optional
            Whether the non parametric model is on gpu, by default False
        logger : bool, optional
            The logger, by default None
        device : str, optional
            The device on which to run the computations, by default "cpu"
        clip_gradient : bool, optional
            Whether to clip the gradient, by default False
        clipping_norm : Optional[float], optional
            The gradient clipping norm, by default None
        clipping_quantile : Optional[float], optional
            The gradient clipping quantile, by default None
        use_wandb : bool, optional
            Whether to use wandb, by default False
        validation_data_generator : Optional[Union[DatasetMeasurementGenerator, UniformMeasurementGenerator, HybridMeasurementGenerator]], optional
            The validation measurement data generator, by default None
        nb_val_steps : Optional[int], optional
            The number of steps for validation, by default None
        val_steps_every : Optional[int], optional
            The number of training steps between each validation, by default None
        schedule_lr : bool, optional
            Whether to schedule the learning rate, by default False
        """

        self.np_model = np_model
        self.bnn = bnn
        self.data_generator = data_generator
        self.n_data = n_data
        self.output_dim = output_dim
        self.out_dir = out_dir
        self.log_space = log_space
        self.device = device
        self.gpu_np_model = gpu_np_model
        self.clip_gradient = clip_gradient
        self.clipping_norm = clipping_norm
        self.clipping_quantile = clipping_quantile
        self.use_wandb = use_wandb
        self.validation_data_generator = validation_data_generator
        self.nb_val_steps = nb_val_steps
        self.val_steps_every = val_steps_every
        self.schedule_lr = schedule_lr

        # Move models to configured device
        if gpu_np_model:
            self.np_model = self.np_model.to(self.device)
        self.bnn = self.bnn.to(self.device)

        # Setup logger
        self.print_info = (
            lambda x: print(x, flush=True) if logger is None else logger.info
        )

    def compute_mmd_given_samples(
        self, nnet_samples: Tensor, np_samples: Tensor
    ) -> float:
        """Compute the MMD discrepency.

        Parameters
        ----------
        nnet_samples : Tensor
            The function samples from the neural network
        np_samples : Tensor
            The function samples from the non parametric model.

        Returns
        -------
        float
            The MMD discrepency.
        """
        # Samples of shape [n samples, n_measurements]
        number_samples = math.floor(nnet_samples.shape[0] / 2)

        nnet_samples_1 = nnet_samples[:number_samples, ...]
        nnet_samples_2 = nnet_samples[number_samples:, ...]
        np_samples_1 = np_samples[:number_samples, ...]
        np_samples_2 = np_samples[number_samples:, ...]

        lengthscale = math.sqrt(nnet_samples_1.shape[1])

        def rbf_kernel(p_samples, q_samples):
            # Samples of shape [n samples, n_measurements]
            return torch.exp(
                -torch.sum((p_samples - q_samples) ** 2, dim=1) / (2 * lengthscale**2)
            )
            # Return Tensor of shape [n_samples]

        term_1 = torch.mean(rbf_kernel(np_samples_1, np_samples_2)).item()
        term_2 = torch.mean(rbf_kernel(nnet_samples_1, nnet_samples_2)).item()
        term_3 = torch.mean(rbf_kernel(nnet_samples_1, np_samples_1)).item()

        return term_1 + term_2 - 2 * term_3

    def compute_mmd(self, theta: Tensor, x: Tensor, n_samples: int) -> float:
        """ompute the MMD discrepency.

        Parameters
        ----------
        theta : Tensor
            The simulator's parameters.
        x : Tensor
            The observations
        n_samples : int
            The numbe of samples to use.

        Returns
        -------
        float
           The MMD discrepency.
        """
        with torch.no_grad():
            # theta, x = self.data_generator.get(n_data)

            x = x.to(self.device)
            theta = theta.to(self.device)
            if not self.gpu_np_model:
                x = x.to("cpu")
                theta = theta.to("cpu")

            np_samples = (
                self.np_model.sample_functions(theta, x, n_samples)
                .detach()
                .float()
                .to(self.device)
            )
            if self.output_dim > 1:
                np_samples = np_samples.squeeze()

            if not self.gpu_np_model:
                x = x.to(self.device)
                theta = theta.to(self.device)

            nnet_samples = (
                self.bnn.sample_functions(theta, x, n_samples).float().to(self.device)
            )
            if self.output_dim > 1:
                nnet_samples = nnet_samples.squeeze()

            return self.compute_mmd_given_samples(nnet_samples, np_samples)


class DistanceBasedPriorMapper(PriorMapper):
    """Adapted from https://github.com/tranbahien/you-need-a-good-prior"""

    def __init__(
        self,
        np_model: GPPrior,
        bnn: Union[
            GaussianNNParametersDistribution,
            HierarchicalGaussianNNParametersDistribution,
        ],
        data_generator: Union[
            DatasetMeasurementGenerator,
            UniformMeasurementGenerator,
            HybridMeasurementGenerator,
        ],
        out_dir: str,
        log_space: bool,
        distance_type: str,
        distance_config: dict,
        output_dim: int = 1,
        n_data: int = 256,
        gpu_np_model: bool = False,
        logger: bool = None,
        device: str = "cpu",
        clip_gradient: bool = False,
        clipping_norm: Optional[float] = None,
        clipping_quantile: Optional[float] = None,
        use_wandb: bool = False,
        validation_data_generator: Optional[
            Union[
                DatasetMeasurementGenerator,
                UniformMeasurementGenerator,
                HybridMeasurementGenerator,
            ]
        ] = None,
        nb_val_steps: Optional[int] = None,
        val_steps_every: Optional[int] = None,
        schedule_lr: bool = False,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        np_model : GPPrior
            The non parametric model.
        bnn : Union[GaussianNNParametersDistribution, HierarchicalGaussianNNParametersDistribution]
            The bayesian neural network
        data_generator : Union[DatasetMeasurementGenerator, UniformMeasurementGenerator, HybridMeasurementGenerator]
            The measurement data generator
        out_dir : str
            directory to save debugging info
        log_space : bool
            Whether the models make prediction in log space
        distance_type : str
            The distance type.
        distance_config : dict
            The distance config.
        output_dim : int, optional
            The output dim, by default 1
        n_data : int, optional
            The number of measurement samples per iteration, by default 256
        gpu_np_model : bool, optional
            Whether the non parametric model is on gpu, by default False
        logger : bool, optional
            The logger, by default None
        device : str, optional
            The device on which to run the computations, by default "cpu"
        clip_gradient : bool, optional
            Whether to clip the gradient, by default False
        clipping_norm : Optional[float], optional
            The gradient clipping norm, by default None
        clipping_quantile : Optional[float], optional
            The gradient clipping quantile, by default None
        use_wandb : bool, optional
            Whether to use wandb, by default False
        validation_data_generator : Optional[Union[DatasetMeasurementGenerator, UniformMeasurementGenerator, HybridMeasurementGenerator]], optional
            The validation measurement data generator, by default None
        nb_val_steps : Optional[int], optional
            The number of steps for validation, by default None
        val_steps_every : Optional[int], optional
            The number of training steps between each validation, by default None
        schedule_lr : bool, optional
            Whether to schedule the learning rate, by default False

        Raises
        ------
        NotImplementedError
            _description_
        """

        super().__init__(
            np_model,
            bnn,
            data_generator,
            out_dir,
            log_space,
            output_dim=output_dim,
            n_data=n_data,
            gpu_np_model=gpu_np_model,
            logger=logger,
            device=device,
            clip_gradient=clip_gradient,
            clipping_norm=clipping_norm,
            clipping_quantile=clipping_quantile,
            use_wandb=use_wandb,
            validation_data_generator=validation_data_generator,
            nb_val_steps=nb_val_steps,
            val_steps_every=val_steps_every,
            schedule_lr=schedule_lr,
        )

        self.distance_type = distance_type
        self.distance_config = distance_config

        if distance_type == "stein_KL":
            self.distance = SteinKLDistance(
                self.np_model,
                self.output_dim,
                distance_config["num_eigs"],
                distance_config["eta"],
                device=self.device,
                gpu_np_model=self.gpu_np_model,
                joint_entropy=distance_config["joint_entropy"],
                loss_divider=distance_config["loss_divider"],
            )

        else:
            raise NotImplementedError(
                "Distance version '{}' does not exist.".format(distance_type)
            )

    def optimize(
        self,
        num_iters: int,
        optimizer_params: dict,
        n_samples: int = 128,
        print_every: int = 1,
        debug: bool = False,
    ) -> Tuple[List, List, List, List, List, List]:
        """Optimize the prior

        Parameters
        ----------
        num_iters : int
            The number of iterations to perform
        optimizer_params : dict
            The optimizer parameters.
        n_samples : int, optional
            The number of samples per iteration, by default 128
        print_every : int, optional
           The numbe of iterations between each print, by default 1
        debug : bool, optional
            whether to run in debug mode, by default False

        Returns
        -------
        Tuple[List, List, List, List, List, List]
            dist_hist,
            normalized_dist_hist,
            mmd_hist,
            val_dist_hist,
            val_normalized_dist,
            val_mmd_hist,

        Raises
        ------
        NotImplementedError
            Invalid optimizer.
        """
        dist_hist = []
        normalized_dist_hist = []
        mmd_hist = []
        val_dist_hist = []
        val_normalized_dist_hist = []
        val_mmd_hist = []

        if self.validation_data_generator:
            val_data = [
                self.validation_data_generator.get(self.n_data)
                for _ in range(self.nb_val_steps)
            ]

        if optimizer_params["optimizer"] == "rmsprop":
            prior_optimizer = torch.optim.RMSprop(
                self.bnn.parameters(), lr=optimizer_params["lr"]
            )
        elif optimizer_params["optimizer"] == "adam":
            prior_optimizer = torch.optim.Adam(
                self.bnn.parameters(), lr=optimizer_params["lr"]
            )
        elif optimizer_params["optimizer"] == "adamw":
            prior_optimizer = torch.optim.AdamW(
                self.bnn.parameters(),
                lr=optimizer_params["lr"],
                weight_decay=optimizer_params["weight_decay"],
            )
        elif optimizer_params["optimizer"] == "sgd":
            prior_optimizer = torch.optim.SGD(
                self.bnn.parameters(),
                lr=optimizer_params["lr"],
                momentum=optimizer_params["init_momentum"],
            )
        else:
            raise NotImplementedError("Invalid optimizer.")

        if self.schedule_lr:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                prior_optimizer, patience=1, verbose=True
            )

        if self.clip_gradient == "quantile":
            prior_optimizer = QuantileClip.as_optimizer(
                optimizer=prior_optimizer,
                quantile=self.clipping_quantile,
                history_length=100,
                global_threshold=True,
            )

        # Prior loop
        for it in range(1, num_iters + 1):
            wandb_log_dict = {}

            # Draw X
            theta, x = self.data_generator.get(self.n_data)
            x = x.to(self.device)
            theta = theta.to(self.device)
            if not self.gpu_np_model:
                x = x.to("cpu")
                theta = theta.to("cpu")

            # Draw functions from NP model
            np_samples = (
                self.np_model.sample_functions(theta, x, n_samples)
                .detach()
                .float()
                .to(self.device)
            )
            if self.output_dim > 1:
                np_samples = np_samples.squeeze()

            if not self.gpu_np_model:
                x = x.to(self.device)
                theta = theta.to(self.device)

            # Draw functions from BNN
            nnet_samples = (
                self.bnn.sample_functions(theta, x, n_samples).float().to(self.device)
            )
            if self.output_dim > 1:
                nnet_samples = nnet_samples.squeeze()

            if not self.log_space:
                nnet_samples = nnet_samples.exp()

            #  It was of size: [n_data, n_func, n_out]
            # will be of size: [n_func, n_data, n_out]
            nnet_samples = nnet_samples.transpose(0, 1)
            np_samples = np_samples.transpose(0, 1)

            dist = self.distance.calculate(theta, x, nnet_samples)

            dist.backward()

            if self.clip_gradient == "threshold":
                grad_norm = nn.utils.clip_grad_norm_(
                    self.bnn.parameters(), self.clipping_norm, error_if_nonfinite=True
                )

            prior_optimizer.step()

            with torch.no_grad():
                mmd = self.compute_mmd_given_samples(nnet_samples, np_samples)

            if "loss_divider" in self.distance_config.keys():
                normalized_dist = dist * self.distance_config["loss_divider"]
            else:
                normalized_dist = dist

            dist_hist.append(float(dist))
            normalized_dist_hist.append(float(normalized_dist))
            mmd_hist.append(mmd)

            if self.use_wandb:
                wandb_log_dict["running_dist"] = dist.item()
                wandb_log_dict["running_normalized_dist"] = normalized_dist.item()
                wandb_log_dict["running_mmd"] = mmd

            if (it % print_every == 0) or it == 1:
                if self.clip_gradient == "threshold":
                    self.print_info(
                        ">>> Iteration # {:3d}: "
                        "Dist {:.4f} "
                        "MMD {:.4f} "
                        "grad norm {:.4f}".format(
                            it, float(dist), float(mmd), float(grad_norm)
                        )
                    )
                else:
                    self.print_info(
                        ">>> Iteration # {:3d}: "
                        "Dist {:.4f} "
                        "MMD {:.4f} ".format(it, float(dist), float(mmd))
                    )

            if self.validation_data_generator and it % self.val_steps_every == 0:
                val_mmd = []
                val_dist = []
                for theta, x in val_data:
                    with torch.no_grad():
                        x = x.to(self.device)
                        theta = theta.to(self.device)
                        if not self.gpu_np_model:
                            x = x.to("cpu")
                            theta = theta.to("cpu")

                        # Draw functions from NP model
                        np_samples = (
                            self.np_model.sample_functions(theta, x, n_samples)
                            .detach()
                            .float()
                            .to(self.device)
                        )
                        if self.output_dim > 1:
                            np_samples = np_samples.squeeze()

                        if not self.gpu_np_model:
                            x = x.to(self.device)
                            theta = theta.to(self.device)

                        # Draw functions from BNN
                        nnet_samples = (
                            self.bnn.sample_functions(theta, x, n_samples)
                            .float()
                            .to(self.device)
                        )
                        if self.output_dim > 1:
                            nnet_samples = nnet_samples.squeeze()

                        if not self.log_space:
                            nnet_samples = nnet_samples.exp()

                        nnet_samples = nnet_samples.transpose(0, 1)
                        np_samples = np_samples.transpose(0, 1)

                        dist = self.distance.calculate(theta, x, nnet_samples)

                        mmd = self.compute_mmd_given_samples(nnet_samples, np_samples)

                    val_dist.append(dist.item())
                    val_mmd.append(mmd)

                val_dist = sum(val_dist) / len(val_dist)
                val_mmd = sum(val_mmd) / len(val_mmd)

                if "loss_divider" in self.distance_config.keys():
                    val_normalized_dist = (
                        val_dist * self.distance_config["loss_divider"]
                    )
                else:
                    val_normalized_dist = val_dist

                if self.schedule_lr:
                    scheduler.step(val_dist)

                self.print_info(
                    ">>> Validation Iteration # {:3d}: "
                    "Dist {:.4f} "
                    "MMD {:.4f} ".format(it, float(val_dist), float(val_mmd))
                )

                val_dist_hist.append(float(val_dist))
                val_normalized_dist_hist.append(float(val_normalized_dist))
                val_mmd_hist.append(val_mmd)

                if self.use_wandb:
                    wandb_log_dict["val_dist"] = val_dist
                    wandb_log_dict["val_normalized_dist"] = val_normalized_dist
                    wandb_log_dict["val_mmd"] = val_mmd

            if self.use_wandb:
                wandb.log(wandb_log_dict)

        # Save accumulated list of intermediate wasserstein values
        if debug:
            values = np.array(self.distance.values_log).reshape(-1, 1)
            path = os.path.join(self.out_dir, "dist_intermediate_values.log")
            np.savetxt(path, values, fmt="%.6e")
            self.print_info("Saved intermediate distance values in: " + path)

        if self.use_wandb:
            wandb.log({"final_val_normalized_dist": val_normalized_dist_hist[-1]})
            wandb.log({"final_val_dist": val_dist_hist[-1]})
            wandb.log({"final_val_mmd": val_mmd_hist[-1]})

        return (
            dist_hist,
            normalized_dist_hist,
            mmd_hist,
            val_dist_hist,
            val_normalized_dist,
            val_mmd_hist,
        )
