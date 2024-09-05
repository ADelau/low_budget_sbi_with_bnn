import os
import shutil
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import wandb
from lampe.data import H5Dataset
from lampe.inference import NPE, NPELoss
from torch import Tensor

from ..benchmarks import Benchmark
from .base import Model, ModelFactory
from .bayesian_methods.hmc import HMCmodel
from .bayesian_methods.vi import VImodel
from .np_priors import GPPrior
from .prior_mappers import (
    DatasetMeasurementGenerator,
    DistanceBasedPriorMapper,
    HybridMeasurementGenerator,
    UniformMeasurementGenerator,
)
from .variational_distributions import (
    GaussianNNParametersDistribution,
    HierarchicalGaussianNNParametersDistribution,
)


class BayesianNPEFactory(ModelFactory):
    """Factory instantiating BayesianNPE models."""

    def __init__(
        self,
        config: dict,
        benchmark: Benchmark,
        simulation_budget: int,
        model_class: Any = None,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        config : dict
            The config file
        benchmark : Benchmark
            The benchmark.
        simulation_budget : int
            The simulation_budget.
        model_class : Any, optional
            The class of the models to instantiate, if not specified, a BayesianNPE
            model is instantiated, by default None
        """
        if model_class is None:
            model_class = BayesianNPEModel

        config_run = config.copy()
        for idx in range(len(config["simulation_budgets"])):
            if config["simulation_budgets"][idx] == simulation_budget:
                break
        config_run["train_batch_size"] = config["train_batch_size"][idx]
        config_run["temperature"] = config["temperature"][idx]
        config_run["max_temperature"] = config["max_temperature"][idx]
        config_run["weight_decay"] = config["weight_decay"][idx]

        super().__init__(config_run, benchmark, simulation_budget, model_class)

    def get_train_time(self, benchmark_time: int, epochs: int) -> int:
        """Get the computational time required to train the model.

        Parameters
        ----------
        benchmark_time : int
            The train computational time associated to the benchmark.
        epochs : int
            The number of epochs for which to train.

        Returns
        -------
        int
            The computational time required to train the model.
        """
        return 2 * super().get_train_time(benchmark_time, epochs)

    def get_coverage_time(self, benchmark_time: int) -> int:
        """Get the computational time required to compute the coverage.

        Parameters
        ----------
        benchmark_time : int
            The coverage computational time associated to the benchmark.

        Returns
        -------
        int
            The computational time required to compute the coverage.
        """
        if self.config["bnn_method"] == "vi":
            nb_networks = self.config["nb_networks"]

        elif self.config["bnn_method"] == "hmc":
            nb_networks = self.config["samples_per_chain"] * self.config["nb_chains"]

        else:
            raise NotImplementedError(
                "bnn_method '{}' not implemented.".format(self.config["bnn_method"])
            )

        return nb_networks * super().get_coverage_time(benchmark_time)

    def get_test_time(self, benchmark_time: int) -> int:
        """Get the computational time required to test the model.

        Parameters
        ----------
        benchmark_time : int
            The test computational time associated to the benchmark.

        Returns
        -------
        int
            The computational time required to test the model.
        """
        if self.config["bnn_method"] == "vi":
            nb_networks = self.config["nb_networks"]

        elif self.config["bnn_method"] == "hmc":
            nb_networks = self.config["samples_per_chain"] * self.config["nb_chains"]

        else:
            raise NotImplementedError(
                "bnn_method '{}' not implemented.".format(self.config["bnn_method"])
            )

        return nb_networks * super().get_test_time(benchmark_time)

    def require_multiple_trainings(self) -> bool:
        """Returns whether the model required multiple trainings.

        Returns
        -------
        bool
            Whether the model required multiple trainings.
        """
        return True

    def nb_trainings_required(self) -> int:
        """Returns whether the number of training required.

        Returns
        -------
        int
            The number of training required.
        """
        if self.config["bnn_method"] == "vi":
            return 1
        elif self.config["bnn_method"] == "hmc":
            return self.config["nb_chains"]
        else:
            raise NotImplementedError(
                "bnn_method '{}' not implemented.".format(self.config["bnn_method"])
            )

    def is_trained(self, id: int) -> bool:
        """Returns whether the model is trained.

        Parameters
        ----------
        id : int
            The id of the model

        Returns
        -------
        bool
            Whether the model is trained.

        Raises
        ------
        NotImplementedError
            The bnn method is not implemented.
        """
        if self.config["bnn_method"] == "vi":
            return self.model_class.is_trained(
                self.config["bnn_method"], self.get_model_path(id)
            )
        elif self.config["bnn_method"] == "hmc":
            return self.model_class.is_trained(
                self.config["bnn_method"],
                self.get_model_path(id // self.config["nb_chains"]),
                id % self.config["nb_chains"],
            )
        else:
            raise NotImplementedError(
                "bnn_method '{}' not implemented.".format(self.config["bnn_method"])
            )


class NPEWithEmbedding(nn.Module):
    """NPE module that embeds observations first."""

    def __init__(
        self,
        npe: nn.Module,
        embedding: nn.Module,
        normalize_observation_fct: Callable,
        unnormalize_observation_fct: Callable,
        normalize_parameters_fct: Callable,
        unnormalize_parameters_fct: Callable,
        normalization_log_jacobian: float,
    ) -> None:
        """Constructor

        Parameters
        ----------
        npe : nn.Module
            The NPE module
        embedding : nn.Module
            The observations embedding module
        normalize_observation_fct : Callable
            The function used to normalize observations.
        unnormalize_observation_fct : Callable
            The function used to unnormalize observations.
        normalize_parameters_fct : Callable
            The function used to normalize simulator's parameters.
        unnormalize_parameters_fct : Callable
           The function used to unnormalize simulator's parameters.
        normalization_log_jacobian : float
            The log Jacobian associated to the normalization.
        """
        super().__init__()
        self.npe = npe
        self.embedding = embedding
        self.normalize_observation_fct = normalize_observation_fct
        self.unnormalize_observation_fct = unnormalize_observation_fct
        self.normalize_parameters_fct = normalize_parameters_fct
        self.unnormalize_parameters_fct = unnormalize_parameters_fct
        self.normalization_log_jacobian = normalization_log_jacobian

    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        """Forward call.

        Parameters
        ----------
        theta : Tensor
            The simulator's parameters.
        x : Tensor
            The observations.

        Returns
        -------
        Tensor
            The predicted posterior log probabilities.
        """
        x = self.normalize_observation_fct(x)
        theta = self.normalize_parameters_fct(theta)
        return self.npe(theta, self.embedding(x)) + self.normalization_log_jacobian

    def sample(self, x: Tensor, shape: Tuple) -> Tensor:
        """Sample simulator's parameters from the model.

        Parameters
        ----------
        x : Tensor
            The observations conditionning the sampling.
        shape : Tuple
            The shape of the samples.

        Returns
        -------
        Tensor
            The samples.
        """
        x = self.normalize_observation_fct(x)
        model_output = self.npe.flow(self.embedding(x)).sample(shape)
        model_output = self.unnormalize_parameters_fct(model_output)
        return model_output


class BayesianLoss(nn.Module):
    """Loss function for Bayesian deep learning."""

    def __init__(
        self,
        estimator: nn.Module,
        likelihoodLoss: Callable,
        prior: Union[
            GaussianNNParametersDistribution,
            HierarchicalGaussianNNParametersDistribution,
        ],
        n_train: int,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        estimator : nn.Module
            The estimator.
        likelihoodLoss : Callable
            The likelihood part of the loss.
        prior : Union[ GaussianNNParametersDistribution, HierarchicalGaussianNNParametersDistribution]
            The prior on weights.
        n_train : int
            The number of training samples.
        """
        super().__init__()
        self.estimator = estimator
        self.likelihoodLoss = likelihoodLoss(estimator)
        self.prior = prior
        self.n_train = n_train
        print("n_train = {}".format(n_train))

    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        """Forward pass computing the loss.

        Parameters
        ----------
        theta : Tensor
            The simulato's parameters.
        x : Tensor
            The observations.

        Returns
        -------
        Tensor
            The loss value
        """
        lik_loss = self.likelihoodLoss(theta, x)
        prior_loss = -self.prior.prior_log_prob(self.estimator)
        return lik_loss * self.n_train + prior_loss

    def resample_prior(self):
        """resample the prior."""
        self.prior.resample_prior()


class BayesianNPEModel(Model):
    """NPE BNN model."""

    def __init__(
        self,
        benchmark: Benchmark,
        model_path: str,
        config: dict,
        normalization_constants: dict,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        benchmark : Benchmark
            The benchmark.
        model_path : str
            The model saving path.
        config : dict
            The config.
        normalization_constants : dict
            The simulator's parameters and observations normalization constants.

        Raises
        ------
        NotImplementedError
            BNN method is not implemented.
        """

        class Embedding(nn.Module):
            """Embedding module with normalization."""

            def __init__(
                self, embedding: nn.Module, normalize_function: Callable
            ) -> None:
                """Constructor.

                Parameters
                ----------
                embedding : nn.Module
                    Embedding module.
                normalize_function : Callable
                    The function used to normalize the observations.
                """
                super().__init__()
                self.embedding = embedding
                self.normalize_function = normalize_function

            def forward(self, x: Tensor) -> Tensor:
                """Module forward pass.

                Parameters
                ----------
                x : Tensor
                    The observations.

                Returns
                -------
                Tensor
                    The embedding.
                """
                return self.embedding(self.normalize_function(x))

        super().__init__(normalization_constants)
        self.benchmark = benchmark
        self.config = config
        self.observable_shape = benchmark.get_observable_shape()
        self.embedding_dim = benchmark.get_embedding_dim()
        self.parameter_dim = benchmark.get_parameter_dim()
        self.device = benchmark.get_device()
        print("device = {}".format(self.device))

        self.model_path = model_path
        os.makedirs(self.model_path, exist_ok=True)
        self.config = config

        self.prior = benchmark.get_prior()

        embedding_kwargs = {}
        if (
            "embedding_nb_layers" in config.keys()
            or "embedding_nb_neurons" in config.keys()
        ):
            embedding_kwargs["nb_layers"] = config["embedding_nb_layers"]
            embedding_kwargs["nb_neurons"] = config["embedding_nb_neurons"]

            embedding_build = benchmark.get_embedding_build(modified=True)
            self.embedding = embedding_build(
                self.embedding_dim, self.observable_shape, **embedding_kwargs
            ).to(self.device)

            self.embedding_dim = config["embedding_nb_neurons"]
        else:
            embedding_build = benchmark.get_embedding_build()
            self.embedding = embedding_build(
                self.embedding_dim, self.observable_shape
            ).to(self.device)

        flow_build, flow_kwargs = benchmark.get_flow_build()

        # Update hyperparams if specified
        if "nb_layers" in config.keys():
            flow_kwargs["hidden_features"] = [
                flow_kwargs["hidden_features"][0]
            ] * config["nb_layers"]

        if "nb_neurons" in config.keys():
            flow_kwargs["hidden_features"] = [
                config["nb_neurons"] for _ in flow_kwargs["hidden_features"]
            ]

        if "nb_transforms" in config.keys():
            flow_kwargs["transforms"] = config["nb_transforms"]

        self.flow = NPE(
            self.parameter_dim, self.embedding_dim, build=flow_build, **flow_kwargs
        ).to(self.device)
        self.model = NPEWithEmbedding(
            self.flow,
            self.embedding,
            self.normalize_observation,
            self.unnormalize_observation,
            self.normalize_parameters,
            self.unnormalize_parameters,
            self.get_normalization_log_jacobian(),
        )

        self.bnn_prior = self.get_bnn_prior()
        self.bnn_prior.to(self.device)

        self.vi_model = NPEWithEmbedding(
            self.flow,
            nn.Identity(),
            nn.Identity(),
            self.unnormalize_observation,
            self.normalize_parameters,
            self.unnormalize_parameters,
            self.get_normalization_log_jacobian(),
        )

        self.embedding = Embedding(self.embedding, self.normalize_observation)

        if config["bnn_method"] == "vi":
            self.model = VImodel(
                self.model, config, self.device, model_path, embedding=None
            )
        elif config["bnn_method"] == "hmc":
            self.model = HMCmodel(self.model, config, self.device, model_path)
        else:
            raise NotImplementedError(
                "bnn_method '{}' not implemented.".format(self.config["bnn_method"])
            )
        self.model = self.model.to(self.device)

    @classmethod
    def is_trained(
        cls, bnn_method: str, model_path: str, chain_id: Optional[int] = None
    ) -> bool:
        """Returns whether the model is trained.

        Parameters
        ----------
        bnn_method : str
            The Bayesian deep learning method.
        model_path : str
            The path of the model.
        chain_id : Optional[int], optional
            The id of the chain, need only to be specified when using HMC,
            by default None

        Returns
        -------
        bool
            Whether the model is trained.
        """

        assert bnn_method in ["vi", "hmc"]
        assert chain_id is not None or bnn_method == "vi"

        if bnn_method == "vi":
            return os.path.exists(os.path.join(model_path, "VIparametrisaton.pt"))
        elif bnn_method == "hmc":
            return os.path.exists(
                os.path.join(model_path, "trained_{}.pt".format(chain_id))
            )

    @classmethod
    def is_initialized(cls, model_path: str) -> bool:
        """Returns whether the model is initialized.

        Parameters
        ----------
        model_path : str
            The path of the model.

        Returns
        -------
        bool
            Whether the model is initialized.
        """
        return os.path.exists(os.path.join(model_path, "bnn_prior.pt"))

    def log_prob(self, theta: Tensor, x: Tensor, id: Optional[int] = None) -> Tensor:
        """Compute the posterior log probability.

        Parameters
        ----------
        theta : Tensor
            The simulator's parameters.
        x : Tensor
            The observations.
        id : Optional[int], optional
            The id of the model to use, if not specififed, compute the
            Bayesian model average, by default None

        Returns
        -------
        Tensor
            The posterior log probabilities.
        """
        return self.model.log_prob(theta, x, id_net=id)

    def log_prob_one_model(self, theta: Tensor, x: Tensor) -> Tensor:
        """Compute the posterior log probability using a single neural network.

        Parameters
        ----------
        theta : Tensor
            The simulator's parameters.
        x : Tensor
            The observations.

        Returns
        -------
        Tensor
            The posterior log probabilities.
        """
        # id is None so random network is used
        return self.model.log_prob(theta, x)

    def sample(self, x: Tensor, shape: Tuple, id: Optional[int] = None) -> Tensor:
        """Sample from a posterior.

        Parameters
        ----------
        x : Tensor
            The observation.
        shape : Tuple
            The shape of the samples.
        id : Optional[int], optional
            The id of the model to use, if not specififed, sample from the
            Bayesian model average, by default None

        Returns
        -------
        Tensor
            The samples.
        """
        x = x.to(self.device)
        return self.model.sample(x, shape, id_net=id)

    def get_posterior_fct(self) -> Callable:
        """Get a posterior generating function.

        Returns
        -------
        Callable
            A function that returns an object with sample and log_prob functions given
            observation.
        """

        def get_posterior(x: Tensor) -> Any:
            """Returns a posterior object.

            Parameters
            ----------
            x : Tensor
                The observation.

            Returns
            -------
            Any
                An object with sample and log_prob functions
            """

            class Posterior:
                """Posterior class."""

                def __init__(
                    self, sampling_fct: Callable, log_prob_fct: Callable
                ) -> None:
                    """Constructor

                    Parameters
                    ----------
                    sampling_fct : Callable
                        The sampling function.
                    log_prob_fct : Callable
                        The posterior log probability function.
                    """
                    self.sample = sampling_fct
                    self.log_prob = log_prob_fct

            return Posterior(
                lambda shape: self.sample(x.to(self.device), shape).cpu(),
                lambda theta: self.log_prob(
                    theta.to(self.device), x.unsqueeze(0).to(self.device)
                )
                .squeeze(0)
                .cpu(),
            )

        return get_posterior

    def prior_log_prob(self, theta: Tensor, x: Tensor, n_estimators: int) -> Tensor:
        """Posterior log probability over simulator's parameters with a priori BMA.

        Parameters
        ----------
        theta : Tensor
            Simulator's parameters.
        x : Tensor
            Observations.
        n_estimators : int
            Number of estimators to compute the Bayesian model average.

        Returns
        -------
        Tensor
            The log probabilities.
        """
        x = x.to(self.device)
        theta = theta.to(self.device)
        outputs = self.bnn_prior.sample_functions(theta, x, n_estimators).squeeze(dim=2)
        outputs = torch.logsumexp(outputs, dim=-1) - np.log(n_estimators)

        return outputs

    def prior_sample(self, x: Tensor, shape: Tuple) -> Tensor:
        """Posterior samples over simulator's parameters with a priori BMA.

        Parameters
        ----------
        x : Tensor
            The observation.
        shape : Tuple
            The shape of the samples.

        Returns
        -------
        Tensor
            The samples.
        """
        x = x.to(self.device)
        nb_samples = np.prod(np.array(shape))
        samples = []
        for _ in range(nb_samples):
            samples.append(self.bnn_prior.sample(x))

        samples = torch.stack(samples, dim=0).view(*shape, -1)

        return samples

    def get_prior_fct(self, n_estimators: int) -> Callable:
        """Get a posterior generating function with a priori BMA.

        Parameters
        ----------
        n_estimators : int
            The number of estimators to approximate the BMA.

        Returns
        -------
        Callable
            The a priori BMA posterior generating function.
        """

        def get_prior(x: Tensor) -> Any:
            """Returns an a priori BMA posterior object.

            Parameters
            ----------
            x : Tensor
                The observation.

            Returns
            -------
            Any
                An object with sample and log_prob functions
            """

            class Prior:
                """A priori BMA posterior"""

                def __init__(
                    self, sampling_fct: Callable, log_prob_fct: Callable
                ) -> None:
                    """Constructor.

                    Parameters
                    ----------
                    sampling_fct : Callable
                        The sampling function.
                    log_prob_fct : Callable
                        The log probability function.
                    """
                    self.sample = sampling_fct
                    self.log_prob = log_prob_fct

            return Prior(
                lambda shape: self.prior_sample(x.to(self.device), shape).cpu(),
                lambda theta: self.prior_log_prob(
                    theta.to(self.device), x.unsqueeze(0).to(self.device), n_estimators
                )
                .squeeze(0)
                .cpu(),
            )

        return get_prior

    def get_bnn_prior(
        self,
    ) -> Union[
        GaussianNNParametersDistribution, HierarchicalGaussianNNParametersDistribution
    ]:
        """Get the BNN prior.

        Returns
        -------
        Union[GaussianNNParametersDistribution, HierarchicalGaussianNNParametersDistribution]
            The BNN prior.

        Raises
        ------
        NotImplementedError
            The prior variational family is not implemented.
        """
        if "init_low_variance_init" in self.config.keys():
            low_variance_init = self.config["init_low_variance_init"]
        else:
            low_variance_init = False

        if low_variance_init:
            std_init_value = self.config["init_std_init_value"]
        else:
            std_init_value = None

        if self.config["prior_variational_family"] == "gaussian":
            return GaussianNNParametersDistribution(
                self.model,
                shared=False,
                low_variance_init=low_variance_init,
                std_init_value=std_init_value,
            )
        elif self.config["prior_variational_family"] == "gaussian_softplus":
            return GaussianNNParametersDistribution(
                self.model,
                shared=False,
                softplus=True,
                beta=self.config["beta"],
                low_variance_init=low_variance_init,
                std_init_value=std_init_value,
            )
        elif self.config["prior_variational_family"] == "shared_gaussian":
            return GaussianNNParametersDistribution(self.model, shared=True)
        elif self.config["prior_variational_family"] == "hierarchical_gaussian":
            return HierarchicalGaussianNNParametersDistribution(
                self.model, shared=False
            )
        elif self.config["prior_variational_family"] == "shared_hierarchical_gaussian":
            return HierarchicalGaussianNNParametersDistribution(self.model, shared=True)
        else:
            raise NotImplementedError("Prior variational family not implemented.")

    def __call__(self, theta: Tensor, x: Tensor) -> Tensor:
        """Posterior log prob function.

        Parameters
        ----------
        theta : Tensor
            The simulator's parameters.
        x : Tensor
            The observations.

        Returns
        -------
        Tensor
            The posterior log probabilities.
        """
        return self.log_prob(theta, x)

    def sampling_enabled(self) -> bool:
        """Returns whether sampling is enabled.

        Returns
        -------
        bool
            Whether sampling is enabled.
        """
        return True

    def save(self) -> None:
        """Save the model."""
        self.model.save()

    def save_init(self) -> None:
        """Save the initialization."""
        torch.save(
            self.bnn_prior.state_dict(), os.path.join(self.model_path, "bnn_prior.pt")
        )

    def load_model(self, chain_id: int, index: int) -> None:
        """Load a model.

        Parameters
        ----------
        chain_id : int
            The id of the chain.
        index : int
            The index of the model within the chain.
        """
        self.embedding.load_state_dict(
            torch.load(
                os.path.join(
                    self.model_path, "embedding_{}_{}.pt".format(chain_id, index)
                )
            )
        )
        self.flow.load_state_dict(
            torch.load(
                os.path.join(self.model_path, "flow_{}_{}.pt".format(chain_id, index))
            )
        )

    def load(self) -> None:
        """Load the model."""
        self.model.load()

    def load_init(self) -> None:
        """Load the initialization."""
        self.bnn_prior.load_state_dict(
            torch.load(
                os.path.join(self.model_path, "bnn_prior.pt"), map_location=self.device
            )
        )

    def delete_models(self, chain_id: int) -> None:
        """Delete the models of a chain.

        Parameters
        ----------
        chain_id : int
            The id of the chain.
        """
        shutil.rmtree(
            os.path.join(self.model_path, "trained_{}.pt".format(chain_id)),
            ignore_errors=True,
        )
        shutil.rmtree(
            os.path.join(self.model_path, "embedding_{}_*.pt".format(chain_id)),
            ignore_errors=True,
        )
        shutil.rmtree(
            os.path.join(self.model_path, "flow_{}_*.pt".format(chain_id)),
            ignore_errors=True,
        )
        shutil.rmtree(os.path.join(self.model_path, "bnn_prior.pt"), ignore_errors=True)

    def train(self) -> None:
        """Put the model in train mode"""
        self.model.train()

    def eval(self) -> None:
        """Put the model in eval mode."""
        self.model.eval()

    def get_loss_fct(self) -> Callable:
        """Get the loss function.

        Returns
        -------
        Callable
            The loss function.
        """
        return NPELoss

    def train_models(
        self,
        train_set: H5Dataset,
        val_set: H5Dataset,
        config: dict,
        chain_id: Optional[int] = None,
    ) -> None:
        """Train the models.

        Parameters
        ----------
        train_set : H5Dataset
            The training set.
        val_set : H5Dataset
            The validation set.
        config : dict
            The config.
        chain_id : Optional[int], optional
            The id of the chain, need to be specified if HMC is used, by default None

        Raises
        ------
        NotImplementedError
            The bnn method is not implemented.
        """

        config = self.config
        loss_fct = self.get_loss_fct()
        if self.config["bnn_method"] == "vi":
            self.model.train_models(
                train_set, val_set, config, loss_fct, self.bnn_prior
            )

        elif self.config["bnn_method"] == "hmc":
            self.model.train_models(
                train_set, val_set, config, chain_id, loss_fct, self.bnn_prior
            )

        else:
            raise NotImplementedError(
                "bnn_method '{}' not implemented.".format(self.config["bnn_method"])
            )

    def initialize(
        self,
        train_set: H5Dataset,
        val_set: H5Dataset,
        config: dict,
        debug: bool = False,
        no_wandb_init: bool = False,
    ) -> None:
        """Initialize the model.

        Parameters
        ----------
        train_set : H5Dataset
            The training set.
        val_set : H5Dataset
            The validation set.
        config : dict
            The config.
        debug : bool, optional
            Whether debug mode is activated, by default False
        no_wandb_init : bool, optional
            Whether wandb is used or not, by default False
        """
        if config["optimize_prior"]:
            return self.optimize_prior(
                train_set, val_set, debug=debug, no_wandb_init=no_wandb_init
            )
        else:
            self.bnn_prior.set_mean(0.0)
            self.bnn_prior.set_std(config["prior_std"])

    def get_np_distribution(
        self,
        measurement_generator: Optional[
            Union[
                DatasetMeasurementGenerator,
                UniformMeasurementGenerator,
                HybridMeasurementGenerator,
            ]
        ] = None,
    ) -> GPPrior:
        """_summary_

        Parameters
        ----------
        measurement_generator : Optional[Union[DatasetMeasurementGenerator,
                                               UniformMeasurementGenerator,
                                               HybridMeasurementGenerator]],
                                               optional
            The measurement generator, by default None

        Returns
        -------
        GPPrior
            The non parametric prior.
        """
        return GPPrior(
            self.benchmark,
            self.config,
            self.device,
            measurement_generator=measurement_generator,
        )

    def wrap_bnn_prior(self, bnn_prior: nn.module) -> nn.Module:
        """Wrap the bnn prior.

        Parameters
        ----------
        bnn_prior : nn.module
            The bnn prior to wrap.

        Returns
        -------
        nn.Module
            The wrapped bnn prior.
        """
        return bnn_prior

    def optimize_prior(
        self,
        train_set: H5Dataset,
        val_set: H5Dataset,
        debug: bool = False,
        no_wandb_init: bool = False,
    ) -> None:
        """Optimize the prior.

        Parameters
        ----------
        train_set : H5Dataset
            The training set.
        val_set : H5Dataset
            The validation set.
        debug : bool, optional
            Whether debug mode is activated, by default False
        no_wandb_init : bool, optional
            Whether wandb is used or not, by default False
        """
        if self.config["use_wandb"] and not no_wandb_init:
            wandb.init(
                project=self.config["wandb_project"],
                entity=self.config["wandb_user"],
                config=self.config,
            )

        # Create measurement dataset

        if self.config["measurement_generator_type"] == "uniform":
            upper = None
            lower = None
            if (
                "automatic_observable_bounds" in self.config.keys()
                and self.config["automatic_observable_bounds"]
            ):
                for _, x in train_set:
                    if lower is None:
                        lower = torch.min(x, dim=0)[0]
                    else:
                        lower = torch.min(torch.cat((x, lower.unsqueeze(0))), dim=0)[0]

                    if upper is None:
                        upper = torch.max(x, dim=0)[0]
                    else:
                        upper = torch.max(torch.cat((x, upper.unsqueeze(0))), dim=0)[0]

                observable_bounds = (lower, upper)

                print("observable_bounds = {}".format(observable_bounds))

            else:
                observable_bounds = self.benchmark.get_observation_domain()

        if self.config["measurement_generator_type"] == "dataset":
            measurement_generator = DatasetMeasurementGenerator(train_set)
            val_measurement_generator = DatasetMeasurementGenerator(val_set)

        elif self.config["measurement_generator_type"] == "uniform":
            measurement_generator = UniformMeasurementGenerator(
                [
                    x * self.config["extend_parameter_domain"]
                    for x in self.benchmark.get_domain()
                ],
                observable_bounds,
            )
            val_measurement_generator = UniformMeasurementGenerator(
                [
                    x * self.config["extend_parameter_domain"]
                    for x in self.benchmark.get_domain()
                ],
                observable_bounds,
            )

        elif self.config["measurement_generator_type"] == "hybrid":
            measurement_generator = HybridMeasurementGenerator(
                [
                    x * self.config["extend_parameter_domain"]
                    for x in self.benchmark.get_domain()
                ],
                train_set,
            )
            val_measurement_generator = HybridMeasurementGenerator(
                [
                    x * self.config["extend_parameter_domain"]
                    for x in self.benchmark.get_domain()
                ],
                val_set,
            )

        else:
            raise NotImplementedError(
                "Measurement generator type {} not implemented.".format(
                    self.config["measurement_generator_type"]
                )
            )

        # Initialize the non parametric distribution to match
        if "automatic_kernel" in self.config.keys() and self.config["automatic_kernel"]:
            np_distribution = self.get_np_distribution(
                measurement_generator=measurement_generator
            )
        else:
            np_distribution = self.get_np_distribution()

        log_dir = os.path.join(self.model_path, "logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        output_dim = 1

        n_data = self.config["measurement_set_size"]
        n_samples = self.config["init_function_samples"]
        log_space = self.config["gp_log_space"]

        init_lr = self.config["init_lr"]
        init_iter = self.config["init_iter"]
        optimizer = self.config["init_optimizer"]

        optimizer_params = {"optimizer": optimizer, "lr": init_lr}
        if optimizer == "sgd":
            optimizer_params["momentum"] = self.config["init_momentum"]
        if optimizer == "adamw":
            optimizer_params["weight_decay"] = self.config["init_weight_decay"]

        use_wandb = self.config["use_wandb"]

        nb_val_steps = self.config["init_nb_val_steps"]
        val_step_every = self.config["init_val_step_every"]
        schedule_init_lr = self.config["schedule_init_lr"]

        distance_type = self.config["distance_type"]
        distance_config = {}

        if distance_type == "stein_KL":
            if self.config["set_eta"]:
                eta = self.config["eta"]
            else:
                eta = None

            if self.config["set_num_eigs"]:
                num_eigs = self.config["num_eigs"]
            else:
                num_eigs = None

            distance_config["eta"] = eta
            distance_config["num_eigs"] = num_eigs
            distance_config["joint_entropy"] = self.config["joint_entropy"]
            distance_config["loss_divider"] = self.config["loss_divider"]

        else:
            raise NotImplementedError(
                "Distance version '{}' does not exist.".format(distance_type)
            )

        clip_gradient = self.config["generator_clip_gradient"]
        if clip_gradient == "threshold":
            clipping_norm = self.config["generator_clipping_norm"]
            clipping_quantile = None
        elif clip_gradient == "quantile":
            clipping_norm = None
            clipping_quantile = self.config["generator_clipping_quantile"]
        else:
            clipping_norm = None
            clipping_quantile = None

        # Create the mapper
        mapper = DistanceBasedPriorMapper(
            np_distribution,
            self.wrap_bnn_prior(self.bnn_prior),
            measurement_generator,
            log_dir,
            log_space,
            distance_type,
            distance_config,
            output_dim=output_dim,
            n_data=n_data,
            gpu_np_model=not (self.device == "cpu"),
            device=self.device,
            clip_gradient=clip_gradient,
            clipping_norm=clipping_norm,
            clipping_quantile=clipping_quantile,
            use_wandb=use_wandb,
            validation_data_generator=val_measurement_generator,
            nb_val_steps=nb_val_steps,
            val_steps_every=val_step_every,
            schedule_lr=schedule_init_lr,
        )

        return mapper.optimize(
            init_iter, optimizer_params, n_samples=n_samples, debug=debug
        )

    def is_ensemble(self) -> bool:
        """Returns whether the model is an ensemble model.

        Returns
        -------
        bool
            Whether the model is an ensemble model.
        """
        return self.get_nb_networks() > 1

    def get_nb_networks(self) -> int:
        """Return the number of models used to compute the Bayasian model average.

        Returns
        -------
        int
            The number of models used to compute the Bayasian model average.

        Raises
        ------
        NotImplementedError
            BNN method is not implemented.
        """

        if self.config["bnn_method"] == "vi":
            return self.config["nb_networks"]
        elif self.config["bnn_method"] == "hmc":
            nb_networks = self.config["samples_per_chain"] * self.config["nb_chains"]
            return nb_networks
        else:
            raise NotImplementedError(
                "bnn_method '{}' not implemented.".format(self.config["bnn_method"])
            )
