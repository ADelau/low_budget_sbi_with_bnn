import os
from typing import Any, Callable, Tuple

import torch
import torch.nn as nn
from lampe.inference import NPE, NPELoss
from torch import Tensor

from ..benchmarks import Benchmark
from .base import Model, ModelFactory


class NPEFactory(ModelFactory):
    def __init__(
        self, config: dict, benchmark: Benchmark, simulation_budget: int
    ) -> None:
        """Constructor.

        Parameters
        ----------
        config : dict
            The config.
        benchmark : Benchmark
            The benchmark.
        simulation_budget : int
            The simulation budget.
        """
        config_run = config.copy()
        for idx in range(len(config["simulation_budgets"])):
            if config["simulation_budgets"][idx] == simulation_budget:
                break
        config_run["train_batch_size"] = config["train_batch_size"][idx]
        config_run["weight_decay"] = config["weight_decay"][idx]

        super().__init__(config_run, benchmark, simulation_budget, NPEModel)

    def get_train_time(self, benchmark_time: float, epochs: int) -> float:
        """Get the model training time.

        Parameters
        ----------
        benchmark_time : float
            The training time associated to the benchmark.
        epochs : int
            The number of epochs.

        Returns
        -------
        float
            The model training time.
        """
        return 2 * super().get_train_time(benchmark_time, epochs)


class NPEWithEmbedding(nn.Module):
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


class NPEModel(Model):
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
        """

        super().__init__(normalization_constants)
        self.observable_shape = benchmark.get_observable_shape()
        self.embedding_dim = benchmark.get_embedding_dim()
        self.parameter_dim = benchmark.get_parameter_dim()
        self.device = benchmark.get_device()

        self.model_path = model_path

        self.prior = benchmark.get_prior()
        self.config = config

        embedding_build = benchmark.get_embedding_build()
        self.embedding = embedding_build(self.embedding_dim, self.observable_shape).to(
            self.device
        )

        flow_build, flow_kwargs = benchmark.get_flow_build()
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

    @classmethod
    def is_trained(cls, model_path: str) -> bool:
        """Return whether the model is trained.

        Parameters
        ----------
        model_path : str
            The model saving path.

        Returns
        -------
        bool
            Whether the model is trained.
        """
        return os.path.exists(
            os.path.join(model_path, "embedding.pt")
        ) and os.path.exists(os.path.join(model_path, "flow.pt"))

    def get_loss_fct(self, config: dict) -> Callable:
        """Get the loss function.

        Parameters
        ----------
        config : dict
            The config.

        Returns
        -------
        Callable
            The loss function.
        """
        return NPELoss

    def log_prob(self, theta: Tensor, x: Tensor) -> Tensor:
        """Compute the posterior log probability

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
        x = x.to(self.device)
        theta = theta.to(self.device)
        return self.model(theta, x)

    def get_posterior_fct(self):
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
                lambda shape: self.model.sample(x.to(self.device), shape).cpu(),
                lambda theta: self.model(
                    theta.to(self.device), x.to(self.device)
                ).cpu(),
            )

        return get_posterior

    def __call__(self, theta: Tensor, x: Tensor) -> Tensor:
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
        torch.save(
            self.embedding.state_dict(), os.path.join(self.model_path, "embedding.pt")
        )
        torch.save(self.flow.state_dict(), os.path.join(self.model_path, "flow.pt"))

    def load(self) -> None:
        """Load the model"""
        self.embedding.load_state_dict(
            torch.load(
                os.path.join(self.model_path, "embedding.pt"), map_location=self.device
            )
        )
        self.flow.load_state_dict(
            torch.load(
                os.path.join(self.model_path, "flow.pt"), map_location=self.device
            )
        )

    def train(self) -> None:
        """Put the model in train mode"""
        self.model.train()

    def eval(self):
        """Put the model in eval mode."""
        self.model.eval()

    def sample(self, x: Tensor, shape: Tuple) -> Tensor:
        """Sample from the posterior.

        Parameters
        ----------
        x : Tensor
            The observation.
        shape : Tuple
            The sample shape.

        Returns
        -------
        Tensor
            The samples.
        """
        x = x.to(self.device)
        return self.model.sample(x, shape)

    def is_ensemble(self) -> bool:
        """Return whether model is an ensemble model.

        Returns
        -------
        bool
            Whether model is an ensemble model.
        """
        if "ensemble" in self.config:
            return self.config["ensemble"]
        return False
