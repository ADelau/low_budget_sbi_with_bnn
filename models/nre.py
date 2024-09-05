import os
from typing import Callable

import torch
import torch.nn as nn
from lampe.inference import NRE, NRELoss
from torch import Tensor

from .base import Model, ModelFactory


class NREFactory(ModelFactory):
    def __init__(
        self, config: dict, benchmark, simulation_budget: int
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

        super().__init__(config_run, benchmark, simulation_budget, NREModel)


class ClassifierWithEmbedding(nn.Module):
    def __init__(
        self,
        classifier: nn.Module,
        embedding: nn.Module,
        normalize_observation_fct: Callable,
        unnormalize_observation_fct: Callable,
        normalize_parameters_fct: Callable,
        unnormalize_parameters_fct: Callable,
    ) -> None:
        """Constructor

        Parameters
        ----------
        classifier : nn.Module
            The classifier module
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
        """
        super().__init__()
        self.classifier = classifier
        self.embedding = embedding
        self.normalize_observation_fct = normalize_observation_fct
        self.unnormalize_observation_fct = unnormalize_observation_fct
        self.normalize_parameters_fct = normalize_parameters_fct
        self.unnormalize_parameters_fct = unnormalize_parameters_fct

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
            The evaluated model.
        """
        x = self.normalize_observation_fct(x)
        theta = self.normalize_parameters_fct(theta)
        return self.classifier(theta, self.embedding(x))


class NREModel(Model):
    def __init__(
        self,
        benchmark,
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
        self.config = config

        self.prior = benchmark.get_prior()

        embedding_build = benchmark.get_embedding_build()
        self.embedding = embedding_build(self.embedding_dim, self.observable_shape).to(
            self.device
        )

        classifier_build, classifier_kwargs = benchmark.get_classifier_build()
        self.classifier = NRE(
            self.parameter_dim,
            self.embedding_dim,
            build=classifier_build,
            **classifier_kwargs
        ).to(self.device)
        self.model = ClassifierWithEmbedding(
            self.classifier,
            self.embedding,
            self.normalize_observation,
            self.unnormalize_observation,
            self.normalize_parameters,
            self.unnormalize_parameters,
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
        ) and os.path.exists(os.path.join(model_path, "classifier.pt"))

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
        return NRELoss

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
        return (
            self.prior.log_prob(theta.cpu())
            + self.model(theta.to(self.device), x.to(self.device)).cpu()
        )

    def __call__(self, theta: Tensor, x: Tensor) -> Tensor:
        return self.log_prob(theta, x)

    def sampling_enabled(self) -> bool:
        """Returns whether sampling is enabled.

        Returns
        -------
        bool
            Whether sampling is enabled.
        """
        return False

    def save(self) -> None:
        """Save the model."""
        torch.save(
            self.embedding.state_dict(), os.path.join(self.model_path, "embedding.pt")
        )
        torch.save(
            self.classifier.state_dict(), os.path.join(self.model_path, "classifier.pt")
        )

    def load(self) -> None:
        """Load the model"""
        self.embedding.load_state_dict(
            torch.load(os.path.join(self.model_path, "embedding.pt"))
        )
        self.classifier.load_state_dict(
            torch.load(os.path.join(self.model_path, "classifier.pt"))
        )

    def train(self) -> None:
        """Put the model in train mode"""
        self.model.train()

    def eval(self):
        """Put the model in eval mode."""
        self.model.eval()

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
