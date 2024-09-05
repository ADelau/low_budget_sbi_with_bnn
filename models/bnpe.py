from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor

from .base import ModelFactory
from .npe import NPEModel


class BNPELoss(nn.Module):
    def __init__(
        self,
        estimator: nn.Module,
        prior: torch.distributions.Distribution,
        lmbda: float = 100.0,
    ) -> None:
        """Constructor

        Parameters
        ----------
        estimator : nn.Module
            The estimator.
        prior : torch.distributions.Distribution
            The prior over simulator's parameters.
        lmbda : float, optional
            The balancing regularization strength, by default 100.0
        """
        super().__init__()

        self.estimator = estimator
        self.prior = prior
        self.lmbda = lmbda

    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        """Loss forward

        Parameters
        ----------
        theta : Tensor
            The simulator's parameters.
        x : Tensor
            The observations.

        Returns
        -------
        Tensor
            The loss value.
        """

        theta_prime = torch.roll(theta, 1, dims=0)

        log_p, log_p_prime = self.estimator(
            torch.stack((theta, theta_prime)),
            x,
        )

        l0 = -log_p.mean()

        # balancing criterion
        # discriminator output = sigmoid(log ratio) = sigmoid(log posterior - log prior)
        lb = (
            (
                torch.sigmoid(log_p - self.prior.log_prob(theta.cpu()).to(log_p.device))
                + torch.sigmoid(
                    log_p_prime
                    - self.prior.log_prob(theta_prime.cpu()).to(log_p_prime.device)
                )
                - 1
            )
            .mean()
            .square()
        )

        return l0 + self.lmbda * lb


class BNPEFactory(ModelFactory):
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

        super().__init__(config_run, benchmark, simulation_budget, BNPEModel)

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
        return 4 * super().get_train_time(benchmark_time, epochs)


class BNPEModel(NPEModel):
    def __init__(
        self,
        benchmark,
        model_path: str,
        config: dict,
        normalization_constants: dict,
    ) -> None:
        """Constructor

        Parameters
        ----------
        benchmark : Benchmark
            The benchmark.
        model_path : str
            The model saving path.
        config : dict
            The config.
        normalization_constants : dict
            The normalization constants.
        """
        super().__init__(benchmark, model_path, config, normalization_constants)

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
        return lambda estimator: BNPELoss(
            estimator, self.prior, lmbda=config["regularization_strength"]
        )
