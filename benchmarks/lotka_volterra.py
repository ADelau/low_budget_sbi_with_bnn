# Reproduced from https://github.com/montefiore-ai/hypothesis

import math
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import zuko
from torch import Tensor

from .base import Benchmark


class LotkaVolterra(Benchmark):
    def __init__(self, data_path, benchmark_name: str = "lotka_volterra"):
        super().__init__(data_path, benchmark_name)

        epsilon = 0.00001
        self.LOWER = -4 * torch.ones(2).float()
        self.UPPER = torch.ones(2).float()
        self.UPPER += epsilon  # Account for half-open interval

        self._initial_state = np.array([50, 100])
        self._duration = 50.0
        self._dt = 0.05
        self._prey_prior = self.get_prior()

    @torch.no_grad()
    def get_prior(self) -> torch.distributions.Distribution:
        """Return the prior associated to this benchmark

        Returns:
            torch.distributions.Distribution: the prior distribution
        """

        return zuko.distributions.BoxUniform(self.LOWER, self.UPPER)

    @torch.no_grad()
    def simulate(self, parameters: Tensor) -> Tensor:
        """Perform a simulation

        Args:
            parameters (Tensor): The parameters conditioning the simulator

        Returns:
            Tensor: the synthetic observation
        """

        latents = self._prey_prior.sample()
        parameters = torch.cat([parameters, latents]).exp().numpy()

        steps = int(self._duration / self._dt) + 1
        states = np.zeros((steps, 2))
        state = np.copy(self._initial_state)
        for step in range(steps):
            x, y = state
            xy = x * y
            propensities = np.array([xy, x, y, xy])
            rates = parameters * propensities
            total_rate = sum(rates)
            if total_rate <= 0.00001:
                break
            normalized_rates = rates / total_rate
            transition = np.random.choice([0, 1, 2, 3], p=normalized_rates)
            if transition == 0:
                state[0] += 1  # Increase predator population by 1
            elif transition == 1:
                state[0] -= 1  # Decrease predator population by 1
            elif transition == 2:
                state[1] += 1  # Increase prey population by 1
            else:
                state[1] -= 1  # Decrease prey population by 1
            states[step, :] = np.copy(state)

        return torch.from_numpy(states).float()

    def get_simulation_batch_size(self) -> int:
        return 128

    def is_vectorized(self) -> bool:
        return False

    def get_parameter_dim(self) -> int:
        return 2

    def get_observable_shape(self) -> tuple[int, ...]:
        return (2001, 2)

    def get_embedding_dim(self) -> int:
        return 16 * 32

    def get_embedding_build(self) -> Callable:
        class Prepare(nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: Tensor) -> Tensor:
                """
                Reshapes the input according to the shape saved in the view data
                structure.
                """
                x = x.view([-1, 1001, 2]).permute((0, 2, 1))
                return x

        def get_embedding(
            embedding_dim: int, observable_shape: tuple[int, ...]
        ) -> nn.Module:
            nb_channels = 16
            nb_conv_layers = 10
            shrink_every = 2
            final_shape = 1001

            for i in range(nb_conv_layers):
                if i % shrink_every == 0:
                    final_shape = math.floor((final_shape - 1) / 2 + 1)
                else:
                    final_shape = final_shape

            cnn = [
                Prepare(),
                nn.Conv1d(in_channels=2, out_channels=nb_channels, kernel_size=1),
            ]

            for i in range(nb_conv_layers):
                if i % shrink_every == 0:
                    stride = 2
                else:
                    stride = 1

                cnn.append(
                    nn.Conv1d(
                        in_channels=nb_channels,
                        out_channels=nb_channels,
                        kernel_size=3,
                        padding=1,
                    )
                )
                cnn.append(nn.SELU())
                cnn.append(nn.MaxPool1d(3, stride=stride, padding=1))

            cnn.append(nn.Flatten())

            return nn.Sequential(*cnn)

        return get_embedding

    def get_classifier_build(self) -> tuple[Callable, dict]:
        return zuko.nn.MLP, {"hidden_features": [256] * 6, "activation": torch.nn.SELU}

    def get_flow_build(self) -> tuple[Callable, dict]:
        return zuko.flows.NSF, {
            "hidden_features": [256] * 3,
            "activation": torch.nn.SELU,
        }

    def get_device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    def get_domain(self) -> tuple[Tensor, Tensor]:
        return self.LOWER, self.UPPER

    def get_observation_domain(self) -> tuple[Tensor, Tensor]:
        lower = 0 * torch.ones((1001, 2)).float()
        upper = 1000 * torch.ones((1001, 2)).float()

        return lower, upper

    def get_nb_cov_samples(self) -> int:
        return 1000

    def get_cov_bins(self) -> int:
        return 100

    def get_simulate_nb_gpus(self) -> int:
        return 0

    def get_simulate_nb_cpus(self) -> int:
        return 1

    def get_simulate_ram(self, block_size: int) -> float:
        return 0.0001 * block_size

    def get_simulate_time(self, block_size: int) -> float:
        return 10 * block_size

    def get_merge_nb_gpus(self) -> int:
        return 0

    def get_merge_nb_cpus(self) -> int:
        return 1

    def get_merge_ram(self, dataset_size: int) -> float:
        return 0.0005 * dataset_size + 2

    def get_merge_time(self, dataset_size: int) -> float:
        return 0.1 * dataset_size + 500

    def get_train_nb_gpus(self) -> int:
        return 1

    def get_train_nb_cpus(self) -> int:
        return 8

    def get_train_ram(self) -> int:
        return 8

    def get_train_time(self, dataset_size: int) -> float:
        return 0.001 * dataset_size + 60

    def get_init_ram(self) -> int:
        return 32

    def get_init_time(self) -> int:
        return 500000

    def get_test_nb_gpus(self) -> int:
        return 1

    def get_test_nb_cpus(self) -> int:
        return 1

    def get_test_ram(self) -> int:
        return 8

    def get_test_time(self, dataset_size: int) -> float:
        return 0.002 * dataset_size + 600

    def get_coverage_nb_gpus(self) -> int:
        return 1

    def get_coverage_nb_cpus(self) -> int:
        return 1

    def get_coverage_ram(self) -> int:
        return 8

    def get_coverage_time(self, dataset_size: int) -> float:
        return 2 * dataset_size + 1800
