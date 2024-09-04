# Reproduced from https://github.com/montefiore-ai/hypothesis

import math
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import zuko
from scipy.signal import convolve2d
from torch import Tensor
from torch.distributions.poisson import Poisson

from .base import Benchmark


class SpatialSIR(Benchmark):
    def __init__(self, data_path) -> None:
        super().__init__(data_path, "spatialsir")

        self.LOWER = torch.zeros(2).float()
        self.UPPER = torch.ones(2).float()

        self.measurement_time = 1.0
        self.lattice_shape = (50, 50)
        self.p_initial_infections = Poisson(float(3))
        self.simulation_step_size = 0.01

    @torch.no_grad()
    def get_prior(self) -> torch.distributions.Distribution:
        """Return the prior associated to this benchmark

        Returns:
            torch.distributions.Distribution: the prior distribution
        """

        return zuko.distributions.BoxUniform(self.LOWER, self.UPPER)

    def _sample_num_initial_infections(self) -> int:
        return int(1 + self.p_initial_infections.sample().item())

    @torch.no_grad()
    def simulate(self, parameters: Tensor) -> Tensor:
        """Perform a simulation

        Args:
            parameters (Tensor): The parameters conditioning the simulator

        Returns:
            Tensor: the synthetic observation
        """

        # Extract the simulation parameters.
        beta = parameters[0].item()  # Infection rate
        gamma = parameters[1].item()  # Recovery rate

        psi = self.measurement_time

        infected = np.zeros(self.lattice_shape, dtype=np.int32)
        recovered = np.zeros(self.lattice_shape, dtype=np.int32)
        kernel = np.ones((3, 3), dtype=np.int32)

        # Seed the grid with the initial infections.
        num_initial_infections = self._sample_num_initial_infections()
        for _ in range(num_initial_infections):
            index_height = np.random.randint(0, self.lattice_shape[0])
            index_width = np.random.randint(0, self.lattice_shape[1])
            infected[index_height][index_width] = 1

        # Derrive the maximum number of simulation steps.
        simulation_steps = int(psi / self.simulation_step_size)
        susceptible = (1 - recovered) * (1 - infected)

        for _ in range(simulation_steps):
            if infected.sum() == 0:
                break
            # Infection
            potential = convolve2d(infected, kernel, mode="same")
            potential *= susceptible
            potential = potential * beta / 8
            next_infected = (
                (potential > np.random.uniform(size=self.lattice_shape)).astype(
                    np.int32
                )
                + infected
            ) * (1 - recovered)
            next_infected = (next_infected > 0).astype(np.int32)
            # Recover
            potential = infected * gamma
            next_recovered = (
                potential > np.random.uniform(size=self.lattice_shape)
            ).astype(np.int32) + recovered
            next_recovered = (next_recovered > 0).astype(np.int32)
            # Next parameters
            recovered = next_recovered
            infected = next_infected
            susceptible = (1 - recovered) * (1 - infected)

        # Convert to tensors
        susceptible = torch.from_numpy(susceptible).view(
            1, self.lattice_shape[0], self.lattice_shape[1]
        )
        infected = torch.from_numpy(infected).view(
            1, self.lattice_shape[0], self.lattice_shape[1]
        )
        recovered = torch.from_numpy(recovered).view(
            1, self.lattice_shape[0], self.lattice_shape[1]
        )
        image = torch.cat([susceptible, infected, recovered], dim=0)

        return image.bool()

    def get_simulation_batch_size(self) -> int:
        return 128

    def is_vectorized(self) -> bool:
        return False

    def get_parameter_dim(self) -> int:
        return 2

    def get_observable_shape(self) -> tuple[int, ...]:
        return (3, 50, 50)

    def get_embedding_dim(self) -> int:
        return 256

    def get_embedding_build(self) -> Callable:
        class View(nn.Module):
            def __init__(self, shape: tuple[int, ...]) -> None:
                super().__init__()
                self.shape = shape

            def forward(self, x: Tensor) -> Tensor:
                """
                Reshapes the input according to the shape saved in the view data
                structure.
                """
                x = x.view(self.shape)
                return x

        def get_embedding(
            embedding_dim: int, observable_shape: tuple[int, ...]
        ) -> nn.Module:
            nb_channels = 16
            nb_conv_layers = 8
            shrink_every = 2
            final_shape = 50

            for i in range(nb_conv_layers):
                if i % shrink_every == 0:
                    final_shape = math.floor((final_shape - 1) / 2 + 1)
                else:
                    final_shape = final_shape

            cnn = [
                View([-1, 3, 50, 50]),
                nn.Conv2d(in_channels=3, out_channels=nb_channels, kernel_size=1),
            ]

            for i in range(nb_conv_layers):
                if i % shrink_every == 0:
                    stride = 2
                else:
                    stride = 1

                cnn.append(
                    nn.Conv2d(
                        in_channels=nb_channels,
                        out_channels=nb_channels,
                        kernel_size=3,
                        padding=1,
                    )
                )
                cnn.append(nn.SELU())
                cnn.append(nn.MaxPool2d(3, stride=stride, padding=1))

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
        lower = 0 * torch.ones((3, 50, 50)).float()
        upper = 1 * torch.ones((3, 50, 50)).float()

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
        return 0.0001 * dataset_size + 16

    def get_merge_time(self, dataset_size: int) -> float:
        return 0.1 * dataset_size + 5000

    def get_train_nb_gpus(self) -> int:
        return 1

    def get_train_nb_cpus(self) -> int:
        return 8

    def get_train_ram(self) -> int:
        return 16

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
        return 16

    def get_test_time(self, dataset_size: int) -> float:
        return 0.002 * dataset_size + 600

    def get_coverage_nb_gpus(self) -> int:
        return 1

    def get_coverage_nb_cpus(self) -> int:
        return 1

    def get_coverage_ram(self) -> int:
        return 16

    def get_coverage_time(self, dataset_size: int) -> float:
        return 2 * dataset_size + 1800
