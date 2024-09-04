# Reproduced from https://github.com/sbi-benchmark/sbibm

import math
from typing import Callable

import torch
import zuko
from torch import Tensor

from .base import Benchmark


class TwoMoons(Benchmark):
    def __init__(self, data_path) -> None:
        super().__init__(data_path, "two_moons")

        self.a_dist = torch.distributions.uniform.Uniform(-math.pi / 2.0, math.pi / 2.0)
        self.r_dist = torch.distributions.normal.Normal(0.1, 0.01)

        self.LOWER = -1 * torch.ones(2).float()
        self.UPPER = 1 * torch.ones(2).float()

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

        a = self.a_dist.sample()
        r = self.r_dist.sample()

        ang = torch.tensor([-math.pi / 4.0])
        c = torch.cos(ang)
        s = torch.sin(ang)

        z0 = c * parameters[0] - s * parameters[1]
        z1 = s * parameters[0] + c * parameters[1]

        return torch.Tensor(
            [torch.cos(a) * r + 0.25 - torch.abs(z0), torch.sin(a) * r + z1]
        )

    def get_simulation_batch_size(self) -> int:
        return 128

    def is_vectorized(self) -> bool:
        return False

    def get_parameter_dim(self) -> int:
        return 2

    def get_observable_shape(self) -> tuple[int, ...]:
        return (2,)

    def get_classifier_build(self) -> tuple[Callable, dict]:
        return zuko.nn.MLP, {"hidden_features": [256] * 6, "activation": torch.nn.SELU}

    def get_flow_build(self) -> tuple[Callable, dict]:
        return zuko.flows.NSF, {
            "hidden_features": [256] * 6,
            "activation": torch.nn.SELU,
        }

    def get_device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    def get_domain(self) -> tuple[Tensor, Tensor]:
        return self.LOWER, self.UPPER

    def get_observation_domain(self) -> tuple[Tensor, Tensor]:
        lower = -3 * torch.ones(2).float()
        upper = 3 * torch.ones(2).float()

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
        return 0.0001 * dataset_size

    def get_merge_time(self, dataset_size: int) -> float:
        return 0.1 * dataset_size

    def get_train_nb_gpus(self) -> int:
        return 1

    def get_train_nb_cpus(self) -> int:
        return 8

    def get_train_ram(self) -> int:
        return 4

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
        return 4

    def get_test_time(self, dataset_size: int) -> float:
        return 0.002 * dataset_size + 600

    def get_coverage_nb_gpus(self) -> int:
        return 1

    def get_coverage_nb_cpus(self) -> int:
        return 1

    def get_coverage_ram(self) -> int:
        return 4

    def get_coverage_time(self, dataset_size: int) -> float:
        return 1 * dataset_size + 1800
