# Reproduced from https://github.com/montefiore-ai/hypothesis

from functools import reduce
from typing import Callable

import torch
import torch.nn as nn
import zuko
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal as Normal

from .base import Benchmark


class SLCP(Benchmark):
    def __init__(self, data_path, benchmark_name: str = "slcp") -> None:
        super().__init__(data_path, benchmark_name)

        self._mu = [0.7, -2.9]
        self._p = torch.distributions.uniform.Uniform(-3.0, 3.0)

        self.LOWER = -3 * torch.ones(2).float()
        self.UPPER = 3 * torch.ones(2).float()

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

        success = False

        while not success:
            try:
                if self._mu is None:
                    mean = torch.tensor(
                        [self._p.sample().item(), self._p.sample().item()]
                    ).float()
                else:
                    mean = torch.tensor(self._mu).float()
                scale = 1.0
                s_1 = parameters[0] ** 2
                s_2 = parameters[1] ** 2
                rho = self._p.sample().tanh()
                covariance = torch.tensor(
                    [
                        [scale * s_1**2, scale * rho * s_1 * s_2],
                        [scale * rho * s_1 * s_2, scale * s_2**2],
                    ]
                )
                normal = Normal(mean, covariance)
                x_out = normal.sample(torch.Size((4,))).view(-1)
                success = True
            except ValueError:
                pass

        return x_out

    def get_simulation_batch_size(self) -> int:
        return 128

    def is_vectorized(self) -> bool:
        return False

    def get_parameter_dim(self) -> int:
        return 2

    def get_observable_shape(self) -> tuple[int, ...]:
        return (8,)

    def get_embedding_build(self, modified=False) -> Callable:
        if modified:

            def get_embedding(
                embedding_dim: int,
                observable_shape: tuple[int, ...],
                nb_layers: int = 6,
                nb_neurons: int = 256,
            ) -> nn.Module:
                modules: list[nn.Module] = []
                modules.append(
                    nn.Linear(reduce(lambda x, y: x * y, observable_shape), nb_neurons)
                )
                modules.append(nn.SELU())

                for i in range(nb_layers):
                    modules.append(nn.Linear(nb_neurons, nb_neurons))
                    modules.append(nn.SELU())

                modules.append(nn.Linear(nb_neurons, nb_neurons))

                return nn.Sequential(*modules)

            return get_embedding
        else:

            def get_embedding(
                embedding_dim: int, observable_shape: tuple[int]
            ) -> nn.Module:
                return nn.Identity()

            return get_embedding

    def get_classifier_build(self) -> tuple[Callable, dict]:
        return zuko.nn.MLP, {"hidden_features": [256] * 6, "activation": torch.nn.SELU}

    def get_flow_build(self) -> tuple[Callable, dict]:
        return zuko.flows.NSF, {
            "hidden_features": [256] * 6,
            "activation": torch.nn.SELU,
        }

    def get_small_flow_build(self) -> tuple[Callable, dict]:
        return zuko.flows.NSF, {
            "hidden_features": [32] * 2,
            "activation": torch.nn.SELU,
        }

    def get_device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    def get_domain(self) -> tuple[Tensor, Tensor]:
        return self.LOWER, self.UPPER

    def get_observation_domain(self) -> tuple[Tensor, Tensor]:
        lower = -30 * torch.ones(8).float()
        upper = 30 * torch.ones(8).float()

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

    def get_init_ram(self) -> int:
        return 32

    def get_init_time(self) -> int:
        return 500000

    def get_train_time(self, dataset_size: int) -> float:
        return 0.001 * dataset_size + 60

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
