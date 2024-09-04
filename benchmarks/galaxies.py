import os
from functools import reduce
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import zuko
from lampe.data import H5Dataset
from torch import Tensor
from torch.utils.data import DataLoader

from .base import Benchmark, Datasets


class Galaxies(Benchmark):
    def __init__(self, data_path, benchmark_name: str = "galaxies") -> None:
        super().__init__(data_path, benchmark_name)

        self._mu = [0.7, -2.9]
        self._p = torch.distributions.uniform.Uniform(-3.0, 3.0)

        self.LOWER = torch.tensor([0.1, 0.6]).float()
        self.UPPER = torch.tensor([0.5, 1.0]).float()

    @torch.no_grad()
    def simulate(self, parameters: Tensor) -> Tensor:
        obs_shape = self.get_observable_shape()
        return torch.zeros_like(obs_shape).float()

    def simulate_block(
        self,
        config: dict,
        dataset: Datasets,
        block_id: int,
        dataset_id: Optional[int] = None,
    ) -> None:
        return

    def is_block_simulated(
        self,
        config: dict,
        dataset: Datasets,
        block_id: int,
        dataset_id: Optional[int] = None,
    ) -> bool:
        return True

    def merge_blocks(
        self,
        config: dict,
        dataset: Datasets,
        dataset_size: int,
        block_ids: list[int],
        dataset_id: Optional[int] = None,
    ) -> None:
        data_file = os.path.join(
            os.path.dirname(__file__), os.path.join(self.data_path, self.benchmark_name)
        )
        if dataset == Datasets.TRAIN:
            x_file = os.path.join(data_file, "train_halos_tpcf_train.npy")
            theta_file = os.path.join(data_file, "train_cosmology_train.csv")
        elif dataset == Datasets.VAL:
            x_file = os.path.join(data_file, "train_halos_tpcf_val.npy")
            theta_file = os.path.join(data_file, "train_cosmology_val.csv")
        else:
            x_file = os.path.join(data_file, "test_halos_tpcf.npy")
            theta_file = os.path.join(data_file, "test_cosmology.csv")
        x = np.load(x_file)
        assert dataset_size <= x.shape[0], "Dataset size is too large"

        # Select random points from the dataset
        indices = np.random.choice(x.shape[0], dataset_size, replace=False)
        x = x[indices]

        thetas = pd.read_csv(theta_file)
        thetas = thetas.iloc[indices]
        thetas = thetas[["Omega_m", "sigma_8"]].values
        thetas = torch.Tensor(thetas)

        assert thetas.shape[0] == x.shape[0], "Theta and x have different sizes"

        data = [(thetas[i], x[i]) for i in range(len(x))]
        dataloader = DataLoader(data, batch_size=1, shuffle=False)
        H5Dataset.store(
            pairs=dataloader,
            file=os.path.join(
                self.get_store_path(dataset, id=dataset_id),
                "dataset_{}.h5".format(dataset_size),
            ),
            size=dataset_size,
        )

        data = H5Dataset(
            os.path.join(
                self.get_store_path(dataset, id=dataset_id),
                "dataset_{}.h5".format(dataset_size),
            )
        )

        xs = torch.stack([x for _, x in data])
        x_mean = xs.mean(dim=0)

        if dataset_size == 1:
            x_std = torch.ones_like(x_mean)
        else:
            x_std = xs.std(dim=0)

        prior = self.get_prior()
        theta_mean = prior.mean
        theta_std = prior.stddev

        torch.save(
            theta_mean,
            os.path.join(
                self.get_store_path(dataset, id=dataset_id),
                "theta_mean_{}.pt".format(dataset_size),
            ),
        )
        torch.save(
            theta_std,
            os.path.join(
                self.get_store_path(dataset, id=dataset_id),
                "theta_std_{}.pt".format(dataset_size),
            ),
        )
        torch.save(
            x_mean,
            os.path.join(
                self.get_store_path(dataset, id=dataset_id),
                "x_mean_{}.pt".format(dataset_size),
            ),
        )
        torch.save(
            x_std,
            os.path.join(
                self.get_store_path(dataset, id=dataset_id),
                "x_std_{}.pt".format(dataset_size),
            ),
        )

    def are_blocks_merged(
        self,
        config: dict,
        dataset: Datasets,
        dataset_size: int,
        dataset_id: Optional[int] = None,
    ) -> bool:
        return os.path.exists(
            os.path.join(
                self.get_store_path(dataset, id=dataset_id),
                "dataset_{}.h5".format(dataset_size),
            )
        )

    @torch.no_grad()
    def get_prior(self) -> torch.distributions.Distribution:
        """Return the prior associated to this benchmark

        Returns:
            torch.distributions.Distribution: the prior distribution
        """

        return zuko.distributions.BoxUniform(self.LOWER, self.UPPER)

    def get_simulation_batch_size(self) -> int:
        return 128

    def is_vectorized(self) -> bool:
        return False

    def get_parameter_dim(self) -> int:
        return 2

    def get_observable_shape(self) -> tuple[int, ...]:
        return (24,)

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

    # TODO: check the values
    def get_observation_domain(self) -> tuple[Tensor, Tensor]:
        lower = -1 * torch.ones(24).float()
        upper = 30 * torch.ones(24).float()

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
        return 0.002 * dataset_size + 60

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
