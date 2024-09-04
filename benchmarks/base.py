import os
from abc import ABC, abstractmethod
from enum import Enum
from itertools import chain
from typing import Callable, Optional

import lampe
import torch
import torch.nn as nn
from torch import Tensor


class Datasets(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    COVERAGE = "coverage"


class Benchmark(ABC):
    def __init__(self, data_path, benchmark_name) -> None:
        self.benchmark_name = benchmark_name
        self.data_path = data_path
        self.create_data_dirs()

    @abstractmethod
    def get_prior(self) -> torch.distributions.Distribution:
        """Return the prior associated to this benchmark

        Returns:
            torch.distributions.Distribution: the prior distribution
        """

        pass

    @abstractmethod
    def simulate(self, parameters: Tensor) -> Tensor:
        """Perform a simulation

        Args:
            parameters (Tensor): The parameters conditioning the simulator

        Returns:
            Tensor: the synthetic observation
        """

        pass

    @abstractmethod
    def get_simulation_batch_size(self) -> int:
        pass

    @abstractmethod
    def is_vectorized(self) -> bool:
        pass

    def create_data_dirs(self) -> None:
        for dataset in Datasets:
            if not os.path.exists(
                os.path.join(
                    os.path.dirname(__file__),
                    os.path.join(
                        os.path.join(self.data_path, self.benchmark_name), dataset.value
                    ),
                )
            ):
                os.makedirs(
                    os.path.join(
                        os.path.dirname(__file__),
                        os.path.join(
                            os.path.join(self.data_path, self.benchmark_name),
                            dataset.value,
                        ),
                    )
                )

    def get_store_path(self, dataset: Datasets, id: Optional[int] = None) -> str:
        if id is not None:
            dataset_name = os.path.join(dataset.value, str(id))
        else:
            dataset_name = dataset.value

        return os.path.join(
            os.path.dirname(__file__),
            os.path.join(
                os.path.join(self.data_path, self.benchmark_name), dataset_name
            ),
        )

    def simulate_block(
        self,
        config: dict,
        dataset: Datasets,
        block_id: int,
        dataset_id: Optional[int] = None,
    ) -> None:
        prior = self.get_prior()
        loader = lampe.data.JointLoader(
            prior,
            self.simulate,
            batch_size=self.get_simulation_batch_size(),
            vectorized=self.is_vectorized(),
        )

        lampe.data.H5Dataset.store(
            loader,
            os.path.join(
                self.get_store_path(dataset, id=dataset_id),
                "block_{}.h5".format(block_id),
            ),
            size=config["block_size"],
        )

    def is_block_simulated(
        self,
        config: dict,
        dataset: Datasets,
        block_id: int,
        dataset_id: Optional[int] = None,
    ) -> bool:
        return os.path.exists(
            os.path.join(
                self.get_store_path(dataset, id=dataset_id),
                "block_{}.h5".format(block_id),
            )
        )

    def merge_blocks(
        self,
        config: dict,
        dataset: Datasets,
        dataset_size: int,
        block_ids: list[int],
        dataset_id: Optional[int] = None,
    ) -> None:

        # data = lampe.data.H5Dataset(
        #     *[
        #         os.path.join(
        #             self.get_store_path(dataset, id=dataset_id),
        #             "block_{}.h5".format(block_id),
        #         )
        #         for block_id in block_ids
        #     ],
        #     batch_size=self.get_simulation_batch_size()
        # )

        # lampe.data.H5Dataset.store(
        #     data,
        #     os.path.join(
        #         self.get_store_path(dataset, id=dataset_id),
        #         "dataset_{}.h5".format(dataset_size),
        #     ),
        #     size=dataset_size,
        # )

        datasets = [
            lampe.data.H5Dataset(
                os.path.join(
                    self.get_store_path(dataset, id=dataset_id),
                    "block_{}.h5".format(block_id),
                ),
                batch_size=self.get_simulation_batch_size(),
            )
            for block_id in block_ids
        ]

        lampe.data.H5Dataset.store(
            pairs=chain(*datasets),
            file=os.path.join(
                self.get_store_path(dataset, id=dataset_id),
                "dataset_{}.h5".format(dataset_size),
            ),
            size=dataset_size,
        )

        data = lampe.data.H5Dataset(
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

        if torch.any(x_std == 0):
            x_std[x_std == 0] = 1

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

    @abstractmethod
    def get_parameter_dim(self) -> int:
        pass

    @abstractmethod
    def get_observable_shape(self) -> tuple[int, ...]:
        pass

    def get_embedding_dim(self) -> int:
        return self.get_observable_shape()[0]

    @abstractmethod
    def get_classifier_build(self) -> tuple[Callable, dict]:
        pass

    @abstractmethod
    def get_flow_build(self) -> tuple[Callable, dict]:
        pass

    def get_embedding_build(self) -> Callable:
        def get_embedding(
            embedding_dim: int, observable_shape: tuple[int]
        ) -> nn.Module:
            return nn.Identity()

        return get_embedding

    def get_train_set(
        self, dataset_size: int, batch_size: int, id: int
    ) -> lampe.data.H5Dataset:
        data = lampe.data.H5Dataset(
            os.path.join(
                self.get_store_path(Datasets.TRAIN, id),
                "dataset_{}.h5".format(dataset_size),
            ),
            batch_size=batch_size,
            shuffle=True,
        )
        return data

    def get_normalization_constants(
        self, dataset_size: int, id: int
    ) -> dict[str, Tensor]:
        theta_mean = torch.load(
            os.path.join(
                self.get_store_path(Datasets.TRAIN, id),
                "theta_mean_{}.pt".format(dataset_size),
            )
        )
        theta_std = torch.load(
            os.path.join(
                self.get_store_path(Datasets.TRAIN, id),
                "theta_std_{}.pt".format(dataset_size),
            )
        )
        x_mean = torch.load(
            os.path.join(
                self.get_store_path(Datasets.TRAIN, id),
                "x_mean_{}.pt".format(dataset_size),
            )
        )
        x_std = torch.load(
            os.path.join(
                self.get_store_path(Datasets.TRAIN, id),
                "x_std_{}.pt".format(dataset_size),
            )
        )

        return {
            "theta_mean": theta_mean,
            "theta_std": theta_std,
            "x_mean": x_mean,
            "x_std": x_std,
        }

    def get_val_set(
        self, dataset_size: int, batch_size: int, id: int
    ) -> lampe.data.H5Dataset:
        data = lampe.data.H5Dataset(
            os.path.join(
                self.get_store_path(Datasets.VAL, id),
                "dataset_{}.h5".format(dataset_size),
            ),
            batch_size=batch_size,
        )
        return data

    def get_test_set(self, dataset_size: int, batch_size: int) -> lampe.data.H5Dataset:
        data = lampe.data.H5Dataset(
            os.path.join(
                self.get_store_path(Datasets.TEST), "dataset_{}.h5".format(dataset_size)
            ),
            batch_size=batch_size,
        )
        return data

    def get_coverage_set(self, dataset_size: int) -> lampe.data.H5Dataset:
        data = lampe.data.H5Dataset(
            os.path.join(
                self.get_store_path(Datasets.COVERAGE),
                "dataset_{}.h5".format(dataset_size),
            )
        )
        return data

    @abstractmethod
    def get_device(self) -> str:
        pass

    @abstractmethod
    def get_nb_cov_samples(self) -> int:
        pass

    @abstractmethod
    def get_cov_bins(self) -> int:
        pass

    @abstractmethod
    def get_simulate_nb_gpus(self) -> int:
        pass

    @abstractmethod
    def get_simulate_nb_cpus(self) -> int:
        pass

    @abstractmethod
    def get_simulate_ram(self, block_size: int) -> float:
        pass

    @abstractmethod
    def get_simulate_time(self, block_size: int) -> float:
        pass

    @abstractmethod
    def get_merge_nb_gpus(self) -> int:
        pass

    @abstractmethod
    def get_merge_nb_cpus(self) -> int:
        pass

    @abstractmethod
    def get_merge_ram(self, dataset_size: int) -> float:
        pass

    @abstractmethod
    def get_merge_time(self, dataset_size: int) -> float:
        pass

    @abstractmethod
    def get_train_nb_gpus(self) -> int:
        pass

    @abstractmethod
    def get_train_nb_cpus(self) -> int:
        pass

    @abstractmethod
    def get_train_ram(self) -> int:
        pass

    @abstractmethod
    def get_train_time(self, dataset_size) -> float:
        pass

    @abstractmethod
    def get_init_ram(self) -> int:
        pass

    @abstractmethod
    def get_init_time(self) -> int:
        pass

    @abstractmethod
    def get_test_nb_gpus(self) -> int:
        pass

    @abstractmethod
    def get_test_nb_cpus(self) -> int:
        pass

    @abstractmethod
    def get_test_ram(self) -> int:
        pass

    @abstractmethod
    def get_test_time(self, dataset_size) -> float:
        pass

    @abstractmethod
    def get_coverage_nb_gpus(self) -> int:
        pass

    @abstractmethod
    def get_coverage_nb_cpus(self) -> int:
        pass

    @abstractmethod
    def get_coverage_ram(self) -> int:
        pass

    @abstractmethod
    def get_coverage_time(self, dataset_size: int) -> float:
        pass
