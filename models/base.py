import copy
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.optim as optim
from lampe.data import H5Dataset
from lampe.utils import GDStep
from torch import Tensor
from tqdm import tqdm


class ModelFactory(ABC):
    """Abstract class for a model factory"""

    def __init__(
        self,
        config: dict,
        benchmark,
        simulation_budget: int,
        model_class: Any,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        config : dict
            The config.
        benchmark : Benchmark
            The benchmark
        simulation_budget : int
            The simulation budget.
        model_class : Any
            The class of model to instantiate with the factory.
        """
        self.experience_dir = os.path.join(
            config["experience_dir"], str(simulation_budget)
        )
        Path(self.experience_dir).mkdir(exist_ok=True, parents=True)

        self.model_class = model_class
        self.benchmark = benchmark
        self.config = config

    def is_trained(self, id: int) -> bool:
        """Return whether the model is trained.

        Parameters
        ----------
        id : int
            The id of the model.

        Returns
        -------
        bool
            Whether the model is trained.
        """
        return self.model_class.is_trained(self.get_model_path(id))

    def is_initialized(self, id: int) -> bool:
        """Returns whether the model has been initialized.

        Parameters
        ----------
        id : int
            The id of the model.

        Returns
        -------
        bool
            Whether the model has been initialized.
        """
        return self.model_class.is_initialized(self.get_model_path(id))

    def instantiate_model(self, id: int, normalization_constants: dict) -> Any:
        """Instantiate a model.

        Parameters
        ----------
        id : int
            The id of the model.
        normalization_constants : dict
            The normalization constants.

        Returns
        -------
        Any
            The instantiated model.
        """
        model_path = self.get_model_path(id)
        Path(model_path).mkdir(exist_ok=True, parents=True)

        model = self.model_class(
            self.benchmark, model_path, self.config, normalization_constants
        )
        model.to(self.benchmark.get_device())
        return model

    def get_model_path(self, id: int) -> str:
        return os.path.join(self.experience_dir, "model_{}".format(id))

    def require_multiple_trainings(self) -> bool:
        return False

    def nb_trainings_required(self) -> int:
        return 1

    def get_train_nb_cpus(self, benchmark_nb_cpus: int) -> int:
        return benchmark_nb_cpus

    def get_train_nb_gpus(self, benchmark_nb_gpus: int) -> int:
        return benchmark_nb_gpus

    def get_train_ram(self, benchmark_ram: int) -> int:
        return benchmark_ram

    def get_train_time(self, benchmark_time: float, epochs: int) -> float:
        return benchmark_time * epochs + 1800

    def get_init_ram(self, benchmark_ram: int) -> int:
        return benchmark_ram

    def get_init_time(self, benchmark_time: float) -> float:
        return benchmark_time

    def get_test_nb_cpus(self, benchmark_nb_cpus: int) -> int:
        return self.get_train_nb_cpus(benchmark_nb_cpus)

    def get_test_nb_gpus(self, benchmark_nb_gpus: int) -> int:
        return self.get_train_nb_gpus(benchmark_nb_gpus)

    def get_test_ram(self, benchmark_ram: int) -> int:
        return self.get_train_ram(benchmark_ram)

    def get_test_time(self, benchmark_time: float) -> float:
        return self.get_train_time(benchmark_time, 1)

    def get_coverage_nb_cpus(self, benchmark_nb_cpus: int) -> int:
        return self.get_train_nb_cpus(benchmark_nb_cpus)

    def get_coverage_nb_gpus(self, benchmark_nb_gpus: int) -> int:
        return self.get_train_nb_gpus(benchmark_nb_gpus)

    def get_coverage_ram(self, benchmark_ram: int) -> int:
        return benchmark_ram

    def get_coverage_time(self, benchmark_time: float) -> float:
        return benchmark_time


class Model(ABC):
    """Abstract class for a model."""

    def __init__(
        self, normalization_constants: dict, config: Optional[dict] = None
    ) -> None:
        """Constructor.

        Parameters
        ----------
        normalization_constants : dict
            The normalization constants.
        config : Optional[dict], optional
            The config, by default None
        """
        self.normalization_constants = normalization_constants
        self.config = config

    def get_sampling_fct(self) -> Callable:
        """Get the sampling function

        Returns
        -------
        Callable
            The sampling function.
        """
        raise NotImplementedError()

    def to(self, device: str) -> None:
        self.model.to(device)

    def initialize(
        self, train_set: H5Dataset, val_set: H5Dataset, config: dict
    ) -> None:
        """Initialize the model

        Parameters
        ----------
        train_set : H5Dataset
            The training set
        val_set : H5Dataset
            The validation set
        config : dict
            The config.
        """
        pass

    def train_models(
        self, train_set: H5Dataset, val_set: H5Dataset, config: dict
    ) -> None:
        """Train the models.

        Parameters
        ----------
        train_set : H5Dataset
            The training set
        val_set : H5Dataset
            The validation set
        config : dict
            The config.
        """
        config = self.config

        learning_rate = float(config["learning_rate"])
        epochs = config["epochs"]
        self.train()
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=config["weight_decay"],
        )

        if config["schedule"]:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, verbose=True, min_lr=float(config["min_lr"])
            )

        step = GDStep(optimizer)

        best_loss = float("inf")
        best_weights = self.model.state_dict()

        loss = self.get_loss_fct(config)(self.model)

        train_losses = []
        val_losses = []

        with tqdm(range(epochs), unit="epoch") as tq:

            for epoch in tq:

                self.train()

                # Perform train steps
                train_loss = (
                    torch.stack(
                        [
                            step(loss(theta.to(self.device), x.to(self.device)))
                            for theta, x in train_set
                        ]
                    )
                    .mean()
                    .item()
                )

                self.eval()

                # Evaluate performance on validation set
                with torch.no_grad():
                    val_loss = (
                        torch.stack(
                            [
                                loss(theta.to(self.device), x.to(self.device))
                                for theta, x in val_set
                            ]
                        )
                        .mean()
                        .item()
                    )

                # Save the weights if they achieve the best validation loss
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_weights = copy.deepcopy(self.model.state_dict())

                if config["schedule"]:
                    scheduler.step(val_loss)

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                tq.set_postfix(train_loss=train_loss, val_loss=val_loss)

        self.model.load_state_dict(best_weights)
        torch.save(train_losses, os.path.join(self.model_path, "train_losses.pt"))
        torch.save(val_losses, os.path.join(self.model_path, "val_losses.pt"))

    def normalize_observation(self, x: Tensor) -> Tensor:
        """Normalize observations

        Parameters
        ----------
        x : Tensor
            The observations.

        Returns
        -------
        Tensor
            The normalized observations.
        """
        x_mean = self.normalization_constants["x_mean"].to(x.device)
        x_std = self.normalization_constants["x_std"].to(x.device)

        # Single observation passed
        if x.shape == x_mean.shape:
            return (x - x_mean) / x_std

        # Batch of observations passed
        else:
            return (x - x_mean.unsqueeze(dim=0)) / x_std.unsqueeze(dim=0)

    def normalize_parameters(self, theta: Tensor) -> Tensor:
        """Normalize simulator's parameters

        Parameters
        ----------
        theta : Tensor
            The simulator's parameters.

        Returns
        -------
        Tensor
            The normalized simulator's parameters.
        """
        theta_mean = self.normalization_constants["theta_mean"].to(theta.device)
        theta_std = self.normalization_constants["theta_std"].to(theta.device)

        # Single observation passed
        if theta.shape == theta_mean.shape:
            return (theta - theta_mean) / theta_std

        # Batch of observations passed
        else:
            return (theta - theta_mean.unsqueeze(dim=0)) / theta_std.unsqueeze(dim=0)

    def unnormalize_observation(self, x: Tensor) -> Tensor:
        """Unnormalize observations

        Parameters
        ----------
        x : Tensor
            The normalized observations.

        Returns
        -------
        Tensor
            The unnormalized observations.
        """
        x_mean = self.normalization_constants["x_mean"].to(x.device)
        x_std = self.normalization_constants["x_std"].to(x.device)

        # Single observation passed
        if x.shape == x_mean.shape:
            return x * x_std + x_mean

        # Batch of observations passed
        else:
            return x * x_std.unsqueeze(dim=0) + x_mean.unsqueeze(dim=0)

    def unnormalize_parameters(self, theta: Tensor) -> Tensor:
        """Unnormalize simulator's parameters

        Parameters
        ----------
        theta : Tensor
            The normalized simulator's parameters.

        Returns
        -------
        Tensor
            The unnormalized simulator's parameters.
        """
        theta_mean = self.normalization_constants["theta_mean"].to(theta.device)
        theta_std = self.normalization_constants["theta_std"].to(theta.device)

        # Single observation passed
        if theta.shape == theta_mean.shape:
            return theta * theta_std + theta_mean

        # Batch of observations passed
        else:
            return theta * theta_std.unsqueeze(dim=0) + theta_mean.unsqueeze(dim=0)

    def get_normalization_log_jacobian(self) -> float:
        """Get the normalization log Jacobian.

        Returns
        -------
        float
            The normalization log Jacobian.
        """
        return -torch.sum(torch.log(self.normalization_constants["theta_std"]))

    @classmethod
    def is_initialized(cls, model_path):
        return True

    def save_init(self):
        pass

    def load_init(self):
        pass

    @classmethod
    @abstractmethod
    def is_trained(cls, model_path):
        pass

    @abstractmethod
    def get_loss_fct(self):
        pass

    @abstractmethod
    def log_prob(self, x, theta):
        pass

    @abstractmethod
    def sampling_enabled(self):
        return False

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def eval(self):
        pass
