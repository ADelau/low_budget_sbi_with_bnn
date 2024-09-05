from typing import Callable

from lampe.inference import BNRELoss

from .base import ModelFactory
from .nre import NREModel


class BNREFactory(ModelFactory):
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

        super().__init__(config_run, benchmark, simulation_budget, BNREModel)


class BNREModel(NREModel):
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
        return lambda estimator: BNRELoss(
            estimator, lmbda=config["regularization_strength"]
        )
