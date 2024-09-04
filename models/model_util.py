from ..benchmarks import Benchmark
from .base import ModelFactory
from .bayesian_npe import BayesianNPEFactory
from .bayesian_nre import BayesianNREFactory
from .bnpe import BNPEFactory
from .bnre import BNREFactory
from .gp_model import GPFactory
from .npe import NPEFactory
from .nre import NREFactory


def load_model_factory(
    config: dict, benchmark: Benchmark, simulation_budget: int
) -> ModelFactory:
    """Load a model factory based on config.

    Parameters
    ----------
    config : dict
        The config.
    benchmark : Benchmark
        The benchmark.
    simulation_budget : int
        The simulation budget.

    Returns
    -------
    ModelFactory
        The loaded model factory

    Raises
    ------
    NotImplementedError
        Model not implemented.
    """
    if config["method"] == "nre":
        return NREFactory(config, benchmark, simulation_budget)

    elif config["method"] == "bnre":
        return BNREFactory(config, benchmark, simulation_budget)

    elif config["method"] == "npe":
        return NPEFactory(config, benchmark, simulation_budget)

    elif config["method"] == "bnpe":
        return BNPEFactory(config, benchmark, simulation_budget)

    elif config["method"] == "bayesian_npe":
        return BayesianNPEFactory(config, benchmark, simulation_budget)

    elif config["method"] == "bayesian_nre":
        return BayesianNREFactory(config, benchmark, simulation_budget)

    elif config["method"] == "gaussian_process":
        return GPFactory(config, benchmark, simulation_budget)

    else:
        raise NotImplementedError("Model not implemented")
