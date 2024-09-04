import torch
from torch import Tensor

from ..benchmarks import Benchmark
from .base import Model, ModelFactory
from .np_priors import GPPrior


class GPFactory(ModelFactory):
    def __init__(
        self, config: dict, benchmark: Benchmark, simulation_budget: int
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
        super().__init__(config, benchmark, simulation_budget, GPModel)


class GPModel(Model):
    def __init__(
        self,
        benchmark: Benchmark,
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
        super().__init__(normalization_constants)

        self.benchmark = benchmark
        self.prior = benchmark.get_prior()
        self.config = config
        self.device = benchmark.get_device()
        self.nb_estimators = config["nb_estimators"]
        self.gp_prior = GPPrior(benchmark, config, self.device)

    def log_prob(self, theta: Tensor, x: Tensor) -> Tensor:
        """Get the posterior log probability

        Parameters
        ----------
        theta : Tensor
            The simulator's parameters.
        x : Tensor
            The observations.

        Returns
        -------
        Tensor
            The posterior log probabilities.
        """
        output = (
            self.gp_prior.sample_functions(theta, x, self.nb_estimators)
            .squeeze(dim=2)
            .mean(dim=1)
        )

        if not self.gp_prior.log_space:
            output = torch.maximum(
                output, torch.Tensor([1e-10]).to(output.device)
            ).log()

        return output

    def train_models(self) -> None:
        pass

    @classmethod
    def is_trained(cls, model_path: str) -> bool:
        return True

    def sampling_enabled(self) -> bool:
        return False

    def __call__(self, theta: Tensor, x: Tensor) -> Tensor:
        return self.log_prob(theta, x)

    def get_loss_fct(self) -> None:
        pass

    def save(self) -> None:
        pass

    def load(self) -> None:
        pass

    def train(self) -> None:
        pass

    def eval(self) -> None:
        pass

    def to(self, device: str):
        pass
