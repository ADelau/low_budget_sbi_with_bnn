import os
import shutil
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from lampe.inference import NRE, NRELoss
from torch import Tensor

from ..benchmarks import Benchmark
from .bayesian_methods.hmc import HMCmodel
from .bayesian_methods.vi import VImodel
from .bayesian_npe import BayesianNPEFactory, BayesianNPEModel
from .nre import ClassifierWithEmbedding


class BayesianNREFactory(BayesianNPEFactory):
    """Factory instantiating BayesianNRE models."""

    def __init__(
        self,
        config: dict,
        benchmark: Benchmark,
        simulation_budget: int,
        model_class: Any = None,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        config : dict
            The config file
        benchmark : Benchmark
            The benchmark.
        simulation_budget : int
            The simulation_budget.
        model_class : Any, optional
            The class of the models to instantiate, if not specified, a BayesianNPE
            model is instantiated, by default None
        """
        if model_class is None:
            model_class = BayesianNREModel

        super().__init__(config, benchmark, simulation_budget, model_class)


class BayesianNREModel(BayesianNPEModel):
    """NRE BNN model."""

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
            The model path
        config : dict
            The config
        normalization_constants : dict
            The normalization constants

        Raises
        ------
        NotImplementedError
            BNN method not implemented.
        """
        super().__init__(benchmark, model_path, config, normalization_constants)

        classifier_build, classifier_kwargs = benchmark.get_classifier_build()

        # Update hyperparams if specified
        if "nb_layers" in config.keys():
            classifier_kwargs["hidden_features"] = [
                classifier_kwargs["hidden_features"][0]
            ] * config["nb_layers"]

        if "nb_neurons" in config.keys():
            classifier_kwargs["hidden_features"] = [
                config["nb_neurons"] for _ in classifier_kwargs["hidden_features"]
            ]

        self.classifier = NRE(
            self.parameter_dim,
            self.embedding_dim,
            build=classifier_build,
            **classifier_kwargs
        ).to(self.device)
        self.model = ClassifierWithEmbedding(
            self.classifier,
            self.embedding,
            self.normalize_observation,
            self.unnormalize_observation,
            self.normalize_parameters,
            self.unnormalize_parameters,
        )

        self.bnn_prior = self.get_bnn_prior()
        self.bnn_prior.to(self.device)

        if config["bnn_method"] == "vi":
            self.model = VImodel(self.model, config, self.device, model_path)
        elif config["bnn_method"] == "hmc":
            self.model = HMCmodel(self.model, config, self.device, model_path)
        else:
            raise NotImplementedError(
                "bnn_method '{}' not implemented.".format(self.config["bnn_method"])
            )
        self.model = self.model.to(self.device)

    def get_loss_fct(self) -> Callable:
        """Get the loss function.

        Returns
        -------
        Callable
            The loss function.
        """
        return NRELoss

    def log_prob(self, theta: Tensor, x: Tensor, id: Optional[int] = None) -> Tensor:
        """Compute the posterior log probability.

        Parameters
        ----------
        theta : Tensor
            The simulator's parameters.
        x : Tensor
            The observations.
        id : Optional[int], optional
            The id of the model to use, if not specififed, compute the
            Bayesian model average, by default None

        Returns
        -------
        Tensor
            The posterior log probabilities.
        """
        log_ratio = self.model.log_prob(theta, x, id_net=id)
        log_prior = self.prior.log_prob(theta.cpu()).to(log_ratio.device)
        return log_ratio + log_prior

    def log_prob_one_model(self, theta: Tensor, x: Tensor) -> Tensor:
        """Compute the posterior log probability using a single neural network.

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
        x = x.to(self.device)
        theta = theta.to(self.device)
        chain_index = np.random.randint(self.nb_chains)
        model_index = np.random.randint(self.expected_nb_models)
        self.load_model(chain_index, model_index)
        self.to(self.device)
        return self.model(theta, x) + self.prior.log_prob(theta.cpu())

    def sample(self, x, shape):
        pass

    def prior_log_prob(self, theta: Tensor, x: Tensor, n_estimators: int) -> Tensor:
        """Posterior log probability over simulator's parameters with a priori BMA.

        Parameters
        ----------
        theta : Tensor
            Simulator's parameters.
        x : Tensor
            Observations.
        n_estimators : int
            Number of estimators to compute the Bayesian model average.

        Returns
        -------
        Tensor
            The log probabilities.
        """
        x = x.to(self.device)
        theta = theta.to(self.device)
        outputs = self.bnn_prior.sample_functions(theta, x, n_estimators).squeeze(dim=2)
        outputs = (
            torch.logsumexp(outputs, dim=-1)
            - np.log(n_estimators)
            + self.prior.log_prob(theta.cpu()).to(self.device)
        )

        return outputs

    def prior_sample(self, x: Tensor, shape: Tuple) -> Tensor:
        pass

    def sampling_enabled(self) -> bool:
        return False

    def save_model(self, chain_id: int, index: int) -> None:
        """Save the model.

        Parameters
        ----------
        chain_id : int
            The id of the chain.
        index : int
            The index of the model within the chain.
        """
        torch.save(
            self.embedding.state_dict(),
            os.path.join(self.model_path, "embedding_{}_{}.pt".format(chain_id, index)),
        )
        torch.save(
            self.flow.state_dict(),
            os.path.join(
                self.model_path, "classifier_{}_{}.pt".format(chain_id, index)
            ),
        )

    def load_model(self, chain_id: int, index: int) -> None:
        """Load a model.

        Parameters
        ----------
        chain_id : int
            The id of the chain.
        index : int
            The index of the model within the chain.
        """
        self.embedding.load_state_dict(
            torch.load(
                os.path.join(
                    self.model_path, "embedding_{}_{}.pt".format(chain_id, index)
                )
            )
        )
        self.flow.load_state_dict(
            torch.load(
                os.path.join(
                    self.model_path, "classifier_{}_{}.pt".format(chain_id, index)
                )
            )
        )

    def delete_models(self, chain_id: int) -> None:
        """Delete the models of a chain.

        Parameters
        ----------
        chain_id : int
            The id of the chain.
        """
        shutil.rmtree(
            os.path.join(self.model_path, "trained_{}.pt".format(chain_id)),
            ignore_errors=True,
        )
        shutil.rmtree(
            os.path.join(self.model_path, "embedding_{}_*.pt".format(chain_id)),
            ignore_errors=True,
        )
        shutil.rmtree(
            os.path.join(self.model_path, "flow_{}_*.pt".format(chain_id)),
            ignore_errors=True,
        )
        shutil.rmtree(os.path.join(self.model_path, "bnn_prior.pt"), ignore_errors=True)

    def wrap_bnn_prior(self, bnn_prior: nn.module) -> nn.Module:
        """Wrap the bnn prior.

        Parameters
        ----------
        bnn_prior : nn.module
            The bnn prior to wrap.

        Returns
        -------
        nn.Module
            The wrapped bnn prior.
        """

        class WrappedPrior(nn.Module):
            def __init__(self, bnn_prior, parameter_prior):
                super().__init__()
                self.bnn_prior = bnn_prior
                self.parameter_prior = parameter_prior

            def sample_functions(self, theta, x, n_samples):
                bnn_samples = self.bnn_prior.sample_functions(theta, x, n_samples)
                return bnn_samples + self.parameter_prior.log_prob(theta.cpu()).to(
                    bnn_samples.device
                )

        return WrappedPrior(bnn_prior, self.prior)
