import math
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Gamma


class GaussianNNParametersDistribution(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        shared: bool = False,
        softplus: bool = False,
        beta: Optional[float] = None,
        low_variance_init: bool = False,
        std_init_value: Optional[float] = None,
        init_distribution: Optional[Any] = None,
    ) -> None:
        """Constructor

        Parameters
        ----------
        model : nn.Module
            The base model.
        shared : bool, optional
            Whether parameter are shared across layer, by default False
        softplus : bool, optional
            Whether to apply a softplus function to std, by default False
        beta : Optional[float], optional
            Softplus beta parameter, by default None
        low_variance_init : bool, optional
            Whether to init std to a low value, by default False
        std_init_value : Optional[float], optional
            The value at which to init the std, by default None
        init_distribution : Optional[Any], optional
            The distribution to use as initialization, by default None

        Raises
        ------
        NotImplementedError
            A layer is not supported by the variational family.
        """

        super().__init__()

        self.model = model
        self.shared = shared
        self.softplus = softplus
        self.beta = beta

        if shared:
            self.m = {
                self.transform_param_name(name): nn.Parameter(torch.zeros((1)))
                for name, param in model.named_parameters()
            }
        else:
            if init_distribution:
                self.m = {
                    self.transform_param_name(name): nn.Parameter(
                        torch.ones_like(param)
                        * init_distribution.m[self.transform_param_name(name)]
                    )
                    for name, param in model.named_parameters()
                }
            elif low_variance_init:
                self.m = {
                    self.transform_param_name(name): nn.Parameter(
                        torch.ones_like(param) * param
                    )
                    for name, param in model.named_parameters()
                }
            else:
                self.m = {
                    self.transform_param_name(name): nn.Parameter(
                        torch.zeros_like(param)
                    )
                    for name, param in model.named_parameters()
                }

        if softplus:
            self.s_ = {}
        else:
            self.log_s_ = {}

        for name, param in model.named_parameters():
            if init_distribution:
                if softplus:
                    if init_distribution.softplus:
                        init_std = init_distribution.s_[self.transform_param_name(name)]
                    else:
                        init_std = torch.exp(
                            init_distribution.log_s_[self.transform_param_name(name)]
                        )
                else:
                    if init_distribution.softplus:
                        init_log_std = torch.log(
                            F.softplus(
                                init_distribution.s_[self.transform_param_name(name)],
                                beta=init_distribution.beta,
                            )
                        )
                    else:
                        init_log_std = init_distribution.log_s_[
                            self.transform_param_name(name)
                        ]

            elif low_variance_init and not shared:
                if softplus:
                    init_std = std_init_value
                else:
                    init_log_std = math.log(std_init_value)

            else:
                if "bias" in name:
                    if softplus:
                        init_std = 1.0
                    else:
                        init_log_std = math.log(1.0)

                elif "weight" in name:
                    if softplus:
                        init_std = 1.0 / param.shape[0]
                    else:
                        init_log_std = math.log(1.0 / param.shape[0])
                else:
                    raise NotImplementedError(
                        "A layer is not supported by the variational family: {}".format(
                            name
                        )
                    )

            if shared:
                if softplus:
                    self.s_[self.transform_param_name(name)] = nn.Parameter(
                        torch.ones((1)) * init_std
                    )
                else:
                    self.log_s_[self.transform_param_name(name)] = nn.Parameter(
                        torch.ones((1)) * init_log_std
                    )
            else:
                if softplus:
                    self.s_[self.transform_param_name(name)] = nn.Parameter(
                        torch.ones_like(param) * init_std
                    )
                else:
                    self.log_s_[self.transform_param_name(name)] = nn.Parameter(
                        torch.ones_like(param) * init_log_std
                    )

        self.m = nn.ParameterDict(self.m)
        if softplus:
            self.s_ = nn.ParameterDict(self.s_)
        else:
            self.log_s_ = nn.ParameterDict(self.log_s_)

    def transform_param_name(self, name: str) -> str:
        """Transform the name of a parameter to avoid problematic characters.

        Parameters
        ----------
        name : str
            The name of the param

        Returns
        -------
        str
            The transformed name.
        """
        return name.replace(".", "@")

    def inverse_transform_param_name(self, name: str) -> str:
        """Reverse transform a parameter name to its original value.

        Parameters
        ----------
        name : str
            The transformed name.

        Returns
        -------
        str
            The reverse transformed name.
        """
        return name.replace("@", ".")

    def sample_functions(self, theta: Tensor, x: Tensor, n_samples: int) -> Tensor:
        """Sample function values.

        Parameters
        ----------
        theta : Tensor
            The simulator's parameters.
        x : Tensor
            The observation
        n_samples : int
            The number of function values to sample.

        Returns
        -------
        Tensor
            The function values.
        """
        outputs = []

        for _ in range(n_samples):

            if self.softplus:
                params = {
                    name: self.m[self.transform_param_name(name)]
                    + torch.randn_like(param)
                    * F.softplus(
                        self.s_[self.transform_param_name(name)], beta=self.beta
                    )
                    for name, param in self.model.named_parameters()
                }
            else:
                params = {
                    name: self.m[self.transform_param_name(name)]
                    + torch.randn_like(param)
                    * torch.exp(self.log_s_[self.transform_param_name(name)])
                    for name, param in self.model.named_parameters()
                }

            output = torch.func.functional_call(
                self.model, params, (theta, x), strict=False
            )
            if len(output.shape) == 0:
                output = output.unsqueeze(dim=0)
            outputs.append(output)

        outputs = torch.stack(outputs, dim=1)
        outputs = outputs.unsqueeze(dim=2)

        return outputs

    def sample(self, x: Tensor) -> Tensor:
        """Sample simulator's parameters from the Baysian Model Average

        Parameters
        ----------
        x : Tensor
            The observation

        Returns
        -------
        Tensor
            Simulator's parameters.
        """
        model_state = self.model.state_dict()
        for name, param in self.model.named_parameters():
            if self.softplus:
                model_state[name] = self.m[
                    self.transform_param_name(name)
                ] + torch.randn_like(param) * F.softplus(
                    self.s_[self.transform_param_name(name)], beta=self.beta
                )
            else:
                model_state[name] = self.m[
                    self.transform_param_name(name)
                ] + torch.randn_like(param) * torch.exp(
                    self.log_s_[self.transform_param_name(name)]
                )

        self.model.load_state_dict(model_state)
        return self.model.sample(x, (1,)).squeeze()

    def set_mean(self, mean: float) -> None:
        """Set the mean to a fixed value.

        Parameters
        ----------
        mean : float
            The valueto set the mean to.
        """
        self.m = nn.ParameterDict(
            {key: mean * torch.ones_like(value) for key, value in self.m.items()}
        )

    def set_std(self, std: float) -> None:
        """Set the std to a fixed value.

        Parameters
        ----------
        std : float
            The value ot set the std to.
        """
        if self.softplus:
            # TODO: set std to inverse softplus
            self.s_ = nn.ParameterDict(
                {
                    key: std * torch.ones_like(value)
                    for key, value in self.log_s_.items()
                }
            )
        else:
            self.log_s_ = nn.ParameterDict(
                {
                    key: np.log(std) * torch.ones_like(value)
                    for key, value in self.log_s_.items()
                }
            )

    def prior_log_prob(self, model: nn.Module) -> Tensor:
        """Compute the prior log probability of a model's weights.

        Parameters
        ----------
        model : nn.Module
            The model.

        Returns
        -------
        Tensor
            The pior log probability of the model.
        """
        log_prior = 0.0
        for name, param in model.named_parameters():
            m = self.m[self.transform_param_name(name)]
            if self.softplus:
                log_s_ = torch.log(
                    F.softplus(self.s_[self.transform_param_name(name)], beta=self.beta)
                )
            else:
                log_s_ = self.log_s_[self.transform_param_name(name)]

            per_param_prior = -0.5 * (
                np.log(2 * np.pi) + 2 * log_s_ + ((param - m) / log_s_.exp()) ** 2
            )
            log_prior += per_param_prior.sum()

        return log_prior

    def get_prior_grad(
        self, name: str, param: Tensor, min_std: Optional[str] = None
    ) -> Tensor:
        """Get the parameter gradient of the log prior.

        Parameters
        ----------
        name : str
            The name of the parameter
        param : Tensor
            The parameter value
        min_std : Optional[str], optional
            The minimal prior std, if specified, any std below this value is set to
            this value, by default None

        Returns
        -------
        Tensor
            The parameter gradient of the log prior.
        """
        m = self.m[self.transform_param_name(name)]
        if self.softplus:
            s_ = F.softplus(self.s_[self.transform_param_name(name)], beta=self.beta)
            if min_std is not None:
                s_ = torch.maximum(s_, torch.Tensor([min_std]).to(s_.device))

            return (param - m) / s_**2
        else:
            log_s_ = self.log_s_[self.transform_param_name(name)]
            if min_std is not None:
                log_s_ = torch.maximum(
                    log_s_, torch.log(torch.Tensor([min_std]).to(log_s_.device))
                )

            return (param - m) / (torch.exp(log_s_) ** 2)

    def resample_prior(self):
        pass

    def sample_network(self) -> nn.Module:
        """Draw weights from the variational posterior and return a list of networks.

        Returns
        -------
        nn.Module
            network with the sampled weights
        """
        params = {}

        for name, param in self.m.named_parameters():
            inverted_name = self.inverse_transform_param_name(name)

            # Sample new parameters from the Gaussian distribution
            if self.softplus:
                s_ = F.softplus(self.s_[name], beta=self.beta)
            else:
                s_ = torch.exp(self.log_s_[name])

            sampled_param = self.m[name] + torch.randn_like(param) * s_

            params[inverted_name] = sampled_param

        return params

    def get_kl_loss(
        self,
        prior_distribution: Any,
        train_size: int,
        temperature: float,
        min_prior_std: Optional[float] = None,
    ) -> torch.Tensor:
        """Compute the KL divergence between the variational distribution and the prior.

        Parameters
        ----------
        prior_distribution : Any
            A GaussianNNParametersDistribution representing the prior.
        train_size : int
            number of training samples.
        temperature : float
           temperature of the KL divergence.
        min_prior_std : Optional[float], optional
            The minimal prior std, if specified, any std below this value is set to
            this value, by default None

        Returns
        -------
        torch.Tensor
            KL divergence between the variational distribution and the prior.
        """
        # Iterate over params
        loss_kl = 0.0

        for name in self.m.keys():
            # Load own params
            p_m_0 = self.m[name]
            if self.softplus:
                p_s_0 = F.softplus(self.s_[name], beta=self.beta)
            else:
                p_s_0 = torch.exp(self.log_s_[name])

            # Load other params
            p_m_1 = prior_distribution.m[name]
            if prior_distribution.softplus:
                p_s_1 = F.softplus(
                    prior_distribution.s_[name], beta=prior_distribution.beta
                )

            else:
                p_s_1 = torch.exp(prior_distribution.log_s_[name])

            if min_prior_std is not None:
                p_s_1 = torch.maximum(
                    p_s_1, torch.Tensor([min_prior_std]).to(p_s_1.device)
                )

            # Compute the different terms
            loss_kl += 0.5 * (
                ((p_s_0**2 + (p_m_0 - p_m_1) ** 2) / p_s_1**2).sum()
                - p_m_0.numel()
                + (p_s_1**2 / p_s_0**2).log().sum()
            )

        return loss_kl * temperature / train_size


class HierarchicalGaussianNNParametersDistribution(nn.Module):
    def __init__(self, model: nn.Module, shared: bool = False) -> None:
        """Constructor

        Parameters
        ----------
        model : nn.Module
            The base model.
        shared : bool, optional
            Whether parameters are shared across layer, by default False
        """
        super().__init__()

        self.model = model
        self.shared = shared

        if shared:
            self.m = {
                self.transform_param_name(key): nn.Parameter(torch.zeros((1)))
                for key, value in dict(model.named_parameters()).items()
            }
        else:
            self.m = {
                self.transform_param_name(key): nn.Parameter(torch.zeros_like(value))
                for key, value in dict(model.named_parameters()).items()
            }

        self.log_shape = {}
        self.log_rate = {}
        for key, value in dict(model.named_parameters()).items():

            if shared:
                self.log_shape[self.transform_param_name(key)] = nn.Parameter(
                    torch.zeros((1))
                )
                self.log_rate[self.transform_param_name(key)] = nn.Parameter(
                    torch.zeros((1))
                )
            else:
                self.log_shape[self.transform_param_name(key)] = nn.Parameter(
                    torch.zeros_like(value)
                )
                self.log_rate[self.transform_param_name(key)] = nn.Parameter(
                    torch.zeros_like(value)
                )

        self.m = nn.ParameterDict(self.m)
        self.log_shape = nn.ParameterDict(self.log_shape)
        self.log_rate = nn.ParameterDict(self.log_rate)

    def transform_param_name(self, name: str) -> str:
        """Transform the name of a parameter to avoid problematic characters.

        Parameters
        ----------
        name : str
            The name of the param

        Returns
        -------
        str
            The transformed name.
        """
        return name.replace(".", "@")

    def inverse_transform_param_name(self, name: str) -> str:
        """Reverse transform a parameter name to its original value.

        Parameters
        ----------
        name : str
            The transformed name.

        Returns
        -------
        str
            The reverse transformed name.
        """
        return name.replace("@", ".")

    def sample_functions(self, theta: Tensor, x: Tensor, n_samples: int) -> Tensor:
        """Sample function values.

        Parameters
        ----------
        theta : Tensor
            The simulator's parameters.
        x : Tensor
            The observation
        n_samples : int
            The number of function values to sample.

        Returns
        -------
        Tensor
            The function values.
        """
        outputs = []

        for _ in range(n_samples):

            s_ = {
                key: 1.0
                / torch.sqrt(
                    Gamma(
                        torch.exp(self.log_shape[self.transform_param_name(key)]),
                        torch.exp(self.log_rate[self.transform_param_name(key)]),
                    ).rsample()
                )
                for key in dict(self.model.named_parameters()).keys()
            }

            param = {
                key: self.m[self.transform_param_name(key)]
                + torch.randn_like(value) * s_[key]
                for key, value in dict(self.model.named_parameters()).items()
            }

            outputs.append(
                torch.func.functional_call(self.model, param, (theta, x), strict=True)
            )

        outputs = torch.stack(outputs, dim=1)
        outputs = outputs.unsqueeze(dim=2)

        return outputs
