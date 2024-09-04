from itertools import islice

import numpy as np
import torch
from lampe.diagnostics import expected_coverage_mc, expected_coverage_ni
from lampe.utils import gridapply
from tqdm import tqdm


def intrinsic_expected_coverage_mc(posterior, pairs, prior, n):

    ranks = []

    with torch.no_grad():
        for theta, x in tqdm(pairs, unit="pair"):
            dist = posterior(x)

            samples = dist.sample((n,))
            mask = dist.log_prob(theta) - prior.log_prob(theta) < dist.log_prob(
                samples
            ) - prior.log_prob(samples)
            rank = mask.sum() / mask.numel()

            ranks.append(rank)

    ranks = torch.stack(ranks).cpu()
    ranks = torch.cat((ranks, torch.tensor([0.0, 1.0])))

    return (
        torch.sort(ranks).values,
        torch.linspace(0.0, 1.0, len(ranks)),
    )


def intrinsic_expected_coverage_ni(posterior, pairs, prior, domain, **kwargs):
    ranks = []

    with torch.no_grad():
        for theta, x in tqdm(pairs, unit="pair"):
            _, log_probs = gridapply(
                lambda theta: posterior(theta, x), domain, **kwargs
            )
            _, log_priors = gridapply(
                lambda theta: prior.log_prob(theta), domain, **kwargs
            )
            log_ratios = log_probs - log_priors
            mask = posterior(theta, x) - prior.lob_prob(theta) < log_ratios
            rank = log_probs[mask].logsumexp(dim=0) - log_probs.flatten().logsumexp(
                dim=0
            )

            ranks.append(rank.exp())

    ranks = torch.stack(ranks).cpu()
    ranks = torch.cat((ranks, torch.tensor([0.0, 1.0])))

    return (
        torch.sort(ranks).values,
        torch.linspace(0.0, 1.0, len(ranks)),
    )


def compute_coverage(
    model, benchmark, config, bounds=None, intrinsic=False, prior=False
):
    """Compute the empirical expected coverage of a model.

    Args:
        model (Model): a model such as defined in models/base.py
        benchmark (Benchmark): a benchmark such as defined in benchmarks/base.py
        config (dict): a dictionary containing the configuration
        bounds (int, int): bounds on dataset indices on which to evaluate the coverage

    Returns:
        (Tensor, Tensor): a tuple (levels, coverages) containing the coverages
        associated to different levels.
    """

    if prior:
        if model.sampling_enabled():
            model_distrib = model.get_prior_fct(config["nb_prior_estimators"])
        else:
            model_distrib = None

        def model_log_prob(theta, x):
            return model.prior_log_prob(theta, x, config["nb_prior_estimators"])

    else:
        if model.sampling_enabled():
            model_distrib = model.get_posterior_fct()
        else:
            model_distrib = None
        model_log_prob = model

    dataset = benchmark.get_coverage_set(config["coverage_set_size"])
    if bounds is not None:
        dataset = islice(dataset, bounds[0], bounds[1])

    if intrinsic:
        if model.sampling_enabled():
            return intrinsic_expected_coverage_mc(
                model_distrib,
                dataset,
                benchmark.get_prior(),
                n=benchmark.get_nb_cov_samples(),
            )
        else:
            return intrinsic_expected_coverage_ni(
                model_log_prob,
                dataset,
                benchmark.get_prior(),
                benchmark.get_domain(),
                bins=benchmark.get_cov_bins(),
                batch_size=config["coverage_batch_size"],
            )
    else:
        if model.sampling_enabled():
            return expected_coverage_mc(
                model_distrib, dataset, n=benchmark.get_nb_cov_samples()
            )
        else:
            return expected_coverage_ni(
                model_log_prob,
                dataset,
                benchmark.get_domain(),
                bins=benchmark.get_cov_bins(),
                batch_size=config["coverage_batch_size"],
            )


def compute_merged_coverages(levels, coverages):
    levels = torch.cat(
        [level if i == 0 else level[1:-1] for i, level in enumerate(levels)]
    )
    levels, _ = torch.sort(levels)
    coverages = torch.linspace(0, 1, steps=levels.shape[0])
    return levels, coverages


def compute_normalized_entropy_log_posterior(model, benchmark, config, bounds=None):
    """Compute the average entropy and normalized log posterior associated to nominal
    parameter value of a model.

    Args:
        model (Model): a model such as defined in models/base.py
        benchmark (Benchmark): a benchmark such as defined in benchmarks/base.py
        config (dict): a dictionary containing the configuration
        bounds (int, int): bounds on dataset indices on which to evaluate the entropy
        and normalized log posterior

    Returns:
        (float, float): a tuple (entropies, nominal_log_probs) containing the average
        entropy and normalized nominal log posterior density
    """

    dataset = benchmark.get_coverage_set(config["coverage_set_size"])
    if bounds is not None:
        dataset = islice(dataset, bounds[0], bounds[1])

    domain = benchmark.get_domain()
    bins = benchmark.get_cov_bins()

    nominal_log_probs = []
    entropies = []

    with torch.no_grad():
        for nominal_theta, x in tqdm(dataset, unit="pair"):
            _, log_probs = gridapply(lambda theta: model(theta, x), domain, bins=bins)

            lower, upper = domain
            dims = len(lower)

            if type(bins) is int:
                bins = [bins] * dims

            log_volume = np.log(
                np.prod(
                    [(u - l) / b for u, l, b in zip(upper.numpy(), lower.numpy(), bins)]
                )
            )

            nominal_log_prob = (
                model(nominal_theta.unsqueeze(0), x.unsqueeze(0)).squeeze(0).item()
            )
            normalizing_constant = (
                log_volume + log_probs.flatten().logsumexp(dim=0).item()
            )
            log_probs = log_probs - normalizing_constant
            nominal_log_prob = nominal_log_prob - normalizing_constant

            log_bin_probs = log_probs + log_volume
            entropy = -torch.sum(torch.exp(log_bin_probs) * log_bin_probs).item()

            nominal_log_probs.append(nominal_log_prob)
            entropies.append(entropy)

    return np.mean(entropies), np.mean(nominal_log_probs)


def merge_scalar_results(results):
    return np.mean(np.array(results))


def compute_log_posterior(model, benchmark, config):
    """Compute the average log posterior associated to nominal parameter value of a
    model.

    Args:
        model (Model): a model such as defined in models/base.py
        benchmark (Benchmark): a benchmark such as defined in benchmarks/base.py
        config (dict): a dictionary containing the configuration

    Returns:
        float: the average nominal log posterior density
    """

    nominal_log_probs = []
    dataset = benchmark.get_test_set(config["test_set_size"], config["test_batch_size"])

    with torch.no_grad():
        for nominal_theta, x in tqdm(dataset, unit="pair"):
            nominal_log_prob = model(nominal_theta, x)
            nominal_log_probs.append(nominal_log_prob)

    return torch.mean(torch.cat(nominal_log_probs)).item()


def compute_balancing_error(model, benchmark, config):
    """Compute the average balancing error of a model.

    Args:
        model (Model): a model such as defined in models/base.py
        benchmark (Benchmark): a benchmark such as defined in benchmarks/base.py
        config (dict): a dictionary containing the configuration

    Returns:
        float: the average balancing error
    """

    dataset = benchmark.get_test_set(config["test_set_size"], config["test_batch_size"])
    d_joints = []
    d_marginals = []
    prior = benchmark.get_prior()

    with torch.no_grad():
        for theta_joint, x in tqdm(dataset, unit="pair"):
            theta_marginal = torch.roll(theta_joint, 1, dims=0)
            d_joint = torch.sigmoid(
                model(theta_joint, x).cpu() - prior.log_prob(theta_joint)
            )
            d_marginal = torch.sigmoid(
                model(theta_marginal, x).cpu() - prior.log_prob(theta_marginal)
            )
            d_joints.append(d_joint)
            d_marginals.append(d_marginal)

        balancing = (
            torch.mean(torch.cat(d_joints)).item()
            + torch.mean(torch.cat(d_marginals)).item()
        )

    return np.absolute(1 - balancing)


def compute_prior_mixture_coef(model, benchmark, config):
    """Compute the mixture coef of prior augmented models.

    Args:
        model (Model): a model such as defined in models/base.py
        benchmark (Benchmark): a benchmark such as defined in benchmarks/base.py
        config (dict): a dictionary containing the configuration

    Returns:
        float: the mixture coef
    """

    prior_mixture_coefs = []
    dataset = benchmark.get_test_set(config["test_set_size"], config["test_batch_size"])

    with torch.no_grad():
        for nominal_theta, x in tqdm(dataset, unit="pair"):
            prior_mixture_coef = torch.sigmoid(
                model.prior_mixture(model.embedding(x)).squeeze()
            )
            prior_mixture_coefs.append(prior_mixture_coef)

    return torch.cat(prior_mixture_coefs)


def compute_aleatoric_mc(model, x, n):

    nb_networks = model.get_nb_networks()
    entropies = []

    for id in range(nb_networks):
        samples = model.sample(x, (n,), id)
        log_probs = model.log_prob(samples, x, id)
        # convert to bits for exact shannon entropy
        entropy = -torch.nanmean(log_probs) * torch.log(torch.tensor([2.0])).to(
            log_probs.device
        )
        entropies.append(entropy)

    return torch.mean(torch.stack(entropies))


def compute_uncertainty_mc(model, pairs, n):
    def uncertainty(theta, x):
        samples = model.sample(x, (n,))
        log_probs = model.log_prob(samples, x)
        # convert to bits for exact shannon entropy
        total = -torch.nanmean(log_probs) * torch.log(torch.tensor([2.0])).to(
            log_probs.device
        )
        if model.is_ensemble():
            aleatoric = compute_aleatoric_mc(model, x, n)
            epistemic = total - aleatoric
        else:
            aleatoric = total
            epistemic = torch.tensor(0.0)
        return total, aleatoric, epistemic

    totals = []
    aleatorics = []
    epistemics = []

    with torch.no_grad():
        for theta, x in tqdm(pairs, unit="pair"):

            total, aleatoric, epistemic = uncertainty(theta, x)

            totals.append(total)
            aleatorics.append(aleatoric)
            epistemics.append(epistemic)

    atu = torch.mean(torch.stack(totals))
    aau = torch.mean(torch.stack(aleatorics))
    aeu = torch.mean(torch.stack(epistemics))
    return atu, aau, aeu


def compute_entropy(log_probas):
    """
    Computes the entropy of a posterior distribution using the posterior probabilities.

    args:
        log_probas(Tensor): probabilities of the samples.
    """

    assert log_probas.dim() == 1, "log_probas must be a 1D tensor."

    probas = torch.exp(log_probas)
    entropy = -torch.nansum(probas * log_probas)
    # convert to bits for exact shannon entropy
    return entropy * torch.log(torch.tensor([2.0])).to(entropy.device)


def compute_aleatoric_ni(model, x, domain, **kwargs):
    nb_networks = model.get_nb_networks()

    entropies = []
    for id in range(nb_networks):

        def log_p(theta):
            return model.log_prob(theta, x, id)

        _, log_probs = gridapply(log_p, domain, **kwargs)
        volume = ((domain[1] - domain[0]) / log_probs[0].shape[0]).prod()
        entropy = compute_entropy(log_probs.flatten()) * volume
        entropies.append(entropy)

    return torch.mean(torch.stack(entropies))


def compute_uncertainty_ni(model, pairs, domain, **kwargs):
    def uncertainty(theta, x):
        _, log_probs = gridapply(lambda theta: log_p(theta, x), domain, bins=256)
        volume = ((domain[1] - domain[0]) / log_probs[0].shape[0]).prod()
        total = compute_entropy(log_probs.flatten()) * volume
        totals.append(total)

        if model.is_ensemble():
            aleatoric = compute_aleatoric_ni(model, x, domain, bins=256)
            epistemic = total - aleatoric
        else:
            aleatoric = total
            epistemic = torch.tensor(0.0)
        return total, aleatoric, epistemic

    totals = []
    aleatorics = []
    epistemics = []

    def log_p(theta, x):
        return model.log_prob(theta, x)

    with torch.no_grad():
        for theta, x in tqdm(pairs, unit="pair"):

            total, aleatoric, epistemic = uncertainty(theta, x)

            totals.append(total)
            aleatorics.append(aleatoric)
            epistemics.append(epistemic)

    atu = torch.mean(torch.stack(totals))
    aau = torch.mean(torch.stack(aleatorics))
    aeu = torch.mean(torch.stack(epistemics))
    return atu, aau, aeu


def compute_uncertainty(model, benchmark, config, **kwargs):
    """Compute the uncertainty of a model. If the model is an ensemble model (bayesian or non-bayesian), it will return the total, aleatoric and epistemic uncertainties. If the model is a non-Bayesian model, it will return the total uncertainty.

    Args:
        model (Model): a model such as defined in models/base.py
        benchmark (Benchmark): a benchmark such as defined in benchmarks/base.py
        config (dict): a dictionary containing the configuration

    Returns:
        float: the uncertainty
    """

    dataset = benchmark.get_coverage_set(config["coverage_set_size"])

    if model.sampling_enabled():
        atu, aau, aeu = compute_uncertainty_mc(
            model, dataset, n=benchmark.get_nb_cov_samples()
        )
    else:
        bounds = benchmark.get_domain()
        atu, aau, aeu = compute_uncertainty_ni(model, dataset, bounds, **kwargs)

    return atu.cpu(), aau.cpu(), aeu.cpu()
