# Low-Budget Simulation-Based Inference with Bayesian Neural Networks

## Abstract
Simulation-based inference methods have been shown to be inaccurate in the data-poor regime, when training simulations are limited or expensive. Under these circumstances, the inference network is particularly prone to overfitting, and using it without accounting for the computational uncertainty arising from the lack of identifiability of the network weights can lead to unreliable results. To address this issue, we propose using Bayesian neural networks in low-budget simulation-based inference, thereby explicitly accounting for the computational uncertainty of the posterior approximation. We design a family of Bayesian neural network priors that are tailored for inference and show that they lead to well-calibrated posteriors on tested benchmarks, even when as few as $O(10)$ simulations are available. This opens up the possibility of performing reliable simulation-based inference using very expensive simulators, as we demonstrate on a problem from the field of cosmology where single simulations are computationally expensive. We show that Bayesian neural networks produce informative and well-calibrated posterior estimates with only a few hundred simulations.

A PDF render of the manuscript is available on [`ArXiV`](https://arxiv.org/abs/2408.15136).

## Reproducing the experiments
First, install all the dependencies from the [requirements.txt](requirements.txt) file. The pipelines performing the experiments can then be executed by running the following command
```
python main.py --config_file <config_file_path>
```
Config files can be found [here](config_files). Optionally, the `--slurm` argument can be added to run on a slurm cluster.
