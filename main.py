import argparse
import glob
import math
import os
import random
import sys

import numpy as np
import torch
from dawgz import after, ensure, job, schedule

from benchmarks import Datasets, load_benchmark
from config_files import read_config
from diagnostics import (compute_balancing_error,
                         compute_coverage,
                         compute_log_posterior,
                         compute_merged_coverages,
                         compute_normalized_entropy_log_posterior,
                         compute_prior_mixture_coef,
                         compute_uncertainty,
                         merge_scalar_results,)
from models import load_model_factory

sys.path.append(os.path.dirname(os.path.realpath(__file__)))


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def seconds_to_time(seconds):
    seconds = math.ceil(seconds)
    minutes = seconds // 60
    seconds = seconds % 60

    hours = minutes // 60
    minutes = minutes % 60

    return "{}:{:02d}:{:02d}".format(hours, minutes, seconds)


if __name__ == "__main__":
    # Increase recursion depth (large workflow)
    sys.setrecursionlimit(10000)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, help="Config file path")
    parser.add_argument(
        "--slurm",
        action="store_true",
        help="Executes the workflow on a Slurm-enabled HPC system (default: false).",
    )

    arguments, _ = parser.parse_known_args()

    config = read_config(arguments.config_file)
    benchmark = load_benchmark(config)

    nb_train_blocks = math.ceil(config["train_set_size"] / config["block_size"])
    nb_val_blocks = math.ceil(config["val_set_size"] / config["block_size"])
    nb_test_blocks = math.ceil(config["test_set_size"] / config["block_size"])
    nb_coverage_blocks = math.ceil(config["coverage_set_size"] / config["block_size"])

    jobs = []
    diagnostics = config["diagnostics"]

    @ensure(
        lambda id: benchmark.is_block_simulated(
            config,
            Datasets.TRAIN,
            id % nb_train_blocks,
            dataset_id=id // nb_train_blocks,
        )
    )
    @job(
        array=nb_train_blocks * config["nb_runs"],
        cpus=benchmark.get_simulate_nb_cpus(),
        gpus=benchmark.get_simulate_nb_gpus(),
        ram="{}GB".format(math.ceil(benchmark.get_simulate_ram(config["block_size"]))),
        time=seconds_to_time(benchmark.get_simulate_time(config["block_size"])),
    )
    def simulate_train(id):
        if config["seed"] is not None:
            set_seed(config["seed"] + id)

        benchmark.simulate_block(
            config,
            Datasets.TRAIN,
            id % nb_train_blocks,
            dataset_id=id // nb_train_blocks,
        )

    jobs.append(simulate_train)

    @ensure(
        lambda id: benchmark.is_block_simulated(
            config, Datasets.VAL, id % nb_val_blocks, dataset_id=id // nb_val_blocks
        )
    )
    @job(
        array=nb_val_blocks * config["nb_runs"],
        cpus=benchmark.get_simulate_nb_cpus(),
        gpus=benchmark.get_simulate_nb_gpus(),
        ram="{}GB".format(math.ceil(benchmark.get_simulate_ram(config["block_size"]))),
        time=seconds_to_time(benchmark.get_simulate_time(config["block_size"])),
    )
    def simulate_val(id):
        if config["seed"] is not None:
            set_seed(config["seed"] + int(1e8) + id)

        benchmark.simulate_block(
            config, Datasets.VAL, id % nb_val_blocks, dataset_id=id // nb_val_blocks
        )

    jobs.append(simulate_val)

    @ensure(
        lambda block_id: benchmark.is_block_simulated(config, Datasets.TEST, block_id)
    )
    @job(
        array=nb_test_blocks,
        cpus=benchmark.get_simulate_nb_cpus(),
        gpus=benchmark.get_simulate_nb_gpus(),
        ram="{}GB".format(math.ceil(benchmark.get_simulate_ram(config["block_size"]))),
        time=seconds_to_time(benchmark.get_simulate_time(config["block_size"])),
    )
    def simulate_test(block_id):
        if config["seed"] is not None:
            set_seed(config["seed"] + int(2e8) + block_id)

        benchmark.simulate_block(config, Datasets.TEST, block_id)

    jobs.append(simulate_test)

    @ensure(
        lambda block_id: benchmark.is_block_simulated(
            config, Datasets.COVERAGE, block_id
        )
    )
    @job(
        array=nb_test_blocks,
        cpus=benchmark.get_simulate_nb_cpus(),
        gpus=benchmark.get_simulate_nb_gpus(),
        ram="{}GB".format(math.ceil(benchmark.get_simulate_ram(config["block_size"]))),
        time=seconds_to_time(benchmark.get_simulate_time(config["block_size"])),
    )
    def simulate_coverage(block_id):
        if config["seed"] is not None:
            set_seed(config["seed"] + int(3e8) + block_id)

        benchmark.simulate_block(config, Datasets.COVERAGE, block_id)

    jobs.append(simulate_coverage)

    @after(simulate_test)
    @ensure(
        lambda: benchmark.are_blocks_merged(
            config, Datasets.TEST, config["test_set_size"]
        )
    )
    @job(
        cpus=benchmark.get_merge_nb_cpus(),
        gpus=benchmark.get_merge_nb_gpus(),
        ram="{}GB".format(math.ceil(benchmark.get_merge_ram(config["test_set_size"]))),
        time=seconds_to_time(benchmark.get_merge_time(config["test_set_size"])),
    )
    def merge_test():
        benchmark.merge_blocks(
            config,
            Datasets.TEST,
            config["test_set_size"],
            [x for x in range(nb_test_blocks)],
        )

    jobs.append(merge_test)

    @after(simulate_coverage)
    @ensure(
        lambda: benchmark.are_blocks_merged(
            config, Datasets.COVERAGE, config["coverage_set_size"]
        )
    )
    @job(
        cpus=benchmark.get_merge_nb_cpus(),
        gpus=benchmark.get_merge_nb_gpus(),
        ram="{}GB".format(
            math.ceil(benchmark.get_merge_ram(config["coverage_set_size"]))
        ),
        time=seconds_to_time(benchmark.get_merge_time(config["coverage_set_size"])),
    )
    def merge_coverage():
        benchmark.merge_blocks(
            config,
            Datasets.COVERAGE,
            config["coverage_set_size"],
            [x for x in range(nb_test_blocks)],
        )

    jobs.append(merge_coverage)

    simulation_budgets = config["simulation_budgets"]

    for simulation_budget in simulation_budgets:
        val_fraction = config["val_fraction"]
        train_size = math.floor(simulation_budget * (1 - val_fraction))
        val_size = math.floor(simulation_budget * val_fraction)

        model_factory = load_model_factory(config, benchmark, simulation_budget)

        @after(simulate_train)
        # @context(train_size=train_size)
        @ensure(
            lambda dataset_id: benchmark.are_blocks_merged(
                config, Datasets.TRAIN, train_size, dataset_id=dataset_id
            )
        )
        @job(
            array=config["nb_runs"],
            cpus=benchmark.get_merge_nb_cpus(),
            gpus=benchmark.get_merge_nb_gpus(),
            ram="{}GB".format(math.ceil(benchmark.get_merge_ram(train_size))),
            time=seconds_to_time(benchmark.get_merge_time(train_size)),
        )
        def merge_train(dataset_id):
            benchmark.merge_blocks(
                config,
                Datasets.TRAIN,
                train_size,
                [x for x in range(nb_train_blocks)],
                dataset_id=dataset_id,
            )

        jobs.append(merge_train)

        @after(simulate_val)
        # @context(val_size=val_size)
        @ensure(
            lambda dataset_id: benchmark.are_blocks_merged(
                config, Datasets.VAL, val_size, dataset_id=dataset_id
            )
        )
        @job(
            array=config["nb_runs"],
            cpus=benchmark.get_merge_nb_cpus(),
            gpus=benchmark.get_merge_nb_gpus(),
            ram="{}GB".format(math.ceil(benchmark.get_merge_ram(val_size))),
            time=seconds_to_time(benchmark.get_merge_time(val_size)),
        )
        def merge_val(dataset_id):
            benchmark.merge_blocks(
                config,
                Datasets.VAL,
                val_size,
                [x for x in range(nb_val_blocks)],
                dataset_id=dataset_id,
            )

        jobs.append(merge_val)

        @after(merge_train, merge_val)
        # @context(train_size=train_size, val_size=val_size, model_factory=model_factory)
        @ensure(lambda job_id: model_factory.is_initialized(job_id))
        @job(
            array=config["nb_runs"],
            cpus=model_factory.get_train_nb_cpus(benchmark.get_train_nb_cpus()),
            gpus=model_factory.get_train_nb_gpus(benchmark.get_train_nb_gpus()),
            ram="{}GB".format(
                math.ceil(model_factory.get_init_ram(benchmark.get_init_ram()))
            ),
            time=seconds_to_time(
                model_factory.get_init_time(benchmark.get_init_time())
            ),
        )
        def initialize(job_id):
            if config["seed"] is not None:
                set_seed(config["seed"] + simulation_budget + job_id)

            batch_size = config["train_batch_size"]
            if isinstance(batch_size, list):
                batch_size = batch_size[-1]

            model = model_factory.instantiate_model(
                job_id, benchmark.get_normalization_constants(train_size, job_id)
            )
            model.initialize(
                benchmark.get_train_set(train_size, batch_size, job_id),
                benchmark.get_val_set(val_size, batch_size, job_id),
                config,
            )
            model.save_init()

        jobs.append(initialize)

        nb_parallel_coverages = config["nb_parallel_coverages"]
        total_coverage_runs = config["nb_runs"] * nb_parallel_coverages

        if nb_parallel_coverages > 1:

            def prior_coverage_ensure_fct(job_id):
                return os.path.exists(
                    os.path.join(
                        model_factory.get_model_path(job_id // nb_parallel_coverages),
                        "prior_levels_{}.pt".format(job_id % nb_parallel_coverages),
                    )
                ) and os.path.exists(
                    os.path.join(
                        model_factory.get_model_path(job_id // nb_parallel_coverages),
                        "prior_coverages_{}.pt".format(job_id % nb_parallel_coverages),
                    )
                )

        else:

            def prior_coverage_ensure_fct(job_id):
                return os.path.exists(
                    os.path.join(
                        model_factory.get_model_path(job_id), "prior_levels.pt"
                    )
                ) and os.path.exists(
                    os.path.join(
                        model_factory.get_model_path(job_id), "prior_coverages.pt"
                    )
                )

        @after(initialize, merge_coverage)
        # @context(model_factory=model_factory)
        @ensure(prior_coverage_ensure_fct)
        @job(
            array=total_coverage_runs,
            cpus=model_factory.get_coverage_nb_cpus(benchmark.get_coverage_nb_cpus()),
            gpus=model_factory.get_coverage_nb_gpus(benchmark.get_coverage_nb_gpus()),
            ram="{}GB".format(
                math.ceil(model_factory.get_coverage_ram(benchmark.get_coverage_ram()))
            ),
            time=seconds_to_time(
                model_factory.get_coverage_time(
                    benchmark.get_coverage_time(
                        config["coverage_set_size"] / nb_parallel_coverages
                    )
                )
            ),
        )
        def prior_coverage(job_id):
            model_id = job_id // nb_parallel_coverages
            model = model_factory.instantiate_model(
                model_id, benchmark.get_normalization_constants(train_size, model_id)
            )
            model.load_init()
            model.to(benchmark.get_device())
            model.eval()

            if nb_parallel_coverages > 1:
                coverage_index = job_id % nb_parallel_coverages
                start = coverage_index * (
                    config["coverage_set_size"] // nb_parallel_coverages
                )
                stop = (coverage_index + 1) * (
                    config["coverage_set_size"] // nb_parallel_coverages
                )
                levels, coverages = compute_coverage(
                    model, benchmark, config, bounds=(start, stop), prior=True
                )
                torch.save(
                    levels,
                    os.path.join(
                        model_factory.get_model_path(model_id),
                        "prior_levels_{}.pt".format(coverage_index),
                    ),
                )
                torch.save(
                    coverages,
                    os.path.join(
                        model_factory.get_model_path(model_id),
                        "prior_coverages_{}.pt".format(coverage_index),
                    ),
                )
            else:
                levels, coverages = compute_coverage(
                    model, benchmark, config, prior=True
                )
                torch.save(
                    levels,
                    os.path.join(
                        model_factory.get_model_path(model_id), "prior_levels.pt"
                    ),
                )
                torch.save(
                    coverages,
                    os.path.join(
                        model_factory.get_model_path(model_id), "prior_coverages.pt"
                    ),
                )

        @after(prior_coverage)
        # @context(model_factory=model_factory)
        @ensure(
            lambda job_id: os.path.exists(
                os.path.join(model_factory.get_model_path(job_id), "prior_levels.pt")
            )
            and os.path.exists(
                os.path.join(model_factory.get_model_path(job_id), "prior_coverages.pt")
            )
        )
        @job(
            array=config["nb_runs"],
            cpus=1,
            gpus=0,
            ram="{}GB".format(
                math.ceil(model_factory.get_coverage_ram(benchmark.get_coverage_ram()))
            ),
            time="1:00:00",
        )
        def merge_prior_coverage_results(job_id):
            levels = [
                torch.load(
                    os.path.join(
                        model_factory.get_model_path(job_id),
                        "prior_levels_{}.pt".format(coverage_index),
                    )
                )
                for coverage_index in range(nb_parallel_coverages)
            ]
            coverages = [
                torch.load(
                    os.path.join(
                        model_factory.get_model_path(job_id),
                        "prior_coverages_{}.pt".format(coverage_index),
                    )
                )
                for coverage_index in range(nb_parallel_coverages)
            ]

            levels, coverages = compute_merged_coverages(levels, coverages)

            torch.save(
                levels,
                os.path.join(model_factory.get_model_path(job_id), "prior_levels.pt"),
            )
            torch.save(
                coverages,
                os.path.join(
                    model_factory.get_model_path(job_id), "prior_coverages.pt"
                ),
            )

        if "prior_coverage" in diagnostics:
            jobs.append(prior_coverage)
            if nb_parallel_coverages > 1:
                jobs.append(merge_prior_coverage_results)

        nb_trainings = config["nb_runs"]
        if model_factory.require_multiple_trainings():
            nb_trainings *= model_factory.nb_trainings_required()

        @after(merge_train, merge_val, initialize)
        # @context(train_size=train_size, val_size=val_size, model_factory=model_factory)
        @ensure(lambda job_id: model_factory.is_trained(job_id))
        @job(
            array=nb_trainings,
            cpus=model_factory.get_train_nb_cpus(benchmark.get_train_nb_cpus()),
            gpus=model_factory.get_train_nb_gpus(benchmark.get_train_nb_gpus()),
            ram="{}GB".format(
                math.ceil(model_factory.get_train_ram(benchmark.get_train_ram()))
            ),
            time=seconds_to_time(
                model_factory.get_train_time(
                    benchmark.get_train_time(train_size + val_size), config["epochs"]
                )
            ),
        )
        def train(job_id):
            if config["seed"] is not None:
                set_seed(config["seed"] + simulation_budget + job_id)

            batch_size = config["train_batch_size"]
            if isinstance(batch_size, list):
                for i in range(len(simulation_budgets)):
                    if simulation_budgets[i] == simulation_budget:
                        batch_size = batch_size[i]
                        break

            if simulation_budget < batch_size:
                batch_size = simulation_budget

            if model_factory.require_multiple_trainings():
                nb_trainings = model_factory.nb_trainings_required()
                model_id = job_id // nb_trainings
                model = model_factory.instantiate_model(
                    model_id,
                    benchmark.get_normalization_constants(train_size, model_id),
                )
                model.load_init()
                model.train_models(
                    benchmark.get_train_set(train_size, batch_size, model_id),
                    benchmark.get_val_set(val_size, batch_size, model_id),
                    config,
                    job_id % nb_trainings,
                )
                model.save()
            else:
                model = model_factory.instantiate_model(
                    job_id, benchmark.get_normalization_constants(train_size, job_id)
                )
                model.load_init()
                model.train_models(
                    benchmark.get_train_set(train_size, batch_size, job_id),
                    benchmark.get_val_set(val_size, batch_size, job_id),
                    config,
                )
                model.save()

        jobs.append(train)

        if nb_parallel_coverages > 1:

            def coverage_ensure_fct(job_id):
                return os.path.exists(
                    os.path.join(
                        model_factory.get_model_path(job_id // nb_parallel_coverages),
                        "levels_{}.pt".format(job_id % nb_parallel_coverages),
                    )
                ) and os.path.exists(
                    os.path.join(
                        model_factory.get_model_path(job_id // nb_parallel_coverages),
                        "coverages_{}.pt".format(job_id % nb_parallel_coverages),
                    )
                )

        else:

            def coverage_ensure_fct(job_id):
                return os.path.exists(
                    os.path.join(model_factory.get_model_path(job_id), "levels.pt")
                ) and os.path.exists(
                    os.path.join(model_factory.get_model_path(job_id), "coverages.pt")
                )

        @after(train, merge_coverage)
        # @context(model_factory=model_factory)
        @ensure(coverage_ensure_fct)
        @job(
            array=total_coverage_runs,
            cpus=model_factory.get_coverage_nb_cpus(benchmark.get_coverage_nb_cpus()),
            gpus=model_factory.get_coverage_nb_gpus(benchmark.get_coverage_nb_gpus()),
            ram="{}GB".format(
                math.ceil(model_factory.get_coverage_ram(benchmark.get_coverage_ram()))
            ),
            time=seconds_to_time(
                model_factory.get_coverage_time(
                    benchmark.get_coverage_time(
                        config["coverage_set_size"] / nb_parallel_coverages
                    )
                )
            ),
        )
        def coverage(job_id):
            model_id = job_id // nb_parallel_coverages
            model = model_factory.instantiate_model(
                model_id, benchmark.get_normalization_constants(train_size, model_id)
            )
            model.load()
            model.to(benchmark.get_device())
            model.eval()

            if nb_parallel_coverages > 1:
                coverage_index = job_id % nb_parallel_coverages
                start = coverage_index * (
                    config["coverage_set_size"] // nb_parallel_coverages
                )
                stop = (coverage_index + 1) * (
                    config["coverage_set_size"] // nb_parallel_coverages
                )
                levels, coverages = compute_coverage(
                    model, benchmark, config, bounds=(start, stop)
                )
                torch.save(
                    levels,
                    os.path.join(
                        model_factory.get_model_path(model_id),
                        "levels_{}.pt".format(coverage_index),
                    ),
                )
                torch.save(
                    coverages,
                    os.path.join(
                        model_factory.get_model_path(model_id),
                        "coverages_{}.pt".format(coverage_index),
                    ),
                )
            else:
                levels, coverages = compute_coverage(model, benchmark, config)
                torch.save(
                    levels,
                    os.path.join(model_factory.get_model_path(model_id), "levels.pt"),
                )
                torch.save(
                    coverages,
                    os.path.join(
                        model_factory.get_model_path(model_id), "coverages.pt"
                    ),
                )

        @after(coverage)
        # @context(model_factory=model_factory)
        @ensure(
            lambda job_id: os.path.exists(
                os.path.join(model_factory.get_model_path(job_id), "levels.pt")
            )
            and os.path.exists(
                os.path.join(model_factory.get_model_path(job_id), "coverages.pt")
            )
        )
        @job(
            array=config["nb_runs"],
            cpus=1,
            gpus=0,
            ram="{}GB".format(
                math.ceil(model_factory.get_coverage_ram(benchmark.get_coverage_ram()))
            ),
            time="1:00:00",
        )
        def merge_coverage_results(job_id):
            levels = [
                torch.load(
                    os.path.join(
                        model_factory.get_model_path(job_id),
                        "levels_{}.pt".format(coverage_index),
                    )
                )
                for coverage_index in range(nb_parallel_coverages)
            ]
            coverages = [
                torch.load(
                    os.path.join(
                        model_factory.get_model_path(job_id),
                        "coverages_{}.pt".format(coverage_index),
                    )
                )
                for coverage_index in range(nb_parallel_coverages)
            ]

            levels, coverages = compute_merged_coverages(levels, coverages)

            torch.save(
                levels, os.path.join(model_factory.get_model_path(job_id), "levels.pt")
            )
            torch.save(
                coverages,
                os.path.join(model_factory.get_model_path(job_id), "coverages.pt"),
            )

        if "coverage" in diagnostics:
            jobs.append(coverage)
            if nb_parallel_coverages > 1:
                jobs.append(merge_coverage_results)

        if nb_parallel_coverages > 1:

            def intrinsic_coverage_ensure_fct(job_id):
                return os.path.exists(
                    os.path.join(
                        model_factory.get_model_path(job_id // nb_parallel_coverages),
                        "intrinsic_levels_{}.pt".format(job_id % nb_parallel_coverages),
                    )
                ) and os.path.exists(
                    os.path.join(
                        model_factory.get_model_path(job_id // nb_parallel_coverages),
                        "intrinsic_coverages_{}.pt".format(
                            job_id % nb_parallel_coverages
                        ),
                    )
                )

        else:

            def intrinsic_coverage_ensure_fct(job_id):
                return os.path.exists(
                    os.path.join(
                        model_factory.get_model_path(job_id), "intrinsic_levels.pt"
                    )
                ) and os.path.exists(
                    os.path.join(
                        model_factory.get_model_path(job_id), "intrinsic_coverages.pt"
                    )
                )

        @after(train, merge_coverage)
        # @context(model_factory=model_factory)
        @ensure(intrinsic_coverage_ensure_fct)
        @job(
            array=total_coverage_runs,
            cpus=model_factory.get_coverage_nb_cpus(benchmark.get_coverage_nb_cpus()),
            gpus=model_factory.get_coverage_nb_gpus(benchmark.get_coverage_nb_gpus()),
            ram="{}GB".format(
                math.ceil(model_factory.get_coverage_ram(benchmark.get_coverage_ram()))
            ),
            time=seconds_to_time(
                model_factory.get_coverage_time(
                    benchmark.get_coverage_time(
                        config["coverage_set_size"] / nb_parallel_coverages
                    )
                )
            ),
        )
        def intrinsic_coverage(job_id):
            model_id = job_id // nb_parallel_coverages
            model = model_factory.instantiate_model(
                model_id, benchmark.get_normalization_constants(train_size, model_id)
            )
            model.load()
            model.to(benchmark.get_device())
            model.eval()

            if nb_parallel_coverages > 1:
                coverage_index = job_id % nb_parallel_coverages
                start = coverage_index * (
                    config["coverage_set_size"] // nb_parallel_coverages
                )
                stop = (coverage_index + 1) * (
                    config["coverage_set_size"] // nb_parallel_coverages
                )
                levels, coverages = compute_coverage(
                    model, benchmark, config, bounds=(start, stop), intrinsic=True
                )
                torch.save(
                    levels,
                    os.path.join(
                        model_factory.get_model_path(model_id),
                        "intrinsic_levels_{}.pt".format(coverage_index),
                    ),
                )
                torch.save(
                    coverages,
                    os.path.join(
                        model_factory.get_model_path(model_id),
                        "intrinsic_coverages_{}.pt".format(coverage_index),
                    ),
                )
            else:
                levels, coverages = compute_coverage(
                    model, benchmark, config, intrinsic=True
                )
                torch.save(
                    levels,
                    os.path.join(
                        model_factory.get_model_path(model_id), "intrinsic_levels.pt"
                    ),
                )
                torch.save(
                    coverages,
                    os.path.join(
                        model_factory.get_model_path(model_id), "intrinsic_coverages.pt"
                    ),
                )

        @after(intrinsic_coverage)
        # @context(model_factory=model_factory)
        @ensure(
            lambda job_id: os.path.exists(
                os.path.join(
                    model_factory.get_model_path(job_id), "intrinsic_levels.pt"
                )
            )
            and os.path.exists(
                os.path.join(
                    model_factory.get_model_path(job_id), "intrinsic_coverages.pt"
                )
            )
        )
        @job(
            array=config["nb_runs"],
            cpus=1,
            gpus=0,
            ram="{}GB".format(
                math.ceil(model_factory.get_coverage_ram(benchmark.get_coverage_ram()))
            ),
            time="1:00:00",
        )
        def merge_intrinsic_coverage_results(job_id):
            levels = [
                torch.load(
                    os.path.join(
                        model_factory.get_model_path(job_id),
                        "intrinsic_levels_{}.pt".format(coverage_index),
                    )
                )
                for coverage_index in range(nb_parallel_coverages)
            ]
            coverages = [
                torch.load(
                    os.path.join(
                        model_factory.get_model_path(job_id),
                        "intrinsic_coverages_{}.pt".format(coverage_index),
                    )
                )
                for coverage_index in range(nb_parallel_coverages)
            ]

            levels, coverages = compute_merged_coverages(levels, coverages)

            torch.save(
                levels,
                os.path.join(
                    model_factory.get_model_path(job_id), "intrinsic_levels.pt"
                ),
            )
            torch.save(
                coverages,
                os.path.join(
                    model_factory.get_model_path(job_id), "intrinsic_coverages.pt"
                ),
            )

        if "intrinsic_coverage" in diagnostics:
            jobs.append(intrinsic_coverage)
            if nb_parallel_coverages > 1:
                jobs.append(merge_intrinsic_coverage_results)

        if nb_parallel_coverages > 1:

            def posterior_entropy_ensure_fct(job_id):
                return (
                    os.path.exists(
                        os.path.join(
                            model_factory.get_model_path(
                                job_id // nb_parallel_coverages
                            ),
                            "entropy_{}.pt".format(job_id % nb_parallel_coverages),
                        )
                    )
                    or "entropy" not in diagnostics
                ) and (
                    os.path.exists(
                        os.path.join(
                            model_factory.get_model_path(
                                job_id // nb_parallel_coverages
                            ),
                            "normalized_nominal_log_prob_{}.pt".format(
                                job_id % nb_parallel_coverages
                            ),
                        )
                    )
                    or "normalized_nominal_log_prob" not in diagnostics
                )

        else:

            def posterior_entropy_ensure_fct(job_id):
                return (
                    os.path.exists(
                        os.path.join(model_factory.get_model_path(job_id), "entropy.pt")
                    )
                    or "entropy" not in diagnostics
                ) and (
                    os.path.exists(
                        os.path.join(
                            model_factory.get_model_path(job_id),
                            "normalized_nominal_log_prob.pt",
                        )
                    )
                    or "normalized_nominal_log_prob" not in diagnostics
                )

        @after(train, merge_coverage)
        # @context(model_factory=model_factory)
        @ensure(posterior_entropy_ensure_fct)
        @job(
            array=total_coverage_runs,
            cpus=model_factory.get_coverage_nb_cpus(benchmark.get_coverage_nb_cpus()),
            gpus=model_factory.get_coverage_nb_gpus(benchmark.get_coverage_nb_gpus()),
            ram="{}GB".format(
                math.ceil(model_factory.get_coverage_ram(benchmark.get_coverage_ram()))
            ),
            time=seconds_to_time(
                model_factory.get_coverage_time(
                    benchmark.get_coverage_time(
                        config["coverage_set_size"] / nb_parallel_coverages
                    )
                )
            ),
        )
        def normalized_entropy_log_posterior(job_id):
            model_id = job_id // nb_parallel_coverages
            model = model_factory.instantiate_model(
                model_id, benchmark.get_normalization_constants(train_size, model_id)
            )
            model.load()
            model.to(benchmark.get_device())
            model.eval()

            if nb_parallel_coverages > 1:
                coverage_index = job_id % nb_parallel_coverages
                start = coverage_index * (
                    config["coverage_set_size"] // nb_parallel_coverages
                )
                stop = (coverage_index + 1) * (
                    config["coverage_set_size"] // nb_parallel_coverages
                )
                (
                    entropy,
                    normalized_nominal_log_prob,
                ) = compute_normalized_entropy_log_posterior(
                    model, benchmark, config, bounds=(start, stop)
                )

                if "entropy" in diagnostics:
                    torch.save(
                        entropy,
                        os.path.join(
                            model_factory.get_model_path(model_id),
                            "entropy_{}.pt".format(coverage_index),
                        ),
                    )
                if "normalized_nominal_log_prob" in diagnostics:
                    torch.save(
                        normalized_nominal_log_prob,
                        os.path.join(
                            model_factory.get_model_path(model_id),
                            "normalized_nominal_log_prob_{}.pt".format(coverage_index),
                        ),
                    )

            else:
                (
                    entropy,
                    normalized_nominal_log_prob,
                ) = compute_normalized_entropy_log_posterior(model, benchmark, config)

                if "entropy" in diagnostics:
                    torch.save(
                        entropy,
                        os.path.join(
                            model_factory.get_model_path(model_id), "entropy.pt"
                        ),
                    )
                if "normalized_nominal_log_prob" in diagnostics:
                    torch.save(
                        normalized_nominal_log_prob,
                        os.path.join(
                            model_factory.get_model_path(model_id),
                            "normalized_nominal_log_prob.pt",
                        ),
                    )

        @after(normalized_entropy_log_posterior)
        # @context(model_factory=model_factory)
        @ensure(
            lambda job_id: (
                os.path.exists(
                    os.path.join(model_factory.get_model_path(job_id), "entropy.pt")
                )
                or "entropy" not in diagnostics
            )
            and (
                os.path.exists(
                    os.path.join(
                        model_factory.get_model_path(job_id),
                        "normalized_nominal_log_prob.pt",
                    )
                )
                or "normalized_nominal_log_prob" not in diagnostics
            )
        )
        @job(
            array=config["nb_runs"],
            cpus=1,
            gpus=0,
            ram="{}GB".format(
                math.ceil(model_factory.get_coverage_ram(benchmark.get_coverage_ram()))
            ),
            time="1:00:00",
        )
        def merge_normalized_entropy_log_posterior_results(job_id):
            if "entropy" in diagnostics:
                entropy = [
                    torch.load(
                        os.path.join(
                            model_factory.get_model_path(job_id),
                            "entropy_{}.pt".format(coverage_index),
                        )
                    )
                    for coverage_index in range(nb_parallel_coverages)
                ]
                entropy = merge_scalar_results(entropy)
                torch.save(
                    entropy,
                    os.path.join(model_factory.get_model_path(job_id), "entropy.pt"),
                )

            if "normalized_nominal_log_prob" in diagnostics:
                normalized_nominal_log_prob = [
                    torch.load(
                        os.path.join(
                            model_factory.get_model_path(job_id),
                            "normalized_nominal_log_prob_{}.pt".format(coverage_index),
                        )
                    )
                    for coverage_index in range(nb_parallel_coverages)
                ]
                normalized_nominal_log_prob = merge_scalar_results(
                    normalized_nominal_log_prob
                )
                torch.save(
                    normalized_nominal_log_prob,
                    os.path.join(
                        model_factory.get_model_path(job_id),
                        "normalized_nominal_log_prob.pt",
                    ),
                )

        if "entropy" in diagnostics or "normalized_nominal_log_prob" in diagnostics:
            jobs.append(normalized_entropy_log_posterior)
            if nb_parallel_coverages > 1:
                jobs.append(merge_normalized_entropy_log_posterior_results)

        @after(train, merge_test)
        # @context(model_factory=model_factory)
        @ensure(
            lambda job_id: os.path.exists(
                os.path.join(
                    model_factory.get_model_path(job_id), "nominal_log_prob.pt"
                )
            )
        )
        @job(
            array=config["nb_runs"],
            cpus=model_factory.get_test_nb_cpus(benchmark.get_test_nb_cpus()),
            gpus=model_factory.get_test_nb_gpus(benchmark.get_test_nb_gpus()),
            ram="{}GB".format(
                math.ceil(model_factory.get_test_ram(benchmark.get_test_ram()))
            ),
            time=seconds_to_time(
                model_factory.get_test_time(
                    benchmark.get_test_time(config["test_set_size"])
                )
            ),
        )
        def log_posterior(job_id):
            model = model_factory.instantiate_model(
                job_id, benchmark.get_normalization_constants(train_size, job_id)
            )
            model.load()
            model.to(benchmark.get_device())
            model.eval()

            nominal_log_prob = compute_log_posterior(model, benchmark, config)
            torch.save(
                nominal_log_prob,
                os.path.join(
                    model_factory.get_model_path(job_id), "nominal_log_prob.pt"
                ),
            )

        if "nominal_log_prob" in diagnostics:
            jobs.append(log_posterior)

        @after(train, merge_test)
        # @context(model_factory=model_factory)
        @ensure(
            lambda job_id: os.path.exists(
                os.path.join(model_factory.get_model_path(job_id), "balancing_error.pt")
            )
        )
        @job(
            array=config["nb_runs"],
            cpus=model_factory.get_test_nb_cpus(benchmark.get_test_nb_cpus()),
            gpus=model_factory.get_test_nb_gpus(benchmark.get_test_nb_gpus()),
            ram="{}GB".format(
                math.ceil(model_factory.get_test_ram(benchmark.get_test_ram()))
            ),
            time=seconds_to_time(
                model_factory.get_test_time(
                    benchmark.get_test_time(config["test_set_size"])
                )
            ),
        )
        def balancing_error(job_id):
            model = model_factory.instantiate_model(
                job_id, benchmark.get_normalization_constants(train_size, job_id)
            )
            model.load()
            model.to(benchmark.get_device())
            model.eval()

            balancing_error = compute_balancing_error(model, benchmark, config)
            torch.save(
                balancing_error,
                os.path.join(
                    model_factory.get_model_path(job_id), "balancing_error.pt"
                ),
            )

        if "balancing_error" in diagnostics:
            jobs.append(balancing_error)

        @after(train, merge_test)
        # @context(model_factory=model_factory)
        @ensure(
            lambda job_id: os.path.exists(
                os.path.join(
                    model_factory.get_model_path(job_id), "prior_mixture_coef.pt"
                )
            )
        )
        @job(
            array=config["nb_runs"],
            cpus=model_factory.get_test_nb_cpus(benchmark.get_test_nb_cpus()),
            gpus=model_factory.get_test_nb_gpus(benchmark.get_test_nb_gpus()),
            ram="{}GB".format(
                math.ceil(model_factory.get_test_ram(benchmark.get_test_ram()))
            ),
            time=seconds_to_time(
                model_factory.get_test_time(
                    benchmark.get_test_time(config["test_set_size"])
                )
            ),
        )
        def prior_mixture_coef(job_id):
            model = model_factory.instantiate_model(
                job_id, benchmark.get_normalization_constants(train_size, job_id)
            )
            model.load()
            model.to(benchmark.get_device())
            model.eval()

            prior_mixture_coef = compute_prior_mixture_coef(model, benchmark, config)
            torch.save(
                prior_mixture_coef,
                os.path.join(
                    model_factory.get_model_path(job_id), "prior_mixture_coef.pt"
                ),
            )

        def uncertainty_ensure_fct(job_id):
            file1 = os.path.exists(
                os.path.join(
                    model_factory.get_model_path(job_id), "total_uncertainty.pt"
                )
            )
            file2 = os.path.exists(
                os.path.join(
                    model_factory.get_model_path(job_id), "aleatoric_uncertainty.pt"
                )
            )
            file3 = os.path.exists(
                os.path.join(
                    model_factory.get_model_path(job_id), "epistemic_uncertainty.pt"
                )
            )
            return file1 and file2 and file3

        @after(train, merge_coverage)
        @ensure(uncertainty_ensure_fct)
        @job(
            array=config["nb_runs"],
            cpus=model_factory.get_coverage_nb_cpus(benchmark.get_coverage_nb_cpus()),
            gpus=model_factory.get_coverage_nb_gpus(benchmark.get_coverage_nb_gpus()),
            ram="{}GB".format(
                math.ceil(model_factory.get_coverage_ram(benchmark.get_coverage_ram()))
            ),
            time=seconds_to_time(
                model_factory.get_coverage_time(
                    benchmark.get_coverage_time(config["coverage_set_size"])
                )
            ),
        )
        def uncertainty(job_id):
            model = model_factory.instantiate_model(
                job_id, benchmark.get_normalization_constants(train_size, job_id)
            )
            model.load()
            model.to(benchmark.get_device())
            model.eval()

            atu, aau, aeu = compute_uncertainty(model, benchmark, config)
            torch.save(
                atu,
                os.path.join(
                    model_factory.get_model_path(job_id), "total_uncertainty.pt"
                ),
            )
            torch.save(
                aau,
                os.path.join(
                    model_factory.get_model_path(job_id), "aleatoric_uncertainty.pt"
                ),
            )
            torch.save(
                aeu,
                os.path.join(
                    model_factory.get_model_path(job_id), "epistemic_uncertainty.pt"
                ),
            )

        if "uncertainty" in diagnostics:
            jobs.append(uncertainty)

    backend = "slurm" if arguments.slurm else "async"
    schedule(*jobs, name="simple.py", backend=backend, prune=True)
