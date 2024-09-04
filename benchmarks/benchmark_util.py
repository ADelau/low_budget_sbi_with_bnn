from .base import Benchmark
from .galaxies import Galaxies
from .lotka_volterra import LotkaVolterra
from .slcp import SLCP
from .spatialsir import SpatialSIR
from .two_moons import TwoMoons


def load_benchmark(config: dict) -> Benchmark:
    if config["benchmark"] == "slcp":
        return SLCP(config["data_path"])
    elif config["benchmark"] == "spatialsir":
        return SpatialSIR(config["data_path"])
    elif config["benchmark"] == "lotka_volterra":
        return LotkaVolterra(config["data_path"])
    elif config["benchmark"] == "two_moons":
        return TwoMoons(config["data_path"])
    elif config["benchmark"] == "galaxies":
        return Galaxies(config["data_path"])
    else:
        raise NotImplementedError("Benchmark not implemented")
