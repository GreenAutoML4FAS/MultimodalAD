"""Contains the automl training of the normalizing flow model."""
from ConfigSpace import Configuration, ConfigurationSpace

from smac import HyperparameterOptimizationFacade, Scenario

import random
from pathlib import Path
from typing import Optional

import numpy
import torch.backends.cudnn

from configuration import Configuration

from train import train
from voraus_ad import Signals, load_torch_dataloaders


# If deterministic CUDA is activated, some calculations cannot be calculated in parallel on the GPU.
# The training will take much longer but is reproducible.
DETERMINISTIC_CUDA = False
DATASET_PATH = Path.home() / "Downloads" / "voraus-ad-dataset-100hz.parquet"
MODEL_PATH: Optional[Path] = Path.cwd() / "model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Make the training reproducible.

if DETERMINISTIC_CUDA:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Disable pylint too-many-variables here for readability.
# The whole training should run in a single function call.

configspace = ConfigurationSpace({
    "ratio": 1.0,
    "epochs": (40, 100),  # 70,
    "frequencyDivider": [1, 2, 4],  # 1,
    "trainGain": 1.0,
    "batchsize": (16, 64),  # 32,
    "nCouplingBlocks": (2, 8),  # 4
    "clamp": (1.0, 2.0),  # 1.2,
    "learningRate": (8e-5, 8e-3),  # 8e-4,
    "normalize": [False, True],  # True,
    "nHiddenLayers": (0, 2),  # 0,
    "scale": (1, 4),  # 2,
    "kernelSize1": 13,  # 13,
    "dilation1": 2,  # 2,
    "kernelSize2": 1,  # 1,
    "dilation2": 1,  # 1,
    "kernelSize3": 1,  # 1,
    "dilation3": 1,  # 1,
    "milestones1": 11,  # [11, 61],
    "milestones2": 61,  # [11, 61],
    "gamma": (0.05, 0.2),  # 0.1,
})


def run_iteration(config: Configuration, seed) -> float:
    # Define the training configuration and hyperparameters of the model.
    configuration = Configuration(
        columns="machine",
        seed=seed,
        epochs=config["epochs"],
        frequencyDivider=config["frequencyDivider"],
        trainGain=config["trainGain"],
        batchsize=config["batchsize"],
        nCouplingBlocks=config["nCouplingBlocks"],
        clamp=config["clamp"],
        learningRate=config["learningRate"],
        normalize=config["normalize"],
        pad=True,
        nHiddenLayers=config["nHiddenLayers"],
        scale=config["scale"],
        kernelSize1=config["kernelSize1"],
        dilation1=config["dilation1"],
        kernelSize2=config["kernelSize2"],
        dilation2=config["dilation2"],
        kernelSize3=config["kernelSize3"],
        dilation3=config["dilation3"],
        milestones=[config["milestones1"], config["milestones2"]],
        gamma=config["gamma"],
        ratio=config["ratio"],
    )
    torch.manual_seed(configuration.seed)
    torch.cuda.manual_seed_all(configuration.seed)
    numpy.random.seed(configuration.seed)
    random.seed(configuration.seed)

    """Run a single iteration of the training."""
    training_results = train(configuration)
    auroc_mean = training_results[-1]["aurocMean"]
    return 1 - auroc_mean


if __name__ == "__main__":
    # Scenario object specifying the optimization environment
    print("RATIO:", configspace.sample_configuration()["ratio"],)
    scenario = Scenario(configspace, deterministic=False, n_trials=100)

    # Use SMAC to find the best configuration/hyperparameters
    smac = HyperparameterOptimizationFacade(scenario, run_iteration)
    incumbent = smac.optimize()

    # Let's calculate the cost of the incumbent
    print("==== Finished optimization ====")
    print("Incumbent:", incumbent)

    auroc_mean = []
    incumbent_cost = smac.validate(incumbent)
    auroc_mean.append(1 - incumbent_cost)
    incumbent_cost = smac.validate(incumbent)
    auroc_mean.append(1 - incumbent_cost)
    incumbent_cost = smac.validate(incumbent)
    auroc_mean.append(1 - incumbent_cost)
    incumbent_cost = smac.validate(incumbent)
    auroc_mean.append(1 - incumbent_cost)
    incumbent_cost = smac.validate(incumbent)
    auroc_mean.append(1 - incumbent_cost)
    print(
        "RATIO:", configspace.sample_configuration()["ratio"],
        auroc_mean,
        numpy.average(auroc_mean), numpy.std(auroc_mean)
    )