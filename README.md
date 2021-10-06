# Distribution matching

This repository contains the code for the experiments presented in 

> Why autoencoders don't work: The downfalls of distribution matching.

## Dependencies
See `setup.py` for necessary pip packages.

Tested with Python 3.8.5 and PyTorch 1.1.1

## Usage

The path variables in `dmatch/utils/io.py` need to be set before running experiments.

#### Flow experiments
Use `experiments/inn_plane.py`

#### MLP experiments with optimal transport losses
Use `experiments/no_recon/tests_no_recon.py`

#### Adversarial trainings
Use `experiments/no_recon/aae.py` 
