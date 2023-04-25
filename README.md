
<p align="center">
  <img alt="logo" src="img/logo.png" width="50%" height="50%">
</p>

[![Tests](https://github.com/BIMSBbioinfo/flexynesis/actions/workflows/tests.yml/badge.svg)](https://github.com/BIMSBbioinfo/flexynesis/actions/workflows/tests.yml)
![benchmarks](https://github.com/BIMSBbioinfo/flexynesis/actions/workflows/benchmarks.yml/badge.svg)

# flexynesis
A deep-learning based multi-modal data integration suite that aims to achieve synesis in a flexible manner

# Benchmarks
For the latest benchmark results see: 
https://bimsbstatic.mdc-berlin.de/akalin/buyar/flexynesis-benchmark-datasets/dashboard.html

The code for the benchmarking pipeline is at: https://github.com/BIMSBbioinfo/flexynesis-benchmarks

# Environment

To create a clone of the development environment, use the `spec-file.txt`:
```
conda create --name flexynesis --file spec-file.txt
conda activate flexynesis
```

To export existing spec-file.txt:
```
conda list --explicit > spec-file.txt
```

# Installation

To install the project using setuptools, you can follow these steps:

1. Clone the project from the Git repository:
```
git clone git@github.com:BIMSBbioinfo/flexynesis.git
```
2. Navigate to the project directory:
```
cd flexynesis
```
3. Create a clone of the development environment, use the `spec-file.txt`:
```
conda create --name flexynesis --file spec-file.txt
conda activate flexynesis
```
4. Install the project using setuptools:
```
python setup.py install
```

# Testing

Test the vignettes 
```
srun python vignettes/DirectPred.py
srun python vignettes/svae.py
```

Run unit tests
```python
pytest -vvv tests/unit
```

This will run all the unit tests in the tests directory.

# Contributing
If you would like to contribute to the project, please open an issue or a pull request on the GitHub repository.

# Documentation

```
pdoc --html --output-dir docs --force flexynesis 
```



