
<p align="center">
  <img alt="logo" src="https://github.com/BIMSBbioinfo/flexynesis/raw/main/img/logo.png" width="50%" height="50%">
</p>

[![Downloads](https://static.pepy.tech/badge/flexynesis)](https://pepy.tech/project/flexynesis)
![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC_BY--NC--ND_4.0-lightgrey.svg)
![benchmarks](https://github.com/BIMSBbioinfo/flexynesis/actions/workflows/benchmarks.yml/badge.svg)
![tutorials](https://github.com/BIMSBbioinfo/flexynesis/actions/workflows/tutorials.yml/badge.svg)

# flexynesis
A deep-learning based multi-omics bulk sequencing data integration suite with a focus on (pre-)clinical 
endpoint prediction. The package includes multiple types of deep learning architectures such as simple 
fully connected networks, supervised variational autoencoders; different options of data layer fusion, 
and automates feature selection and hyperparameter optimisation. The tools are continuosly benchmarked 
on publicly available datasets mostly related to the study of cancer. Some of the applications of the methods 
we develop are drug response modeling in cancer patients or preclinical models (such as cell lines and 
patient-derived xenografts), cancer subtype prediction, or any other clinically relevant outcome prediction
that can be formulated as a regression or classification problem. 

<p align="center">
  <img alt="workflow" src="https://github.com/BIMSBbioinfo/flexynesis/raw/main/img/graphical_abstract.jpg">
</p>

# Documentation

A detailed documentation of classes and functions in this repository can be found [here](https://bimsbstatic.mdc-berlin.de/akalin/buyar/flexynesis/site/index.html).

# Benchmarks

For the latest benchmark results see: 
https://bimsbstatic.mdc-berlin.de/akalin/buyar/flexynesis-benchmark-datasets/dashboard.html

The code for the benchmarking pipeline is at: https://github.com/BIMSBbioinfo/flexynesis-benchmarks

# Quick Start

## Install 

```
# create an environment with python 3.11 
conda create --name flexynesisenv python==3.11
conda activate flexynesisenv
# install latest version from pypi (https://pypi.org/project/flexynesis)
# make sure to use python3.11*
python -m pip install flexynesis --upgrade  
```

## Options

For a full set of command-line options:
```
flexynesis -h 
```

## Test the installation

Download a dataset and test the flexynesis installation on a test run. 
```
curl -L -o dataset1.tgz \
https://bimsbstatic.mdc-berlin.de/akalin/buyar/flexynesis-benchmark-datasets/dataset1.tgz

tar -xzvf dataset1.tgz

conda activate flexynesisenv

flexynesis --data_path dataset1 \
  --model_class DirectPred \
  --target_variables Erlotinib \
  --fusion_type early \
  --hpo_iter 1 \
  --features_min 50 \
  --features_top_percentile 5 \
  --log_transform False \
  --data_types gex,cnv \
  --outdir . \
  --prefix erlotinib_direct \
  --early_stop_patience 3 \
  --use_loss_weighting False \
  --evaluate_baseline_performance False
```

## Accelerating with GPUs

If you have access to GPUs on your system, they can be used to accelerate the training of models. 
However, making GPUs accessible to `torch` is system-specific. Please contact your system administrator 
to make sure you have accessible GPUs and methods to access them. 

### With Slurm 

If you have [Slurm Workload Manager] in your system, you can call `flexynesis` as follows: 

```
conda activate flexynesisenv
srun --gpus=1 --pty flexynesis --use_gpu ...otherarguments
```

### GridEngine

If you have an HPC sytem running GridEngine with GPU nodes, you may be allowed to request a node with 
GPUs. The important thing here is to request a GPU node with the proper **CUDA** version installed on it. 

```
# request 1 GPU device node with CUDA version 12
qrsh -l gpu=1,cuda12
# activate your environment
conda activate flexynesisenv
flexynesis --use_gpu ...otherarguments 
```

# Input Dataset Structure

```txt
InputFolder/
| --  train 
|    |-- omics1.csv 
|    |-- omics2.csv
|    |--  ... 
|    |-- clin.csv

| --  test 
|    |-- omics1.csv 
|    |-- omics2.csv
|    |--  ... 
|    |-- clin.csv
```

## File contents

### clin.csv
`clin.csv` contains the sample metadata. The first column contains unique sample identifiers. 
The other columns contain sample-associated clinical variables. 
`NA` values are allowed in the clinical variables. 

```csv
v1,v2
s1,a,b
s2,c,d
s3,e,f
```

### omics.csv 
The first column of the feature tables must be unique feature identifiers (e.g. gene names). 
The column names must be sample identifiers that should overlap with those in the `clin.csv`. 
They don't have to be completely identical or in the same order. Samples from the `clin.csv` that are not represented
in the omics table will be dropped. 

```txt
s1,s2,s3
g1,0,1,2
g2,3,3,5
g3,2,3,4
```

### Concordance between train/test splits
The corresponding omics files in train/test splits must contain overlapping feature names (they don't 
have to be identical or in the same order). 
The `clin.csv` files in train/test must contain matching clinical variables. 

# Guix

You can also create a reproducible development environment or build a reproducible package of Flexynesis with [GNU Guix](https://guix.gnu.org).  You will need at least the Guix channels listed in `channels.scm`.  It also helps to have authorized the Inria substitute server to get binaries for CUDA-enabled packages.  See [this page](https://hpc.guix.info/channels/non-free/) for instructions on how to configure fetching binary substitutes from the build servers.

You can build a Guix package from the current committed state of your git checkout and using the specified state of Guix like this:

```sh
guix time-machine -C channels.scm -- \
    build --no-grafts -f guix.scm
```

To enter an environment containing just Flexynesis:

```sh
guix time-machine -C channels.scm -- \
    shell --no-grafts -f guix.scm
```

To enter a development environment to hack on Flexynesis:

```sh
guix time-machine -C channels.scm -- \
    shell --no-grafts -Df guix.scm
```

Do this to build a Docker image containing this package together with a matching Python installation:

```sh
guix time-machine -C channels.scm -- \
  pack -C none \
  -e '(load "guix.scm")' \
  -f docker \
  -S /bin=bin -S /lib=lib -S /share=share \
  glibc-locales coreutils bash python
```

# Defining Kernel for Jupyter Notebook

For interactively using flexynesis on Jupyter notebooks, one can define the kernel to make
flexynesis and its dependencies available on the jupyter session. 

Assuming you have already defined an environment and installed the package: 
```
conda activate flexynesisenv 
python -m ipykernel install --user --name "flexynesisenv" --display-name "flexynesisenv"
```

# Compiling Notebooks

`papermill` can be used to compile the tutorials under `examples/tutorials`. 

If the purpose is to quickly check if the notebook can be run; set HPO_ITER to 1. 
This sets hyperparameter optimisation steps to 1. 
For longer training runs to see more meaningful results from the notebook, increase this number to e.g. 50. 

Example: 

```
papermill examples/tutorials/brca_subtypes.ipynb brca_subtypes.ipynb -p HPO_ITER 1 
```

The output from papermill can be converted to an html file as follows:

```
jupyter nbconvert --to html brca_subtypes.ipynb 
```

# Documentation

Documentation generated using [mkdocs](https://mkdocstrings.github.io/) 

```
pip install mkdocstrings[python]
mkdocs build --clean
```



