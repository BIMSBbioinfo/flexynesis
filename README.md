
<p align="center">
  <img alt="logo" src="https://github.com/BIMSBbioinfo/flexynesis/raw/main/img/logo.png" width="50%" height="50%">
</p>

[![Downloads](https://static.pepy.tech/badge/flexynesis)](https://pepy.tech/project/flexynesis)
![benchmarks](https://github.com/BIMSBbioinfo/flexynesis/actions/workflows/benchmarks.yml/badge.svg)
![tutorials](https://github.com/BIMSBbioinfo/flexynesis/actions/workflows/tutorials.yml/badge.svg)

# flexynesis

A deep-learning based multi-omics bulk sequencing data integration suite with a focus on (pre-)clinical 
endpoint prediction. The package includes multiple types of deep learning architectures such as simple 
fully connected networks, supervised variational autoencoders, graph convolutional networks, multi-triplet networks
different options of data layer fusion, and automates feature selection and hyperparameter optimisation. The tools are continuosly benchmarked on publicly available datasets mostly related to the study of cancer. Some of the applications of the methods 
we develop are drug response modeling in cancer patients or preclinical models (such as cell lines and 
patient-derived xenografts), cancer subtype prediction, or any other clinically relevant outcome prediction
that can be formulated as a regression, classification, survival, or cross-modality prediction problem. 

<p align="center">
  <img alt="workflow" src="https://github.com/BIMSBbioinfo/flexynesis/raw/main/img/graphical_abstract.jpg">
</p>

# Citing our work

In order to refer to our work, please cite our manuscript currently available at [BioRxiv](https://biorxiv.org/cgi/content/short/2024.07.16.603606v1). 

# Getting started with Flexynesis

## Command-line tutorial

- [Getting Started with Flexynesis](https://bimsbstatic.mdc-berlin.de/akalin/buyar/flexynesis/site/getting_started/)

## Jupyter notebooks for interactive usage

- [Modeling Breast Cancer Subtypes](https://github.com/BIMSBbioinfo/flexynesis/blob/main/examples/tutorials/brca_subtypes.ipynb)
- [Survival Markers of Lower Grade Gliomas](https://github.com/BIMSBbioinfo/flexynesis/blob/main/examples/tutorials/survival_subtypes_LGG_GBM.ipynb)
- [Unsupervised Analysis of Bone Marrow Cells](https://github.com/BIMSBbioinfo/flexynesis/blob/main/examples/tutorials/unsupervised_analysis_single_cell.ipynb)


# Benchmarks

For the latest benchmark results see: 
https://bimsbstatic.mdc-berlin.de/akalin/buyar/flexynesis-benchmark-datasets/dashboard.html

The code for the benchmarking pipeline is at: https://github.com/BIMSBbioinfo/flexynesis-benchmarks


# Documentation

Documentation generated using [mkdocs](https://mkdocstrings.github.io/) 

```
pip install mkdocstrings[python]
mkdocs build --clean
```



