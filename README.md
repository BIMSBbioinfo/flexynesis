
<p align="center">
  <img alt="logo" src="https://github.com/BIMSBbioinfo/flexynesis/raw/main/img/logo.png" width="40%">
</p>

[![Downloads](https://static.pepy.tech/badge/flexynesis)](https://pepy.tech/project/flexynesis)
![benchmarks](https://github.com/BIMSBbioinfo/flexynesis/actions/workflows/benchmarks.yml/badge.svg)
![tutorials](https://github.com/BIMSBbioinfo/flexynesis/actions/workflows/tutorials.yml/badge.svg)

# flexynesis

Flexynesis: a flexible deep learning toolkit for interpretable multi-omics integration and clinical outcome prediction.

Flexynesis is a deep learning suite for multi-omics data integration, designed for (pre-)clinical endpoint prediction. It supports diverse neural architectures — from fully connected networks and supervised variational autoencoders to graph convolutional and multi-triplet models — with flexible options for omics layer fusion, automated feature selection, and hyperparameter optimization.

Built with interpretability in mind, Flexynesis incorporates integrated gradients (via Captum) for marker discovery, helping researchers move beyond black-box models.

The framework is continuously benchmarked on public datasets, particularly in oncology, and has been applied to tasks such as drug response prediction in patients and preclinical models (cell lines, PDXs), cancer subtype classification, and clinically relevant outcomes in regression, classification, survival, and cross-modality settings.

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



