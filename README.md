
<p align="center">
  <img alt="logo" src="https://github.com/BIMSBbioinfo/flexynesis/raw/main/img/logo.png" width="40%">
</p>

[![Downloads](https://static.pepy.tech/badge/flexynesis)](https://pepy.tech/project/flexynesis)
[![Docker Pulls](https://img.shields.io/docker/pulls/borauyar/flexynesis?logo=docker)](https://hub.docker.com/r/borauyar/flexynesis)
![Tutorials 3.11](https://img.shields.io/github/actions/workflow/status/BIMSBbioinfo/flexynesis/tutorials.yml?branch=main&job=Tutorials%20Python%203.11&label=Tutorials:%20Python%203.11)
![Tutorials 3.12](https://img.shields.io/github/actions/workflow/status/BIMSBbioinfo/flexynesis/tutorials.yml?branch=main&job=Tutorials%20Python%203.12&label=Tutorials:%20Python%203.12)
![Tutorials 3.x](https://img.shields.io/github/actions/workflow/status/BIMSBbioinfo/flexynesis/tutorials.yml?branch=main&job=Tutorials%20Python%203.x&label=Tutorials:%20Python%203.x)
![Python 3.11](https://img.shields.io/github/actions/workflow/status/BIMSBbioinfo/flexynesis/models.yml?branch=main&job=Python%203.11&label=Models:%20Python%203.11)
![Python 3.12](https://img.shields.io/github/actions/workflow/status/BIMSBbioinfo/flexynesis/models.yml?branch=main&job=Python%203.12&label=Models:%20Python%203.12)
![Python 3.x](https://img.shields.io/github/actions/workflow/status/BIMSBbioinfo/flexynesis/models.yml?branch=main&job=Python%203.x&label=Models:%20Python%203.x%20(latest))

# Flexynesis: deep learning toolkit for interpretable multi-omics integration and clinical outcome prediction

Flexynesis is a deep learning suite for multi-omics data integration, designed for (pre-)clinical endpoint prediction. It supports diverse neural architectures — from fully connected networks and supervised variational autoencoders to graph convolutional and multi-triplet models — with flexible options for omics layer fusion, automated feature selection, and hyperparameter optimization.

Built with interpretability in mind, Flexynesis incorporates integrated gradients (via Captum) for marker discovery, helping researchers move beyond black-box models.

The framework is continuously benchmarked on public datasets, particularly in oncology, and has been applied to tasks such as drug response prediction in patients and preclinical models (cell lines, PDXs), cancer subtype classification, and clinically relevant outcomes in regression, classification, survival, and cross-modality settings.

<p align="center">
  <img alt="workflow" src="https://github.com/BIMSBbioinfo/flexynesis/raw/main/img/graphical_abstract.jpg">
</p>

# Installation

Flexynesis requires **Python 3.11+**.  
You can install the latest release from PyPI:

```bash
pip install flexynesis
```

# Citing our work

In order to refer to our work, please cite our manuscript currently available at [BioRxiv](https://biorxiv.org/cgi/content/short/2024.07.16.603606v1). 

# Getting started with Flexynesis

## Command-line tutorial

- [Getting Started with Flexynesis](https://bimsbstatic.mdc-berlin.de/akalin/buyar/flexynesis/site/getting_started/)

## Jupyter notebooks for interactive usage

- [Modeling Breast Cancer Subtypes](https://github.com/BIMSBbioinfo/flexynesis/blob/main/examples/tutorials/brca_subtypes.ipynb)
- [Survival Markers of Lower Grade Gliomas](https://github.com/BIMSBbioinfo/flexynesis/blob/main/examples/tutorials/survival_subtypes_LGG_GBM.ipynb)
- [Unsupervised Analysis of Bone Marrow Cells](https://github.com/BIMSBbioinfo/flexynesis/blob/main/examples/tutorials/unsupervised_analysis_single_cell.ipynb)

## Running Flexynesis on [Galaxy](https://usegalaxy.eu/)

- See [Galaxy Training Network Tutorials](https://github.com/BIMSBbioinfo/flexynesis/discussions/107)

## Docker

- See how to run a [Docker image of Flexynesis](https://github.com/BIMSBbioinfo/flexynesis/discussions/110#discussion-8836611). 

# Benchmarks

For the latest benchmark results see: 
https://bimsbstatic.mdc-berlin.de/akalin/buyar/flexynesis-benchmark-datasets/dashboard.html

The code for the benchmarking pipeline is at: https://github.com/BIMSBbioinfo/flexynesis-benchmarks

# Documentation

[Flexynesis Documentation](https://bimsbstatic.mdc-berlin.de/akalin/buyar/flexynesis/site/getting_started/) was generated using [mkdocs](https://mkdocstrings.github.io/) 

```
pip install mkdocstrings[python]
mkdocs build --clean
```

# Contact

For questions, suggestions, or collaborations: Open an [issue](https://github.com/BIMSBbioinfo/flexynesis/issues) or create a [discussion](https://github.com/BIMSBbioinfo/flexynesis/discussions).  

# License 

Flexynesis is released under a modifed MIT Licence for Academic and Non-Commercial Usage. 

© 2025 Bioinformatics and Omic Data Science Platform, Max Delbrück Center for Molecular Medicine (MDC).


