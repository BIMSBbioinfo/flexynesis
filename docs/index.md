
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
  --evaluate_baseline_performance
```

# Tutorial for getting started 

See our [tutorial](https://bimsbstatic.mdc-berlin.de/akalin/buyar/flexynesis/site/getting_started/) for how to use Flexynesis in different scenarios. 

