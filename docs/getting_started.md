# Getting Started with Flexynesis 

## Quick Start

### Install 

```
# create an environment with python 3.11 
conda create --name flexynesisenv python==3.11
conda activate flexynesisenv
# install latest version from pypi (https://pypi.org/project/flexynesis)
# make sure to use python3.11*
python -m pip install flexynesis --upgrade  
```

### Test the installation

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

## Input Dataset Description

Flexynesis expects as input a path to a data folder with the following structure:

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

### File contents

#### clin.csv
`clin.csv` contains the sample metadata. The first column contains unique sample identifiers. 
The other columns contain sample-associated clinical variables. 
`NA` values are allowed in the clinical variables. 

```csv
v1,v2
s1,a,b
s2,c,d
s3,e,f
```

#### omics.csv 
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

#### Concordance between train/test splits
The corresponding omics files in train/test splits must contain overlapping feature names (they don't 
have to be identical or in the same order). 
The `clin.csv` files in train/test must contain matching clinical variables. 


## Download a curated dataset

Before using Flexynesis on your own dataset, it is highly recommended that you familiarize yourself with datasets we have already curated and used for training and testing Flexynesis models. 

Below you can find examples of how we can utilize Flexynesis from the command-line in multi-omic data integration for clinical variable prediction. 

In order to demonstrate the various command-line options and different ways to run Flexynesis, we will use a multi-omic dataset of Lower Grade Glioma (LGG) and Glioblastoma Multiforme (GBM) Merged Cohorts. The data were downloaded from [Cbioportal](https://www.cbioportal.org/study/summary?id=lgggbm_tcga_pub). The dataset was split into 70/30 train/test splits and used as input to Flexynesis. 

```
wget -O lgggbm_tcga_pub_processed.tgz https://bimsbstatic.mdc-berlin.de/akalin/buyar/flexynesis-benchmark-datasets/lgggbm_tcga_pub_processed.tgz 
tar -xzvf lgggbm_tcga_pub_processed.tgz
```

The example dataset contains 556 training samples and 238 testing samples. Each sample has both copy number variation and mutation data. The mutation data was converted into a binary matrix of genes-vs-samples where the value of a gene for a given sample is set to 1 if the gene is mutated in that sample, or it is set to 0 if no mutation was found for that gene. 

## Supervised training

### Minimal setup 

For supervised training, the minimum required options to run Flexynesis are

1. Path to a dataset folder 
2. Selection of a tool/model 
3. One target variable which can be numerical or categorical for regression/classification tasks. 
4. List of data types to use for modeling. Here we use the prefix of the filename that is available in the train/test folders (e.g. mut.csv => mut). While flexynesis is built for multi-omic integration, a single data modality is also acceptable. 

While it is not a required argument, we set the hyperparameter optimisation steps to 1 to avoid lengthy run times for demonstration purposes. 

```
flexynesis  --data_path lgggbm_tcga_pub_processed \
            --model_class DirectPred \
            --target_variables KARNOFSKY_PERFORMANCE_SCORE \
            --data_types mut \
            --hpo_iter 1
```

### Multi-modal training 
In the case where we want to use multiple data modalities, we provide a comma separated list of data type names as input:

For example, if we wanted to utilize both mutation and CNA data matrices for training: 

```
flexynesis  --data_types mut,cna  <... other arguments> 
```

### Different options for the outcome variables 

Flexynesis supports both single-task and multi-task training. We can provide one or more target variables and optionally survival variables as input and Flexynesis will build the appropriate model architecture. If the selected variable is numerical, a Multi-Layered-Perceptron (MLP) with MSE loss will be used. If a categorical variable is provided, an MLP with cross-entropy-loss will be utilized. If survival variables are provided, an MLP with Cox-Proportional-Hazards loss will be attached to the model.  

All the user has to do is to provide a list of variable names:

#### Example: Regression

The target variable `KARNOFSKY_PERFORMANCE_SCORE` is a numerical value, so it will be built as a regression problem. 

```
flexynesis  --data_path lgggbm_tcga_pub_processed \
            --model_class DirectPred \
            --target_variables KARNOFSKY_PERFORMANCE_SCORE \
            --data_types mut,cna \
            --hpo_iter 1
```

#### Example: Classification

The target varible `HISTOLOGICAL_DIAGNOSIS` is a categorical variable, so it will be built as a classification problem. 
```
flexynesis  --data_path lgggbm_tcga_pub_processed \
            --model_class DirectPred \
            --target_variables HISTOLOGICAL_DIAGNOSIS  \
            --data_types mut,cna \
            --hpo_iter 1
```

#### Example: Survival 

For survival analysis, two separate variables are required, where the first variable is a numeric  `event` variable (consisting of 0's or 1's, where 1 means an event such as disease progression or death has occurred). The second variable is also a numeric `time` variable, which indicates how much time it took since last patient follow-up. 

```
flexynesis  --data_path lgggbm_tcga_pub_processed \
            --model_class DirectPred \
            --surv_event_var OS_STATUS \
            --surv_time_var OS_MONTHS \
            --data_types mut,cna \
            --hpo_iter 1
```


#### Example: Mixed/multi-task model 

Flexynesis can be trained with multiple target variables, which can be a mixture of regression/classification/survival tasks. 
```
flexynesis  --data_path lgggbm_tcga_pub_processed \
            --model_class DirectPred \
            --target_variables HISTOLOGICAL_DIAGNOSIS,KARNOFSKY_PERFORMANCE_SCORE \
            --surv_event_var OS_STATUS \
            --surv_time_var OS_MONTHS \
            --data_types mut,cna \
            --hpo_iter 1
```


### Using different model architectures

For the supervised tasks, the user can easily switch between different model architectures.  

```
flexynesis  --data_path lgggbm_tcga_pub_processed \
            --model_class [DirectPred|supervised_vae|MultiTripletNetwork|GNN|CrossModalPred] \
            --target_variables HISTOLOGICAL_DIAGNOSIS,KARNOFSKY_PERFORMANCE_SCORE \
            --surv_event_var OS_STATUS \
            --surv_time_var OS_MONTHS \
            --data_types mut,cna \
            --hpo_iter 1
```

#### Model-specific exceptions

However there are model-specific exceptions due to the nature of the model architectures. 

1. `MultiTripletNetwork` requires the first target variable to be a `categorical` variable. Triplet loss works 
by definition on categorical variables. 
2. `GNN`: in the case of multi-omics input, the features should have the same naming convention. Another restriction for GNNs is that 
it only works if the omics features are "genes". For instance, if the features are CpG methylation sites, it wouldn't work. The reason is that GNNs require a prior knowledge network, which is currently set to use STRING database. Other model architectures can work on any kind of features, where feature nomenclature is not important. The current implementation of GNNs is also using `early` fusion type by default. 

GNNs have an additional option called `--gnn_conv_type`, which determines the type of graph convolution algorithm. By default it is set to `GC`, but it can be change to `SAGE` or `GCN`. 


### Modality fusion options

Flexynesis currently supports two main ways of fusing different omics data modalities:
1. Early fusion: The input data matrices are initially concatenated and pushed through the networks
2. Intermediate fusion: The input data matrices are initially pushed through the networks to obtain a modality-specific embedding space, which then gets concatenated to serve as input for the supervisor MLPs. 

Fusion option can be set using the `--fusion` flag

```
flexynesis  --data_path lgggbm_tcga_pub_processed \
            --model_class DirectPred \
            --target_variables HISTOLOGICAL_DIAGNOSIS \
            --data_types mut,cna \
            --fusion intermediate \
            --hpo_iter 1 
```


## Unsupervised Training 

In the absence of any target variables or survival variables, we can use a VAE architecture to carry out unsupervised training. 

Set model class to `supervised_vae` and leave variable arguments out. 

```
flexynesis  --data_path lgggbm_tcga_pub_processed \
            --model_class supervised_vae \
            --data_types mut,cna \
            --hpo_iter 1
```

## Cross-modality Training

We have implemented a special case of VAEs where the input data layers and output data layers can be set to different data modalities. 
The purpose of a cross-modality encoder is to learn embeddings that can translate from one data modality to another. Crossmodality encoder we implemented supports both single/multiple input layers and also one or more target/survival variables can be added to the model. 

The user needs to provide which data layers to be used as input and which ones to be used as output (reconstruction target). 

```
flexynesis  --data_path lgggbm_tcga_pub_processed \
            --model_class CrossModalPred \
            --data_types mut,cna \
            --input_layers mut \
            --output_layers cna \
            --hpo_iter 1
```

Both input and output layers can be set to one or more data modalities, where the modalities are determined by the `--data_types` flag. 
If the `--data_types` is set to "mut,cna"; the `--input_layers` can be set to `mut`, `mut,cna`, or `cna`, while the `--output_layers` can be set to `mut`, `mut,cna`, and `cna`. However, if the `--input_layers` and `--output_layers` are set to the same values, then it will behave as `supervised_vae` because the goal of the reconstruction would be identical to the input layers. 

Multi-modal input and multiple target variables: 

```
flexynesis  --data_path lgggbm_tcga_pub_processed \
            --model_class CrossModalPred \
            --data_types mut,cna \
            --input_layers mut,cna \
            --output_layers cna \
            --target_variables HISTOLOGICAL_DIAGNOSIS,AGE \
            --hpo_iter 1
```

## Fine-tuning options 

To enable fine-tuning, where Flexynesis builds a model on the training dataset, fine-tunes it on a portion of the test dataset, and evaluates the model on the remaining test samples, set the `--finetuning_samples` to a positive integer. 

For instance, to fine-tune the model on a randomly drawn subset of 50 samples:

```
flexynesis  --data_path lgggbm_tcga_pub_processed \
            --model_class DirectPred \
            --data_types mut,cna \
            --target_variables HISTOLOGICAL_DIAGNOSIS \
            --finetuning_samples 50 \
            --hpo_iter 1
```

## Feature filtering options 

Flexynesis will by default do feature selection using multiple flags. 

1. `--variance_threshold 1` : will remove the lowest 1% of the features based on their variances in the training data.
2. `--features_top_percentile 20`: This will trigger the "Laplacian Scoring" module to rank features by this score and the top 20% of the features will be kept. 
3. `--correlation_threshold 0.8`: Among the top ranking features, highly redundant features based on a pearson correlation score cut-off are dropped, based on the laplacian score rankings. 
4. `--restrict_to_features <filepath>`: If the user provides a path to a list of feature names, the analysis will be restricted to only these features. 

## Hyperparameter optimisation

Flexynesis will run by default for 100 hyperparameter optimisation steps. It will stop the procedure if no improvement has been observed in the last 10 iterations. We can change these with the following flags: `--hpo_iter` and `--hpo_patience`. 

```
flexynesis  --data_path lgggbm_tcga_pub_processed \
            --model_class DirectPred \
            --data_types mut,cna \
            --target_variables HISTOLOGICAL_DIAGNOSIS \
            --hpo_iter 50 \
            --hpo_patience 20
```

## Accelerating with GPUs

If you have access to GPUs on your system, they can be used to accelerate the training of models using the `--use_gpu` flag. 

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

## Using Guix

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
