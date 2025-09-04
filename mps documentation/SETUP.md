# Setting Up Flexynesis with Mamba

This guide explains how to set up the Flexynesis package using Mamba for environment management.

## Prerequisites

- Mamba installed (comes with Miniconda or Mambaforge)
- Git (to clone the repository if not already done)

## Step 1: Clone the Repository

If you haven't already, clone the Flexynesis repository:

```bash
git clone https://github.com/huseyincavusbi/flexynesis-mps.git
cd flexynesis-mps
```

## Step 2: Create the Mamba Environment

Use the provided `environment.yml` file to create a new Mamba environment:

```bash
mamba env create -f environment.yml
```

This will create an environment named `flexynesis-test` with all the basic dependencies.

## Step 3: Activate the Environment

Activate the newly created environment:

```bash
mamba activate flexynesis-test
```

## Step 4: Install Flexynesis in Editable Mode

Install the Flexynesis package in editable mode to include all dependencies and allow for development changes:

```bash
pip install -e .
```

This installs the package and pulls in additional dependencies from `pyproject.toml`.

## Step 5: Verify the Installation

Test that Flexynesis is working correctly:

```bash
flexynesis --help
```

You should see the help message with all available options.

## Step 6: Run a Quick Test (Optional)

To ensure everything is working, you can run a quick test with sample data:

```bash
# Download test dataset
curl -L -o dataset1.tgz https://bimsbstatic.mdc-berlin.de/akalin/buyar/flexynesis-benchmark-datasets/dataset1.tgz
tar -xzvf dataset1.tgz

# Run test
flexynesis --data_path dataset1 --model_class DirectPred --target_variables Erlotinib --hpo_iter 1 --features_top_percentile 5 --data_types gex,cnv
```

This should complete within a minute on a typical CPU.

## Notes

- Editable mode (`-e`) allows changes to the source code to be reflected immediately without reinstalling.
- The environment includes PyTorch, Lightning, and other ML libraries.
- For development, use this setup to modify and test changes to Flexynesis.

## Troubleshooting

If you encounter issues with Mamba activation, ensure Mamba is properly initialized:

```bash
mamba init zsh  # or your shell
source ~/.zshrc
```

Then retry the activation step.
