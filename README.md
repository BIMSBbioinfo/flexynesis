# flexynesis
A deep-learning based multi-modal data integration suite that aims to achieve synesis in a flexible manner

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


This will run all the tests in the tests directory.

# Contributing
If you would like to contribute to the project, please open an issue or a pull request on the GitHub repository.

# Documentation

```
pdoc --html --output-dir docs flexynesis 
```



