# CONTRIBUTING

Information for anyone who would like to contribute to this repository.

## Repository map

```bash
├── .git
├── .github
│   └── workflows         # Github continuous integration set up
├── examples              # Python demos: jupyter notebooks
│   ├── data
│   ├── example1outputs
│   ├── example2outputs
├── glmsingle             # Python implementation
│   ├── cod
│   ├── design
│   ├── gmm
│   ├── hrf
│   ├── ols
│   ├── ssq
│   └── utils
├── matlab                # MATLAB implementation
│   ├── examples          # MATLAB demos
│   ├── fracridge
│   └── utilities
└── tests                 # Python and MATLAB tests
    └── data

```

## Generic

### Makefile

### pre-commit

## Matlab

### Style guide

### Tests

#### Demos

### Continuous integration

## Python

All the packages required to help with the python development of GLMsingle can
be installed with:

```bash
pip install -r requirements_dev.txt
```

### Style guide

[black](https://black.readthedocs.io/en/stable/) (code formater) and
[flake8](https://flake8.pycqa.org/en/latest/) (style guide enforcement) are used
on the test code base.

Ypu can use make to run them automatically with

```bash
make lint/black # to run black
make lint/flake8 # to run flake8
make lint # to run both
```

### Tests

The tests can be run with with pytest via the make command:

```bash
make test-python
```

#### Demos

The jupyter notebook are tested with the
[`nbmake` plugin for pytest](https://pypi.org/project/nbmake/).

They can be run with the make command:

```bash
make test-notebooks
```

### Continuous integration

We use Github to run several workflows for continuous integration.

#### Tests

The python tests are run by the workflow:
`.github/workflows/run_tests_python.yaml`.

Those tests should be run with every push on the `master` branch and on pull
request that target the `master` branch.

#### Demos

The demos in the `examples` folder are run automatically in Github CI at regular
intervals.

The jupyter notebooks are run by the workflow
`.github/workflows/run_demos_python.yaml`.
