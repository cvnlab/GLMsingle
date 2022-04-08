# CONTRIBUTING

Information for anyone who would like to contribute to this repository.

- [CONTRIBUTING](#contributing)
  - [Repository map](#repository-map)
  - [Generic](#generic)
    - [Makefile](#makefile)
    - [pre-commit](#pre-commit)
  - [Matlab](#matlab)
    - [Style guide](#style-guide)
    - [Tests](#tests)
      - [Adding new tests](#adding-new-tests)
    - [Continuous integration](#continuous-integration)
      - [Tests](#tests-1)
      - [Demos](#demos)
  - [Python](#python)
    - [Style guide](#style-guide-1)
    - [Tests](#tests-2)
      - [Demos](#demos-1)
    - [Continuous integration](#continuous-integration-1)

## Repository map

```bash
├── .git
├── .github
│   └── workflows         # Github continuous integration set up
├── examples              # Python demos
│   ├── data
│   ├── example1outputs
│   └── example2outputs
├── glmsingle             # Python implementation
│   ├── cod
│   ├── design
│   ├── gmm
│   ├── hrf
│   ├── ols
│   ├── ssq
│   └── utils
├── matlab                # Matlab implementation
│   ├── examples          # Matlab demos
│   ├── fracridge         # Fracridge submodule
│   └── utilities
└── tests                 # Python and Matlab tests
    ├── data              # Data used as inputs for the tests
    └── expected          # Expected results of the tests
│       └── matlab

```

## Generic

### Makefile

A `Makefile` is used to help set / and automate some things.

In a terminal type `make help` to see what some of the different "recipes" you
can run with this `Makefile`.

See
[here for a short intro on using `Makefiles`](https://the-turing-way.netlify.app/reproducible-research/make.html)

### pre-commit

You can use the [`pre-commit` python package](https://pre-commit.com/) in this
repo to make sure you only commit properly formatted files (for example `.yml`
files).

1. Install `pre-commit`

```bash
$ pip3 install pre-commit
```

It is also included in `requirements_dev.txt`, so it will installed by running:

```bash
$ pip3 install -r requirements_dev.txt
```

The `.pre-commit-config.yml` file defines the checks to run when committing
files.

1. Run the following command to install the `pre-commit` "hooks"

```bash
$ pre-commit install
```

## Matlab

### Style guide

The [`miss_hit` python package](https://misshit.org/) is used to help ensure a
consistent coding style for some of the MATLAB code.

`miss_hit` can check code style, do a certain amount of automatic code
reformating and prevent the code complexity from getting out of hand by running
static code analysis (Static analysis can is a way to measure and track software
quality metrics without additional code like tests).

`miss_hit` is quite configurable via the use of `miss_hit.cfg` files.

Install `miss_hit`:

```bash
$ pip3 install miss_hit
```

It is also included in `requirements_dev.txt`, so it will installed by running:

```bash
$ pip3 install -r requirements_dev.txt
```

Style-check your program:

```bash
$ mh_style --fix path_to_folder_or_m_file
```

Make sure your code does not get too complex:

```bash
$ mh_metric --ci
```

You can rule several of those checks by simply typing

```bash
make lint/miss_hit
```

### Tests

For an introduction to testing see
[here](https://the-turing-way.netlify.app/reproducible-research/make.html).

Running the tests require to have the following toolboxes in your MATLAB path:

- the [MOxUnit testing framework](https://github.com/MOxUnit/MOxUnit) to run the
  tests
  ([see installation procedure](https://github.com/MOxUnit/MOxUnit#installation))
- [MOcov](https://github.com/MOcov/MOcov)) to get a code coverage estimate
  ([see installation procedure](https://github.com/MOcov/MOcov#installation))

All the tests are in the `tests` folder in files starting with `test_*.m`.

To Download the data required for running the tests (this data is common for
MATLAB and python tests), type:

```bash
make tests/data/nsdcoreexampledataset.mat
```

Only some specfici results are checked by the system tests: those can be found
in `tests/expected/matlab`

To run **all** the tests and get code coverage, you can

1. type the following in a terminal

```
make test-matlab
```

1. run `moxunit_runtests` in MATLAB to run all `test_*.m` files in in the
   present working directory.

1. run the `run_tests.m` in MATLAB

You can also run all the tests contained in a specific `test_*.m` file directly,
by running that file only.

#### Adding new tests

A typical MoxUnit test file starts with with `test_` and would look something
like this.

```matlab
function test_suite=test_sum_of_squares

    try % assignment of 'localfunctions' is necessary in Matlab >= 2016
        test_functions=localfunctions();
    catch % no problem; early Matlab versions can use initTestSuite fine
    end
    initTestSuite();

end

function test_sum_of_squares_basic

    % given
    a = 2;
    b = 3;

    % when
    result = sum_of_squares([a, b])

    % then
    expected = 13;
    assertEqual(result, expected);

end

% New tests can added as new sub functions

```

### Continuous integration

We use Github to run several workflows for continuous integration.

#### Tests

The matlab tests are run by the workflow:
`.github/workflows/run_tests_matlab.yaml`. It sets up MATLAB, Moxunit and Mocov
and then then calls `.github/workflows/run_tests_ci.m` to run the tests via
`run_tests.m`.

Those tests should be run with every push on the `master` branch and on pull
request that target the `master` branch.

#### Demos

The demos in the `matlab/examples` folder are run automatically in Github CI at
regular intervals.

The matlab demos are run by the workflow:
`.github/workflows/run_demos_matlab.yaml`. The demos are run by calling
`.github/workflows/run_demos_ci.m` and also each demo is run via a MoxUnit test
(see `.github/workflows/test_demos.m`) to make sure that if the first one
crashes, then the second one will still be run (easier than setting up parallel
jobs in CI).

## Python

### Style guide

### Tests

#### Demos

### Continuous integration
