# CONTRIBUTING

Information for anyone who would like to contribute to this repository.

## Repository map

```bash
├── .git
├── .github
│   └── workflows         # Github continuous integration set up
├── examples
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
├── matlab                # Matlab implementation
│   ├── examples
│   ├── fracridge
│   └── utilities
└── tests                 # Python and Matlab tests
    └── data

```

## Generic

### Makefile

A `Makefile` is used to help set / and automate some things.

In a terminal type `make help` to see what are the different "recipes" you can
run with this `Makefile`.

### pre-commit

## Matlab

### Style guide

### Tests

Running the tests require to have the following toolboes in your MATLAB path:

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

To run **all** the tests and get code coverage, you can

1. type the following in a terminal

```
make test-matlab
```

2. run the `run_tests.m` in MATLAB

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

#### Demos

### Continuous integration

## Python

### Style guide

### Tests

#### Demos

### Continuous integration
