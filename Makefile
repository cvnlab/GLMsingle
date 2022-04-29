.DEFAULT_GOAL := help

BROWSER := python -c "$$BROWSER_PYSCRIPT"

# TODO make more general to use the local matlab version
MATLAB = /usr/local/MATLAB/R2017a/bin/matlab
MATLAB_ARG    = -nodisplay -nosplash -nodesktop

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

# determines what "make help" will show
define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

################################################################################
# 	GENERIC
.PHONY: help clean clean-test lint install_dev

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)
clean: clean-build clean-pyc clean-test ## remove all build, test, coverage artifacts

clean-test: ## remove test and coverage artifacts
	rm -rf .tox/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache
	rm -rf tests/data
	rm -rf test/outputs

install_dev: ## install for both matlab and python developpers
	pip install -e .
	pip install -r requirements_dev.txt

lint: lint/black lint/flake8 lint/miss_hit ## check style

test: test-matlab test-python

tests/data/nsdcoreexampledataset.mat: 
	mkdir tests/data
	curl -fsSL --retry 5 -o "tests/data/nsdcoreexampledataset.mat" https://osf.io/k89b2/download

################################################################################

################################################################################
# 	MATLAB

.PHONY: lint/miss_hit

lint/miss_hit: ## lint and checks matlab code
	mh_style --fix tests && mh_metric --ci tests && mh_lint tests

test-matlab: tests/data/nsdcoreexampledataset.mat
	$(MATLAB) $(MATLAB_ARG) -r "run_tests; exit()"

coverage-matlab: test-matlab
	$(BROWSER) coverage_html/index.html

################################################################################

################################################################################
# 	PYTHON

.PHONY: clean-build clean-pyc coverage-python install lint/flake8 lint/black

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

lint/flake8: ## check style with flake8
	flake8 tests
lint/black: ## check style with black
	black tests

test-python: tests/data/nsdcoreexampledataset.mat ## run tests quickly with the default Python
	pytest 

test-notebooks:
	pytest  --nbmake --nbmake-timeout=3000 "./examples"
test-all: ## run tests on every Python version with tox
	tox

coverage-python: ## check code coverage quickly with the default Python
	coverage run --source glmsingle -m pytest
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

install: clean ## install the package to the active Python's site-packages
	python setup.py install

################################################################################