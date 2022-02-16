# GLMsingle

![image](https://user-images.githubusercontent.com/35503086/151108958-24479034-c7f7-4734-b903-9046ba6a78ac.png)

-------------------------------------------------------------------------------------------

GLMsingle is a toolbox for obtaining accurate single-trial estimates
in fMRI time-series data. We provide both MATLAB and Python implementations. 

**The GLMsingle preprint, which describes the technique in detail, 
is available on bioRxiv (https://www.biorxiv.org/content/10.1101/2022.01.31.478431v1).**

GLMsingle can be viewed as a wholesale replacement of its predecessor,
GLMdenoise (http://github.com/kendrickkay/GLMdenoise).

For additional information, please see the Wiki page of
the GLMsingle repository (https://github.com/kendrickkay/GLMsingle/wiki).

If you have questions or discussion points, please use the Discussions
feature of this github repository, or alternatively, e-mail
Kendrick (kay@umn.edu). If you find a bug, please let us know by
raising a Github issue.

## MATLAB

To use the GLMsingle toolbox, add it to your MATLAB path:
  addpath(genpath('GLMsingle/matlab'));

You will also need to download and add fracridge to your path.
It is available here: https://github.com/nrdg/fracridge

## Example scripts

We provide a number of example scripts that demonstrate usage of GLMsingle. You can browse these example scripts here:

(Python Example 1 - event-related design) https://htmlpreview.github.io/?https://github.com/kendrickkay/GLMsingle/blob/main/examples/example1.html

(Python Example 2 - block design) https://htmlpreview.github.io/?https://github.com/kendrickkay/GLMsingle/blob/main/examples/example2.html

(MATLAB Example 1 - event-related design) https://htmlpreview.github.io/?https://github.com/kendrickkay/GLMsingle/blob/main/matlab/examples/example1preview/example1.html

(MATLAB Example 2 - block design) https://htmlpreview.github.io/?https://github.com/kendrickkay/GLMsingle/blob/main/matlab/examples/example2preview/example2.html

If you would like to run these example scripts, the Python versions are available in `/GLMsingle/examples`, and the MATLAB versions are available in `/GLMsingle/matlab/examples`. Each notebook contains a full walkthrough of the process of loading an example dataset and design matrix, estimating neural responses using GLMsingle, estimating the reliability of responses at each voxel, and comparing those achieved via GLMsingle to those achieved using a baseline GLM.

## Python

To install: 

```bash
pip install git+https://github.com/cvnlab/GLMsingle.git
```

Running the demos requires:

- jupyter notebook or jupyter lab.

```bash
pip install jupyterlab
```

Code dependencies: see requirements.txt

Notes:
* Please note that GLMsingle is not (yet) compatible with Python 3.9 (due to an incompatibility between scikit-learn and Python 3.9). Please use Python 3.8 or earlier.
* Currently, numpy has a 4GB limit for the pickle files it writes; thus, GLMsingle will crash if the file outputs exceed that size. One workaround is to turn off "disk saving" and instead get the outputs of GLMsingle in your workspace and save the outputs yourself to HDF5 format.

## Additional information

For additional information, please visit the Wiki page associated with this
repository: https://github.com/kendrickkay/GLMsingle/wiki

Terms of use: This content is licensed under a BSD 3-Clause License.

If you use GLMsingle in your research, please cite the following paper:

* [Prince, J.S., Charest, I., Kurzawski, J.W., Pyles, J.A., Tarr, M.J., Kay, K.N. GLMsingle: a toolbox for improving single-trial fMRI response estimates. bioRxiv (2022).](https://www.biorxiv.org/content/10.1101/2022.01.31.478431v1)

## Change history

* 2021/10/12 - Version 1.0 of GLMsingle is now released. A git tag has been added to the repo.
* 2021/05/21 - The core code is complete, but is in "beta" and we are generating tutorial examples of usage. The initial 1.0 release should be forthcoming.
