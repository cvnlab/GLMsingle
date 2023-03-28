# GLMsingle

![image](https://user-images.githubusercontent.com/35503086/151108958-24479034-c7f7-4734-b903-9046ba6a78ac.png)

-------------------------------------------------------------------------------------------

GLMsingle is a toolbox for obtaining accurate single-trial estimates
in fMRI time-series data. We provide both MATLAB and Python implementations. 

GLMsingle is detailed in the following paper:

**[Prince, J.S., Charest, I., Kurzawski, J.W., Pyles, J.A., Tarr, M., Kay, K.N. Improving the accuracy of single-trial fMRI response estimates using GLMsingle. *eLife* (2022).](https://doi.org/10.7554/eLife.77599)**

For additional documentation and FAQ on GLMsingle,
please see: **https://glmsingle.readthedocs.io**

For a lecture overview, implementation guide, and demo of GLMsingle,
please see: **https://cbmm.mit.edu/video/glmsingle-toolbox-improving-single-trial-fmri-response-estimates**

For a video walkthrough of the figure outputs from GLMsingle,
please see: **https://www.youtube.com/watch?v=aZFh-YUZUYE**

GLMsingle can be viewed as a wholesale replacement of its predecessor,
GLMdenoise (http://github.com/kendrickkay/GLMdenoise).

If you have questions or discussion points, please use the Discussions
feature of this github repository. If you find a bug, 
please let us know by raising a github Issue.

## MATLAB

To install: 

```bash
git clone --recurse-submodules https://github.com/cvnlab/GLMsingle.git
```

This will also clone [`fracridge`](https://github.com/nrdg/fracridge) as a submodule.

To use the GLMsingle toolbox, add it and `fracridge` to your MATLAB path by running the `setup.m` script.

## Python

To install: 

```bash
pip install git+https://github.com/cvnlab/GLMsingle.git
```

Running the example scripts requires:

- installing jupyter notebook or jupyter lab
- cloning the GLMsingle repository in order to get the example scripts located in `examples`:

```bash
pip install jupyterlab
git clone --recurse-submodules https://github.com/cvnlab/GLMsingle.git
```

Code dependencies: see [requirements.txt](./requirements.txt)

Notes:
* Currently, numpy has a 4GB limit for the pickle files it writes; thus, GLMsingle will crash if the file outputs exceed that size. One workaround is to turn off "disk saving" and instead get the outputs of GLMsingle in your workspace and save the outputs yourself to HDF5 format.

## Example scripts

We provide a number of example scripts that demonstrate usage of GLMsingle. You can browse these example scripts here:

(Python Example 1 - event-related design) https://htmlpreview.github.io/?https://github.com/kendrickkay/GLMsingle/blob/main/examples/example1.html

(Python Example 2 - block design) https://htmlpreview.github.io/?https://github.com/kendrickkay/GLMsingle/blob/main/examples/example2.html

(MATLAB Example 1 - event-related design) https://htmlpreview.github.io/?https://github.com/kendrickkay/GLMsingle/blob/main/matlab/examples/example1preview/example1.html

(MATLAB Example 2 - block design) https://htmlpreview.github.io/?https://github.com/kendrickkay/GLMsingle/blob/main/matlab/examples/example2preview/example2.html

If you would like to run these example scripts, the Python versions are available in `/GLMsingle/examples`, and the MATLAB versions are available in `/GLMsingle/matlab/examples`.

The first two notebooks contain a full walkthrough of the process of loading an example dataset and design matrix, estimating neural responses using GLMsingle, estimating the reliability of responses at each voxel, and comparing those achieved via GLMsingle to those achieved using a baseline GLM.

The remaining notebooks illustrate a number of other analyses that may be useful for the user to browse.

## Additional information

Terms of use: This content is licensed under a BSD 3-Clause License.

If you use GLMsingle in your research, please cite the following paper:

* [Prince, J.S., Charest, I., Kurzawski, J.W., Pyles, J.A., Tarr, M., Kay, K.N. Improving the accuracy of single-trial fMRI response estimates using GLMsingle. *eLife* (2022).](https://doi.org/10.7554/eLife.77599)

## Change history

* 2023/03/27 - More diagnostic figures added.
* 2023/03/26 - A number of new useful diagnostics are computed: these include diagnostics of the design matrix (information is shown in the command window), new figure visualizations, and a new run-wise FIR model that is used to generate diagnostic figures. These changes are currently implemented in MATLAB and will be ported to Python soon. The changes are described in the function documentation and covered in the new video walkthrough of figure outputs (see above).
* 2022/11/28 - Version 1.1 of GLMsingle is now released. A git tag has been added to the repo. This version corresponds to what is described in the Prince et al. 2022 paper.
* 2021/10/12 - Version 1.0 of GLMsingle is now released. A git tag has been added to the repo.
* 2021/05/21 - The core code is complete, but is in "beta" and we are generating tutorial examples of usage. The initial 1.0 release should be forthcoming.
