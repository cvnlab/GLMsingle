from os.path import join
from os.path import abspath
from os.path import dirname
import scipy
import scipy.io as sio
import numpy as np
from glmsingle.glmsingle import GLM_single

test_dir = dirname(abspath(__file__))
expected_dir = join(test_dir, "expected", "python")
data_dir = join(test_dir, "data")
data_file = join(data_dir, "nsdcoreexampledataset.mat")

output_dir = join(test_dir, "outputs")


expected = {"typeb": {}, "typec": {}, "typed": {}}

# TODO use same expected data for Python and MATLAB ?
# TODO in both cases expected results should probably be in a set of CSV files
# tmp = sio.loadmat(join(expected_dir, "TYPEB_FITHRF.mat"))
# expected["typeb"]["HRFindex"] = tmp["HRFindex"]

tmp = np.load(join(expected_dir, "TYPEB_FITHRF_HRFindex.npy"), allow_pickle=True)
expected["typeb"]["HRFindex"] = tmp
tmp = np.load(join(expected_dir, "TYPEC_FITHRF_HRFindex.npy"), allow_pickle=True)
expected["typec"]["HRFindex"] = tmp
tmp = np.load(join(expected_dir, "TYPED_FITHRF_HRFindex.npy"), allow_pickle=True)
expected["typed"]["HRFindex"] = tmp
tmp = np.load(join(expected_dir, "TYPED_FITHRF_R2.npy"), allow_pickle=True)
expected["typed"]["R2"] = tmp

def test_GLM_single_system():

    X = sio.loadmat(data_file)

    data = []
    design = []
    for r in range(3):
        data.append(X["data"][0, r][50:70, 7:27, 0:1, :])
        design.append(scipy.sparse.csr_matrix.toarray(X["design"][0, r]))

    stimdur = X["stimdur"][0][0]
    tr = X["tr"][0][0]

    opt = {"wantmemoryoutputs": [1, 1, 1, 1]}

    # OPTIONAL: PUT THIS IN?
    # opt['wantlibrary'] = 0

    glmsingle_obj = GLM_single(opt)
    glm_results = glmsingle_obj.fit(
        design, data, stimdur, tr, outputdir=join(output_dir, "python")
    )

    # TODO read results directly from "glm_results" and not from file on disk?
    results = {"typeb": {}, "typec": {}, "typed": {}}

    tmp = np.load(join(output_dir, "python", "TYPEB_FITHRF.npy"), allow_pickle=True)
    results["typeb"]["HRFindex"] = tmp.item(0)["HRFindex"]
    tmp = np.load(join(output_dir, "python", "TYPEC_FITHRF_GLMDENOISE.npy"), allow_pickle=True)
    results["typec"]["HRFindex"] = tmp.item(0)["HRFindex"]
    tmp = np.load(join(output_dir, "python", "TYPED_FITHRF_GLMDENOISE_RR.npy"), allow_pickle=True)
    results["typed"]["HRFindex"] = tmp.item(0)["HRFindex"]


    assert (results["typeb"]["HRFindex"] == expected["typeb"]["HRFindex"]).all
    assert (results["typec"]["HRFindex"] == expected["typec"]["HRFindex"]).all
    assert (results["typed"]["HRFindex"] == expected["typed"]["HRFindex"]).all
