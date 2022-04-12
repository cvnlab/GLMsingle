from os.path import join
from os.path import abspath
from os.path import dirname
import scipy
import scipy.io as sio
from glmsingle.glmsingle import GLM_single

test_dir = dirname(abspath(__file__))
data_dir = join(test_dir, "data")
data_file = join(data_dir, "nsdcoreexampledataset.mat")

output_dir = join(test_dir, "outputs")


def test_GLM_single_system():

    X = sio.loadmat(data_file)

    data = []
    design = []
    for r in range(3):  # range(len(X['data'][0])):
        data.append(X["data"][0, r][50:70, 7:27, 0:1, :])
        design.append(scipy.sparse.csr_matrix.toarray(X["design"][0, r]))

    stimdur = X["stimdur"][0][0]
    tr = X["tr"][0][0]

    opt = {"wantmemoryoutputs": [1, 1, 1, 1]}

    # OPTIONAL: PUT THIS IN?
    # opt['wantlibrary'] = 0

    glmsingle_obj = GLM_single(opt)
    glmsingle_obj.fit(design, data, stimdur, tr, outputdir=join(output_dir, "python"))
