"""An all-zero extra_regressors matrix for run 1 must not discard the
regressors of the other runs (matches the per-run handling in the matlab
GLMestimatemodel.m)."""
import numpy as np

from glmsingle.hrf.gethrf import getcanonicalhrf
from glmsingle.hrf.normalisemax import normalisemax
from glmsingle.ols.glm_estimatemodel import glm_estimatemodel

TR = 1.0
LENGTHS = (100, 120)
HRF = normalisemax(getcanonicalhrf(1, TR), dim='global')


def make_inputs(seed=0):
    rng = np.random.default_rng(seed)
    design, data = [], []
    for n in LENGTHS:
        d = np.zeros((n, 2))
        onsets = np.arange(5, n - 15, 6)
        d[onsets, np.arange(len(onsets)) % 2] = 1
        design.append(d)
        data.append(100 + rng.standard_normal((30, n)))
    return design, data, rng


def fit(design, data, extra):
    opt = {'extra_regressors': extra, 'maxpolydeg': [2, 2], 'suppressoutput': 1}
    results = glm_estimatemodel(design, data, 1, TR, 'assume', HRF, 0, opt)[0]
    return results['betasmd'], opt


def test_zero_run1_keeps_other_runs_regressors():
    design, data, rng = make_inputs()
    extra = [np.zeros((LENGTHS[0], 3)), rng.standard_normal((LENGTHS[1], 3))]

    betas, opt = fit(design, data, extra)

    # run 2's regressors survive the defaults handling
    assert opt['extra_regressors'][1] is extra[1]

    # and were actually projected out: betas differ from a fit with no regressors
    betas_none, _ = fit(design, data, [False])
    assert not np.allclose(betas, betas_none)

    # and match a fit where run 1 simply has no regressors (None)
    betas_ref, _ = fit(design, data, [None, extra[1]])
    assert np.allclose(betas, betas_ref)
