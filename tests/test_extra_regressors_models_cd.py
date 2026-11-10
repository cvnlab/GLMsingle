"""Models C and D must keep the user's extra regressors, also when no
GLMdenoise PCs are added (matches matlab GLMestimatesingletrial.m, cvnlab#213).
"""
import numpy as np
import pytest

import glmsingle.glmsingle as gs
from glmsingle import GLM_single

TR = 1.0
LENGTHS = (100, 120)
NREG = 3


def make_data(nvox=50, ncond=4, seed=0):
    rng = np.random.default_rng(seed)
    design, data = [], []
    for n in LENGTHS:
        d = np.zeros((n, ncond))
        onsets = np.arange(5, n - 15, 6)
        d[onsets, np.arange(len(onsets)) % ncond] = 1
        design.append(d)
        data.append(100 + rng.standard_normal((nvox, n)))
    return design, data


@pytest.fixture
def calls(monkeypatch):
    """(wantfracridge, per-run extra regressors) of every multi-run fit."""
    seen = []
    real = gs.glm_estimatemodel

    def spy(design, data, *args):
        opt = args[-1]
        if isinstance(design, list):
            seen.append((opt.get('wantfracridge', 0),
                         list(opt['extra_regressors'])))
        return real(design, data, *args)

    monkeypatch.setattr(gs, 'glm_estimatemodel', spy)
    return seen


@pytest.mark.parametrize('wantglmdenoise,wantfracridge', [(1, 0), (0, 1), (1, 1)])
def test_user_regressors_in_models_c_and_d(tmp_path, calls,
                                           wantglmdenoise, wantfracridge):
    design, data = make_data()
    rng = np.random.default_rng(1)
    regs = [rng.standard_normal((n, NREG)) for n in LENGTHS]
    opt = {'wantlibrary': 1, 'wantglmdenoise': wantglmdenoise,
           'wantfracridge': wantfracridge, 'n_pcs': 2,
           'extra_regressors': regs,
           'wantfileoutputs': [0, 0, 0, 0], 'wantmemoryoutputs': [0, 0, 0, 1]}
    GLM_single(opt).fit(design, data, 1, TR,
                        outputdir=str(tmp_path / 'out'),
                        figuredir=str(tmp_path / 'fig'))

    assert calls
    ncols = set()
    for _, extra in calls:
        for r, x in enumerate(extra):
            # the user's regressors come first, any PCs are appended after
            assert x is not None
            np.testing.assert_array_equal(x[:, :NREG], regs[r])
            ncols.add(x.shape[1])
    # some fit used no PCs at all; before the fix it had no regressors
    assert NREG in ncols
    if wantfracridge:
        assert any(fr for fr, _ in calls)
