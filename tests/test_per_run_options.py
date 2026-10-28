"""Single-run fits (run-wise FIR, LSS) must get the options of their own run.

Runs have different lengths, so a fit that gets another run's
extra regressors or polynomial degree is detectable from the data length.
"""
import numpy as np
import pytest

import glmsingle.glmsingle as gs
from glmsingle import GLM_single

TR = 1.0
LENGTHS = (100, 120)
DEGS = [1, 4]


def make_data(lengths=LENGTHS, nvox=30, ncond=4, seed=0):
    rng = np.random.default_rng(seed)
    design, data = [], []
    for n in lengths:
        d = np.zeros((n, ncond))
        onsets = np.arange(5, n - 15, 6)
        d[onsets, np.arange(len(onsets)) % ncond] = 1
        design.append(d)
        data.append(100 + rng.standard_normal((nvox, n)))
    return design, data


def fit(tmp_path, design, data, **opt):
    opt = {'wantlibrary': 1, 'wantglmdenoise': 0, 'wantfracridge': 0,
           'wantfileoutputs': [0, 0, 0, 0], 'wantmemoryoutputs': [0, 1, 0, 0],
           **opt}
    return GLM_single(opt).fit(design, data, 1, TR,
                               outputdir=str(tmp_path / 'out'),
                               figuredir=str(tmp_path / 'fig'))


@pytest.fixture
def calls(monkeypatch):
    """(run length, maxpolydeg, extra regressor rows) of every single-run fit."""
    seen = []
    real = gs.glm_estimatemodel

    def spy(design, data, *args):
        opt = args[-1]
        if not isinstance(design, list):
            regs = opt['extra_regressors'][0]
            # the fitter treats this as run 0 and reads entry 0 of each option
            seen.append((data.shape[-1], opt['maxpolydeg'][0],
                         None if regs is None or regs is False else regs.shape[0]))
        return real(design, data, *args)

    monkeypatch.setattr(gs, 'glm_estimatemodel', spy)
    return seen


@pytest.mark.parametrize('wantlss', [0, 1])
def test_maxpolydeg_per_run(tmp_path, calls, wantlss):
    design, data = make_data()
    fit(tmp_path, design, data, maxpolydeg=DEGS, wantlss=wantlss)
    assert calls
    for n, deg, _ in calls:
        assert deg == DEGS[LENGTHS.index(n)]


@pytest.mark.parametrize('wantlss', [0, 1])
def test_extra_regressors_per_run(tmp_path, calls, wantlss):
    design, data = make_data()
    rng = np.random.default_rng(1)
    regs = [rng.standard_normal((n, 3)) for n in LENGTHS]
    fit(tmp_path, design, data, extra_regressors=regs, wantlss=wantlss)
    assert calls
    for n, _, rows in calls:
        assert rows == n
