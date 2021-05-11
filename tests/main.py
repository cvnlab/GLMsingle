from unittest import TestCase, skipIf
from numpy.random import RandomState
import numpy
import pandas
import socket
import os


class MainClassTest(TestCase):

    @skipIf(
        socket.gethostname() == 'colles-d1800479',
        'RandomState different on Jaspers workstation'
    )
    @skipIf(
        'TRAVIS' in os.environ,
        'Skipping this test on Travis CI.'
    )
    def test_fit_assume_hrf(self):
        from glmdenoise.glmsingle import GLM_single
        rng = RandomState(seed=156336647)
        design1 = pandas.DataFrame([
            {'onset': 0,  'duration': 0.5, 'trial_type': 'foo'},
            {'onset': 8,  'duration': 0.5, 'trial_type': 'bar'},
            {'onset': 16, 'duration': 0.5, 'trial_type': 'foo'},
            {'onset': 24, 'duration': 0.5, 'trial_type': 'bar'}
        ])
        design = [design1] * 3
        brain = numpy.array([0.1, 1, 1, 1, 0.1])
        response = numpy.ones([8, 5]) * 0.5
        # foo voxel
        response[:, 2] = numpy.array([0.5, 1, 0.5, 0.5, 0.5, 1, 0.5, 0.5])
        # bar voxel
        response[:, 3] = numpy.array([0.5, 0.5, 0.5, 1, 0.5, 0.5, 0.5, 1])
        data = [
            (rng.rand(8, 5) + response) * brain,
            (rng.rand(8, 5) + response) * brain,
            (rng.rand(8, 5) + response) * brain,
        ]
        gd = GLM_single(opt={'wantlss': 0})
        results = gd.fit(design, data, 4.0, stimdur=0.5)
        self.assertEqual(results['pcnum'], 1)
