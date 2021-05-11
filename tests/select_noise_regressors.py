from unittest import TestCase
import numpy


class SelectNoiseRegressorsTest(TestCase):

    def test_select_with_default_cutoff(self):
        from glmsingle.utils.select_noise_regressors import (
            select_noise_regressors
        )
        r2_nrs = numpy.array([2, 3, 4, 4.2, 4.2])
        n = select_noise_regressors(r2_nrs)
        self.assertEqual(n, 3)
