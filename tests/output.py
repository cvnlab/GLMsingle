from unittest import TestCase
from unittest.mock import Mock, patch
import numpy
from numpy.testing import assert_array_equal


class OutputTests(TestCase):

    @patch('glmsingle.io.output.nibabel')
    @patch('glmsingle.io.output.numpy')
    @patch('glmsingle.io.output.makedirs')
    def test_ensures_directory(self, makedirs, _np, _nb):
        from glmsingle.io.output import Output
        output = Output()
        filepath='/home/johndoe/data/myproject/run_1.nii'
        output.configure_from(sample_file=filepath)
        output.save_variable(11, 'foo')
        makedirs.assert_called_with(
            '/home/johndoe/data/myproject/glmsingle'
        )

    @patch('glmsingle.io.output.nibabel')
    def test_file_path_non_bids(self, nibabel):
        from glmsingle.io.output import Output
        output = Output()
        filepath='/home/johndoe/data/myproject/run_1.nii'
        output.configure_from(sample_file=filepath)
        self.assertEqual(
            output.file_path('bar', 'xyz'),
            '/home/johndoe/data/myproject/glmsingle/bar.xyz'
        )

    def test_file_path_bids(self):
        from glmsingle.io.output import Output
        output = Output()
        bids = Mock()
        bids.root = '/d'
        output.fit_bids_context(bids, sub='1', ses='2', task='a')
        self.assertEqual(
            output.file_path('bar', 'xyz'),
            '/d/derivatives/glmsingle/sub-1/ses-2/sub-1_ses-2_task-a_bar.xyz'
        )

    def test_file_path_bid_no_ses(self):
        from glmsingle.io.output import Output
        output = Output()
        bids = Mock()
        bids.root = '/d'
        output.fit_bids_context(bids, sub='1', ses=None, task='a')
        self.assertEqual(
            output.file_path('bar', 'xyz'),
            '/d/derivatives/glmsingle/sub-1/sub-1_task-a_bar.xyz'
        )

    @patch('glmsingle.io.output.nibabel')
    def test_image(self, nibabel):
        from glmsingle.io.output import Output
        output = Output()
        output.img = Mock()
        output.img.shape = (2, 3, 4, 10) # xyzt
        data = numpy.random.rand(8, 2*3*4)
        output.save_image(data, 'my_img')
        assert_array_equal(
            nibabel.Nifti1Image.call_args[0][0],
            numpy.moveaxis(data, 0, -1).reshape([2, 3, 4, 8])
        )
