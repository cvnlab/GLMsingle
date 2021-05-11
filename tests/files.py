from unittest import TestCase, skip
from unittest.mock import Mock, patch


class FilesTests(TestCase):

    @patch('glmsingle.io.files.load_nifti')
    @patch('glmsingle.io.files.pandas')
    @patch('glmsingle.io.files.GLM_single')
    @patch('glmsingle.io.files.Output')
    def test_run_files_can_create_output(self, Output, GLM_single,
        pandas, load_nifti):
        from glmsingle.io.files import run_files
        out = Output.return_value
        data1, design1 = Mock(), Mock()
        run_files([data1], [design1], 1.0)
        out.configure_from.assert_called_with(sample_file=data1)

    @patch('glmsingle.io.files.load_nifti')
    @patch('glmsingle.io.files.pandas')
    @patch('glmsingle.io.files.GLM_single')
    def test_run_files_plots_figs(self, GLM_single, pandas, load_nifti):
        from glmsingle.io.files import run_files
        glmsingle = GLM_single.return_value
        out = Mock()
        data1, design1 = Mock(), Mock()
        run_files([data1], [design1], 1.0, out=out)
        glmsingle.plot_figures.assert_called_with(out.create_report())

    @patch('glmsingle.io.files.load_nifti')
    @patch('glmsingle.io.files.pandas')
    @patch('glmsingle.io.files.GLM_single')
    def test_run_files_saves_data(self, GLM_single, pandas, load_nifti):
        from glmsingle.io.files import run_files
        glmsingle = GLM_single.return_value
        glmsingle.results.get.side_effect = lambda k: '$' + k
        glmsingle.full_image.side_effect = lambda i: i
        out = Mock()
        data1, design1 = Mock(), Mock()
        run_files([data1], [design1], 1.0, out=out)
        out.save_image.assert_called_with('$pseudo_t_stats', 'pseudo_t_stats')
        out.save_variable.assert_called_with('$xval', 'xval')
