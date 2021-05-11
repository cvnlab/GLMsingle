from unittest import TestCase, skip
from unittest.mock import Mock, patch


class BidsTests(TestCase):

    @patch('glmsingle.io.bids.BIDSLayout')
    def test_match_run_files(self, BIDSLayout):
        from glmsingle.io.bids import BidsDirectory
        layout = BIDSLayout.return_value
        layout.parse_file_entities.side_effect = lambda f: {'run': f[-1]}
        bids = BidsDirectory('')
        bids.index()
        files1 = ['f1_r1', 'f1_r2', 'f1_r3']
        files2 = ['f2_r1', 'f2_r3']
        files1, files2 = bids.match_run_files(files1, files2)
        self.assertEqual(files1, ['f1_r1', 'f1_r3'])
        self.assertEqual(files2, ['f2_r1', 'f2_r3'])
        files3 = ['f3_r1', 'f3_r2', 'f3_r4', 'f3_r6']
        files4 = ['f4_r1', 'f4_r2', 'f4_r3', 'f4_r4', 'f4_r5', 'f4_r6']
        files3, files4 = bids.match_run_files(files3, files4)
        self.assertEqual(files3, ['f3_r1', 'f3_r2', 'f3_r4', 'f3_r6'])
        self.assertEqual(files4, ['f4_r1', 'f4_r2', 'f4_r4', 'f4_r6'])
