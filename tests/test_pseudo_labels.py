"""Small CPU checks for the released pseudo-label alignment operations."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools' / 'pseudo_labels'))
from align_mvsec_quantiles import align_quantiles
from generate_dsec_da2_lidar import bounded_da2_depth, disparity_to_depth
from generate_mvsec_teacher import fit_inverse_affine, sequence_root


class PseudoLabelTests(unittest.TestCase):
    def test_mvsec_inverse_affine_fit(self):
        relative = np.linspace(1, 10, 64, dtype=np.float32).reshape(8, 8)
        gt = 1 / (0.2 * relative + 0.1)
        np.testing.assert_allclose(fit_inverse_affine(relative, gt), gt, rtol=1e-6)
        self.assertEqual(sequence_root(Path('/data'), 'mvsec_dataset_day2_train'),
                         Path('/data/mvsec_dataset_day2/train'))
        with self.assertRaises(ValueError):
            fit_inverse_affine(np.ones((8, 8)), gt)

    def test_mvsec_monotonic_clamped_and_cropped(self):
        teacher = np.arange(100, dtype=np.float32).reshape(10, 10) / 2 + 1
        gt = teacher * 2
        output, count = align_quantiles(teacher, gt, num_quantiles=3,
                                        quantile_min=0.25, quantile_max=0.75)
        valid_teacher = teacher[:8, :8]
        valid = (valid_teacher > 0.001) & (gt[:8, :8] < 80)
        knots = np.quantile(valid_teacher[valid], [0.25, 0.5, 0.75])
        targets = np.quantile(gt[:8, :8][valid], [0.25, 0.5, 0.75])
        np.testing.assert_allclose(output, np.interp(valid_teacher, knots, targets))
        self.assertEqual(output.shape, (8, 8))
        self.assertEqual(count, int(valid.sum()))
        self.assertEqual(output[0, 0], np.float32(targets[0]))
        self.assertEqual(output[-1, -1], np.float32(targets[-1]))

    def test_mvsec_rejects_invalid_fit(self):
        with self.assertRaises(ValueError):
            align_quantiles(np.ones((8, 8)), np.ones((8, 8)))
        with self.assertRaises(ValueError):
            align_quantiles(np.ones((8, 8)), np.zeros((8, 8)))

    def test_dsec_inverse_depth_fit(self):
        relative = np.linspace(0.1, 1.0, 64, dtype=np.float32).reshape(8, 8)
        depth = 1.0 / (0.2 * relative + 0.1)
        result = bounded_da2_depth(relative, depth, np.ones((8, 8), dtype=bool), 0.5, 80)
        np.testing.assert_allclose(result, depth, rtol=1e-5)
        result = disparity_to_depth(np.array([2, 0], dtype=np.float32), 100, 0.5)
        self.assertEqual(result[0], 25)
        self.assertTrue(np.isnan(result[1]))


if __name__ == '__main__':
    unittest.main()
