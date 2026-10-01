"""Geometric checks for the exploratory similarity transform."""

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from align_ceph_landmarks import fit_similarity, pil_inverse_affine, transform_points  # noqa: E402


class SimilarityTests(unittest.TestCase):
    def test_recovers_rotation_uniform_scale_and_translation(self):
        source = np.array([[0, 0], [10, 0], [0, 20], [30, 15]], dtype=float)
        angle = np.deg2rad(12)
        rotation = np.array([[np.cos(angle), np.sin(angle)],
                             [-np.sin(angle), np.cos(angle)]])
        target = 0.37 * source @ rotation + [42, 18]
        scale, fitted_rotation, translation = fit_similarity(source, target, np.ones(len(source)))
        np.testing.assert_allclose(scale, 0.37, atol=1e-12)
        np.testing.assert_allclose(fitted_rotation, rotation, atol=1e-12)
        np.testing.assert_allclose(translation, [42, 18], atol=1e-12)
        np.testing.assert_allclose(transform_points(source, scale, fitted_rotation, translation), target, atol=1e-12)

    def test_pil_inverse_maps_target_to_original_source(self):
        source = np.array([[200.0, 500.0], [850.0, 1200.0], [2100.0, 300.0]])
        target = np.array([[20.0, 55.0], [60.0, 95.0], [105.0, 30.0]])
        scale, rotation, translation = fit_similarity(source, target, np.ones(3))
        a, b, c, d, e, f = pil_inverse_affine(scale, rotation, translation)
        projected = transform_points(source, scale, rotation, translation)
        recovered = np.stack([a * projected[:, 0] + b * projected[:, 1] + c,
                              d * projected[:, 0] + e * projected[:, 1] + f], axis=1)
        np.testing.assert_allclose(recovered, source, atol=1e-10)

    def test_rejects_coincident_points(self):
        source = np.ones((3, 2))
        target = np.array([[1, 1], [2, 2], [3, 3]], dtype=float)
        with self.assertRaisesRegex(ValueError, "coincident"):
            fit_similarity(source, target, np.ones(3))


if __name__ == "__main__":
    unittest.main()
